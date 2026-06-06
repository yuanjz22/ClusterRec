import os
import torch
import random
import time
import os
import sys
import gc
import numpy as np

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR

from src.cr_seqllm import *
from SeqRec.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference, SeqDataset_Validation


def setup_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    if args.device.type == 'hpu':
        import habana_frameworks.torch.distributed.hccl
        init_process_group(backend="hccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    # htcore.set_device(rank)

def inference(args):
    print('LLMRec start inference\n')
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(inference_,
             args=(world_size,args),
             nprocs=world_size,
             join=True)
    else:
        inference_(0,0,args)

def inference_(rank,world_size,args):
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if args.device == 'hpu':
            args.device = torch.device('hpu')
        else:
            args.device = 'cuda:' + str(rank)

    random.seed(42)

    
    dataset = data_partition(args.inference_data, args, path=f'./SeqRec/data_{args.inference_data}/{args.inference_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    args.usernum = usernum

    model = llmrec_model(args).to(args.device)
    
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u]) + len(user_valid.get(u, [])) + len(user_test.get(u, []))
    print('average interaction length: %.2f' % (cc / len(user_train)))

    train_data_set = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)
    
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(valid_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size, pin_memory=True, shuffle=True)
    
    eval_set_use = eval_set[1]
    if len(eval_set_use)>10000:
        users = random.sample(list(eval_set_use), 10000)
    else:
        users = list(eval_set_use)
    
    user_list = []
    for u in users:
        if len(user_test[u]) < 1: continue
        user_list.append(u)

    # Split users into head/tail by interaction length (only enabled with --tail_split)
    # Top split_rate by interaction count are head users, bottom split_rate are tail users
    tail_split_enabled = getattr(args, 'tail_split', False)
    split_rate = getattr(args, 'split_rate', 0.2)

    if tail_split_enabled:
        interaction_lengths = {u: len(user_train[u]) + len(user_valid.get(u, [])) for u in user_list}
        sorted_users = sorted(user_list, key=lambda u: interaction_lengths[u], reverse=True)
        total_users = len(sorted_users)
        head_count = int(total_users * split_rate)
        tail_count = int(total_users * split_rate)
        head_users = sorted_users[:head_count]
        tail_users = sorted_users[total_users - tail_count:]
        head_threshold = interaction_lengths[sorted_users[head_count - 1]] if head_count > 0 else 0
        tail_threshold = interaction_lengths[sorted_users[total_users - tail_count]] if tail_count > 0 else 0
        print(f'Tail split enabled (top {split_rate*100:.0f}% / bottom {split_rate*100:.0f}%)')
        print(f'Head users (top {split_rate*100:.0f}%, interaction >= {head_threshold}): {len(head_users)}')
        print(f'Tail users (bottom {split_rate*100:.0f}%, interaction <= {tail_threshold}): {len(tail_users)}')

    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, pin_memory=True)

    train_data_loader_all = DataLoader(train_data_set,
                                batch_size=len(train_data_set),
                                pin_memory=True,
                                shuffle=False)
    u_all, seq_all, pos_all, neg_all = next(iter(train_data_loader_all))
    u_all, seq_all, pos_all, neg_all = (
        u_all.numpy(), seq_all.numpy(), pos_all.numpy(), neg_all.numpy()
    )
    model.pre_train_phase1([u_all,seq_all, pos_all, neg_all],Group_=True)

    # Load the best model
    model.load_model(args, phase2_epoch=args.phase2_epoch, best=True)

    model.eval()

    model.users = 0.0
    model.NDCG = 0.0
    model.HT = 0.0
    model.NDCG_20 = 0.0
    model.HIT_20 = 0.0
    model.NDCG_30 = 0.0
    model.HIT_30 = 0.0
    model.NDCG_40 = 0.0
    model.HIT_40 = 0.0
    model.NDCG_50 = 0.0
    model.HIT_50 = 0.0
    model.all_embs = None
    
    # Clean up memory, consistent with training
    gc.collect()
    torch.cuda.empty_cache()

    total_start = time.time()
    with torch.no_grad():
        for _, data in enumerate(inference_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()

            model([u,seq,pos,neg, rank, None, 'original'], mode='generate_batch')

    total_elapsed = time.time() - total_start
    user_time = time.time() - model.user_start

    # Save full results
    group_results = {}
    group_results['all'] = {
        'users': model.users, 'NDCG': model.NDCG, 'HT': model.HT,
        'NDCG_20': model.NDCG_20, 'HIT_20': model.HIT_20,
        'NDCG_30': model.NDCG_30, 'HIT_30': model.HIT_30,
        'NDCG_40': model.NDCG_40, 'HIT_40': model.HIT_40,
        'NDCG_50': model.NDCG_50, 'HIT_50': model.HIT_50,
    }

    # Run inference separately for head/tail users (reuse cached all_embs)
    if tail_split_enabled:
        for group_name, group_user_list in [('head', head_users), ('tail', tail_users)]:
            if len(group_user_list) == 0:
                continue
            group_data_set = SeqDataset_Inference(user_train, user_valid, user_test, group_user_list, itemnum, args.maxlen)
            group_data_loader = DataLoader(group_data_set, batch_size=args.batch_size_infer, pin_memory=True)

            model.users = 0.0
            model.NDCG = 0.0
            model.HT = 0.0
            model.NDCG_20 = 0.0
            model.HIT_20 = 0.0
            model.NDCG_30 = 0.0
            model.HIT_30 = 0.0
            model.NDCG_40 = 0.0
            model.HIT_40 = 0.0
            model.NDCG_50 = 0.0
            model.HIT_50 = 0.0

            with torch.no_grad():
                for _, data in enumerate(group_data_loader):
                    u, seq, pos, neg = data
                    u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                    model([u,seq,pos,neg, rank, None, 'original'], mode='generate_batch')

            group_results[group_name] = {
                'users': model.users, 'NDCG': model.NDCG, 'HT': model.HT,
                'NDCG_20': model.NDCG_20, 'HIT_20': model.HIT_20,
                'NDCG_30': model.NDCG_30, 'HIT_30': model.HIT_30,
                'NDCG_40': model.NDCG_40, 'HIT_40': model.HIT_40,
                'NDCG_50': model.NDCG_50, 'HIT_50': model.HIT_50,
            }

    if args.inference_data == args.rec_pre_trained_data:
        out_dir = f'./inference/{args.save_dir}/{args.rec_pre_trained_data}_{args.llm}_{args.phase2_epoch}_results.txt'
    else:
        out_dir = f'./inference/{args.save_dir}/{args.rec_pre_trained_data}-{args.inference_data}_{args.llm}_{args.phase2_epoch}_results.txt'

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    f = open(out_dir, 'a')

    for group_name, result in group_results.items():
        num_users = result['users']
        if num_users == 0:
            continue
        if tail_split_enabled:
            threshold_info = head_threshold if group_name == 'head' else tail_threshold
            f.write(f'\n===== {group_name} (users: {int(num_users)}, threshold: {threshold_info}) =====\n')
        else:
            f.write(f'\n===== {group_name} (users: {int(num_users)}) =====\n')
        f.write(f'NDCG: {result["NDCG"]/num_users}, HR: {result["HT"]/num_users}\n')
        f.write(f'NDCG20: {result["NDCG_20"]/num_users}, HR20: {result["HIT_20"]/num_users}\n')
        f.write(f'NDCG30: {result["NDCG_30"]/num_users}, HR30: {result["HIT_30"]/num_users}\n')
        f.write(f'NDCG40: {result["NDCG_40"]/num_users}, HR40: {result["HIT_40"]/num_users}\n')
        f.write(f'NDCG50: {result["NDCG_50"]/num_users}, HR50: {result["HIT_50"]/num_users}\n')

    f.write(f'\ntotal_reference_time: {total_elapsed:.4f} s; user_time: {user_time:.4f} s')

    f.close()

    