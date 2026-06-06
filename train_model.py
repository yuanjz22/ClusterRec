import os
import torch
import random
import time
import os
import sys
import gc

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
        
def train_model(args):
    print('LLMRec start train\n')
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(train_model_,
             args=(world_size,args),
             nprocs=world_size,
             join=True)
    else:
        train_model_(0, 0, args)

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
  

def train_model_(rank,world_size,args):
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if args.device == 'hpu':
            args.device = torch.device('hpu')
        else:
            args.device = 'cuda:' + str(rank)

    # random seed
    random.seed(args.random_seed)

    

    dataset = data_partition(args.rec_pre_trained_data, args, path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    args.usernum = usernum

    model = llmrec_model(args).to(args.device)

    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    # Init Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)
    
    
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(valid_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size, pin_memory=True, shuffle=True)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))
    scheduler = LambdaLR(adam_optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    epoch_start_idx = 1
    T = 0.0
    best_perform = 0
    early_stop = 0
    early_thres = 10
    t0 = time.time()
    total_train_time = 0.0
    total_eval_time = 0.0
    completed_epochs = 0
    
    eval_set_use = eval_set[1]
    if len(eval_set_use)>10000:
        users = random.sample(list(eval_set_use), 10000)
    else:
        users = list(eval_set_use)
    
    user_list = []
    for u in users:
        if len(user_test[u]) < 1: continue
        user_list.append(u)

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

    # start training 
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        epoch_train_time = 0.0
        epoch_eval_time = 0.0
        for step, data in enumerate(train_data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_train_start = time.time()
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u,seq,pos,neg], optimizer=adam_optimizer, batch_iter=[epoch,args.num_epochs + 1,step,num_batch], mode='phase2')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_train_time += time.time() - step_train_start
            if step % (num_batch//10) ==0 and step !=0:
                eval_set_use = eval_set[0]
                if len(eval_set_use)>10000:
                    users = random.sample(list(eval_set_use), 10000)
                else:
                    users = list(eval_set_use)
                
                user_list_valid = []
                for u in users:
                    if len(user_valid[u]) < 1: continue
                    user_list_valid.append(u)
                valid_data_set = SeqDataset_Validation(user_train, user_valid, user_list_valid, itemnum, args.maxlen)
                valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size_infer, pin_memory=True, shuffle=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mid_eval_start = time.time()

                model.eval()
                model.extract_group_rep()

                num_valid_batch = len(user_valid.keys())//args.batch_size_infer
                model.users = 0.0
                model.NDCG = 0.0
                model.HT = 0.0
                model.NDCG_20 = 0.0
                model.HIT_20 = 0.0
                model.all_embs = None
                with torch.no_grad():
                    for _, data in enumerate(valid_data_loader):
                        print("Validation, early stop:", early_stop)
                        u, seq, pos, neg = data
                        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
                        model([u,seq,pos,neg, rank, None, 'original'], mode='generate_batch')
                        
                        
                perform = model.HT/model.users

                if perform >= best_perform:
                    best_perform = perform
                    if rank == 0:
                        if args.multi_gpu: model.module.save_model(args, epoch2=epoch, best=True)
                        else: model.save_model(args,  epoch2=epoch, best=True)
                    
                    model.users = 0.0
                    model.NDCG = 0.0
                    model.HT = 0.0
                    model.NDCG_20 = 0.0
                    model.HIT_20 = 0.0

                    model.extract_group_rep()

                    with torch.no_grad():
                        for _, data in enumerate(inference_data_loader):
                            print("Testing")
                            u, seq, pos, neg = data
                            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
                            model([u,seq,pos,neg, rank, None, 'original'], mode='generate_batch')


                    out_dir = f'./models/{args.save_dir}/'
                    out_dir = out_dir[:-1] + 'best/'
                    
                    out_dir += f'{args.rec_pre_trained_data}_'
                    
                    out_dir += f'{args.llm}_{epoch}_results.txt'
                    
                    f = open(out_dir, 'a')
                    f.write(f'NDCG: {model.NDCG/model.users}, HR: {model.HT/model.users}\n')
                    f.write(f'NDCG20: {model.NDCG_20/model.users}, HR20: {model.HIT_20/model.users}\n')
                    f.close()
                    
                    early_stop = 0
                else:
                    model.save_model(args,  epoch2=epoch)
                    early_stop +=1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                epoch_eval_time += time.time() - mid_eval_start

                if early_stop == early_thres:
                    partial_ratio = step / num_batch if num_batch > 0 else 1.0
                    total_train_time += epoch_train_time
                    total_eval_time += epoch_eval_time
                    completed_epochs += partial_ratio
                    print(f'Epoch {epoch} (partial {partial_ratio:.2f}) | Train time: {epoch_train_time:.2f}s | Eval time: {epoch_eval_time:.2f}s')
                    avg_train = total_train_time / completed_epochs
                    avg_eval = total_eval_time / completed_epochs
                    print(f'Completed {completed_epochs:.2f} epochs | Avg train/epoch: {avg_train:.2f}s | Avg eval/epoch: {avg_eval:.2f}s | Avg total/epoch: {avg_train + avg_eval:.2f}s')
                    print(f'Total train time: {total_train_time:.2f}s | Total eval time: {total_eval_time:.2f}s | Total: {time.time() - t0:.2f}s')
                    print("Terminating Train (early stopping)")
                    return
                model.train()
                scheduler.step()
                
        if rank == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_epoch_eval_start = time.time()

            model.eval()
            num_valid_batch = len(user_valid.keys())//args.batch_size_infer
            model.users = 0.0
            model.NDCG = 0.0
            model.HT = 0.0
            model.NDCG_20 = 0.0
            model.HIT_20 = 0.0
            model.all_embs = None
            
            eval_set_use = eval_set[0]
            if len(eval_set_use)>10000:
                users = random.sample(list(eval_set_use), 10000)
            else:
                users = list(eval_set_use)
            
            user_list_valid = []
            for u in users:
                if len(user_valid[u]) < 1: continue
                user_list_valid.append(u)
            valid_data_set = SeqDataset_Validation(user_train, user_valid, user_list_valid, itemnum, args.maxlen)
            valid_data_loader = DataLoader(valid_data_set, batch_size = args.batch_size_infer, pin_memory=True, shuffle=True)

            model.extract_group_rep()

            with torch.no_grad():
                for _, data in enumerate(valid_data_loader):
                    print("Validation, early stop:", early_stop)
                    u, seq, pos, neg = data
                    u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
                    model([u,seq,pos,neg, rank, None, 'original'], mode='generate_batch')

            if perform >= best_perform:
                best_perform = perform
                if rank ==0:
                    if args.multi_gpu: model.module.save_model(args, epoch2=epoch, best=True)
                    else: model.save_model(args,  epoch2=epoch, best=True)
                
                model.users = 0.0
                model.NDCG = 0.0
                model.HT = 0.0
                model.NDCG_20 = 0.0
                model.HIT_20 = 0.0

                model.extract_group_rep()

                with torch.no_grad():
                    for _, data in enumerate(inference_data_loader):
                        print("Testing")
                        u, seq, pos, neg = data
                        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()                        
                        model([u,seq,pos,neg, rank, None, 'original'], mode='generate_batch')


                out_dir = f'./models/{args.save_dir}/'
                out_dir = out_dir[:-1] + 'best/'
                
                out_dir += f'{args.rec_pre_trained_data}_'
                
                out_dir += f'{args.llm}_{epoch}_results.txt'
                
                f = open(out_dir, 'a')
                f.write(f'NDCG: {model.NDCG/model.users}, HR: {model.HT/model.users}\n')
                f.write(f'NDCG20: {model.NDCG_20/model.users}, HR20: {model.HIT_20/model.users}\n')
                f.close()

                
                early_stop = 0
            else:
                model.save_model(args,  epoch2=epoch)
                early_stop +=1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_eval_time += time.time() - end_epoch_eval_start

            if early_stop == early_thres:
                total_train_time += epoch_train_time
                total_eval_time += epoch_eval_time
                completed_epochs += 1
                print(f'Epoch {epoch} | Train time: {epoch_train_time:.2f}s | Eval time: {epoch_eval_time:.2f}s')
                avg_train = total_train_time / completed_epochs
                avg_eval = total_eval_time / completed_epochs
                print(f'Completed {completed_epochs:.2f} epochs | Avg train/epoch: {avg_train:.2f}s | Avg eval/epoch: {avg_eval:.2f}s | Avg total/epoch: {avg_train + avg_eval:.2f}s')
                print(f'Total train time: {total_train_time:.2f}s | Total eval time: {total_eval_time:.2f}s | Total: {time.time() - t0:.2f}s')
                print("Terminating Train (early stopping)")
                return
            model.train()
            scheduler.step()

        total_train_time += epoch_train_time
        total_eval_time += epoch_eval_time
        completed_epochs += 1
        print(f'Epoch {epoch} | Train time: {epoch_train_time:.2f}s | Eval time: {epoch_eval_time:.2f}s | Total: {epoch_train_time + epoch_eval_time:.2f}s')

    
    avg_train = total_train_time / completed_epochs if completed_epochs > 0 else 0
    avg_eval = total_eval_time / completed_epochs if completed_epochs > 0 else 0
    print(f'Completed {completed_epochs} epochs | Avg train/epoch: {avg_train:.2f}s | Avg eval/epoch: {avg_eval:.2f}s | Avg total/epoch: {avg_train + avg_eval:.2f}s')
    print(f'Total train time: {total_train_time:.2f}s | Total eval time: {total_eval_time:.2f}s | Total: {time.time() - t0:.2f}s')
    if args.multi_gpu:
        destroy_process_group()
    return
