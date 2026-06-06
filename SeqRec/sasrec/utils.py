import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import os
from datetime import datetime
from pytz import timezone
from torch.utils.data import Dataset
from tqdm import tqdm

import pickle

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# DataSet for ddp
class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len): 
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        print("Initializing with num_user:", num_user)

        
    def __len__(self):
        return self.num_user
        
    def __getitem__(self, idx):
        user_id = idx + 1
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
    
        nxt = self.user_train[user_id][-1]
        length_idx = self.max_len - 1
        
        # user의 seq set
        ts = set(self.user_train[user_id])
        for i in reversed(self.user_train[user_id][:-1]):
            seq[length_idx] = i
            pos[length_idx] = nxt
            if nxt != 0: neg[length_idx] = random_neq(1, self.num_item + 1, ts)
            nxt = i
            length_idx -= 1
            if length_idx == -1: break

        return user_id, seq, pos, neg

class SeqDataset_Inference(Dataset):
    def __init__(self, user_train, user_valid, user_test,use_user, num_item, max_len): 
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.num_user = len(use_user)
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user
        print("Initializing with num_user:", self.num_user)

        
    def __len__(self):
        return self.num_user
        
    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len -1
        try:
            seq[idx] = self.user_valid[user_id][0]
            idx -= 1
        except:
            idx = self.max_len - 1
        for i in reversed(self.user_train[user_id]):
            seq[idx] = i
            idx -=1
            if idx ==-1: break
        rated = set(self.user_train[user_id])
        rated.add(0)
        pos = self.user_test[user_id][0]
        neg = []
        for _ in range(1):
            t = np.random.randint(1,self.num_item+1)
            while t in rated: t = np.random.randint(1,self.num_item+1)
            neg.append(t)
        neg = np.array(neg)
        return user_id, seq, pos, neg
    
    
class SeqDataset_Validation(Dataset):
    def __init__(self, user_train, user_valid,use_user, num_item, max_len): 
        self.user_train = user_train
        self.user_valid = user_valid
        self.num_user = len(use_user)
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user
        print("Initializing with num_user:", self.num_user)

        
    def __len__(self):
        return self.num_user
        
    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len -1
        for i in reversed(self.user_train[user_id]):
            seq[idx] = i
            idx -=1
            if idx ==-1: break
        rated = set(self.user_train[user_id])
        rated.add(0)
        pos = self.user_valid[user_id][0]
        neg = []
        for _ in range(1):
            t = np.random.randint(1,self.num_item+1)
            while t in rated: t = np.random.randint(1,self.num_item+1)
            neg.append(t)
        neg = np.array(neg)
        return user_id, seq, pos, neg
    
# train/val/test data generation
def data_partition(fname, args, path=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    # assume user/item index starting from 1

    for t in ['train','valid','test']:
        if path == None:  
            f = open(f'./../data_{args.dataset}/{fname}_{t}.txt', 'r')
        else:
            f = open(path+f'_{t}.txt','r')
            
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            if t =='train':
                user_train[u].append(i)
            elif t =='valid':
                user_valid[u].append(i)
            elif t =='test':
                user_test[u].append(i)
    
    
    eval_set = [set(user_valid.keys()), set(user_test.keys())]

    
    return [user_train, user_valid, user_test, usernum, itemnum, eval_set]

# evaluate on test set
def evaluate(model, dataset, args, mode= 1, ranking = 10):
    [train, valid, test, usernum, itemnum, eval_set] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    
    valid_user = 0.0

    eval_set = eval_set[mode]
    if len(eval_set)>10000:
        users = random.sample(list(eval_set), 10000)
    else:
        users = list(eval_set)
    
    num_candi = 99
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        try:
            seq[idx] = valid[u][0]
            idx -= 1
        except:
            idx = args.maxlen - 1
            
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
    
        his = train[u] + valid[u] + item_idx
        his = set(his)
        his.add(0)
        items = set([i for i in range(1,itemnum+1)])
        items = list(items.difference(his))
        if len(items) > num_candi:
            samples = random.sample(items, num_candi)
            item_idx = item_idx +  samples
        else:
            item_idx = item_idx + items

        l_ = [i for i in range(len(item_idx))]
        random.shuffle(l_)
        
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC
        
        rank = predictions[l_].argsort().argsort()[l_.index(0)].item()
        valid_user += 1
        

        if rank < ranking:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user



# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, eval_set] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    
    eval_set = eval_set[0]
    if len(eval_set)>10000:
        users = random.sample(list(eval_set), 10000)
    else:
        users = list(eval_set)
        
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user

def evaluate_with_users(model, dataset, args, user_list, mode=1, ranking=10):
    """Evaluate on a specific list of users (for tail split evaluation)."""
    [train, valid, test, usernum, itemnum, eval_set] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    num_candi = 99

    for u in user_list:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        try:
            seq[idx] = valid[u][0]
            idx -= 1
        except:
            idx = args.maxlen - 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        his = train[u] + valid[u] + item_idx
        his = set(his)
        his.add(0)
        items = set([i for i in range(1, itemnum + 1)])
        items = list(items.difference(his))
        if len(items) > num_candi:
            samples = random.sample(items, num_candi)
            item_idx = item_idx + samples
        else:
            item_idx = item_idx + items

        l_ = [i for i in range(len(item_idx))]
        random.shuffle(l_)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions[l_].argsort().argsort()[l_.index(0)].item()
        valid_user += 1

        if rank < ranking:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user == 0:
        return 0.0, 0.0
    return NDCG / valid_user, HT / valid_user


def save_eval(model, dataset, args):
    model.eval()
    
    with torch.no_grad():
        print('Evaluate')
        t_test = evaluate(model, dataset, args)
        print('\n')
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    with open(f'./../data_{args.dataset}/Results.txt', 'w') as f:
        sys.stdout = f
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        sys.stdout = sys.__stdout__ 
        