# Code based on https://github.com/Sein-Kim/LLM-SRec

from ast import parse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import warnings
warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")
warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")

import sys
import argparse

from utils import *
from train_model import *
from inference_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--llm", type=str, default='llama-3b', help='flan_t5, llama, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument("--rec_pre_trained_data", type=str, default='Industrial_and_Scientific',help='Electronics, Movies_and_TV, CDs_and_Vinyl, Industrial_and_Scientific')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--token", action='store_true')

    parser.add_argument("--save_dir", type=str, default='seqllm')

    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--batch_size_infer', default=20, type=int)
    
    parser.add_argument('--infer_epoch', default=1, type=int)
    
    parser.add_argument('--maxlen', default=128, type=int)#50
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    parser.add_argument('--nn_parameter', default=False, action='store_true')
    
    parser.add_argument('--user_group_num', default=20, type=int, help='the number of groups')
    parser.add_argument('--interact_max_num', default=10, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--gated', default=False, action='store_true')

    parser.add_argument('--niter', default=20, type=int)
    parser.add_argument('--group_weight_threshold',default=0.99, type=float)
    parser.add_argument('--phase2_epoch',default=2, type=int)
    parser.add_argument('--inference_data', default=None, type=str)
    parser.add_argument('--candidate_num', default=4, type=int, help='number of candidates per sample (1 positive + N negatives)')
    parser.add_argument('--loss_type', default='softmax', type=str, choices=['softmax', 'gbce'], help='recommendation loss type: softmax (cross-entropy) or gbce')
    parser.add_argument('--gbce_t', default=0.75, type=float, help='gBCE calibration parameter t (0=no calibration, 1=full calibration)')
    
    parser.add_argument('--user_lambda', default=0.01, type=float, help='weight for user-to-center compactness loss')
    parser.add_argument('--center_lambda', default=0.001, type=float, help='weight for center separation loss')
    parser.add_argument('--tail_split', default=False, action='store_true', help='enable head/tail user split evaluation')
    parser.add_argument('--split_rate', default=0.2, type=float, help='proportion of top/bottom users for head/tail split (default: 0.2)')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed')

    args = parser.parse_args()

    if args.inference_data is None:
        args.inference_data = args.rec_pre_trained_data
    
    if args.device =='hpu':
        args.device = torch.device('hpu')
    else:
        args.device = 'cuda:' + str(args.device)
    
    if args.train:
        train_model(args)
    else:
        inference(args)
