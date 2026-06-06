## ClusterRec: Cluster-Aware Recommendation with Large Language Models for Efficient and Scalable Inference

This repository is designed for implementing ClusterRec.

## Overview

In this [paper](https://arxiv.org/abs/2502.13909), we first demonstrate through a series of experiments that existing LLM4Rec models do not fully capture sequential information both during training and inference. Then, we propose a simple yet effective LLM-based sequential recommender, called LLM-SRec, a method that enhances the integration of sequential information into LLMs by distilling the user representations extracted from a pre-trained CF-SRec model into LLMs.

- We use LLaMA-3.2-3b-instruct.

## Env Setting
```
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## Pre-train CF-RecSys (SASRec)

The data ([Amazon 2023](https://amazon-reviews-2023.github.io/)) is automatically downloaded when using the SASRec training code provided below.

```
cd SeqRec/sasrec
python main.py --device 0 --dataset Industrial_and_Scientific
```

We have also implemented LLM-SRec on the Gaudi-v2 environment (Note that `--nn_parameter` must be used for training models on Gaudi-v2):
```
python main.py --device hpu --dataset Industrial_and_Scientific --nn_parameter
```

## Train - Item Retrieval
The model saves when the best validation score is reached during training and performs inference on the test set.

```
python main.py --device 0 --train --rec_pre_trained_data Industrial_and_Scientific --save_dir model_train --batch_size 20
```

For Gaudi-v2:
```
python main.py --device hpu --train --rec_pre_trained_data Industrial_and_Scientific --save_dir model_train --batch_size 20 --nn_parameter
```

For our method:
```
CUDA_VISIBLE_DEVICES=2 python main.py --device 0 --train --rec_pre_trained_data Industrial_and_Scientific --save_dir Industrial_and_Scientific/model_train_32_lr0.0001x_soft_e2e_newsasrec_thres0.99 --batch_size 20 --user_group_num 32 --group_weight_threshold 0.99
```

For candi
```
CUDA_VISIBLE_DEVICES=2 python main.py --device 0 --train --rec_pre_trained_data Industrial_and_Scientific --save_dir Industrial_and_Scientific/model_train_36_lr0.0001x_candi_16 --batch_size 20 --user_group_num 36 --group_weight_threshold 0.99 --candidate_num 16
```

## Inference
```
CUDA_VISIBLE_DEVICES=2 python main.py --device 0 --rec_pre_trained_data Industrial_and_Scientific --save_dir Industrial_and_Scientific/model_train_32_lr0.0001x_soft_e2e_newsasrec_thres0.99 --batch_size 20 --user_group_num 32 --phase2_epoch 3 --group_weight_threshold 0.99
```

