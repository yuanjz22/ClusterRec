#!/bin/bash

learing_rate=0.0001
user_group_num=32
dataset="Industrial_and_Scientific"
group_weight_threshold=0.99
phase2_epoch=2
GPU_ID=4

random_seed=0

echo "Inference: user_group_num: ${user_group_num}, dataset: ${dataset}, group_weight_threshold: ${group_weight_threshold}, phase2_epoch: ${phase2_epoch}, GPU_ID: ${GPU_ID}"

save_dir=$dataset/model_train_${user_group_num}_lr${learing_rate}x_soft_e2e_newsasrec_thres${group_weight_threshold}

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --device 0 \
    --rec_pre_trained_data $dataset \
    --save_dir $save_dir \
    --batch_size 20 \
    --user_group_num ${user_group_num} \
    --phase2_epoch ${phase2_epoch} \
    --group_weight_threshold $group_weight_threshold