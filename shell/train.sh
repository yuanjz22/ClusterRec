#!/bin/bash

learing_rate=0.0001
user_group_num=32
dataset="Industrial_and_Scientific"
group_weight_threshold=0.99
GPU_ID=2
random_seed=42
batch_size=20

save_dir=$dataset/model_train_${user_group_num}_lr${learing_rate}x_soft_e2e_thres${group_weight_threshold}

echo "user_group_num: ${user_group_num}, dataset: ${dataset}, group_weight_threshold: ${group_weight_threshold}, GPU_ID: ${GPU_ID}, batch_size: ${batch_size}, random_seed: ${random_seed}"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --device 0 \
    --train \
    --rec_pre_trained_data "$dataset" \
    --stage2_lr $learing_rate \
    --save_dir "$save_dir" \
    --batch_size $batch_size \
    --user_group_num ${user_group_num} \
    --phase2_epoch 2 \
    --group_weight_threshold $group_weight_threshold \
    --random_seed $random_seed