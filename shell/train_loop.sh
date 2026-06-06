#!/bin/bash

DATASET="Electronics"
# Learning rate
LR="0.0001"

# Group weight threshold
GROUP_WEIGHT_THRESHOLD="0.99"

# Loss function type: softmax or gbce
LOSS_TYPE="softmax"

# Number of candidates (1 positive + N negatives), default 4
CANDIDATE_NUM=4

# GPU id
GPU_ID=0

# batch size
batch_size=20

# Group numbers
GROUP_NUMS=(46 64 52)

for user_group_num in "${GROUP_NUMS[@]}"; do
    if [ "${LOSS_TYPE}" = "gbce" ]; then
        SAVE_DIR="${DATASET}/model_train_${user_group_num}_lr${LR}x_soft_e2e_thres${GROUP_WEIGHT_THRESHOLD}_gbce"
    else
        SAVE_DIR="${DATASET}/model_train_${user_group_num}_lr${LR}x_soft_e2e_thres${GROUP_WEIGHT_THRESHOLD}"
    fi

    if [ "${CANDIDATE_NUM}" -ne 4 ]; then
        SAVE_DIR="${DATASET}/model_train_${user_group_num}_lr${LR}x_soft_candi_${CANDIDATE_NUM}"
    fi

    echo "=========================================="
    echo "Running user_group_num=${user_group_num} on GPU ${GPU_ID}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --device 0 \
        --train \
        --rec_pre_trained_data ${DATASET} \
        --save_dir ${SAVE_DIR} \
        --batch_size ${batch_size} \
        --stage2_lr ${LR} \
        --user_group_num ${user_group_num} \
        --group_weight_threshold ${GROUP_WEIGHT_THRESHOLD} \
        --loss_type ${LOSS_TYPE} \
        --candidate_num ${CANDIDATE_NUM} 
    echo "Finished user_group_num=${user_group_num}"
    echo ""
done

echo "${GROUP_NUMS[@]} All experiments completed!"
