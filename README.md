# ClusterRec

> **Cluster-Aware Recommendation with Large Language Models for Efficient and Scalable Inference**

This repository contains the source code and experimental scripts for the paper *"Cluster-Aware Recommendation with Large Language Models for Efficient and Scalable Inference"*.

## Requirements

- **LLM Backbone**: LLaMA-3.2-3B-Instruct
- **Python**: 3.8+
- **Key Dependencies**: PyTorch, Transformers, PEFT, BitsAndBytes, Sentence-Transformers

### Environment Setup

```bash
conda create -n [envname] python=3.10 pip
conda activate [envname]
pip install -r requirements.txt
```

## Data Preparation

We use the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset. Since different categories in Amazon Reviews 2023 store their metadata in different formats (some use Parquet, others use JSONL), we provide two download scripts accordingly:

```bash
# For datasets with Parquet metadata (e.g., Electronics)
bash SeqRec/download_huggingface.bash

# For datasets with JSONL metadata (e.g., Movies_and_TV)
bash SeqRec/download_jsonl.bash
```

> **Note**: Please modify the dataset category name in the script before running. The downloaded data will be stored in `SeqRec/data_[dataset_name]/`.

## Usage

### Step 1: Pre-train CF-RecSys (SASRec)

Train the collaborative filtering sequential recommender:

```bash
cd SeqRec/sasrec
python main.py --device 0 --dataset Industrial_and_Scientific
```

Evaluate the pre-trained model:

```bash
cd SeqRec/sasrec
python main.py --device 0 --dataset Industrial_and_Scientific --save_dir <your_save_dir> --inference_only
```

### Step 2: Train ClusterRec

Train the full ClusterRec model. The best checkpoint is automatically saved based on validation performance each epoch, and inference is performed on the test set upon completion.

```bash
# Modify the parameters (dataset, GPU_ID, user_group_num, etc.) in the script before running
bash shell/train.sh
```

Or run manually with custom parameters:

```bash
python main.py \
    --device 0 \
    --train \
    --rec_pre_trained_data Industrial_and_Scientific \
    --save_dir <save_dir> \
    --batch_size 20 \
    --user_group_num 32 \
    --group_weight_threshold 0.99
```



### Step 3: Inference

Run inference with a trained model:

```bash
# Modify the parameters (dataset, GPU_ID, user_group_num, etc.) in the script before running
bash shell/inference.sh
```

Or run manually:

```bash
python main.py \
    --device 0 \
    --rec_pre_trained_data Industrial_and_Scientific \
    --save_dir <save_dir> \
    --batch_size 20 \
    --user_group_num 32 \
    --group_weight_threshold 0.99 \
    --phase2_epoch <best_epoch>
```

## Project Structure

```
LLM-ClusterRec/
├── main.py                
├── train_model.py          
├── inference_model.py     
├── utils.py                 
├── src/
│   ├── cr_seqllm.py         
│   ├── cr_seqllm4rec.py   
│   └── recsys_model.py   
├── SeqRec/
│   └── sasrec/         
│       ├── main.py
│       ├── model.py
│       ├── utils.py
│       └── data_preprocess.py
├── shell/             
│   ├── train.sh
│   ├── train_loop.sh
│   └── inference.sh
└── requirements.txt
```
