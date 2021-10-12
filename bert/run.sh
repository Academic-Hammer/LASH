#!/bin/bash

# train the model
# dataset: dblp_cn, dblp_cv dblp_ml dblp_nlp
dataset=$1
cuda=$2

gpu_ids=(${cuda//,/})
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29436 train_multilabel.py \
    --dataset $dataset


