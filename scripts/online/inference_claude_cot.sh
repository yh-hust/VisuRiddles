#!/bin/bash
NGPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
model_name=claude-3-7-sonnet-20250219
tp=1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR=output/$model_name
LOG_DIR=$SAVE_DIR/logs/qwen_mpdoc_eval_${TIMESTAMP}.log
model=$model_name
dataset_path='abstract'
use_cot=1

mkdir -p "$SAVE_DIR"
mkdir -p "$SAVE_DIR/logs"

DISTRIBUTED_ARGS="--nproc_per_node=$NGPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
SCRIPT_ARGS="--model $model --save_dir $SAVE_DIR --dataset_path $dataset_path --use_cot $use_cot"
export CUDA_VISIBLE_DEVICES=1
python evaluate/inference_online.py $SCRIPT_ARGS 2>&1 | tee -a $LOG_DIR
