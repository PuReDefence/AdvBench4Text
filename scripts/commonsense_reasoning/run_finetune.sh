#!/bin/bash

DEVICE=0
DATASET_NAMES=("siqa" "csqa")
MODEL_NAME="bert-base-uncased"
DEFENCE_METHOD="finetune"

BATCH_SIZE="16"
LEARNING_RATE="2e-5"
EXP_NAME="b$BATCH_SIZE-lr$LEARNING_RATE"

# Define placeholders or use environment variables
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/output_dir"}
export HF_HOME=${HF_HOME:-"/path/to/huggingface_home"}
export PYTHON_PATH=${PYTHON_PATH:-"/path/to/python_executable"}

MODEL_NAME_WITHOUT_ORG=$(echo "$MODEL_NAME" | cut -d '/' -f 2)
OUTPUT_DIR=$OUTPUT_DIR/$MODEL_NAME_WITHOUT_ORG/$DATASET_NAME/$DEFENCE_METHOD/$EXP_NAME

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    CUDA_VISIBLE_DEVICES=$DEVICE python $PWD/scripts/commonsense_reasoning/run_finetune.py \
        --model_name_or_path $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --pad_to_max_length \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 8 \
        --eval_split_name "validation" \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs 4 \
        --num_warmup_fraction 0.1 \
        --ignore_mismatched_sizes \
        --save_best_checkpoint \
        --seed 1016 \
        --output_dir $OUTPUT_DIR
done