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
CHECKPOINT_DIR="$OUTPUT_DIR/$MODEL_NAME_WITHOUT_ORG/$DATASET_NAME/$DEFENCE_METHOD/$EXP_NAME"
OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME_WITHOUT_ORG/$DATASET_NAME/$DEFENCE_METHOD/$EXP_NAME"

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON_PATH $PWD/scripts/commonsense_reasoning/run_attack.py \
        --dataset_name $DATASET_NAME \
        --model_name_or_path $CHECKPOINT_DIR \
        --ignore_mismatched_sizes \
        --pad_to_max_length \
        --num_examples "-1" \
        --output_dir $OUTPUT_DIR
done