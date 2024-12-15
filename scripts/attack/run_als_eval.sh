#!/bin/bash

DEVICE=0
DATASET_NAMES=("sst2" "mr" "mrpc" "scitail")
DATASET_CONFIG_NAME="default"
MODEL_NAME="bert-base-uncased"
DEFENCE_METHOD="als"

BATCH_SIZE="32"
LEARNING_RATE="5e-5"
ALPHA="0.1"
EXP_NAME="a$ALPHA-b$BATCH_SIZE-lr$LEARNING_RATE"

# Define placeholders or use environment variables
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/output_dir"}
export HF_HOME=${HF_HOME:-"/path/to/huggingface_home"}
export PYTHON_PATH=${PYTHON_PATH:-"/path/to/python_executable"}

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    MODEL_NAME_WITHOUT_ORG=$(echo "$MODEL_NAME" | cut -d '/' -f 2)
    CKPT_DIR="$OUTPUT_DIR/$MODEL_NAME_WITHOUT_ORG/$DATASET_NAME/$DEFENCE_METHOD/$EXP_NAME"
    OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME_WITHOUT_ORG/$DATASET_NAME/$DEFENCE_METHOD/$EXP_NAME"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON_PATH $PWD/scripts/attack/run_attack.py \
        --dataset_name $DATASET_NAME \
        --dataset_config_name $DATASET_CONFIG_NAME \
        --model_name_or_path $CKPT_DIR \
        --output_dir $OUTPUT_DIR \
        --pad_to_max_length \
        --num_devices 1 \
        --attackers textfooler textbugger \
        --save_attack_results \
        --ignore_mismatched_sizes
done