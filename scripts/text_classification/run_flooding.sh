#!/bin/bash

DEVICE=0
DATASET_NAMES=("sst2" "mr" "mrpc" "scitail")
DATASET_CONFIG_NAME="default"
MODEL_NAME="bert-base-uncased"
DEFENCE_METHOD="flooding"

BATCH_SIZE="32"
LEARNING_RATE="5e-5"
FLOOD_LEVEL="0.1"
EXP_NAME="f$FLOOD_LEVEL-b$BATCH_SIZE-lr$LEARNING_RATE"

# Define placeholders or use environment variables
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/output_dir"}
export HF_HOME=${HF_HOME:-"/path/to/huggingface_home"}
export PYTHON_PATH=${PYTHON_PATH:-"/path/to/python_executable"}

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    MODEL_NAME_WITHOUT_ORG=$(echo "$MODEL_NAME" | cut -d '/' -f 2)
    OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME_WITHOUT_ORG/$DATASET_NAME/$DEFENCE_METHOD/$EXP_NAME"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON_PATH $PWD/scripts/text_classification/run_flooding.py \
        --model_name_or_path $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --dataset_config_name $DATASET_CONFIG_NAME \
        --num_train_epochs 4 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 8 \
        --learning_rate $LEARNING_RATE \
        --num_warmup_fraction 0.1 \
        --checkpointing_steps epoch \
        --pad_to_max_length \
        --seed 1016 \
        --save_best_checkpoint \
        --flood_level $FLOOD_LEVEL \
        --output_dir $OUTPUT_DIR
done