#!/bin/bash

# Set GPU device (optional)
# export CUDA_VISIBLE_DEVICES=0

# Paths and hyperparameters
TRAIN_JSON="../DATA/train.json"
VAL_JSON="../DATA/val.json"
TRAIN_DIR="coco_train" #replace with path to coco train directory 
VAL_DIR="coco_val" #replace with path to coco val directory (the inner most directory that contain all the images)
SAVE_PATH="siglip_model.pt"
NUM_WORKERS=8
NUM_TASKS=5
BATCH_SIZE=64
EPOCHS=10
PATIENCE=3

# Run the training script
python train.py \
  $TRAIN_JSON \
  $VAL_JSON \
  $TRAIN_DIR \
  $VAL_DIR \
  $SAVE_PATH \
  $NUM_WORKERS \
  $NUM_TASKS \
  $BATCH_SIZE \
  $EPOCHS \
  $PATIENCE 2>&1 | tee training_log.txt
