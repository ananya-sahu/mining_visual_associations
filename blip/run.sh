#!/bin/bash

# Set GPU device (optional)
export CUDA_VISIBLE_DEVICES=0

# Paths and hyperparameters
TRAIN_JSON="../DATA/train.json"
VAL_JSON="../DATA/val.json"
TRAIN_DIR="coco_train" #replace with path to coco train directory 
VAL_DIR="coco_val" #replace with path to coco val directory (the inner most directory that contain all the images)
SAVE_PATH="blip_model.pt"
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
  $PATIENCE 2>&1 | tee training_log_blip.txt

# ------------------ Evaluation 1------------------

# Evaluation config (SEPARATE from VAL_JSON/VAL_DIR)
EVAL_JSON="mm_poem.json"     # path to test/eval JSON
EVAL_IMG_DIR="Eval1_Imgs"             # path to test images
EVAL_LOG="eval_1_log_blip.txt"

# Clear eval log before writing
> $EVAL_LOG

# Baseline evaluation
echo -e "\n=== Baseline Evaluation ===" | tee -a $EVAL_LOG
python eval_1.py \
  true \
  none \
  $EVAL_JSON \
  $EVAL_IMG_DIR \
  none 2>&1 | tee -a $EVAL_LOG

# Evaluation with task labels 0 through 4
for TASK in {0..4}; do
  echo -e "\n=== Evaluation with Task Label $TASK ===" | tee -a $EVAL_LOG
  python eval_1.py \
    false \
    $SAVE_PATH \
    $EVAL_JSON \
    $EVAL_IMG_DIR \
    $TASK 2>&1 | tee -a $EVAL_LOG
done

# ------------------ Evaluation 2 (SigLIP Metaphor) ------------------

# Eval 2 config
EVAL_LOG_2="eval_3_log_blip.txt"

# Label directories (exactly 6 required)
LABEL_ROOT_0="dummy_data"
LABEL_ROOT_1="dummy_data/label_0"
LABEL_ROOT_2="dummy_data/label_1"
LABEL_ROOT_3="dummy_data/label_2"
LABEL_ROOT_4="dummy_data/label_3"
LABEL_ROOT_5="dummy_data/label_4"

# Clear eval log
> $EVAL_LOG_2

# Baseline
echo -e "\n=== [Eval 2] Baseline ===" | tee -a $EVAL_LOG_2
python eval_2.py \
  true \
  none \
  $LABEL_ROOT_0 \
  $LABEL_ROOT_1 \
  $LABEL_ROOT_2 \
  $LABEL_ROOT_3 \
  $LABEL_ROOT_4 \
  $LABEL_ROOT_5 2>&1 | tee -a $EVAL_LOG_2

python eval_2.py \
  false \
  $SAVE_PATH \
  $LABEL_ROOT_0 \
  $LABEL_ROOT_1 \
  $LABEL_ROOT_2 \
  $LABEL_ROOT_3 \
  $LABEL_ROOT_4 \
  $LABEL_ROOT_5 2>&1 | tee -a $EVAL_LOG_2


# ------------------ Evaluation 3 (SigLIP Contrastive) ------------------
EVAL_LOG_3="eval3_log_blip.txt"
PICKLE_FILE="/content/dummy_contrastive.pkl"  # Update to real contrastive pickle if needed

# Clear eval log
> $EVAL_LOG_3

# Baseline evaluation
echo -e "\n=== [Eval 2] Baseline (Contrastive) ===" | tee -a $EVAL_LOG_3
python eval_3.py \
  none \
  true \
  $PICKLE_FILE 2>&1 | tee -a $EVAL_LOG_3

# Evaluation with task labels 0 through 4

echo -e "\n=== [Eval 3] Task Labels $TASK" | tee -a $EVAL_LOG_3
python eval_3.py \
  $SAVE_PATH \
  false \
  $PICKLE_FILE \
  $TASK 2>&1 | tee -a $EVAL_LOG_3