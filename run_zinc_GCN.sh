#!/bin/bash

# Check if a GPU is available
GPU_AVAILABLE=$(command -v nvidia-smi >/dev/null 2>&1 && echo "yes" || echo "no")

if [ "$GPU_AVAILABLE" == "yes" ]; then
  GPU=0
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "GPU detected. Using GPU $GPU."
else
  export CUDA_VISIBLE_DEVICES=""
  echo "No GPU detected. Running on CPU."
fi

# Define common parameters
DATASET="zinc"
BETA=1
LR=0.001
MIN_LR=1e-6
WEIGHT_DECAY=0.001
HIDDEN=300
EPOCHS=1
BATCH_SIZE=256
TRAILS=10
MEMORY=False
PROTOTYPE=False
ME_BATCH_N=3
# Define arrays for domain and shift
DOMAINS=('scaffold' 'size')
SHIFTS=('no_shift' 'covariate' 'concept')

mkdir -p ./log/GCN/

# Loop through each combination of domain and shift
for DOMAIN in "${DOMAINS[@]}"
do
  for SHIFT in "${SHIFTS[@]}"
  do
    # Define log file for the current combination
    LOG_FILE="./log/GCN/${DATASET}_${DOMAIN}_${SHIFT}.log"

    # Run the experiment with nohup and log output
    nohup python -u train_non_causal.py \
      --dataset $DATASET \
      --domain $DOMAIN \
      --shift $SHIFT \
      --beta $BETA \
      --lr $LR \
      --min_lr $MIN_LR \
      --weight_decay $WEIGHT_DECAY \
      --hidden $HIDDEN \
      --epoch $EPOCHS \
      --batch_size $BATCH_SIZE \
      --trails $TRAILS \
      --memory $MEMORY \
      --prototype $PROTOTYPE \
      --me_batch_n $ME_BATCH_N \
      >> $LOG_FILE 2>&1 &

    echo "GCN: Running ZINC dataset with domain $DOMAIN and shift $SHIFT. Logging to $LOG_FILE."
  done
done
