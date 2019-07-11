#!/bin/bash

set -ex

# check shared memory limits
ipcs -lm
df -h | grep shm

# launch tensorboard in background for the whole duration (even if terminal is closed)
tensorboard --port $CDSW_PUBLIC_PORT --logdir $PWD/../tb_logger > /dev/null 2>&1 & disown

# run code in pytorch env
source activate pytorch

NUM_GPUS=6      # for GPU training
MODEL=EDVR
MODEL_SIZE=M    # Large (L) or Medium (M)
LR=4e-4         # learning rate
BS=16           # batch size per GPU
EXT=_noinplace_rb

### stage 1
#python train.py -opt options/train/train_${MODEL}_woTSA_${MODEL_SIZE}.yml \
#    > ${MODEL}_woTSA_train_${LR}_bs${BS}_${MODEL_SIZE}.log 2>&1 & disown

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=4321 \
  train.py -opt options/train/train_${MODEL}_woTSA_${MODEL_SIZE}.yml \
  --launcher pytorch \
  > ${MODEL}_woTSA_train_${LR}_bs${BS}_${MODEL_SIZE}${EXT}.log 2>&1 & disown

### stage 2
#python train.py -opt options/train/train_${MODEL}_${MODEL_SIZE}.yml \
#    > ${MODEL}_train_${LR}_bs${BS}_${MODEL_SIZE}.log 2>&1 & disown

#python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=4321 \
#  train.py -opt options/train/train_${MODEL}_${MODEL_SIZE}.yml \
#  --launcher pytorch \
#  > ${MODEL}_train_${LR}_bs${BS}_${MODEL_SIZE}.log 2>&1 & disown
