#!/bin/bash

set -ex

# check shared memory limits
ipcs -lm
df -h | grep shm

# launch tensorboard in background for the whole duration (even if terminal is closed)
# waits 15s before running tensorboard to make sure tb_logger is created and filled at first
( sleep 15; tensorboard --port 8080 --logdir $PWD/../tb_logger > /dev/null 2>&1 ) & disown

# run code
source activate pytorch
#python train.py -opt options/train/train_EDVR_woTSA_M.yml

num_gpus=5 # for GPU training
python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=4321 \
  train.py -opt options/train/train_EDVR_woTSA_M.yml \
  --launcher pytorch \
  > edvr_train.log 2>&1 & disown
