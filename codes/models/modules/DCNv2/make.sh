#!/usr/bin/env bash

# You may need to modify the following paths before compiling.

CUDA_HOME=/usr/local/cuda \
CUDNN_INCLUDE_DIR=/usr/local/cuda/include \
CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
CFLAGS="-mfma -mavx2" \
python setup.py build develop
