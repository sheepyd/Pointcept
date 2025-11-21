#!/bin/bash
# Script to train Point Transformer V3 on Tomato dataset

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate pointcept environment
conda activate pointcept

# Check environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

export CUDA_VISIBLE_DEVICES=0  # Modify this to use different GPUs

# Add Pointcept to Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Number of GPUs to use
NUM_GPUS=1

# Config file
CONFIG="configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py"

# Experiment name (will create exp folder with this name)
EXP_NAME="ptv3_tomato20480_$(date +%Y%m%d_%H%M%S)"

# Training command
python tools/train.py \
    --config-file ${CONFIG} \
    --num-gpus ${NUM_GPUS} \
    --options save_path=exp/tomato/${EXP_NAME}

echo "Training finished! Results saved to exp/tomato/${EXP_NAME}"
