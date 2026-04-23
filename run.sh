#!/bin/bash
# SBATCH --partition=gpu
# SBATCH --gpus-per-node=v100:1
# SBATCH --time=4:00:00

source $HOME/venvs/soogyeong/bin/activate

# python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml