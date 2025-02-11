#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=3D-CNN
#SBATCH --output=3D-CNN.out
#SBATCH --error=3D-CNN.err
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --constraint=a40

source ~/hpc-share/capstone_model_training/bin/activate

python main.py