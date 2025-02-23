#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=Train_CNN
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00

# Define variables
BATCH_FILE="batchfile-here.py"
MODEL_NAME="b3-[namehere]"
# DATASET=~/hpc-share/tiles/gDenoised/ Will copy this folder. options: gDenoised, gRaw, sDenoised, sRaw
DATASET=~/hpc-share/tiles/gDenoised/
GITHUB_URL="https://raw.githubusercontent.com/OSU-Enhancing-Deformation-Analysis/CNN-motion-model/refs/heads/main/batch_models/$BATCH_FILE"
PYTHON_ENV=~/hpc-share/capstone_model_training/bin/activate


# Download the batch file using wget with the token
wget -O "$BATCH_FILE" "$GITHUB_URL"
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to download batch file!"
    exit 1
fi

# Source the Python environment
if [[ -f "$PYTHON_ENV" ]]; then
    source "$PYTHON_ENV"
else
    echo "Error: Python environment not found!"
    exit 1
fi

# Copy the tile dataset and rename it to 'tiles'
if [[ -d "$DATASET" ]]; then
    cp -r "$DATASET" ./tiles
else
    echo "Error: Tile dataset not found!"
    exit 1
fi

echo "Setup complete!"

python "$BATCH_FILE" "$MODEL_NAME"