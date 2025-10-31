#!/bin/bash

# Script to download the Stable Diffusion model weights

# Define the URL for the Stable Diffusion weights
WEIGHTS_URL="https://path_to_stable_diffusion_weights/sd-v1-4.ckpt"

# Define the destination directory
DEST_DIR="./stable_diffusion/checkpoints"

# Create the destination directory if it doesn't exist
mkdir -p $DEST_DIR

# Download the weights
echo "Downloading Stable Diffusion weights..."
curl -L $WEIGHTS_URL -o "$DEST_DIR/sd-v1-4.ckpt"

echo "Download complete!"