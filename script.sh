#!/bin/bash

set -e

echo "AgentEditor Installation Script"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo ""
echo "[1/5] Creating conda environment 'agenteditor'..."
if conda env list | grep -q "^agenteditor "; then
    echo "Warning: Environment 'agenteditor' already exists."
    read -p "Do you want to remove it and reinstall? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n agenteditor -y
        conda create -n agenteditor python=3.10 -y
    else
        echo "Using existing environment."
    fi
else
    conda create -n agenteditor python=3.10 -y
fi

# Activate environment
echo ""
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agenteditor

# Install PyTorch
echo ""
echo "[2/5] Installing PyTorch..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -ne 0 ]; then
    echo "Error: PyTorch installation failed!"
    exit 1
fi

# Install Python dependencies
echo ""
echo "[3/5] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies may have conflicts."
    echo "Attempting to resolve..."
fi

# Download SAM weights
echo ""
echo "[4/5] Downloading SAM model weights..."
mkdir -p ckpts

if [ ! -f "ckpts/sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM ViT-H weights..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ckpts/sam_vit_h_4b8939.pth
else
    echo "SAM ViT-H weights already exist, skipping download."
fi

# Install submodules (CUDA extensions)
echo ""
echo "[5/5] Installing submodules..."
echo "This may take several minutes. Please ensure you have:"
echo "  - CUDA toolkit installed (matching PyTorch CUDA version)"
echo "  - C++ compiler (gcc/g++)"
echo ""

echo "Installing diff-gaussian-rasterization-lang..."
if [ -d "submodules/diff-gaussian-rasterization-lang" ]; then
    cd submodules/diff-gaussian-rasterization-lang
    pip install -e .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install diff-gaussian-rasterization-lang!"
        echo "This is a critical dependency. Please check:"
        echo "  1. CUDA toolkit is installed and matches PyTorch"
        echo "  2. gcc/g++ compiler is available"
        echo "  3. nvcc is in PATH"
        exit 1
    fi
    cd ../..
else
    echo "Warning: submodules/diff-gaussian-rasterization-lang not found!"
    echo "Please run: git submodule update --init --recursive"
fi

echo ""
echo "Installing simple-knn..."
if [ -d "submodules/simple-knn" ]; then
    cd submodules/simple-knn
    pip install -e .
    cd ../..
fi

echo ""
echo "Installing segment-anything-langsplat..."
if [ -d "submodules/segment-anything-langsplat" ]; then
    cd submodules/segment-anything-langsplat
    pip install -e .
    cd ../..
fi

echo ""
echo "Installing MobileSAM-lang..."
if [ -d "submodules/MobileSAM-lang" ]; then
    cd submodules/MobileSAM-lang
    pip install -e .
    if [ -f "weights/mobile_sam.pt" ]; then
        cp weights/mobile_sam.pt ../../ckpts/
        echo "MobileSAM weights copied to ckpts/"
    fi
    cd ../..
fi

echo ""
echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate agenteditor"
echo ""
