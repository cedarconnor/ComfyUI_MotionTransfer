#!/bin/bash
#
# Build CUDA kernels for Motion Transfer nodes (Linux/macOS)
#
# Requirements:
#   - CUDA Toolkit 11.x or 12.x
#   - nvcc compiler
#   - Compatible GPU (compute capability >= 5.0)
#
# Usage:
#   chmod +x build.sh
#   ./build.sh
#

set -e

echo "Building Motion Transfer CUDA kernels..."

# Detect CUDA architecture (adjust for your GPU)
# Common values:
#   - RTX 30xx/40xx: sm_86, sm_89
#   - RTX 20xx: sm_75
#   - GTX 10xx: sm_61
#   - Automatic detection
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Detected GPU: $GPU_NAME"

    # Map GPU to compute capability (heuristic)
    if [[ "$GPU_NAME" == *"RTX 40"* ]]; then
        SM_ARCH="sm_89"
    elif [[ "$GPU_NAME" == *"RTX 30"* ]]; then
        SM_ARCH="sm_86"
    elif [[ "$GPU_NAME" == *"RTX 20"* ]]; then
        SM_ARCH="sm_75"
    elif [[ "$GPU_NAME" == *"GTX 10"* ]]; then
        SM_ARCH="sm_61"
    else
        SM_ARCH="sm_75"  # Default to RTX 2000 series
        echo "Warning: Unknown GPU, using default sm_75"
    fi
else
    SM_ARCH="sm_75"
    echo "Warning: nvidia-smi not found, using default sm_75"
fi

echo "Compiling for architecture: $SM_ARCH"

# Compile TileWarp kernel
echo "[1/3] Compiling tile_warp_kernel.cu..."
nvcc -shared -Xcompiler -fPIC \
    -arch=$SM_ARCH \
    -O3 \
    -o libtile_warp_kernel.so \
    tile_warp_kernel.cu

# Compile BarycentricWarp kernel
echo "[2/3] Compiling barycentric_warp_kernel.cu..."
nvcc -shared -Xcompiler -fPIC \
    -arch=$SM_ARCH \
    -O3 \
    -o libbarycentric_warp_kernel.so \
    barycentric_warp_kernel.cu

# Compile GuidedFilter kernel
echo "[3/3] Compiling guided_filter_kernel.cu..."
nvcc -shared -Xcompiler -fPIC \
    -arch=$SM_ARCH \
    -O3 \
    -o libguided_filter_kernel.so \
    guided_filter_kernel.cu

echo "✓ Build complete! CUDA kernels compiled successfully."
echo ""
echo "Generated libraries:"
ls -lh *.so
echo ""
echo "You can now use CUDA acceleration in ComfyUI Motion Transfer nodes."
