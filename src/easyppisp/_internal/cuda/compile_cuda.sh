#!/bin/bash
set -e
export TORCH_CUDA_ARCH_LIST="7.5"
apt-get update && apt-get install -y python3 python3-pip
cd /workspaces/easyppisp/src/easyppisp/_internal/cuda
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python3 setup.py build_ext --inplace
