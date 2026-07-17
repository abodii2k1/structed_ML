#!/bin/bash
# One-time environment setup on the HPC (run on the login node, needs internet).
# Recreates the exact local environment: torch 2.11 (CUDA 13) + PyG 2.7 stack.
set -e

CONDA_ENV=${CONDA_ENV:-structml}

eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV python=3.11 -y
conda activate $CONDA_ENV

# torch wheel bundles its own CUDA runtime - no cluster CUDA module needed.
# IMPORTANT: pick a CUDA build the node's GPU DRIVER actually supports. Check with
# `nvidia-smi` (top-right "CUDA Version: X.Y") BEFORE running this script. The
# default `pip install torch` grabs a CUDA-13 build; if the driver only supports
# 12.x (common on older cluster nodes), that build silently falls back to CPU
# (no error - just device: cpu in the logs and 10-50x slower training). Below
# targets CUDA 12.1, which works with any driver reporting CUDA 12.1 or newer.
# If `nvidia-smi` shows something older than 12.1, use cu118 instead (edit both
# lines below from "cu121" to "cu118" and drop the trailing "+cu121" from torch).
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121

# PyG + the neighbor-sampling backend (pyg_lib preferred, torch-sparse as fallback;
# NeighborLoader needs at least one of them)
pip install torch_geometric==2.7.0
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.11.0+cu121.html \
  || pip install torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu121.html

pip install relbench==2.1.2 pytorch-frame==0.3.0 sentence-transformers==5.6.0 \
            scikit-learn==1.6.1 pandas matplotlib nbformat

python - << 'EOF'
import torch, torch_geometric
print("torch", torch.__version__, "| cuda available at runtime:", torch.cuda.is_available())
print("pyg", torch_geometric.__version__, "| pyg-lib:", torch_geometric.typing.WITH_PYG_LIB,
      "| torch-sparse:", torch_geometric.typing.WITH_TORCH_SPARSE)
EOF

echo ""
echo "Env '$CONDA_ENV' ready. If pyg-lib AND torch-sparse both show False above,"
echo "neighbor sampling will fail - install one of them for your exact torch+CUDA"
echo "combination from https://data.pyg.org/whl/"
