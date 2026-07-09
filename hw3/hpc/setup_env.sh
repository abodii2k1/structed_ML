#!/bin/bash
# One-time environment setup on the HPC (run on the login node, needs internet).
# Recreates the exact local environment: torch 2.11 (CUDA 13) + PyG 2.7 stack.
set -e

CONDA_ENV=${CONDA_ENV:-structml}

eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV python=3.11 -y
conda activate $CONDA_ENV

# torch wheel bundles its own CUDA runtime - no cluster CUDA module needed
pip install torch==2.11.0

# PyG + the neighbor-sampling backend (pyg_lib preferred, torch-sparse as fallback;
# NeighborLoader needs at least one of them)
pip install torch_geometric==2.7.0
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.11.0+cu130.html \
  || pip install torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu130.html

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
