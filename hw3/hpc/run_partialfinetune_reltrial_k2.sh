#!/bin/bash
#SBATCH --job-name=structml_partialft_reltrial_k2
#SBATCH --output=logs/partialft_reltrial_k2_%j.out
#SBATCH --error=logs/partialft_reltrial_k2_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Aspect 3 follow-up: partial MiniLM fine-tune, num_unfrozen=2 only, on a dedicated
# 30,000/10,000-seed sample for rel-trial, 3 seeds = 3 runs (~21h, fits one 24h job).
# Standalone: no dependency on prep or any other job. Builds its own (separate, larger)
# A3 subgraph cache on first run - shared with the other k value for this dataset, so
# whichever of run_partialfinetune_reltrial_k1.sh / _k2.sh runs first builds the cache and
# the other reuses it (both must run on the SAME account/filesystem to share the cache;
# if splitting k1/k2 across two accounts, each side builds its own cache once).
# Run this on your own HPC account with:
#   sbatch run_partialfinetune_reltrial_k2.sh
# (Split across HPC accounts however works best: by dataset - relstack vs reltrial via
# run_partialfinetune_relstack.sh / run_partialfinetune_reltrial.sh - or by unfreeze-config
# within a dataset via this k1/k2 pair - or both, across up to 4 accounts.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML PARTIAL LLM FINE-TUNE: rel-trial (k=2)"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting at: $(date)"
echo ""

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export MPLBACKEND=Agg
export STRUCTML_ARTIFACTS="$(cd .. && pwd)/artifacts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Artifacts: $STRUCTML_ARTIFACTS"
echo ""

python hpc_partial_finetune.py rel-trial --k 2
EXIT_CODE=$?

echo ""
echo "================================================"
echo "PARTIAL LLM FINE-TUNE rel-trial k=2 COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
