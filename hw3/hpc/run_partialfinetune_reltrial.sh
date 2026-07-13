#!/bin/bash
#SBATCH --job-name=structml_partialft_reltrial
#SBATCH --output=logs/partialft_reltrial_%j.out
#SBATCH --error=logs/partialft_reltrial_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Aspect 3 follow-up: partial MiniLM fine-tune (last 1 or 2 layers unfrozen) on a
# dedicated 30,000/10,000-seed sample for rel-trial, 3 seeds each = 6 runs.
# Standalone: no dependency on prep or any other job. Builds its own (separate,
# larger) A3 subgraph cache on first run. Run this on your own HPC account with:
#   sbatch run_partialfinetune_reltrial.sh
# (Your labmate runs run_partialfinetune_relstack.sh on theirs - independent jobs.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML PARTIAL LLM FINE-TUNE: rel-trial"
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

python hpc_partial_finetune.py rel-trial
EXIT_CODE=$?

echo ""
echo "================================================"
echo "PARTIAL LLM FINE-TUNE rel-trial COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
