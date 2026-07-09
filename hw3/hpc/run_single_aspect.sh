#!/bin/bash
#SBATCH --job-name=structml_aspect
#SBATCH --output=logs/aspect_%x_%j.out
#SBATCH --error=logs/aspect_%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Usage: sbatch run_single_aspect.sh <prep|a1|a2|a3|a4>
ASPECT=$1
if [ -z "$ASPECT" ]; then
    echo "Error: missing aspect argument (prep|a1|a2|a3|a4)"
    exit 1
fi

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML ASPECT RUN: $ASPECT"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting at: $(date)"
echo ""

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

cd "$(dirname "$0")"
export MPLBACKEND=Agg
export STRUCTML_ARTIFACTS="$(cd .. && pwd)/artifacts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Artifacts: $STRUCTML_ARTIFACTS"
echo ""

python hpc_run_aspect.py "$ASPECT"
EXIT_CODE=$?

echo ""
echo "================================================"
echo "ASPECT $ASPECT COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
