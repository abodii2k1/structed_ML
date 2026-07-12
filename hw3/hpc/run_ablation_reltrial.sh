#!/bin/bash
#SBATCH --job-name=structml_ablation_reltrial
#SBATCH --output=logs/ablation_reltrial_%j.out
#SBATCH --error=logs/ablation_reltrial_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Leave-one-relation-out edge ablation on rel-trial (hetero-SAGE, 3 seeds).
# Standalone: no dependency on prep or any other job. Builds its own graph cache on
# first run if one isn't already present. Run this on your own HPC account with:
#   sbatch run_ablation_reltrial.sh
# (Your labmate runs run_ablation_relstack.sh on theirs - the two jobs are independent.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML EDGE ABLATION: rel-trial"
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

python hpc_edge_ablation.py rel-trial
EXIT_CODE=$?

echo ""
echo "================================================"
echo "EDGE ABLATION rel-trial COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
