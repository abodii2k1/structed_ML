#!/bin/bash
#SBATCH --job-name=structml_sagebasis_reltrial
#SBATCH --output=logs/sagebasis_reltrial_%j.out
#SBATCH --error=logs/sagebasis_reltrial_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Aspect 2 follow-up: SAGE basis-decomposition sweep (num_bases in {1,2,4,8,30}) on
# rel-trial, 3 seeds each = 15 runs. Standalone: no dependency on prep or any other job.
# Run this on your own HPC account with:
#   sbatch run_sagebasis_reltrial.sh
# (Your labmate runs run_sagebasis_relstack.sh on theirs - the two jobs are independent.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML SAGE BASIS SWEEP: rel-trial"
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

python hpc_sage_basis.py rel-trial
EXIT_CODE=$?

echo ""
echo "================================================"
echo "SAGE BASIS SWEEP rel-trial COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
