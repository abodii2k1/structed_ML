#!/bin/bash
#SBATCH --job-name=structml_llmfrozen_large_reltrial
#SBATCH --output=logs/llmfrozen_large_reltrial_%j.out
#SBATCH --error=logs/llmfrozen_large_reltrial_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Aspect 3 fine-tuning-family experiment 1/3: frozen `llm` baseline on the SAME
# 30,000/10,000-seed sample and train'/val'/test protocol as the k=1/k=2 partial
# fine-tune (run_partialfinetune_reltrial.sh), for rel-trial, 3 seeds. Together the two
# scripts give the 3 experiments for this dataset (frozen, k=1, k=2) - run both on your
# own account:
#   sbatch run_llmfrozen_large_reltrial.sh
#   sbatch run_partialfinetune_reltrial.sh
# (Your labmate runs the *_relstack.sh pair on theirs.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML FROZEN LLM BASELINE (large sample): rel-trial"
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

python hpc_llm_frozen_large.py rel-trial
EXIT_CODE=$?

echo ""
echo "================================================"
echo "FROZEN LLM BASELINE (large sample) rel-trial COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
