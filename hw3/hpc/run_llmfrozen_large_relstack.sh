#!/bin/bash
#SBATCH --job-name=structml_llmfrozen_large_relstack
#SBATCH --output=logs/llmfrozen_large_relstack_%j.out
#SBATCH --error=logs/llmfrozen_large_relstack_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Aspect 3 fine-tuning-family experiment 1/3: frozen `llm` baseline on the SAME
# 30,000/10,000-seed sample and train'/val'/test protocol as the k=1/k=2 partial
# fine-tune (run_partialfinetune_relstack.sh), for rel-stack, 3 seeds. Together the two
# scripts give the 3 experiments for this dataset (frozen, k=1, k=2) - run both on your
# own account:
#   sbatch run_llmfrozen_large_relstack.sh
#   sbatch run_partialfinetune_relstack.sh
# (Your labmate runs the *_reltrial.sh pair on theirs.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML FROZEN LLM BASELINE (large sample): rel-stack"
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

python hpc_llm_frozen_large.py rel-stack
EXIT_CODE=$?

echo ""
echo "================================================"
echo "FROZEN LLM BASELINE (large sample) rel-stack COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
