#!/bin/bash
#SBATCH --job-name=structml_finetune_reltrial
#SBATCH --output=logs/finetune_reltrial_%j.out
#SBATCH --error=logs/finetune_reltrial_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

# Aspect 3 follow-up: fine-tuned MiniLM (llm-finetuned) on rel-trial, 3 seeds.
# Standalone: no dependency on prep or any other job. Builds/upgrades its own cached
# A3 subgraph on first run if needed. Run this on your own HPC account with:
#   sbatch run_finetune_reltrial.sh
# (Your labmate runs run_finetune_relstack.sh on theirs - the two jobs are independent.)

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML LLM FINE-TUNE: rel-trial"
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

python hpc_finetune_llm.py rel-trial
EXIT_CODE=$?

echo ""
echo "================================================"
echo "LLM FINE-TUNE rel-trial COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
