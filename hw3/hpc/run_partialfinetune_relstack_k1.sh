#!/bin/bash
#SBATCH --job-name=structml_partialft_relstack_k1
#SBATCH --output=logs/partialft_relstack_k1_%j.out
#SBATCH --error=logs/partialft_relstack_k1_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=2

# Aspect 3 supplementary: partial MiniLM fine-tune (last 1 layer unfrozen) for rel-stack,
# 3 seeds = 3 runs. Runs on the SAME unified 30,000/10,000/20,000 sample and the SAME
# protocol as id/column/llm, reported on the held-out test split via task.evaluate() - so
# the only thing that differs from frozen `llm` is the adaptation.
#
# WALLTIME: this is the tightest job in the project - ~19.8h measured (6.6h/seed) against
# the cluster's 24h cap. That should fit, but there is little margin if the node is slower.
# It is safe either way: results are written after EVERY seed, so if it hits the wall
# partway, just resubmit - finished seeds are skipped and it continues from where it
# stopped. Two resubmits would cover the worst realistic case.
#
# Needs the unified A3 subgraph cache. Submit via submit_person_a.sh, which runs
# run_aspect3.sh first and makes this wait for it - launching this and the rel-trial k=1
# job before the cache exists would have them race to build the same thing.

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML PARTIAL LLM FINE-TUNE: rel-stack (k=1)"
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

python hpc_partial_finetune.py rel-stack --k 1
EXIT_CODE=$?

echo ""
echo "================================================"
echo "PARTIAL LLM FINE-TUNE rel-stack k=1 COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
