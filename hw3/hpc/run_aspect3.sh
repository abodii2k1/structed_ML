#!/bin/bash
#SBATCH --job-name=structml_a3
#SBATCH --output=logs/a3_%j.out
#SBATCH --error=logs/a3_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4

# Aspect 3 official: id / column / llm x 2 datasets x 3 seeds, on the unified
# 30,000/10,000/20,000 sample, reported on the held-out test split via task.evaluate().
#
# Run this FIRST on person A's account: it builds the unified Aspect 3 subgraph cache
# (artifacts/aspect3_subgraphs_v2/) that the k=1 fine-tune jobs also need. Launching the
# k=1 jobs before this finishes would have them race to build the same cache.
# submit_person_a.sh wires that dependency for you.
#
# Memory: the unified cache carries column TensorFrames + frozen MiniLM embeddings +
# tokenized text for every node, so it is several GB per dataset - hence --mem=96G.

CONDA_ENV=${CONDA_ENV:-structml}

echo "================================================"
echo "STRUCTML ASPECT 3 (id/column/llm, held-out test)"
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
echo "Artifacts: $STRUCTML_ARTIFACTS"
echo ""

python hpc_run_aspect.py a3
EXIT_CODE=$?

echo ""
echo "================================================"
echo "ASPECT 3 COMPLETE - exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================================"
exit $EXIT_CODE
