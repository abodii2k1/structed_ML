#!/bin/bash

# Submit the full converged-protocol rerun:
#   1 prep job (downloads data, builds graph caches + A3 subgraphs if missing)
#   then the 4 aspect jobs IN PARALLEL (they write to separate result/curve files,
#   and only read the shared caches, so parallel is safe).
#
# If you rsync-ed artifacts/graph_cache and artifacts/aspect3_subgraphs from the
# local machine, prep finds everything cached and finishes in ~2 minutes.

cd "$(dirname "$0")"
mkdir -p logs

echo "================================================"
echo "STRUCTML CONVERGED-PROTOCOL RERUN"
echo "================================================"
echo "Protocol: 30 epochs, patience 6, 3 seeds x both datasets, loss curves logged"
echo ""

PREP_ID=$(sbatch --parsable --job-name=prep run_single_aspect.sh prep)
echo "Submitted prep - Job ID: $PREP_ID"

ASPECT_IDS=()
for a in a1 a2 a3 a4; do
    JOB_ID=$(sbatch --parsable --job-name=$a --dependency=afterok:$PREP_ID run_single_aspect.sh $a)
    ASPECT_IDS+=($JOB_ID)
    echo "Submitted $a - Job ID: $JOB_ID (waits for prep)"
done

echo ""
echo "================================================"
echo "ALL JOBS SUBMITTED"
echo "================================================"
echo "prep: $PREP_ID | aspects: ${ASPECT_IDS[@]}"
echo ""
echo "Monitor with:  watch -n 30 'squeue -u \$USER'"
echo "View logs:     tail -f logs/aspect_*.out"
echo ""
echo "Expected times (on a modern HPC GPU):"
echo "  prep: ~2 min if caches were rsync-ed, else 1-3 h to build"
echo "  a1: ~5-10 h   a2: ~4-8 h   a3: ~1 h   a4: ~8-15 h  (all run in parallel)"
echo ""
echo "Everything is resumable: if a job dies, just resubmit the same aspect -"
echo "it skips finished runs and continues."
echo ""
echo "When done, copy results back to the local machine:"
echo "  rsync -av <server>:$(cd .. && pwd)/artifacts/*.csv ~/structed_ML/hw3/artifacts/"
