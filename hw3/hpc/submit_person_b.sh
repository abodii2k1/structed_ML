#!/bin/bash

# ============ PERSON B: Aspects 1, 2, 4 + the basis sweep (~31 GPU-h, ~12h wall-clock) ============
#
# Why this grouping: none of these need Aspect 3's big subgraph cache - they stream from the
# plain graph cache instead, so this account never builds or stores the multi-GB A3 blobs.
# All five jobs run in parallel once prep is done (they write to separate result files).
#
#   prep          - datasets + graph caches (~2 min if rsynced, else 1-3h)
#   a1            - 6 directionality variants x 2 datasets x 3 seeds   (~7.5h)
#   a2            - 4 homo/hetero variants x 2 x 3 + param control     (~6h)
#   a4            - 12 depth/skip settings x 2 x 3                     (~11.5h)
#   basis rel-stack - num_bases {8, 12, 16} x 3 seeds                  (~5.7h)
#   basis rel-trial - num_bases {12, 16, 20} x 3 seeds                 (~0.1h)
#
# Everything is resumable: if a job dies, resubmit it - finished runs are skipped.

cd "$(dirname "$0")"
mkdir -p logs

echo "================================================"
echo "PERSON B: Aspects 1, 2, 4 + SAGE basis sweep"
echo "================================================"
echo "Protocol: 30 epochs, patience 6, 3 seeds, train -> val (early stop) -> test (task.evaluate)"
echo ""

# None of B's jobs touch the Aspect 3 subgraph, so don't spend hours building it here.
# Passed explicitly rather than relying on the environment propagating to the job.
PREP_ID=$(sbatch --parsable --job-name=prep \
    --export=ALL,STRUCTML_SKIP_A3_PREP=1 run_single_aspect.sh prep)
echo "Submitted prep - Job ID: $PREP_ID (skipping A3 subgraph - not needed on this account)"

IDS=()
for a in a1 a2 a4; do
    JOB_ID=$(sbatch --parsable --job-name=$a --dependency=afterok:$PREP_ID run_single_aspect.sh $a)
    IDS+=($JOB_ID)
    echo "Submitted $a - Job ID: $JOB_ID (waits for prep)"
done

for ds in relstack reltrial; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID run_sagebasis_${ds}.sh)
    IDS+=($JOB_ID)
    echo "Submitted basis $ds - Job ID: $JOB_ID (waits for prep)"
done

echo ""
echo "prep: $PREP_ID | jobs: ${IDS[@]}"
echo ""
echo "Monitor:   watch -n 30 'squeue -u \$USER'"
echo "Logs:      tail -f logs/aspect_*.out logs/sagebasis_*.out"
echo ""
echo "When done, send results home:"
echo "  rsync -av <server>:$(cd .. && pwd)/artifacts/aspect{1,2,4}_results.csv \\"
echo "            <server>:$(cd .. && pwd)/artifacts/loss_curves_A{1,2,4}.csv \\"
echo "            <server>:$(cd .. && pwd)/artifacts/aspect2_sage_basis_*.csv \\"
echo "            ~/structed_ML/hw3/artifacts/"
