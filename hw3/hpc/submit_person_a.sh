#!/bin/bash

# ===================== PERSON A: all of Aspect 3 (~26 GPU-h, ~20h wall-clock) =====================
#
# Why this grouping: every Aspect 3 job needs the unified subgraph cache
# (artifacts/aspect3_subgraphs_v2/, several GB per dataset and expensive to build - it runs
# MiniLM over every node). Keeping all A3 work on one account builds that cache exactly once
# per dataset instead of duplicating it across accounts. Person B's jobs need only the plain
# graph cache, never this one.
#
#   1. a3        - id/column/llm, both datasets, 3 seeds  (~1-2h + cache build)
#                  ALSO builds the A3 subgraph cache the k=1 jobs depend on.
#   2. k1 jobs   - partial fine-tune (last MiniLM layer), launched in parallel once (1) is done
#                  rel-stack ~19.8h  |  rel-trial ~5h
#
# Everything is resumable: if a job dies, resubmit it - finished runs are skipped.

cd "$(dirname "$0")"
mkdir -p logs

echo "================================================"
echo "PERSON A: Aspect 3 (official + k=1 fine-tune)"
echo "================================================"
echo "Protocol: 30 epochs, patience 6, 3 seeds, train -> val (early stop) -> test (task.evaluate)"
echo ""

A3_ID=$(sbatch --parsable run_aspect3.sh)
echo "Submitted a3 (builds the A3 cache too) - Job ID: $A3_ID"

K1_STACK=$(sbatch --parsable --dependency=afterok:$A3_ID run_partialfinetune_relstack_k1.sh)
echo "Submitted k=1 rel-stack - Job ID: $K1_STACK (waits for a3)"

K1_TRIAL=$(sbatch --parsable --dependency=afterok:$A3_ID run_partialfinetune_reltrial_k1.sh)
echo "Submitted k=1 rel-trial - Job ID: $K1_TRIAL (waits for a3)"

echo ""
echo "Monitor:   watch -n 30 'squeue -u \$USER'"
echo "Logs:      tail -f logs/a3_*.out logs/partialft_*_k1_*.out"
echo ""
echo "Expected:  a3 ~1-2h (+ cache build)  |  k=1 rel-stack ~19.8h  |  k=1 rel-trial ~5h"
echo ""
echo "When done, send results home:"
echo "  rsync -av <server>:$(cd .. && pwd)/artifacts/aspect3_*results.csv \\"
echo "            <server>:$(cd .. && pwd)/artifacts/loss_curves_A3.csv \\"
echo "            <server>:$(cd .. && pwd)/artifacts/aspect3_partial_finetune_*.csv \\"
echo "            ~/structed_ML/hw3/artifacts/"
