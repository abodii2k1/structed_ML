# Running the held-out-test rerun on the HPC

What this runs: **every experiment in the project, re-run under a proper three-way split** -
train on `train`, select the checkpoint on `val` (early stopping), then score that checkpoint
**once** on RelBench's real held-out `test` split via `task.evaluate()`.

Protocol, identical everywhere: **30 epochs max, early stopping patience 6, 3 seeds
(42/43/44), both datasets, per-epoch loss curves logged** (Aspect 4 additionally caps
1000 steps/epoch). The scripts execute the actual notebook cells (`final.ipynb` is the
single source of truth), so what runs on the cluster is byte-identical to the notebook.

## Why the rerun

The previous protocol used the same validation split twice: once to pick the best epoch
(early stopping) and again as the reported number. That is selection leakage - the reported
score belongs to whichever checkpoint happened to fit that particular validation set best,
so it is optimistically biased.

RelBench masks the test target column by default (`get_table("test")` returns input columns
only), but `EntityTask.evaluate(pred)` loads the real labels internally and scores against
them without ever exposing them. That gives a genuinely once-touched test number, and it
needs no data sacrificed from train.

Two things were verified before committing to this (see `final.ipynb`):
- the test labels really are present and complete (825 rel-trial / 88,137 rel-stack rows, zero nulls);
- **order alignment** - `task.evaluate()` matches `pred[i]` to `target[i]` positionally, so the
  test loader must yield seeds in test-table row order. Proven with an oracle model that emits
  each seed's own entity id: all 825 rel-trial positions came back exactly equal. A silent
  misalignment here would have produced plausible-looking ~0.5 AUROCs that were pure garbage.

## What changed in Aspect 3

Aspect 3 now uses **one unified sample per dataset** for every variant (`id`, `column`, `llm`,
`llm-partial`), so the only thing that differs between them is the strategy:

| | train | val | test |
|---|---|---|---|
| rel-stack | 30,000 | 10,000 | 20,000 |
| rel-trial | 11,994 *(capped: all of train)* | 960 *(all)* | 825 *(all)* |

- train/val are label-stratified; **test is sampled uniformly** - stratifying it would mean
  choosing which test rows to be scored on by looking at their labels.
- This also fixes a real weakness: the old 2,000-seed val had only **56 positives** on
  rel-stack, and since all 3 seeds shared that same fixed subsample, the reported +/- std
  measured only training randomness and not sampling noise at all. The new val has ~281.
- The cache (`artifacts/aspect3_subgraphs_v2/`) carries column TensorFrames + frozen MiniLM
  embeddings + tokenized text for every node, so it is several GB per dataset. Tokens are
  stored narrow (int32 ids / int8 masks, cast back to long at use) - at this size that is a
  multi-GB saving.

## 1. Send the code (and optionally the caches) to the server

```bash
# code + small artifacts
rsync -av --progress --exclude 'artifacts/graph_cache' --exclude 'artifacts/aspect3_subgraphs*' \
    ~/structed_ML/hw3 <user>@<server>:~/structed_ML/

# OPTIONAL but recommended - the prebuilt graph caches (~12 GB).
# Skipping this is fine: the prep job rebuilds them on the cluster (~1-3 h).
rsync -av --progress ~/structed_ML/hw3/artifacts/graph_cache \
    <user>@<server>:~/structed_ML/hw3/artifacts/
```

## 2. One-time environment setup (login node, has internet)

```bash
cd ~/structed_ML/hw3/hpc
bash setup_env.sh          # creates conda env "structml"
```

## 3. Submit - split across two accounts

The work splits so that **the expensive Aspect 3 subgraph cache is built on one account only**.
Person B's jobs stream from the plain graph cache and never touch it.

```bash
# Person A  (~26 GPU-h, ~20 h wall-clock)
bash submit_person_a.sh

# Person B  (~31 GPU-h, ~12 h wall-clock)
bash submit_person_b.sh
```

| | job | what | est. time |
|---|---|---|---|
| **A** | a3 | id/column/llm x 2 datasets x 3 seeds (**builds the A3 cache**) | ~1-2 h + build |
| **A** | k=1 rel-stack | partial fine-tune, last MiniLM layer | ~19.8 h |
| **A** | k=1 rel-trial | partial fine-tune, last MiniLM layer | ~5 h |
| **B** | prep | datasets + graph caches (skips the A3 cache) | ~2 min cached / 1-3 h fresh |
| **B** | a1 | 6 directionality variants x 2 x 3 | ~7.5 h |
| **B** | a2 | 4 homo/hetero variants x 2 x 3 + param control | ~6 h |
| **B** | a4 | 12 depth/skip settings x 2 x 3 | ~11.5 h |
| **B** | basis rel-stack | num_bases {8, 12, 16} x 3 seeds | ~5.7 h |
| **B** | basis rel-trial | num_bases {12, 16, 20} x 3 seeds | ~0.1 h |

On A, the two k=1 jobs wait on `a3` (`--dependency=afterok`) because all three need the same
A3 cache - launching them together would race to build it. On B, everything runs in parallel
after prep.

The basis sweep is restricted to the region that answers the question: rel-stack's relations
are severely imbalanced in edge count (232x between largest and smallest), which is the
fragmentation being tested, so it keeps the mid-range where the full-vs-shared transition
happens; rel-trial's are comparatively balanced (~10x, no severe outlier), so only the top of
its range is worth confirming.

Everything is resumable - if a job dies or hits walltime, resubmit it; finished runs are
skipped.

## 4. Bring the results home

```bash
# from person A
rsync -av <server>:~/structed_ML/hw3/artifacts/aspect3_results.csv \
          <server>:~/structed_ML/hw3/artifacts/loss_curves_A3.csv \
          <server>:~/structed_ML/hw3/artifacts/aspect3_partial_finetune_*.csv \
          ~/structed_ML/hw3/artifacts/

# from person B
rsync -av <server>:~/structed_ML/hw3/artifacts/aspect{1,2,4}_results.csv \
          <server>:~/structed_ML/hw3/artifacts/loss_curves_A{1,2,4}.csv \
          <server>:~/structed_ML/hw3/artifacts/aspect2_sage_basis_*.csv \
          ~/structed_ML/hw3/artifacts/
```

The previous val-only-protocol results are kept locally in
`artifacts/_archive_valonly_protocol/` for comparison. They were archived (not deleted) so
the resume logic cannot mistake them for finished runs under the new protocol.

## Outputs

Every results CSV now carries both the selection-split and the held-out-test metrics:

- `AUROC`, `AUPRC`, `precision`, `recall` - on **val** (the checkpoint-selection split)
- `test_roc_auc`, `test_average_precision`, `test_accuracy`, `test_f1` - from
  `task.evaluate()` on the **held-out test split**
- `test_precision_valthr`, `test_recall_valthr` - test precision/recall at the best-F1
  threshold chosen on val

Note RelBench's own `test_f1`/`test_accuracy` use a **hardcoded 0.5 threshold**, so they are
not comparable to this report's val precision/recall (which use a val-tuned threshold) - that
is what `test_precision_valthr`/`test_recall_valthr` are for.

Files:
- `artifacts/aspect{1,2,3,4}_results.csv` - final metrics, one row per run
- `artifacts/loss_curves_A{1,2,3,4}.csv` - per-epoch train loss / val loss / val AUROC
- `artifacts/aspect2_sage_basis_*.csv` - basis sweep
- `artifacts/aspect3_partial_finetune_*.csv` - k=1 partial fine-tune
- `logs/*.out` - live training prints per job
