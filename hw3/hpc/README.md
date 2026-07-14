# Running the converged-protocol rerun on the HPC

What this runs: every official experiment of all 4 aspects at the converged
protocol - **30 epochs max, early stopping patience 6, 3 seeds (42/43/44) on both
datasets, per-epoch loss curves logged** (Aspect 4 additionally: 1000 steps/epoch).
The scripts execute the actual notebook cells (`final.ipynb` is the single source
of truth), so what runs on the cluster is byte-identical to the notebook.

## 1. Send the code (and optionally the caches) to the server

```bash
# code + small artifacts (~400 MB, includes the A3 subgraphs)
rsync -av --progress --exclude 'artifacts/graph_cache' \
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

## 3. Submit everything

```bash
cd ~/structed_ML/hw3/hpc
bash submit_all.sh
```

This submits: `prep` (data download + cache build, skipped if caches present),
then `a1 a2 a3 a4` in parallel once prep succeeds.

| job | what | est. time |
|---|---|---|
| prep | datasets, graph caches, A3 subgraphs | ~2 min cached / 1-3 h fresh |
| a1 | 6 directionality variants x 2 datasets x 3 seeds | 5-10 h |
| a2 | 4 homo/hetero variants x 2 x 3 + param-matched control | 4-8 h |
| a3 | id/column/llm x 2 x 3 | ~1 h |
| a4 | 12 depth/skip settings x 2 x 3 | 8-15 h |

Everything is resumable - if a job dies or hits walltime, resubmit the same
aspect: `sbatch --job-name=a4 run_single_aspect.sh a4`. Finished runs are skipped.

## 4. Bring the results home

```bash
rsync -av <user>@<server>:~/structed_ML/hw3/artifacts/aspect*_results.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/loss_curves_A*.csv \
          ~/structed_ML/hw3/artifacts/
```

(The old 10-epoch results stay archived locally as `artifacts/*_10ep.csv` for the
budget-sensitivity comparison in the report.)

## Outputs

- `artifacts/aspect{1,2,3,4}_results.csv` - final metrics, one row per run
- `artifacts/loss_curves_A{1,2,3,4}.csv` - per-epoch train loss / val loss / val AUROC
- `logs/aspect_*.out` - live training prints per job

## Supplementary: fine-tuned LLM encoding for Aspect 3 (2 more independent jobs)

Tests whether Aspect 3's `llm` strategy loses to `column` because serializing a row to
text loses structure, or because the frozen MiniLM embedding just can't adapt to the
task. Fine-tunes MiniLM (~22.7M params) end-to-end instead of using the official
strategy's frozen precomputed embedding. See the `a3-finetune-0` markdown cell in
`final.ipynb` for the full reasoning and the empirical checks behind the implementation
(PyG can't slice raw text lists, so text is pre-tokenized into fixed-length tensors
instead; discriminative LR so fine-tuning doesn't wreck the pretrained weights).

Split by dataset into two standalone jobs, no shared dependency, so two people on two
different HPC accounts can each run one:

```bash
# on your account
sbatch run_finetune_relstack.sh

# on your labmate's account
sbatch run_finetune_reltrial.sh
```

Heavier than everything else in this project: MiniLM's transformer now runs a live
forward+backward pass every mini-batch (vs. a cached lookup + one linear layer for the
frozen `llm` strategy), so expect this to be noticeably slower per epoch.

Bring the results home the same way:

```bash
rsync -av <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_finetune_results.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_finetune_loss_curves.csv \
          ~/structed_ML/hw3/artifacts/
```

Output: `artifacts/aspect3_finetune_results.csv` (one row per seed: AUROC, AUPRC,
precision, recall, learned params, train time) and
`artifacts/aspect3_finetune_loss_curves.csv` (per-epoch curves, same format as the
official aspects) - kept separate from `aspect3_results.csv` since this is a
supplementary follow-up, not an official strategy.

## Supplementary: SAGE basis-decomposition sweep for Aspect 2 (2 more independent jobs)

Motivated by the edge-count listing: rel-stack's 11 relations are severely imbalanced
(232x gap between the largest, `votes.PostId->posts`, and the smallest,
`votes.UserId->users`), while rel-trial's 15 are comparatively balanced (10x
top-to-bottom, no severe outlier). Tests whether basis decomposition (the original
R-GCN fix for exactly this failure mode - thin-data relations share a small set of
basis matrices instead of each estimating a full independent one) helps, and whether
it helps differently on the two datasets given how differently skewed their edge
counts are.

Deliberately built on **GraphSAGE**, not RGCN: the assignment's Aspect 2 model list is
GraphSAGE/GAT/HGT (RGCN is only listed under Aspect 1), so this is a hand-built
`BasisSAGEConv` that reproduces `SAGEConv`'s exact formula
(`out = lin_l(mean_neighbors) + lin_r(self)`, no bias on the self term) but constructs
each relation's `lin_l`/`lin_r` weight matrices from a small shared set of basis
matrices instead of each relation owning fully independent ones - same SAGE
computation as the official hetero model, only the parameterization of the weights
changes. `num_bases = num_relations` is designed to reproduce the official hetero
SAGE result's structure (every relation can still learn its own independent weights,
just re-expressed through a large-enough basis) as the "no sharing" reference point.
See the `sage-basis-0` markdown cell in `final.ipynb` for the full reasoning and the
empirical checks behind it (verified `scatter(..., reduce='mean')` returns zero, not
NaN, for destination nodes with no contributing edges, and smoke-tested the whole
forward/backward pass on real data before committing to the full sweep).

Sweeps `num_bases` in `{1, 2, 4, 8, 12, 16, 20, 22}` for rel-stack and
`{1, 2, 4, 8, 12, 16, 20, 24, 28, 30}` for rel-trial x 3 seeds (24-30 runs per dataset)
- denser near the top end than a naive log-spaced sweep, since the gap between 8 and the
full relation count is where the full-vs-shared transition actually happens and was
originally the least-resolved part of the curve. `run_sage_basis()` is resumable, so
rel-trial's already-collected `{1,2,4,8,30}` points are skipped, not recomputed. Same
split-by-dataset pattern as the other
two supplementary studies:

```bash
# on your account
sbatch run_sagebasis_relstack.sh

# on your labmate's account
sbatch run_sagebasis_reltrial.sh
```

Bring the results home the same way:

```bash
rsync -av <user>@<server>:~/structed_ML/hw3/artifacts/aspect2_sage_basis_results.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/aspect2_sage_basis_loss_curves.csv \
          ~/structed_ML/hw3/artifacts/
```

Output: `artifacts/aspect2_sage_basis_results.csv` (one row per num_bases per seed) and
`artifacts/aspect2_sage_basis_loss_curves.csv` (per-epoch curves) - kept separate from
`aspect2_results.csv` since this is a supplementary follow-up, not an official variant.

## Supplementary: partial LLM fine-tune + more data for Aspect 3 (2 more independent jobs)

Follow-up to the full-finetune `llm-finetuned` result, which underperformed the frozen
`llm` strategy on rel-trial (0.6388 vs 0.6545) - likely overfitting, since the official
shared A3 sample is only 6,000 training seeds (~12 batches/epoch, never hitting the
500-step cap), so a 30-epoch budget means the 22.7M-parameter fully-unfrozen MiniLM saw
the same ~12 batches up to 30 times. Tests two changes at once: (1) freeze all but the
last 1 or 2 MiniLM transformer layers instead of the whole model (verified directly:
frozen params never get gradient, unfrozen ones always do, zero leakage - `k=1` leaves
1.77M trainable params, `k=2` leaves 3.55M, both far below the full fine-tune's 22.7M),
(2) train on a dedicated, separately-cached 30,000/10,000-seed sample instead of the
official 6,000/2,000 one (does not touch or replace the official A3 subsample - those
strategies need to stay on the identical fixed sample per the assignment's requirement).
See the `a3-partial-0` markdown cell in `final.ipynb` for the full reasoning.

Same split-by-dataset pattern as the other supplementary studies:

```bash
# on your account
sbatch run_partialfinetune_relstack.sh

# on your labmate's account
sbatch run_partialfinetune_reltrial.sh
```

Heavier than the full fine-tune: 5x more training data per epoch, so expect this to
take noticeably longer per run despite fewer trainable parameters.

Bring the results home the same way:

```bash
rsync -av <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_partial_finetune_results.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_partial_finetune_loss_curves.csv \
          ~/structed_ML/hw3/artifacts/
```

Output: `artifacts/aspect3_partial_finetune_results.csv` (one row per num_unfrozen per
seed) and `artifacts/aspect3_partial_finetune_loss_curves.csv` (per-epoch curves) -
kept separate from both `aspect3_results.csv` and `aspect3_finetune_results.csv`.
