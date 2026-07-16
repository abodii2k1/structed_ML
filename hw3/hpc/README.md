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

## Supplementary: LLM fine-tuning family for Aspect 3 (4 jobs, 2 each for 2 people)

Tests whether Aspect 3's `llm` strategy loses to `column` because serializing a row to
text loses structure, or because the frozen MiniLM embedding just can't adapt to the
task - and if adaptation helps, how much of MiniLM needs to be unfrozen. Exactly 3
experiments per dataset, all on the **identical** dedicated 30,000/10,000-seed sample
(`a3_build_or_load_large`, separate from the official 6,000/2,000 A3 subsample - does
not touch or replace it) and the **identical** protocol, so the only thing that changes
between them is how much of MiniLM can adapt:

- **frozen** - all of MiniLM stays frozen, only a linear projection trains (same
  architecture as the official `A3Model(..., strategy="llm", ...)`).
- **k=1** - last 1 MiniLM transformer layer unfrozen (1.77M trainable params).
- **k=2** - last 2 MiniLM transformer layers unfrozen (3.55M trainable params).

(Freezing verified directly: frozen params never receive a gradient, unfrozen ones
always do, zero leakage either direction.)

**Protocol note:** early stopping and the final reported metric do not share the same
validation labels. `a3_loaders_split()` (defined in `aspect3-1`) carves the official
train pool itself, label-stratified, into train'/val' - val' is used only to pick the
best checkpoint. The official `val` pool is left untouched during training and
evaluated exactly once, after the checkpoint is already fixed, to produce the reported
number. This is **not** applied to the official `id`/`column`/`llm` comparison in
`aspect3_results.csv`, or to Aspects 1/2/4 - only to these 3 experiments. See the
`a3-partial-0` / `a3-llm-frozen-large-0` markdown cells in `final.ipynb` for the full
reasoning.

Split into 4 scripts so two people can each run two (one dataset's frozen job + that
same dataset's k=1/k=2 job):

```bash
# person A, your account
sbatch run_llmfrozen_large_relstack.sh
sbatch run_partialfinetune_relstack.sh

# person B, labmate's account
sbatch run_llmfrozen_large_reltrial.sh
sbatch run_partialfinetune_reltrial.sh
```

The frozen job is cheap (embedding lookup only, well under a minute per seed once the
large-sample cache is warm); the k=1/k=2 job is heavier (MiniLM's transformer runs a
live forward+backward pass every mini-batch on 5x the official sample size), so expect
it to take noticeably longer.

Bring the results home:

```bash
rsync -av <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_llm_frozen_large_results.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_llm_frozen_large_loss_curves.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_partial_finetune_results.csv \
          <user>@<server>:~/structed_ML/hw3/artifacts/aspect3_partial_finetune_loss_curves.csv \
          ~/structed_ML/hw3/artifacts/
```

Output: `artifacts/aspect3_llm_frozen_large_results.csv` (one row per seed) and
`artifacts/aspect3_partial_finetune_results.csv` (one row per num_unfrozen per seed),
plus matching `..._loss_curves.csv` files for each - kept separate from
`aspect3_results.csv` since this is a supplementary follow-up, not an official strategy.
