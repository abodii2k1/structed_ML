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

## Supplementary: edge-type ablation (2 independent jobs, split across accounts)

Motivated by Aspect 2's rel-stack finding: trains hetero-SAGE (3 seeds) then masks
each relation's edges one at a time at evaluation time (no retraining) to measure its
actual contribution to validation AUROC. See the `edge-ablation-0` markdown cell in
`final.ipynb` for the full reasoning.

Split into two **standalone** jobs (no shared dependency, no `submit_all.sh`) so two
people on two different HPC accounts can each run one:

```bash
# on your account
sbatch run_ablation_relstack.sh

# on your labmate's account
sbatch run_ablation_reltrial.sh
```

Each script builds its own graph cache on first run if one isn't already present, so
either can run standalone on a fresh account after `setup_env.sh`. Resumable like
everything else - a rerun skips any (relation, seed) already in the output CSV.

Bring the results home the same way as step 4 above:

```bash
rsync -av <user>@<server>:~/structed_ML/hw3/artifacts/edge_ablation_*.csv \
          ~/structed_ML/hw3/artifacts/
```

Output: `artifacts/edge_ablation_{rel-stack,rel-trial}.csv` - one row per relation per
seed (AUROC with that relation masked, and its drop from the unmasked baseline), plus
a `__baseline__` row per seed.

## Supplementary: fine-tuned LLM encoding for Aspect 3 (2 more independent jobs)

Tests whether Aspect 3's `llm` strategy loses to `column` because serializing a row to
text loses structure, or because the frozen MiniLM embedding just can't adapt to the
task. Fine-tunes MiniLM (~22.7M params) end-to-end instead of using the official
strategy's frozen precomputed embedding. See the `a3-finetune-0` markdown cell in
`final.ipynb` for the full reasoning and the empirical checks behind the implementation
(PyG can't slice raw text lists, so text is pre-tokenized into fixed-length tensors
instead; discriminative LR so fine-tuning doesn't wreck the pretrained weights).

Same split-by-dataset pattern as the edge ablation above - two standalone jobs, no
shared dependency:

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
