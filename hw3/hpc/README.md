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
