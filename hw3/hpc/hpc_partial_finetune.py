"""Run the partial LLM fine-tune (k=1, k=2) Aspect 3 follow-up on the HPC by executing
the ACTUAL notebook cells (single source of truth - no code duplication with final.ipynb).

Two of five experiments in a 5-way Aspect 3 strategy comparison (the other three -
id/column/llm frozen - are `hpc_a3_large_sample.py`): freeze all but the last 1 or 2
MiniLM transformer layers, trained on a dedicated 30,000/10,000-seed sample (separate
from the official shared 6,000/2,000 A3 sample). Same protocol as every other result in
this project (early stopping and the final reported metric both from the official val
split) - not a genuine held-out test; see the report's Limitations section for why this
was disclosed rather than fixed, given time constraints. See the `a3-partial-0`
markdown cell in final.ipynb for the full reasoning and the empirical checks behind it
(verified the freezing mechanism directly: frozen params never get gradient, unfrozen
ones always do, zero leakage either direction).

Usage:
    python hpc_partial_finetune.py rel-stack           # both num_unfrozen configs (1 and 2)
    python hpc_partial_finetune.py rel-trial
    python hpc_partial_finetune.py rel-stack --k 1      # only num_unfrozen=1 (~21h alone,
    python hpc_partial_finetune.py rel-stack --k 2      # vs ~42h for both) - splits one
                                                         # dataset's work across two HPC
                                                         # accounts alongside the by-dataset
                                                         # split (run_partialfinetune_*.sh)
    python hpc_partial_finetune.py rel-stack --dry      # just list which cells would run

Set STRUCTML_ARTIFACTS to redirect outputs (defaults to the repo's hw3/artifacts).
Resumable: results are saved after every (num_unfrozen, seed); rerunning skips
whatever already finished (matches every other HPC runner in this project).
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "final.ipynb")
os.environ.setdefault("STRUCTML_ARTIFACTS", os.path.join(HERE, "..", "artifacts"))
os.makedirs(os.environ["STRUCTML_ARTIFACTS"], exist_ok=True)

dataset = sys.argv[1] if len(sys.argv) > 1 else None
dry = "--dry" in sys.argv
assert dataset in ("rel-stack", "rel-trial"), \
    f"usage: python hpc_partial_finetune.py <rel-stack|rel-trial>, got {dataset!r}"
os.environ["A3_PARTIAL_DATASET"] = dataset

if "--k" in sys.argv:
    k = sys.argv[sys.argv.index("--k") + 1]
    assert k in ("1", "2"), f"--k must be 1 or 2, got {k!r}"
    os.environ["A3_PARTIAL_UNFROZEN"] = k

CELLS = ["aspect1-1", "aspect1-3", "aspect3-1", "a3-partial-1", "a3-partial-2"]

nb = json.load(open(NB))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

print(f"### hpc_partial_finetune: {dataset} | artifacts -> {os.environ['STRUCTML_ARTIFACTS']}", flush=True)

if dry:
    for cid in CELLS:
        assert cid in src, f"cell {cid} missing from notebook!"
        print("would exec:", cid)
    sys.exit(0)

g = globals()
for cid in CELLS:
    assert cid in src, f"cell {cid} missing from notebook!"
    print(f"\n===== exec cell {cid} =====", flush=True)
    exec(compile(src[cid], cid, "exec"), g)

print(f"\n### DONE: partial finetune for {dataset}", flush=True)
