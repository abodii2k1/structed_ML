"""Run the leave-one-relation-out edge ablation on the HPC by executing the ACTUAL
notebook cells (single source of truth - no code duplication with final.ipynb).

Trains hetero-SAGE (3 seeds) on the given dataset, then for each relation masks its
edges at evaluation time on the trained model and records the validation AUROC drop.
Motivated by Aspect 2's finding that homogeneous SAGE beats heterogeneous SAGE on
rel-stack - this measures whether some relations are essentially untrained noise that
hetero pays a parameter cost for while homo never has to.

Usage:
    python hpc_edge_ablation.py rel-stack
    python hpc_edge_ablation.py rel-trial
    python hpc_edge_ablation.py rel-stack --dry   # just list which cells would run

Set STRUCTML_ARTIFACTS to redirect outputs (defaults to the repo's hw3/artifacts).
Resumable: results are saved after every relation; rerunning skips whatever already
finished (matches every other HPC runner in this project).
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
    f"usage: python hpc_edge_ablation.py <rel-stack|rel-trial>, got {dataset!r}"
os.environ["EDGE_ABLATION_DATASET"] = dataset

CELLS = ["aspect1-1", "aspect1-3", "aspect2-1", "edge-ablation-1", "edge-ablation-2"]

nb = json.load(open(NB))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

print(f"### hpc_edge_ablation: {dataset} | artifacts -> {os.environ['STRUCTML_ARTIFACTS']}", flush=True)

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

print(f"\n### DONE: edge ablation for {dataset}", flush=True)
