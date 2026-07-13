"""Run the RGCN basis-decomposition sweep on the HPC by executing the ACTUAL notebook
cells (single source of truth - no code duplication with final.ipynb).

Motivated by the edge-count listing showing rel-stack's relations are severely
imbalanced (232x gap between largest and smallest) while rel-trial's are comparatively
balanced. Sweeps num_bases in {1, 2, 4, 8, num_relations} to test whether sharing
structure across relations (instead of each relation training an independent, possibly
data-starved weight matrix) helps - and whether it helps differently on the two
datasets given their very different edge-count distributions. See the `rgcn-basis-0`
markdown cell in final.ipynb for the full reasoning and the empirical checks behind it.

Usage:
    python hpc_rgcn_basis.py rel-stack
    python hpc_rgcn_basis.py rel-trial
    python hpc_rgcn_basis.py rel-stack --dry   # just list which cells would run

Set STRUCTML_ARTIFACTS to redirect outputs (defaults to the repo's hw3/artifacts).
Resumable: results are saved after every (num_bases, seed); rerunning skips whatever
already finished (matches every other HPC runner in this project).
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
    f"usage: python hpc_rgcn_basis.py <rel-stack|rel-trial>, got {dataset!r}"
os.environ["RGCN_BASIS_DATASET"] = dataset

CELLS = ["aspect1-1", "aspect1-3", "rgcn-basis-1", "rgcn-basis-2"]

nb = json.load(open(NB))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

print(f"### hpc_rgcn_basis: {dataset} | artifacts -> {os.environ['STRUCTML_ARTIFACTS']}", flush=True)

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

print(f"\n### DONE: rgcn-basis sweep for {dataset}", flush=True)
