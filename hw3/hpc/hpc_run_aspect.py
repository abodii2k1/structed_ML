"""Run one aspect's official experiment on the HPC by executing the ACTUAL notebook
cells (single source of truth - no code duplication with final.ipynb).

Usage:
    python hpc_run_aspect.py prep   # download data, build graph caches + A3 subgraphs
    python hpc_run_aspect.py a1     # Aspect 1 (directionality)
    python hpc_run_aspect.py a2     # Aspect 2 (heterogeneity + param-matched control)
    python hpc_run_aspect.py a3     # Aspect 3 (node features)
    python hpc_run_aspect.py a4     # Aspect 4 (depth / oversmoothing)
    python hpc_run_aspect.py a1 --dry   # just list which cells would run

Set STRUCTML_ARTIFACTS to redirect outputs (defaults to the repo's hw3/artifacts).
All runs are resumable: results are saved after every run, loss curves after every
epoch; rerunning skips whatever already finished.
"""
import json, os, sys, gc

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "final.ipynb")
os.environ.setdefault("STRUCTML_ARTIFACTS", os.path.join(HERE, "..", "artifacts"))
os.makedirs(os.environ["STRUCTML_ARTIFACTS"], exist_ok=True)

nb = json.load(open(NB))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

CELLS = {
    "prep": ["aspect1-1", "aspect1-2", "aspect1-3", "aspect3-1"],
    "a1":   ["aspect1-1", "aspect1-2", "aspect1-3", "aspect1-4"],
    "a2":   ["aspect1-1", "aspect1-2", "aspect1-3", "aspect2-1", "aspect2-2", "aspect2-2b-control"],
    "a3":   ["aspect1-1", "aspect1-2", "aspect1-3", "aspect3-1", "aspect3-2"],
    "a4":   ["aspect1-1", "aspect1-2", "aspect1-3", "aspect4-1", "aspect4-2"],
}

aspect = sys.argv[1]
dry = "--dry" in sys.argv
assert aspect in CELLS, f"unknown aspect {aspect}; choose from {list(CELLS)}"
print(f"### hpc_run_aspect: {aspect} | artifacts -> {os.environ['STRUCTML_ARTIFACTS']}", flush=True)

if dry:
    for cid in CELLS[aspect]:
        assert cid in src, f"cell {cid} missing from notebook!"
        print("would exec:", cid)
    sys.exit(0)

g = globals()
for cid in CELLS[aspect]:
    print(f"\n===== exec cell {cid} =====", flush=True)
    exec(compile(src[cid], cid, "exec"), g)

if aspect == "prep":
    # build/download everything the aspect jobs will need, one dataset at a time
    for name, tname in g["TASKS"]:
        print(f"\n### prep: graph cache for {name}", flush=True)
        data, col_stats = g["build_or_load_graph"](name)
        g["get_task"](name, tname, download=True)
        del data, col_stats
        gc.collect()
    # The Aspect 3 subgraph is big and slow to build (MiniLM over every node) and is needed
    # ONLY by the a3 / partial-finetune jobs. Set STRUCTML_SKIP_A3_PREP=1 on an account that
    # runs only a1/a2/a4/basis (see submit_person_b.sh) to skip building it there.
    if os.environ.get("STRUCTML_SKIP_A3_PREP") == "1":
        print("\n### prep: skipping aspect-3 subgraph (STRUCTML_SKIP_A3_PREP=1)", flush=True)
    else:
        for name, tname in g["TASKS"]:
            print(f"\n### prep: aspect-3 subgraph for {name}", flush=True)
            blob = g["a3_build_or_load"](name, tname)
            del blob
            gc.collect()

print(f"\n### DONE: {aspect}", flush=True)
