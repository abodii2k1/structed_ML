"""Run the fine-tuned-LLM Aspect 3 follow-up on the HPC by executing the ACTUAL
notebook cells (single source of truth - no code duplication with final.ipynb).

Fine-tunes MiniLM (all-MiniLM-L6-v2, ~22.7M params) end-to-end as the Aspect 3 node
encoder, instead of the official `llm` strategy's frozen precomputed embedding.
Motivated by: does `llm` lose to `column` because serializing a row to text loses
structure no fine-tuning can fix, or because the frozen embedding just can't adapt?
See the `a3-finetune-0` markdown cell in final.ipynb for full reasoning.

Usage:
    python hpc_finetune_llm.py rel-stack
    python hpc_finetune_llm.py rel-trial
    python hpc_finetune_llm.py rel-stack --dry   # just list which cells would run

Set STRUCTML_ARTIFACTS to redirect outputs (defaults to the repo's hw3/artifacts).
Resumable: results are saved after every seed; rerunning skips whatever already
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
    f"usage: python hpc_finetune_llm.py <rel-stack|rel-trial>, got {dataset!r}"
os.environ["A3_FT_DATASET"] = dataset

CELLS = ["aspect1-1", "aspect1-3", "aspect3-1", "a3-finetune-1", "a3-finetune-2"]

nb = json.load(open(NB))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

print(f"### hpc_finetune_llm: {dataset} | artifacts -> {os.environ['STRUCTML_ARTIFACTS']}", flush=True)

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

print(f"\n### DONE: llm-finetuned for {dataset}", flush=True)
