"""Run the frozen `llm` baseline on the SAME large 30,000/10,000 sample and the same
train'/val'/test protocol as the partial fine-tune follow-up, by executing the ACTUAL
notebook cells (single source of truth - no code duplication with final.ipynb).

Together with `hpc_partial_finetune.py` (k=1, k=2), this gives exactly 3 fine-tuning-
family experiments per dataset - frozen, k=1, k=2 - all on the identical sample and
protocol, so only the freezing depth changes between them. See the
`a3-llm-frozen-large-0` markdown cell in final.ipynb for the full reasoning.

Usage:
    python hpc_llm_frozen_large.py rel-stack
    python hpc_llm_frozen_large.py rel-trial
    python hpc_llm_frozen_large.py rel-stack --dry   # just list which cells would run

Set STRUCTML_ARTIFACTS to redirect outputs (defaults to the repo's hw3/artifacts).
Resumable: results are saved after every seed; rerunning skips whatever already
finished. Cheap: frozen embedding lookup, no live MiniLM forward/backward - well under
a minute per seed once the large-sample subgraph cache is warm.
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
    f"usage: python hpc_llm_frozen_large.py <rel-stack|rel-trial>, got {dataset!r}"
os.environ["A3_LLMFROZEN_DATASET"] = dataset

# a3-partial-1 defines a3_build_or_load_large / ensure_llm_emb / A3PartialFinetuneModel
# etc. (only definitions, no training triggered by importing it); a3-partial-2 (which
# triggers the k=1/k=2 training loop) is deliberately excluded here.
CELLS = ["aspect1-1", "aspect1-3", "aspect3-1", "a3-partial-1",
         "a3-llm-frozen-large-1", "a3-llm-frozen-large-2"]

nb = json.load(open(NB))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

print(f"### hpc_llm_frozen_large: {dataset} | artifacts -> {os.environ['STRUCTML_ARTIFACTS']}", flush=True)

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

print(f"\n### DONE: frozen llm baseline (large sample) for {dataset}", flush=True)
