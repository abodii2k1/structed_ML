"""Regenerate the five aspect architecture figures for the StructML final project.

Paper-style figures (light fills, thin rules, numbered stages, the ablated block
highlighted): each figure shows the shared pipeline once and then isolates the one
component that the aspect ablates. Writes architecture_A1..A5 as PNG (300 dpi) and
PDF (vector) into OUT.

    python make_architecture_diagrams.py            # -> artifacts/
    python make_architecture_diagrams.py somedir    # -> somedir/
"""
import sys
import numpy as np

OUT = "artifacts"


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# ---------------------------------------------------------------- palette ---
INK    = "#20242b"      # primary text / arrows
MUTED  = "#71787f"      # secondary text
RULE   = "#b9c0c7"      # light borders

DATA_F, DATA_E = "#e4edf6", "#8fb2ce"   # data / graph objects  (blue)
ENC_F,  ENC_E  = "#e0efe1", "#8dbf94"   # encoders              (green)
MP_F,   MP_E   = "#fbe6cb", "#e0a75f"   # message passing       (amber)
HEAD_F, HEAD_E = "#f6dde6", "#d494ac"   # prediction head       (pink)
AUX_F,  AUX_E  = "#e8e4f2", "#a397c9"   # reshaping ops         (violet)
ABL_F,  ABL_E  = "#fffaf0", "#c99a3f"   # ablated block halo    (amber, dashed)
GREY_F, GREY_E = "#eceef0", "#a9b0b7"   # type-erased / neutral

BLUE   = "#2a6ebb"
RED    = "#c0392b"
GREEN  = "#3f8f52"
PURPLE = "#6c52a3"

FILLS = {"data": (DATA_F, DATA_E), "enc": (ENC_F, ENC_E), "mp": (MP_F, MP_E),
         "head": (HEAD_F, HEAD_E), "aux": (AUX_F, AUX_E), "grey": (GREY_F, GREY_E)}


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def title(ax, x, y, text, fs=15):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK)


def subtitle(ax, x, y, text, fs=9.5, color=MUTED, ha="center", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style)


def stage(ax, x, y, text, num=None, fs=9.5, ha="left"):
    """Bold numbered stage header sitting above a block (TabLLM-style)."""
    label = f"{num}. {text}" if num is not None else text
    ax.text(x, y, label, ha=ha, va="center", fontsize=fs,
            fontweight="bold", color=INK)


def box(ax, cx, cy, w, h, label, sub=None, kind="mp", fs=10, sfs=7.6,
        lw=1.1, dashed=False, mono=False):
    f, e = FILLS[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=f, edgecolor=e, lw=lw,
        linestyle="--" if dashed else "-", zorder=3))
    if sub:
        ax.text(cx, cy + h / 2 - 0.24, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=4,
                family="monospace" if mono else None)
        ax.text(cx, cy + h / 2 - 0.24 - 0.20 - 0.115 * (sub.count("\n") + 1),
                sub, ha="center", va="center", fontsize=sfs, color=MUTED, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=INK, zorder=4,
                family="monospace" if mono else None)
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def halo(ax, x0, y0, w, h, note=None, color=ABL_E, fill=ABL_F, nfs=8):
    """Dashed highlight around the block being ablated."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill, edgecolor=color, lw=1.3,
                                linestyle=(0, (5, 3)), zorder=1))
    if note:
        ax.text(x0 + w / 2, y0 + h + 0.14, note, ha="center", va="bottom",
                fontsize=nfs, color=color, fontweight="bold")


def arrow(ax, p0, p1, rad=0.0, color=INK, lw=1.2, ms=11, dashed=False, z=5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
        shrinkA=1, shrinkB=1, zorder=z,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def flow(ax, x0, x1, y, label=None, fs=7.4, color=INK, dy=0.13, lw=1.2):
    """Horizontal arrow between two blocks with an italic data label above it."""
    arrow(ax, (x0, y), (x1, y), color=color, lw=lw)
    if label:
        ax.text((x0 + x1) / 2, y + dy, label, ha="center", va="bottom",
                fontsize=fs, style="italic", color=BLUE, zorder=6)


def out_box(ax, cx, cy, w, h, text, fs=8.4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="white", edgecolor=BLUE, lw=1.1,
                                linestyle=(0, (4, 2.5)), zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            style="italic", color=BLUE, zorder=4)


def mini_table(ax, x, y_top, name, headers, rows, cw, rh=0.24, fs=6.4,
               accent="#41505e", key_cols=()):
    """Small database table. key_cols indexes get a tinted background."""
    w = cw * len(headers)
    ax.text(x + w / 2, y_top + 0.07, name, ha="center", va="bottom",
            fontsize=fs + 1.4, fontweight="bold", color=INK, family="monospace")
    for j, h in enumerate(headers):
        ax.add_patch(Rectangle((x + j * cw, y_top - rh), cw, rh,
                               facecolor=accent, edgecolor="white", lw=0.6, zorder=3))
        ax.text(x + j * cw + cw / 2, y_top - rh / 2, h, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4)
    for i, row in enumerate(rows):
        yy = y_top - (i + 2) * rh
        for j, v in enumerate(row):
            fc = "#f2f6fa" if j in key_cols else "white"
            ax.add_patch(Rectangle((x + j * cw, yy), cw, rh, facecolor=fc,
                                   edgecolor="#dcdfe3", lw=0.6, zorder=3))
            ax.text(x + j * cw + cw / 2, yy + rh / 2, str(v), ha="center",
                    va="center", fontsize=fs, color=INK, zorder=4)
    bottom = y_top - (1 + len(rows)) * rh
    return x + w, bottom, (y_top - rh + bottom) / 2


def node(ax, x, y, r=0.13, color=BLUE, fc=None, label=None, lfs=6.2, z=6, lw=1.1):
    ax.add_patch(Circle((x, y), r, facecolor=fc or color, edgecolor=color,
                        lw=lw, zorder=z))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=lfs,
                color="white", fontweight="bold", zorder=z + 1)


def link(ax, p0, p1, color=RULE, lw=1.0, rad=0.0, z=4, dashed=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-", color=color, lw=lw,
                                 zorder=z, shrinkA=0, shrinkB=0,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))


def legend(ax, x, y, items, fs=8.2, gap=2.05, bw=0.30, bh=0.19):
    """items: list of (kind|'dash'|'out', text)."""
    cx = x
    for kind, text in items:
        if kind == "out":
            ax.add_patch(FancyBboxPatch((cx, y - bh / 2), bw, bh,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor="white", edgecolor=BLUE, lw=1.0,
                                        linestyle=(0, (3, 2))))
        elif kind == "abl":
            ax.add_patch(FancyBboxPatch((cx, y - bh / 2), bw, bh,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=ABL_F, edgecolor=ABL_E, lw=1.0,
                                        linestyle=(0, (3, 2))))
        else:
            f, e = FILLS[kind]
            ax.add_patch(FancyBboxPatch((cx, y - bh / 2), bw, bh,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=f, edgecolor=e, lw=1.0))
        ax.text(cx + bw + 0.12, y, text, ha="left", va="center", fontsize=fs,
                color=MUTED)
        cx += gap


def save(fig, name, outdir=None):
    import os
    outdir = outdir or OUT
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight",
                facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)


# ==========================================================================
# Figure 1 - Aspect 1
# ==========================================================================
def rdb_tables(ax, x, cy):
    """Two example rel-stack tables with the FK -> PK link drawn."""
    r1, b1, m1 = mini_table(ax, x, cy + 1.02, "users", ["Id", "AboutMe"],
                            [["1", "ML fan..."], ["2", "SQL dev..."]],
                            cw=0.62, key_cols=(0,))
    r2, b2, m2 = mini_table(ax, x, cy - 0.30, "posts", ["Id", "OwnerId", "Title"],
                            [["7", "1", "How to..."], ["8", "2", "Why is..."]],
                            cw=0.62, key_cols=(0, 1))
    # FK -> PK curve: posts.OwnerId  ->  users.Id
    arrow(ax, (x + 0.93, b2 + 0.62), (x + 0.31, b1 + 0.02), rad=0.55,
          color=BLUE, lw=1.0, ms=8)
    ax.text(x - 0.16, cy + 0.12, "FK\u2192PK", fontsize=6.6, color=BLUE,
            rotation=90, ha="center", va="center", style="italic")
    ax.text(x + 0.95, b2 - 0.20, "+ comments, votes, badges \u2026",
            fontsize=6.6, color=MUTED, style="italic", ha="center")
    return r2


def graph_sketch(ax, cx, cy):
    """Typed PK-FK graph: one node per row, forward + reverse edge per link."""
    box(ax, cx, cy, 2.15, 2.05, "", kind="data", lw=1.1)
    ax.text(cx, cy + 0.80, "make_pkey_fkey_graph", ha="center", va="center",
            fontsize=7.0, family="monospace", color=INK, fontweight="bold")
    u, p, v = (cx + 0.62, cy + 0.28), (cx - 0.55, cy + 0.10), (cx - 0.05, cy - 0.60)
    link(ax, p, u, color="#93a3b4", lw=1.4)
    link(ax, v, p, color="#93a3b4", lw=1.4)
    node(ax, *u, r=0.20, color="#2a6ebb", fc="#2a6ebb", label="u")
    node(ax, *p, r=0.20, color="#3f8f52", fc="#3f8f52", label="p")
    node(ax, *v, r=0.20, color="#c0392b", fc="#c0392b", label="v")
    ax.text(cx + 0.10, cy + 0.42, "f2p", fontsize=6.2, color=MUTED, style="italic")
    ax.text(cx - 0.02, cy + 0.05, "rev_f2p", fontsize=6.2, color=MUTED,
            style="italic", ha="center", rotation=-9)
    ax.text(cx, cy - 1.24, "one node per row; each PK\u2013FK link \u2192\n"
                           "forward f2p + reverse rev_f2p edge",
            ha="center", va="center", fontsize=6.7, color=MUTED, style="italic")


def sample_sketch(ax, cx, cy):
    """Time-respecting 2-hop neighbour sample around a seed."""
    box(ax, cx, cy, 2.15, 2.05, "", kind="data", lw=1.1)
    ax.text(cx, cy + 0.80, "NeighborLoader", ha="center", va="center",
            fontsize=7.0, family="monospace", color=INK, fontweight="bold")
    ax.add_patch(Circle((cx, cy - 0.08), 0.72, facecolor="none",
                        edgecolor="#b8c6d4", lw=0.9, ls=(0, (3, 2)), zorder=4))
    hop1 = [(cx - 0.42, cy + 0.22), (cx + 0.42, cy + 0.22), (cx - 0.02, cy - 0.52)]
    hop2 = [(cx - 0.66, cy + 0.52), (cx + 0.68, cy + 0.50), (cx + 0.40, cy - 0.58)]
    for a, b in zip(hop1, hop2):
        link(ax, a, b, color="#c3ced8", lw=0.9)
        node(ax, *b, r=0.09, color="#9db4c8", fc="#dce6ee")
    for h in hop1:
        link(ax, (cx, cy - 0.08), h, color="#9db4c8", lw=1.0)
        node(ax, *h, r=0.11, color="#5c86ad", fc="#9db4c8")
    node(ax, cx, cy - 0.08, r=0.15, color="#c0392b", fc="#c0392b")
    ax.text(cx, cy - 1.24, "512 seed entities; \u2264128 neighbours/hop, 2 hops;\n"
                           "only the past ($t \\leq t_{seed}$)",
            ha="center", va="center", fontsize=6.7, color=MUTED, style="italic")


def wiring_panel(ax, x0, y0, w, h, name, tag, mode):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#fcfcfd", edgecolor=RULE, lw=1.0, zorder=1))
    cx = x0 + w / 2
    ax.text(cx, y0 + h - 0.33, name, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(cx, y0 + h - 0.68, tag, ha="center", va="center", fontsize=8.0,
            color=MUTED, style="italic")

    yc = y0 + 1.42
    lx, rx = cx - 1.10, cx + 1.10
    for x, lab, col in [(lx, "posts\nrow", "#3f8f52"), (rx, "users\nrow", "#2a6ebb")]:
        ax.add_patch(FancyBboxPatch((x - 0.50, yc - 0.30), 1.00, 0.60,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=col, lw=1.2, zorder=5))
        ax.text(x, yc, lab, ha="center", va="center", fontsize=7.2, color=INK, zorder=6)
    ax.text(lx, yc - 0.46, "child (FK)", ha="center", fontsize=6.4, color=MUTED)
    ax.text(rx, yc - 0.46, "parent (PK)", ha="center", fontsize=6.4, color=MUTED)

    if mode == "D":
        arrow(ax, (lx + 0.54, yc), (rx - 0.54, yc), color=BLUE, lw=1.5, ms=12)
        ax.text(cx, yc + 0.16, "$W$", ha="center", fontsize=9, color=BLUE)
        ax.text(cx, yc + 0.44, "f2p", ha="center", fontsize=6.8, color=BLUE,
                style="italic")
        ax.text(cx, y0 + 0.52, "reverse edges never wired \u2192\nparent \u2192 child flow is absent",
                ha="center", va="center", fontsize=7.4, color=RED, style="italic")
    elif mode == "U":
        arrow(ax, (lx + 0.54, yc + 0.10), (rx - 0.54, yc + 0.10), rad=-0.42,
              color=BLUE, lw=1.5, ms=12)
        arrow(ax, (rx - 0.54, yc - 0.10), (lx + 0.54, yc - 0.10), rad=-0.42,
              color=BLUE, lw=1.5, ms=12)
        ax.text(cx, yc + 0.62, "$W$", ha="center", fontsize=9, color=BLUE)
        ax.text(cx, yc - 0.72, "$W$  (same module object)", ha="center",
                fontsize=8, color=BLUE)
        ax.text(cx, y0 + 0.52, "one transform shared by both directions\n"
                               "\u2192 identical parameter count to MPNN-D",
                ha="center", va="center", fontsize=7.4, color=MUTED, style="italic")
    else:
        arrow(ax, (lx + 0.54, yc + 0.10), (rx - 0.54, yc + 0.10), rad=-0.42,
              color=BLUE, lw=1.5, ms=12)
        arrow(ax, (rx - 0.54, yc - 0.10), (lx + 0.54, yc - 0.10), rad=-0.42,
              color=RED, lw=1.5, ms=12)
        ax.text(cx, yc + 0.62, "$W_{fwd}$", ha="center", fontsize=9, color=BLUE)
        ax.text(cx, yc - 0.75, "$W_{rev}$  (independent)", ha="center",
                fontsize=8, color=RED)
        ax.text(cx, y0 + 0.52, "separate weights + separate aggregation\n"
                               "\u2192 2\u00d7 the message-passing parameters",
                ha="center", va="center", fontsize=7.4, color=MUTED, style="italic")


def make_a1():
    W, H = 14.8, 9.0
    fig, ax = canvas(W, H)
    title(ax, W / 2, H - 0.30,
          "Aspect 1 \u2014 Message Directionality: one shared pipeline, three edge-wiring modes")
    subtitle(ax, W / 2, H - 0.66,
             "Backbones: GraphSAGE (mean) and GAT (4 heads).  Datasets: rel-stack, rel-trial.  "
             "Everything except stage 5 is held fixed.")

    cy = 6.35
    hy = cy + 1.32          # stage-header row

    # 1 - raw DB
    stage(ax, 0.25, hy, "Relational database", 1)
    x_end = rdb_tables(ax, 0.35, cy)

    # 2 - graph
    stage(ax, 2.72, hy, "PK\u2013FK graph", 2)
    graph_sketch(ax, 3.80, cy)
    flow(ax, x_end + 0.10, 2.68, cy, "rows +\nPK\u2013FK links")

    # 3 - sampling
    stage(ax, 5.35, hy, "Temporal sampling", 3)
    sample_sketch(ax, 6.43, cy)
    flow(ax, 4.90, 5.30, cy, "HeteroData")

    # 4 - encoder
    stage(ax, 7.98, hy, "Node features", 4)
    box(ax, 8.75, cy, 2.05, 1.40, "HeteroEncoder",
        "typed per-column encoders\n(num / cat / GloVe text)\nfused \u2192 1 vector per row",
        kind="enc", fs=9, sfs=6.9)
    flow(ax, 7.55, 7.85, cy, "mini-batch\nsubgraph")

    # 5 - message passing (the ablated block)
    halo(ax, 10.35, cy - 0.78, 2.30, 1.78)
    stage(ax, 10.43, hy, "Message passing", 5, ha="left")
    box(ax, 11.50, cy + 0.11, 2.05, 1.28, "HeteroConv \u00d7 2",
        "one conv per wired edge type\n(SAGEConv | GATConv);\nsummed over edge types, ReLU",
        kind="mp", fs=9, sfs=6.7)
    ax.text(11.50, cy - 1.02, "\u2193  the ablated block", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=ABL_E)
    flow(ax, 9.75, 10.32, cy + 0.11, None)
    ax.text(10.03, cy + 0.42, "$x_v \\in \\mathbb{R}^{128}$", ha="center",
            fontsize=7.0, style="italic", color=BLUE)

    # 6 - head
    stage(ax, 13.00, hy, "Readout", 6)
    box(ax, 13.88, cy + 0.42, 1.55, 1.00, "MLP head",
        "Linear\u2192ReLU\u2192Linear", kind="head", fs=9, sfs=7.0)
    arrow(ax, (12.73, cy + 0.11), (13.10, cy + 0.42), lw=1.2)
    ax.text(12.94, cy + 0.74, "$h_{seed}$", ha="center", fontsize=7.0,
            style="italic", color=BLUE)
    arrow(ax, (13.88, cy - 0.10), (13.88, cy - 0.50), lw=1.1)
    out_box(ax, 13.88, cy - 0.92, 1.55, 0.78,
            "logit $\\hat{y}$\nper seed entity", fs=7.4)

    # ---- bottom band: the three modes -------------------------------------
    ax.plot([0.35, W - 0.35], [4.42, 4.42], color=RULE, lw=0.9)
    ax.text(W / 2, 4.06,
            "Stage 5 in detail \u2014 the only thing that changes: which edge types get a "
            "convolution, and whether the two directions share one weight matrix",
            ha="center", va="center", fontsize=10, fontweight="bold", color=INK)

    pw, gap = 4.55, 0.30
    x0 = (W - 3 * pw - 2 * gap) / 2
    for i, (nm, tg, md) in enumerate([
            ("MPNN-D", "forward edges only", "D"),
            ("MPNN-U", "both directions, shared weights", "U"),
            ("Dir-GNN", "both directions, separate weights", "G")]):
        wiring_panel(ax, x0 + i * (pw + gap), 0.86, pw, 2.95, nm, tg, md)

    legend(ax, x0, 0.40, [("data", "graph / data op"), ("enc", "encoder"),
                          ("mp", "message passing"), ("head", "prediction head"),
                          ("abl", "ablated block"), ("out", "model output")],
           gap=2.20)
    save(fig, "architecture_A1")


# ==========================================================================
# Figure 2 - Aspect 2
# ==========================================================================
TYPE_COL = {"u": "#2a6ebb", "p": "#3f8f52", "v": "#c0392b"}


def typed_sketch(ax, cx, cy, typed=True):
    """Small 3-type graph: one W per relation (typed) vs. one shared W (untyped)."""
    u, p, v = (cx + 0.60, cy + 0.30), (cx - 0.60, cy + 0.30), (cx, cy - 0.45)
    if typed:
        link(ax, p, u, color=BLUE, lw=1.6)
        link(ax, v, p, color=RED, lw=1.6)
        link(ax, v, u, color=GREEN, lw=1.6)
        ax.text(cx, cy + 0.46, "$W_{r_1}$", ha="center", fontsize=7.4, color=BLUE)
        ax.text(cx - 0.62, cy - 0.20, "$W_{r_2}$", ha="center", fontsize=7.4, color=RED)
        ax.text(cx + 0.62, cy - 0.20, "$W_{r_3}$", ha="center", fontsize=7.4, color=GREEN)
        for pt, k in [(u, "u"), (p, "p"), (v, "v")]:
            node(ax, *pt, r=0.155, color=TYPE_COL[k], fc=TYPE_COL[k], label=k, lfs=6.0)
    else:
        for a, b in [(p, u), (v, p), (v, u)]:
            link(ax, a, b, color="#8b949c", lw=1.6)
        ax.text(cx, cy + 0.46, "$W$", ha="center", fontsize=7.4, color=INK)
        ax.text(cx - 0.66, cy - 0.20, "$W$", ha="center", fontsize=7.4, color=INK)
        ax.text(cx + 0.66, cy - 0.20, "$W$", ha="center", fontsize=7.4, color=INK)
        for pt in (u, p, v):
            node(ax, *pt, r=0.155, color="#8b949c", fc="#c6ccd2")


def make_a2():
    W, H = 15.8, 10.6
    fig, ax = canvas(W, H)
    title(ax, W / 2, H - 0.32,
          "Aspect 2 \u2014 Heterogeneity: identical inputs and readout, typed vs. type-erased message passing")
    subtitle(ax, W / 2, H - 0.68,
             "Four variants: {GraphSAGE, HGT} \u00d7 {heterogeneous, homogeneous}.  "
             "Only stages 5\u20138 differ; stages 1\u20134 and 9\u201310 are shared by all four.")

    # ---------------- shared front-end (stages 1-4) -------------------------
    ty = 8.75
    hy = ty + 0.80
    stage(ax, 0.45, hy, "Raw RDB", 1)
    box(ax, 1.55, ty, 2.05, 1.20, "Tables", "one row \u2192 one node;\nPK\u2013FK links \u2192 edges",
        kind="data", fs=9.2, sfs=7.0)
    stage(ax, 3.10, hy, "Build graph", 2)
    box(ax, 4.30, ty, 2.25, 1.20, "make_pkey_fkey_graph",
        "forward f2p_* +\nreverse rev_f2p_* edges", kind="data", fs=8.2, sfs=7.0, mono=True)
    flow(ax, 2.62, 3.15, ty, None)
    stage(ax, 5.90, hy, "Sample", 3)
    box(ax, 7.25, ty, 2.55, 1.20, "NeighborLoader",
        "512 seeds, \u2264128/hop, 2 hops,\ntime-respecting", kind="data", fs=9.2, sfs=7.0)
    flow(ax, 5.45, 5.95, ty, None)
    stage(ax, 8.95, hy, "Node features", 4)
    box(ax, 10.35, ty, 2.55, 1.20, "HeteroEncoder",
        "typed per-column encoders;\neach table keeps its own", kind="enc", fs=9.2, sfs=7.0)
    flow(ax, 8.55, 9.05, ty, None)

    # routing line: X_type  ->  both branches
    fx = 12.10
    arrow(ax, (11.65, ty), (fx - 0.05, ty), lw=1.2)
    ax.plot([fx, fx], [ty, 7.92], color=INK, lw=1.2, zorder=2)
    ax.plot([0.30, fx], [7.92, 7.92], color=INK, lw=1.2, zorder=2)
    ax.plot([0.30, 0.30], [7.92, 3.05], color=INK, lw=1.2, zorder=2)
    arrow(ax, (0.30, 6.55), (0.78, 6.55), lw=1.2)
    arrow(ax, (0.30, 3.05), (0.78, 3.05), lw=1.2)
    ax.text(6.2, 7.76, "$X_{type}$ \u2014 one 128-d matrix per node type, still typed: "
                       "identical tensors enter both branches",
            ha="center", va="center", fontsize=7.6, style="italic", color=BLUE)

    # ---------------- branch A: heterogeneous -------------------------------
    ax.add_patch(FancyBboxPatch((0.80, 5.30), 11.95, 2.30,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#f7fafd", edgecolor="#9dbbd6", lw=1.1, zorder=1))
    ax.text(1.05, 7.38, "Heterogeneous \u2014 types kept through message passing",
            ha="left", va="center", fontsize=10, fontweight="bold", color=BLUE)
    ay = 6.45
    box(ax, 3.10, ay, 3.35, 1.30, "HeteroConv \u00d7 2   (5a, 6a)",
        "SAGEConv per edge type | native HGTConv (4 heads):\n"
        "own weights per relation; summed per node; ReLU",
        kind="mp", fs=9.2, sfs=6.7)
    typed_sketch(ax, 6.35, ay, typed=True)
    ax.text(6.35, ay - 0.92, "one weight matrix per relation",
            ha="center", fontsize=7.2, color=MUTED, style="italic")
    flow(ax, 4.82, 5.35, ay, None)
    box(ax, 9.55, ay, 2.55, 1.30, "Take entity rows   (7a)",
        "keep the entity type's matrix;\nfirst 512 rows = this batch's seeds",
        kind="aux", fs=9.2, sfs=6.9)
    flow(ax, 7.35, 8.25, ay, "$h_v^{(2)}$ per\nnode type", fs=6.8)
    arrow(ax, (10.85, ay), (11.80, ay), lw=1.2)

    # ---------------- branch B: homogeneous ---------------------------------
    ax.add_patch(FancyBboxPatch((0.80, 1.95), 11.95, 2.35,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#faf9fd", edgecolor="#b3a8d3", lw=1.1, zorder=1))
    ax.text(1.05, 4.08, "Homogeneous \u2014 types erased before message passing",
            ha="left", va="center", fontsize=10, fontweight="bold", color=PURPLE)
    by = 2.90
    box(ax, 2.90, by, 2.80, 1.55, "collapse()   (5b)",
        "concat every $X_{type}$ into one $x_{all}$\n(store each type's row offset);\n"
        "shift + merge all edge indices\n\u2192 one untyped node & edge set",
        kind="aux", fs=9.2, sfs=6.9, mono=False)
    typed_sketch(ax, 5.85, by + 0.10, typed=False)
    ax.text(5.85, by - 0.80, "one shared weight matrix\nfor every relation",
            ha="center", fontsize=7.2, color=MUTED, style="italic")
    flow(ax, 4.35, 4.90, by, None)
    box(ax, 8.55, by, 2.75, 1.55, "Single Conv \u00d7 2   (6b, 7b)",
        "ONE type-agnostic convolution\nfor all nodes: plain SAGEConv, or\n"
        "HGTConv on an explicit single-type\nHeteroData wrapper; ReLU",
        kind="mp", fs=9.2, sfs=6.9)
    flow(ax, 6.85, 7.15, by, "($x_{all}$, $ei_{all}$)", fs=6.8)
    box(ax, 11.40, by, 2.45, 1.55, "Slice by offset   (8b)",
        "cut the entity type's rows\nback out of $h_{all}$ using\nits stored offset",
        kind="aux", fs=9.2, sfs=6.9)
    flow(ax, 9.95, 10.15, by, "$h_{all}^{(2)}$", fs=6.8)

    # ---------------- shared readout ---------------------------------------
    box(ax, 14.35, 4.72, 1.90, 1.05, "MLP head   (9)",
        "Linear\u2192ReLU\u2192Linear", kind="head", fs=9.2, sfs=7.0)
    arrow(ax, (11.85, ay), (13.42, 5.05), rad=-0.10, lw=1.2)
    arrow(ax, (12.65, by), (13.42, 4.42), rad=0.10, lw=1.2)
    ax.text(13.05, 5.62, "$h_{seed}$", fontsize=7.2, style="italic", color=BLUE)
    ax.text(13.05, 3.72, "$h_{seed}$", fontsize=7.2, style="italic", color=BLUE)
    arrow(ax, (14.35, 4.18), (14.35, 3.70), lw=1.1)
    out_box(ax, 14.35, 3.24, 1.90, 0.85,
            "logit $\\hat{y}$ (10)\nper seed entity", fs=7.6)

    # ---------------- note + legend ----------------------------------------
    ax.add_patch(FancyBboxPatch((0.80, 0.42), 11.95, 0.82,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="#fbfbfc", edgecolor=RULE, lw=0.9))
    ax.text(6.78, 0.83,
            "Isolation: identical $X_{type}$ (stage 4) and identical head (stage 9), so any gap is due to message passing alone. "
            "Heterogeneous models own more weights\n(one set per relation), so parameter counts are reported and a "
            "parameter-matched control (widened homogeneous SAGE, \u201chomo-wide\u201d) is run on rel-trial.",
            ha="center", va="center", fontsize=7.8, color=MUTED)

    for i, (k, t) in enumerate([("data", "data op"), ("enc", "encoder"),
                                ("mp", "message passing"), ("aux", "reshaping op"),
                                ("head", "head"), ("out", "model output")]):
        legend(ax, 13.45, 1.62 - 0.26 * i, [(k, t)], fs=7.6)
    save(fig, "architecture_A2")


# ==========================================================================
# Figure 3 - Aspect 3
# ==========================================================================
def panel(ax, x0, y0, w, h, tag, name, color):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#fcfdfe", edgecolor=color, lw=1.2, zorder=1))
    cx = x0 + w / 2
    ax.text(x0 + 0.22, y0 + h - 0.30, tag, ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=color)
    ax.text(x0 + w - 0.22, y0 + h - 0.30, name, ha="right", va="center",
            fontsize=8.0, color=MUTED, style="italic")
    return cx


def foot(ax, cx, y0, sub, params):
    ax.text(cx, y0 + 0.60, sub, ha="center", va="center", fontsize=7.2, color=MUTED)
    ax.text(cx, y0 + 0.24, params, ha="center", va="center", fontsize=7.6,
            color=INK, fontweight="bold")


def chip(ax, cx, cy):
    ax.add_patch(FancyBboxPatch((cx - 0.78, cy - 0.20), 1.56, 0.40,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="#eef4fa", edgecolor=BLUE, lw=1.1, zorder=5))
    ax.text(cx, cy, "$x_v \\in \\mathbb{R}^{128}$", ha="center", va="center",
            fontsize=8.4, color=BLUE, zorder=6)


def make_a3():
    W, H = 15.6, 10.6
    fig, ax = canvas(W, H)
    title(ax, W / 2, H - 0.30,
          "Aspect 3 \u2014 Node Features: three swappable input encoders, one identical HGT downstream")
    subtitle(ax, W / 2, H - 0.66,
             "Exactly one of 4a / 4b / 4c is active per run. Stages 5\u20138 are bit-for-bit identical, "
             "so every difference is attributable to the initial node representation.")

    ty, hy = 8.85, 9.62
    stage(ax, 0.45, hy, "Raw RDB", 1)
    r, b, m = mini_table(ax, 1.00, ty + 0.52, "studies", ["NctId", "Phase", "Summary"],
                         [["NCT01", "2", "A study of\u2026"], ["NCT02", "3", "Trial for\u2026"]],
                         cw=0.72, key_cols=(0,))
    ax.text(2.08, b - 0.20, "+ conditions, sponsors, \u2026", ha="center",
            fontsize=6.6, color=MUTED, style="italic")

    stage(ax, 3.60, hy, "Build graph", 2)
    box(ax, 4.80, ty, 2.30, 1.15, "make_pkey_fkey_graph",
        "one node per row;\nf2p_* + rev_f2p_* edges", kind="data", fs=8.2, sfs=7.0, mono=True)
    flow(ax, r + 0.15, 3.63, ty, None)

    stage(ax, 6.65, hy, "Fixed cached subsample", 3)
    box(ax, 8.62, ty, 3.55, 1.15, "Stratified seeds + 2-hop neighbourhoods",
        "6 000 train / 2 000 val seeds (positive rate preserved);\n"
        "time-respecting fan-out [6, 6]; built once, cached to disk",
        kind="data", fs=8.6, sfs=6.8)
    flow(ax, 5.97, 6.83, ty, None)
    arrow(ax, (10.42, ty), (10.85, ty), lw=1.2)
    ax.text(10.95, ty, "the SAME subgraph \u2014 same nodes,\nedges, seeds and labels \u2014 is used\n"
                       "by all three strategies and all seeds",
            ha="left", va="center", fontsize=7.8, style="italic", color=BLUE)

    ax.text(0.45, 7.92, "Stage 4 \u2014 the ablated component: how each row becomes its "
                        "initial 128-d vector",
            ha="left", va="center", fontsize=10, fontweight="bold", color=INK)

    fy = 7.58
    py0, ph = 3.72, 3.55
    pw, gap = 4.85, 0.32
    x0s = [0.42, 0.42 + pw + gap, 0.42 + 2 * (pw + gap)]
    centers = [x + pw / 2 for x in x0s]
    ax.plot([8.62, 8.62], [ty - 0.58, fy], color=INK, lw=1.2, zorder=2)
    ax.plot([centers[0], centers[2]], [fy, fy], color=INK, lw=1.2, zorder=2)
    for c in centers:
        arrow(ax, (c, fy), (c, py0 + ph - 0.02), lw=1.2)

    # --- 4a  id ------------------------------------------------------------
    cx = panel(ax, x0s[0], py0, pw, ph, "4a  id encoding", "no-features baseline", PURPLE)
    foot(ax, cx, py0,
         "a pure lookup table: cell values are never read; transductive \u2014\n"
         "a slot is only learned if that node is seen during training",
         "learned parameters (rel-stack): 25.6 M")
    ax.add_patch(FancyBboxPatch((cx - 2.22, py0 + 2.02), 0.95, 0.60,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="white", edgecolor=PURPLE, lw=1.1, zorder=4))
    ax.text(cx - 1.74, py0 + 2.32, "$i = 37$", ha="center", va="center",
            fontsize=9, color=INK, zorder=5)
    ax.text(cx - 1.74, py0 + 2.80, "node index only", ha="center", fontsize=7.4, color=MUTED)
    tx, tb = cx - 0.30, py0 + 1.78
    for k in range(4):
        hit = (k == 2)
        ax.add_patch(Rectangle((tx, tb + k * 0.26), 1.50, 0.26,
                               facecolor="#efeaf7" if hit else "white",
                               edgecolor=PURPLE if hit else "#d3cde3",
                               lw=1.4 if hit else 0.7, zorder=4))
    ax.text(tx + 0.75, tb + 1.16, "nn.Embedding($N_{type}$, 128)", ha="center",
            fontsize=7.2, family="monospace", color=INK)
    arrow(ax, (cx - 1.24, py0 + 2.32), (tx - 0.04, py0 + 2.43), color=PURPLE, lw=1.2, ms=10)
    ax.plot([tx + 1.50, tx + 1.88], [tb + 0.65, tb + 0.65], color=PURPLE, lw=1.2, zorder=5)
    ax.plot([tx + 1.88, tx + 1.88], [tb + 0.65, py0 + 1.22], color=PURPLE, lw=1.2, zorder=5)
    arrow(ax, (tx + 1.88, py0 + 1.22), (cx + 0.82, py0 + 1.22), color=PURPLE, lw=1.2, ms=10)
    chip(ax, cx, py0 + 1.22)

    # --- 4b  column-wise ---------------------------------------------------
    cx = panel(ax, x0s[1], py0, pw, ph, "4b  column-wise encoding",
               "torch_frame", GREEN)
    foot(ax, cx, py0,
         "one encoder per column, chosen by the column's semantic type;\n"
         "the per-column embeddings are fused into a single row vector",
         "learned parameters (rel-stack): 3.6 M")
    for j, ((cell, st), col) in enumerate(zip(
            [("NctId=NCT01", "cat"), ("Phase=2", "num"), ("Summary=\u2026", "text")],
            [GREEN, PURPLE, BLUE])):
        xx = cx - 1.45 + j * 1.45
        ax.add_patch(FancyBboxPatch((xx - 0.62, py0 + 2.74), 1.24, 0.36,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="white", edgecolor="#c6ccd2", lw=0.9, zorder=4))
        ax.text(xx, py0 + 2.92, cell, ha="center", va="center", fontsize=6.6,
                color=INK, zorder=5)
        arrow(ax, (xx, py0 + 2.72), (xx, py0 + 2.52), color="#9aa2aa", lw=1.0, ms=8)
        ax.add_patch(FancyBboxPatch((xx - 0.48, py0 + 2.06), 0.96, 0.44,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=ENC_F, edgecolor=col, lw=1.2, zorder=4))
        ax.text(xx, py0 + 2.28, f"Enc$_{{{st}}}$", ha="center", va="center",
                fontsize=7.8, color=col, zorder=5, fontweight="bold")
        arrow(ax, (xx, py0 + 2.04), (xx, py0 + 1.88), color="#9aa2aa", lw=1.0, ms=8)
        for i in range(5):
            ax.add_patch(Rectangle((xx - 0.45 + i * 0.18, py0 + 1.60), 0.18, 0.26,
                                   facecolor=col + "26", edgecolor=col, lw=0.7, zorder=5))
    ax.plot([cx - 1.45, cx + 1.45], [py0 + 1.50, py0 + 1.50], color="#9aa2aa", lw=1.0, zorder=4)
    ax.text(cx + 1.68, py0 + 1.50, "fuse", ha="left", va="center", fontsize=7, color=MUTED)
    arrow(ax, (cx, py0 + 1.50), (cx, py0 + 1.44), color="#9aa2aa", lw=1.0, ms=9)
    chip(ax, cx, py0 + 1.22)

    # --- 4c  LLM -----------------------------------------------------------
    cx = panel(ax, x0s[2], py0, pw, ph, "4c  LLM encoding",
               "frozen sentence-transformer", BLUE)
    foot(ax, cx, py0,
         "the row is serialized to text and embedded once, offline;\n"
         "MiniLM is never fine-tuned \u2014 only the projection is learned",
         "learned parameters (rel-stack): 1.6 M")
    ax.add_patch(FancyBboxPatch((cx - 2.15, py0 + 2.72), 4.30, 0.44,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="white", edgecolor="#c6ccd2", lw=0.9, zorder=4))
    ax.text(cx, py0 + 2.94, "\u201cNctId=NCT01, Phase=2, Summary=A study of\u2026\u201d",
            ha="center", va="center", fontsize=7.2, color=INK, style="italic", zorder=5)
    ax.text(cx + 2.10, py0 + 2.46, "serialize the row", ha="right", va="center",
            fontsize=6.8, color=MUTED, style="italic")
    arrow(ax, (cx - 1.15, py0 + 2.70), (cx - 1.15, py0 + 2.42), color="#9aa2aa", lw=1.0, ms=8)
    box(ax, cx - 1.15, py0 + 2.06, 1.90, 0.62, "\u2744  MiniLM (frozen)", kind="data", fs=8.4)
    ax.text(cx - 1.15, py0 + 1.60, "all-MiniLM-L6-v2", ha="center", fontsize=6.8,
            color=MUTED, family="monospace")
    arrow(ax, (cx - 0.18, py0 + 2.06), (cx + 0.38, py0 + 2.06), color=BLUE, lw=1.2, ms=10)
    ax.text(cx + 0.10, py0 + 2.22, "384-d", ha="center", fontsize=6.8, color=BLUE,
            style="italic")
    box(ax, cx + 1.25, py0 + 2.06, 1.70, 0.62, "Linear(384,128)", kind="enc",
        fs=8.4, mono=True)
    ax.text(cx + 1.25, py0 + 1.60, "the only trained part", ha="center", fontsize=6.8,
            color=GREEN, style="italic")
    ax.plot([cx + 1.25, cx + 1.25], [py0 + 1.75, py0 + 1.40], color=BLUE, lw=1.2, zorder=4)
    arrow(ax, (cx + 1.25, py0 + 1.40), (cx + 0.82, py0 + 1.24), color=BLUE, lw=1.2, ms=10)
    chip(ax, cx, py0 + 1.22)

    # ---------------- stages 5-8 (shared) -----------------------------------
    dy = 2.30
    for c in centers:
        arrow(ax, (c, py0), (c, py0 - 0.30), lw=1.2)
    ax.plot([centers[0], centers[2]], [py0 - 0.30, py0 - 0.30], color=INK, lw=1.2, zorder=2)
    ax.plot([W / 2, W / 2], [py0 - 0.30, dy + 0.82], color=INK, lw=1.2, zorder=2)
    arrow(ax, (W / 2, dy + 0.90), (W / 2, dy + 0.66), lw=1.2)
    ax.text(W / 2 + 0.18, py0 - 0.58, "all three emit exactly the same shape: "
                                      "$x_v \\in \\mathbb{R}^{128}$ per node",
            ha="left", va="center", fontsize=7.8, style="italic", color=BLUE)

    ax.add_patch(FancyBboxPatch((3.05, dy - 0.92), 9.55, 1.55,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#fbfbfc", edgecolor=RULE, lw=1.0, zorder=1))
    ax.text(3.28, dy + 0.44, "Stages 5\u20138 \u2014 identical for all three strategies",
            ha="left", va="center", fontsize=8.6, fontweight="bold", color=MUTED)
    box(ax, 5.10, dy - 0.30, 2.65, 0.92, "HGTConv \u00d7 2   (5, 6)",
        "4-head typed attention", kind="mp", fs=9, sfs=7.0)
    box(ax, 8.40, dy - 0.30, 2.10, 0.92, "Seed rows   (7)",
        "the 6 000 / 2 000 seeds", kind="aux", fs=9, sfs=7.0)
    flow(ax, 6.45, 7.33, dy - 0.30, "$h_v^{(2)}$", fs=7.0)
    box(ax, 11.30, dy - 0.30, 1.95, 0.92, "MLP head   (8)",
        "Linear\u2192ReLU\u2192Linear", kind="head", fs=9, sfs=7.0)
    flow(ax, 9.47, 10.30, dy - 0.30, "$h_{seed}$", fs=7.0)
    arrow(ax, (12.30, dy - 0.30), (12.95, dy - 0.30), lw=1.2)
    out_box(ax, 13.90, dy - 0.30, 1.75, 0.82, "logit $\\hat{y}$\nper seed entity", fs=7.6)

    ax.text(W / 2, 0.68,
            "Complexity is part of the comparison: the id table stores one learned 128-d slot per node "
            "(25.6 M parameters on rel-stack); the column encoders are moderate (3.6 M);\n"
            "the LLM route learns the least of all (1.6 M) but pays a heavy one-off "
            "embedding and storage cost before training even starts.",
            ha="center", va="center", fontsize=8, color=MUTED)
    save(fig, "architecture_A3")


# ==========================================================================
# Figure 4 - Aspect 4
# ==========================================================================
def hchip(ax, cx, cy, text, w=0.95, color=BLUE):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 0.20), w, 0.40,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor="#eef4fa", edgecolor=color, lw=1.0, zorder=5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.2, color=color, zorder=6)


def scatter_blob(ax, cx, cy, spread, seed, n=14, color=BLUE, rmax=0.46):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.normal(0, spread, size=2)
        if p[0] ** 2 + p[1] ** 2 <= rmax ** 2:
            pts.append(p)
    for px, py in pts:
        ax.add_patch(Circle((cx + px, cy + py * 0.75), 0.055, facecolor=color,
                            edgecolor="none", alpha=0.75, zorder=5))


def make_a4():
    W, H = 15.4, 9.9
    fig, ax = canvas(W, H)
    title(ax, W / 2, H - 0.30,
          "Aspect 4 \u2014 Depth and Oversmoothing: a GCN stack of depth L, with an optional residual skip")
    subtitle(ax, W / 2, H - 0.66,
             "12 configurations: L \u2208 {1, 2, 3, 4, 6, 8} \u00d7 {no skip, skip}. "
             "Only stage 6 changes \u2014 encoder, features, sampled subgraph and training budget are fixed.")

    # ---------------- trunk (stages 1-5) ------------------------------------
    ty, hy = 8.35, 9.05
    stage(ax, 0.42, hy, "Raw RDB", 1)
    box(ax, 1.32, ty, 1.75, 1.05, "Tables", "one row \u2192 one node", kind="data",
        fs=9.2, sfs=7.0)
    stage(ax, 2.60, hy, "Build graph", 2)
    box(ax, 3.85, ty, 2.30, 1.05, "make_pkey_fkey_graph",
        "f2p_* + rev_f2p_* edges", kind="data", fs=8.2, sfs=7.0, mono=True)
    flow(ax, 2.22, 2.68, ty, None)
    stage(ax, 5.30, hy, "Sample \u2014 FIXED", 3)
    box(ax, 7.00, ty, 3.05, 1.05, "NeighborLoader  [10, 10]",
        "256 seeds; the SAME 2-hop subgraph is\nhanded to every depth setting",
        kind="data", fs=9.0, sfs=6.9)
    flow(ax, 5.03, 5.45, ty, None)
    stage(ax, 8.80, hy, "Node features", 4)
    box(ax, 10.00, ty, 2.05, 1.05, "HeteroEncoder", "typed per-column encoders",
        kind="enc", fs=9.2, sfs=6.9)
    flow(ax, 8.55, 8.95, ty, None)
    stage(ax, 11.85, hy, "Erase types", 5)
    box(ax, 13.25, ty, 2.35, 1.05, "collapse()",
        "concat per-type $X$ into $x_{all}$;\nmerge every edge set (as in Aspect 2)",
        kind="aux", fs=9.2, sfs=6.6)
    flow(ax, 11.05, 12.06, ty, None)

    # route down into the GCN panel
    ax.plot([13.25, 13.25], [ty - 0.53, 7.30], color=INK, lw=1.2, zorder=2)
    ax.plot([1.05, 13.25], [7.30, 7.30], color=INK, lw=1.2, zorder=2)
    arrow(ax, (1.05, 7.30), (1.05, 6.62), lw=1.2)
    ax.text(7.15, 7.44, "$h^{(0)} = x_{all}$  +  one merged edge set", ha="center",
            va="bottom", fontsize=7.8, style="italic", color=BLUE)

    # ---------------- stage 6: the GCN stack --------------------------------
    px0, py0, pw, ph = 0.45, 3.55, 9.35, 3.00
    halo(ax, px0, py0, pw, ph)
    ax.text(px0 + 0.25, py0 + ph - 0.30,
            "6.  GCN stack \u2014 the ablated component: the same block repeated $L$ times",
            ha="left", va="center", fontsize=10, fontweight="bold", color="#a8791f")

    gy = py0 + 1.55
    hchip(ax, 1.05, gy, "$h^{(0)}$")
    xs = [2.55, 4.65, 7.55]
    for i, gx in enumerate(xs):
        if i == 2:
            ax.text(6.10, gy, "\u2026", ha="center", va="center", fontsize=14, color=MUTED)
        box(ax, gx, gy, 1.55, 0.72, "GCNConv + ReLU", kind="mp", fs=8.2)
        ax.text(gx, gy - 0.52, f"layer {'1' if i == 0 else ('2' if i == 1 else 'L')}",
                ha="center", fontsize=7.0, color=MUTED)
        # optional residual around this layer
        arrow(ax, (gx - 0.80, gy + 0.30), (gx + 0.80, gy + 0.30), rad=-0.55,
              color=RED, lw=1.1, ms=9)
    arrow(ax, (1.55, gy), (1.75, gy), lw=1.1)
    arrow(ax, (3.35, gy), (3.85, gy), lw=1.1)
    ax.text(3.60, gy + 0.14, "$h^{(1)}$", ha="center", fontsize=7.0, style="italic",
            color=BLUE)
    arrow(ax, (5.45, gy), (5.75, gy), lw=1.1)
    arrow(ax, (6.45, gy), (6.75, gy), lw=1.1)
    arrow(ax, (8.35, gy), (8.75, gy), lw=1.1)
    hchip(ax, 9.20, gy, "$h^{(L)}$")

    ax.text(4.85, py0 + 2.46, "$\\times\\, L$,   $L \\in \\{1, 2, 3, 4, 6, 8\\}$",
            ha="center", va="center", fontsize=9, color="#a8791f", fontweight="bold")
    ax.text(px0 + 0.30, py0 + 0.62,
            "baseline (no skip):   $h^{(l)} = \\mathrm{ReLU}(\\mathrm{conv}(h^{(l-1)}))$",
            ha="left", va="center", fontsize=8.6, color=INK)
    ax.text(px0 + 0.30, py0 + 0.28,
            "mitigation (skip):   $h^{(l)} = \\mathrm{ReLU}(h^{(l-1)} + "
            "\\mathrm{conv}(h^{(l-1)}))$",
            ha="left", va="center", fontsize=8.6, color=RED)
    ax.text(px0 + 5.45, py0 + 0.45, "the red loop = the residual carrying each layer's\n"
                                    "input around its convolution (skip variant only)",
            ha="left", va="center", fontsize=7.6, color=RED, style="italic")

    # ---------------- stages 7-8 -------------------------------------------
    box(ax, 11.20, 5.30, 2.30, 1.25, "Slice by offset   (7)",
        "cut the entity type's rows\nback out of $h^{(L)}$;\nfirst 256 = this batch's seeds",
        kind="aux", fs=9.2, sfs=6.9)
    arrow(ax, (9.72, 5.10), (10.05, 5.30), lw=1.2)
    box(ax, 14.10, 5.30, 1.90, 1.25, "MLP head   (8)",
        "Linear\u2192ReLU\n\u2192Linear", kind="head", fs=9.2, sfs=6.9)
    flow(ax, 12.38, 13.13, 5.30, "$h_{seed}$", fs=7.0)
    arrow(ax, (14.10, 4.65), (14.10, 4.25), lw=1.1)
    out_box(ax, 14.10, 3.85, 1.90, 0.72, "logit $\\hat{y}$ per seed", fs=7.4)

    # ---------------- what we measure ---------------------------------------
    ax.plot([0.45, W - 0.40], [3.15, 3.15], color=RULE, lw=0.9)
    ax.text(0.45, 2.86, "What is measured on $h^{(L)}$ \u2014 the same tensor that feeds the head",
            ha="left", va="center", fontsize=10, fontweight="bold", color=INK)

    ax.add_patch(FancyBboxPatch((0.45, 0.42), 8.20, 2.15,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#fbfcfd", edgecolor=RULE, lw=1.0, zorder=1))
    for cx, spread, lab, sd in [(2.05, 0.30, "shallow (L = 2)", 1),
                                (6.05, 0.07, "deep (L = 8), no skip", 5)]:
        ax.add_patch(Circle((cx, 1.75), 0.58, facecolor="white", edgecolor="#dfe4e9",
                            lw=0.9, zorder=3))
        scatter_blob(ax, cx, 1.75, spread, sd)
        ax.text(cx, 1.00, lab, ha="center", fontsize=7.8, color=MUTED)
    arrow(ax, (2.80, 1.75), (5.30, 1.75), lw=1.3, color="#a8791f")
    ax.text(4.05, 2.28, "node embeddings collapse toward each other",
            ha="center", fontsize=7.8, color="#a8791f", style="italic")
    ax.text(4.05, 1.98, "increasing depth", ha="center", fontsize=7.2,
            color="#a8791f", style="italic")
    ax.text(4.05, 0.66, "cos_sim \u2192 1        dir_energy \u2192 0",
            ha="center", fontsize=8.4, color=INK, family="monospace")
    ax.text(7.45, 1.75, "+ the four\ndownstream\nmetrics vs. $L$", ha="center",
            va="center", fontsize=7.8, color=MUTED)

    ax.add_patch(FancyBboxPatch((8.95, 0.42), 6.05, 2.15,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#fbfcfd", edgecolor=RULE, lw=1.0, zorder=1))
    ax.text(9.20, 2.28, "Isolation \u2014 what \u201cdepth\u201d means here",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=MUTED)
    ax.text(9.20, 1.35,
            "Because the sampled 2-hop subgraph (stage 3) is identical at every\n"
            "depth, adding layers adds propagation rounds over the same\n"
            "neighbourhood \u2014 it does not enlarge the receptive field. A depth-\n"
            "dependent sampler would instead grow the subgraph exponentially\n"
            "and blow the 8 GB budget, confounding depth with sample size.",
            ha="left", va="center", fontsize=7.7, color=MUTED)

    save(fig, "architecture_A4")


# ==========================================================================
# Figure 5 - Aspect 5
# ==========================================================================
def schema_chip(ax, cx, cy, name, color, w=1.30, h=0.34):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="white", edgecolor=color, lw=1.1, zorder=4))
    ax.text(cx, cy, name, ha="center", va="center", fontsize=7.4, color=color,
            family="monospace", zorder=5)


def card(ax, x0, y0, w, h, num, head, body, color):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#fcfdfe", edgecolor=color, lw=1.2, zorder=1))
    ax.add_patch(Circle((x0 + 0.36, y0 + h - 0.34), 0.17, facecolor=color,
                        edgecolor=color, zorder=5))
    ax.text(x0 + 0.36, y0 + h - 0.34, str(num), ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=6)
    ax.text(x0 + 0.62, y0 + h - 0.34, head, ha="left", va="center", fontsize=9.4,
            fontweight="bold", color=color)
    ax.text(x0 + w / 2, y0 + 0.42, body, ha="center", va="center", fontsize=6.9,
            color=MUTED)
    return x0 + w / 2


def make_a5():
    W, H = 15.6, 10.9
    fig, ax = canvas(W, H)
    title(ax, W / 2, H - 0.30,
          "Aspect 5 \u2014 From HGT to a Foundation Model: what must change to pretrain on one "
          "database and reuse it on another")
    subtitle(ax, W / 2, H - 0.66,
             "Dry design. The blocker is that every HGT weight is stored in a dictionary keyed by a "
             "schema-specific type name, so nothing survives a change of schema.")

    # ---------------- the blocker -------------------------------------------
    y0, hgt = 7.55, 2.10
    ax.add_patch(FancyBboxPatch((0.45, y0), 14.70, hgt,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#fdf8f8", edgecolor="#dfb3ad", lw=1.2, zorder=1))
    ax.text(0.72, y0 + hgt - 0.30, "The blocker \u2014 standard HGT is welded to one schema",
            ha="left", va="center", fontsize=10, fontweight="bold", color=RED)

    cy = y0 + 0.82
    ax.text(1.55, cy + 0.72, "source DB (rel-stack)", ha="center", fontsize=7.6,
            color=MUTED, style="italic")
    for i, (nm, c) in enumerate([("users", BLUE), ("posts", GREEN), ("votes", "#c07a2b")]):
        schema_chip(ax, 1.55, cy + 0.34 - i * 0.42, nm, c)

    box(ax, 5.60, cy, 3.60, 1.30, "HGT parameters",
        "$W_{\\mathrm{users}}$, $W_{\\mathrm{posts}}$, $W_{\\mathrm{votes}}$, "
        "$W_{(\\mathrm{votes},\\,f2p,\\,\\mathrm{posts})}$ \u2026\n\n"
        "one query/key/value matrix per node-type NAME,\n"
        "one attention/message matrix per relation NAME",
        kind="mp", fs=9.4, sfs=7.0)
    arrow(ax, (2.30, cy), (3.75, cy), lw=1.2)
    ax.text(3.02, cy + 0.14, "fit", ha="center", fontsize=7.2, style="italic", color=MUTED)

    ax.text(12.05, cy + 0.72, "target DB (rel-trial)", ha="center", fontsize=7.6,
            color=MUTED, style="italic")
    for i, (nm, c) in enumerate([("studies", PURPLE), ("sponsors", "#0f8a8a"),
                                 ("conditions", "#b3549a")]):
        schema_chip(ax, 12.05, cy + 0.34 - i * 0.42, nm, c, w=1.55)
    arrow(ax, (7.45, cy), (11.10, cy), lw=1.3, color=RED, dashed=True)
    ax.text(9.25, cy + 0.42, "transfer the pretrained weights", ha="center",
            fontsize=7.6, style="italic", color=RED)
    ax.text(9.30, cy - 0.30, "\u2717  no dictionary entry exists for these names \u2014\n"
                             "there is no weight to load, and the column\n"
                             "encoders are per-table too",
            ha="center", va="top", fontsize=7.4, color=RED)

    # ---------------- four required changes ---------------------------------
    ax.text(0.45, 7.15, "Four required changes", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(3.30, 7.15, "\u2014 each one replaces something schema-specific with something "
                        "schema-agnostic", ha="left", va="center", fontsize=8.6, color=MUTED)

    cy0, ch = 4.35, 2.45
    cw, cgap = 3.45, 0.30
    xs = [0.45 + i * (cw + cgap) for i in range(4)]

    # card 1 - input
    c1 = card(ax, xs[0], cy0, cw, ch, 1, "Schema-agnostic input",
              "encoders keyed by the column's SEMANTIC TYPE\n"
              "(numeric / categorical / text / time), shared across\n"
              "databases, plus one frozen text LM; feature spaces\n"
              "normalized so a new table lands in the same space",
              GREEN)
    for j, st in enumerate(["num", "cat", "text"]):
        xx = c1 - 1.05 + j * 1.05
        ax.add_patch(FancyBboxPatch((xx - 0.42, cy0 + 1.35), 0.84, 0.38,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=ENC_F, edgecolor=ENC_E, lw=1.0, zorder=4))
        ax.text(xx, cy0 + 1.54, f"Enc$_{{{st}}}$", ha="center", va="center",
                fontsize=7.4, color=GREEN, fontweight="bold", zorder=5)
    ax.text(c1, cy0 + 1.10, "shared by every database", ha="center", fontsize=6.8,
            color=MUTED, style="italic")

    # card 2 - message passing
    c2 = card(ax, xs[1], cy0, cw, ch, 2, "Schema-agnostic MP",
              "a hypernetwork generates each relation's weights\n"
              "from a metadata embedding of that relation (name,\n"
              "the tables it joins, its column descriptions), so an\n"
              "unseen relation still gets weights at transfer time",
              "#b8791f")
    box(ax, c2 - 0.88, cy0 + 1.55, 1.35, 0.46, "$m_r$  metadata", kind="data", fs=7.6)
    arrow(ax, (c2 - 0.18, cy0 + 1.55), (c2 + 0.18, cy0 + 1.55), lw=1.1, ms=9)
    box(ax, c2 + 0.95, cy0 + 1.55, 1.30, 0.46, "$g_\\theta \\rightarrow W_r$",
        kind="mp", fs=8.2)
    ax.text(c2, cy0 + 1.12, "no per-type-name dictionary anywhere", ha="center",
            fontsize=6.8, color=MUTED, style="italic")

    # card 3 - pretraining
    c3 = card(ax, xs[2], cy0, cw, ch, 3, "Task-agnostic pretraining",
              "self-supervised objectives that need no task labels,\n"
              "so the backbone can be trained on any database:\n"
              "masked-cell reconstruction, PK\u2013FK link prediction,\n"
              "or a contrastive node objective",
              BLUE)
    for j, t in enumerate(["mask a cell", "predict a link", "contrast nodes"]):
        ax.add_patch(FancyBboxPatch((c3 - 1.32, cy0 + 1.72 - j * 0.32), 2.64, 0.27,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=DATA_F, edgecolor=DATA_E, lw=0.9, zorder=4))
        ax.text(c3, cy0 + 1.855 - j * 0.32, t, ha="center", va="center",
                fontsize=7.0, color=BLUE, zorder=5)

    # card 4 - readout
    c4 = card(ax, xs[3], cy0, cw, ch, 4, "Transferable readout",
              "the task head is detachable: on a new dataset,\n"
              "attach a fresh head and either linear-probe (freeze\n"
              "the backbone) or fine-tune it \u2014 nothing task-specific\n"
              "has to survive the move between databases",
              "#b5628a")
    box(ax, c4 - 0.80, cy0 + 1.55, 1.50, 0.46, "shared backbone", kind="mp", fs=7.6)
    box(ax, c4 + 0.95, cy0 + 1.55, 1.20, 0.46, "new head", kind="head", fs=7.8)
    arrow(ax, (c4 - 0.02, cy0 + 1.55), (c4 + 0.32, cy0 + 1.55), lw=1.1, ms=9, dashed=True)
    ax.text(c4, cy0 + 1.12, "detach / re-attach", ha="center", fontsize=6.8,
            color=MUTED, style="italic")

    # ---------------- the proposed experiment -------------------------------
    ax.plot([0.45, 15.15], [4.05, 4.05], color=RULE, lw=0.9)
    ax.text(0.45, 3.75, "How to test whether pretraining actually helps",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color=INK)

    ey = 2.05
    box(ax, 1.85, ey, 2.55, 1.55, "Pretrain on SOURCE",
        "rel-stack, self-supervised:\nno task labels used;\nkeep the backbone only",
        kind="mp", fs=9.2, sfs=7.0)
    arrow(ax, (3.20, ey), (4.05, ey), lw=1.3)
    ax.text(3.62, ey + 0.16, "backbone\nweights", ha="center", va="bottom",
            fontsize=6.8, style="italic", color=BLUE)

    settings = [("train from scratch", "no pretraining \u2014 the control", GREY_E),
                ("pretrain \u2192 linear-probe", "backbone frozen, head only", PURPLE),
                ("pretrain \u2192 fine-tune", "backbone + head updated", GREEN)]
    for i, (nm, sub, col) in enumerate(settings):
        yy = ey + 0.98 - i * 0.72
        ax.add_patch(FancyBboxPatch((4.15, yy - 0.28), 3.55, 0.56,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=col, lw=1.2, zorder=4))
        ax.text(4.32, yy + 0.10, nm, ha="left", va="center", fontsize=8.2,
                color=col, fontweight="bold", zorder=5)
        ax.text(4.32, yy - 0.13, sub, ha="left", va="center", fontsize=6.8,
                color=MUTED, zorder=5)
    ax.text(5.92, ey - 1.10, "evaluate all three on the TARGET database (rel-trial),\n"
                             "sweeping the fraction of target labels that is used",
            ha="center", va="center", fontsize=7.6, color=MUTED, style="italic")

    # few-shot curve
    px, py, pwd, phd = 8.85, 0.95, 4.20, 2.35
    ax.add_patch(FancyBboxPatch((px - 0.55, py - 0.45), pwd + 2.35, phd + 0.85,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#fbfcfd", edgecolor=RULE, lw=1.0, zorder=1))
    ax.plot([px, px, px + pwd], [py + phd, py, py], color=INK, lw=1.1, zorder=4)
    ax.text(px + pwd / 2, py - 0.28, "fraction of target labels used  \u2192",
            ha="center", fontsize=7.6, color=MUTED)
    ax.text(px - 0.30, py + phd / 2, "AUROC on target", rotation=90, ha="center",
            va="center", fontsize=7.6, color=MUTED)

    t = np.linspace(0.03, 1.0, 100)
    def curve(a, b):
        return py + phd * (a + b * (1 - np.exp(-3.2 * t))) / 1.0
    ax.plot(px + pwd * t, curve(0.05, 0.55), color="#8b949c", lw=1.9, zorder=5)
    ax.plot(px + pwd * t, curve(0.30, 0.40), color=PURPLE, lw=1.9, zorder=5)
    ax.plot(px + pwd * t, curve(0.40, 0.40), color=GREEN, lw=1.9, zorder=5)
    ax.text(px + pwd + 0.10, curve(0.05, 0.55)[-1], "from scratch", fontsize=7.4,
            color="#8b949c", va="center")
    ax.text(px + pwd + 0.10, curve(0.30, 0.40)[-1], "linear-probe", fontsize=7.4,
            color=PURPLE, va="center")
    ax.text(px + pwd + 0.10, curve(0.40, 0.40)[-1], "fine-tune", fontsize=7.4,
            color=GREEN, va="center")

    ax.add_patch(Rectangle((px, py), pwd * 0.22, phd, facecolor="#fff3d6",
                           edgecolor="none", alpha=0.7, zorder=2))
    ax.text(px + pwd * 0.11, py + phd + 0.16, "low-label regime", ha="center",
            fontsize=7.2, color="#a8791f", fontweight="bold")
    ax.annotate("", xy=(px + pwd * 0.11, curve(0.40, 0.40)[8]),
                xytext=(px + pwd * 0.11, curve(0.05, 0.55)[8]),
                arrowprops=dict(arrowstyle="<->", color="#a8791f", lw=1.3), zorder=6)
    ax.text(px + pwd * 0.24, py + phd * 0.16,
            "this gap is the evidence", ha="left", va="center", fontsize=7.2,
            color="#a8791f", style="italic", zorder=6)

    ax.text(W / 2, 0.26,
            "Pretraining is effective if the pretrained curves sit above the from-scratch curve, "
            "and above all if the gap is largest when target labels are scarce.",
            ha="center", va="center", fontsize=7.6, color=MUTED)

    save(fig, "architecture_A5")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT = sys.argv[1]
    make_a1(); make_a2(); make_a3(); make_a4(); make_a5()
    print(f"wrote architecture_A1..A5 (.png + .pdf) to {OUT}/")
