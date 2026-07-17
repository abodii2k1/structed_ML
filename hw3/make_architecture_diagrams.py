"""Regenerate the four aspect architecture diagrams in the course-slide style:
raw example tables as inputs, solid colored boxes for components (operations),
data (inputs/outputs) written on the arrows, and every pipeline step shown and
numbered (the numbers match the step-by-step tables in report.md).
Overwrites artifacts/architecture_A{1..4}.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = "artifacts"

# palette (kept consistent with the loss-curve figures already in the report)
C_ENC = "#c8e6c9"   # encoder green
C_MP = "#ffe0b2"    # message passing orange
C_HEAD = "#f8bbd0"  # prediction head pink
C_AUX = "#e6e0f0"   # auxiliary transform purple
C_GRAPH = "#d7e6f5"  # graph building / sampling blue
C_EDGE = "#333333"
C_DATA = "#1a4f8b"  # blue italic = data on arrows
C_HDR = "#555555"


def draw_table(ax, x, y_top, title, headers, rows, col_ws, row_h=0.30, fs=6.5):
    w = sum(col_ws)
    ax.text(x + w / 2, y_top + 0.06, title, ha="center", va="bottom",
            fontsize=fs + 2, fontweight="bold")
    cx = x
    for h_txt, cw in zip(headers, col_ws):
        ax.add_patch(Rectangle((cx, y_top - row_h), cw, row_h,
                               facecolor=C_HDR, edgecolor="#333333", lw=0.6))
        ax.text(cx + cw / 2, y_top - row_h / 2, h_txt, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold")
        cx += cw
    for i, row in enumerate(rows):
        ry = y_top - (i + 2) * row_h
        cx = x
        for val, cw in zip(row, col_ws):
            ax.add_patch(Rectangle((cx, ry), cw, row_h,
                                   facecolor="white", edgecolor="#999999", lw=0.5))
            ax.text(cx + cw / 2, ry + row_h / 2, str(val), ha="center",
                    va="center", fontsize=fs)
            cx += cw
    return x + w, y_top - (1 + len(rows)) * row_h


def step_num(ax, x, y, n):
    ax.add_patch(Circle((x, y), 0.17, facecolor="#333333", edgecolor="#333333",
                        lw=1.0, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7,
            fontweight="bold", color="white", zorder=7)


def comp(ax, cx, cy, w, h, title, sub=None, color=C_MP, fs=9, sfs=6.4, num=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="#333333", lw=1.3))
    if sub:
        n_sub = sub.count("\n") + 1
        ax.text(cx, cy + h / 2 - 0.26, title, ha="center", va="center",
                fontsize=fs, fontweight="bold")
        ax.text(cx, cy - h / 2 + 0.11 * n_sub + 0.12, sub, ha="center",
                va="center", fontsize=sfs)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold")
    if num is not None:
        step_num(ax, cx - w / 2, cy + h / 2, num)


def data_box(ax, cx, cy, w, h, text, fs=7.5, num=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=C_DATA,
                                lw=1.2, linestyle="--"))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            style="italic", color=C_DATA)
    if num is not None:
        step_num(ax, cx - w / 2, cy + h / 2, num)


def arrow(ax, p0, p1, rad=0.0, color=C_EDGE, lw=1.5, ms=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def dlab(ax, x, y, text, fs=6.5, ha="center", va="bottom", color=C_DATA):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, style="italic", color=color)


def legend_line(ax, x, y, fs=8):
    ax.add_patch(FancyBboxPatch((x, y - 0.09), 0.42, 0.24,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=C_MP, edgecolor="#333333", lw=1.0))
    ax.text(x + 0.55, y, "= component (operation)", fontsize=fs, va="center")
    ax.text(x + 2.65, y, "labeled arrow = data flowing (input / output)",
            fontsize=fs, va="center", style="italic", color=C_DATA)
    ax.add_patch(FancyBboxPatch((x + 6.6, y - 0.09), 0.42, 0.24,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="white", edgecolor=C_DATA,
                                lw=1.0, linestyle="--"))
    ax.text(x + 7.15, y, "= model output", fontsize=fs, va="center",
            style="italic", color=C_DATA)
    step_num(ax, x + 9.0, y, "n")
    ax.text(x + 9.25, y, "= step number (see table in report)", fontsize=fs,
            va="center")


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def stack_tables(ax, x, y_tops):
    """The three rel-stack example tables, stacked; returns (right_edge, centers)."""
    specs = [
        ("users", ["Id", "AboutMe", "Created"],
         [["1", "ML fan...", "2019-03"], ["2", "SQL dev...", "2020-11"]],
         [0.38, 0.85, 0.78]),
        ("posts", ["Id", "OwnerId", "Title"],
         [["7", "1", "How to..."], ["8", "2", "Why is..."]],
         [0.38, 0.72, 0.91]),
        ("votes", ["Id", "PostId", "UserId"],
         [["3", "7", "2"], ["4", "8", "1"]],
         [0.38, 0.82, 0.81]),
    ]
    centers, right = [], x
    for (title, hdrs, rows, ws), y_top in zip(specs, y_tops):
        r, bottom = draw_table(ax, x, y_top, title, hdrs, rows, ws)
        right = max(right, r)
        centers.append((y_top + bottom) / 2 - 0.15)
    ax.text(x + 1.3, y_tops[-1] - 1.12, "... + comments, badges, ... (all tables)",
            ha="center", fontsize=6.5, style="italic")
    return right, centers


def shared_row1(ax, cy, sample_sub, tab_label="Raw RDB (rel-stack)"):
    """Steps 1-4 shared by A1/A2/A4: tables -> build graph -> sample -> encode.
    Returns the encoder box (cx, w, h)."""
    tab_tops = [cy + 1.55, cy + 0.4, cy - 0.75]
    ax.text(1.45, cy + 1.95, tab_label, ha="center", fontsize=9, fontweight="bold")
    step_num(ax, 0.25, cy + 2.0, 1)
    right, centers = stack_tables(ax, 0.35, tab_tops)

    comp(ax, 4.7, cy, 2.1, 1.7, "Build graph",
         "make_pkey_fkey_graph:\none node per table row;\neach PK-FK link $\\rightarrow$ f2p_* +\nrev_f2p_* edge; + timestamps",
         color=C_GRAPH, num=2)
    for i, tc in enumerate(centers):
        arrow(ax, (right + 0.05, tc), (3.65, cy + (1 - i) * 0.4))
    dlab(ax, 3.28, cy + 1.35, "rows +\nPK-FK links", fs=6.5)

    comp(ax, 8.05, cy, 2.5, 1.7, "Sample subgraph", sample_sub,
         color=C_GRAPH, num=3)
    arrow(ax, (5.75, cy), (6.8, cy))
    dlab(ax, 6.27, cy + 0.12, "full typed graph\n(HeteroData)", fs=6.3)

    comp(ax, 11.65, cy, 2.1, 1.7, "HeteroEncoder",
         "per-column typed encoders\n(numeric / categorical /\nGloVe text), fused into\none vector per row",
         color=C_ENC, num=4)
    arrow(ax, (9.3, cy), (10.6, cy))
    dlab(ax, 9.95, cy + 0.12, "mini-batch subgraph:\nseed rows + sampled\nneighbors + their edges", fs=6.3)
    return 11.65, 2.1, 1.7


A1_SAMPLE = ("NeighborLoader: batch of 512\nlabeled seed users; $\\leq$128 neigh-"
             "\nbors per edge type per hop,\n2 hops, only past (time $\\leq$ seed)")


# --------------------------------------------------------------------------
# Aspect 1 - directionality
# --------------------------------------------------------------------------
def make_a1():
    fig, ax = new_ax(15.0, 12.2)
    ax.text(7.5, 11.85, "Aspect 1 - Message Directionality: shared pipeline, "
            "three edge-wiring modes", ha="center", fontsize=13, fontweight="bold")

    cy1 = 9.55
    enc_cx, enc_w, enc_h = shared_row1(ax, cy1, A1_SAMPLE)

    # row 2 (right -> left)
    cy2 = 6.45
    arrow(ax, (enc_cx, cy1 - enc_h / 2), (12.85, cy2 + 0.9), rad=-0.15)
    dlab(ax, 12.5, 8.1, "$x_v \\in \\mathbb{R}^{128}$\nfor every batch node", fs=6.5,
         ha="left")

    comp(ax, 12.85, cy2, 2.3, 1.8, "MP layer 1",
         "per wired edge type: SAGEConv\n(mean over sampled neighbors)\nor GATConv (4-head attention);\nsum over edge types; ReLU",
         color=C_MP, num=5)
    ax.text(12.85, cy2 - 1.12, "edge wiring = mode (below)", ha="center",
            fontsize=6.8, style="italic")
    arrow(ax, (11.7, cy2), (10.55, cy2))
    dlab(ax, 11.12, cy2 + 0.12, "$h_v^{(1)}$ per node", fs=6.5)

    comp(ax, 9.5, cy2, 2.1, 1.8, "MP layer 2",
         "same structure,\nfresh weights;\nsame mode wiring",
         color=C_MP, num=6)
    arrow(ax, (8.45, cy2), (7.35, cy2))
    dlab(ax, 7.9, cy2 + 0.12, "$h_v^{(2)}$ per node", fs=6.5)

    comp(ax, 6.35, cy2, 2.0, 1.8, "Take seed rows",
         "first 512 rows of the\nentity (users) store =\nthis batch's labeled\nseed entities",
         color=C_AUX, num=7)
    arrow(ax, (5.35, cy2), (4.5, cy2))
    dlab(ax, 4.92, cy2 + 0.12, "$h_{seed}$", fs=6.5)

    comp(ax, 3.6, cy2, 1.8, 1.8, "MLP Head",
         "Linear(128,128)\n$\\rightarrow$ ReLU $\\rightarrow$\nLinear(128,1)",
         color=C_HEAD, num=8)
    arrow(ax, (2.7, cy2), (2.05, cy2))
    data_box(ax, 1.12, cy2, 1.8, 1.1,
             "logit $\\hat{y}$ per seed;\nsigmoid $\\rightarrow$ prob.;\nBCE loss in training", fs=6.5)

    # ---- bottom band: the three wiring modes -----------------------------
    ax.text(7.5, 4.85, "Step 5/6 detail - the varied component: edge wiring "
            "inside each message-passing layer", ha="center", fontsize=10.5,
            fontweight="bold")
    panels = [
        (0.5, "MPNN-D (directed)", "forward edges only"),
        (5.35, "MPNN-U (undirected)", "both directions, one shared W"),
        (10.2, "Dir-GNN", "both directions, separate weights"),
    ]
    for px, title, sub in panels:
        ax.add_patch(FancyBboxPatch((px, 1.9), 4.3, 2.65,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#f8f8f8", edgecolor="#888888", lw=1.0))
        ax.text(px + 2.15, 4.28, title, ha="center", fontsize=9.5, fontweight="bold")
        ax.text(px + 2.15, 3.98, sub, ha="center", fontsize=7.5, style="italic")
        for dx, name in [(0.9, "post row\n(child)"), (3.4, "user row\n(parent)")]:
            ax.add_patch(FancyBboxPatch((px + dx - 0.62, 2.7), 1.24, 0.75,
                                        boxstyle="round,pad=0.02,rounding_size=0.06",
                                        facecolor="white", edgecolor="#666666", lw=1.0))
            ax.text(px + dx, 3.07, name, ha="center", va="center", fontsize=7.5)
    px = panels[0][0]
    arrow(ax, (px + 1.6, 3.25), (px + 2.7, 3.25), color="#1565c0")
    dlab(ax, px + 2.15, 3.45, "msg via W (f2p)", fs=7.5, color="#1565c0")
    ax.text(px + 2.15, 2.3, "reverse edges: unused", ha="center",
            fontsize=7.5, color="#aa3333", style="italic")
    px = panels[1][0]
    arrow(ax, (px + 1.6, 3.3), (px + 2.7, 3.3), rad=-0.35, color="#1565c0")
    dlab(ax, px + 2.15, 3.62, "msg via W (f2p)", fs=7.5, color="#1565c0")
    arrow(ax, (px + 2.7, 2.9), (px + 1.6, 2.9), rad=-0.35, color="#1565c0")
    ax.text(px + 2.15, 2.2, "msg via the SAME W (rev_f2p)", ha="center",
            fontsize=7.5, color="#1565c0", style="italic")
    px = panels[2][0]
    arrow(ax, (px + 1.6, 3.3), (px + 2.7, 3.3), rad=-0.35, color="#1565c0")
    dlab(ax, px + 2.15, 3.62, "msg via $W_{fwd}$ (f2p)", fs=7.5, color="#1565c0")
    arrow(ax, (px + 2.7, 2.9), (px + 1.6, 2.9), rad=-0.35, color="#c62828")
    ax.text(px + 2.15, 2.2, "msg via separate $W_{rev}$ (rev_f2p)", ha="center",
            fontsize=7.5, color="#c62828", style="italic")

    legend_line(ax, 1.4, 1.35)
    fig.savefig(f"{OUT}/architecture_A1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Aspect 2 - heterogeneity
# --------------------------------------------------------------------------
def make_a2():
    fig, ax = new_ax(16.0, 12.8)
    ax.text(8.0, 12.45, "Aspect 2 - Heterogeneity: typed vs. untyped message "
            "passing (shared steps 1-4 and 9-10)", ha="center", fontsize=13,
            fontweight="bold")

    cy1 = 10.25
    enc_cx, enc_w, enc_h = shared_row1(ax, cy1, A1_SAMPLE)

    # fan out to the two branches
    hy, ly = 7.15, 4.35
    arrow(ax, (enc_cx, cy1 - enc_h / 2), (14.05, hy + 0.95), rad=-0.15)
    dlab(ax, 13.6, 8.95, "$x_v \\in \\mathbb{R}^{128}$ per node, still\ntyped (one "
         "matrix $X_{type}$\nper node type)", fs=6.8, ha="left")
    ax.plot([enc_cx + 1.05, 15.82, 15.82], [cy1, cy1, ly], color=C_EDGE, lw=1.5)
    arrow(ax, (15.82, ly), (15.66, ly))
    dlab(ax, 15.0, cy1 + 0.12, "same $x_v$", fs=6.3)

    # --- heterogeneous branch ---------------------------------------------
    ax.text(10.9, 8.32, "Heterogeneous branch: types kept", fontsize=10,
            fontweight="bold", color="#1565c0", ha="center")
    comp(ax, 14.05, hy, 2.5, 1.9, "HeteroConv layer 1",
         "one SAGEConv per edge type\n(mean over neighbors) or native\nHGTConv (4-head attention);\nsum over edge types; ReLU",
         color=C_MP, num="5a")
    arrow(ax, (12.8, hy), (11.75, hy))
    dlab(ax, 12.27, hy + 0.12, "$h_v^{(1)}$ per\nnode type", fs=6.3)
    comp(ax, 10.7, hy, 2.1, 1.9, "HeteroConv layer 2",
         "same structure,\nfresh weights", color=C_MP, fs=8.5, num="6a")
    arrow(ax, (9.65, hy), (8.6, hy))
    dlab(ax, 9.12, hy + 0.12, "$h_v^{(2)}$ per\nnode type", fs=6.3)
    comp(ax, 7.55, hy, 2.1, 1.9, "Take entity rows",
         "keep the entity type's\n(users) embeddings;\nfirst 512 rows = this\nbatch's seed entities",
         color=C_AUX, num="7a")

    # --- homogeneous branch -------------------------------------------------
    ax.text(9.9, 3.05, "Homogeneous branch: types erased", fontsize=10,
            fontweight="bold", color="#6a4fa3", ha="center")
    comp(ax, 14.35, ly, 2.6, 1.9, "collapse()",
         "concat all $X_{type}$ into one $x_{all}$\n(record each type's row offset);\nshift every edge index by the\noffsets; merge into one edge set",
         color=C_AUX, num="5b")
    arrow(ax, (13.05, ly), (12.0, ly))
    dlab(ax, 12.52, ly + 0.12, "($x_{all}$, $ei_{all}$):\none untyped graph", fs=6.3)
    comp(ax, 10.95, ly, 2.1, 1.9, "single Conv layer 1",
         "ONE type-agnostic\nSAGEConv / HGTConv-on-\n1-type-graph (see note)\nfor ALL nodes; ReLU",
         color=C_MP, fs=8, num="6b")
    arrow(ax, (9.9, ly), (9.0, ly))
    dlab(ax, 9.45, ly + 0.12, "$h_{all}^{(1)}$", fs=6.5)
    comp(ax, 8.0, ly, 2.0, 1.9, "single Conv layer 2",
         "same structure,\nfresh weights", color=C_MP, fs=8, num="7b")
    arrow(ax, (7.0, ly), (6.15, ly))
    dlab(ax, 6.57, ly + 0.12, "$h_{all}^{(2)}$", fs=6.5)
    comp(ax, 5.15, ly, 2.0, 1.9, "Slice by offset",
         "cut the entity type's\nrows back out of $h_{all}$\nusing its stored offset;\nfirst 512 = seeds",
         color=C_AUX, num="8b")

    # --- shared head --------------------------------------------------------
    head_cx, head_cy = 2.6, 5.9
    arrow(ax, (6.5, hy), (3.35, head_cy + 0.55), rad=0.12)
    dlab(ax, 5.1, 7.28, "$h_{seed} \\in \\mathbb{R}^{128}$", fs=6.8, ha="left")
    arrow(ax, (4.15, ly), (3.35, head_cy - 0.5), rad=-0.12)
    dlab(ax, 3.35, 4.72, "$h_{seed}$", fs=6.8, ha="right")
    comp(ax, head_cx, head_cy, 1.9, 1.5, "MLP Head (shared)",
         "Linear(128,128) $\\rightarrow$ ReLU\n$\\rightarrow$ Linear(128,1)",
         color=C_HEAD, fs=8, num=9)
    arrow(ax, (head_cx, head_cy - 0.75), (head_cx, head_cy - 1.5))
    data_box(ax, head_cx, head_cy - 2.1, 2.0, 1.15,
             "logit $\\hat{y}$ per seed;\nsigmoid $\\rightarrow$ prob.;\nBCE loss in training",
             fs=6.5, num=10)

    # note box
    ax.add_patch(FancyBboxPatch((3.0, 1.15), 11.6, 1.15,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#f8f8f8", edgecolor="#888888", lw=1.0))
    ax.text(8.8, 1.72, "Note (HGT-homo, step 6b): heterogeneity is disabled at the "
            "data level - the merged graph is wrapped in an explicit single-type "
            "HeteroData\n(all nodes type \"node\", all edges (\"node\",\"to\",\"node\")) "
            "before calling the same HGTConv, so its type-lookup machinery sees "
            "exactly one type.",
            ha="center", va="center", fontsize=8)

    legend_line(ax, 1.8, 0.55)
    fig.savefig(f"{OUT}/architecture_A2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Aspect 3 - node features
# --------------------------------------------------------------------------
def make_a3():
    fig, ax = new_ax(15.5, 12.0)
    ax.text(7.75, 11.65, "Aspect 3 - Node Features: three swappable input "
            "encoders, identical downstream model", ha="center", fontsize=13,
            fontweight="bold")

    # row 1: table -> build graph -> fixed cached subsample
    cy1 = 10.0
    ax.text(1.75, 10.75, "one node = one row", ha="center", fontsize=9,
            fontweight="bold")
    step_num(ax, 0.3, 10.8, 1)
    r_edge, bottom = draw_table(
        ax, 0.35, 10.35, "studies (rel-trial)",
        ["NctId", "Phase", "Summary"],
        [["NCT01", "2", "A study of..."], ["NCT02", "3", "Trial for..."]],
        [0.55, 0.55, 1.4])
    ax.text(1.6, bottom - 0.22, "... + conditions, sponsors, ... (all tables)",
            ha="center", fontsize=6.5, style="italic")
    row_c = (10.35 + bottom) / 2 - 0.15

    comp(ax, 5.0, cy1, 2.1, 1.7, "Build graph",
         "make_pkey_fkey_graph:\none node per table row;\neach PK-FK link $\\rightarrow$ f2p_* +\nrev_f2p_* edge; + timestamps",
         color=C_GRAPH, num=2)
    arrow(ax, (r_edge + 0.05, row_c), (3.95, cy1))
    dlab(ax, 3.45, cy1 + 0.35, "rows +\nPK-FK links", fs=6.5)

    comp(ax, 8.9, cy1, 2.9, 1.7, "Fixed cached subsample",
         "label-stratified seed studies (6000\ntrain / 2000 val) + their 2-hop time-\nrespecting neighborhoods (fan-out\n[6,6]); cached once, shared by all 3",
         color=C_GRAPH, num=3)
    arrow(ax, (6.05, cy1), (7.45, cy1))
    dlab(ax, 6.75, cy1 + 0.12, "full typed graph\n(HeteroData)", fs=6.3)

    # fan from the cached subgraph down to the three encoders
    ax.text(12.6, 10.55, "exactly ONE of steps 4a / 4b / 4c\nis active per run; "
            "all three see the\nsame cached subgraph", fontsize=7.5,
            style="italic", ha="center")

    enc_cx, mid = 3.6, 4.6
    encs = [
        ("4a", 6.6, "id encoder",
         "nn.Embedding($N_{type}$, 128) per node\ntype: pure lookup by node index;\nignores every cell value; transductive",
         C_AUX, "node index $i$ only"),
        ("4b", 4.6, "column encoder",
         "HeteroEncoder: per-column typed\nencoders (torch_frame), fused\ninto one vector per row",
         C_ENC, "typed cell values"),
        ("4c", 2.6, "llm encoder",
         "row $\\rightarrow$ string \"NctId=NCT01,\nPhase=2, ...\" $\\rightarrow$ frozen MiniLM\n(384-d) $\\rightarrow$ learned Linear(384,128)",
         "#d7e6f5", "row serialized to text"),
    ]
    # elbow from subsample box down-left to a fan point
    fan_x, fan_y = 1.0, 4.6
    ax.plot([8.9, 8.9], [cy1 - 0.85, 8.35], color=C_EDGE, lw=1.5)
    ax.plot([8.9, fan_x], [8.35, 8.35], color=C_EDGE, lw=1.5)
    ax.plot([fan_x, fan_x], [8.35, fan_y], color=C_EDGE, lw=1.5)
    dlab(ax, 4.9, 8.47, "cached subgraph: same nodes, edges, seed labels for all "
         "three encoders", fs=6.8)
    for num, ecy, name, sub, col, in_lab in encs:
        arrow(ax, (fan_x, fan_y), (enc_cx - 1.3, ecy),
              rad=-0.12 if ecy > fan_y else (0.12 if ecy < fan_y else 0.0))
        comp(ax, enc_cx, ecy, 2.6, 1.5, name, sub, color=col, fs=8.5, num=num)
    dlab(ax, 1.5, 6.3, "node index $i$ only", fs=6.3)
    dlab(ax, 1.75, 4.72, "typed cell values", fs=6.3)
    dlab(ax, 1.3, 2.45, "row serialized\nto text", fs=6.3, va="top")

    # downstream (identical for all three)
    hgt1_cx = 7.15
    for _, ecy, *_ in encs:
        arrow(ax, (enc_cx + 1.3, ecy), (hgt1_cx - 1.15, mid + (ecy - mid) * 0.18))
    dlab(ax, 5.45, 4.85, "$x_v \\in \\mathbb{R}^{128}$", fs=7)
    comp(ax, hgt1_cx, mid, 2.3, 1.8, "HGTConv layer 1",
         "4-head attention with per-\nnode-type and per-relation\nprojections; messages\nsummed per node; ReLU",
         color=C_MP, fs=8.5, num=5)
    arrow(ax, (8.3, mid), (9.15, mid))
    dlab(ax, 8.72, mid + 0.12, "$h_v^{(1)}$", fs=6.5)
    comp(ax, 10.15, mid, 2.0, 1.8, "HGTConv layer 2",
         "same structure,\nfresh weights", color=C_MP, fs=8.5, num=6)
    arrow(ax, (11.15, mid), (11.95, mid))
    dlab(ax, 11.55, mid + 0.12, "$h_v^{(2)}$", fs=6.5)
    comp(ax, 12.95, mid, 2.0, 1.8, "Take seed rows",
         "keep the 6000 / 2000\nseed studies'\nembeddings",
         color=C_AUX, fs=8.5, num=7)
    arrow(ax, (12.95, mid - 0.9), (12.95, mid - 1.75))
    dlab(ax, 13.15, mid - 1.35, "$h_{seed}$", fs=6.5, ha="left")
    comp(ax, 12.95, mid - 2.5, 2.0, 1.4, "MLP Head",
         "Linear(128,128) $\\rightarrow$ ReLU\n$\\rightarrow$ Linear(128,1)",
         color=C_HEAD, fs=8.5, num=8)
    arrow(ax, (11.95, mid - 2.5), (11.3, mid - 2.5))
    data_box(ax, 10.25, mid - 2.5, 1.95, 1.1,
             "logit $\\hat{y}$ per seed;\nsigmoid $\\rightarrow$ prob.;\nBCE loss in training", fs=6.5)

    legend_line(ax, 1.0, 0.45)
    fig.savefig(f"{OUT}/architecture_A3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Aspect 4 - depth / oversmoothing
# --------------------------------------------------------------------------
def make_a4():
    fig, ax = new_ax(15.5, 11.8)
    ax.text(7.75, 11.45, "Aspect 4 - Depth / Oversmoothing: GCN stack of depth L, "
            "optional skip connection", ha="center", fontsize=13, fontweight="bold")

    cy1 = 9.2
    enc_cx, enc_w, enc_h = shared_row1(
        ax, cy1,
        "NeighborLoader: batch of 256\nseed entities; $\\leq$10 neighbors\nper edge type per hop, 2 hops,\nonly past (time $\\leq$ seed) - FIXED\nacross all depths L")

    # row 2 (right -> left)
    cy2 = 5.7
    arrow(ax, (enc_cx, cy1 - enc_h / 2), (13.6, cy2 + 1.0), rad=-0.15)
    dlab(ax, 13.15, 7.6, "$x_v \\in \\mathbb{R}^{128}$ per node,\nper node type", fs=6.5,
         ha="left")

    comp(ax, 13.6, cy2, 2.4, 1.8, "collapse()",
         "concat per-type $X$ into $x_{all}$\n(record row offsets); shift +\nmerge all edge indices\n(same as Aspect 2)",
         color=C_AUX, num=5)
    arrow(ax, (12.4, cy2), (11.45, cy2))
    dlab(ax, 11.88, cy2 + 0.12, "$h^{(0)} = x_{all}$ +\nmerged edge set", fs=6)

    # GCN block (step 6)
    g_cx = 9.95
    ax.add_patch(FancyBboxPatch((g_cx - 1.5, cy2 - 1.15), 3.0, 2.25,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="none", edgecolor="#8d6e63",
                                lw=1.4, linestyle="--"))
    ax.text(g_cx, cy2 + 1.28, "repeat $\\times L$,  $L \\in \\{1,2,3,4,6,8\\}$",
            ha="center", fontsize=8.5, fontweight="bold", color="#6d4c41")
    step_num(ax, g_cx - 1.5, cy2 + 1.1, 6)
    comp(ax, g_cx, cy2 + 0.42, 2.2, 0.95, "GCNConv + ReLU",
         "deg.-normalized sum over\nneighbors; in $h^{(l-1)}$, out $h^{(l)}$",
         color=C_MP, fs=8.5, sfs=6)
    arrow(ax, (g_cx + 1.1, cy2 - 0.06), (g_cx - 1.1, cy2 - 0.06), rad=-0.4,
          color="#c62828", lw=1.2, ms=10)
    ax.text(g_cx, cy2 - 0.82, "optional skip:\n$h^{(l)}$ = ReLU($h^{(l-1)}$ + conv($h^{(l-1)}$))",
            ha="center", va="center", fontsize=6.6, color="#c62828", style="italic")

    arrow(ax, (8.45, cy2), (7.55, cy2))
    dlab(ax, 8.0, cy2 + 0.12, "$h^{(L)}$", fs=6.5)
    comp(ax, 6.55, cy2, 2.0, 1.8, "Slice by offset",
         "cut the entity type's rows\nout of $h^{(L)}$ by stored\noffset; first 256 rows =\nseed entities",
         color=C_AUX, fs=8.5, num=7)
    arrow(ax, (5.55, cy2), (4.7, cy2))
    dlab(ax, 5.12, cy2 + 0.12, "$h_{seed}$", fs=6.5)
    comp(ax, 3.8, cy2, 1.8, 1.8, "MLP Head",
         "Linear(128,128)\n$\\rightarrow$ ReLU $\\rightarrow$\nLinear(128,1)",
         color=C_HEAD, fs=8.5, num=8)
    arrow(ax, (2.9, cy2), (2.25, cy2))
    data_box(ax, 1.25, cy2, 1.85, 1.1,
             "logit $\\hat{y}$ per seed;\nsigmoid $\\rightarrow$ prob.;\nBCE loss in training", fs=6.5)

    ax.add_patch(FancyBboxPatch((2.4, 2.2), 10.8, 1.35,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#f8f8f8", edgecolor="#888888", lw=1.0))
    ax.text(7.8, 2.87, "Because the sampled 2-hop subgraph (step 3) is identical "
            "across all depths, depth L only adds propagation rounds over the\n"
            "same neighborhood, not a larger receptive field. Baseline (no-skip): "
            "$h^{(l)}$ = ReLU(conv($h^{(l-1)}$)). Oversmoothing is measured on "
            "$h^{(L)}$\n(cos_sim, dir_energy) alongside the downstream metrics.",
            ha="center", va="center", fontsize=8)

    legend_line(ax, 1.8, 1.4)
    fig.savefig(f"{OUT}/architecture_A4.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_a1()
    make_a2()
    make_a3()
    make_a4()
    print("wrote architecture_A1..A4.png")
