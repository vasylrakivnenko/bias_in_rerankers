"""
Generate the three figures for the legal paper, from the saved experiment outputs.
All figures are vector PDFs written to legal_paper/figures/.

Run:  ../venv/bin/python generate_figures.py

READ THIS BEFORE EDITING
------------------------
Three input schemas changed in Aug 2026 (see REVIEW_TODO.md):

  * results/single_stage_summary.json  (NEW, from code/analyze_single_stage.py)
    Authoritative per-model statistics: 14 models (the byte-identical
    semantic-ranker-default-003 is already excluded), three-way M/tie/F splits,
    tie-aware percentages, cluster-bootstrap 95% CIs, vendor strings.
    Figure 1 reads this file and never re-globs results/*_full_raw.json.

  * results/compounding_bge_pipeline.json
    slate.<cat>.{emb,rr,pipe}_top<k>_m are PERCENTAGES 0-100 (the raw mean count
    out of k lives in the parallel *_count field). Do NOT divide by the
    shortlist size again -- the old code did `100 * value / 3` and would now
    render ~2600%.

  * results/deidentification_bge.json
    conditions.<c>.pct_male is a LEGACY per-category dict of FRACTIONS (0-1).
    conditions.<c>.by_category.<cat>.pct_male and .overall.pct_male are the new
    PERCENTAGES (0-100). This script uses the new percentage fields only.

House style: the paper is written for a reader with no technical background.
No jargon in any axis label, legend or title -- say "re-ranker", "retriever",
"the man's document", "counterfactual pairs". Figures must stay legible in
grayscale, so every categorical fill is separated by lightness as well as hue
and every segment carries its number.

REVIEW_ROUND2 A3: "search step" / "ranking model" were an earlier, simplified
vocabulary the author asked to be reverted -- the paper now consistently says
"retriever" / "re-ranker" (paper.md agrees). A9: paths below are now relative
to this file rather than hard-coded to one machine's home directory, so this
script (and the reproduction command in the top-level README) works after a
fresh `git clone` anywhere.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = str(REPO_ROOT / "results")
FIGDIR = str(REPO_ROOT / "legal_paper" / "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"font.size": 9, "font.family": "serif"})

# Muted palette, kept from the previous version. Lightness is deliberately
# spread out so the fills survive a grayscale print.
MALE = "#3b6ea5"      # blue      (lightness ~0.42)
FEMALE = "#c2607a"    # rose      (lightness ~0.53)
MALE_DARK = "#2f5474"
INK = "#333333"
MUTED = "#777777"
GRID = "#dddddd"
REFERENCE = "#929292"

# Shared figure typography. All three PDFs are authored at the same width and
# inserted at the same manuscript width, so these sizes remain consistent after
# LaTeX scaling.
FONT_AXIS = 9.0
FONT_TICK = 8.4
FONT_LEGEND = 8.2
FONT_VALUE = 7.8
FONT_NOTE = 7.6

# Every figure is inserted at 0.78 of the manuscript's 6.5-inch text block.
# A common source width makes type, strokes, markers, and legends scale equally.
W_FIG = 6.00


def style(ax, axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if axis == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    else:
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.0)


def _load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Figure 1: a direct, paired comparison for each re-ranker. The previous
# scatter plot encoded vendors as shapes and models as numbers, forcing the
# reader to decode two axes and a long key. This dumbbell view puts one named
# model on each row and asks one question: how far does the model move between
# female-typed and male-typed occupations?
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "BAAI/bge-reranker-v2-m3": "BAAI: BGE v2-m3",
    "Cohere-rerank-v4.0-fast": "Cohere: v4 fast",
    "Cohere-rerank-v4.0-pro": "Cohere: v4 pro",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Open: MiniLM-L12-v2",
    "ibm-granite-granite-embedding-reranker-english-r2": "IBM: Granite r2",
    "jinaai/jina-reranker-v2-base-multilingual": "Jina: reranker v2",
    "rerank-1": "Voyage: rerank-1",
    "rerank-2": "Voyage: rerank-2",
    "rerank-2-lite": "Voyage: rerank-2-lite",
    "rerank-2.5": "Voyage: rerank-2.5",
    "rerank-2.5-lite": "Voyage: rerank-2.5-lite",
    "semantic-ranker-default-002": "Google: default-002",
    "semantic-ranker-default-004": "Google: default-004",
    "semantic-ranker-fast-004": "Google: fast-004",
}


def fig_single_stage():
    s = _load("single_stage_summary.json")
    models = s["models"]
    # REVIEW_TODO A1: the deprecated Google model returned scores byte-identical
    # to its successor and must never appear as a fourteenth/fifteenth point.
    assert "semantic-ranker-default-003" not in models, \
        "A1: semantic-ranker-default-003 must be excluded from the summary"
    assert s["n_models"] == len(models), "summary n_models disagrees with its model table"

    rows = []
    for key, m in models.items():
        male = m["by_category"]["male"]
        fem = m["by_category"]["female"]
        rows.append({
            "name": key,
            "label": MODEL_LABELS.get(key, key.split("/")[-1]),
            # Tie-aware: ties count half a win each (REVIEW_TODO A2). Tie rates
            # reach 22.7% on one model, so plain "score gap > 0" would silently
            # credit every tie to the woman's document.
            "male_jobs": male["tie_aware_pct_male"],
            "female_jobs": fem["tie_aware_pct_male"],
        })

    # Largest shift first. Every difference is positive; sorting makes both the
    # shared direction and the variation in its size visible immediately.
    rows.sort(key=lambda r: r["male_jobs"] - r["female_jobs"], reverse=True)

    # Keep the artwork compact and strongly horizontal. The paper's prose and
    # caption carry the takeaway, so the figure itself needs only the key,
    # directly labelled rows, and the measurement axis.
    fig_w = W_FIG
    fig_h = 3.00
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_box = [0.32, 0.31, 0.66, 0.55]
    ax = fig.add_axes(ax_box)
    ax.axvline(50, color=REFERENCE, lw=0.9, ls="--", zorder=1)

    ypos = np.arange(len(rows))
    for y, r in zip(ypos, rows):
        ax.plot([r["female_jobs"], r["male_jobs"]], [y, y],
                color="#b9c0c7", lw=2.0, solid_capstyle="round", zorder=2)
    ax.scatter([r["female_jobs"] for r in rows], ypos, s=34, color=FEMALE,
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.scatter([r["male_jobs"] for r in rows], ypos, s=34, color=MALE,
               edgecolor="white", linewidth=0.6, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=FONT_TICK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Male-associated document favored (exact ties split evenly, %)",
                  fontsize=FONT_AXIS)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    style(ax, axis="x")

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.5,
               color=MALE, markeredgecolor="white", markeredgewidth=0.5,
               label="male-typed occupations"),
        Line2D([], [], linestyle="none", marker="o", markersize=5.5,
               color=FEMALE, markeredgecolor="white", markeredgewidth=0.5,
               label="female-typed occupations"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               frameon=False, fontsize=FONT_LEGEND, ncol=2, handletextpad=0.4,
               columnspacing=1.4)
    # Place this directly over x=50; tying it to ax_box avoids the misleading
    # left-shift that a hard-coded page coordinate caused in the first draft.
    fig.text(ax_box[0] + 0.5 * ax_box[2], ax_box[1] + ax_box[3] + 0.008,
             "50 = equal split", fontsize=FONT_NOTE, color=MUTED,
             ha="center", va="bottom")

    fig.savefig(f"{FIGDIR}/fig_single_stage.pdf")
    plt.close(fig)
    print("wrote fig_single_stage.pdf  (%d directly labelled models; shifts %.1f--%.1f points)"
          % (len(rows),
             min(r["male_jobs"] - r["female_jobs"] for r in rows),
             max(r["male_jobs"] - r["female_jobs"] for r in rows)))
    return rows


# ---------------------------------------------------------------------------
# Figure 2: shortlist composition by stage and job category.
# ---------------------------------------------------------------------------
def fig_pipeline():
    d = _load("compounding_bge_pipeline.json")
    slate = d["slate"]
    k = d["headline_shortlist"]

    cats = ["male", "female", "neutral", "ALL"]
    catlabel = {"male": "Male-typed\noccupations",
                "female": "Female-typed\noccupations",
                "neutral": "Evenly split\noccupations",
                "ALL": "All\noccupations"}
    stages = [(f"emb_top{k}_m", "Retriever only", "#9bbcd6"),
              (f"rr_top{k}_m", "Reranker only", "#5a8bbd"),
              (f"pipe_top{k}_m", "Full pipeline", "#23456b")]

    x = np.arange(len(cats))
    w = 0.26
    fig = plt.figure(figsize=(W_FIG, 3.50))
    ax = fig.add_axes([0.14, 0.24, 0.72, 0.70])

    for i, (key, lab, col) in enumerate(stages):
        # Already a percentage 0-100 in the new schema -- no division by the
        # shortlist size (the raw mean count is in <key>_count).
        vals = [slate[c][key] for c in cats]
        los = [slate[c][key] - slate[c][key + "_ci"][0] for c in cats]
        his = [slate[c][key + "_ci"][1] - slate[c][key] for c in cats]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col, zorder=3)
        ax.errorbar(x + (i - 1) * w, vals, yerr=[los, his], fmt="none",
                    ecolor="#404040", elinewidth=0.8, capsize=2.0,
                    capthick=0.8, zorder=4)
        for b, v, h in zip(bars, vals, his):
            ax.text(b.get_x() + b.get_width() / 2, v + h + 1.6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=FONT_VALUE, color=INK)

    ax.axhline(50, color=REFERENCE, lw=0.9, ls="--", zorder=2)
    ax.text(1.03, 50, "equal shares\n(50%)", transform=ax.get_yaxis_transform(),
            color=MUTED, fontsize=FONT_NOTE, ha="left", va="center", linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([catlabel[c] for c in cats], fontsize=FONT_TICK)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.set_ylabel("Male-associated share of\nthe top-%d shortlist (%%)" % k,
                  fontsize=FONT_AXIS, linespacing=1.25)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT_LEGEND, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.045), handlelength=1.3,
               columnspacing=1.6)
    style(ax)

    fig.savefig(f"{FIGDIR}/fig_pipeline.pdf")
    plt.close(fig)
    print("wrote fig_pipeline.pdf  (top-%d %%male: " % k
          + ", ".join("%s search %.1f / model %.1f / pipeline %.1f"
                      % (c, slate[c][f"emb_top{k}_m"], slate[c][f"rr_top{k}_m"],
                         slate[c][f"pipe_top{k}_m"]) for c in cats) + ")")
    return {c: {s[0]: slate[c][s[0]] for s in stages} for c in cats}


# ---------------------------------------------------------------------------
# Figure 3: de-identification -- tie-aware balance for every counterfactual pair.
# ---------------------------------------------------------------------------
def fig_deid():
    d = _load("deidentification_bge.json")
    conds = d["conditions"]

    # Story order: no transform -> case-only control -> each half on its own ->
    # the two naive grammatical rewrites -> the grammar-aware rewrite -> the
    # full transform.
    order = [
        ("original", "Original documents"),
        ("lowercase_only", "Lowercased only\n(no markers removed)"),
        ("names_only", "Names removed only"),
        ("pronouns_only", "Pronouns neutralised only"),
        ("gram_her_their", "Simple rewrite: her $\\rightarrow$ their"),
        ("gram_her_them", "Simple rewrite: her $\\rightarrow$ them"),
        ("gram_pos", "Next-word rule for 'her'"),
        ("full", "Names removed and\npronouns neutralised"),
    ]
    order = [(k, lab) for k, lab in order if k in conds]
    # A condition present in the data but absent from `order` would be silently
    # dropped from the figure while the caption still says "each rule" -- which
    # is exactly what happened to `lowercase_only` when it was first added.
    missing = [k for k in conds if k not in {o[0] for o in order}]
    if missing:
        raise KeyError(
            f"conditions present in the results but missing from Figure 3's order "
            f"list: {missing}. Add a label for each, or the figure will disagree "
            f"with Table A8 and its own caption."
        )

    fig = plt.figure(figsize=(W_FIG, 3.50))
    ax = fig.add_axes([0.30, 0.27, 0.68, 0.65])

    ypos = np.arange(len(order))[::-1]
    bar_h = 0.62
    for y, (key, _) in zip(ypos, order):
        # New schema: overall.pct_male / pct_tie / pct_female are PERCENTAGES
        # (0-100) and sum to 100. The legacy conditions.<c>.pct_male dict is a
        # per-category dict of FRACTIONS (0-1) and is deliberately not used.
        o = conds[key]["overall"]
        # Split each exact tie evenly between the two sides. This is the same
        # tie-aware convention used in Figures 1 and 2, and makes a row of all
        # ties read visually as the balanced 50/50 result it represents. The
        # caption and footer still state that the last two rows are exact ties,
        # rather than observed wins for either document.
        segs = [(o["pct_male"] + 0.5 * o["pct_tie"], MALE, "white"),
                (o["pct_female"] + 0.5 * o["pct_tie"], FEMALE, "white")]
        left = 0.0
        for val, col, txtcol in segs:
            if val <= 0:
                continue
            ax.barh(y, val, bar_h, left=left, color=col, zorder=3,
                    edgecolor="white", linewidth=0.7)
            if val >= 7:
                ax.text(left + val / 2, y, f"{val:.0f}", ha="center", va="center",
                        fontsize=FONT_VALUE, color=txtcol, zorder=5)
            left += val
        lo, hi = o["tie_aware_pct_male_ci"]
        if hi - lo > 0.5:
            ax.plot([lo, hi], [y + bar_h / 2 + 0.16] * 2, color=MALE_DARK,
                    lw=0.9, zorder=6, solid_capstyle="butt")
            for e in (lo, hi):
                ax.plot([e, e], [y + bar_h / 2 + 0.09, y + bar_h / 2 + 0.23],
                        color=MALE_DARK, lw=0.9, zorder=6)

    # Drawn behind the bars, so it shows in the gaps between them and never
    # runs through a number.
    ax.axvline(50, color=REFERENCE, lw=0.9, ls="--", zorder=2)
    ax.text(51.5, len(order) - 0.30, "50%", color=MUTED, fontsize=FONT_NOTE,
            ha="left", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in order], fontsize=FONT_TICK, linespacing=1.25)
    ax.set_ylim(-0.65, len(order) + 0.15)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share after splitting exact ties evenly (%s pairs, %%)"
                  % f"{d['n_triples']:,}", fontsize=FONT_AXIS)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    style(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend_items = [
        Patch(facecolor=MALE, label="male-associated side"),
        Patch(facecolor=FEMALE, label="female-associated side"),
    ]
    fig.legend(handles=legend_items, frameon=False, fontsize=FONT_LEGEND,
               loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
               handlelength=1.1, handleheight=0.9, columnspacing=1.8,
               handletextpad=0.5)

    fig.savefig(f"{FIGDIR}/fig_deid.pdf")
    plt.close(fig)
    print("wrote fig_deid.pdf  ("
          + "; ".join("%s M%.0f/T%.0f/F%.0f"
                      % (k, conds[k]["overall"]["pct_male"],
                         conds[k]["overall"]["pct_tie"],
                         conds[k]["overall"]["pct_female"]) for k, _ in order)
          + ")")
    return {k: conds[k]["overall"] for k, _ in order}


if __name__ == "__main__":
    fig_single_stage()
    fig_pipeline()
    fig_deid()
    print("done ->", FIGDIR)
