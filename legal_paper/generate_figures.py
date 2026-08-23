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
import matplotlib.patheffects as pe
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
TIE = "#d8d8d8"       # pale grey (lightness ~0.85)
MALE_DARK = "#2f5474"
INK = "#333333"
MUTED = "#777777"
GRID = "#dddddd"

# The paper's figures are included at 0.78--0.82 of a 6.5in text block, so the
# figures are authored at close to their printed size: no font shrinkage.
W_NARROW = 5.10   # 0.78 * 6.5in
W_WIDE = 5.35     # 0.82 * 6.5in


def style(ax, axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if axis == "y":
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    else:
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Figure 1: where each of the 14 re-rankers sits, male-typed jobs against
# female-typed jobs. Tie-aware percentages (ties split evenly), cluster-
# bootstrap 95% CIs on both axes, vendor by marker shape, points numbered.
# ---------------------------------------------------------------------------

# Vendor families for the marker shapes (REVIEW_TODO B7). The three
# downloadable models are pooled as "open weights".
VENDOR_SHAPE = [
    ("Voyage AI", "o"),
    ("Cohere", "s"),
    ("Google Vertex AI", "^"),
    ("IBM", "D"),
    ("Open weights", "v"),
]


def _vendor_family(vendor_string):
    v = vendor_string.lower()
    if "open weights" in v:
        return "Open weights"
    if "voyage" in v:
        return "Voyage AI"
    if "cohere" in v:
        return "Cohere"
    if "google" in v:
        return "Google Vertex AI"
    if "ibm" in v:
        return "IBM"
    return "Open weights"


def _place_labels(xs, ys, xlim, ylim, ax_w_pt, ax_h_pt, radius=9.5):
    """Pick a non-colliding offset direction for each point's number label.

    The search runs in typographic points on the printed page, so the choice
    does not depend on the data units or the axis aspect. Greedy: for each
    point try eight compass directions and keep the one whose label lands
    farthest from every other point and from the labels already placed.
    """
    px = [(x - xlim[0]) / (xlim[1] - xlim[0]) * ax_w_pt for x in xs]
    py = [(y - ylim[0]) / (ylim[1] - ylim[0]) * ax_h_pt for y in ys]
    dirs = [(1, 0), (0.7, 0.7), (0, 1), (-0.7, 0.7),
            (-1, 0), (-0.7, -0.7), (0, -1), (0.7, -0.7)]
    placed, out = [], []
    for i in range(len(xs)):
        best, best_score = dirs[0], -1.0
        for dx, dy in dirs:
            qx, qy = px[i] + dx * radius, py[i] + dy * radius
            if not (4 < qx < ax_w_pt - 4 and 4 < qy < ax_h_pt - 4):
                continue
            others = [((qx - px[j]) ** 2 + (qy - py[j]) ** 2) ** 0.5
                      for j in range(len(xs)) if j != i]
            others += [((qx - rx) ** 2 + (qy - ry) ** 2) ** 0.5 for rx, ry in placed]
            score = min(others) if others else 1e9
            if score > best_score:
                best, best_score = (dx, dy), score
        placed.append((px[i] + best[0] * radius, py[i] + best[1] * radius))
        out.append(best)
    return out


def _short_name(name):
    """Display form of a model id: drop the hosting org, keep the product."""
    name = name.split("/")[-1]
    for prefix in ("Cohere-", "ibm-granite-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _pack(prefix, entries, width):
    """Wrap `entries` into lines of at most `width` characters, never splitting
    an entry (textwrap would break inside model ids at their hyphens)."""
    lines, cur = [], prefix
    for e in entries:
        candidate = cur + ("  " if cur.strip() else "") + e
        if len(candidate) > width and cur.strip() != prefix.strip():
            lines.append(cur)
            cur = " " * 4 + e
        else:
            cur = candidate
    lines.append(cur)
    return "\n".join(lines)


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
            "family": _vendor_family(m["vendor"]),
            # Tie-aware: ties count half a win each (REVIEW_TODO A2). Tie rates
            # reach 22.7% on one model, so plain "score gap > 0" would silently
            # credit every tie to the woman's document.
            "x": male["tie_aware_pct_male"],
            "xlo": male["tie_aware_pct_male_ci"][0],
            "xhi": male["tie_aware_pct_male_ci"][1],
            "y": fem["tie_aware_pct_male"],
            "ylo": fem["tie_aware_pct_male_ci"][0],
            "yhi": fem["tie_aware_pct_male_ci"][1],
        })

    # Same ordering as the appendix tables (analyze_single_stage.py sorts by the
    # tie-aware male-typed-jobs share, descending), so point 1..14 in the figure
    # is row 1..14 of Table A1/A2.
    rows.sort(key=lambda r: -r["x"])
    for i, r in enumerate(rows, start=1):
        r["n"] = i

    xlim, ylim = (40, 100), (0, 95)
    fig_h = 4.95
    fig = plt.figure(figsize=(W_NARROW, fig_h))
    ax_box = [0.135, 0.386, 0.845, 0.521]
    ax = fig.add_axes(ax_box)

    ax.axhline(50, color="#aaaaaa", lw=0.9, ls="--", zorder=1)
    ax.axvline(50, color="#aaaaaa", lw=0.9, ls="--", zorder=1)

    for r in rows:
        ax.errorbar(r["x"], r["y"],
                    xerr=[[r["x"] - r["xlo"]], [r["xhi"] - r["x"]]],
                    yerr=[[r["y"] - r["ylo"]], [r["yhi"] - r["y"]]],
                    fmt="none", ecolor="#9fb6c9", elinewidth=0.8,
                    capsize=1.6, capthick=0.8, zorder=2)
    for fam, marker in VENDOR_SHAPE:
        sel = [r for r in rows if r["family"] == fam]
        if not sel:
            continue
        ax.scatter([r["x"] for r in sel], [r["y"] for r in sel],
                   s=34, marker=marker, color=MALE, edgecolor="white",
                   linewidth=0.6, zorder=4)

    ax_w_pt = ax_box[2] * W_NARROW * 72
    ax_h_pt = ax_box[3] * fig_h * 72
    offsets = _place_labels([r["x"] for r in rows], [r["y"] for r in rows],
                            xlim, ylim, ax_w_pt, ax_h_pt)
    for r, (dx, dy) in zip(rows, offsets):
        ax.annotate(str(r["n"]), (r["x"], r["y"]), textcoords="offset points",
                    xytext=(dx * 9.5, dy * 9.5 - 2.2), fontsize=7.2,
                    color=INK, ha="center", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    ax.text(99, 93, "favors the man\nwhatever the job", ha="right", va="top",
            fontsize=7.0, color=MUTED, style="italic", linespacing=1.25)
    ax.text(99, 2, "follows the stereotype:\nthe man for male-typed jobs,\n"
                   "the woman for female-typed jobs",
            ha="right", va="bottom", fontsize=7.0, color=MUTED, style="italic",
            linespacing=1.25)
    ax.text(41, 2, "favors the woman\nwhatever the job", ha="left", va="bottom",
            fontsize=7.0, color=MUTED, style="italic", linespacing=1.25)

    ax.set_xlabel("Male-typed jobs: % of comparisons favoring the man's document",
                  fontsize=8.6)
    ax.set_ylabel("Female-typed jobs: % favoring\nthe man's document",
                  fontsize=8.6, linespacing=1.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8)
    fig.text(0.5, 0.960,
             "Which document each re-ranker favors (%d models)" % len(rows),
             ha="center", va="top", fontsize=10.5)
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.6)

    # One legend that doubles as the numbered key: the marker shape gives the
    # vendor, the numbers key each point to the appendix tables (whose rows are
    # in this same order).
    handles, labels = [], []
    for fam, marker in VENDOR_SHAPE:
        sel = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["n"])
        if not sel:
            continue
        handles.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=4.2, color=MALE,
                              markeredgecolor="white", markeredgewidth=0.5))
        labels.append(_pack(f"{fam}:",
                            [f"{r['n']} {_short_name(r['name'])}" for r in sel],
                            width=96))
    leg = fig.legend(handles, labels, loc="lower left",
                     bbox_to_anchor=(0.035, 0.012), frameon=False, fontsize=6.2,
                     handletextpad=0.7, labelspacing=0.7, borderpad=0.0)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.text(0.035, 0.303,
             "Each point is one re-ranker, numbered as in the appendix tables. "
             "Bars are 95% ranges\nthat allow for the same occupations recurring "
             "across comparisons. Ties are split evenly.",
             fontsize=6.4, color=MUTED, linespacing=1.4, va="top")

    fig.savefig(f"{FIGDIR}/fig_single_stage.pdf")
    plt.close(fig)
    print("wrote fig_single_stage.pdf  (%d models, 1=%s ... %d=%s)"
          % (len(rows), rows[0]["name"], len(rows), rows[-1]["name"]))
    return rows


# ---------------------------------------------------------------------------
# Figure 2: shortlist composition by stage and job category.
# ---------------------------------------------------------------------------
def fig_pipeline():
    d = _load("compounding_bge_pipeline.json")
    slate = d["slate"]
    k = d["headline_shortlist"]

    cats = ["male", "female", "neutral", "ALL"]
    catlabel = {"male": "Male-typed\njobs", "female": "Female-typed\njobs",
                "neutral": "Evenly split\njobs", "ALL": "All jobs"}
    stages = [(f"emb_top{k}_m", "Retriever only", "#9bbcd6"),
              (f"rr_top{k}_m", "Re-ranker only", "#5a8bbd"),
              (f"pipe_top{k}_m", "Full pipeline", "#23456b")]

    x = np.arange(len(cats))
    w = 0.26
    fig = plt.figure(figsize=(W_WIDE, 3.95))
    ax = fig.add_axes([0.155, 0.245, 0.700, 0.60])

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
                    ha="center", va="bottom", fontsize=7.0, color=INK)

    ax.axhline(50, color="#c0392b", lw=1.0, ls="--", zorder=2)
    ax.text(1.03, 50, "equal shares\n(50%)", transform=ax.get_yaxis_transform(),
            color="#c0392b", fontsize=7.0, ha="left", va="center", linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([catlabel[c] for c in cats], fontsize=8.6)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.set_ylabel("Share of the top-%d shortlist\ngoing to the man's name (%%)" % k,
                  fontsize=8.6, linespacing=1.4)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=8)
    fig.text(0.5, 0.975, "Who reaches the shortlist from a pool of equally\n"
             "qualified candidates", ha="center", va="top", fontsize=10.5,
             linespacing=1.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.0, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.075), handlelength=1.3,
               columnspacing=1.6)
    style(ax)

    fig.text(0.5, 0.028,
             "Bars are 95% ranges that allow for the same occupations recurring "
             "across simulated pools.",
             ha="center", fontsize=6.4, color=MUTED)

    fig.savefig(f"{FIGDIR}/fig_pipeline.pdf")
    plt.close(fig)
    print("wrote fig_pipeline.pdf  (top-%d %%male: " % k
          + ", ".join("%s search %.1f / model %.1f / pipeline %.1f"
                      % (c, slate[c][f"emb_top{k}_m"], slate[c][f"rr_top{k}_m"],
                         slate[c][f"pipe_top{k}_m"]) for c in cats) + ")")
    return {c: {s[0]: slate[c][s[0]] for s in stages} for c in cats}


# ---------------------------------------------------------------------------
# Figure 3: de-identification -- three-way split of every counterfactual pair.
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
        ("gram_pos", "Grammar-aware rewrite"),
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

    fig = plt.figure(figsize=(W_WIDE, 4.00))
    ax = fig.add_axes([0.305, 0.245, 0.675, 0.620])

    ypos = np.arange(len(order))[::-1]
    bar_h = 0.62
    for y, (key, _) in zip(ypos, order):
        # New schema: overall.pct_male / pct_tie / pct_female are PERCENTAGES
        # (0-100) and sum to 100. The legacy conditions.<c>.pct_male dict is a
        # per-category dict of FRACTIONS (0-1) and is deliberately not used.
        o = conds[key]["overall"]
        segs = [(o["pct_male"], MALE, "white"),
                (o["pct_tie"], TIE, INK),
                (o["pct_female"], FEMALE, "white")]
        left = 0.0
        for val, col, txtcol in segs:
            if val <= 0:
                continue
            ax.barh(y, val, bar_h, left=left, color=col, zorder=3,
                    edgecolor="white", linewidth=0.7)
            if val >= 7:
                ax.text(left + val / 2, y, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.4, color=txtcol, zorder=5)
            left += val
        lo, hi = o["pct_male_ci"]
        if hi - lo > 0.5:
            ax.plot([lo, hi], [y + bar_h / 2 + 0.16] * 2, color=MALE_DARK,
                    lw=0.9, zorder=6, solid_capstyle="butt")
            for e in (lo, hi):
                ax.plot([e, e], [y + bar_h / 2 + 0.09, y + bar_h / 2 + 0.23],
                        color=MALE_DARK, lw=0.9, zorder=6)

    # Drawn behind the bars, so it shows in the gaps between them and never
    # runs through a number.
    ax.axvline(50, color="#c0392b", lw=1.0, ls="--", zorder=2)
    ax.text(51.5, len(order) - 0.30, "50%", color="#c0392b", fontsize=6.8,
            ha="left", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in order], fontsize=8.0, linespacing=1.25)
    ax.set_ylim(-0.65, len(order) + 0.15)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of the %s counterfactual pairs (%%)"
                  % f"{d['n_triples']:,}", fontsize=8.6)
    ax.tick_params(axis="x", labelsize=8)
    fig.text(0.5, 0.965, "Who scores higher after each de-identification rule",
             ha="center", va="top", fontsize=10.5)
    style(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend_items = [
        Patch(facecolor=MALE, label="the man's document scored higher"),
        Patch(facecolor=TIE, label="exact tie"),
        Patch(facecolor=FEMALE, label="the woman's document scored higher"),
    ]
    fig.legend(handles=legend_items, frameon=False, fontsize=7.0,
               loc="lower center", bbox_to_anchor=(0.5, 0.088), ncol=3,
               handlelength=1.1, handleheight=0.9, columnspacing=1.3,
               handletextpad=0.5)

    fig.text(0.5, 0.052,
             "The last two rules leave the two documents word-for-word identical, "
             "so every pair must tie.\nWhiskers show the 95% range for the share "
             "favoring the man's document.",
             ha="center", va="top", fontsize=6.4, color=MUTED, linespacing=1.4)

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
