"""
Is the headline contrast an artefact of cutting the BLS share of women at 30/70?

The audit labels an occupation male-typed when the BLS share of women is below
30, female-typed above 70 and evenly split between 40 and 60, and drops the
30-40 and 60-70 bands as ambiguous. A reviewer called the cut-points a
researcher degree of freedom. This script treats the BLS share as the
continuous variable it is and reruns the headline under other cut-points, all
from the banked scores (no model calls):

    continuous      per model, over the 52 kept occupations: Spearman rho and
                    Kendall tau between an occupation's tie-aware male-favored
                    share and its share of women; a two-sided permutation
                    p-value for rho (the shares of women are permuted across
                    occupations, 100,000 permutations per model), Holm-adjusted
                    across the models; and the OLS slope of the share on the
                    share of women, in points of male-favored share per 10
                    points of women's share
    continuous_82   the same for the models scored on all 82 occupations,
                    adding every dropped occupation that has a BLS share of
                    women (the ambiguous-band ones; occupations with no BLS
                    row cannot be placed on the axis and stay out)
    cutoffs         the contrast (male-typed minus female-typed tie-aware
                    share) at 30/70, at 25/75 (within the kept occupations)
                    and, for the fully scored models, at 35/65 (which brings
                    ambiguous-band occupations back in), each with an
                    occupation-clustered percentile bootstrap interval
                    (1,000 draws), and the models whose sign changes

Every per-occupation share is recomputed from the raw files and checked
against results/single_stage_summary.json for the 52 kept occupations; the
30/70 contrast is checked against the summary's category shares.

Table A4 (results/tex/continuous_bls_check.tex) prints, per model, rho, its
Holm-adjusted permutation p and the slope over the 52 kept occupations; the
contrast at 30/70 and at 25/75, with the 25/75 interval; and the 35/65
contrast, which exists only for the models scored on all 82 occupation words
(an en dash for the others, as the note under the table says).

Output: results/continuous_bls_check.json, results/tex/continuous_bls_check.tex
Run:    venv/bin/python code/continuous_bls_check.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import SEED, stable_seed  # noqa: E402
from interval_sensitivity import holm  # noqa: E402
from labels import CSV_PATH, load_labels  # noqa: E402
from model_labels import MODEL_LABELS, model_label  # noqa: E402

SCRIPT_VERSION = "continuous_bls_check.py v1.0"
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "single_stage_summary.json"
OUT = RESULTS / "continuous_bls_check.json"
TEX = RESULTS / "tex" / "continuous_bls_check.tex"
TEX_ORDER_SRC = RESULTS / "tex" / "contrast_intervals.tex"
ALPHA = 0.05
# (male-typed below, female-typed above) in % women; "30/70" is the paper's rule
CUTOFFS = {"30/70": (30.0, 70.0), "25/75": (25.0, 75.0), "35/65": (35.0, 65.0)}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_bls():
    """occupation -> {pct_women (float or None), label, bls_title} for all 82 rows.

    bls_title contains commas, so the csv module, never a split on commas.
    """
    out = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pct = (r.get("pct_women") or "").strip()
            out[r["occupation"].strip()] = {
                "pct_women": float(pct) if pct else None,
                "label": (r.get("label") or "").strip().lower(),
                "bls_title": (r.get("bls_title") or "").strip()}
    return out


def occupation_counts(raw_rows):
    """occupation -> (tie-aware male half-wins / 2, pairs), from the raw rows.

    The tie-aware male-favored share is (wins_M + 0.5 * ties) / n, the
    statistic of analyze_single_stage.ClusterStat; `delta` is the raw file's
    score_male - score_female.
    """
    w, n = defaultdict(float), defaultdict(int)
    for r in raw_rows:
        d = r["delta"]
        w[r["occupation"]] += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
        n[r["occupation"]] += 1
    return {o: (w[o], n[o]) for o in sorted(n)}


# ---------------------------------------------------------------------------
# continuous analysis
# ---------------------------------------------------------------------------

def _pearson_rows(X, y):
    """Pearson correlation of every row of X with y."""
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    return (Xc @ yc) / (np.sqrt((Xc ** 2).sum(axis=1)) * np.sqrt((yc ** 2).sum()))


def continuous_stats(x, y, n_perm, rng):
    """Association between the share of women (x) and the male-favored share (y).

    Spearman rho is Pearson's r on average ranks (ties in x: occupations that
    share a BLS row), so the permutation distribution is built by permuting the
    x ranks across occupations, holding the y ranks fixed; p = (1 + #{|rho*| >=
    |rho_obs|}) / (1 + B), two-sided, the convention of interval_sensitivity.
    Kendall's tau is tau-b (scipy); its asymptotic p is kept for reference
    only. The slope is ordinary least squares of y on x, reported per 10
    points of women's share.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = sps.spearmanr(x, y)
    tau = sps.kendalltau(x, y)
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    obs = float(_pearson_rows(rx[None, :], ry)[0])
    if abs(obs - rho.statistic) > 1e-9:
        raise AssertionError("rank-Pearson does not reproduce scipy's Spearman rho")
    perm = rng.permuted(np.tile(rx, (n_perm, 1)), axis=1)      # each row shuffled
    stat = _pearson_rows(perm, ry)
    # 1e-12 so a permutation that reproduces the observed ranks is never lost
    # to floating-point summation order
    more = int((np.abs(stat) >= abs(obs) - 1e-12).sum())
    slope, intercept = np.polyfit(x, y, 1)
    return {"n_occupations": int(len(x)),
            "spearman_rho": float(rho.statistic),
            "perm_p_two_sided": (1 + more) / (1 + n_perm),
            "n_permutations_at_least_as_extreme": more,
            "kendall_tau": float(tau.statistic),
            "kendall_p_asymptotic": float(tau.pvalue),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "slope_per_10_points": float(10.0 * slope),
            "intercept": float(intercept)}


def continuous_section(per_model, occupation_set, n_occ):
    """Overview over the per-model continuous statistics, Holm-adjusted.

    Holm is applied across exactly the models passed in, so the input dicts are
    copied: the same per-model statistics can sit in a 29-model section and in
    a 14-model comparison with different adjusted p-values.
    """
    per_model = {m: dict(v) for m, v in per_model.items()}
    order = sorted(per_model)
    for m, p in zip(order, holm([per_model[m]["perm_p_two_sided"] for m in order])):
        per_model[m]["p_holm"] = float(p)
    rho = {m: v["spearman_rho"] for m, v in per_model.items()}

    def n_sig(sign, key):
        return sum(1 for m, v in per_model.items()
                   if v[key] < ALPHA and (rho[m] < 0 if sign == "negative" else rho[m] > 0))

    return {"occupation_set": occupation_set,
            "n_occupations": n_occ,
            "n_models": len(per_model),
            "alpha": ALPHA,
            "adjustment": f"Holm across the {len(per_model)} models",
            "n_models_rho_negative": sum(v < 0 for v in rho.values()),
            "n_models_tau_negative": sum(v["kendall_tau"] < 0 for v in per_model.values()),
            "n_models_slope_negative": sum(v["slope_per_10_points"] < 0
                                           for v in per_model.values()),
            # either direction, then split by the sign of rho
            "n_models_perm_p_below_alpha": sum(v["perm_p_two_sided"] < ALPHA
                                               for v in per_model.values()),
            "n_models_holm_below_alpha": sum(v["p_holm"] < ALPHA for v in per_model.values()),
            "n_models_negative_perm_significant": n_sig("negative", "perm_p_two_sided"),
            "n_models_negative_holm_significant": n_sig("negative", "p_holm"),
            "n_models_positive_perm_significant": n_sig("positive", "perm_p_two_sided"),
            "n_models_positive_holm_significant": n_sig("positive", "p_holm"),
            "models_positive_perm_significant": sorted(
                m for m, v in per_model.items() if v["perm_p_two_sided"] < ALPHA and rho[m] > 0),
            "median_rho": statistics.median(rho.values()),
            "min_rho": min(rho.values()), "max_rho": max(rho.values()),
            "median_tau": statistics.median(v["kendall_tau"] for v in per_model.values()),
            "median_slope_per_10_points": statistics.median(
                v["slope_per_10_points"] for v in per_model.values()),
            "min_slope_per_10_points": min(v["slope_per_10_points"] for v in per_model.values()),
            "max_slope_per_10_points": max(v["slope_per_10_points"] for v in per_model.values()),
            "models_rho_not_negative": sorted(m for m, v in rho.items() if v >= 0),
            "models_perm_p_not_below_alpha": sorted(
                m for m, v in per_model.items() if v["perm_p_two_sided"] >= ALPHA),
            "models_not_holm_significant": sorted(
                m for m, v in per_model.items() if v["p_holm"] >= ALPHA),
            "models": {m: per_model[m] for m in order}}


# ---------------------------------------------------------------------------
# cut-points
# ---------------------------------------------------------------------------

def assign(pct, lo, hi):
    """male-typed below `lo` % women, female-typed above `hi`, else neither."""
    if pct is None:
        return None
    return "male" if pct < lo else "female" if pct > hi else None


def cutoff_contrast(counts, groups, B, rng):
    """Pooled contrast and its occupation-clustered percentile bootstrap.

    `counts` maps occupation -> (tie-aware male half-wins, pairs); `groups`
    maps "male"/"female" -> occupations. The pooled share of a group is the
    ratio of sums over its occupations, so resampling occupations within each
    group with replacement and re-summing their counts is the cluster
    bootstrap of analyze_single_stage.ClusterStat, with B draws.
    """
    W = {c: np.array([counts[o][0] for o in groups[c]]) for c in groups}
    N = {c: np.array([counts[o][1] for o in groups[c]], dtype=float) for c in groups}
    share = {c: float(W[c].sum() / N[c].sum() * 100) for c in groups}
    draws = {}
    for c in ("male", "female"):
        idx = rng.integers(0, len(W[c]), size=(B, len(W[c])))
        draws[c] = W[c][idx].sum(1) / N[c][idx].sum(1) * 100
    lo, hi = np.percentile(draws["male"] - draws["female"], [2.5, 97.5])
    return {"contrast": share["male"] - share["female"],
            "tie_aware_pct_male": share,
            "ci": [float(lo), float(hi)],
            "above_zero": bool(lo > 0), "below_zero": bool(hi < 0)}


def cutoff_section(per_model, groups, occupation_set, lo, hi, bls):
    return {"occupation_set": occupation_set,
            "male_typed_below_pct_women": lo, "female_typed_above_pct_women": hi,
            "n_male": len(groups["male"]), "n_female": len(groups["female"]),
            "n_models": len(per_model),
            "occupations": {c: [{"occupation": o, "pct_women": bls[o]["pct_women"]}
                                for o in groups[c]] for c in ("male", "female")},
            "n_models_positive": sum(v["contrast"] > 0 for v in per_model.values()),
            "n_models_ci_above_zero": sum(v["above_zero"] for v in per_model.values()),
            "n_models_ci_below_zero": sum(v["below_zero"] for v in per_model.values()),
            "median_contrast": statistics.median(v["contrast"] for v in per_model.values()),
            "min_contrast": min(v["contrast"] for v in per_model.values()),
            "max_contrast": max(v["contrast"] for v in per_model.values()),
            "models_not_positive": sorted(m for m, v in per_model.items() if v["contrast"] <= 0),
            "models_ci_not_above_zero": sorted(m for m, v in per_model.items()
                                               if not v["above_zero"]),
            "models": {m: per_model[m] for m in sorted(per_model)}}


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _num(x, nd=1):
    """Fixed decimals with a true minus sign; a value rounding to zero is not -0."""
    t = f"{x:.{nd}f}"
    if float(t) == 0:
        t = f"{0.0:.{nd}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def _fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def _ci(v):
    lo, hi = v["ci"]
    return f"[{_num(lo)}, {_num(hi)}]"


def tex_order(models, contrast30):
    """Model order of results/tex/contrast_intervals.tex, else by 30/70 contrast."""
    inv = {v: k for k, v in MODEL_LABELS.items()}
    if TEX_ORDER_SRC.exists():
        order = []
        for line in TEX_ORDER_SRC.read_text().splitlines():
            if "&" not in line:
                continue
            label = (line.split("&")[0].strip()
                     .replace(r"\_", "_").replace(r"\&", "&").replace(r"\%", "%"))
            if label in inv and inv[label] in models:
                order.append(inv[label])
        if sorted(order) == sorted(models):
            return order, str(TEX_ORDER_SRC.relative_to(ROOT))
    return sorted(models, key=lambda m: -contrast30[m]), "descending 30/70 contrast"


def write_tex(order, cont, c30, c25, c35, n_perm, B, order_note, n_kept, n_words):
    """Table A4. The 35/65 contrast exists only for the models in `c35`, the
    fully scored ones; every other model gets an en dash, and the note under
    the tabular says why. The two-line headers keep the eight columns within
    the text width at the paper's small type."""
    n_other = len(order) - len(c35)
    lines = [f"% Generated by code/continuous_bls_check.py ({n_perm:,} permutations per "
             f"model for the Spearman p; B = {B:,} bootstrap draws per cut-point "
             f"interval) -- do not edit by hand.",
             f"% Row order: {order_note}.",
             r"\begin{tabular}{lrrrrrlr}", r"\toprule",
             r" & \multicolumn{3}{c}{Share vs.\ BLS \% women} & "
             r"\multicolumn{4}{c}{Contrast, cut-points} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-8}",
             r" & Spearman & perm.\ $p$ & slope / & & & 25/75 & \\",
             r"Model & $\rho$ & (Holm) & 10 pts & 30/70 & 25/75 & 95\% interval & "
             r"35/65$^{\dagger}$ \\",
             r"\midrule"]
    for m in order:
        v = cont[m]
        lines.append(f"{_esc(model_label(m))} & {_num(v['spearman_rho'], 2)} & "
                     f"{_fmt_p(v['p_holm'])} & {_num(v['slope_per_10_points'])} & "
                     f"{_num(c30[m]['contrast'])} & {_num(c25[m]['contrast'])} & "
                     f"{_ci(c25[m])} & {_num(c35[m]['contrast']) if m in c35 else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip{\footnotesize\raggedright Every column but the last uses only "
              f"the {n_kept} retained occupations; the 25/75 interval is an "
              f"occupation-clustered percentile bootstrap ({B:,} draws). "
              r"$^{\dagger}$Only for the " f"{len(c35)} models scored on all {n_words} "
              f"occupation words (-- for the other {n_other}): the 35/65 cut-points bring in "
              r"occupations from the excluded bands, which the other models were not scored "
              r"on.\par}"]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Continuous and alternative-cut-point checks "
                                             "of the BLS labelling")
    ap.add_argument("--permutations", type=int, default=100_000,
                    help="permutations per model for the Spearman p-value")
    ap.add_argument("--bootstrap", type=int, default=1_000,
                    help="occupation-clustered bootstrap draws per cut-point contrast")
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()

    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    bls = load_bls()
    summary = json.loads(SUMMARY.read_text())
    kept = sorted(labels)                                   # the 52 the paper uses
    if any(bls[o]["pct_women"] is None for o in kept):
        raise ValueError("a kept occupation has no BLS share of women")
    with_pct = sorted(o for o, v in bls.items() if v["pct_women"] is not None)
    added = sorted(set(with_pct) - set(kept))               # ambiguous bands, by construction
    no_pct = sorted(o for o, v in bls.items() if v["pct_women"] is None)
    if any(assign(bls[o]["pct_women"], *CUTOFFS["30/70"]) or
           40 <= bls[o]["pct_women"] <= 60 for o in added):
        raise AssertionError("an added occupation is not in the 30-40 / 60-70 bands")

    # per-occupation counts from the raw files, checked against the summary
    counts, full = {}, []
    for model, meta in sorted(summary["models"].items()):
        raw = json.loads((RESULTS / meta["raw_file"]).read_text())["results"]
        counts[model] = occupation_counts(raw)
        occs = set(counts[model])
        if not occs >= set(kept):
            raise ValueError(f"{model}: raw file lacks a kept occupation")
        if occs >= set(with_pct):
            full.append(model)
        elif occs != set(kept):
            raise ValueError(f"{model}: raw file covers neither the 52 nor all 82 occupations")
        for cat, block in meta["occupation_consistency"].items():
            for o, v in block["per_occupation"].items():
                w, n = counts[model][o]
                if abs(w / n * 100 - v["tie_aware_pct_male"]) > 1e-9:
                    raise AssertionError(f"{model}/{o}: share disagrees with the summary")
    if len(full) != 14:
        raise AssertionError(f"expected 14 fully scored models, found {len(full)}")
    if any(set(counts[m]) != set(bls) for m in full):     # the table note says "all 82"
        raise AssertionError("a fully scored model lacks one of the occupation words")
    pct = {o: bls[o]["pct_women"] for o in with_pct}

    def share(model, o):
        w, n = counts[model][o]
        return w / n * 100

    # ---- A: continuous, 52 kept occupations, every model ----------------------
    cont = {}
    for model in counts:
        rng = np.random.default_rng(stable_seed(model, "continuous", "kept_52"))
        cont[model] = continuous_stats([pct[o] for o in kept], [share(model, o) for o in kept],
                                       args.permutations, rng)
    continuous = continuous_section(cont, "kept_52", len(kept))
    continuous["definition"] = (
        "per model: Spearman rho, Kendall tau-b and the OLS slope between an "
        "occupation's tie-aware % male-favored share and its BLS % women, over the "
        "52 kept occupations; the permutation p permutes % women across occupations")
    continuous["n_permutations"] = args.permutations
    continuous["min_attainable_p"] = 1 / (1 + args.permutations)
    # one more view: the median share across models, per occupation
    med = [statistics.median(share(m, o) for m in counts) for o in kept]
    continuous["across_model_median_share"] = continuous_stats(
        [pct[o] for o in kept], med, args.permutations,
        np.random.default_rng(stable_seed("across_model_median", "continuous", "kept_52")))

    # ---- B: the fully scored models, adding the ambiguous-band occupations -----
    cont82 = {}
    for model in full:
        rng = np.random.default_rng(stable_seed(model, "continuous", "all_with_pct_women"))
        cont82[model] = continuous_stats([pct[o] for o in with_pct],
                                         [share(model, o) for o in with_pct],
                                         args.permutations, rng)
        cont82[model]["rho_on_kept_52"] = cont[model]["spearman_rho"]
        cont82[model]["rho_change"] = cont82[model]["spearman_rho"] - cont[model]["spearman_rho"]
        cont82[model]["slope_on_kept_52_per_10_points"] = cont[model]["slope_per_10_points"]
    continuous_82 = continuous_section(cont82, "all_with_pct_women", len(with_pct))
    continuous_82["definition"] = (
        "as `continuous`, for the models scored on all 82 occupations, over every "
        "occupation with a BLS % women: the 52 kept plus the dropped ones in the "
        "ambiguous 30-40 / 60-70 bands")
    continuous_82["n_permutations"] = args.permutations
    continuous_82["n_occupations_added"] = len(added)
    continuous_82["added_occupations"] = [
        {"occupation": o, "pct_women": pct[o], "bls_title": bls[o]["bls_title"]} for o in added]
    continuous_82["n_occupations_without_pct_women"] = len(no_pct)
    continuous_82["occupations_without_pct_women"] = no_pct
    continuous_82["models_fully_scored"] = full
    sub = {m: cont[m] for m in full}
    continuous_82["same_models_on_kept_52"] = {
        k: v for k, v in continuous_section(dict(sub), "kept_52", len(kept)).items()
        if k.startswith(("n_models", "median", "min", "max"))}
    continuous_82["median_rho_change"] = statistics.median(
        v["rho_change"] for v in cont82.values())
    continuous_82["max_abs_rho_change"] = max(abs(v["rho_change"]) for v in cont82.values())
    continuous_82["n_models_rho_moved_toward_zero"] = sum(
        abs(v["spearman_rho"]) < abs(v["rho_on_kept_52"]) for v in cont82.values())

    # ---- C: cut-points ---------------------------------------------------------
    schemes = {}
    for name, (lo, hi) in CUTOFFS.items():
        occ_set, universe, models = ("kept_52", kept, list(counts))
        if name == "35/65":                 # needs the ambiguous bands -> fully scored only
            occ_set, universe, models = ("all_with_pct_women", with_pct, full)
        groups = {c: [o for o in universe if assign(pct[o], lo, hi) == c]
                  for c in ("male", "female")}
        per = {}
        for model in models:
            rng = np.random.default_rng(stable_seed(model, "cutoff", name))
            per[model] = cutoff_contrast(counts[model], groups, args.bootstrap, rng)
        schemes[name] = cutoff_section(per, groups, occ_set, lo, hi, bls)
    # the 30/70 contrast must be the paper's headline
    for model, v in schemes["30/70"]["models"].items():
        cat = summary["models"][model]["by_category"]
        if abs(v["contrast"] - (cat["male"]["tie_aware_pct_male"]
                                - cat["female"]["tie_aware_pct_male"])) > 1e-9:
            raise AssertionError(f"{model}: 30/70 contrast disagrees with the summary")
    base = {c: {o for o in kept if labels[o] == c} for c in ("male", "female")}
    for name in ("25/75", "35/65"):
        g = {c: {d["occupation"] for d in schemes[name]["occupations"][c]}
             for c in ("male", "female")}
        schemes[name]["removed_relative_to_30_70"] = {c: sorted(base[c] - g[c]) for c in g}
        schemes[name]["added_relative_to_30_70"] = {c: sorted(g[c] - base[c]) for c in g}
        schemes[name]["change_in_contrast_vs_30_70"] = {
            m: v["contrast"] - schemes["30/70"]["models"][m]["contrast"]
            for m, v in schemes[name]["models"].items()}
        schemes[name]["median_abs_change_vs_30_70"] = statistics.median(
            abs(x) for x in schemes[name]["change_in_contrast_vs_30_70"].values())
        schemes[name]["max_abs_change_vs_30_70"] = max(
            abs(x) for x in schemes[name]["change_in_contrast_vs_30_70"].values())
        schemes[name]["sign_flips_vs_30_70"] = sorted(
            m for m, v in schemes[name]["models"].items()
            if (v["contrast"] > 0) != (schemes["30/70"]["models"][m]["contrast"] > 0))
    s3070 = schemes["30/70"]["models"]
    schemes["35/65"]["same_models_at_30_70"] = {
        "n_models_positive": sum(s3070[m]["contrast"] > 0 for m in full),
        "n_models_ci_above_zero": sum(s3070[m]["above_zero"] for m in full),
        "median_contrast": statistics.median(s3070[m]["contrast"] for m in full)}
    s2575 = schemes["25/75"]["models"]
    schemes["35/65"]["same_models_at_25_75"] = {
        "n_models_positive": sum(s2575[m]["contrast"] > 0 for m in full),
        "n_models_ci_above_zero": sum(s2575[m]["above_zero"] for m in full),
        "median_contrast": statistics.median(s2575[m]["contrast"] for m in full)}
    cutoffs = {
        "definition": "tie-aware % male-favored on male-typed minus female-typed "
                      "occupations, with male-typed = BLS % women below the first "
                      "cut-point and female-typed = above the second; 95% percentile "
                      "interval from an occupation-clustered bootstrap within each group",
        "bootstrap_B": args.bootstrap,
        "n_models_positive_at_30_70_and_25_75": sum(
            s3070[m]["contrast"] > 0 and s2575[m]["contrast"] > 0 for m in counts),
        "n_models_ci_above_zero_at_30_70_and_25_75": sum(
            s3070[m]["above_zero"] and s2575[m]["above_zero"] for m in counts),
        "n_fully_scored_models_positive_at_all_three": sum(
            all(schemes[s]["models"][m]["contrast"] > 0 for s in CUTOFFS) for m in full),
        "sign_flips": {f"{s} vs 30/70": schemes[s]["sign_flips_vs_30_70"]
                       for s in ("25/75", "35/65")},
        "schemes": schemes}

    out = {"script_version": SCRIPT_VERSION,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "label_source": label_source,
           "seed": SEED,
           "seed_note": "every permutation and bootstrap stream is "
                        "np.random.default_rng(stable_seed(model, purpose, ...)) with "
                        "analyze_single_stage.stable_seed, derived from `seed`",
           "n_permutations": args.permutations,
           "bootstrap_B": args.bootstrap,
           "alpha": ALPHA,
           "definition": "occupation-level tie-aware % male-favored share "
                         "((wins_M + 0.5 * ties) / pairs) against the BLS CPS Table 11 "
                         "% women, continuously and under alternative cut-points",
           "inputs": ["results/single_stage_summary.json", "results/*_full_raw.json",
                      str(CSV_PATH.relative_to(ROOT))],
           "n_models": len(counts),
           "n_models_fully_scored": len(full),
           "occupation_sets": {
               "kept_52": {"n": len(kept),
                           "label_counts": {c: sum(1 for o in kept if labels[o] == c)
                                            for c in ("male", "female", "neutral")}},
               "all_with_pct_women": {"n": len(with_pct), "n_added": len(added),
                                      "added": added, "n_without_pct_women": len(no_pct)}},
           "continuous": continuous,
           "continuous_82": continuous_82,
           "cutoffs": cutoffs}
    order, order_note = tex_order(list(counts), {m: v["contrast"] for m, v in s3070.items()})
    out["tex_row_order"] = order_note
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    cont, cont82 = continuous["models"], continuous_82["models"]     # with the Holm p's
    write_tex(order, cont, s3070, s2575, schemes["35/65"]["models"], args.permutations,
              args.bootstrap, order_note, len(kept), len(bls))

    # ---- console -------------------------------------------------------------
    print(f"{'model':<44}{'rho':>7}{'p':>8}{'Holm':>8}{'tau':>7}{'slope/10':>10}"
          f"{'c30/70':>8}{'c25/75':>8}{'c35/65':>8}{'rho63':>7}")
    for m in order:
        v = cont[m]
        c35 = f"{schemes['35/65']['models'][m]['contrast']:8.1f}" if m in full else f"{'--':>8}"
        r63 = f"{cont82[m]['spearman_rho']:7.2f}" if m in full else f"{'--':>7}"
        print(f"{m[:43]:<44}{v['spearman_rho']:7.2f}{v['perm_p_two_sided']:8.4f}"
              f"{v['p_holm']:8.4f}{v['kendall_tau']:7.2f}{v['slope_per_10_points']:10.1f}"
              f"{s3070[m]['contrast']:8.1f}{s2575[m]['contrast']:8.1f}{c35}{r63}")
    c = continuous
    print(f"\ncontinuous (52 occ., {c['n_models']} models): rho<0 {c['n_models_rho_negative']}, "
          f"perm p<{ALPHA} {c['n_models_perm_p_below_alpha']} "
          f"({c['n_models_negative_perm_significant']} negative, "
          f"{c['n_models_positive_perm_significant']} positive), Holm<{ALPHA} "
          f"{c['n_models_holm_below_alpha']} ({c['n_models_negative_holm_significant']} "
          f"negative), median rho {c['median_rho']:.2f} "
          f"[{c['min_rho']:.2f}, {c['max_rho']:.2f}], median tau {c['median_tau']:.2f}, "
          f"median slope/10 {c['median_slope_per_10_points']:.1f}; "
          f"not Holm-significant: {c['models_not_holm_significant']}")
    c = continuous_82
    k = c["same_models_on_kept_52"]
    print(f"continuous_82 ({c['n_occupations']} occ. = 52 + {c['n_occupations_added']} added, "
          f"{c['n_models']} models): rho<0 {c['n_models_rho_negative']} (52: "
          f"{k['n_models_rho_negative']}), perm p<{ALPHA} {c['n_models_perm_p_below_alpha']} "
          f"(52: {k['n_models_perm_p_below_alpha']}), Holm<{ALPHA} "
          f"{c['n_models_holm_below_alpha']} (52: {k['n_models_holm_below_alpha']}), median rho "
          f"{c['median_rho']:.2f} (52: {k['median_rho']:.2f}), median change "
          f"{c['median_rho_change']:+.3f}, max |change| {c['max_abs_rho_change']:.3f}")
    print("  added: " + ", ".join(f"{d['occupation']} ({d['pct_women']})"
                                  for d in c["added_occupations"]))
    for name, s in schemes.items():
        print(f"cutoff {name} ({s['occupation_set']}, {s['n_male']} M / {s['n_female']} F, "
              f"{s['n_models']} models): positive {s['n_models_positive']}, CI>0 "
              f"{s['n_models_ci_above_zero']}, CI<0 {s['n_models_ci_below_zero']}, median "
              f"{s['median_contrast']:.1f} [{s['min_contrast']:.1f}, {s['max_contrast']:.1f}]; "
              f"not positive: {s['models_not_positive']}")
        if "added_relative_to_30_70" in s:
            print(f"  vs 30/70: added {s['added_relative_to_30_70']}, removed "
                  f"{s['removed_relative_to_30_70']}, sign flips {s['sign_flips_vs_30_70']}, "
                  f"median |change| {s['median_abs_change_vs_30_70']:.1f}, max "
                  f"{s['max_abs_change_vs_30_70']:.1f}")
    print(f"Saved -> {OUT}\nSaved -> {TEX}  (row order: {order_note})")


if __name__ == "__main__":
    main()
