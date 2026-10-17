"""
How much do the single-stage intervals depend on what they resample?

The audit's intervals resample occupations and hold the names, templates and
query phrasings fixed, so they describe uncertainty over occupations for this
benchmark's inputs (reviewer item 5). This script recomputes each model's
headline contrast -- tie-aware male-favored share on male-typed occupations
minus the same on female-typed occupations -- under three resampling schemes:

    occupation     occupations, within category (the paper's scheme)
    bls_category   BLS CPS categories, within category; occupations that share a
                   BLS row (e.g. accountant / auditor) move together
    occupation_x_name
                   occupations within category AND the ten name pairs,
                   independently (a two-way cluster bootstrap), so the interval
                   also covers uncertainty over which names were chosen

v1.2 adds robustness statistics, each computed from the same banked scores and
each drawing from its own seeded random stream, so every value v1.1 wrote is
unchanged (the new blocks follow `models` in the JSON):

    neutral_level      the evenly split occupations' tie-aware % male-favored,
                       with 95% intervals resampling occupations, and
                       occupations x name pairs jointly; how many models'
                       intervals contain 50 (closed interval, the rule of
                       analyze_single_stage.count_neutral_indeterminate).
                       B = 1,000,000 here (--neutral-bootstrap): see main()
    label_permutation  two-sided permutation test of the contrast: the labels
                       of the 39 male- or female-typed occupations are shuffled
                       with the group sizes (18 / 21) fixed (1,000,000 seeded
                       permutations per model, --permutations); Holm-adjusted
                       across the models. Also the last column of Table A2.
    full_name_cross    the contrast when, within each (occupation, template,
                       query) cell, every male-associated document is compared
                       with every female-associated one (10 x 10), not only
                       with its own pair
    within_template,   the contrast within each document template and within
    within_query       each query phrasing

Output: results/interval_sensitivity.json, results/tex/contrast_intervals.tex
Run:    venv/bin/python code/interval_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import stable_seed  # noqa: E402
from labels import CSV_PATH, load_labels  # noqa: E402

SCRIPT_VERSION = "interval_sensitivity.py v1.2"
RESULTS = ROOT / "results"
OUT = RESULTS / "interval_sensitivity.json"
TEX = RESULTS / "tex" / "contrast_intervals.tex"
SCHEMES = ("occupation", "bls_category", "occupation_x_name")
NEUTRAL_SCHEMES = ("occupation", "occupation_x_name")
ALPHA = 0.05


def cell_tables(raw_rows, occs, names):
    """Tie-aware male wins and counts per (occupation, name pair)."""
    oi = {o: i for i, o in enumerate(occs)}
    ni = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(occs), len(names)))
    N = np.zeros((len(occs), len(names)))
    for r in raw_rows:
        if r["occupation"] not in oi:
            continue
        d = r["score_male"] - r["score_female"]
        o, n = oi[r["occupation"]], ni[r["name_pair"]]
        W[o, n] += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
        N[o, n] += 1
    return W, N


def contrast_draws(W, N, cat, clusters, B, rng, resample_names, chunk=10_000):
    """B bootstrap values of (male-typed share - female-typed share), in points.

    `clusters` maps each occupation index to a cluster id; clusters are
    resampled with replacement within each category. Vectorised: drawing k
    clusters with replacement is a multinomial(k, uniform) count vector, so a
    whole chunk of draws is a matrix product rather than a Python loop. In the
    two-way scheme one set of name-pair weights is shared by both categories
    within a draw.
    """
    n_occ, n_names = W.shape
    out = np.empty(B)
    by_cat, expand = {}, {}
    for c in ("male", "female"):
        ids = sorted({clusters[i] for i in range(n_occ) if cat[i] == c})
        by_cat[c] = ids
        # occupation x cluster indicator: cluster counts -> occupation weights
        E = np.zeros((len(ids), n_occ))
        pos = {k: j for j, k in enumerate(ids)}
        for i in range(n_occ):
            if cat[i] == c:
                E[pos[clusters[i]], i] = 1.0
        expand[c] = E
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        w_name = (rng.multinomial(n_names, np.full(n_names, 1 / n_names), size=b)
                  if resample_names else np.ones((b, n_names)))
        share = {}
        for c in ("male", "female"):
            k = len(by_cat[c])
            a = rng.multinomial(k, np.full(k, 1 / k), size=b) @ expand[c]   # (b, n_occ)
            share[c] = (np.einsum("bo,on,bn->b", a, W, w_name)
                        / np.einsum("bo,on,bn->b", a, N, w_name) * 100)
        out[start:start + b] = share["male"] - share["female"]
    return out


# ---------------------------------------------------------------------------
# v1.2 robustness statistics. Each takes its own seeded generator, so none of
# them consumes a draw from the streams above.
# ---------------------------------------------------------------------------

def level_draws(W, N, B, rng, resample_names, chunk=10_000):
    """B bootstrap values of one category's tie-aware % male-favored.

    The category's occupations are resampled with replacement and, in the
    two-way scheme, the name pairs too, independently -- the same multinomial
    construction as `contrast_draws`, for a single category.
    """
    n_occ, n_names = W.shape
    out = np.empty(B)
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        w_name = (rng.multinomial(n_names, np.full(n_names, 1 / n_names), size=b)
                  if resample_names else np.ones((b, n_names)))
        a = rng.multinomial(n_occ, np.full(n_occ, 1 / n_occ), size=b)
        out[start:start + b] = (np.einsum("bo,on,bn->b", a, W, w_name)
                                / np.einsum("bo,on,bn->b", a, N, w_name) * 100)
    return out


def neutral_level(model, W, N, B):
    """Evenly split occupations: point share and its 95% intervals.

    An interval "contains 50" when lo <= 50 <= hi -- closed, as in
    analyze_single_stage.count_neutral_indeterminate: a bound landing exactly
    on 50 has not excluded it.
    """
    out = {"tie_aware_pct_male": float(W.sum() / N.sum() * 100), "schemes": {}}
    for scheme in NEUTRAL_SCHEMES:
        rng = np.random.default_rng(stable_seed(model, scheme, "neutral_level"))
        lo, hi = np.percentile(level_draws(W, N, B, rng, scheme == "occupation_x_name"),
                               [2.5, 97.5])
        out["schemes"][scheme] = {"ci": [float(lo), float(hi)],
                                  "contains_50": bool(lo <= 50.0 <= hi),
                                  "above_50": bool(lo > 50.0), "below_50": bool(hi < 50.0)}
    return out


def label_permutation_test(W, N, is_male, B, rng, chunk=20_000):
    """Two-sided occupation-label permutation test of the contrast.

    Null: an occupation's stereotype label is unrelated to how the model treats
    it, so the labels of the male- and female-typed occupations are
    exchangeable. Each permutation gives the male-typed label to a uniformly
    random set of as many occupations as carry it now (the k_m smallest of k
    i.i.d. uniform keys), so the group sizes stay fixed, and recomputes the
    contrast.

    With n pairs in every occupation the pooled contrast is the difference of
    the two groups' mean per-occupation shares, i.e. (k*S - k_m*T) /
    (k_m*k_f*2n) * 100 points, where S is the male-typed group's total of
    tie-aware half-wins (2*wins + ties, an integer per occupation) and T the
    total over all k occupations. The test compares the integer k*S - k_m*T, so
    a permuted statistic equal to the observed one is never lost to rounding.
    p = (1 + #{|stat*| >= |stat_obs|}) / (1 + B) (two-sided); one-sided p's use
    stat* >= stat_obs (upper) and stat* <= stat_obs (lower).
    """
    halves = 2.0 * W.sum(axis=1)
    h = np.rint(halves).astype(np.int64)
    if not np.array_equal(h, halves):
        raise ValueError("tie-aware half-wins must be whole numbers")
    pairs = N.sum(axis=1)
    if not np.all(pairs == pairs[0]):
        raise ValueError("occupations carry different numbers of pairs, so the "
                         "integer statistic does not equal the pooled contrast")
    k, k_m = len(h), int(is_male.sum())
    k_f, total = k - k_m, int(h.sum())
    obs = k * int(h[is_male].sum()) - k_m * total
    more2 = upper = lower = 0
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        pick = np.argpartition(rng.random((b, k)), k_m - 1, axis=1)[:, :k_m]
        stat = k * h[pick].sum(axis=1) - k_m * total
        more2 += int((np.abs(stat) >= abs(obs)).sum())
        upper += int((stat >= obs).sum())
        lower += int((stat <= obs).sum())
    return {"contrast_from_statistic": obs / (k_m * k_f * 2 * float(pairs[0])) * 100,
            "pairs_per_occupation": int(pairs[0]),
            "n_permutations_at_least_as_extreme": more2,
            "p_two_sided": (1 + more2) / (1 + B),
            "p_one_sided_upper": (1 + upper) / (1 + B),
            "p_one_sided_lower": (1 + lower) / (1 + B)}


def holm(p):
    """Holm (1979) step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def full_name_cross(raw_rows, labels, names):
    """Contrast over every male-associated x female-associated document in a cell.

    A cell is one (occupation, template, query phrasing); it holds one pair per
    name pair. Instead of comparing each male-associated document only with its
    own pair's female-associated document, compare it with all of them
    (len(names)^2 comparisons per cell), ties counting half. Also returns the
    contrast from the diagonal alone (each document with its own pair), which
    must equal the paired contrast.
    """
    cells = {}
    for r in raw_rows:
        c = labels.get(r["occupation"])
        if c not in ("male", "female"):
            continue
        cell = cells.setdefault((c, r["occupation"], r["template_style"], r["query_type"]), {})
        if r["name_pair"] in cell:
            raise ValueError(f"two rows for one name pair in cell {r['occupation']!r}, "
                             f"{r['template_style']!r}, {r['query_type']!r}")
        cell[r["name_pair"]] = (r["score_male"], r["score_female"])
    wins = {"male": 0.0, "female": 0.0}
    diag = {"male": 0.0, "female": 0.0}
    n_cells = {"male": 0, "female": 0}
    for (c, *_), cell in cells.items():
        if sorted(cell) != list(names):
            raise ValueError("every cell must hold exactly one pair per name pair")
        m = np.array([cell[x][0] for x in names])
        f = np.array([cell[x][1] for x in names])
        gt, eq = m[:, None] > f[None, :], m[:, None] == f[None, :]
        wins[c] += gt.sum() + 0.5 * eq.sum()
        diag[c] += np.trace(gt) + 0.5 * np.trace(eq)
        n_cells[c] += 1
    k = len(names)
    share = {c: float(wins[c] / (n_cells[c] * k * k) * 100) for c in wins}
    paired = {c: float(diag[c] / (n_cells[c] * k) * 100) for c in diag}
    return {"contrast": share["male"] - share["female"],
            "tie_aware_pct_male": share,
            "n_cells": n_cells,
            "comparisons_per_cell": k * k,
            "diagonal_contrast": paired["male"] - paired["female"]}


def stratum_contrasts(raw_rows, labels, field):
    """The contrast within each level of `field` (template_style or query_type)."""
    wins, n = defaultdict(float), defaultdict(int)
    for r in raw_rows:
        c = labels.get(r["occupation"])
        if c not in ("male", "female"):
            continue
        d = r["score_male"] - r["score_female"]
        wins[r[field], c] += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
        n[r[field], c] += 1
    return {lv: (wins[lv, "male"] / n[lv, "male"] - wins[lv, "female"] / n[lv, "female"]) * 100
            for lv in sorted({key[0] for key in n})}


def stratum_section(per_model, field, description):
    levels = sorted({lv for v in per_model.values() for lv in v})
    if any(sorted(v) != levels for v in per_model.values()):
        raise ValueError(f"every model must cover the same {field} levels")
    positive_all = sorted(m for m, v in per_model.items() if all(x > 0 for x in v.values()))
    return {"definition": description, "field": field, "levels": levels,
            "n_models_positive_in_all_levels": len(positive_all),
            "models_not_positive_in_all_levels": sorted(set(per_model) - set(positive_all)),
            "n_model_level_cells": sum(len(v) for v in per_model.values()),
            "n_model_level_cells_positive": sum(x > 0 for v in per_model.values()
                                                for x in v.values()),
            "models": per_model}


def fmt_p(p):
    """Compact p for the table: '<0.001', else three decimals below 0.1, else two."""
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def write_tex(models, B, path, p_holm, n_perm):
    """Booktabs tabular, one row per model, sorted by contrast (descending).

    The last column is the two-sided occupation-label permutation p-value,
    Holm-adjusted across the models (`label_permutation` in the JSON).
    """
    from model_labels import model_label

    def esc(s):
        return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    def num(x):
        # One decimal with a true minus sign, matching the de-identification
        # tables; a value that rounds to zero prints as 0.0, never -0.0.
        t = f"{x:.1f}"
        if t == "-0.0":
            t = "0.0"
        return "$-$" + t[1:] if t.startswith("-") else t

    def ci(v):
        lo, hi = v["ci"]
        return f"[{num(lo)}, {num(hi)}]"

    lines = [f"% Generated by code/interval_sensitivity.py (B = {B:,} draws per scheme; "
             f"{n_perm:,} label permutations per model) -- do not edit by hand.",
             r"\begin{tabular}{lrlllr}", r"\toprule",
             r" & & \multicolumn{3}{c}{95\% interval, resampling} & Permutation \\",
             r"\cmidrule(l){3-5}",
             r"Model & Contrast & Occupation & BLS category & Occ.\,$\times$\,name & $p$ (Holm) \\",
             r"\midrule"]
    for m, v in sorted(models.items(), key=lambda kv: -kv[1]["contrast"]):
        s = v["schemes"]
        lines.append(f"{esc(model_label(m))} & {num(v['contrast'])} & "
                     + " & ".join(ci(s[k]) for k in SCHEMES)
                     + f" & {fmt_p(p_holm[m])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Interval sensitivity to the resampling unit")
    ap.add_argument("--bootstrap", type=int, default=50_000)
    # One model's evenly-split interval ends within 0.1 point of 50 (rerank-2.5-lite,
    # occupations x names: lower end 49.93-49.95 at B = 2,000,000). At B = 50,000
    # Monte Carlo error alone puts that end on either side of 50 depending on the
    # seed, so these intervals use a B at which no model's "contains 50"
    # classification depends on the seed.
    ap.add_argument("--neutral-bootstrap", type=int, default=1_000_000,
                    help="bootstrap draws for the evenly split occupations' intervals")
    ap.add_argument("--permutations", type=int, default=1_000_000,
                    help="label permutations per model for the permutation test")
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()
    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        bls = {r["occupation"]: r["bls_title"] for r in csv.DictReader(fh)}

    summary = json.loads((RESULTS / "single_stage_summary.json").read_text())
    occs = sorted(o for o, c in labels.items() if c in ("male", "female"))
    cat = [labels[o] for o in occs]
    is_male = np.array([c == "male" for c in cat])
    neutral_occs = sorted(o for o, c in labels.items() if c == "neutral")
    neutral, perm, cross = {}, {}, {}
    strata = {"template_style": {}, "query_type": {}}
    cluster_of = {"occupation": {i: o for i, o in enumerate(occs)},
                  "bls_category": {i: bls[o] for i, o in enumerate(occs)},
                  "occupation_x_name": {i: o for i, o in enumerate(occs)}}

    models = {}
    for model, meta in summary["models"].items():
        raw = json.loads((RESULTS / meta["raw_file"]).read_text())["results"]
        names = sorted({r["name_pair"] for r in raw})
        W, N = cell_tables(raw, occs, names)
        point = {c: W[[i for i in range(len(occs)) if cat[i] == c]].sum()
                 / N[[i for i in range(len(occs)) if cat[i] == c]].sum() * 100
                 for c in ("male", "female")}
        res = {"contrast": point["male"] - point["female"], "schemes": {}}
        for scheme in SCHEMES:
            rng = np.random.default_rng(stable_seed(model, scheme, "contrast"))
            draws = contrast_draws(W, N, cat, cluster_of[scheme], args.bootstrap, rng,
                                   resample_names=scheme == "occupation_x_name")
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res["schemes"][scheme] = {"ci": [float(lo), float(hi)],
                                      "above_zero": bool(lo > 0), "below_zero": bool(hi < 0),
                                      "excludes_zero": bool(lo > 0 or hi < 0)}
        models[model] = res
        print(f"{model:<52} {res['contrast']:6.1f}  "
              + "  ".join(f"{s}: [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]"
                          for s, v in res["schemes"].items()), flush=True)

        # ---- v1.2 additions (own seeded streams; nothing above is touched) ----
        Wn, Nn = cell_tables(raw, neutral_occs, names)
        neutral[model] = neutral_level(model, Wn, Nn, args.neutral_bootstrap)
        ss_neutral = meta["by_category"]["neutral"]["tie_aware_pct_male"]
        if abs(neutral[model]["tie_aware_pct_male"] - ss_neutral) > 1e-9:
            raise AssertionError(f"{model}: neutral share disagrees with single_stage_summary")
        perm[model] = label_permutation_test(
            W, N, is_male, args.permutations,
            np.random.default_rng(stable_seed(model, "label_permutation")))
        if abs(perm[model]["contrast_from_statistic"] - res["contrast"]) > 1e-9:
            raise AssertionError(f"{model}: permutation statistic is not the pooled contrast")
        cross[model] = full_name_cross(raw, labels, names)
        if abs(cross[model]["diagonal_contrast"] - res["contrast"]) > 1e-9:
            raise AssertionError(f"{model}: the cross's diagonal must equal the paired contrast")
        for field in strata:
            strata[field][model] = stratum_contrasts(raw, labels, field)
        print(f"{'':<52} neutral {neutral[model]['tie_aware_pct_male']:5.1f}  "
              f"perm p {perm[model]['p_two_sided']:.2g}  "
              f"full cross {cross[model]['contrast']:5.1f}", flush=True)

    out = {"script_version": SCRIPT_VERSION,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "label_source": label_source, "bootstrap_B": args.bootstrap,
           "definition": "tie-aware % male-favored, male-typed minus female-typed occupations",
           "n_occupations": len(occs),
           "n_bls_categories": len({bls[o] for o in occs}),
           "n_models": len(models),
           # interval entirely above zero / entirely below zero / either
           "n_models_above_zero": {s: sum(m["schemes"][s]["above_zero"]
                                          for m in models.values()) for s in SCHEMES},
           "n_models_below_zero": {s: sum(m["schemes"][s]["below_zero"]
                                          for m in models.values()) for s in SCHEMES},
           "models_below_zero": {s: sorted(k for k, m in models.items()
                                           if m["schemes"][s]["below_zero"]) for s in SCHEMES},
           "n_models_excluding_zero": {s: sum(m["schemes"][s]["excludes_zero"]
                                              for m in models.values()) for s in SCHEMES},
           "models": models}

    # ---- v1.2 blocks, after `models` so every earlier key is unchanged ----
    ss_count = summary["n_neutral_indeterminate_models"]
    classified_differently = []
    for m, v in sorted(neutral.items()):
        lo, hi = summary["models"][m]["by_category"]["neutral"]["tie_aware_pct_male_ci"]
        if (lo <= 50.0 <= hi) != v["schemes"]["occupation"]["contains_50"]:
            classified_differently.append({"model": m, "single_stage_ci": [lo, hi],
                                           "this_ci": v["schemes"]["occupation"]["ci"]})
    out["neutral_level"] = {
        "definition": "tie-aware % male-favored on the evenly split (neutral-labelled) "
                      "occupations, with 95% percentile intervals",
        "n_occupations": len(neutral_occs),
        "bootstrap_B": args.neutral_bootstrap,
        "bootstrap_B_note": "larger than the contrast intervals' B because one model's "
                            "interval ends within 0.1 point of 50 (rerank-2.5-lite, "
                            "occupation_x_name); at B = 50,000 Monte Carlo error alone "
                            "decides which side of 50 that end falls",
        "schemes": {"occupation": "the evenly split occupations, with replacement",
                    "occupation_x_name": "those occupations and the name pairs, "
                                         "independently (two-way)"},
        "interval_rule": "contains 50 when lo <= 50 <= hi (closed), as "
                         "analyze_single_stage.count_neutral_indeterminate",
        "n_models_containing_50": {s: sum(v["schemes"][s]["contains_50"] for v in neutral.values())
                                   for s in NEUTRAL_SCHEMES},
        "n_models_above_50": {s: sum(v["schemes"][s]["above_50"] for v in neutral.values())
                              for s in NEUTRAL_SCHEMES},
        "n_models_below_50": {s: sum(v["schemes"][s]["below_50"] for v in neutral.values())
                              for s in NEUTRAL_SCHEMES},
        "models_containing_50": {s: sorted(m for m, v in neutral.items()
                                           if v["schemes"][s]["contains_50"])
                                 for s in NEUTRAL_SCHEMES},
        # the occupation-only count should equal analyze_single_stage's, which
        # uses the same resampling with B = summary bootstrap_B and its own seeds
        "single_stage_check": {
            "n_neutral_indeterminate_models": ss_count,
            "single_stage_bootstrap_B": summary["bootstrap_B"],
            "equal": sum(v["schemes"]["occupation"]["contains_50"]
                         for v in neutral.values()) == ss_count,
            "models_classified_differently": classified_differently},
        "models": neutral}

    order = sorted(perm)
    for m, p in zip(order, holm([perm[m]["p_two_sided"] for m in order])):
        perm[m]["p_holm"] = float(p)
    pairs_per_occ = sorted({v["pairs_per_occupation"] for v in perm.values()})
    pos = sorted(m for m in perm if models[m]["contrast"] > 0 and perm[m]["p_holm"] < ALPHA)
    neg = sorted(m for m in perm if models[m]["contrast"] < 0 and perm[m]["p_holm"] < ALPHA)
    out["label_permutation"] = {
        "definition": "two-sided permutation test of the contrast: the labels of the "
                      "male- and female-typed occupations are permuted with the group "
                      "sizes fixed; p = (1 + #{|stat*| >= |stat_obs|}) / (1 + B)",
        "n_permutations": args.permutations,
        "n_occupations": {"male": int(is_male.sum()), "female": int((~is_male).sum())},
        "n_distinct_label_assignments": math.comb(len(occs), int(is_male.sum())),
        "pairs_per_occupation": pairs_per_occ,
        "equal_pairs_per_occupation": len(pairs_per_occ) == 1,
        "statistic_note": "every male- and female-typed occupation carries the same "
                          "number of pairs in every model, so the permuted statistic "
                          "(difference of the groups' mean per-occupation tie-aware "
                          "shares) is the pooled contrast in models.<m>.contrast",
        "min_attainable_p": 1 / (1 + args.permutations),
        "adjustment": f"Holm across the {len(perm)} models",
        "alpha": ALPHA,
        "n_models_positive_holm_significant": len(pos),
        "n_models_negative_holm_significant": len(neg),
        "models_positive_holm_significant": pos,
        "models_negative_holm_significant": neg,
        "models_not_holm_significant": sorted(set(perm) - set(pos) - set(neg)),
        # cross-check: one-sided (upper) p below a Bonferroni 0.05 / n_models
        "n_models_one_sided_upper_p_below_bonferroni":
            sum(v["p_one_sided_upper"] < ALPHA / len(perm) for v in perm.values()),
        "models": {m: {k: v[k] for k in ("p_two_sided", "p_holm", "p_one_sided_upper",
                                         "p_one_sided_lower",
                                         "n_permutations_at_least_as_extreme")}
                   for m, v in sorted(perm.items())}}

    shapes = {json.dumps([v["n_cells"], v["comparisons_per_cell"]]) for v in cross.values()}
    if len(shapes) != 1:
        raise AssertionError("every model's cross must cover the same cells")
    out["full_name_cross"] = {
        "definition": "within each (occupation, template, query) cell, every "
                      "male-associated document's score compared with every "
                      "female-associated document's score (10 x 10), ties counting "
                      "half; tie-aware % male-favored on male-typed minus female-typed "
                      "occupations",
        "n_cells": next(iter(cross.values()))["n_cells"],
        "comparisons_per_cell": next(iter(cross.values()))["comparisons_per_cell"],
        "n_models_positive": sum(v["contrast"] > 0 for v in cross.values()),
        "models_not_positive": sorted(m for m, v in cross.items() if v["contrast"] <= 0),
        "models": {m: {"contrast": v["contrast"], "tie_aware_pct_male": v["tie_aware_pct_male"]}
                   for m, v in sorted(cross.items())}}
    out["within_template"] = stratum_section(
        strata["template_style"], "template_style",
        "male-typed minus female-typed tie-aware % male-favored within each document template")
    out["within_query"] = stratum_section(
        strata["query_type"], "query_type",
        "male-typed minus female-typed tie-aware % male-favored within each query phrasing")

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    write_tex(models, args.bootstrap, TEX, {m: v["p_holm"] for m, v in perm.items()},
              args.permutations)
    for k in ("n_models_above_zero", "n_models_below_zero", "n_models_excluding_zero"):
        print(k, json.dumps(out[k]))
    print("neutral_level n_models_containing_50",
          json.dumps(out["neutral_level"]["n_models_containing_50"]),
          "| single_stage_check", json.dumps(out["neutral_level"]["single_stage_check"]))
    lp = out["label_permutation"]
    print(f"label_permutation: Holm p < {ALPHA}: positive {lp['n_models_positive_holm_significant']}, "
          f"negative {lp['n_models_negative_holm_significant']}; not significant "
          f"{lp['models_not_holm_significant']}")
    print(f"full_name_cross: {out['full_name_cross']['n_models_positive']} models positive")
    for k in ("within_template", "within_query"):
        s = out[k]
        print(f"{k}: {s['n_models_positive_in_all_levels']} models positive in all levels; "
              f"{s['n_model_level_cells_positive']}/{s['n_model_level_cells']} cells positive")


if __name__ == "__main__":
    main()
