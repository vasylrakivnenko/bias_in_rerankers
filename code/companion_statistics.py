"""
Robustness statistics for the companion experiment on published résumé examples.

Table A6 (results/tex/companion_table.tex, written by code/companion_table.py)
reports each reranker's tie-aware male-favored share of the companion
comparisons, overall and by occupation group, and its top-3 flip rate. This
script adds, per reranker, from the banked outputs only -- no model is loaded,
nothing is rescored and the résumé corpus is not read:

  1. in-window shares: the same shares without the comparisons whose every
     gender marker fell outside the model's 512-token window. There the model
     saw two identical inputs, which tie and so count as half male-favored;
     also the largest absolute change from Table A6's shares.
  2. the male-typed minus female-typed difference, on all comparisons and
     in-window.
  3. magnitude: mean |score_male - score_female| in-window, the same model's
     mean |score_male - score_female| in the synthetic audit (its
     results/*_full_raw.json, retained occupations only), and their ratio.
  4. net top-3 flips (female-form-only minus male-form-only), with a
     descriptive two-sided exact binomial p. The comparisons share pools (20
     focal comparisons per pool, and a résumé can sit in several pools), so the
     flips are not independent draws and the p is descriptive only.
  5. which of the 24 job categories fall in each occupation group.

Reads:  results/realistic_pool_four_rerankers.json        (Table A6's source)
        results/realistic_pool_four_rerankers_rows.json.gz (comparison-level rows)
        results/single_stage_summary.json, results/*_full_raw.json (synthetic audit)
Output: results/companion_statistics.json
Run:    venv/bin/python code/companion_statistics.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from labels import load_labels  # noqa: E402

SCRIPT_VERSION = "companion_statistics.py v1.0"
RESULTS = ROOT / "results"
SRC = RESULTS / "realistic_pool_four_rerankers.json"
ROWS = RESULTS / "realistic_pool_four_rerankers_rows.json.gz"
AUDIT = RESULTS / "single_stage_summary.json"
OUT = RESULTS / "companion_statistics.json"
# Table A6's columns after "All", in order, with the names the table prints
GROUPS = ("male", "female", "neutral", "unknown")
GROUP_LABELS = {"male": "male-typed", "female": "female-typed", "neutral": "evenly split",
                "unknown": "unlabeled"}
CELLS = ("overall",) + GROUPS
DEPTH = 3


def tie_aware(deltas):
    """Tie-aware % male-favored, computed as realistic_pool_experiment does."""
    n = len(deltas)
    return 100 * (sum(d > 0 for d in deltas) + 0.5 * sum(d == 0 for d in deltas)) / n


def shares(rows):
    out = {"overall": tie_aware([r["delta"] for r in rows])}
    for g in GROUPS:
        out[g] = tie_aware([r["delta"] for r in rows if r["stereotype"] == g])
    return out


def synthetic_mean_abs_delta(model, audit, labels):
    """Mean |score_male - score_female| over the model's retained synthetic pairs."""
    meta = audit["models"][model]
    rows = [r for r in json.loads((RESULTS / meta["raw_file"]).read_text())["results"]
            if r["occupation"] in labels]
    if len(rows) != meta["n_pairs"]:
        raise AssertionError(f"{model}: {len(rows)} retained synthetic pairs, "
                             f"single_stage_summary says {meta['n_pairs']}")
    scores = [s for r in rows for s in (r["score_male"], r["score_female"])]
    return (float(np.mean([abs(r["score_male"] - r["score_female"]) for r in rows])),
            meta["raw_file"], len(rows), [min(scores), max(scores)])


def analyse(short, entry, rows, audit, labels):
    s = entry["summary"]
    model = entry["model"]
    if len(rows) != s["overall"]["n"]:
        raise AssertionError(f"{short}: {len(rows)} rows, summary says {s['overall']['n']}")
    for r in rows:
        if r["delta"] != r["score_male"] - r["score_female"]:
            raise AssertionError(f"{short}: a row's delta is not score_male - score_female")

    inside = [r for r in rows if r["marker_in_token_window"]]
    outside = [r for r in rows if not r["marker_in_token_window"]]
    if len(outside) != entry["token_window"]["n_marker_outside_token_window"]:
        raise AssertionError(f"{short}: outside-window count disagrees with the summary")

    # 1. shares; the all-comparison ones must reproduce Table A6's source
    every, within = shares(rows), shares(inside)
    banked = {"overall": s["overall"]["score_gap"]["tie_aware_pct_male"],
              **{g: s["by_stereotype"][g]["score_gap"]["tie_aware_pct_male"] for g in GROUPS}}
    for c in CELLS:
        if round(every[c], 2) != banked[c]:
            raise AssertionError(f"{short}: recomputed {c} share {every[c]} does not "
                                 f"reproduce the banked {banked[c]}")
    change = {c: within[c] - every[c] for c in CELLS}
    worst = max(CELLS, key=lambda c: abs(change[c]))

    # 3. magnitude
    comp_abs = float(np.mean([abs(r["delta"]) for r in inside]))
    comp_abs_all = float(np.mean([abs(r["delta"]) for r in rows]))
    if round(comp_abs_all, 6) != s["overall"]["score_gap"]["mean_abs_delta"]:
        raise AssertionError(f"{short}: mean |delta| does not reproduce the banked value")
    syn_abs, raw_file, n_syn, syn_range = synthetic_mean_abs_delta(model, audit, labels)
    comp_scores = [x for r in rows for x in (r["score_male"], r["score_female"])]

    # 4. top-3 flips, recomputed from the exact-k ranks and checked against the summary
    m_in = [r["rank_male"] <= DEPTH for r in rows]
    f_in = [r["rank_female"] <= DEPTH for r in rows]
    male_only = sum(a and not b for a, b in zip(m_in, f_in))
    female_only = sum(b and not a for a, b in zip(m_in, f_in))
    top = s["overall"][f"top{DEPTH}"]
    if (male_only, female_only) != (top["male_form_in_only"], top["female_form_in_only"]):
        raise AssertionError(f"{short}: recomputed top-{DEPTH} flips disagree with the summary")
    flip_pools = {r["pool_id"] for r, a, b in zip(rows, m_in, f_in) if a != b}

    return {
        "model": model,
        "n_comparisons": len(rows),
        "n_outside_window": len(outside),
        "n_in_window": len(inside),
        "outside_window_all_ties": all(r["delta"] == 0 for r in outside),
        "n_in_window_by_group": {g: sum(r["stereotype"] == g for r in inside) for g in GROUPS},
        "tie_aware_pct_male": {"all_comparisons": every, "in_window": within,
                               "change_in_window_minus_all": change},
        "max_abs_change": abs(change[worst]),
        "max_abs_change_cell": worst,
        "male_minus_female": {"all_comparisons": every["male"] - every["female"],
                              "in_window": within["male"] - within["female"]},
        "mean_abs_delta": {"companion_in_window": comp_abs,
                           "companion_all_comparisons": comp_abs_all,
                           "synthetic_audit": syn_abs,
                           "ratio_synthetic_to_companion_in_window": syn_abs / comp_abs},
        "synthetic_audit": {"raw_file": raw_file, "n_pairs": n_syn},
        "score_range": {"companion": [min(comp_scores), max(comp_scores)],
                        "synthetic_audit": syn_range},
        f"top{DEPTH}": {
            "n_flips": male_only + female_only,
            "male_form_only": male_only,
            "female_form_only": female_only,
            "net_female_form": female_only - male_only,
            "binomial_p_two_sided": float(binomtest(female_only, male_only + female_only,
                                                    0.5).pvalue),
            "n_pools_with_a_flip": len(flip_pools)},
    }


def main():
    ap = argparse.ArgumentParser(description="Companion-experiment robustness statistics")
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()
    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    data = json.loads(SRC.read_text())
    with gzip.open(ROWS, "rt", encoding="utf-8") as fh:
        rows_file = json.load(fh)
    audit = json.loads(AUDIT.read_text())
    if rows_file["pool_membership_sha256"] != data["provenance"]["sampling"]["pool_membership_sha256"]:
        raise AssertionError("the rows file and the summary come from different pools")

    models = {short: analyse(short, entry, rows_file["models"][short]["rows"], audit, labels)
              for short, entry in data["models"].items()}

    # 5. job categories: one query per category, each with one occupation group
    group_of = {}
    for short in data["models"]:
        for r in rows_file["models"][short]["rows"]:
            if group_of.setdefault(r["category"], r["stereotype"]) != r["stereotype"]:
                raise AssertionError(f"category {r['category']} has two occupation groups")
    by_group = {g: sorted(c for c, x in group_of.items() if x == g) for g in GROUPS}
    if {g: len(v) for g, v in by_group.items() if v} != data["provenance"]["queries"]["stereotype_counts"]:
        raise AssertionError("category groups disagree with the summary's stereotype_counts")
    if len(group_of) != data["provenance"]["queries"]["n_queries"]:
        raise AssertionError("expected one job category per query")

    ratios = {k: v["mean_abs_delta"]["ratio_synthetic_to_companion_in_window"]
              for k, v in models.items()}
    worst = max(models, key=lambda k: models[k]["max_abs_change"])
    sampling = data["provenance"]["sampling"]
    out = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {"summary": str(SRC.relative_to(ROOT)), "rows": str(ROWS.relative_to(ROOT)),
                   "summary_script_version": data["script_version"],
                   "synthetic_audit": str(AUDIT.relative_to(ROOT)),
                   "synthetic_label_source": label_source},
        "group_labels": GROUP_LABELS,
        "n_models": len(models),
        # top-level counts the paper can quote directly
        "n_models_male_minus_female_positive_all_comparisons":
            sum(v["male_minus_female"]["all_comparisons"] > 0 for v in models.values()),
        "n_models_male_minus_female_positive_in_window":
            sum(v["male_minus_female"]["in_window"] > 0 for v in models.values()),
        "n_models_overall_share_below_50_all_comparisons":
            sum(v["tie_aware_pct_male"]["all_comparisons"]["overall"] < 50 for v in models.values()),
        "n_models_overall_share_below_50_in_window":
            sum(v["tie_aware_pct_male"]["in_window"]["overall"] < 50 for v in models.values()),
        "max_abs_share_change_in_window": models[worst]["max_abs_change"],
        "max_abs_share_change_in_window_where": {"model": worst,
                                                 "cell": models[worst]["max_abs_change_cell"]},
        "mean_abs_delta_ratio_synthetic_to_companion_min": min(ratios.values()),
        "mean_abs_delta_ratio_synthetic_to_companion_max": max(ratios.values()),
        "mean_abs_delta_ratio_min_model": min(ratios, key=ratios.get),
        "mean_abs_delta_ratio_max_model": max(ratios, key=ratios.get),
        f"n_models_net_top{DEPTH}_flips_favor_female_form":
            sum(v[f"top{DEPTH}"]["net_female_form"] > 0 for v in models.values()),
        "n_job_categories": len(group_of),
        "n_job_categories_male_typed": len(by_group["male"]),
        "n_job_categories_female_typed": len(by_group["female"]),
        "n_job_categories_evenly_split": len(by_group["neutral"]),
        "n_job_categories_unlabeled": len(by_group["unknown"]),
        "job_categories_by_group": by_group,
        "notes": {
            "window": "outside-window comparisons are rows with marker_in_token_window false: "
                      "every gender marker lies beyond the model's tokenizer window "
                      f"({next(iter(data['models'].values()))['token_window']['tokenizer_model_max_length']} "
                      "tokens) for the (query, résumé) pair, so the two forms' token ids are "
                      "identical and the scores tie. Counts differ by tokenizer (MiniLM "
                      "uses a different vocabulary). The character-based "
                      "marker_in_scored_window heuristic is not used.",
            "shares": "tie-aware % male-favored (ties count half), full precision; Table A6 "
                      "prints the all-comparison values to one decimal",
            "magnitude": "mean |score_male - score_female| on each model's own output scale "
                         "(its CrossEncoder default: scores in (0, 1) for BGE and Jina, "
                         "unbounded logits for MiniLM). Both runs score a given model on the "
                         "same scale (see score_range), so each ratio compares like with "
                         "like; ratios are not comparable to raw score gaps across models",
            "flips": f"top-{DEPTH} membership from the exact-k ranks; net_female_form = "
                     "female-form-only minus male-form-only. The binomial p tests "
                     "female_form_only ~ Binomial(n_flips, 1/2) and is descriptive only: "
                     f"the {sampling['n_focal_comparisons']:,} comparisons come from "
                     f"{sampling['n_pools']} pools of "
                     f"{sampling['n_focal_comparisons'] // sampling['n_pools']} focal "
                     f"comparisons that share their other members, and "
                     f"{sampling['n_distinct_resumes_scored']:,} distinct résumés fill "
                     "them, so flips are not independent",
        },
        "models": models,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    for short, v in models.items():
        t = v[f"top{DEPTH}"]
        print(f"{short:<9} outside window {v['n_outside_window']:>3}  "
              f"overall {v['tie_aware_pct_male']['all_comparisons']['overall']:.2f} -> "
              f"{v['tie_aware_pct_male']['in_window']['overall']:.2f}  "
              f"max |change| {v['max_abs_change']:.2f} ({v['max_abs_change_cell']})  "
              f"M-F {v['male_minus_female']['all_comparisons']:+.2f} / "
              f"{v['male_minus_female']['in_window']:+.2f}  "
              f"|delta| ratio {v['mean_abs_delta']['ratio_synthetic_to_companion_in_window']:.1f}  "
              f"top-3 net female {t['net_female_form']:+d} of {t['n_flips']} "
              f"(p = {t['binomial_p_two_sided']:.3g})")
    print("job categories:", {g: len(c) for g, c in by_group.items()})
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
