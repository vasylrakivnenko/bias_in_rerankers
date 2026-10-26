"""
The companion experiment at each reranker's full input window, against the banked run.

The banked companion run (results/realistic_pool_four_rerankers.json) scored every
(query, résumé) pair truncated to 512 tokens. Two of its four rerankers accept longer
inputs: BAAI/bge-reranker-v2-m3 up to 8,192 tokens and
jinaai/jina-reranker-v2-base-multilingual up to 1,024. code/realistic_pool_experiment.py
re-scored the same pools and pairs at those windows; this script compares the two runs
comparison by comparison, from the saved outputs only -- no model is loaded.

For each model and each run (512 tokens; full window):
  1. tie-aware male-favored shares, overall and by occupation group, and the male-typed
     minus female-typed difference (M-F), with a 95% bootstrap interval that resamples
     pools within each job category (the 24 categories are fixed by design, so the
     interval reflects which résumés were drawn, not which occupations);
  2. comparisons whose every marker lay beyond the window, and the share of ties;
  3. mean |score_male - score_female| over the in-window comparisons, and the synthetic
     audit's over it;
  4. top-3 flips by the form they favor, with a descriptive two-sided binomial p;
  5. the same by how the pair was built (prepended name; swap in window; swap cut off).
And, pairing the runs: how often a comparison keeps its winner (male form, female form,
or tie), overall and by its construction in the 512-token run; the change in M-F with a
paired bootstrap interval; and the comparisons the 512-token window cut off, now scored.

Further rerankers were scored on the same comparisons with whole documents only: three
hosted ones (Voyage rerank-3 and rerank-2.5, Cohere rerank-v4.0-pro; 32,000-token windows)
and three long-window open-weight ones (gte-reranker-modernbert-base and
granite-embedding-reranker-english-r2 at 8,192 tokens on the Mac; zerank-2 at 8,192 on an
RTX 4090, one input per batch). For them the same statistics are reported for the one run,
beside the model's synthetic-audit contrast; Voyage's comparisons are also split by whether
their request kept rounded scores. Cohere's Rerank 5 (rerank-v5.0-pro and -fast, released
2026-10-21, after the audit) was scored the same way on Cohere's own API; its synthetic
contrast comes from code/post_audit_synthetic.py, which scored the audit's pairs with it
(results/post_audit/), and these models are flagged post_audit. An overview lists, for every
model, the synthetic contrast and the whole-document one, with their rank correlation across
models (also without the post-audit models).

Continuous check (every run): the 7 male-typed against the 3 female-typed categories uses 10
of the corpus's 24 job categories, so each run is also tested on all 21 categories with a BLS
women's share (1.8% to 79.2%): Spearman rho between a category's tie-aware male-favored
share and its share of women, the least-squares slope per 10 points of women's share, and a
permutation p that shuffles the women's shares across categories -- the categories, not the
résumés, are the units, as in the synthetic audit's continuous check.

Reads:  results/realistic_pool_four_rerankers.json (+ _rows.json.gz)   (512 tokens)
        results/realistic_pool_full_window_<model>.json (+ _rows.json.gz)
        results/interval_sensitivity.json            (synthetic-audit contrasts)
        results/single_stage_summary.json, results/*_full_raw.json   (synthetic audit)
        results/post_audit/<model>_synthetic.json   (models released after the audit)
Output: results/companion_full_window.json
Run:    venv/bin/python code/companion_full_window.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from companion_statistics import (AUDIT, CONSTRUCTIONS, DEPTH, GROUPS, construction,  # noqa: E402
                                  flip, shares, synthetic_mean_abs_delta, tie_aware)
from realistic_pool_experiment import load_queries  # noqa: E402
from labels import load_labels  # noqa: E402

SCRIPT_VERSION = "companion_full_window.py v1.1"
RESULTS = ROOT / "results"
BANK = RESULTS / "realistic_pool_four_rerankers.json"
FULL = {"bge-m3": RESULTS / "realistic_pool_full_window_bge-m3.json",
        "jina": RESULTS / "realistic_pool_full_window_jina.json"}
# Rerankers scored on whole documents only (no 512-token run to pair with)
FULL_ONLY = {m: RESULTS / f"realistic_pool_full_window_{m}.json" for m in (
    "voyage-rerank-3", "voyage-rerank-2.5", "cohere-v4-pro",
    "gte-modernbert", "granite-r2", "zerank-2", "cohere-v5-pro", "cohere-v5-fast",
    # the rest of the audit's rerankers that read a whole resume (October 2026)
    "voyage-rerank-2", "voyage-rerank-3-lite", "voyage-rerank-2.5-lite", "voyage-rerank-1",
    "voyage-rerank-2-lite", "cohere-v4-fast", "cohere-v3.5", "qwen3-8b", "nemotron-vl",
    "qwen3-0.6b", "qwen3-4b", "mxbai-large-v2", "nemotron-1b", "jina-v3")}
# jina-reranker-v3 is listwise and order-sensitive: with each synthetic pair's female form
# sent first, only 65% of the audit's pairs keep their winner. Every request sends a unit's
# male form first, so a second whole-CV pass sends the female form first (--female-first),
# and for this model both contrasts are counterbalanced: a pair's gap is the mean of its gaps
# in the two orders (synthetic side: the audit's "reversed" arm, kept in its raw file).
COUNTERBALANCED = {"jina-v3": RESULTS / "realistic_pool_full_window_jina-v3_female_first.json"}
# Released after the audit: the audit's pairs scored by code/post_audit_synthetic.py
POST_AUDIT = {m: RESULTS / "post_audit" / f"{m}_synthetic.json"
              for m in ("rerank-v5.0-pro", "rerank-v5.0-fast")}
# The synthetic audit scored Granite through IBM's API under this name; the open weights
# used here reproduce it (99.9% of winners, contrast 15.2 vs 15.3; reproducibility_checks)
SYNTHETIC_KEY = {"ibm-granite/granite-embedding-reranker-english-r2":
                 "ibm-granite-granite-embedding-reranker-english-r2",
                 # the audit reached these through OpenRouter; the whole-CV runs call the
                 # backend that served them (Cohere, Fireworks) or OpenRouter's free route
                 "rerank-v3.5": "cohere/rerank-v3.5",
                 "accounts/fireworks/models/qwen3-reranker-8b": "qwen/qwen3-reranker-8b",
                 "nvidia/llama-nemotron-rerank-vl-1b-v2:free": "nvidia/llama-nemotron-rerank-vl-1b-v2"}
SYNTHETIC_CONTRASTS = RESULTS / "interval_sensitivity.json"
OUT = RESULTS / "companion_full_window.json"
N_BOOT = 10_000
N_PERM = 100_000
SEED = 20261019


def model_rng(short):
    """The model's own random stream, so adding or dropping a model leaves the others'
    bootstrap and permutation draws unchanged."""
    return np.random.default_rng([SEED, int(hashlib.sha256(short.encode()).hexdigest()[:8], 16)])


def load_run(path):
    data = json.loads(path.read_text())
    rows_file = {m["rows_file"] for m in data["models"].values()}.pop()
    with gzip.open(ROOT / rows_file, "rt", encoding="utf-8") as fh:
        rows = json.load(fh)
    if rows["pool_membership_sha256"] != data["provenance"]["sampling"]["pool_membership_sha256"]:
        raise AssertionError(f"{path.name}: rows and summary come from different pools")
    return data, rows


def winner(r):
    return "male" if r["delta"] > 0 else "female" if r["delta"] < 0 else "tie"


def flips(rows):
    f = [flip(r) for r in rows]
    m, w = f.count("male"), f.count("female")
    return {"n_flips": m + w, "flip_rate_pct": 100 * (m + w) / len(rows) if rows else None,
            "male_form_only": m, "female_form_only": w, "net_female_form": w - m,
            "binomial_p_two_sided": float(binomtest(w, m + w, 0.5).pvalue) if m + w else None}


def group_stats(rows, syn_abs):
    """n, ties, share, mean |delta| (and the synthetic audit's over it) and top-3 flips."""
    if not rows:
        return {"n": 0}
    mean_abs = float(np.mean([abs(r["delta"]) for r in rows]))
    by = {g: [r["delta"] for r in rows if r["stereotype"] == g] for g in ("male", "female")}
    return {"n": len(rows), "n_ties": sum(r["delta"] == 0 for r in rows),
            "tie_aware_pct_male": tie_aware([r["delta"] for r in rows]),
            "male_minus_female": (tie_aware(by["male"]) - tie_aware(by["female"])
                                  if by["male"] and by["female"] else None),
            "mean_abs_delta": mean_abs,
            "ratio_synthetic_to_companion": syn_abs / mean_abs if mean_abs else None,
            f"top{DEPTH}": flips(rows)}


def run_stats(rows, syn_abs):
    inside = [r for r in rows if r["marker_in_token_window"]]
    every = shares(rows)
    comp_abs = float(np.mean([abs(r["delta"]) for r in inside]))
    return {
        "n_comparisons": len(rows),
        "n_outside_window": len(rows) - len(inside),
        "pct_ties": 100 * sum(r["delta"] == 0 for r in rows) / len(rows),
        "tie_aware_pct_male": every,
        "tie_aware_pct_male_in_window": shares(inside),
        "male_minus_female": every["male"] - every["female"],
        "mean_abs_delta_in_window": comp_abs,
        "ratio_synthetic_to_companion_in_window": syn_abs / comp_abs,
        f"top{DEPTH}": flips(rows),
        "by_construction": {c: group_stats([r for r in rows if construction(r) == c], syn_abs)
                            for c in CONSTRUCTIONS},
    }


def continuous_check(rows, pct_women, rng):
    """Male-favored share against the BLS share of women, one point per job category."""
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r["delta"])
    cats = sorted(c for c in by if pct_women.get(c) is not None)
    x = np.array([pct_women[c] for c in cats])
    y = np.array([tie_aware(by[c]) for c in cats])
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(N_PERM)])
    return {"n_categories": len(cats), "spearman_rho": rho,
            "slope_per_10_points_women": float(np.polyfit(x, y, 1)[0] * 10),
            "permutation_p_two_sided": float((1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12))
                                             / (N_PERM + 1)),
            "n_permutations": N_PERM,
            "by_category": {c: {"pct_women": float(pct_women[c]), "tie_aware_pct_male": float(v)}
                            for c, v in zip(cats, y)}}


def bootstrap_m_minus_f(rows_a, rows_b, rng):
    """M-F of each run and their difference, resampling pools within each job category.

    Each pool belongs to one category and so to one occupation group; a replicate draws,
    for every category, as many of its pools as it has, with replacement, and both runs
    use the same draw, so the interval of the difference is paired."""
    per_pool = defaultdict(lambda: np.zeros(3))  # n, tie-aware male wins in run a, in run b
    pool_cat, pool_group = {}, {}
    for ra, rb in zip(rows_a, rows_b):
        g = ra["stereotype"]
        if g not in ("male", "female"):
            continue
        acc = per_pool[ra["pool_id"]]
        acc[0] += 1
        acc[1] += (ra["delta"] > 0) + 0.5 * (ra["delta"] == 0)
        acc[2] += (rb["delta"] > 0) + 0.5 * (rb["delta"] == 0)
        pool_cat[ra["pool_id"]], pool_group[ra["pool_id"]] = ra["category"], g
    pools = sorted(per_pool)
    stats = np.array([per_pool[p] for p in pools])
    male = np.array([pool_group[p] == "male" for p in pools])
    by_cat = defaultdict(list)
    for i, p in enumerate(pools):
        by_cat[pool_cat[p]].append(i)
    draws = np.concatenate([rng.choice(idx, size=(N_BOOT, len(idx)), replace=True)
                            for idx in by_cat.values()], axis=1)
    picked = stats[draws]                        # (N_BOOT, n_pools, 3)
    is_male = male[draws]
    def contrast(col):
        m = (picked[..., col] * is_male).sum(1) / (picked[..., 0] * is_male).sum(1)
        f = (picked[..., col] * ~is_male).sum(1) / (picked[..., 0] * ~is_male).sum(1)
        return 100 * (m - f)
    a, b = contrast(1), contrast(2)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"ci_512": ci(a), "ci_full_window": ci(b), "ci_change": ci(b - a),
            "n_bootstrap": N_BOOT, "resampling": "pools within each job category, paired"}


def own_marker_stats(rows):
    """The comparisons built from the résumé's own name or pronouns (mode swap)."""
    sw = [r for r in rows if r["mode"] == "swap"]
    by = {g: [r["delta"] for r in sw if r["stereotype"] == g] for g in GROUPS}
    return {"n": len(sw), "n_outside_window": sum(not r["marker_in_token_window"] for r in sw),
            "n_by_group": {g: len(d) for g, d in by.items()},
            "tie_aware_pct_male": tie_aware([r["delta"] for r in sw]),
            "tie_aware_pct_male_by_group": {g: tie_aware(d) if d else None for g, d in by.items()},
            "male_minus_female": (tie_aware(by["male"]) - tie_aware(by["female"])
                                  if by["male"] and by["female"] else None),
            f"top{DEPTH}": flips(sw)}


def bf16_round(x):
    """x rounded to the nearest bfloat16 (ties to even), as Voyage's rounded responses are."""
    b = np.array(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return float(b.astype(np.uint32).view(np.float32))


def counterbalance(short, rows, bank_rows, audit, key, labels, rng, pct_women):
    """Both orders of an order-sensitive model: the female-first pass's statistics, how
    often a comparison keeps its winner across the two orders, and the counterbalanced
    contrasts -- whole-CV and synthetic -- from each pair's mean gap."""
    ff, ff_rows = load_run(COUNTERBALANCED[short])
    if not ff["config"].get("female_first"):
        raise AssertionError(f"{short}: {COUNTERBALANCED[short].name} is not a female-first run")
    if ff_rows["pool_membership_sha256"] != bank_rows["pool_membership_sha256"]:
        raise AssertionError(f"{short}: the female-first run used other pools")
    other = sorted(ff_rows["models"][short]["rows"], key=lambda r: (r["pool_id"], r["focal_index"]))
    same = ("pool_id", "focal_index", "focal_rid", "mode", "stereotype", "category")
    if len(other) != len(rows) or any(tuple(a[k] for k in same) != tuple(b[k] for k in same)
                                      for a, b in zip(rows, other)):
        raise AssertionError(f"{short}: the two orders do not hold the same comparisons")
    mean = [{**a, "delta": (a["delta"] + b["delta"]) / 2} for a, b in zip(rows, other)]
    raw = json.loads((RESULTS / audit["models"][key]["raw_file"]).read_text())["results"]
    syn = [r for r in raw if r["occupation"] in labels]
    syn_gap = {g: [((r["score_male"] - r["score_female"])
                    + (r["arms"]["reversed"][0] - r["arms"]["reversed"][1])) / 2
                   for r in syn if labels[r["occupation"]] == g] for g in ("male", "female")}
    sh = shares(mean)
    return {"female_first_run": str(COUNTERBALANCED[short].relative_to(ROOT)),
            "male_first": {"tie_aware_pct_male": shares(rows),
                           "male_minus_female": shares(rows)["male"] - shares(rows)["female"]},
            "female_first": {"tie_aware_pct_male": shares(other),
                             "male_minus_female": shares(other)["male"] - shares(other)["female"]},
            "pct_same_winner_both_orders": 100 * sum(winner(a) == winner(b)
                                                     for a, b in zip(rows, other)) / len(rows),
            "counterbalanced": {"tie_aware_pct_male": sh,
                                "male_minus_female": sh["male"] - sh["female"],
                                "ci": bootstrap_m_minus_f(mean, mean, rng)["ci_512"],
                                "continuous_check": continuous_check(mean, pct_women, rng)},
            "synthetic_counterbalanced_male_minus_female":
                tie_aware(syn_gap["male"]) - tie_aware(syn_gap["female"])}


def hosted_model(short, bank_rows, audit, labels, synthetic_contrast, rng, pct_women):
    """A reranker scored on whole documents only: the statistics of its one run."""
    full, full_rows = load_run(FULL_ONLY[short])
    if full_rows["pool_membership_sha256"] != bank_rows["pool_membership_sha256"]:
        raise AssertionError(f"{short}: the run used other pools")
    if full_rows["resumes"] != bank_rows["resumes"]:
        raise AssertionError(f"{short}: the résumé forms differ from the banked run's")
    rows = sorted(full_rows["models"][short]["rows"], key=lambda r: (r["pool_id"], r["focal_index"]))
    if len(rows) != len(bank_rows["models"]["bge-m3"]["rows"]):
        raise AssertionError(f"{short}: {len(rows)} comparisons, not the full sample")
    if not all(r["marker_in_token_window"] for r in rows):
        raise AssertionError(f"{short}: a comparison was not read whole")
    entry = full["models"][short]
    model = entry["model"]
    key = SYNTHETIC_KEY.get(model, model)
    if model in POST_AUDIT:
        post = json.loads(POST_AUDIT[model].read_text())
        if post["model"] != model or post.get("canary_limit_queries"):
            raise AssertionError(f"{short}: {POST_AUDIT[model].name} is not a full run of {model}")
        syn_abs, syn_contrast = post["headline"]["mean_abs_delta"], post["headline"]["contrast"]
    else:
        syn_abs = synthetic_mean_abs_delta(key, audit, labels)[0]
        syn_contrast = synthetic_contrast[key]["contrast"]
    stats = run_stats(rows, syn_abs)
    out = {"model": model, "api": entry.get("api"), "post_audit": model in POST_AUDIT,
           "config": {k: full["config"].get(k) for k in (
               "max_length", "token_budget", "batch_size", "revision", "model_dtype", "device")},
           "synthetic_audit_male_minus_female": syn_contrast,
           "full_window": stats,
           "bootstrap": {k: v for k, v in bootstrap_m_minus_f(rows, rows, rng).items()
                         if k in ("ci_512", "n_bootstrap", "resampling")},
           "own_marker_swaps": own_marker_stats(rows),
           "continuous_check": continuous_check(rows, pct_women, rng)}
    out["bootstrap"]["ci"] = out["bootstrap"].pop("ci_512")
    if short in COUNTERBALANCED and COUNTERBALANCED[short].exists():
        out["counterbalanced"] = counterbalance(short, rows, bank_rows, audit, key, labels,
                                                rng, pct_women)
    if (entry.get("api") or {}).get("vendor") == "voyage":
        coarse = [r for r in rows if r.get("api_coarse_response")]
        fine = [r for r in rows if not r.get("api_coarse_response")]
        out["by_response_precision"] = {
            "full_precision": {"n": len(fine), **({"tie_aware_pct_male": shares(fine),
                                                   "male_minus_female": shares(fine)["male"]
                                                   - shares(fine)["female"]} if fine else {})},
            "rounded": {"n": len(coarse)}}
        # Sensitivity: every score rounded to bfloat16, as most of the synthetic audit's
        # Voyage responses were (rounded pairs that meet tie, and count half)
        rounded = [{**r, "delta": bf16_round(r["score_male"]) - bf16_round(r["score_female"])}
                   for r in rows]
        sh = shares(rounded)
        out["all_scores_rounded_to_bf16"] = {
            "pct_ties": 100 * sum(r["delta"] == 0 for r in rounded) / len(rounded),
            "tie_aware_pct_male": sh, "male_minus_female": sh["male"] - sh["female"]}
    return out


def main():
    ap = argparse.ArgumentParser(description="Companion experiment: 512 tokens vs full window")
    ap.add_argument("--models", nargs="+", choices=list(FULL) + list(FULL_ONLY),
                    default=list(FULL) + [m for m, p in FULL_ONLY.items() if p.exists()])
    args = ap.parse_args()
    labels, label_source = load_labels(strict=True)
    audit = json.loads(AUDIT.read_text())
    bank, bank_rows = load_run(BANK)
    synthetic_contrast = json.loads(SYNTHETIC_CONTRASTS.read_text())["models"]
    pct_women = {q.category: q.pct_women for q in load_queries(None, None, None, 0)[0]}
    hosted = {short: hosted_model(short, bank_rows, audit, labels, synthetic_contrast,
                                  model_rng(short), pct_women)
              for short in args.models if short in FULL_ONLY}
    models = {}
    for short in [m for m in args.models if m in FULL]:
        rng = model_rng(short)
        full, full_rows = load_run(FULL[short])
        if full_rows["pool_membership_sha256"] != bank_rows["pool_membership_sha256"]:
            raise AssertionError(f"{short}: the full-window run used other pools")
        if full["models"][short]["model"] != bank["models"][short]["model"]:
            raise AssertionError(f"{short}: a different model")
        if full_rows["resumes"] != bank_rows["resumes"]:
            raise AssertionError(f"{short}: the résumé forms differ from the banked run's")
        a = sorted(bank_rows["models"][short]["rows"], key=lambda r: (r["pool_id"], r["focal_index"]))
        b = sorted(full_rows["models"][short]["rows"], key=lambda r: (r["pool_id"], r["focal_index"]))
        same = ("pool_id", "focal_index", "focal_rid", "mode", "stereotype", "category")
        if len(a) != len(b) or any(tuple(x[k] for k in same) != tuple(y[k] for k in same)
                                   for x, y in zip(a, b)):
            raise AssertionError(f"{short}: the two runs do not hold the same comparisons")
        syn_abs = synthetic_mean_abs_delta(bank["models"][short]["model"], audit, labels)[0]
        s512, sfull = run_stats(a, syn_abs), run_stats(b, syn_abs)

        # pairing: same winner, overall and by construction in the 512-token run
        keep = {}
        for c in ("all",) + CONSTRUCTIONS:
            pairs = [(x, y) for x, y in zip(a, b) if c == "all" or construction(x) == c]
            keep[c] = {"n": len(pairs),
                       "pct_same_winner": (100 * sum(winner(x) == winner(y) for x, y in pairs)
                                           / len(pairs)) if pairs else None,
                       "pct_same_top3_membership_both_forms": (100 * sum(
                           (x["rank_male"] <= DEPTH, x["rank_female"] <= DEPTH)
                           == (y["rank_male"] <= DEPTH, y["rank_female"] <= DEPTH)
                           for x, y in pairs) / len(pairs)) if pairs else None}
        cut = [y for x, y in zip(a, b) if construction(x) == "swap_truncated"]
        # the pairs built from the résumé's own name or pronouns, whatever the window
        own = {"window_512": own_marker_stats(a), "full_window": own_marker_stats(b)}
        boot = bootstrap_m_minus_f(a, b, rng)
        models[short] = {
            "model": bank["models"][short]["model"],
            "max_length_full_window": full["config"]["max_length"],
            "full_window_run": {k: full["models"][short][k] for k in (
                "n_pairs_scored", "n_pairs_resumed_from_checkpoint",
                "n_pairs_identical_inputs_scored_once", "score_seconds", "pairs_per_second")},
            "full_window_config": {k: full["config"][k] for k in (
                "max_length", "token_budget", "batching", "device")},
            "window_512": s512,
            "full_window": sfull,
            "m_minus_f_change": sfull["male_minus_female"] - s512["male_minus_female"],
            "bootstrap": boot,
            "continuous_check": {"window_512": continuous_check(a, pct_women, rng),
                                 "full_window": continuous_check(b, pct_women, rng)},
            "same_winner": keep,
            "own_marker_swaps": own,
            "cut_at_512_now_scored": {
                **group_stats(cut, syn_abs),
                "n_still_outside_window": sum(not r["marker_in_token_window"] for r in cut),
                "tie_aware_pct_male_by_group": {
                    g: (tie_aware([r["delta"] for r in cut if r["stereotype"] == g])
                        if any(r["stereotype"] == g for r in cut) else None) for g in GROUPS},
                "n_by_group": {g: sum(r["stereotype"] == g for r in cut) for g in GROUPS}},
        }

    # every model: synthetic contrast against the whole-document one (BGE and Jina at their
    # own windows; Jina's 1,024 tokens still cut most résumés)
    overview = {}
    for short, v in models.items():
        overview[short] = {"model": v["model"], "reads_every_resume_whole":
                           v["full_window"]["n_outside_window"] == 0 and short != "jina",
                           "synthetic": synthetic_contrast[SYNTHETIC_KEY.get(v["model"], v["model"])]["contrast"],
                           "whole_document": v["full_window"]["male_minus_female"],
                           "ci": v["bootstrap"]["ci_full_window"],
                           "continuous": v["continuous_check"]["full_window"]}
    for short, v in hosted.items():
        cb = v.get("counterbalanced")  # an order-sensitive model: both orders averaged
        overview[short] = {"model": v["model"], "reads_every_resume_whole": True,
                           "post_audit": v["post_audit"],
                           "synthetic": (cb["synthetic_counterbalanced_male_minus_female"] if cb
                                         else v["synthetic_audit_male_minus_female"]),
                           "whole_document": (cb["counterbalanced"]["male_minus_female"] if cb
                                              else v["full_window"]["male_minus_female"]),
                           "ci": cb["counterbalanced"]["ci"] if cb else v["bootstrap"]["ci"],
                           "continuous": (cb["counterbalanced"]["continuous_check"] if cb
                                          else v["continuous_check"]),
                           **({"counterbalanced": True} if cb else {})}
    from scipy.stats import spearmanr, pearsonr

    def correlate(entries):
        xs = [o["synthetic"] for o in entries]
        ys = [o["whole_document"] for o in entries]
        return ({"n_models": len(xs), "spearman_rho": float(spearmanr(xs, ys)[0]),
                 "pearson_r": float(pearsonr(xs, ys)[0])} if len(xs) >= 3 else None)
    correlation = correlate(list(overview.values()))
    if any(o.get("post_audit") for o in overview.values()):
        correlation["audit_models_only"] = correlate(
            [o for o in overview.values() if not o.get("post_audit")])
    out = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {"window_512": str(BANK.relative_to(ROOT)),
                   "full_window": {k: str({**FULL, **FULL_ONLY}[k].relative_to(ROOT))
                                   for k in args.models},
                   "synthetic_contrasts": str(SYNTHETIC_CONTRASTS.relative_to(ROOT)),
                   "synthetic_audit": str(AUDIT.relative_to(ROOT)),
                   "synthetic_label_source": label_source},
        "notes": {
            "window": "full window = the model's own maximum input length (8,192 tokens for "
                      "bge-reranker-v2-m3, 1,024 for jina-reranker-v2-base-multilingual); the "
                      "banked run truncated every (query, résumé) pair to 512. Same pools, "
                      "résumé forms, queries, seeds and tie-break keys in both runs.",
            "shares": "tie-aware % male-favored (ties count half), as in companion_statistics.py",
            "bootstrap": "pools resampled with replacement within each of the 24 job categories, "
                         "the same draw for both runs; it reflects which résumés were drawn, "
                         "not which occupations, and the female-typed group has 3 categories",
            "same_winner": "winner = male form, female form, or tie (equal scores); "
                           "by_construction groups follow the 512-token run",
            "hosted_models": "scored once, on whole documents (32,000-token windows; Voyage "
                             "with truncation off, Cohere with max_tokens_per_doc 32000); "
                             "synthetic_audit_male_minus_female is the model's tie-aware "
                             "contrast in the synthetic audit (interval_sensitivity.json). "
                             "Voyage sometimes returns every score rounded to bfloat16; such "
                             "a request was re-sent, and by_response_precision sets apart "
                             "the comparisons whose request kept rounded scores",
            "counterbalanced": "jina-reranker-v3 is listwise and order-sensitive; its overview "
                               "row averages each pair's gap over the male-first run and a "
                               "female-first run, and its synthetic contrast likewise over the "
                               "audit's same and reversed arms (see COUNTERBALANCED)",
            "random_streams": f"each model draws its bootstrap and permutations from its own "
                              f"stream, seeded by {SEED} and its short name (model_rng), so "
                              f"adding or dropping a model leaves the others' intervals and "
                              f"p values unchanged",
            "post_audit": "models released after the audit (Cohere rerank-v5.0-pro and -fast, "
                          "2026-10-21), called on Cohere's own API; their synthetic contrast "
                          "is the audit's statistic over the same 8,320 retained pairs, scored "
                          "in October 2026 by code/post_audit_synthetic.py "
                          "(results/post_audit/<model>_synthetic.json), not by the audit",
        },
        "models": models,
        "hosted_models": hosted,
        "overview": overview,
        "overview_correlation": correlation,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    for short, v in models.items():
        print(f"== {short} ({v['model']}), 512 -> {v['max_length_full_window']} tokens")
        for run in ("window_512", "full_window"):
            s = v[run]
            t = s[f"top{DEPTH}"]
            ci = v["bootstrap"]["ci_512" if run == "window_512" else "ci_full_window"]
            print(f"  {run:<12} cut {s['n_outside_window']:>3}  ties {s['pct_ties']:4.1f}%  "
                  f"all {s['tie_aware_pct_male']['overall']:.1f}  "
                  + " ".join(f"{g[:4]} {s['tie_aware_pct_male'][g]:.1f}" for g in GROUPS)
                  + f"  M-F {s['male_minus_female']:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]  "
                  f"|d| {s['mean_abs_delta_in_window']:.3g} (synthetic/companion "
                  f"{s['ratio_synthetic_to_companion_in_window']:.1f})  "
                  f"top-3 flips {t['flip_rate_pct']:.1f}% M/F {t['male_form_only']}/{t['female_form_only']}")
        c = v["bootstrap"]["ci_change"]
        print(f"  M-F change {v['m_minus_f_change']:+.1f} [{c[0]:+.1f}, {c[1]:+.1f}]; same winner "
              + ", ".join(f"{k} {x['pct_same_winner']:.1f}% (n {x['n']})"
                          for k, x in v["same_winner"].items() if x["n"]))
        cut = v["cut_at_512_now_scored"]
        print(f"  cut at 512, now scored: n {cut['n']}, still cut {cut['n_still_outside_window']}, "
              f"ties {cut['n_ties']}, male-favored {cut['tie_aware_pct_male']:.1f}% "
              f"(by group " + ", ".join(f"{g[:4]} {x:.1f} n {cut['n_by_group'][g]}"
                                        for g, x in cut["tie_aware_pct_male_by_group"].items()
                                        if x is not None)
              + f"), top-3 flips M/F {cut[f'top{DEPTH}']['male_form_only']}/"
              f"{cut[f'top{DEPTH}']['female_form_only']}")
    for short, v in models.items():
        for run, o in v["own_marker_swaps"].items():
            print(f"  {short} own-marker swaps, {run}: n {o['n']} (cut {o['n_outside_window']}), "
                  f"male-favored {o['tie_aware_pct_male']:.1f}%, "
                  + ", ".join(f"{g[:4]} {x:.1f} (n {o['n_by_group'][g]})"
                              for g, x in o['tie_aware_pct_male_by_group'].items() if x is not None)
                  + f", M-F {o['male_minus_female']:+.1f}, top-3 flips M/F "
                  f"{o[f'top{DEPTH}']['male_form_only']}/{o[f'top{DEPTH}']['female_form_only']}")
    for short, v in hosted.items():
        s, t, ci = v["full_window"], v["full_window"][f"top{DEPTH}"], v["bootstrap"]["ci"]
        print(f"== {short} ({v['model']}), whole documents; synthetic M-F "
              f"{v['synthetic_audit_male_minus_female']:+.1f}")
        print(f"  ties {s['pct_ties']:4.1f}%  all {s['tie_aware_pct_male']['overall']:.1f}  "
              + " ".join(f"{g[:4]} {s['tie_aware_pct_male'][g]:.1f}" for g in GROUPS)
              + f"  M-F {s['male_minus_female']:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]  "
              f"|d| {s['mean_abs_delta_in_window']:.3g} (synthetic/companion "
              f"{s['ratio_synthetic_to_companion_in_window']:.1f})  top-3 flips "
              f"{t['flip_rate_pct']:.1f}% M/F {t['male_form_only']}/{t['female_form_only']}")
        for c, g in s["by_construction"].items():
            if g.get("n"):
                print(f"  {c:<15} n {g['n']:>5} ties {g['n_ties']:>4} male-fav "
                      f"{g['tie_aware_pct_male']:5.1f}%  M-F {g['male_minus_female']:+.1f}  "
                      f"flips M/F {g[f'top{DEPTH}']['male_form_only']}"
                      f"/{g[f'top{DEPTH}']['female_form_only']}")
        o = v["own_marker_swaps"]
        print(f"  own-marker swaps: n {o['n']}, male-favored {o['tie_aware_pct_male']:.1f}%, M-F "
              f"{o['male_minus_female']:+.1f}")
        if "by_response_precision" in v:
            print(f"  by response precision: {v['by_response_precision']}")
            b = v["all_scores_rounded_to_bf16"]
            print(f"  all scores rounded to bf16: ties {b['pct_ties']:.1f}%, "
                  f"all {b['tie_aware_pct_male']['overall']:.1f}, M-F {b['male_minus_female']:+.1f}")
        if v.get("counterbalanced"):
            c = v["counterbalanced"]
            print(f"  order: M-F male first {c['male_first']['male_minus_female']:+.1f}, female first "
                  f"{c['female_first']['male_minus_female']:+.1f}, both averaged "
                  f"{c['counterbalanced']['male_minus_female']:+.1f}; same winner in both orders "
                  f"{c['pct_same_winner_both_orders']:.1f}%; synthetic averaged "
                  f"{c['synthetic_counterbalanced_male_minus_female']:+.1f}")
        if v["api"]:
            print(f"  api: {v['api']}")
    print("== overview: synthetic contrast -> whole-document contrast")
    for short, o in sorted(overview.items(), key=lambda kv: -kv[1]["synthetic"]):
        c = o["continuous"]
        print(f"  {short:<18} {o['synthetic']:+6.1f} -> {o['whole_document']:+6.1f} "
              f"[{o['ci'][0]:+.1f}, {o['ci'][1]:+.1f}]  over {c['n_categories']} categories: "
              f"rho {c['spearman_rho']:+.2f} (p {c['permutation_p_two_sided']:.3f}), "
              f"{c['slope_per_10_points_women']:+.1f} pts per +10 pts women"
              + ("" if o["reads_every_resume_whole"] else "  (most résumés still cut)")
              + ("  (released after the audit)" if o.get("post_audit") else "")
              + ("  (both orders averaged)" if o.get("counterbalanced") else ""))
    if correlation:
        print(f"  across {correlation['n_models']} models: Spearman {correlation['spearman_rho']:.2f}, "
              f"Pearson {correlation['pearson_r']:.2f}")
        a = correlation.get("audit_models_only")
        if a:
            print(f"  across the {a['n_models']} audit models only: Spearman {a['spearman_rho']:.2f}, "
                  f"Pearson {a['pearson_r']:.2f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
