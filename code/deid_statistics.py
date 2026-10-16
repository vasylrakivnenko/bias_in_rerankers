"""
Paired statistics behind the de-identification sentences of the empirical paper.

`deidentification_experiment.py` reports each condition on its own: a share, a
mean |score gap| and an occupation-clustered interval for each. Three sentences
in the paper compare conditions, and a comparison needs a paired statistic
(FACTCHECK_ROUND3 S2, S3b, S3c):

  gap_vs_original   For `names_only` and `pronouns_only` on bge-reranker-v2-m3:
                    the difference in mean |score gap| against the untransformed
                    text, with a PAIRED occupation-clustered bootstrap (both
                    conditions resampled on the same occupations in each draw).

  utility           Top-1 occupation matching, `original` vs `full`: the counts
                    behind the two percentages and an exact two-sided McNemar test
                    on the ranking events whose outcome changed.

  distinct          How many distinct (query, male doc, female doc) comparisons
                    each condition has. Rules that replace every name with one
                    placeholder collapse the ten name pairs into one text.

It re-scores only what those statistics need, locally, and asserts that its
recomputed per-condition values equal the ones already published in
`results/deidentification_bge.json`, so it cannot silently disagree with the
tables.

Output (never touches any other results file):
    results/deid_statistics_bge.json

Run:  venv/bin/python code/deid_statistics.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import stable_seed  # noqa: E402
from deidentification_experiment import (  # noqa: E402
    RESULTS, SCORED_CONDITIONS, SEED, baseline_scores, build_texts, key,
    score_pairs, utility_test)
from labels import load_labels  # noqa: E402
from synthetic_dataset import QUERY_TYPES, generate_dataset  # noqa: E402

SCRIPT_VERSION = "deid_statistics.py v1.0 (FACTCHECK_ROUND3 S2/S3b/S3c)"
MODEL = "BAAI/bge-reranker-v2-m3"
PAIRED_CONDITIONS = ("names_only", "pronouns_only")
OUT = RESULTS / "deid_statistics_bge.json"


def paired_gap_bootstrap(per_occ, cond, seed, B):
    """Difference in mean |gap| (cond - original), occupation-clustered, paired."""
    occs = sorted(per_occ)
    s0 = np.array([per_occ[o]["original"][0] for o in occs])
    s1 = np.array([per_occ[o][cond][0] for o in occs])
    n = np.array([per_occ[o]["original"][1] for o in occs], dtype=float)
    assert np.array_equal(n, [per_occ[o][cond][1] for o in occs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(occs), size=(B, len(occs)))
    nb = n[idx].sum(1)
    diff = s1[idx].sum(1) / nb - s0[idx].sum(1) / nb
    point = s1.sum() / n.sum() - s0.sum() / n.sum()
    return {"original_mean_abs_gap": float(s0.sum() / n.sum()),
            "condition_mean_abs_gap": float(s1.sum() / n.sum()),
            "difference": float(point),
            "difference_ci": [float(x) for x in np.percentile(diff, [2.5, 97.5])],
            "share_of_draws_above_zero": float((diff > 0).mean()),
            "bootstrap_B": B,
            "n_occupations": len(occs)}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description="Paired statistics for the de-identification text")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    published = json.loads((RESULTS / "deidentification_bge.json").read_text())
    assert published["reranker"] == MODEL and published["label_source"] == label_source
    triples = generate_dataset(perturbation_levels=["names_and_pronouns"], query_types=None)
    rows, _ = build_texts(triples, labels)

    # ---- distinct comparisons per condition (no scoring needed) ----
    distinct = {"original": len({(r["t"].query, r["t"].doc_male, r["t"].doc_female)
                                 for r in rows})}
    for cond in SCORED_CONDITIONS:
        distinct[cond] = len({r["cond"][cond] for r in rows})

    # ---- paired mean |gap| ----
    base, base_src = baseline_scores(MODEL, rows, device)
    need = {}
    for r in rows:
        for cond in PAIRED_CONDITIONS:
            q, dm, df = r["cond"][cond]
            need[(q, dm)] = None
            need[(q, df)] = None
    sc = score_pairs(MODEL, list(need), device)

    per_occ = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for r in rows:
        t = r["t"]
        sm, sf = base[key(t.occupation, t.query_type, t.name_pair, t.template_style)]
        cell = per_occ[t.occupation]["original"]
        cell[0] += abs(sm - sf); cell[1] += 1
        for cond in PAIRED_CONDITIONS:
            q, dm, df = r["cond"][cond]
            cell = per_occ[t.occupation][cond]
            cell[0] += abs(sc[(q, dm)] - sc[(q, df)]); cell[1] += 1

    gap = {}
    for cond in PAIRED_CONDITIONS:
        g = paired_gap_bootstrap(per_occ, cond, stable_seed(MODEL, cond, "paired_gap"),
                                 args.bootstrap)
        for which, pub_cond in (("original_mean_abs_gap", "original"),
                                ("condition_mean_abs_gap", cond)):
            pub = published["conditions"][pub_cond]["mean_abs_delta"]
            assert abs(g[which] - pub) < 1e-6, (
                f"{cond}: recomputed {which} {g[which]:.8f} != published {pub:.8f}; "
                "the local re-score does not reproduce deidentification_bge.json")
        gap[cond] = g

    # ---- utility: original vs full, paired by ranking event ----
    u = utility_test(MODEL, triples, labels, device, list(QUERY_TYPES.keys()),
                     ("original", "full"), return_hits=True)
    h0, h1 = u["original"]["hits"], u["full"]["hits"]
    assert [x[:3] for x in h0] == [x[:3] for x in h1]
    for cond in ("original", "full"):
        pub = published["utility"][cond]["top1_pct"]
        assert abs(u[cond]["top1_pct"] - pub) < 1e-9, (
            f"utility {cond}: recomputed {u[cond]['top1_pct']} != published {pub}")
    ok0 = [x[3] == 1 for x in h0]
    ok1 = [x[3] == 1 for x in h1]
    b = sum(a and not c for a, c in zip(ok0, ok1))   # right before, wrong after
    c = sum(c and not a for a, c in zip(ok0, ok1))   # wrong before, right after
    utility = {"n_ranking_events": len(h0),
               "pool_size": u["config"]["pool_size"],
               "top1_correct_original": sum(ok0),
               "top1_correct_full": sum(ok1),
               "top3_pct_original": u["original"]["top3_pct"],
               "top3_pct_full": u["full"]["top3_pct"],
               "discordant_right_before_wrong_after": b,
               "discordant_wrong_before_right_after": c,
               "mcnemar_exact_p_two_sided": mcnemar_exact(b, c)}

    out = {"script_version": SCRIPT_VERSION,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "reranker": MODEL, "device": device, "baseline_source": base_src,
           "label_source": label_source, "seed": SEED,
           "gap_vs_original": gap, "utility": utility,
           "distinct_comparisons": distinct, "n_rows": len(rows)}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("gap_vs_original", "utility", "distinct_comparisons")},
                     indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
