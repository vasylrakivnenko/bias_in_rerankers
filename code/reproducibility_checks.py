"""
Do the same model's scores reproduce across hardware and serving paths?

Two kinds of comparison, each on the audit's full set of counterfactual pairs:

  cross_hardware   one open-weight model scored on two machines with the same
                   code: the RunPod NVIDIA RTX 4090 (CUDA) run in
                   results/partial/ against the Apple M4 Pro (MPS) run in
                   results/deid_pairs/ (its untransformed `original` scores).
  selfhost_vs_api  a model the audit reaches through a vendor API, re-scored
                   from its public weights on our own GPU
                   (results/selfhost/) against the API scores
                   (results/*_full_raw.json).

For each: the largest and mean absolute score difference, the share of pairs
whose winner (male version higher / tie / female version higher) agrees, and
both runs' male-typed minus female-typed contrast. Score scales can differ
between serving paths, so score differences are reported only where the scale
is the same; agreement and contrast are always comparable.

A third block, `vendor_drift`, summarises a comparison already banked by
code/order_reversal_test.py: its October 2026 re-query of every API model in
the original request layout (`same`) against the banked audit scores
(`same_vs_banked`), grouped by the date each model was first scored -- the
February 2026 audit, about eight months before the re-query, or the day of
the re-query itself. Per group: the largest |contrast change|, how many
contrast-change intervals exclude zero (closed interval: an end exactly at
zero does not exclude it), and the range of the share of pairs with the same
winner.

Offline: every input is a banked results file; nothing is scored or re-queried
and no model is loaded.

Output: results/reproducibility_checks.json
Run:    venv/bin/python code/reproducibility_checks.py
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from labels import load_labels  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "reproducibility_checks.json"
ORDER_TEST = RESULTS / "order_reversal_test.json"
AUDIT = RESULTS / "single_stage_summary.json"

CROSS_HARDWARE = {   # model -> (RunPod checkpoint, Mac pair-level file)
    "BAAI/bge-reranker-base": ("partial/BAAI__bge-reranker-base.jsonl",
                               "deid_pairs/BAAI_bge-reranker-base.json"),
}
SELFHOST = {          # self-hosted raw file -> API raw file
    "selfhost/Qwen3-Reranker-8B_full_raw.json": "qwen3-reranker-8b_full_raw.json",
    "selfhost/granite-embedding-reranker-english-r2_full_raw.json":
        "ibm-granite-granite-embedding-reranker-english-r2_full_raw.json",
}


def key(r):
    return f"{r['occupation']}|{r['query_type']}|{r['name_pair']}|{r['template_style']}"


def contrast(pairs, labels):
    share = {}
    for c in ("male", "female"):
        d = np.array([m - f for k, (m, f) in pairs.items() if labels[k.split("|")[0]] == c])
        share[c] = float(np.mean(np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)))) * 100
    return share["male"] - share["female"]


def compare(a, b, labels, same_scale):
    # February API files also hold the 30 occupations the analysis drops; compare
    # on the analysed pairs, all of which each run must cover.
    a = {k: v for k, v in a.items() if k.split("|")[0] in labels}
    b = {k: v for k, v in b.items() if k.split("|")[0] in labels}
    keys = sorted(set(a) & set(b))
    assert len(keys) == len(a) == len(b), "the two runs must cover the same analysed pairs"
    sa = np.array([a[k] for k in keys])
    sb = np.array([b[k] for k in keys])
    win_a = np.sign(sa[:, 0] - sa[:, 1])
    win_b = np.sign(sb[:, 0] - sb[:, 1])
    both = (win_a != 0) & (win_b != 0)
    out = {"n_pairs": len(keys),
           "pct_same_winner": float((win_a == win_b).mean() * 100),
           # ties count as disagreements above; coarse hosted scores tie often,
           # so also report agreement where neither run ties
           "pct_tie_a": float((win_a == 0).mean() * 100),
           "pct_tie_b": float((win_b == 0).mean() * 100),
           "pct_same_winner_excluding_ties": float((win_a[both] == win_b[both]).mean() * 100),
           "n_pairs_neither_ties": int(both.sum()),
           "contrast_a": contrast(a, labels), "contrast_b": contrast(b, labels)}
    if same_scale:
        diff = np.abs(sa - sb)
        out.update({"max_abs_score_diff": float(diff.max()),
                    "mean_abs_score_diff": float(diff.mean())})
    return out


def from_raw(path):
    d = json.loads(path.read_text())
    return {key(r): (r["score_male"], r["score_female"]) for r in d["results"]}, d


def vendor_drift():
    """Summarise the order test's re-query of each API model against its banked scores.

    Reads `models.<m>.comparisons.same_vs_banked` from the order test's summary
    for the API models (access "API" in the single-stage summary) and groups
    them by `date_scored`. Nothing is re-queried.
    """
    ot = json.loads(ORDER_TEST.read_text())
    audit = json.loads(AUDIT.read_text())["models"]
    api = sorted(m for m in ot["models"] if audit[m]["access"] == "API")
    if len(api) != ot["api_overview"]["n_models"]:
        raise AssertionError(f"{len(api)} API models in the order test's summary, but its "
                             f"api_overview counts {ot['api_overview']['n_models']}")
    per_model = {}
    for m in api:
        c = ot["models"][m]["comparisons"]["same_vs_banked"]
        lo, hi = c["contrast_change_ci"]
        scored, requeried = audit[m]["date_scored"], ot["models"][m]["scored"]
        per_model[m] = {
            "date_scored": scored,
            "requeried": requeried,
            "days_between": (_dt.date.fromisoformat(requeried[:10])
                             - _dt.date.fromisoformat(scored)).days,
            "n_pairs": ot["models"][m]["n_pairs"],
            "contrast_change": c["contrast_change"],
            "contrast_change_ci": c["contrast_change_ci"],
            "ci_excludes_zero": lo > 0 or hi < 0,
            "pct_pairs_same_winner": c["pct_pairs_same_winner"]}
    groups = {}
    for date in sorted({v["date_scored"] for v in per_model.values()}):
        g = {m: v for m, v in per_model.items() if v["date_scored"] == date}
        change = {m: abs(v["contrast_change"]) for m, v in g.items()}
        same = {m: v["pct_pairs_same_winner"] for m, v in g.items()}
        excl = sorted(m for m, v in g.items() if v["ci_excludes_zero"])
        groups[date] = {
            "n_models": len(g),
            "models": sorted(g),
            "days_between_scoring_and_requery": sorted({v["days_between"] for v in g.values()}),
            "max_abs_contrast_change": max(change.values()),
            "models_max_abs_contrast_change": sorted(m for m in g
                                                     if change[m] == max(change.values())),
            "n_contrast_change_ci_excluding_zero": len(excl),
            "models_contrast_change_ci_excluding_zero": excl,
            "min_pct_pairs_same_winner": min(same.values()),
            "models_min_pct_pairs_same_winner": sorted(m for m in g
                                                       if same[m] == min(same.values())),
            "max_pct_pairs_same_winner": max(same.values()),
            "models_max_pct_pairs_same_winner": sorted(m for m in g
                                                       if same[m] == max(same.values()))}
    return {
        "source": "results/order_reversal_test.json: models.<m>.comparisons.same_vs_banked",
        "definition": "October 2026 re-query in the original request layout (order test arm "
                      "`same`) against the banked audit scores, on the order test's pairs "
                      f"({ot['per_category']} occupations per category x 4 query phrasings x "
                      "4 templates x 10 name pairs). contrast_change: re-query minus banked "
                      "male-typed minus "
                      "female-typed contrast (tie-aware points), with the order test's 95% "
                      "paired occupation-bootstrap interval; pct_pairs_same_winner: share "
                      "of pairs whose sign of score_male - score_female agrees",
        "interval_rule": "an interval excludes zero when lo > 0 or hi < 0 (closed: an end "
                         "exactly at zero does not exclude it)",
        "group_by": "single_stage_summary.json models.<m>.date_scored",
        "n_models": len(api),
        "n_models_by_date_scored": {d: g["n_models"] for d, g in groups.items()},
        "groups": groups,
        "models": per_model}


def main():
    labels, _ = load_labels()
    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "cross_hardware": {}, "selfhost_vs_api": {}}

    for model, (pod_file, mac_file) in CROSS_HARDWARE.items():
        pod, meta = {}, {}
        for line in (RESULTS / pod_file).read_text().splitlines():
            rec = json.loads(line)
            if "_meta" in rec:
                meta = rec["_meta"]
            else:
                pod.update({key(r): (r["score_male"], r["score_female"]) for r in rec["rows"]})
        mac_d = json.loads((RESULTS / mac_file).read_text())
        mac = {p["key"]: tuple(p["original"]) for p in mac_d["pairs"]}
        assert meta["revision"] == mac_d["model_revision"], "both runs must use the same weights"
        out["cross_hardware"][model] = {"a": f"{meta['gpu']} (CUDA)", "b": "Apple M4 Pro (MPS)",
                                        "revision": meta["revision"],
                                        **compare(pod, mac, labels, same_scale=True)}

    for sh, api in SELFHOST.items():
        if not (RESULTS / sh).exists():
            print(f"  skip {sh}: not scored yet")
            continue
        a, ad = from_raw(RESULTS / sh)
        b, bd = from_raw(RESULTS / api)
        out["selfhost_vs_api"][ad["model"]] = {
            "a": f"self-hosted on {ad['self_hosted']['gpu']}", "b": f"API ({bd['model']})",
            "revision": ad["self_hosted"]["revision"],
            **compare(a, b, labels, same_scale=False)}

    out["vendor_drift"] = vendor_drift()

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "vendor_drift"}, indent=1))
    for date, g in out["vendor_drift"]["groups"].items():
        print(f"vendor drift, first scored {date}: {g['n_models']} models, "
              f"{g['days_between_scoring_and_requery']} days before the re-query; "
              f"max |contrast change| {g['max_abs_contrast_change']:.2f} "
              f"({', '.join(g['models_max_abs_contrast_change'])}); "
              f"{g['n_contrast_change_ci_excluding_zero']} intervals exclude zero; "
              f"same winner {g['min_pct_pairs_same_winner']:.1f}-"
              f"{g['max_pct_pairs_same_winner']:.1f}%")


if __name__ == "__main__":
    main()
