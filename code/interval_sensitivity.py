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

Output: results/interval_sensitivity.json
Run:    venv/bin/python code/interval_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import stable_seed  # noqa: E402
from labels import CSV_PATH, load_labels  # noqa: E402

SCRIPT_VERSION = "interval_sensitivity.py v1.0"
RESULTS = ROOT / "results"
OUT = RESULTS / "interval_sensitivity.json"
SCHEMES = ("occupation", "bls_category", "occupation_x_name")


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


def contrast_draws(W, N, cat, clusters, B, rng, resample_names):
    """B bootstrap values of (male-typed share - female-typed share), in points.

    `clusters` maps each occupation index to a cluster id; clusters are
    resampled with replacement within each category.
    """
    n_occ, n_names = W.shape
    out = np.empty(B)
    by_cat = {c: sorted({clusters[i] for i in range(n_occ) if cat[i] == c})
              for c in ("male", "female")}
    members = {}
    for i in range(n_occ):
        members.setdefault(clusters[i], []).append(i)
    for b in range(B):
        w_name = (np.bincount(rng.integers(0, n_names, n_names), minlength=n_names)
                  if resample_names else np.ones(n_names))
        share = {}
        for c in ("male", "female"):
            picks = rng.choice(len(by_cat[c]), size=len(by_cat[c]), replace=True)
            a = np.zeros(n_occ)
            for p in picks:
                a[members[by_cat[c][p]]] += 1
            share[c] = (a @ W @ w_name) / (a @ N @ w_name) * 100
        out[b] = share["male"] - share["female"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Interval sensitivity to the resampling unit")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()
    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        bls = {r["occupation"]: r["bls_title"] for r in csv.DictReader(fh)}

    summary = json.loads((RESULTS / "single_stage_summary.json").read_text())
    occs = sorted(o for o, c in labels.items() if c in ("male", "female"))
    cat = [labels[o] for o in occs]
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
            res["schemes"][scheme] = {"ci": [float(lo), float(hi)], "excludes_zero": bool(lo > 0)}
        models[model] = res
        print(f"{model:<52} {res['contrast']:6.1f}  "
              + "  ".join(f"{s}: [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]"
                          for s, v in res["schemes"].items()), flush=True)

    out = {"script_version": SCRIPT_VERSION,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "label_source": label_source, "bootstrap_B": args.bootstrap,
           "definition": "tie-aware % male-favored, male-typed minus female-typed occupations",
           "n_occupations": len(occs),
           "n_bls_categories": len({bls[o] for o in occs}),
           "n_models_excluding_zero": {s: sum(m["schemes"][s]["excludes_zero"]
                                              for m in models.values()) for s in SCHEMES},
           "models": models}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["n_models_excluding_zero"]))


if __name__ == "__main__":
    main()
