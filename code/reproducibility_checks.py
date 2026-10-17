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
    out = {"n_pairs": len(keys),
           "pct_same_winner": float((win_a == win_b).mean() * 100),
           "contrast_a": contrast(a, labels), "contrast_b": contrast(b, labels)}
    if same_scale:
        diff = np.abs(sa - sb)
        out.update({"max_abs_score_diff": float(diff.max()),
                    "mean_abs_score_diff": float(diff.mean())})
    return out


def from_raw(path):
    d = json.loads(path.read_text())
    return {key(r): (r["score_male"], r["score_female"]) for r in d["results"]}, d


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

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
