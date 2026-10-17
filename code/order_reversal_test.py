"""
Request-order test for the API rerankers (reviewer item 1, FACTCHECK_ROUND3 S4).

The single-stage audit sent each query's 80 documents to an API reranker in
one request laid out as [male_0, female_0, male_1, female_1, ...]: the
male-associated version always came first. Its level statistics therefore
assume that a vendor scores each document independently of where it sits in
the request. This script tests that assumption directly by re-sending the
same requests in three layouts, in the same session:

    same       the original layout, male version first in every pair
    repeat     the identical request again, which measures run-to-run noise
    reversed   the female version first in every pair, same pair order
    shuffled   all 80 documents in a seeded random order

`reversed` and `shuffled` are compared with `same` (position effect), against
the `repeat` vs `same` difference as the noise floor; `same` is compared with
the banked February scores (vendor drift since the audit).

Sample: a seeded, category-balanced subset of the retained occupations
(`--per-category` each of male-, female- and neutral-typed) crossed with all
four query phrasings. Each request reproduces the original document list for
its query, read in the original order from the banked raw file.

Outputs:
    results/order_test/<model>_raw.json      per-pair scores in every arm
    results/order_reversal_test.json         summary across models

Run:  venv/bin/python code/order_reversal_test.py --model rerank-2
Keys: VOYAGE_API_KEY, AZURE_API_KEY (Cohere on Azure AI), OPENROUTER_API_KEY, from the environment
or the repository's git-ignored .env file; Google Vertex AI Ranking uses the
active gcloud login and the billed project in GCP_PROJECT.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import stable_seed  # noqa: E402
from labels import load_labels  # noqa: E402
from synthetic_dataset import generate_dataset  # noqa: E402

SCRIPT_VERSION = "order_reversal_test.py v1.1"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "order_test"
SUMMARY = RESULTS / "order_reversal_test.json"
SEED = 20260822
ARMS = ("same", "repeat", "reversed", "shuffled")
CATEGORIES = ("male", "female", "neutral")

COHERE_ENDPOINT = "https://ai-vasyl-0670.services.ai.azure.com/providers/cohere/v2/rerank"
COHERE_MODELS = ("Cohere-rerank-v4.0-fast", "Cohere-rerank-v4.0-pro")
VOYAGE_MODELS = ("rerank-1", "rerank-2", "rerank-2-lite", "rerank-2.5", "rerank-2.5-lite",
                 "rerank-3", "rerank-3-lite")
GOOGLE_MODELS = ("semantic-ranker-default-002", "semantic-ranker-default-004",
                 "semantic-ranker-fast-004")
OPENROUTER_MODELS = {   # project model name -> OpenRouter id (as in run_experiment.py)
    "cohere/rerank-v3.5": "cohere/rerank-v3.5",
    "qwen/qwen3-reranker-8b": "qwen/qwen3-reranker-8b",
    "nvidia/llama-nemotron-rerank-vl-1b-v2": "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
}


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def key(occ, qt, pair, tmpl):
    return f"{occ}|{qt}|{pair}|{tmpl}"


# ---------------------------------------------------------------------------
# Vendor calls: one request -> one score per document, in input order
# ---------------------------------------------------------------------------

class Throttle:
    """Keep a rolling minute under a token and a request budget.

    The Azure AI deployment allows 250,000 tokens and 250 requests per minute;
    one 80-document request is several thousand tokens, so the token budget
    binds first. Tokens are estimated as characters / 4, which overestimates
    for English and so errs on the safe side.
    """

    def __init__(self, tpm, rpm):
        self.tpm, self.rpm, self.log = tpm, rpm, []

    def wait(self, text_chars):
        need = text_chars / 4
        while True:
            now = time.time()
            self.log = [(t, n) for t, n in self.log if now - t < 60]
            if (sum(n for _, n in self.log) + need <= self.tpm
                    and len(self.log) + 1 <= self.rpm):
                self.log.append((now, need))
                return
            time.sleep(1.0)


# 80% of the Azure AI limits for the Cohere rerank deployments.
AZURE_THROTTLE = Throttle(tpm=200_000, rpm=200)


def make_scorer(model):
    if model in COHERE_MODELS:
        import requests
        headers = {"api-key": os.environ["AZURE_API_KEY"], "Content-Type": "application/json"}

        def score(query, docs):
            AZURE_THROTTLE.wait(len(query) * len(docs) + sum(map(len, docs)))
            payload = {"model": model, "query": query, "documents": docs, "top_n": len(docs)}
            r = requests.post(COHERE_ENDPOINT, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return {x["index"]: float(x["relevance_score"]) for x in r.json()["results"]}
        return score
    if model in VOYAGE_MODELS:
        import voyageai
        vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

        def score(query, docs):
            res = vo.rerank(query, docs, model=model, top_k=len(docs))
            return {x.index: float(x.relevance_score) for x in res.results}
        return score
    if model in OPENROUTER_MODELS:
        import requests
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                   "Content-Type": "application/json"}

        def score(query, docs):
            payload = {"model": OPENROUTER_MODELS[model], "query": query,
                       "documents": docs, "top_n": len(docs)}
            r = requests.post("https://openrouter.ai/api/v1/rerank", json=payload,
                              headers=headers, timeout=120)
            r.raise_for_status()
            got = {x["index"]: float(x["relevance_score"]) for x in r.json()["results"]}
            assert len(got) == len(docs), "OpenRouter dropped documents"
            return got
        return score
    if model in GOOGLE_MODELS:
        import subprocess

        import requests
        project = os.environ["GCP_PROJECT"]
        url = (f"https://discoveryengine.googleapis.com/v1/projects/{project}"
               f"/locations/global/rankingConfigs/default_ranking_config:rank")
        token = {}

        def headers(refresh=False):
            if refresh or "v" not in token:
                token["v"] = subprocess.run(["gcloud", "auth", "print-access-token"],
                                            capture_output=True, text=True,
                                            check=True).stdout.strip()
            return {"Authorization": f"Bearer {token['v']}", "Content-Type": "application/json",
                    "X-Goog-User-Project": project}

        def score(query, docs):
            payload = {"model": model, "query": query,
                       "records": [{"id": str(i), "content": d} for i, d in enumerate(docs)]}
            r = requests.post(url, json=payload, headers=headers(), timeout=60)
            if r.status_code == 401:                      # tokens last an hour
                r = requests.post(url, json=payload, headers=headers(refresh=True), timeout=60)
            r.raise_for_status()
            got = {int(x["id"]): float(x["score"]) for x in r.json()["records"]}
            assert len(got) == len(docs), "the ranking API dropped records"
            return got
        return score
    raise ValueError(f"unsupported model {model!r}")


def call_with_retry(fn, *args, tries=6):
    for attempt in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001  (vendor SDKs raise assorted types)
            if attempt == tries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"    retry in {wait}s after: {str(e)[:120]}", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Sample and requests
# ---------------------------------------------------------------------------

def build_requests(model, labels, per_category):
    raw = json.loads((RESULTS / f"{model.split('/')[-1]}_full_raw.json").read_text())
    assert raw["model"] == model
    texts = {key(t.occupation, t.query_type, t.name_pair, t.template_style):
             (t.doc_male, t.doc_female)
             for t in generate_dataset(perturbation_levels=["names_and_pronouns"],
                                       query_types=None)}

    rng = np.random.default_rng(SEED)
    chosen = []
    for cat in CATEGORIES:
        occs = sorted(o for o, c in labels.items() if c == cat)
        chosen += list(rng.choice(occs, size=min(per_category, len(occs)), replace=False))
    chosen = set(chosen)

    by_query = defaultdict(list)          # original request order is file order
    for r in raw["results"]:
        if r["occupation"] in chosen:
            by_query[r["query"]].append(r)
    reqs = []
    for q, rows in by_query.items():
        pairs = []
        for r in rows:
            k = key(r["occupation"], r["query_type"], r["name_pair"], r["template_style"])
            dm, df = texts[k]
            pairs.append({"key": k, "occupation": r["occupation"],
                          "category": labels[r["occupation"]],
                          "doc_male": dm, "doc_female": df,
                          "banked_male": r["score_male"], "banked_female": r["score_female"]})
        reqs.append((q, pairs))
    return sorted(reqs), sorted(chosen)


def layout(pairs, arm, seed):
    """Documents for one request plus, per pair, (male index, female index)."""
    docs, pos = [], []
    for p in pairs:
        if arm == "reversed":
            docs += [p["doc_female"], p["doc_male"]]
            pos.append((len(docs) - 1, len(docs) - 2))
        else:
            docs += [p["doc_male"], p["doc_female"]]
            pos.append((len(docs) - 2, len(docs) - 1))
    if arm == "shuffled":
        perm = np.random.default_rng(seed).permutation(len(docs))
        where = {old: new for new, old in enumerate(perm)}
        docs = [docs[i] for i in perm]
        pos = [(where[m], where[f]) for m, f in pos]
    return docs, pos


def run_model(model, labels, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{model.replace('/', '_')}_raw.json"
    reqs, chosen = build_requests(model, labels, args.per_category)
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("script_version") == SCRIPT_VERSION and prev.get("occupations") == chosen:
            print(f"  reusing {out.name}")
            return prev
    score = make_scorer(model)
    rows = []
    t0 = time.time()
    for i, (q, pairs) in enumerate(reqs):
        for arm in ARMS:
            docs, pos = layout(pairs, arm, stable_seed(model, q, "shuffle"))
            got = call_with_retry(score, q, docs)
            for p, (im, jf) in zip(pairs, pos):
                p[f"{arm}_male"], p[f"{arm}_female"] = got[im], got[jf]
        rows += [{k: v for k, v in p.items() if not k.startswith("doc_")} | {"query": q}
                 for p in pairs]
        if (i + 1) % 12 == 0:
            print(f"  {model}: {i + 1}/{len(reqs)} queries ({time.time() - t0:.0f}s)", flush=True)
    res = {"script_version": SCRIPT_VERSION, "model": model,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "occupations": chosen, "n_requests_per_arm": len(reqs),
           "docs_per_request": 2 * len(reqs[0][1]), "rows": rows}
    out.write_text(json.dumps(res, indent=1) + "\n")
    return res


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def tie_aware_male(d):
    return float(np.mean(np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)))) * 100


def contrast(rows, src):
    d = {c: np.array([r[f"{src}_male"] - r[f"{src}_female"] for r in rows if r["category"] == c])
         for c in CATEGORIES}
    return tie_aware_male(d["male"]) - tie_aware_male(d["female"])


def paired_contrast_ci(rows, a, b, seed, B):
    occs = sorted({r["occupation"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[r["occupation"]].append(r)
    rng = np.random.default_rng(seed)
    cat = {o: by[o][0]["category"] for o in occs}
    groups = {c: [o for o in occs if cat[o] == c] for c in CATEGORIES}
    diffs = []
    for _ in range(B):
        # resample occupations within each category so every draw has all three
        samp = [r for c in CATEGORIES
                for o in rng.choice(groups[c], size=len(groups[c]), replace=True)
                for r in by[o]]
        diffs.append(contrast(samp, b) - contrast(samp, a))
    return [float(x) for x in np.percentile(diffs, [2.5, 97.5])]


def summarise(res, B):
    rows, model = res["rows"], res["model"]
    s = {"n_pairs": len(rows), "n_occupations": len(res["occupations"]),
         "n_requests_per_arm": res["n_requests_per_arm"],
         "docs_per_request": res["docs_per_request"], "arms": {}}
    for src in ("banked",) + ARMS:
        d = np.array([r[f"{src}_male"] - r[f"{src}_female"] for r in rows])
        s["arms"][src] = {
            "tie_aware_pct_male": tie_aware_male(d),
            "by_category": {c: tie_aware_male(np.array(
                [r[f"{src}_male"] - r[f"{src}_female"] for r in rows if r["category"] == c]))
                for c in CATEGORIES},
            "contrast_male_minus_female": contrast(rows, src)}
    comps = {"repeat_vs_same": ("same", "repeat"), "reversed_vs_same": ("same", "reversed"),
             "shuffled_vs_same": ("same", "shuffled"), "same_vs_banked": ("banked", "same")}
    s["comparisons"] = {}
    for name, (a, b) in comps.items():
        diff = np.array([r[f"{b}_{g}"] - r[f"{a}_{g}"] for r in rows for g in ("male", "female")])
        gap_a = np.array([r[f"{a}_male"] - r[f"{a}_female"] for r in rows])
        gap_b = np.array([r[f"{b}_male"] - r[f"{b}_female"] for r in rows])
        s["comparisons"][name] = {
            "max_abs_score_diff": float(np.abs(diff).max()),
            "mean_abs_score_diff": float(np.abs(diff).mean()),
            "pct_scores_identical": float((diff == 0).mean() * 100),
            "pct_pairs_same_winner": float((np.sign(gap_a) == np.sign(gap_b)).mean() * 100),
            "mean_gap_change": float((gap_b - gap_a).mean()),
            "contrast_change": s["arms"][b]["contrast_male_minus_female"]
                               - s["arms"][a]["contrast_male_minus_female"],
            "contrast_change_ci": paired_contrast_ci(rows, a, b,
                                                     stable_seed(model, name, "contrast"), B)}
    return s


def listwise_from_raw(model, labels):
    """Order arms already scored for a listwise open-weight model.

    code/score_open_models.py scores listwise rerankers (all of a query's
    documents in one pass) in the same four layouts as this test and stores the
    extra arms on each row, so no rescoring is needed. Its own run is both the
    baseline and the `same` arm.
    """
    raw = json.loads((RESULTS / f"{model.split('/')[-1]}_full_raw.json").read_text())
    rows = []
    for r in raw["results"]:
        row = {"occupation": r["occupation"], "category": labels[r["occupation"]],
               "banked_male": r["score_male"], "banked_female": r["score_female"],
               "same_male": r["score_male"], "same_female": r["score_female"]}
        for arm, (m, f) in r["arms"].items():
            row[f"{arm}_male"], row[f"{arm}_female"] = m, f
        rows.append(row)
    return {"model": model, "rows": rows, "generated": raw["scored_at"],
            "occupations": sorted({r["occupation"] for r in rows}),
            "n_requests_per_arm": len({r["query"] for r in raw["results"]}),
            "docs_per_request": 2 * sum(1 for r in raw["results"]
                                        if r["query"] == raw["results"][0]["query"])}


def main():
    ap = argparse.ArgumentParser(description="Request-order test for API rerankers")
    ap.add_argument("--listwise-from-raw", action="append", default=None,
                    help="summarise the stored order arms of a listwise open-weight model "
                         "(e.g. jinaai/jina-reranker-v3) instead of calling an API")
    ap.add_argument("--model", action="append", default=None,
                    help="model to test (repeatable); default: every Cohere, Voyage and "
                         "Google model")
    ap.add_argument("--per-category", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()
    load_env()
    labels, label_source = load_labels(strict=not args.allow_legacy_labels)

    if args.listwise_from_raw:
        todo = [(m, lambda m=m: listwise_from_raw(m, labels)) for m in args.listwise_from_raw]
    else:
        todo = [(m, lambda m=m: run_model(m, labels, args))
                for m in args.model or (COHERE_MODELS + VOYAGE_MODELS + GOOGLE_MODELS
                                        + tuple(OPENROUTER_MODELS))]
    for model, get in todo:
        print(f"== {model}", flush=True)
        res = get()
        # re-read just before writing, so runs for different models can overlap
        summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {"models": {}}
        summary["models"][model] = summarise(res, args.bootstrap) | {
            "scored": res["generated"]}
        summary.update({"script_version": SCRIPT_VERSION, "label_source": label_source,
                        "seed": SEED, "bootstrap_B": args.bootstrap,
                        "per_category": args.per_category,
                        "generated": _dt.datetime.now().isoformat(timespec="seconds")})
        api = {m: v for m, v in summary["models"].items()
               if m in COHERE_MODELS + VOYAGE_MODELS + GOOGLE_MODELS + tuple(OPENROUTER_MODELS)}

        def moved(v, arm):
            lo, hi = v["comparisons"][f"{arm}_vs_same"]["contrast_change_ci"]
            return lo > 0 or hi < 0
        summary["api_overview"] = {
            "n_models": len(api),
            "n_pairs_per_model": sorted({v["n_pairs"] for v in api.values()}),
            "models_contrast_moved_by_reversal": sorted(m for m, v in api.items() if moved(v, "reversed")),
            "models_contrast_moved_by_shuffle": sorted(m for m, v in api.items() if moved(v, "shuffled")),
            "n_models_unmoved": sum(not moved(v, "reversed") and not moved(v, "shuffled")
                                    for v in api.values())}
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
        c = summary["models"][model]["comparisons"]
        print(f"  repeat vs same:   max|diff| {c['repeat_vs_same']['max_abs_score_diff']:.2e}, "
              f"same winner {c['repeat_vs_same']['pct_pairs_same_winner']:.1f}%", flush=True)
        print(f"  reversed vs same: max|diff| {c['reversed_vs_same']['max_abs_score_diff']:.2e}, "
              f"same winner {c['reversed_vs_same']['pct_pairs_same_winner']:.1f}%, "
              f"contrast change {c['reversed_vs_same']['contrast_change']:+.2f} "
              f"{c['reversed_vs_same']['contrast_change_ci']}", flush=True)
        print(f"  same vs banked:   max|diff| {c['same_vs_banked']['max_abs_score_diff']:.2e}, "
              f"contrast change {c['same_vs_banked']['contrast_change']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
