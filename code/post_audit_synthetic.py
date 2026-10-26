"""
The synthetic audit's pairs, scored by a hosted reranker released after the audit.

The audit's scores (results/*_full_raw.json, February 2026) are banked and never re-run,
and analyze_single_stage.py reads every results/*_full_raw.json, so those files define its
29 models. A reranker released later -- Cohere's Rerank 5, 2026-10-21 -- is scored here on
the 8,320 pairs the audit's statistics use: the 52 occupations with a BLS label, which is
all generate_dataset now yields (the February run also scored 30 occupations the labels
drop). They are regenerated with the audit's own call (the names_and_pronouns perturbation,
all four query types) and checked, pair by pair and in order, against the banked file of the
vendor's previous model; the stereotype field is the BLS label, not the banked legacy one.
As run_experiment.py did, each query is one request holding its 80 documents, both forms of
a pair together and the male form first -- the same requests the audit sent for these 208
queries.

Requests go through realistic_pool_experiment.APIScorer: throttled, retried, each response
checked (one finite score per document, not all equal) and saved to a checkpoint as it
lands, so an interrupted run resumes where it stopped; a ledger beside the checkpoint counts
every request sent. A canary (--limit-queries N) sends the first N of the same requests,
which the full run then reuses.

The output keeps the raw files' row schema under another name,
results/post_audit/<model>_synthetic.json, so no audit script picks it up. It also holds the
audit's headline contrast -- tie-aware % male-favored on the BLS male-typed occupations minus
the same on the female-typed ones -- for the new model and, as a check of this computation,
for the reference model, which must reproduce its contrast in results/interval_sensitivity.json.

Run:  venv/bin/python code/post_audit_synthetic.py --model cohere-v5-pro [--resume]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from labels import load_labels  # noqa: E402
from realistic_pool_experiment import API_RERANKERS, APIScorer  # noqa: E402
from synthetic_dataset import QUERY_TYPES, generate_dataset  # noqa: E402

SCRIPT_VERSION = "post_audit_synthetic.py v1.0"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "post_audit"
PARTIAL = RESULTS / "partial"
PERTURBATION = ["names_and_pronouns"]
# The banked audit model each new one is checked against: same pairs, same order
REFERENCE = {"cohere-v5-pro": "Cohere-rerank-v4.0-pro",
             "cohere-v5-fast": "Cohere-rerank-v4.0-fast"}
IDENTITY = ("query", "query_type", "occupation", "name_pair", "template_style", "perturbation")
GROUPS = ("male", "female", "neutral")


def tie_aware(deltas):
    """Tie-aware % male-favored: ties count half."""
    return 100 * (sum(d > 0 for d in deltas) + 0.5 * sum(d == 0 for d in deltas)) / len(deltas)


def headline(rows, labels):
    """The audit's statistics over the retained (BLS-labelled) occupations."""
    kept = [r for r in rows if r["occupation"] in labels]
    by = {g: [r["delta"] for r in kept if labels[r["occupation"]] == g] for g in GROUPS}
    share = {g: tie_aware(d) for g, d in by.items() if d}
    scores = [s for r in kept for s in (r["score_male"], r["score_female"])]
    return {"n_pairs": len(kept), "n_by_group": {g: len(d) for g, d in by.items()},
            "tie_aware_pct_male": {"overall": tie_aware([r["delta"] for r in kept]), **share},
            "contrast": share["male"] - share["female"],
            "tie_rate_pct": 100 * sum(r["delta"] == 0 for r in kept) / len(kept),
            "n_distinct_scores": len(set(scores)), "score_range": [min(scores), max(scores)],
            "mean_abs_delta": sum(abs(r["delta"]) for r in kept) / len(kept)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(REFERENCE))
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="canary: score only the first N queries (their requests are reused)")
    ap.add_argument("--resume", action="store_true", help="reuse finished requests")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check the pairs and requests, send nothing")
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels, label_source = load_labels(strict=True)
    triples = generate_dataset(PERTURBATION, list(QUERY_TYPES))
    ref_name = REFERENCE[args.model]
    ref = json.loads((RESULTS / f"{ref_name}_full_raw.json").read_text())
    if ref["perturbation_levels"] != PERTURBATION or ref["query_types"] != list(QUERY_TYPES):
        raise AssertionError(f"{ref_name} was scored on another design")
    ref_rows = [r for r in ref["results"] if r["occupation"] in labels]
    if [tuple(getattr(t, k) for k in IDENTITY) for t in triples] != \
            [tuple(r[k] for k in IDENTITY) for r in ref_rows]:
        raise AssertionError(f"the regenerated pairs are not {ref_name}'s, in its order")
    if any(t.stereotype != labels[t.occupation] for t in triples):
        raise AssertionError("a regenerated pair does not carry its BLS label")
    dataset_sha = hashlib.sha256(json.dumps(
        [[getattr(t, k) for k in IDENTITY] + [t.doc_male, t.doc_female] for t in triples]
    ).encode()).hexdigest()
    # the contrast computed here must be the audit's, checked on the reference model
    ref_check = headline(ref_rows, labels)
    banked = json.loads((RESULTS / "interval_sensitivity.json").read_text())["models"]
    if abs(ref_check["contrast"] - banked[ref_name]["contrast"]) > 1e-9:
        raise AssertionError(f"{ref_name}: contrast {ref_check['contrast']} here, "
                             f"{banked[ref_name]['contrast']} in interval_sensitivity.json")

    queries = list(dict.fromkeys(t.query for t in triples))
    if args.limit_queries:
        keep = set(queries[:args.limit_queries])
        todo = [(i, t) for i, t in enumerate(triples) if t.query in keep]
    else:
        todo = list(enumerate(triples))
    keys, pairs = [], []
    for i, t in todo:
        for form, doc in (("male", t.doc_male), ("female", t.doc_female)):
            keys.append((t.query, i, form))
            pairs.append((t.query, doc))

    PARTIAL.mkdir(parents=True, exist_ok=True)
    ckpt = PARTIAL / f"post_audit_synthetic.{args.model}.scores.jsonl"
    # one request per query, its 80 documents in the audit's order (no token cap)
    scorer = APIScorer(args.model, docs_per_request=80, tokens_per_request=10 ** 9,
                       max_requests=args.max_requests, coarse_retries=8,
                       checkpoint=None if args.dry_run else ckpt,
                       checkpoint_meta={"experiment": SCRIPT_VERSION.split()[0],
                                        "dataset_sha256": dataset_sha},
                       resume=args.resume)
    plan = scorer.plan_requests(list(zip(keys, pairs)))
    expected = [[k for k in keys if k[0] == q] for q in dict.fromkeys(k[0] for k in keys)]
    if [[k for k, _ in req] for req in plan] != expected or any(len(r) != 80 for r in plan):
        raise AssertionError("the requests are not the audit's: one per query, 80 documents, "
                             "pairs in order, the male form first")
    print(f"{args.model} ({scorer.model_name}): {len(todo):,} pairs, {len(plan)} requests; "
          f"checked against {ref_name}", flush=True)
    if args.dry_run:
        print(f"dry run: pairs and requests match {ref_name}; its contrast here "
              f"{ref_check['contrast']:.6f}, as banked; dataset {dataset_sha[:16]}; nothing sent")
        return
    t0 = time.time()
    scorer.score(pairs, keys)
    elapsed = time.time() - t0

    rows = []
    for i, t in todo:
        sm, sf = float(scorer.cache[(t.query, i, "male")]), float(scorer.cache[(t.query, i, "female")])
        rows.append({**{k: getattr(t, k) for k in IDENTITY}, "stereotype": t.stereotype,
                     "score_male": sm, "score_female": sf, "delta": sm - sf})
    out = {"model": scorer.model_name, "short": args.model,
           "script_version": SCRIPT_VERSION,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "scored_at": _dt.date.today().isoformat(),
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "canary_limit_queries": args.limit_queries or None,
           "perturbation_levels": PERTURBATION, "query_types": list(QUERY_TYPES),
           "dataset_sha256": dataset_sha, "n_triples": len(rows),
           "api": {"vendor": scorer.vendor, "endpoint": getattr(scorer, "_endpoint", None),
                   "settings": scorer.settings, "documents_per_request": 80,
                   "billed_unit": scorer.billed_unit, **scorer.ledger},
           "elapsed_seconds": elapsed, "label_source": label_source}
    if not args.limit_queries:
        agree = sum((r["delta"] > 0) - (r["delta"] < 0) == (q["delta"] > 0) - (q["delta"] < 0)
                    for r, q in zip(rows, ref_rows, strict=True))
        out["headline"] = headline(rows, labels)
        out["reference"] = {"model": ref_name, "headline": ref_check,
                            "pct_same_winner": 100 * agree / ref_check["n_pairs"]}
    else:
        out["canary"] = {"tie_aware_pct_male": tie_aware([r["delta"] for r in rows]),
                         "n_ties": sum(r["delta"] == 0 for r in rows)}
    out["results"] = rows
    path = args.out or (PARTIAL / f"{scorer.model_name}_synthetic.canary.json"
                        if args.limit_queries else OUT_DIR / f"{scorer.model_name}_synthetic.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1)[:4000])
    print(f"written {path}")


if __name__ == "__main__":
    main()
