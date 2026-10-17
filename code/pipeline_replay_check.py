"""
Replay-validity check for the two-stage pipeline simulation (reviewer finding).

code/compounding_experiment.py simulates retriever -> reranker pipelines over
20-document pools (10 male-associated + 10 female-associated versions of one
occupation's bios).  For the API rerankers it does not call the API: it REPLAYS
the scores banked by the single-stage audit, where every document was scored
inside an 80-document request.  A deployed pipeline would send the reranker
only the 10 documents the retriever kept.  If a vendor's score for a document
depends on what else is in the request (listwise normalisation, batch-relative
calibration, a different serving path for short requests), the replayed top-3
is not what the pipeline would actually return.

This script tests that directly.  For a seeded sample of pools it

  1. takes the retriever's top-10 for the pool from the cached bge-large scores
     (results/compounding_emb_cache.json) using the simulation's own rule
     (compounding_experiment.build_slates + _order; retriever scores have no
     ties in any of the 832 pools, so the top-10 is deterministic),
  2. sends exactly those 10 documents, in retriever-rank order, to the API as
     one request,
  3. compares the top-3 implied by those live scores with the top-3 implied by
     the banked scores for the same 10 documents.

Male-associated share of the top-3 is computed with
compounding_experiment.analyse_slate itself (seeded random tie-breaks averaged
over `--repeats` draws), for the replayed and the live slate with the SAME
seed, so the two arms see identical tie-break draws and differ only in scores.

Sample: every retained occupation (52) contributes `--per-occupation` pools
(default 2, so 104 pools); for each occupation the template styles and query
phrasings are drawn without replacement, so no occupation repeats a template
or a phrasing.  Occupation weights are therefore equal, as in the sweep.  The
same pools are used for every model.

Outputs:
    results/pipeline_replay_partial/<model>.jsonl         one line per request (resume)
    results/pipeline_replay_partial/<model>.status.json   ok / skipped + reason
    results/pipeline_replay_check.json                    summary across models

Run (one process per vendor, vendors in parallel; each vendor sequential):
    venv/bin/python code/pipeline_replay_check.py --vendor voyage
    venv/bin/python code/pipeline_replay_check.py --vendor azure
    venv/bin/python code/pipeline_replay_check.py --vendor google
    venv/bin/python code/pipeline_replay_check.py --vendor openrouter
    venv/bin/python code/pipeline_replay_check.py --summarise-only
Keys: as code/order_reversal_test.py (.env at the repo root; gcloud login for Vertex).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import load_raw, stable_seed  # noqa: E402
from compounding_experiment import (  # noqa: E402  (reuse the simulation's own rules)
    HEADLINE_SHORTLIST, HEADLINE_TOPK, _order, analyse_slate, build_slates,
    embedding_scores, key,
)
from labels import load_labels  # noqa: E402
from order_reversal_test import (  # noqa: E402  (vendor request code)
    COHERE_MODELS, GOOGLE_MODELS, OPENROUTER_MODELS, VOYAGE_MODELS, load_env, make_scorer,
)
from synthetic_dataset import generate_dataset  # noqa: E402

SCRIPT_VERSION = "pipeline_replay_check.py v1.0"
RESULTS = ROOT / "results"
PARTIAL = RESULTS / "pipeline_replay_partial"
OUT = RESULTS / "pipeline_replay_check.json"
SWEEP = RESULTS / "compounding_all_rerankers.json"
ORDER_TEST = RESULTS / "order_reversal_test.json"
SEED = 20260822
CATEGORIES = ("male", "female", "neutral")
K, S = HEADLINE_TOPK, HEADLINE_SHORTLIST

VENDORS = {
    "voyage": tuple(VOYAGE_MODELS),
    "azure": tuple(COHERE_MODELS),
    "google": tuple(GOOGLE_MODELS),
    "openrouter": tuple(OPENROUTER_MODELS),
}
UNTESTED = {
    "ibm-granite-granite-embedding-reranker-english-r2":
        "not reachable: the Azure ML online-endpoint URL used for the audit is not on record, "
        "so no request can be sent; its sweep value (the sweep's upper endpoint) is unvalidated",
}


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def draw_sample(labels, slate_keys, per_occ):
    """[(occ, template, query_type, category)], seeded, equal pools per occupation."""
    by_occ = defaultdict(list)
    for sk in slate_keys:
        by_occ[sk[0]].append(sk)
    rng = np.random.default_rng(SEED)
    out = []
    for occ in sorted(by_occ):
        tmpls = sorted({sk[1] for sk in by_occ[occ]})
        qts = sorted({sk[2] for sk in by_occ[occ]})
        n = min(per_occ, len(tmpls), len(qts))
        ts = rng.choice(tmpls, size=n, replace=False)
        qs = rng.choice(qts, size=n, replace=False)
        for t, q in zip(ts, qs):
            sk = (occ, str(t), str(q), labels[occ])
            assert sk in slate_keys, sk
            out.append(sk)
    return out


def sample_hash(sample):
    return hashlib.sha256(json.dumps(sample).encode()).hexdigest()[:16]


def pool_id(sk):
    return "|".join(sk[:3])


# ---------------------------------------------------------------------------
# Context shared by every model: triples, texts, retriever scores, pools
# ---------------------------------------------------------------------------

class Context:
    def __init__(self, args):
        self.labels, self.label_source = load_labels(strict=not args.allow_legacy_labels)
        self.triples = generate_dataset(perturbation_levels=["names_and_pronouns"],
                                        query_types=None)
        self.emb = embedding_scores(self.triples, refresh=False)
        self.raw = load_raw()
        self.rr_maps = {
            name: {key(r["occupation"], r["query_type"], r["name_pair"], r["template_style"]):
                   (r["score_male"], r["score_female"]) for r in meta["rows"]}
            for name, meta in self.raw.items()}
        # Per-pool document texts and query, in build_slates' bookkeeping order
        # (male, female per name pair), so index i here is index i in the slate.
        self.pool_docs, self.pool_query, self.pool_pairs = (defaultdict(list), {},
                                                            defaultdict(list))
        for t in self.triples:
            cat = self.labels.get(t.occupation)
            if cat is None:
                continue
            sk = (t.occupation, t.template_style, t.query_type, cat)
            self.pool_docs[sk] += [t.doc_male, t.doc_female]
            self.pool_pairs[sk] += [t.name_pair, t.name_pair]
            q = self.pool_query.setdefault(sk, t.query)
            assert q == t.query, "one query per pool"
        any_model = next(iter(self.rr_maps))
        self.slate_keys = set(build_slates(self.triples, self.emb, self.rr_maps[any_model],
                                           self.labels))
        self.sample = draw_sample(self.labels, self.slate_keys, args.per_occupation)
        self.sample_hash = sample_hash(self.sample)

    def slates(self, model):
        allsl = build_slates(self.triples, self.emb, self.rr_maps[model], self.labels)
        out = {sk: allsl[sk] for sk in self.sample}
        assert all(len(s["is_male"]) == 2 * K for s in out.values()), "pool size"
        return out

    def retrieved(self, s):
        """Retriever top-K indices, best first (no retriever ties exist; asserted)."""
        assert len(set(s["emb"].tolist())) == len(s["emb"]), "retriever tie in pool"
        return _order(s["emb"], np.arange(len(s["emb"])))[:K]


# ---------------------------------------------------------------------------
# Requests with checkpointing, canary and backoff
# ---------------------------------------------------------------------------

class Transient(Exception):
    pass


def _status_code(e):
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    if code is None:
        code = getattr(e, "http_status", None) or getattr(e, "status_code", None)
    return code


def is_transient(e):
    code = _status_code(e)
    if code in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    name = type(e).__name__
    return any(s in name for s in ("RateLimit", "ServiceUnavailable", "ServerError",
                                   "APIConnection", "Timeout", "ConnectionError",
                                   "ChunkedEncoding"))


def call_with_backoff(fn, *a, tries=8, base=5.0, cap=180.0):
    for attempt in range(tries):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001  (vendor SDKs raise assorted types)
            code = _status_code(e)
            if not is_transient(e) or attempt == tries - 1:
                raise RuntimeError(f"{type(e).__name__} (HTTP {code}): {str(e)[:300]}") from e
            wait = min(cap, base * 2 ** attempt)
            print(f"    transient {type(e).__name__} (HTTP {code}); retry {attempt + 1}/"
                  f"{tries - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def fname(model):
    return model.replace("/", "_")


def read_jsonl(path, shash):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sample_hash") != shash:
                raise SystemExit(f"{path} was written for a different sample "
                                 f"({rec.get('sample_hash')} != {shash}); move it aside")
            done[rec["pool_id"]] = rec
    return done


def write_status(model, status):
    PARTIAL.mkdir(parents=True, exist_ok=True)
    (PARTIAL / f"{fname(model)}.status.json").write_text(json.dumps(status, indent=1) + "\n")


def canary_ok(scores):
    v = np.asarray(scores, dtype=float)
    return bool(np.all(np.isfinite(v)) and len(np.unique(v)) > 1)


def score_model(model, ctx, args):
    PARTIAL.mkdir(parents=True, exist_ok=True)
    path = PARTIAL / f"{fname(model)}.jsonl"
    done = read_jsonl(path, ctx.sample_hash)
    todo = [sk for sk in ctx.sample if pool_id(sk) not in done]
    print(f"== {model}: {len(done)} pools banked, {len(todo)} to request", flush=True)
    if not todo:
        write_status(model, {"model": model, "status": "ok", "n_pools": len(done),
                             "sample_hash": ctx.sample_hash,
                             "finished": _dt.datetime.now().isoformat(timespec="seconds")})
        return
    try:
        score = make_scorer(model)
    except Exception as e:  # noqa: BLE001
        write_status(model, {"model": model, "status": "skipped",
                             "reason": f"could not build client: {type(e).__name__}: {e}"[:400]})
        print(f"   SKIP {model}: client error", flush=True)
        return
    slates = ctx.slates(model)
    # The canary covers the first three requests of the sample; if they are already
    # banked from an earlier run they are re-checked from the checkpoint.
    canary_ids = [pool_id(sk) for sk in ctx.sample[:3]]
    for pid in canary_ids:
        if pid in done and not canary_ok(done[pid]["actual"]):
            write_status(model, {"model": model, "status": "skipped",
                                 "reason": f"canary failed on banked request {pid}"})
            print(f"   SKIP {model}: canary failed (checkpoint)", flush=True)
            return
    t0 = time.time()
    n_new = 0
    with open(path, "a") as fh:
        for sk in todo:
            pid = pool_id(sk)
            s = slates[sk]
            ret = ctx.retrieved(s)
            docs = [ctx.pool_docs[sk][i] for i in ret]
            q = ctx.pool_query[sk]
            try:
                got = call_with_backoff(score, q, docs)
            except Exception as e:  # noqa: BLE001
                write_status(model, {"model": model, "status": "skipped",
                                     "reason": f"persistent error after retries: {e}"[:600],
                                     "n_pools_banked": len(done) + n_new})
                print(f"   SKIP {model}: {str(e)[:200]}", flush=True)
                return
            actual = [float(got[i]) for i in range(len(docs))]
            if pid in canary_ids and not canary_ok(actual):
                write_status(model, {"model": model, "status": "skipped",
                                     "reason": f"canary failed: request {pid} returned "
                                               f"non-finite or constant scores {actual}"})
                print(f"   SKIP {model}: canary failed", flush=True)
                return
            rec = {"model": model, "pool_id": pid, "occupation": sk[0], "template_style": sk[1],
                   "query_type": sk[2], "category": sk[3], "query": q,
                   "sample_hash": ctx.sample_hash,
                   "retrieved_slate_idx": [int(i) for i in ret],
                   "is_male": [bool(s["is_male"][i]) for i in ret],
                   "name_pair": [ctx.pool_pairs[sk][i] for i in ret],
                   "banked": [float(s["rr"][i]) for i in ret],
                   "actual": actual,
                   "requested_at": _dt.datetime.now().isoformat(timespec="seconds")}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_new += 1
            if pid == canary_ids[-1]:
                print(f"   canary passed for {model}", flush=True)
            if n_new % 20 == 0 or n_new == len(todo):
                print(f"   {model}: {len(done) + n_new}/{len(ctx.sample)} pools "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    write_status(model, {"model": model, "status": "ok", "n_pools": len(done) + n_new,
                         "sample_hash": ctx.sample_hash,
                         "finished": _dt.datetime.now().isoformat(timespec="seconds")})


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def pool_metrics(model, rec, s, args):
    """Replayed vs live top-3 for one pool, both via analyse_slate with one seed."""
    ret = np.asarray(rec["retrieved_slate_idx"])
    assert np.allclose(s["rr"][ret], rec["banked"]), "banked scores changed since request"
    live = dict(s)
    live["rr"] = s["rr"].astype(float).copy()
    live["rr"][ret] = rec["actual"]           # only the retrieved docs are re-scored
    seed = stable_seed(model, rec["pool_id"], "replay-check")
    rep = analyse_slate(s, K, (S,), np.random.default_rng(seed), args.repeats)
    act = analyse_slate(live, K, (S,), np.random.default_rng(seed), args.repeats)
    # Set identity under the same tie-break draws (analyse_slate's own draw sequence).
    rng = np.random.default_rng(seed)
    same3 = same1 = 0.0
    for _ in range(args.repeats):
        tb = rng.permutation(len(s["is_male"]))
        r_ret = _order(s["emb"], tb)[:K]
        assert set(r_ret) == set(ret), "retriever top-K is deterministic"
        a = r_ret[_order(s["rr"][r_ret], tb[r_ret])]
        b = r_ret[_order(live["rr"][r_ret], tb[r_ret])]
        same3 += set(a[:S]) == set(b[:S])
        same1 += a[0] == b[0]
    banked, actual = np.asarray(rec["banked"]), np.asarray(rec["actual"])
    rho = spearmanr(banked, actual)[0] if len(set(banked)) > 1 and len(set(actual)) > 1 \
        else float("nan")
    return {"occupation": rec["occupation"], "category": rec["category"],
            "replay_m": rep[f"pipe_top{S}_m"] / S * 100, "actual_m": act[f"pipe_top{S}_m"] / S * 100,
            "same_top3": same3 / args.repeats, "same_top1": same1 / args.repeats,
            "abs_diff": np.abs(actual - banked), "rho": rho, "banked": banked, "actual": actual}


def cluster_boot(rows, fn, seed, B):
    """Occupation-clustered bootstrap, occupations resampled within category."""
    by = defaultdict(list)
    for r in rows:
        by[r["occupation"]].append(r)
    groups = {c: sorted(o for o in by if by[o][0]["category"] == c) for c in CATEGORIES}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        samp = [r for c in CATEGORIES if groups[c]
                for o in rng.choice(groups[c], size=len(groups[c]), replace=True) for r in by[o]]
        vals.append(fn(samp))
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def summarise_model(model, ctx, recs, args, sweep, order_test):
    slates = ctx.slates(model)
    by_pid = {pool_id(sk): sk for sk in ctx.sample}
    rows = [pool_metrics(model, rec, slates[by_pid[rec["pool_id"]]], args) for rec in recs]
    mean = lambda f: (lambda rs: float(np.mean([r[f] for r in rs])))  # noqa: E731
    diff = lambda rs: float(np.mean([r["actual_m"] - r["replay_m"] for r in rs]))  # noqa: E731
    B = args.bootstrap
    seed = lambda tag: stable_seed(model, "replay-check", tag)  # noqa: E731
    d = diff(rows)
    d_ci = cluster_boot(rows, diff, seed("diff"), B)
    banked = np.concatenate([r["banked"] for r in rows])
    actual = np.concatenate([r["actual"] for r in rows])
    absd = np.concatenate([r["abs_diff"] for r in rows])
    slope, intercept = np.polyfit(banked, actual, 1)
    pearson = float(np.corrcoef(banked, actual)[0, 1])
    # "Same scale" = the live scores sit on the banked scale: a near-identity linear
    # relation.  Only then is a raw mean |score difference| meaningful.
    scale_match = bool(0.8 <= slope <= 1.25 and pearson >= 0.9
                       and abs(np.median(actual) - np.median(banked)) <= 0.1 * (np.ptp(banked) or 1))
    sw = sweep.get("models", {}).get(model, {}).get("slate", {}).get("ALL", {})
    sweep_val = sw.get(f"pipe_top{S}_m_pct_male")
    ot = order_test.get("models", {}).get(model, {}).get("comparisons", {})
    out = {
        "n_pools": len(rows),
        "n_occupations": len({r["occupation"] for r in rows}),
        "n_pools_by_category": {c: sum(r["category"] == c for r in rows) for c in CATEGORIES},
        "docs_per_request": K,
        "pct_pools_identical_top3_set": 100 * mean("same_top3")(rows),
        "pct_pools_identical_top3_set_ci": [100 * x for x in
                                            cluster_boot(rows, mean("same_top3"), seed("same3"), B)],
        "pct_pools_identical_top1": 100 * mean("same_top1")(rows),
        "replay_pct_male_top3": mean("replay_m")(rows),
        "replay_pct_male_top3_ci": cluster_boot(rows, mean("replay_m"), seed("replay"), B),
        "actual_pct_male_top3": mean("actual_m")(rows),
        "actual_pct_male_top3_ci": cluster_boot(rows, mean("actual_m"), seed("actual"), B),
        "paired_diff_actual_minus_replay_pp": d,
        "paired_diff_ci": d_ci,
        "paired_diff_ci_excludes_zero": bool(d_ci[0] > 0 or d_ci[1] < 0),
        "by_category": {c: {"replay_pct_male_top3": mean("replay_m")([r for r in rows if r["category"] == c]),
                            "actual_pct_male_top3": mean("actual_m")([r for r in rows if r["category"] == c])}
                        for c in CATEGORIES if any(r["category"] == c for r in rows)},
        "score_agreement": {
            "scale_match": scale_match,
            "mean_abs_score_diff": float(absd.mean()) if scale_match else None,
            "max_abs_score_diff": float(absd.max()) if scale_match else None,
            "mean_abs_score_diff_raw": float(absd.mean()),
            "pct_scores_identical": float((absd == 0).mean() * 100),
            "linear_fit_actual_on_banked": {"slope": float(slope), "intercept": float(intercept)},
            "pearson_r": pearson,
            "mean_within_pool_spearman": float(np.nanmean([r["rho"] for r in rows])),
            "banked_range": [float(banked.min()), float(banked.max())],
            "actual_range": [float(actual.min()), float(actual.max())],
        },
        "sweep_reference": {
            "sweep_pct_male_top3_all_pools": sweep_val,
            "sweep_ci": sw.get(f"pipe_top{S}_m_pct_male_ci"),
            "sweep_plus_paired_diff": None if sweep_val is None else sweep_val + d,
            "sweep_plus_paired_diff_ci": None if sweep_val is None else
                [sweep_val + d_ci[0], sweep_val + d_ci[1]],
            "note": "sweep value over all 832 pools shifted by this sample's paired "
                    "difference; an estimate of the value a 10-document request would give",
        },
        "drift_reference_80doc": {
            "source": "results/order_reversal_test.json comparisons (80-document requests)",
            "same_vs_banked_mean_abs_score_diff": ot.get("same_vs_banked", {}).get("mean_abs_score_diff"),
            "same_vs_banked_pct_pairs_same_winner": ot.get("same_vs_banked", {}).get("pct_pairs_same_winner"),
            "repeat_vs_same_pct_pairs_same_winner": ot.get("repeat_vs_same", {}).get("pct_pairs_same_winner"),
        },
        "first_request": min(r["requested_at"] for r in recs),
        "last_request": max(r["requested_at"] for r in recs),
    }
    out["replay_supported"] = not out["paired_diff_ci_excludes_zero"]
    return out


def summarise(ctx, args):
    sweep = json.loads(SWEEP.read_text()) if SWEEP.exists() else {}
    order_test = json.loads(ORDER_TEST.read_text()) if ORDER_TEST.exists() else {}
    models, untested = {}, dict(UNTESTED)
    for vendor, ms in VENDORS.items():
        for m in ms:
            st_path = PARTIAL / f"{fname(m)}.status.json"
            st = json.loads(st_path.read_text()) if st_path.exists() else {"status": "not run"}
            recs = list(read_jsonl(PARTIAL / f"{fname(m)}.jsonl", ctx.sample_hash).values())
            if st.get("status") == "ok" and len(recs) == len(ctx.sample):
                models[m] = summarise_model(m, ctx, recs, args, sweep, order_test) | {"vendor": vendor}
            else:
                untested[m] = (st.get("reason") or st.get("status", "not run")) + \
                    f" ({len(recs)}/{len(ctx.sample)} pools banked)"
    sp = sweep.get("pipeline_spread", {})
    sweep_all = {m: v["slate"]["ALL"][f"pipe_top{S}_m_pct_male"]
                 for m, v in sweep.get("models", {}).items()}
    api_all = set(UNTESTED) | {m for ms in VENDORS.values() for m in ms}
    supported = sorted(m for m, v in models.items() if v["replay_supported"])
    not_supported = sorted(m for m, v in models.items() if not v["replay_supported"])
    # Local open-weight rerankers are pointwise cross-encoders scored one pair at a
    # time in the audit, so their replay is exact by construction; they are not
    # API calls and are not tested here.
    local = sorted(m for m in sweep_all if m not in api_all)

    def span(ms):
        ms = [m for m in ms if m in sweep_all]
        if not ms:
            return None
        lo, hi = min(ms, key=sweep_all.get), max(ms, key=sweep_all.get)
        return {"min": sweep_all[lo], "min_model": lo, "max": sweep_all[hi], "max_model": hi,
                "n_models": len(ms)}
    adj = {m: v["sweep_reference"]["sweep_plus_paired_diff"] for m, v in models.items()
           if v["sweep_reference"]["sweep_plus_paired_diff"] is not None}
    adjusted_all = {**{m: sweep_all[m] for m in local}, **adj}
    lo = min(adjusted_all, key=adjusted_all.get) if adjusted_all else None
    hi = max(adjusted_all, key=adjusted_all.get) if adjusted_all else None
    payload = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "label_source": ctx.label_source,
        "question": "Does replaying 80-document audit scores reproduce the top-3 an API reranker "
                    "returns when it is sent only the retriever's top-10 documents?",
        "sample": {
            "definition": f"{args.per_occupation} pools per retained occupation, seeded "
                          f"(seed {SEED}); per occupation the template styles and query "
                          f"phrasings are drawn without replacement; identical pools for "
                          f"every model",
            "seed": SEED, "per_occupation": args.per_occupation,
            "n_pools": len(ctx.sample), "n_pools_in_sweep": len(ctx.slate_keys),
            "n_occupations": len({sk[0] for sk in ctx.sample}),
            "n_pools_by_category": {c: sum(sk[3] == c for sk in ctx.sample) for c in CATEGORIES},
            "sample_hash": ctx.sample_hash,
            "pools": [pool_id(sk) for sk in ctx.sample],
        },
        "method": {
            "retriever": "BAAI/bge-large-en-v1.5 cached scores; top-10 per pool by the "
                         "simulation's ordering rule (no retriever ties in any pool)",
            "request": "exactly the 10 retrieved documents, in retriever-rank order, one request "
                       "per pool, query text as in the audit",
            "replay": "banked single-stage scores (80-document requests) for the same 10 documents",
            "top3_share": f"compounding_experiment.analyse_slate, pipe_top{S}_m, seeded random "
                          f"tie-breaks averaged over {args.repeats} draws; the replay and live arms "
                          f"share the seed and therefore every tie-break draw",
            "identical_top3_set": "share of tie-break draws (shared between arms) in which the "
                                  "two top-3 sets coincide, averaged over pools",
            "paired_diff": "mean over pools of (live - replay) male-associated % of the top-3",
            "bootstrap": f"occupation-clustered, occupations resampled within stereotype "
                         f"category, B={args.bootstrap}, percentile 95% CI",
            "replay_supported": "paired-difference 95% CI includes zero",
            "scale_match": "live-on-banked linear fit slope in [0.8, 1.25], Pearson r >= 0.9, "
                           "medians within 10% of the banked range; mean_abs_score_diff is "
                           "reported only when true",
        },
        "models": models,
        "untested": untested,
        "overview": {
            "n_models_tested": len(models),
            "models_replay_supported": supported,
            "models_replay_not_supported": not_supported,
            "sweep_spread_reported": {k: sp.get(k) for k in ("min", "min_model", "max", "max_model",
                                                              "n_models")},
            "sweep_span_api_validated": span(supported),
            "sweep_span_validated_plus_local": span(supported + local),
            "sweep_span_adjusted_api_plus_local": None if lo is None else {
                "min": adjusted_all[lo], "min_model": lo, "max": adjusted_all[hi],
                "max_model": hi, "n_models": len(adjusted_all),
                "note": "API models at sweep value + paired difference; local pointwise models "
                        "at their sweep value; IBM Granite excluded (untested)"},
            "local_models_not_tested": local,
            "local_models_note": "open-weight pointwise cross-encoders scored locally; their "
                                 "score for a pair does not depend on the other documents "
                                 "(the listwise jina-reranker-v3 is already excluded from the sweep)",
        },
    }
    with open(str(OUT) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved -> {OUT}")
    print(f"{'model':<42}{'same top3':>10}{'replay %M':>11}{'live %M':>9}{'diff':>8}  95% CI"
          f"{'':>8}{'mean|d|':>9}")
    for m, v in sorted(models.items(), key=lambda x: x[0]):
        ci = v["paired_diff_ci"]
        mad = v["score_agreement"]["mean_abs_score_diff"]
        print(f"{m:<42}{v['pct_pools_identical_top3_set']:>9.1f}%"
              f"{v['replay_pct_male_top3']:>11.1f}{v['actual_pct_male_top3']:>9.1f}"
              f"{v['paired_diff_actual_minus_replay_pp']:>+8.1f}  [{ci[0]:+.1f}, {ci[1]:+.1f}]"
              f"{'' if mad is None else f'{mad:>9.4f}'}")
    for m, why in untested.items():
        print(f"untested: {m}: {why[:120]}")


def main():
    ap = argparse.ArgumentParser(description="Pipeline replay-validity check for API rerankers")
    ap.add_argument("--vendor", action="append", choices=sorted(VENDORS), default=None)
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--per-occupation", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between requests (s)")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--allow-legacy-labels", action="store_true")
    args = ap.parse_args()
    load_env()
    os.environ.setdefault("GCP_PROJECT", "law-ai-for-legal-help")
    ctx = Context(args)
    print(f"Sample: {len(ctx.sample)} pools ({ctx.sample_hash}), "
          f"{len({sk[0] for sk in ctx.sample})} occupations", flush=True)
    if not args.summarise_only:
        todo = list(args.model or [])
        for v in args.vendor or []:
            todo += VENDORS[v]
        if not todo:
            todo = [m for ms in VENDORS.values() for m in ms]
        for m in todo:
            score_model(m, ctx, args)
    summarise(ctx, args)


if __name__ == "__main__":
    main()
