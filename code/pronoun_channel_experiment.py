"""
Pronoun-channel test: names replaced by one placeholder, pronouns kept, on every
reachable audited reranker.

A reviewer objected that given names carry signals other than gender (ethnicity,
age), so the audit's contrast cannot be attributed to gender alone. The cleanest
gender-only test replaces the name with a placeholder and keeps the pronouns, so
the two documents of a pair differ ONLY in he/his/him versus she/her. That is
the `names_only` condition of code/deidentification_experiment.py (names ->
"the candidate", pronouns kept, case preserved, lowercase=False). The texts are
built by that script's own build_texts(); nothing about the transform is
re-implemented here.

Once every name is the same placeholder, the ten name pairs collapse into one,
so each model scores 832 distinct comparisons (52 retained occupations x 4
templates x 4 query phrasings) = 1,664 query-document scorings, as 208 requests
of 8 documents: one request per query, laid out as in the audit (male version
first within each pair, templates in the audit's order).

Models
  * The 4 local rerankers already scored under `names_only` by the
    de-identification run: read from results/deid_pairs/*.json (not re-scored).
  * The 15 API rerankers of code/order_reversal_test.py, scored here through
    its make_scorer / Throttle / call_with_retry / load_env.
  * The 8 open-weight rerankers outside the de-identification run, plus a
    self-hosted copy of IBM Granite r2 (its Azure endpoint cannot be
    re-queried), scored here on the local GPU (MPS) through
    code/score_open_models.py's load_backend recipes at the pinned revisions.
  * Excluded: jinaai/jina-reranker-v3 (see EXCLUDED).

Long-run discipline
  * Canary. API: the first request of every session is checked (every
    document scored, finite, not all equal) before any further request; its
    result is kept as that query's data, so it costs no extra request. Local:
    score_open_models' occupation-matching canary (12 queries x 12 documents,
    top-1 >= 75%, finite, varied), then the same check on the first query.
  * Checkpoints. Every finished query is appended to
    results/pronoun_channel/partial/<model>.jsonl; a restart resumes there.
  * Rate limits. A per-vendor request throttle (order_reversal_test.Throttle);
    on HTTP 429 the Retry-After header (else 30 s), and after a 5xx a fixed
    20 s, is waited out before call_with_retry's own back-off; a daily-quota
    429 or any non-retryable 4xx ends the model at once.
  * Request cap. Every HTTP request to a model's API (canary and retries
    included) is counted at the transport layer and persisted, so a restart
    cannot exceed --max-requests (at most 220).
  * Give-up. A query that still fails after call_with_retry's attempts ends
    that model; the reason is recorded in results/pronoun_channel/<model>.json.
  * Status. results/pronoun_channel/partial/status_<batch>.json after every
    query; milestones_<batch>.log gets a line at 25/50/75/100% of the batch.
    No contrast is computed while a batch runs.
  * Drift check (local models). The untransformed audit documents of 2 queries
    are re-scored and compared with the banked scores.

Analysis (one code path for every model): tie-aware % male-favoured by category;
the male-typed minus female-typed contrast with an occupation-clustered
percentile bootstrap (interval_sensitivity.contrast_draws, 50,000 draws, seeded
with the seed that file uses, so the change from the original contrast is a
paired comparison); the original contrast from results/interval_sensitivity.json.
The pronoun-only design makes one comparison per cell (occupation x query
phrasing x template) where the original audit makes ten, one per name pair, so
the original scores are also compared with each cell's ten names averaged
(`original_name_averaged`, same seed, so that change is paired too). Re-run
checks say how far the original side moves on the machine that scored the
pronoun-only texts: the drift check of the re-run models and, for the four
de-identification-run models, that run's own re-score of the original
documents (`deid_rerun_checks`); the overview repeats the comparison without
the models whose check did not reproduce every banked winner.

Outputs
    results/pronoun_channel/partial/       checkpoints, request ledgers, status
    results/pronoun_channel/<model>.json   pair-level scores (or the failure reason)
    results/pronoun_channel_summary.json   every model, counts, checks
    results/tex/pronoun_channel.tex        tabular-only fragment (booktabs)

Run (keys from the git-ignored .env; Google uses the active gcloud login):
    GCP_PROJECT=law-ai-for-legal-help venv/bin/python code/pronoun_channel_experiment.py --batch api
    venv/bin/python code/pronoun_channel_experiment.py --batch local
    PYTHONPATH=<dir with sentence-transformers 6.1.0> \
        venv/bin/python code/pronoun_channel_experiment.py --batch local   # zerank-2, mxbai (as in the audit)
    venv/bin/python code/pronoun_channel_experiment.py --tables-only       # no scoring
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import order_reversal_test as ort  # noqa: E402  (API backend, reused as is)
from analyze_single_stage import stable_seed  # noqa: E402
from deid_transforms import DEFAULT_PLACEHOLDER, deidentify  # noqa: E402
from deidentification_experiment import LOCAL_RERANKERS as DEID_MODELS  # noqa: E402
from deidentification_experiment import NAMES, build_texts  # noqa: E402
from interval_sensitivity import cell_tables, contrast_draws  # noqa: E402
from labels import load_labels  # noqa: E402
from model_labels import model_label  # noqa: E402
from synthetic_dataset import NAME_PAIRS, QUERY_TYPES, TEMPLATES, generate_dataset  # noqa: E402

SCRIPT_VERSION = "pronoun_channel_experiment.py v1.0"
CONDITION = "names_only"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "pronoun_channel"
PARTIAL = OUT_DIR / "partial"
DEID_PAIRS = RESULTS / "deid_pairs"
SUMMARY = RESULTS / "pronoun_channel_summary.json"
TEX = RESULTS / "tex" / "pronoun_channel.tex"
SEED = 20260822
HARD_REQUEST_CAP = 220
CATEGORIES = ("male", "female", "neutral")
PLACEHOLDER_PAIR = f"{DEFAULT_PLACEHOLDER}/{DEFAULT_PLACEHOLDER}"
EXPECTED = {"n_comparisons": 832, "n_queries": 208, "n_scorings": 1664,
            "n_distinct_query_doc_strings": 1664}

API_MODELS = (ort.COHERE_MODELS + ort.VOYAGE_MODELS + ort.GOOGLE_MODELS
              + tuple(ort.OPENROUTER_MODELS))
LOCAL_MODELS = (
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "ibm-granite/granite-embedding-reranker-english-r2",
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "BAAI/bge-reranker-large",
    "Qwen/Qwen3-Reranker-0.6B",
    "nvidia/llama-nemotron-rerank-1b-v2",
    "Qwen/Qwen3-Reranker-4B",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "zeroentropy/zerank-2-reranker",
)
# Self-hosted copy -> the audited (API-served) model it stands in for.
SELFHOST_OF = {"ibm-granite/granite-embedding-reranker-english-r2":
               "ibm-granite-granite-embedding-reranker-english-r2"}
SELFHOST_RAW = {"ibm-granite/granite-embedding-reranker-english-r2":
                RESULTS / "selfhost" / "granite-embedding-reranker-english-r2_full_raw.json"}
SELFHOST_NOTE = ("self-hosted copy (open weights, local MPS) of the audited API model "
                 "ibm-granite-granite-embedding-reranker-english-r2, whose Azure ML endpoint "
                 "cannot be re-queried; its original contrast is the API model's")
# The audit scored these two with sentence-transformers 6.1.0 (their repositories'
# LogitScore modules need it); the rest with 5.2.3. A local batch only runs the
# models whose sentence-transformers major version matches the running one.
ST6_MODELS = {"zeroentropy/zerank-2-reranker", "mixedbread-ai/mxbai-rerank-large-v2"}
EXCLUDED = {
    "jinaai/jina-reranker-v3": (
        "listwise reranker: it scores all of a request's documents jointly, so its scores "
        "depend on the request's composition, and requests of 8 placeholder documents are not "
        "comparable with its audit requests of 80 documents; its original contrast is about "
        "zero (see original_contrast / original_ci)"),
}


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def slug(model: str) -> str:
    return model.replace("/", "__")


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def write_json_atomic(path: Path, obj, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Texts: deidentification_experiment.build_texts, `names_only`, collapsed
# ---------------------------------------------------------------------------

def build_comparisons(labels):
    """(query -> its comparisons in the audit's order, design checks).

    Each comparison is one (occupation, query phrasing, template): the ten name
    pairs give the same two strings once the name is a placeholder, which is
    asserted, not assumed.
    """
    triples = generate_dataset(perturbation_levels=["names_and_pronouns"], query_types=None)
    rows, _ = build_texts(triples, labels)
    comps: OrderedDict = OrderedDict()
    for rec in rows:
        t = rec["t"]
        q, dm, df = rec["cond"][CONDITION]
        k = (t.occupation, t.query_type, t.template_style)
        c = comps.get(k)
        if c is None:
            comps[k] = {"occupation": t.occupation, "category": rec["cat"],
                        "query_type": t.query_type, "template_style": t.template_style,
                        "query": q, "doc_male": dm, "doc_female": df, "n_name_pairs": 1}
        else:
            assert (c["query"], c["doc_male"], c["doc_female"]) == (q, dm, df), (
                f"name pairs did not collapse to one comparison for {k}")
            c["n_name_pairs"] += 1
    groups: OrderedDict = OrderedDict()
    for c in comps.values():
        groups.setdefault(c["query"], []).append(c)

    names = sorted({n for pair in NAME_PAIRS for n in pair} | {"Smith"})
    name_re = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)
    docs = [d for c in comps.values() for d in (c["doc_male"], c["doc_female"])]
    n_occ = {cat: len({c["occupation"] for c in comps.values() if c["category"] == cat})
             for cat in CATEGORIES}
    checks = {
        "n_triples_in": len(rows),
        "n_occupations": sum(n_occ.values()),
        "n_occupations_by_category": n_occ,
        "n_templates": len({c["template_style"] for c in comps.values()}),
        "n_query_phrasings": len({c["query_type"] for c in comps.values()}),
        "n_comparisons": len(comps),
        "n_comparisons_by_category": {cat: sum(c["category"] == cat for c in comps.values())
                                      for cat in CATEGORIES},
        "n_queries": len(groups),
        "n_scorings": 2 * len(comps),
        "n_distinct_query_doc_strings": len({(c["query"], d) for c in comps.values()
                                             for d in (c["doc_male"], c["doc_female"])}),
        "docs_per_request": sorted({2 * len(g) for g in groups.values()}),
        "name_pairs_collapsed_per_comparison": sorted({c["n_name_pairs"] for c in comps.values()}),
        "templates_in_audit_order_in_every_request": all(
            [c["template_style"] for c in g] == list(TEMPLATES) for g in groups.values()),
        "two_documents_differ_in_every_pair": all(c["doc_male"] != c["doc_female"]
                                                  for c in comps.values()),
        "no_given_name_or_surname_left": not any(name_re.search(d) for d in docs),
        "placeholder_in_every_document": all(DEFAULT_PLACEHOLDER in d for d in docs),
        # Mapping every gendered pronoun to one token makes the two documents
        # identical, so pronouns are the only thing that differs between them.
        "documents_differ_only_in_pronouns": all(
            deidentify(c["doc_male"], "pronouns_only", None, lowercase=False)
            == deidentify(c["doc_female"], "pronouns_only", None, lowercase=False)
            for c in comps.values()),
        "query_unchanged": all(c["query"] == QUERY_TYPES[c["query_type"]].format(
            occupation=c["occupation"]) for c in comps.values()),
    }
    bad = {k: (checks[k], v) for k, v in EXPECTED.items() if checks[k] != v}
    bad.update({k: v for k, v in checks.items() if v is False})
    if checks["docs_per_request"] != [8] or checks["name_pairs_collapsed_per_comparison"] != [10]:
        bad["layout"] = (checks["docs_per_request"], checks["name_pairs_collapsed_per_comparison"])
    if bad:
        raise SystemExit(f"design check failed: {bad}")
    return groups, checks


def request_layout(group):
    """One request's documents (male version first in each pair) and positions."""
    docs, pos = [], []
    for c in group:
        docs += [c["doc_male"], c["doc_female"]]
        pos.append((len(docs) - 2, len(docs) - 1))
    return docs, pos


def query_rows(group, scores, pos):
    return [{"occupation": c["occupation"], "category": c["category"],
             "query_type": c["query_type"], "template_style": c["template_style"],
             "score_male": scores[i], "score_female": scores[j]}
            for c, (i, j) in zip(group, pos)]


# ---------------------------------------------------------------------------
# Checkpoints, per-model result files, batch status
# ---------------------------------------------------------------------------

def ck_path(model):
    return PARTIAL / f"{slug(model)}.jsonl"


def result_path(model):
    return OUT_DIR / f"{slug(model)}.json"


def load_checkpoint(model):
    """(header meta or None, query -> record), tolerating a torn last line."""
    meta, done = None, OrderedDict()
    p = ck_path(model)
    if p.exists():
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    print(f"  {model}: ignoring a torn last checkpoint line", flush=True)
                    continue
                raise
            if "_meta" in rec:
                meta = rec["_meta"]
            elif "query" in rec:
                done[rec["query"]] = rec
    return meta, done


def sessions_path(model):
    return PARTIAL / f"{slug(model)}.sessions.json"


def load_sessions(model):
    p = sessions_path(model)
    return json.loads(p.read_text()) if p.exists() else []


def finalize_model(model, groups, access, extra):
    """Write results/pronoun_channel/<model>.json once every query is on disk."""
    meta, done = load_checkpoint(model)
    if len(done) != len(groups) or set(done) != set(groups):
        return False
    rows = [r for q in groups for r in done[q]["rows"]]
    stamps = sorted(rec["at"] for rec in done.values())
    sessions = load_sessions(model)
    # the backend OpenRouter named in each response, read back from the checkpoint
    # so that every path that finalizes (including a resumed, already complete
    # model) records it
    if access == "API":
        served = sorted({str(rec["served_by"]) for rec in done.values() if rec.get("served_by")})
        extra = {**extra, "served_by": served or None}
    payload = {
        "status": "complete", "model": model, "access": access,
        "self_hosted_copy_of": SELFHOST_OF.get(model),
        "script_version": SCRIPT_VERSION, "condition": CONDITION,
        "name_placeholder": DEFAULT_PLACEHOLDER, "lowercase": False,
        "score_pair_order": ["male", "female"],
        "scored_from": stamps[0], "scored_to": stamps[-1],
        "active_seconds": round(sum(s.get("seconds", 0) for s in sessions), 1),
        "n_queries": len(done), "n_pairs": len(rows),
        "meta": meta, "sessions": sessions, **extra,
        "pairs": rows,
    }
    write_json_atomic(result_path(model), payload, indent=None)
    return True


def record_failure(model, access, reason, groups, extra=None):
    _, done = load_checkpoint(model)
    payload = {"status": "failed", "model": model, "access": access,
               "self_hosted_copy_of": SELFHOST_OF.get(model),
               "script_version": SCRIPT_VERSION, "condition": CONDITION,
               "reason": reason, "recorded": now(),
               "n_queries_done": len(done), "n_queries_total": len(groups),
               "sessions": load_sessions(model), **(extra or {})}
    write_json_atomic(result_path(model), payload)
    print(f"  !! {model}: GAVE UP -- {reason}", flush=True)


class BatchStatus:
    """status_<batch>.json after every query; a milestone line at each 25%."""

    RESOLVED = ("complete", "failed", "FAILED")

    def __init__(self, batch, models, units):
        PARTIAL.mkdir(parents=True, exist_ok=True)
        self.batch, self.units = batch, units
        self.path = PARTIAL / f"status_{batch}.json"
        self.log_path = PARTIAL / f"milestones_{batch}.log"
        self.lock = threading.Lock()
        self.state = {"batch": batch, "script_version": SCRIPT_VERSION, "started": now(),
                      "pid": os.getpid(), "units_per_model": units, "models": {}}
        for m in models:
            _, done = load_checkpoint(m)
            prev = result_path(m)
            prev = json.loads(prev.read_text()).get("status") if prev.exists() else None
            st = ("complete" if prev == "complete" and len(done) == units
                  else "failed (earlier run)" if prev == "failed" else "pending")
            self.state["models"][m] = {"done": len(done), "total": units, "status": st}
        pct = self.fraction() * 100
        self.next_pct = (int(pct + 1e-9) // 25 + 1) * 25
        self._log(f"start: {pct:.1f}% of the batch already resolved "
                  f"({len(models)} models x {units} queries)")
        self._write()

    def fraction(self):
        tot = len(self.state["models"]) * self.units
        res = sum(v["total"] if v["status"].startswith(self.RESOLVED) else v["done"]
                  for v in self.state["models"].values())
        return res / tot if tot else 1.0

    def _write(self):
        self.state["updated"] = now()
        self.state["pct_resolved"] = round(100 * self.fraction(), 2)
        write_json_atomic(self.path, self.state)

    def _log(self, msg):
        line = f"{now()}  [{self.batch}] {msg}"
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def update(self, model, **kw):
        with self.lock:
            self.state["models"][model].update(kw)
            self.state["models"][model]["updated"] = now()
            self._write()
            pct = self.fraction() * 100
            while self.next_pct <= 100 and pct >= self.next_pct - 1e-9:
                ms = self.state["models"]
                groups = {k: sorted(m for m, v in ms.items() if v["status"].startswith(k))
                          for k in ("complete", "running", "failed", "FAILED")}
                self._log(f"MILESTONE {self.next_pct}%: "
                          f"{sum(v['done'] for v in ms.values())} of "
                          f"{len(ms) * self.units} queries scored; "
                          f"complete {len(groups['complete'])}, running {groups['running']}, "
                          f"failed {groups['failed'] + groups['FAILED']}")
                self.next_pct += 25

    def finish(self):
        with self.lock:
            self.state["finished"] = now()
            self._write()
            self._log(f"batch process finished: {self.fraction() * 100:.1f}% resolved; "
                      + ", ".join(f"{m}: {v['status']}" for m, v in self.state["models"].items()))


# ---------------------------------------------------------------------------
# API backend: order_reversal_test's scorers, with a request cap and 429 handling
# ---------------------------------------------------------------------------

class GiveUp(BaseException):
    """Stop scoring this model now. A BaseException on purpose: call_with_retry
    retries any Exception, and a cap, a daily quota or a 4xx must not be retried."""


class RequestLedger:
    """Every HTTP request sent to one model's API, persisted after each one."""

    def __init__(self, model, cap):
        self.model, self.cap = model, cap
        self.path = PARTIAL / f"{slug(model)}.requests.json"
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.n = d.get("n_requests", 0)
        self.by_outcome = d.get("by_outcome", {})
        self.lock = threading.Lock()

    def remaining(self):
        return self.cap - self.n

    def before(self):
        with self.lock:
            if self.n >= self.cap:
                raise GiveUp(f"request cap reached ({self.n} of {self.cap} requests sent)")
            self.n += 1
            self._save()

    def after(self, outcome):
        with self.lock:
            self.by_outcome[outcome] = self.by_outcome.get(outcome, 0) + 1
            self._save()

    def summary(self):
        return {"n_requests": self.n, "cap": self.cap, "by_outcome": dict(self.by_outcome)}

    def _save(self):
        write_json_atomic(self.path, {"model": self.model, **self.summary(), "updated": now()})


_CTX = threading.local()
_HOOKED = False


def install_request_counter():
    """Count every requests.Session.request made while a thread has a ledger set.

    order_reversal_test.make_scorer sends through `requests` for Cohere,
    OpenRouter and Google, and the voyageai client does too, so this sees every
    request each scorer makes (Google's token-refresh re-send included). The
    count is taken before the request is sent, so the cap can never be passed.
    """
    global _HOOKED
    if _HOOKED:
        return
    import requests
    original = requests.sessions.Session.request

    def counted(self, method, url, *args, **kwargs):
        ledger = getattr(_CTX, "ledger", None)
        if ledger is None:
            return original(self, method, url, *args, **kwargs)
        ledger.before()
        try:
            resp = original(self, method, url, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            ledger.after(f"no response ({type(e).__name__})")
            raise
        ledger.after(str(resp.status_code))
        if resp.ok and "openrouter.ai" in str(url):
            try:
                j = resp.json()
                _CTX.served_by = j.get("model") or j.get("provider")
            except ValueError:
                pass
        return resp

    requests.sessions.Session.request = counted
    _HOOKED = True


def vendor(model):
    if model in ort.COHERE_MODELS:
        return "azure"
    if model in ort.VOYAGE_MODELS:
        return "voyage"
    if model in ort.GOOGLE_MODELS:
        return "google"
    if model in ort.OPENROUTER_MODELS:
        return "openrouter"
    raise ValueError(model)


def is_free(model):
    return ort.OPENROUTER_MODELS.get(model, "").endswith(":free")


def throttle_for(model):
    """(Throttle, rpm): well under each vendor's published or observed limits.

    Paid OpenRouter models started at 60 rpm; transient HTTP 503s from the
    upstream provider at that rate (3 in 71 requests for qwen3-reranker-8b on
    2026-10-16) threatened the request cap, so they now run at 20 rpm.
    """
    v = vendor(model)
    if v == "openrouter":
        rpm = 15 if is_free(model) else 20
    else:
        rpm = {"voyage": 120, "google": 100, "azure": 100}[v]
    return ort.Throttle(tpm=200_000 if v == "azure" else 1_000_000, rpm=rpm), rpm


def http_error_info(e):
    resp = getattr(e, "response", None)
    status = getattr(resp, "status_code", None) or getattr(e, "http_status", None)
    headers = (getattr(resp, "headers", None) or getattr(e, "headers", None) or {})
    body = getattr(resp, "text", None) or getattr(e, "http_body", None) or str(e)
    return status, headers, str(body)


def retry_after_seconds(headers):
    for k in ("Retry-After", "retry-after"):
        v = headers.get(k) if hasattr(headers, "get") else None
        if v:
            try:
                return float(v)
            except ValueError:
                return None
    return None


NOT_RETRYABLE = {400, 401, 403, 404, 405, 413, 422}
SERVER_UNAVAILABLE = {500, 502, 503, 504}
SERVER_WAIT = 20.0


def guarded_scorer(score, throttle, log):
    """order_reversal_test's scorer behind a throttle, with explicit 429 handling."""
    def fn(query, docs):
        throttle.wait(len(query) * len(docs) + sum(map(len, docs)))
        try:
            got = score(query, docs)
        except GiveUp:
            raise
        except Exception as e:  # noqa: BLE001
            status, headers, body = http_error_info(e)
            if status == 429:
                low = body.lower()
                if "per-day" in low or "per day" in low or "daily" in low:
                    raise GiveUp(f"HTTP 429, daily quota exhausted: {body[:200]}") from None
                ra = retry_after_seconds(headers)
                wait = min(ra if ra is not None else 30.0, 300.0)
                log(f"rate limited (HTTP 429); waiting {wait:.0f}s before call_with_retry's back-off")
                time.sleep(wait)
                raise
            if status in NOT_RETRYABLE:
                raise GiveUp(f"HTTP {status}, not retryable: {body[:200]}") from None
            if status in SERVER_UNAVAILABLE:
                # upstream outages cluster in time: wait them out rather than spend
                # capped requests inside one
                log(f"HTTP {status}; waiting {SERVER_WAIT:.0f}s before call_with_retry's back-off")
                time.sleep(SERVER_WAIT)
            raise
        missing = [i for i in range(len(docs)) if i not in got or not math.isfinite(got[i])]
        if missing:
            raise ValueError(f"{len(missing)} of {len(docs)} scores missing or not finite")
        return [float(got[i]) for i in range(len(docs))]
    return fn


def canary_verdict(scores, n_docs):
    return {"n_scores": len(scores), "n_docs": n_docs,
            "finite": all(math.isfinite(x) for x in scores),
            "n_distinct": len(set(scores)),
            "passed": (len(scores) == n_docs and all(math.isfinite(x) for x in scores)
                       and len(set(scores)) > 1)}


def run_api_model(model, groups, status, args):
    access = "API"
    meta, done = load_checkpoint(model)
    ledger = RequestLedger(model, args.max_requests)
    todo = [q for q in groups if q not in done]
    if not todo:
        finalize_model(model, groups, access, {"requests": ledger.summary()})
        status.update(model, status="complete", done=len(done))
        return
    if ledger.remaining() < len(todo):
        reason = (f"request budget insufficient to finish: {ledger.n} of {ledger.cap} requests "
                  f"already sent, {len(todo)} queries left")
        record_failure(model, access, reason, groups, {"requests": ledger.summary()})
        status.update(model, status=f"failed: {reason}")
        return
    if args.limit_queries:
        todo = todo[:args.limit_queries]
    throttle, rpm = throttle_for(model)

    def log(msg):
        print(f"[api] {model}: {msg}", flush=True)

    session = {"started": now(), "requests_before": ledger.n, "throttle_rpm": rpm,
               "queries_before": len(done)}
    t0 = time.time()
    reason, q = None, None
    try:
        score = guarded_scorer(ort.make_scorer(model), throttle, log)
    except Exception as e:  # noqa: BLE001  (a missing key or project id)
        reason = f"could not build the scorer: {type(e).__name__}: {e}"
        score = None
    status.update(model, status="running", done=len(done))
    _CTX.ledger = ledger
    try:
        for i, q in enumerate(todo if score else []):
            docs, pos = request_layout(groups[q])
            _CTX.served_by = None
            scores = ort.call_with_retry(score, q, docs)
            if i == 0:   # canary: the session's first request, checked before any other
                verdict = canary_verdict(scores, len(docs)) | {"query": q}
                session["canary"] = verdict
                if not verdict["passed"]:
                    raise GiveUp(f"canary failed: {verdict}")
                log(f"canary passed ({verdict['n_scores']} scores, finite, "
                    f"{verdict['n_distinct']} distinct values)")
                if meta is None:
                    meta = {"model": model, "access": access, "vendor": vendor(model),
                            "script_version": SCRIPT_VERSION, "condition": CONDITION,
                            "started": now(), "canary": verdict,
                            "request_layout": "one request per query: 8 documents, male "
                                              "version first in each pair, templates in "
                                              "the audit's order",
                            "max_requests": ledger.cap, "throttle_rpm": rpm,
                            "gcp_project": (os.environ.get("GCP_PROJECT")
                                            if vendor(model) == "google" else None),
                            "openrouter_id": ort.OPENROUTER_MODELS.get(model)}
                    append_jsonl(ck_path(model), {"_meta": meta})
            rec = {"query": q, "at": now(), "scores": scores,
                   "served_by": getattr(_CTX, "served_by", None),
                   "rows": query_rows(groups[q], scores, pos)}
            append_jsonl(ck_path(model), rec)
            done[q] = rec
            status.update(model, done=len(done))
            if len(done) % 52 == 0:
                log(f"{len(done)}/{len(groups)} queries ({ledger.n} requests so far)")
    except GiveUp as g:
        reason = str(g)
    except Exception as e:  # noqa: BLE001  (call_with_retry ran out of attempts)
        reason = (f"query {q!r} still failed after call_with_retry's 6 attempts: "
                  f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        _CTX.ledger = None
        session.update({"ended": now(), "seconds": round(time.time() - t0, 1),
                        "requests_after": ledger.n, "queries_after": len(done),
                        "outcome": reason or "ok"})
        sessions = load_sessions(model) + [session]
        write_json_atomic(sessions_path(model), sessions)
    served = sorted({str(r.get("served_by")) for r in done.values() if r.get("served_by")})
    extra = {"requests": ledger.summary(), "served_by": served or None}
    if reason:
        record_failure(model, access, reason, groups, extra)
        status.update(model, status=f"failed: {reason[:160]}", done=len(done))
    elif finalize_model(model, groups, access, extra):
        status.update(model, status="complete", done=len(done))
        log(f"complete: {len(done)} queries, {ledger.n} requests")
    else:
        status.update(model, status="stopped (--limit-queries)", done=len(done))


def run_api_batch(args, groups):
    ort.load_env()
    install_request_counter()
    models = [m for m in API_MODELS if not args.model or m in args.model]
    status = BatchStatus("api", models, len(groups))
    by_vendor: OrderedDict = OrderedDict()
    for m in models:   # the free OpenRouter model has its own limits, so its own queue
        by_vendor.setdefault(vendor(m) + ("-free" if is_free(m) else ""), []).append(m)

    def worker(ms):
        for m in ms:
            try:
                run_api_model(m, groups, status, args)
            except BaseException as e:  # noqa: BLE001  (one model must not stop its queue)
                reason = f"unexpected {type(e).__name__}: {str(e)[:300]}"
                record_failure(m, "API", reason, groups)
                status.update(m, status=f"failed: {reason[:160]}")

    print(f"API batch: {len(models)} models in {len(by_vendor)} parallel vendor queues: "
          + "; ".join(f"{v}: {len(ms)}" for v, ms in by_vendor.items()), flush=True)
    threads = [threading.Thread(target=worker, args=(ms,), name=v)
               for v, ms in by_vendor.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    status.finish()


# ---------------------------------------------------------------------------
# Local backend: score_open_models' recipes on MPS
# ---------------------------------------------------------------------------

def hardware():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        chip = ""
    return chip or platform.processor() or platform.machine()


def free_accelerator_memory():
    gc.collect()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def banked_raw_path(model):
    return SELFHOST_RAW.get(model, RESULTS / f"{model.split('/')[-1]}_full_raw.json")


def drift_queries(orig_groups, labels):
    """Two audit queries, one male-typed and one female-typed occupation (seeded)."""
    rng = random.Random(SEED)
    out = []
    for cat in ("male", "female"):
        qs = [q for q, ts in orig_groups.items() if labels[ts[0].occupation] == cat]
        out.append(rng.choice(qs))
    return out


def drift_check(model, score, ctx):
    """Re-score the untransformed audit documents of 2 queries, as the audit laid
    them out, and compare with the banked scores."""
    import score_open_models as som
    path = banked_raw_path(model)
    raw = json.loads(path.read_text())
    assert raw["model"] == model, f"{path.name} holds {raw['model']!r}, expected {model!r}"
    bank = {(r["occupation"], r["query_type"], r["name_pair"], r["template_style"]):
            (r["score_male"], r["score_female"]) for r in raw["results"]}
    diffs, same, n, ties_b, ties_n, gaps_b = [], 0, 0, 0, 0, []
    for q in ctx["drift_queries"]:
        pairs = ctx["orig_groups"][q]
        docs, pos = som.layout(pairs, "same", 0)
        s = score(q, docs)
        for t, (i, j) in zip(pairs, pos):
            bm, bf = bank[(t.occupation, t.query_type, t.name_pair, t.template_style)]
            nm, nf = float(s[i]), float(s[j])
            diffs += [abs(nm - bm), abs(nf - bf)]
            same += int(np.sign(nm - nf) == np.sign(bm - bf))
            ties_b += int(bm == bf)
            ties_n += int(nm == nf)
            gaps_b.append(abs(bm - bf))
            n += 1
    return {"queries": ctx["drift_queries"], "n_pairs": n, "n_scores": 2 * n,
            "banked_file": rel(path), "banked_scored_at": raw.get("scored_at"),
            "banked_hardware": (raw.get("self_hosted") or {}).get("gpu"),
            "max_abs_score_diff": float(max(diffs)), "mean_abs_score_diff": float(np.mean(diffs)),
            "pct_pairs_same_winner": 100.0 * same / n,
            "n_ties_banked": ties_b, "n_ties_now": ties_n,
            "banked_mean_abs_pair_gap": float(np.mean(gaps_b)),
            "layout": "audit layout: the query's 40 pairs in one call, male version first"}


def run_local_model(model, groups, status, args, ctx):
    import score_open_models as som
    import sentence_transformers
    import torch
    import transformers
    access = "self-hosted copy (local)" if model in SELFHOST_OF else "local"
    spec = som.MODELS[model]
    meta, done = load_checkpoint(model)
    drift_p = PARTIAL / f"{slug(model)}.drift.json"
    todo = [q for q in groups if q not in done]
    if not todo and drift_p.exists() and meta:
        finalize_model(model, groups, access, {"drift_check": json.loads(drift_p.read_text())})
        status.update(model, status="complete", done=len(done))
        return
    if args.limit_queries:
        todo = todo[:args.limit_queries]

    def log(msg):
        print(f"[local] {model}: {msg}", flush=True)

    session = {"started": now(), "queries_before": len(done)}
    t0 = time.time()
    status.update(model, status="loading", done=len(done))
    reason, score = None, None
    try:
        from run_records import pinned_revision
        audit_revision = pinned_revision(
            model, experiment="selfhost_check" if spec.get("selfhost") else "audit")
        revision = som.pinned_or_current(model, spec, (meta or {}).get("revision"))
        score, revision = som.load_backend(model, spec, revision)
        if meta and revision != meta["revision"]:
            raise RuntimeError(f"loaded {revision}, checkpoint was scored at {meta['revision']}")
        session["load_seconds"] = round(time.time() - t0, 1)
        log(f"loaded {revision[:12]} on {som.device()} in {session['load_seconds']}s")

        # canary 1: score_open_models' occupation-matching canary
        hits, flat = 0, []
        for q, docs, target in ctx["canary"]:
            s = [float(x) for x in score(q, docs)]
            flat += s
            hits += int(max(range(len(s)), key=s.__getitem__) == target)
        n_c = len(ctx["canary"])
        finite = all(math.isfinite(x) for x in flat)
        varied = len({round(x, 6) for x in flat}) > 1
        canary = {"top1": f"{hits}/{n_c}", "finite": finite, "varied": varied,
                  "passed": finite and varied and hits >= 0.75 * n_c}
        session["canary"] = canary
        log(f"canary {canary}")
        if not canary["passed"]:
            raise GiveUp(f"occupation-matching canary failed: {canary}")

        if not drift_p.exists():
            drift = drift_check(model, score, ctx)
            write_json_atomic(drift_p, drift)
            log(f"drift check vs banked: max|diff| {drift['max_abs_score_diff']:.3g}, "
                f"same winner {drift['pct_pairs_same_winner']:.1f}% of {drift['n_pairs']} pairs")

        status.update(model, status="running", done=len(done))
        for i, q in enumerate(todo):
            docs, pos = request_layout(groups[q])
            scores = [float(x) for x in score(q, docs)]
            verdict = canary_verdict(scores, len(docs))
            if i == 0:
                session["first_query_check"] = verdict | {"query": q}
            # the session's first query must pass the full canary; later ones must be finite
            ok = verdict["passed"] if i == 0 else verdict["finite"]
            if not ok:
                raise GiveUp(f"scores failed the check on query {q!r}: {verdict}")
            if meta is None:
                meta = {"model": model, "access": access, "script_version": SCRIPT_VERSION,
                        "condition": CONDITION, "started": now(), "revision": revision,
                        "audit_revision": audit_revision,
                        "revision_matches_audit": revision == audit_revision,
                        "backend": spec["backend"], "dtype": spec["dtype"],
                        "batch": spec["batch"], "device": som.device(), "hardware": hardware(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "sentence_transformers": sentence_transformers.__version__,
                        "sentence_transformers_from": rel(Path(sentence_transformers.__file__))
                        if str(Path(sentence_transformers.__file__).resolve()).startswith(str(ROOT))
                        else "overlay outside the repository (pip install --no-deps --target "
                             "<dir> sentence-transformers==6.1.0, prepended to PYTHONPATH)",
                        "python": platform.python_version(), "canary": canary}
                append_jsonl(ck_path(model), {"_meta": meta})
            rec = {"query": q, "at": now(), "scores": scores,
                   "rows": query_rows(groups[q], scores, pos)}
            append_jsonl(ck_path(model), rec)
            done[q] = rec
            status.update(model, done=len(done))
            if len(done) % 52 == 0:
                log(f"{len(done)}/{len(groups)} queries")
    except GiveUp as g:
        reason = str(g)
    except Exception as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {str(e)[:300]}"
        traceback.print_exc()
    finally:
        session.update({"ended": now(), "seconds": round(time.time() - t0, 1),
                        "queries_after": len(done), "outcome": reason or "ok"})
        write_json_atomic(sessions_path(model), load_sessions(model) + [session])
        score = None
        free_accelerator_memory()
    extra = {"drift_check": json.loads(drift_p.read_text()) if drift_p.exists() else None}
    if reason:
        record_failure(model, access, reason, groups, extra)
        status.update(model, status=f"failed: {reason[:160]}", done=len(done))
    elif finalize_model(model, groups, access, extra):
        status.update(model, status="complete", done=len(done))
        log(f"complete: {len(done)} queries")
    else:
        status.update(model, status="stopped (--limit-queries)", done=len(done))


def run_local_batch(args, groups, labels):
    import score_open_models as som
    import sentence_transformers
    models = [m for m in LOCAL_MODELS if not args.model or m in args.model]
    st_major = int(sentence_transformers.__version__.split(".")[0])
    eligible = [m for m in models if (m in ST6_MODELS) == (st_major >= 6)]
    status = BatchStatus("local", models, len(groups))
    orig_groups, _ = som.work_units()
    ctx = {"canary": som.canary_set(orig_groups, labels), "orig_groups": orig_groups,
           "drift_queries": drift_queries(orig_groups, labels)}
    print(f"Local batch (sentence-transformers {sentence_transformers.__version__}, "
          f"device {som.device()}): {len(eligible)} of {len(models)} models eligible here; "
          f"drift queries {ctx['drift_queries']}", flush=True)
    for m in models:
        if m not in eligible and not status.state["models"][m]["status"].startswith("complete"):
            status.update(m, status=f"deferred: needs sentence-transformers "
                                    f"{'6' if m in ST6_MODELS else '5'}.x")
    for m in eligible:
        try:
            run_local_model(m, groups, status, args, ctx)
        except BaseException as e:  # noqa: BLE001  (one model must not stop the queue)
            if isinstance(e, KeyboardInterrupt):
                raise
            reason = f"unexpected {type(e).__name__}: {str(e)[:300]}"
            record_failure(m, "local", reason, groups)
            status.update(m, status=f"failed: {reason[:160]}")
            free_accelerator_memory()
    status.finish()


# ---------------------------------------------------------------------------
# Analysis: the same code path for every model
# ---------------------------------------------------------------------------

def deid_rows(model):
    """The de-identification run's `names_only` pairs, collapsed to 832 comparisons."""
    path = DEID_PAIRS / f"{model.replace('/', '_')}.json"
    d = json.loads(path.read_text())
    assert d["reranker"] == model and d["score_pair_order"] == ["male", "female"]
    comps = OrderedDict()
    for p in d["pairs"]:
        occ, qt, _pair, tmpl = p["key"].split("|")
        k = (occ, qt, tmpl)
        sm, sf = p[CONDITION]
        if k in comps:
            assert (comps[k]["score_male"], comps[k]["score_female"]) == (sm, sf), (
                f"{path.name}: name pairs disagree for {k}")
            continue
        comps[k] = {"occupation": occ, "category": p["category"], "query_type": qt,
                    "template_style": tmpl, "score_male": sm, "score_female": sf}
    run = json.loads((RESULTS / f"deidentification_{model.replace('/', '_')}.json").read_text())
    info = {"access": "local", "source_file": rel(path),
            "scored": run["runtime"]["date"], "scored_by": run["script_version"],
            "device": run["device"], "revision": d["model_revision"],
            "runtime": run["runtime"],
            # that run's own check: its untransformed re-score vs the banked scores
            "deid_run_determinism_check": run.get("determinism_check")}
    return list(comps.values()), info


def scored_rows(model):
    p = result_path(model)
    if not p.exists():
        return None, {"reason": "not run"}
    d = json.loads(p.read_text())
    if d.get("status") != "complete":
        return None, {"reason": d.get("reason", d.get("status"))}
    meta = d.get("meta") or {}
    info = {"access": d["access"], "source_file": rel(p),
            "scored": f"{d['scored_from']} to {d['scored_to']}",
            "scored_from": d["scored_from"], "scored_to": d["scored_to"],
            "active_seconds": d.get("active_seconds"),
            "requests": d.get("requests"), "served_by": d.get("served_by"),
            "drift_check": d.get("drift_check"),
            "canary": meta.get("canary"),
            "revision": meta.get("revision"),
            "revision_matches_audit": meta.get("revision_matches_audit"),
            "device": meta.get("device"), "hardware": meta.get("hardware"),
            "runtime": {k: meta.get(k) for k in ("torch", "transformers", "sentence_transformers",
                                                  "dtype", "batch", "backend") if k in meta}
            or None}
    return d["pairs"], info


def split(rows):
    out = {}
    for c in ("all",) + CATEGORIES:
        d = np.array([r["score_male"] - r["score_female"] for r in rows
                      if c == "all" or r["category"] == c])
        n, m, t = len(d), int((d > 0).sum()), int((d == 0).sum())
        out[c] = {"n": n, "pct_male": 100.0 * m / n, "pct_tie": 100.0 * t / n,
                  "pct_female": 100.0 * (n - m - t) / n,
                  "tie_aware_pct_male": 100.0 * (m + 0.5 * t) / n}
    return out


def contrast_of(by):
    return by["male"]["tie_aware_pct_male"] - by["female"]["tie_aware_pct_male"]


def cell_key(r):
    return r["occupation"], r["query_type"], r["template_style"]


def name_averaged(raw_rows, labels):
    """(pair rows, cell rows) of a banked original file, labelled occupations only.

    The pronoun-only design makes one comparison per cell (occupation x query
    phrasing x template), because the ten name pairs collapse into one; the
    original audit makes ten, and each of them also carries its name pair's own
    preferences, which pull a share towards 50%. Comparing the mean of a cell's
    ten male-version scores with the mean of its ten female-version scores gives
    the original audit the same one comparison per cell, so that the two
    contrasts differ in the documents and not in how each cell is counted. The
    means are compared exactly, on each score's shortest decimal form: Google's
    four-decimal scores, for instance, give equal decimal means whose binary sums
    differ by 1e-16. A cell row stores the two means as floats, which are
    asserted to order the same way as the exact means.
    """
    names = [f"{m}/{f}" for m, f in NAME_PAIRS]
    pair_rows, cells = [], OrderedDict()
    for r in raw_rows:
        cat = labels.get(r["occupation"])
        if cat is None:
            continue
        row = {"occupation": r["occupation"], "category": cat, "query_type": r["query_type"],
               "template_style": r["template_style"], "name_pair": r["name_pair"],
               "score_male": r["score_male"], "score_female": r["score_female"]}
        pair_rows.append(row)
        cell = cells.setdefault(cell_key(row), {})
        if row["name_pair"] in cell:
            raise SystemExit(f"two rows for name pair {row['name_pair']!r} in cell {cell_key(row)}")
        cell[row["name_pair"]] = row
    cell_rows = []
    for (occ, qt, tmpl), cell in cells.items():
        if sorted(cell) != sorted(names):
            raise SystemExit(f"cell {(occ, qt, tmpl)} does not hold each of the ten name pairs once")
        exact = [sum(Fraction(repr(cell[n][k])) for n in names) / len(names)
                 for k in ("score_male", "score_female")]
        sm, sf = float(exact[0]), float(exact[1])
        if (sm > sf) - (sm < sf) != (exact[0] > exact[1]) - (exact[0] < exact[1]):
            raise SystemExit(f"cell {(occ, qt, tmpl)}: the float means do not order as the exact ones")
        cell_rows.append({"occupation": occ, "category": labels[occ], "query_type": qt,
                          "template_style": tmpl, "score_male": sm, "score_female": sf,
                          "n_name_pairs": len(names)})
    return pair_rows, cell_rows


def deid_rerun_check(model, labels, banked_path, drift_queries, original_contrast):
    """The de-identification run's own scoring of the ORIGINAL documents against the
    banked original scores: for the four models read from that run, the
    counterpart of the re-run models' drift check.

    That run scored all 8,320 untransformed pairs in the same session, on the same
    machine, as its `names_only` texts. It kept those re-scores, as `original` in
    results/deid_pairs/, where no bank existed (BGE base) or the bank failed its
    determinism check (jina-reranker-v2); winners are then compared pair by pair,
    as in drift_check (a tie is its own outcome). Where the bank passed and was
    kept as `original`, the run recorded only the largest score difference d: a
    pair's winner can change only if its banked gap is at most 2d, so those pairs
    are undetermined, the agreement is a lower bound (exactly 100% when no pair is
    that close), and the contrast can move by at most one tie-aware win per
    undetermined pair. The pairs of the drift check's two queries are also
    reported on their own, beside the re-run models' figure for the same pairs.
    """
    s = model.replace("/", "_")
    pairs_path, run_path = DEID_PAIRS / f"{s}.json", RESULTS / f"deidentification_{s}.json"
    d, run = json.loads(pairs_path.read_text()), json.loads(run_path.read_text())
    raw = json.loads(banked_path.read_text())
    assert d["reranker"] == model == raw["model"] and d["score_pair_order"] == ["male", "female"]
    bank = {(r["occupation"], r["query_type"], r["name_pair"], r["template_style"]): r
            for r in raw["results"]}
    saved = not d["baseline_source"].startswith("banked")
    dc = run["determinism_check"]
    bound = None if saved else dc["max_abs_score_diff"]
    tally = {"all": [0, 0, 0], "drift": [0, 0, 0]}      # pairs, same winner, undetermined
    changes = {"reversed": 0, "tie_to_winner": 0, "winner_to_tie": 0}
    undetermined = {c: 0 for c in CATEGORIES}
    diffs, banked_rows, rescored_rows = [], [], []
    for p in d["pairs"]:
        occ, qt, pair, tmpl = p["key"].split("|")
        b, cat = bank[(occ, qt, pair, tmpl)], labels[occ]
        assert p["category"] == cat, f"{pairs_path.name}: category of {p['key']!r}"
        bm, bf = b["score_male"], b["score_female"]
        rm, rf = p["original"]
        banked_rows.append({"category": cat, "score_male": bm, "score_female": bf})
        if saved:
            rescored_rows.append({"category": cat, "score_male": rm, "score_female": rf})
            diffs += [abs(rm - bm), abs(rf - bf)]
            sb, sr = np.sign(bm - bf), np.sign(rm - rf)
            same, und = int(sb == sr), 0
            if not same:
                changes["tie_to_winner" if sb == 0 else "winner_to_tie" if sr == 0
                        else "reversed"] += 1
        else:
            if (rm, rf) != (bm, bf):
                raise SystemExit(f"{pairs_path.name}: `original` should be the bank, "
                                 f"{banked_path.name}, but differs at {p['key']!r}")
            und = int(abs(bm - bf) <= 2 * bound)
            same = 1 - und
            undetermined[cat] += und
        for g in ("all", "drift") if b["query"] in drift_queries else ("all",):
            tally[g] = [tally[g][0] + 1, tally[g][1] + same, tally[g][2] + und]

    def agreement(n, same, und):
        return {"n_pairs": n, "pct_pairs_same_winner": None if und else 100.0 * same / n,
                "pct_pairs_same_winner_at_least": 100.0 * same / n, "n_pairs_undetermined": und}

    banked_contrast = contrast_of(split(banked_rows))
    if abs(banked_contrast - original_contrast) > 1e-9:
        raise SystemExit(f"{model}: the banked pairs of {pairs_path.name} do not reproduce the "
                         f"original contrast")
    n, same, und = tally["all"]
    rescored = contrast_of(split(rescored_rows)) if saved else None
    n_cat = {c: sum(r["category"] == c for r in banked_rows) for c in ("male", "female")}
    return {
        "source": f"{rel(pairs_path)} (`original`); {rel(run_path)} (determinism_check)",
        "rerun": {"device": run["device"], "script_version": run["script_version"],
                  **run["runtime"]},
        "banked_file": rel(banked_path), "banked_scored_at": raw.get("scored_at"),
        "banked_hardware": (raw.get("self_hosted") or {}).get("gpu"),
        "baseline_source": d["baseline_source"],
        "rescored_originals_saved": saved,
        "method": ("pair by pair: the run's re-scores of the original documents are saved"
                   if saved else
                   "bound: only the bank was kept, with the run's largest score difference; a "
                   "pair whose banked gap is at most twice that is undetermined"),
        "max_abs_score_diff": float(max(diffs)) if saved else bound,
        "mean_abs_score_diff": float(np.mean(diffs)) if saved else dc["mean_abs_score_diff"],
        **agreement(n, same, und),
        "winner_changes": changes if saved else None,
        "drift_check_pairs": ({"queries": list(drift_queries), **agreement(*tally["drift"])}
                              if drift_queries else None),
        "all_winners_reproduced": None if und else same == n,
        "banked_original_contrast": banked_contrast,
        "rescored_original_contrast": rescored,
        "contrast_change": None if rescored is None else rescored - banked_contrast,
        "abs_contrast_change_bound": None if saved else
        100.0 * sum(undetermined[c] / n_cat[c] for c in ("male", "female")),
    }


def set_statistics(models, ms):
    """Medians and counts over the models `ms`: the pronoun-only contrast against the
    original one and against the name-averaged original (point estimates), and the
    paired change from the name-averaged original."""
    def med(v):
        return statistics.median(v) if v else None

    orig = [models[m]["original"]["contrast"] for m in ms]
    pron = [models[m]["contrast"] for m in ms]
    avg = [models[m]["original_name_averaged"]["contrast"] for m in ms]
    ch = [models[m]["change_vs_original_name_averaged"] for m in ms]
    return {"n_models_tested": len(ms),
            "median_original_contrast": med(orig),
            "median_contrast": med(pron),
            "median_original_name_averaged_contrast": med(avg),
            "n_contrast_above_original": sum(p > o for p, o in zip(pron, orig)),
            "n_contrast_above_original_name_averaged": sum(p > a for p, a in zip(pron, avg)),
            "median_change_vs_original_name_averaged": med([c["change"] for c in ch]),
            "n_change_vs_original_name_averaged_above_zero": sum(c["above_zero"] for c in ch),
            "n_change_vs_original_name_averaged_below_zero": sum(c["below_zero"] for c in ch)}


class ContrastFrame:
    """interval_sensitivity's cell tables and draws, over the male/female occupations."""

    def __init__(self, labels, B):
        self.occs = sorted(o for o, c in labels.items() if c in ("male", "female"))
        self.cat = [labels[o] for o in self.occs]
        self.clusters = {i: o for i, o in enumerate(self.occs)}
        self.B = B

    def point_and_draws(self, rows, seed):
        names = sorted({r.get("name_pair", PLACEHOLDER_PAIR) for r in rows})
        rows = [r if "name_pair" in r else dict(r, name_pair=PLACEHOLDER_PAIR) for r in rows]
        W, N = cell_tables(rows, self.occs, names)
        share = {c: W[[i for i, k in enumerate(self.cat) if k == c]].sum()
                 / N[[i for i, k in enumerate(self.cat) if k == c]].sum() * 100
                 for c in ("male", "female")}
        draws = contrast_draws(W, N, self.cat, self.clusters, self.B,
                               np.random.default_rng(seed), resample_names=False)
        return float(share["male"] - share["female"]), draws


def ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def analyse(args, labels, label_source, design):
    frame = ContrastFrame(labels, args.bootstrap)
    iv = json.loads((RESULTS / "interval_sensitivity.json").read_text())
    audit = json.loads((RESULTS / "single_stage_summary.json").read_text())["models"]
    order_test = json.loads(ort.SUMMARY.read_text())["models"] if ort.SUMMARY.exists() else {}
    models, missing = {}, {}
    for m in tuple(DEID_MODELS) + API_MODELS + LOCAL_MODELS:
        if m in DEID_MODELS:
            rows, info = deid_rows(m)
            info["source"] = "de-identification run (names_only), not re-scored"
        else:
            rows, info = scored_rows(m)
            if rows is None:
                missing[m] = info["reason"]
                continue
            info["source"] = "scored by this script"
        audit_key = SELFHOST_OF.get(m, m)
        seed = stable_seed(audit_key, "occupation", "contrast")   # interval_sensitivity's seed
        point, draws = frame.point_and_draws(rows, seed)
        lo, hi = ci(draws)

        orig_iv = iv["models"][audit_key]
        raw_path = SELFHOST_RAW.get(m, RESULTS / audit[audit_key]["raw_file"])
        raw_rows = json.loads(raw_path.read_text())["results"]
        p0, d0 = frame.point_and_draws(raw_rows, seed)    # same draws: paired
        reproduces = (m not in SELFHOST_OF and abs(p0 - orig_iv["contrast"]) < 1e-9
                      and np.allclose(ci(d0), orig_iv["schemes"]["occupation"]["ci"],
                                      rtol=0, atol=1e-9))
        if m not in SELFHOST_OF and not reproduces:
            raise SystemExit(f"{m}: recomputed original contrast does not reproduce "
                             f"interval_sensitivity.json")
        change = draws - d0
        by = split(rows)
        # The original audit with each cell's ten names averaged, one comparison per
        # cell as in the pronoun-only design: from the audited model's banked file
        # (the `original` column, IBM Granite's API scores), checked at pair level
        # against interval_sensitivity.json; the paired change is taken against
        # the same original scores as change_vs_original.
        audit_raw = RESULTS / audit[audit_key]["raw_file"]
        pair_rows, avg_rows = name_averaged(json.loads(audit_raw.read_text())["results"], labels)
        pair_point = contrast_of(split(pair_rows))
        if abs(pair_point - orig_iv["contrast"]) > 1e-9:
            raise SystemExit(f"{m}: the pair-level contrast of {audit_raw.name} does not reproduce "
                             f"interval_sensitivity.json")
        if sorted(map(cell_key, avg_rows)) != sorted(map(cell_key, rows)):
            raise SystemExit(f"{m}: the name-averaged cells are not the pronoun-only comparisons")
        pa, da = frame.point_and_draws(avg_rows, seed)
        by_avg = split(avg_rows)
        if abs(pa - contrast_of(by_avg)) > 1e-9:
            raise SystemExit(f"{m}: the two name-averaged contrasts disagree")
        lo_a, hi_a = ci(da)
        if m in SELFHOST_OF:
            pa0, da0 = frame.point_and_draws(name_averaged(raw_rows, labels)[1], seed)
        else:
            pa0, da0 = pa, da
        change_avg = draws - da0
        # How far the original side itself moves between runs, for reading `change`:
        # API models -- the order test's re-query of the original layout vs the bank;
        # local models -- this run's drift check (MPS here vs the banked hardware).
        if m in API_MODELS:
            c = (order_test.get(m) or {}).get("comparisons", {}).get("same_vs_banked")
            drift_ref = None if c is None else {
                "kind": "vendor drift: order test's same-layout re-query vs the banked scores",
                "source": f"results/order_reversal_test.json models[{m!r}]"
                          ".comparisons.same_vs_banked",
                "re_queried": order_test[m].get("scored"),
                "n_pairs": order_test[m].get("n_pairs"),
                "contrast_change": c["contrast_change"], "contrast_change_ci": c["contrast_change_ci"],
                "pct_pairs_same_winner": c["pct_pairs_same_winner"]}
        elif info.get("drift_check"):
            dc = info["drift_check"]
            drift_ref = {"kind": "hardware/library drift: this run's re-score of 2 audit queries "
                                 f"on {info.get('hardware')} ({info.get('device')}) vs the "
                                 f"banked scores ({dc.get('banked_hardware')})",
                         "pct_pairs_same_winner": dc["pct_pairs_same_winner"],
                         "max_abs_score_diff": dc["max_abs_score_diff"],
                         "n_pairs": dc["n_pairs"]}
        else:
            drift_ref = {"kind": "the de-identification run's determinism check",
                         **(info.get("deid_run_determinism_check") or {})}
        models[m] = {
            "label": model_label(audit_key) + (" (self-hosted copy)" if m in SELFHOST_OF else ""),
            "table_label": model_label(audit_key),
            "dagger": m in SELFHOST_OF,
            "self_hosted_copy_of": SELFHOST_OF.get(m),
            "note": SELFHOST_NOTE if m in SELFHOST_OF else None,
            "access": info["access"],
            "n_comparisons": len(rows),
            "by_category": {c: by[c] for c in CATEGORIES},
            "overall": by["all"],
            "contrast": point, "ci": [lo, hi],
            "above_zero": lo > 0, "below_zero": hi < 0, "includes_zero": lo <= 0 <= hi,
            "original": {"contrast": orig_iv["contrast"],
                         "ci": orig_iv["schemes"]["occupation"]["ci"],
                         "above_zero": orig_iv["schemes"]["occupation"]["above_zero"],
                         "source": f"results/interval_sensitivity.json models[{audit_key!r}]"},
            "original_name_averaged": {
                "contrast": pa, "ci": [lo_a, hi_a],
                "above_zero": lo_a > 0, "below_zero": hi_a < 0, "includes_zero": lo_a <= 0 <= hi_a,
                "n_comparisons": len(avg_rows),
                "n_ties": sum(r["score_male"] == r["score_female"] for r in avg_rows),
                "by_category": {c: by_avg[c] for c in CATEGORIES},
                "pair_level_contrast": pair_point,
                "pair_level_reproduces_original": True,
                "source": rel(audit_raw)},
            "change_vs_original": {
                "against": rel(raw_path) + (" (the self-hosted copy's own original scores)"
                                            if m in SELFHOST_OF else ""),
                "original_contrast": p0, "original_ci": ci(d0),
                "reproduces_interval_sensitivity": bool(reproduces) if m not in SELFHOST_OF
                else None,
                "change": point - p0, "ci": ci(change),
                "above_zero": ci(change)[0] > 0, "below_zero": ci(change)[1] < 0,
                "original_side_drift_reference": drift_ref},
            "change_vs_original_name_averaged": {
                "against": rel(raw_path) + (" (the self-hosted copy's own original scores)"
                                            if m in SELFHOST_OF else ""),
                "original_name_averaged_contrast": pa0, "original_name_averaged_ci": ci(da0),
                "change": point - pa0, "ci": ci(change_avg),
                "above_zero": ci(change_avg)[0] > 0, "below_zero": ci(change_avg)[1] < 0},
            "scoring": info,
        }
        print(f"{m:<52} original {orig_iv['contrast']:6.1f}   name-averaged {pa:6.1f}   "
              f"pronouns only {point:6.1f} [{lo:6.1f}, {hi:6.1f}]   change {point - p0:+6.1f}",
              flush=True)

    order = sorted(models, key=lambda k: -models[k]["original"]["contrast"])
    tested = order
    contrasts = [models[m]["contrast"] for m in tested]
    above = [m for m in tested if models[m]["above_zero"]]
    below = [m for m in tested if models[m]["below_zero"]]
    incl = [m for m in tested if models[m]["includes_zero"]]
    neutral = [models[m]["by_category"]["neutral"]["tie_aware_pct_male"] for m in tested]
    ch = [models[m]["change_vs_original"] for m in tested]
    overview = {
        "n_models_audited": len(audit),
        "n_models_planned": len(DEID_MODELS) + len(API_MODELS) + len(LOCAL_MODELS),
        "n_models_tested": len(tested),
        "n_above_zero": len(above), "n_below_zero": len(below), "n_including_zero": len(incl),
        "models_below_zero": below, "models_including_zero": incl,
        "n_point_estimate_above_zero": sum(c > 0 for c in contrasts),
        "median_contrast": statistics.median(contrasts) if contrasts else None,
        "min_contrast": min(contrasts) if contrasts else None,
        "max_contrast": max(contrasts) if contrasts else None,
        "median_original_contrast": statistics.median(
            models[m]["original"]["contrast"] for m in tested) if tested else None,
        "n_original_above_zero": sum(models[m]["original"]["above_zero"] for m in tested),
        "median_change_vs_original": statistics.median(c["change"] for c in ch) if ch else None,
        "n_change_above_zero": sum(c["above_zero"] for c in ch),
        "n_change_below_zero": sum(c["below_zero"] for c in ch),
        "n_change_including_zero": sum(not c["above_zero"] and not c["below_zero"] for c in ch),
        "median_neutral_tie_aware_pct_male": statistics.median(neutral) if neutral else None,
        "n_neutral_above_50": sum(v > 50 for v in neutral),
        "n_neutral_below_50": sum(v < 50 for v in neutral),
        "models_missing": missing,
    }
    # Hardware drift of the locally re-run models (paper caption): how often the
    # Apple M4 Pro re-run of sampled ORIGINAL pairs picks the banked winner.
    drift = [models[m]["scoring"]["drift_check"]["pct_pairs_same_winner"] for m in tested
             if models[m]["scoring"].get("drift_check")]
    overview["n_drift_checked"] = len(drift)
    overview["drift_min_pct_same_winner"] = min(drift) if drift else None
    overview["drift_max_pct_same_winner"] = max(drift) if drift else None
    overview["n_drift_below_100"] = sum(v < 100 for v in drift)
    by_drift = {m: models[m]["scoring"]["drift_check"]["pct_pairs_same_winner"] for m in tested
                if models[m]["scoring"].get("drift_check")}
    overview["n_drift_100"] = sum(v == 100 for v in by_drift.values())
    overview["models_drift_100"] = {m: v for m, v in by_drift.items() if v == 100}
    overview["models_drift_below_100"] = {m: v for m, v in by_drift.items() if v < 100}

    # The same question for the de-identification-run models, from that run's own
    # re-score of the original documents (on all pairs, and on the drift check's
    # two queries); Table A3 prints one agreement per re-run model.
    queries = {tuple(models[m]["scoring"]["drift_check"]["queries"]) for m in by_drift}
    if len(queries) > 1:
        raise SystemExit(f"the drift checks did not all use the same queries: {sorted(queries)}")
    queries = queries.pop() if queries else ()
    deid_checks = {m: deid_rerun_check(m, labels, RESULTS / audit[m]["raw_file"], queries,
                                       models[m]["original"]["contrast"])
                   for m in tested if m in DEID_MODELS}
    for m in tested:
        if m in by_drift:
            models[m]["rerun_agreement"] = {
                "pct_pairs_same_winner": by_drift[m], "lower_bound": False,
                "n_pairs": models[m]["scoring"]["drift_check"]["n_pairs"],
                "pairs": "the original documents of the drift check's two sampled queries",
                "source": "drift_checks"}
        elif m in deid_checks:
            c = deid_checks[m]
            models[m]["rerun_agreement"] = {
                "pct_pairs_same_winner": c["pct_pairs_same_winner_at_least"],
                "lower_bound": c["pct_pairs_same_winner"] is None,
                "n_pairs": c["n_pairs"],
                "pairs": "every pair, from the de-identification run's re-score of the original "
                         "documents",
                "source": "deid_rerun_checks"}
        else:
            models[m]["rerun_agreement"] = None
    overview["n_deid_rerun_checked"] = len(deid_checks)
    overview["models_deid_rerun_100"] = {m: c["pct_pairs_same_winner"] for m, c in deid_checks.items()
                                         if c["all_winners_reproduced"] is True}
    overview["models_deid_rerun_below_100"] = {m: c["pct_pairs_same_winner"]
                                               for m, c in deid_checks.items()
                                               if c["all_winners_reproduced"] is False}
    overview["models_deid_rerun_undetermined"] = {m: c["pct_pairs_same_winner_at_least"]
                                                  for m, c in deid_checks.items()
                                                  if c["all_winners_reproduced"] is None}

    # The pronoun-only contrast against the original and the name-averaged original,
    # over every tested model (where the overview already holds a statistic, the two
    # must agree) and without the models whose re-run did not reproduce every banked
    # winner, so that a hardware difference cannot drive the comparison.
    stats_all = set_statistics(models, tested)
    if any(overview[k] != v for k, v in stats_all.items() if k in overview):
        raise SystemExit("set_statistics disagrees with the overview")
    overview.update(stats_all)
    drop = overview["models_drift_below_100"]
    overview["excluding_drift_below_100"] = {
        "rule": "without the re-run open-weight models whose drift check (the original documents "
                "of two sampled queries, re-scored on the machine that scored the pronoun-only "
                "texts) did not reproduce every banked winner",
        "models_dropped": drop,
        **set_statistics(models, [m for m in tested if m not in drop])}
    if overview["models_deid_rerun_below_100"]:
        drop2 = {**drop, **overview["models_deid_rerun_below_100"]}
        overview["excluding_drift_below_100_and_deid_rerun_below_100"] = {
            "rule": "also without the de-identification-run models whose re-score of all their "
                    "original pairs did not reproduce every banked winner; a model whose "
                    "agreement is only bounded (models_undetermined_kept) stays in",
            "models_dropped": drop2,
            "models_undetermined_kept": overview["models_deid_rerun_undetermined"],
            **set_statistics(models, [m for m in tested if m not in drop2])}

    def window(kind):
        ms = [models[m]["scoring"] for m in tested
              if models[m]["scoring"]["source"] == "scored by this script"
              and (models[m]["access"] == "API") == (kind == "api")]
        if not ms:
            return None
        return {"first_scored": min(s["scored_from"] for s in ms),
                "last_scored": max(s["scored_to"] for s in ms),
                "sum_of_model_active_seconds": round(sum(s.get("active_seconds") or 0
                                                         for s in ms), 1)}

    # Calendar dates of the new scoring (paper text): one date if both batches
    # started and finished on the same day.
    overview["scoring_dates"] = sorted({w[k][:10] for w in (window("api"), window("local")) if w
                                        for k in ("first_scored", "last_scored")})

    requests_ = {m: models[m]["scoring"]["requests"] for m in tested
                 if models[m]["scoring"].get("requests")}
    failed = {}
    for m in missing:
        p = result_path(m)
        if p.exists():
            d = json.loads(p.read_text())
            failed[m] = {"reason": d.get("reason"), "requests": d.get("requests"),
                         "n_queries_done": d.get("n_queries_done")}
    failed_requests = {m: v["requests"] for m, v in failed.items() if v.get("requests")}
    out = {
        "script_version": SCRIPT_VERSION,
        "generated": now(),
        "label_source": label_source,
        "condition": {
            "name": CONDITION,
            "description": "names replaced by one placeholder, gendered pronouns kept: the two "
                           "documents of a pair differ only in he/his/him versus she/her",
            "name_placeholder": DEFAULT_PLACEHOLDER, "lowercase": False, "case_preserved": True,
            "texts_built_by": "code/deidentification_experiment.py build_texts() "
                              "(code/deid_transforms.py variant names_only)"},
        "definition": "contrast = tie-aware % male-favored on male-typed occupations minus the "
                      "same on female-typed occupations (ties count half); points",
        "definition_original_name_averaged": (
            "the contrast of the audited model's banked original scores with each cell "
            "(occupation x query phrasing x template) as one comparison: the mean of its ten "
            "male-version scores against the mean of its ten female-version scores, compared "
            "exactly on the scores' decimal forms (equal means count half), so that every cell "
            "weighs the same, as in the pronoun-only design; same seed and draws as the "
            "pronoun-only contrast, so change_vs_original_name_averaged is a paired interval"),
        "design": design,
        "bootstrap": {"B": args.bootstrap, "method": "percentile",
                      "resampling": "occupations with replacement within category "
                                    "(interval_sensitivity.py `occupation` scheme, "
                                    "contrast_draws with resample_names=False)",
                      "seed": "stable_seed(<audited model id>, 'occupation', 'contrast'), the "
                              "seed interval_sensitivity.py uses, so the pronouns-only and "
                              "original draws resample the same occupations and "
                              "change_vs_original is a paired interval",
                      "interval": "95%, individual (not adjusted for the number of models)"},
        "models_tested": tested,
        "excluded": {m: {"reason": why, "original_contrast": iv["models"][m]["contrast"],
                         "original_ci": iv["models"][m]["schemes"]["occupation"]["ci"]}
                     for m, why in EXCLUDED.items()},
        "failed_or_skipped": failed,
        "overview": overview,
        "drift_checks": {m: models[m]["scoring"]["drift_check"] for m in tested
                         if models[m]["scoring"].get("drift_check")},
        "deid_rerun_checks": deid_checks,
        "requests": {"per_model": requests_,
                     "failed_models": failed_requests,
                     "total_completed_models": sum(r["n_requests"] for r in requests_.values()),
                     "total_all_models": sum(r["n_requests"] for r in requests_.values())
                     + sum(r["n_requests"] for r in failed_requests.values()),
                     "cap_per_model": HARD_REQUEST_CAP,
                     "counted": "every HTTP request sent to the model's API, canary and "
                                "retries included (requests.Session.request)"},
        "scoring_windows": {"api": window("api"), "local": window("local")},
        "models": {m: models[m] for m in order},
    }
    write_json_atomic(SUMMARY, out, indent=2)
    write_tex(out, args.bootstrap)
    print("\n" + json.dumps({k: v for k, v in overview.items() if k != "models_missing"},
                            indent=1), flush=True)
    if missing:
        print(f"missing: {json.dumps(missing, indent=1)}", flush=True)
    print(f"Saved -> {rel(SUMMARY)}\nSaved -> {rel(TEX)}", flush=True)
    return out


def num(x):
    """One decimal with a true minus sign, as in the other generated tables;
    a value that rounds to zero prints as 0.0, never -0.0."""
    t = f"{x:.1f}"
    if t == "-0.0":
        t = "0.0"
    return "$-$" + t[1:] if t.startswith("-") else t


def write_tex(out, B):
    def esc(s):
        return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    def agreement(a):
        """The re-run's same-winner share (%); a lower bound is rounded down, and the
        mark takes no width, so that the numbers stay aligned."""
        if a is None:
            return "--"
        v = a["pct_pairs_same_winner"]
        txt = r"$\geq$" + f"{math.floor(v * 10) / 10:.1f}" if a["lower_bound"] else f"{v:.1f}"
        return txt + (r"\rlap{$^{\ddagger}$}" if a["source"] == "deid_rerun_checks" else "")

    lines = [f"% Generated by code/pronoun_channel_experiment.py (B = {B:,} occupation-clustered "
             "draws) -- do not edit by hand.",
             r"\begin{tabular}{lrrrlr}", r"\toprule",
             r" & \multicolumn{2}{c}{Names and pronouns} & \multicolumn{2}{c}{Pronouns only} "
             r"& Re-run \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & Original & Name-avg. & Contrast & 95\% interval & agreement (\%) \\",
             r"\midrule"]
    for m, v in out["models"].items():   # already sorted by original contrast, descending
        lo, hi = v["ci"]
        dag = r"$^{\dagger}$" if v["dagger"] else ""
        lines.append(f"{esc(v['table_label'])}{dag} & {num(v['original']['contrast'])} & "
                     f"{num(v['original_name_averaged']['contrast'])} & "
                     f"{num(v['contrast'])} & [{num(lo)}, {num(hi)}] & "
                     f"{agreement(v['rerun_agreement'])} \\\\")
    agree = [v["rerun_agreement"] for v in out["models"].values() if v["rerun_agreement"]]
    n_drift = sorted({a["n_pairs"] for a in agree if a["source"] == "drift_checks"})
    n_deid = sorted({a["n_pairs"] for a in agree if a["source"] == "deid_rerun_checks"})
    note = (r"Name-avg.: the original scores with each cell's ten male-version and ten "
            r"female-version scores averaged before the comparison, so that each occupation, "
            r"template, and query phrasing gives one comparison, as in the pronoun-only design. "
            r"Re-run agreement: share of original pairs whose winner is reproduced when their "
            r"documents are re-scored on the machine that scored the pronoun-only texts"
            + (f": the {' or '.join(f'{n:,}' for n in n_drift)} pairs of two sampled queries"
               if n_drift else ""))
    if n_deid:
        note += (r", or $^{\ddagger}$all " + " or ".join(f"{n:,}" for n in n_deid)
                 + " pairs, re-scored by the de-identification run"
                 + (r" ($\geq$: a lower bound, as that run kept only its largest score difference)"
                    if any(a["lower_bound"] for a in agree) else ""))
    note += r". --: API model, not re-run."
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip{\footnotesize\raggedright " + note + r"\par}"]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pronoun-channel (names_only) test, all reachable models")
    ap.add_argument("--batch", choices=("api", "local"),
                    help="score one batch; without it (or with --tables-only) only analyse")
    ap.add_argument("--tables-only", action="store_true",
                    help="no scoring: rebuild the summary JSON and the tex fragment from the "
                         "saved pair-level files")
    ap.add_argument("--model", action="append", default=None,
                    help="restrict the batch to these models (repeatable)")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="stop each model after this many queries in this session (testing); "
                         "the scores go into the real checkpoint and are reused")
    ap.add_argument("--max-requests", type=int, default=HARD_REQUEST_CAP,
                    help=f"per-model API request cap, canary and retries included "
                         f"(at most {HARD_REQUEST_CAP})")
    ap.add_argument("--bootstrap", type=int, default=50_000)
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()
    if args.max_requests > HARD_REQUEST_CAP:
        ap.error(f"--max-requests may not exceed {HARD_REQUEST_CAP}")
    if args.tables_only and args.batch:
        ap.error("--tables-only never scores; drop --batch")

    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    groups, design = build_comparisons(labels)
    design = {"label_source": label_source, **design,
              "request_layout": "one request per query: its 8 documents, male version first "
                                "within each pair, templates in the audit's order ("
                                + ", ".join(TEMPLATES) + ")"}
    print(f"Design: {design['n_comparisons']} comparisons = {design['n_occupations']} "
          f"occupations x {design['n_templates']} templates x {design['n_query_phrasings']} "
          f"query phrasings; {design['n_queries']} requests of {design['docs_per_request'][0]} "
          f"documents = {design['n_scorings']} scorings per model; all checks passed", flush=True)

    if args.batch == "api":
        t0 = time.time()
        run_api_batch(args, groups)
        print(f"API batch wall time {time.time() - t0:.0f}s", flush=True)
    elif args.batch == "local":
        t0 = time.time()
        run_local_batch(args, groups, labels)
        print(f"local batch wall time {time.time() - t0:.0f}s", flush=True)
    else:
        analyse(args, labels, label_source, design)


if __name__ == "__main__":
    main()
