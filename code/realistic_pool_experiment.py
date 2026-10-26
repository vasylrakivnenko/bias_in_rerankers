"""G3 -- flip rate at the margin, on real resumes in a realistic pool.

REVIEW_TODO.md section G3. THE headline number of the realistic-pool study.

For each job query we build a pool of K=20 real resumes with genuine quality
variation, score it with a focal candidate in their original form, then score it
again with the focal candidate's gender-swapped twin substituted. We measure how
often the focal candidate's shortlist membership changes (top-3 / top-5 / top-10)
when ONLY the name and the pronouns change, and in which direction the flips run.

This converts the tiny score gaps of the synthetic study into selection outcomes
in a pool where the candidates are *not* identical, which is the empirical
backbone of the disparate-impact argument.

This module is also the shared infrastructure for the other three scripts of the
study (leakage_probe.py, utility_retention.py, ner_coverage.py): corpus loading,
O*NET query construction, the SSA name->gender table, twin construction and the
batched re-ranker score cache all live here and are imported from there.

De-identification itself is NOT implemented here -- it is imported from
code/deid_transforms.py, the single source of truth (see that module's docstring).

Run `--demo` to execute end to end on a small, obviously synthetic stand-in
corpus that this module generates itself; results from that path are stamped
`validation_only: true` and are not evidence for anything.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DEFAULT_SEED = 20260822

# Stamped into every output. Bump whenever a change can move a number.
#   g3-2026-10-16-exact-k: shortlist membership is decided by an exact-k rule
#     (each scoring of a pool selects exactly k candidates; score ties are broken
#     by a seeded per-(pool, resume) key that is identical in the two
#     counterfactual conditions). The earlier rule -- "in top-k iff average rank
#     <= k" -- could select fewer or more than k candidates at a tied cutoff; its
#     flip counts are still reported, as `legacy_average_rank_rule`, for
#     comparison. Comparison-level rows go to a companion file.
SCRIPT_VERSION = "g3-2026-10-16-exact-k"

# Candidate locations for each input, tried in order. The data-acquisition agent
# may land these under slightly different names; every one of them is also
# overridable from the command line.
CORPUS_CANDIDATES = [
    DATA / "resumes" / "resumes.csv",
    DATA / "resumes.csv",
    DATA / "resumes.jsonl",
    DATA / "resume" / "Resume.csv",
    DATA / "resumes" / "Resume.csv",
    DATA / "raw" / "resumes.csv",
    DATA / "raw" / "Resume.csv",
]
ONET_CANDIDATES = [
    DATA / "onet" / "Occupation Data.txt",
    DATA / "raw" / "onet_extract" / "db_30_3_text" / "Occupation Data.txt",
]
TASKS_CANDIDATES = [
    DATA / "onet" / "Task Statements.txt",
    DATA / "raw" / "onet_extract" / "db_30_3_text" / "Task Statements.txt",
]
CAT2SOC_CANDIDATES = [
    DATA / "category_to_soc.csv",
    ROOT / "code" / "category_to_soc.csv",
]
NAMES_CANDIDATES = [
    DATA / "ssa_names.csv",
    DATA / "names_gender.csv",
    DATA / "raw" / "ssa_names.zip",
    DATA / "raw" / "names.zip",
]

RERANKERS = {
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "bge-base": "BAAI/bge-reranker-base",
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "jina": "jinaai/jina-reranker-v2-base-multilingual",
    # long-window open-weight rerankers, for the full-window re-run only
    "gte-modernbert": "Alibaba-NLP/gte-reranker-modernbert-base",
    "granite-r2": "ibm-granite/granite-embedding-reranker-english-r2",
    "zerank-2": "zeroentropy/zerank-2-reranker",
    "mxbai-large-v2": "mixedbread-ai/mxbai-rerank-large-v2",
}
# Hosted rerankers: (vendor, model id as the synthetic audit names it, documented
# context window in tokens). Both windows exceed every resume in the corpus, and
# APIScorer makes sure the whole document is read. Cohere's Rerank 5 (released
# 2026-10-21, after the audit) is called on Cohere's own API; its synthetic-pair scores
# come from code/post_audit_synthetic.py, not from the audit.
API_RERANKERS = {
    "voyage-rerank-3": ("voyage", "rerank-3", 32_000),
    "voyage-rerank-2.5": ("voyage", "rerank-2.5", 32_000),
    "cohere-v4-pro": ("azure-cohere", "Cohere-rerank-v4.0-pro", 32_000),
    "cohere-v5-pro": ("cohere", "rerank-v5.0-pro", 32_000),
    "cohere-v5-fast": ("cohere", "rerank-v5.0-fast", 32_000),
    # The audit's other hosted rerankers whose window holds every resume (the longest
    # (query, resume) input is about 4,700 tokens; Google's rankers cut each record at
    # 512 or 1,024 tokens and are left out). Each goes to the backend that served it in
    # the audit: Cohere's rerank-v3.5 and Fireworks' Qwen3-Reranker-8B were reached through
    # OpenRouter then, NVIDIA's VL reranker still is.
    "voyage-rerank-2": ("voyage", "rerank-2", 16_000),
    "voyage-rerank-3-lite": ("voyage", "rerank-3-lite", 32_000),
    "voyage-rerank-2.5-lite": ("voyage", "rerank-2.5-lite", 32_000),
    "voyage-rerank-1": ("voyage", "rerank-1", 8_000),
    "voyage-rerank-2-lite": ("voyage", "rerank-2-lite", 8_000),
    "cohere-v4-fast": ("azure-cohere", "Cohere-rerank-v4.0-fast", 32_000),
    "cohere-v3.5": ("cohere", "rerank-v3.5", 4_096),
    "qwen3-8b": ("fireworks", "accounts/fireworks/models/qwen3-reranker-8b", 40_960),
    "nemotron-vl": ("openrouter", "nvidia/llama-nemotron-rerank-vl-1b-v2:free", 10_240),
    # Not hosted, but listwise: jina-reranker-v3 scores a query's documents together, so it
    # takes the same request layout, each "request" one rerank() call on the local model
    # (at the audit's pinned revision), with documents cut at 8,192 tokens, not 2,048
    "jina-v3": ("local-listwise", "jinaai/jina-reranker-v3", 8_192),
}
COHERE_API = "https://api.cohere.com/v2/rerank"
FIREWORKS_API = "https://api.fireworks.ai/inference/v1/rerank"
OPENROUTER_API = "https://openrouter.ai/api/v1/rerank"
RERANKERS.update({short: spec[1] for short, spec in API_RERANKERS.items()})


class MissingData(SystemExit):
    """Raised with an actionable message when a required input is absent."""

    def __init__(self, what: str, tried: Sequence[Path], how: str) -> None:
        tried_s = "\n".join(f"    {p}" for p in tried)
        super().__init__(
            f"\nMISSING INPUT: {what}\n"
            f"  looked in:\n{tried_s}\n"
            f"  how to get it: {how}\n"
            f"  or pass an explicit path on the command line.\n"
            f"  or run with --demo to validate the pipeline on a synthetic stand-in.\n"
        )


def _first_existing(cands: Sequence[Path]) -> Path | None:
    for p in cands:
        if p.exists() and p.stat().st_size > 4096:
            return p
    for p in cands:  # tolerate small-but-real files (e.g. a hand-made csv)
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Resume:
    rid: str
    category: str
    text: str
    names: list[str] = field(default_factory=list)   # all detected PERSON spans
    carried_given: str | None = None                 # given name the document actually carried
    carried_gender: str | None = None                # 'M' | 'F' | None -- gender the doc carried
    carried_source: str | None = None                # 'name' | 'pronoun'
    male_text: str | None = None                     # the male form of the counterfactual pair
    female_text: str | None = None                   # the female form
    mode: str | None = None                          # 'swap' | 'inject'
    default_form: str = "male"                       # form used when a pool filler, not focal
    injected: tuple[str, str, str] | None = None     # (male given, female given, surname)
    # set by apply_marker_labels (--marker-labels): each form's prepended name and body
    heads: dict[str, str] | None = None
    bodies: dict[str, str] | None = None
    owner_markers: dict | None = None                # what the labels changed, for the rows file
    unlabelled_markers: bool = False                 # carries a marker the labels do not cover

    def form(self, which: str) -> str:
        if which == "male":
            return self.male_text or self.text
        if which == "female":
            return self.female_text or self.text
        return self.form(self.default_form)          # 'filler'


@dataclass
class Query:
    qid: str            # SOC code
    category: str       # resume-corpus category this SOC maps to
    title: str
    text: str
    stereotype: str = "unknown"
    pct_women: float | None = None


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_TEXT_COLS = ["resume_str", "resume", "text", "content", "resume_text", "body", "raw"]
_CAT_COLS = ["category", "label", "job_category", "occupation", "field"]
_ID_COLS = ["id", "resume_id", "doc_id", "uuid"]


def _pick(colnames: Sequence[str], wanted: Sequence[str]) -> str | None:
    low = {c.lower().strip(): c for c in colnames}
    for w in wanted:
        if w in low:
            return low[w]
    for w in wanted:  # substring fallback ("Resume_str" vs "resume string")
        for lc, orig in low.items():
            if w in lc:
                return orig
    return None


def _rows_from_file(path: Path) -> list[dict]:
    suf = path.suffix.lower()
    if suf in (".jsonl", ".ndjson"):
        return [json.loads(l) for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(obj, dict):
            for k in ("data", "rows", "records", "resumes"):
                if isinstance(obj.get(k), list):
                    return obj[k]
            raise SystemExit(f"{path}: JSON object with no obvious list of records")
        return obj
    if suf == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise SystemExit(f"{path} is parquet but pyarrow is not installed: {e}")
        return pq.read_table(path).to_pylist()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def load_corpus(path: Path | None = None, min_words: int = 60, max_chars: int = 20000) -> list[Resume]:
    """Load the resume corpus. Column names are auto-detected."""
    p = Path(path) if path else _first_existing(CORPUS_CANDIDATES)
    if p is None or not p.exists():
        raise MissingData(
            "resume corpus (livecareer / Wilson & Caliskan)",
            [Path(path)] if path else CORPUS_CANDIDATES,
            "HuggingFace `opensporks/resumes` (pip install datasets) or Kaggle "
            "`snehaanbhawal/resume-dataset`; see REVIEW_TODO.md G1.3",
        )
    rows = _rows_from_file(p)
    if not rows:
        raise SystemExit(f"{p} parsed to zero rows")
    cols = list(rows[0].keys())
    tcol = _pick(cols, _TEXT_COLS)
    ccol = _pick(cols, _CAT_COLS)
    icol = _pick(cols, _ID_COLS)
    if tcol is None:
        raise SystemExit(f"{p}: cannot find a resume-text column among {cols}")
    if ccol is None:
        raise SystemExit(f"{p}: cannot find a category column among {cols}")

    out: list[Resume] = []
    for i, r in enumerate(rows):
        text = (r.get(tcol) or "").strip()
        if len(text.split()) < min_words:
            continue
        cat = str(r.get(ccol) or "").strip().upper()
        if not cat:
            continue
        rid = str(r.get(icol) or f"{p.stem}-{i}")
        out.append(Resume(rid=rid, category=cat, text=text[:max_chars]))
    if not out:
        raise SystemExit(f"{p}: every row was filtered out (min_words={min_words})")
    return out


# ---------------------------------------------------------------------------
# Queries: category -> SOC -> O*NET description
# ---------------------------------------------------------------------------


def _label_from_pct_women(p: float | None) -> str:
    """The A8 rule, applied identically here and in code/labels.py."""
    if p is None:
        return "unknown"
    if p < 30:
        return "male"
    if p > 70:
        return "female"
    if 40 <= p <= 60:
        return "neutral"
    return "drop"


def load_queries(
    onet_path: Path | None = None,
    cat2soc_path: Path | None = None,
    tasks_path: Path | None = None,
    include_tasks: int = 0,
) -> tuple[list[Query], dict]:
    onet = Path(onet_path) if onet_path else _first_existing(ONET_CANDIDATES)
    c2s = Path(cat2soc_path) if cat2soc_path else _first_existing(CAT2SOC_CANDIDATES)
    if c2s is None:
        raise MissingData(
            "category_to_soc.csv (resume category -> SOC code)",
            [Path(cat2soc_path)] if cat2soc_path else CAT2SOC_CANDIDATES,
            "hand-written, ~24 rows: category,soc_code[,onet_title][,pct_women][,label]; REVIEW_TODO.md G2",
        )

    soc_rows: dict[str, tuple[str, str]] = {}
    if onet is not None:
        with onet.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                code = (row.get("O*NET-SOC Code") or row.get("onetsoc_code") or "").strip()
                if code:
                    soc_rows[code] = ((row.get("Title") or "").strip(),
                                      (row.get("Description") or "").strip())

    tasks: dict[str, list[str]] = defaultdict(list)
    if include_tasks:
        tp = Path(tasks_path) if tasks_path else _first_existing(TASKS_CANDIDATES)
        if tp is not None:
            with tp.open(newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    code = (row.get("O*NET-SOC Code") or "").strip()
                    t = (row.get("Task") or "").strip()
                    if code and t and len(tasks[code]) < include_tasks:
                        tasks[code].append(t)

    def resolve(code: str) -> tuple[str, str] | None:
        code = code.strip()
        if code in soc_rows:
            return soc_rows[code]
        for suffix in (".00", ".01"):
            if code + suffix in soc_rows:
                return soc_rows[code + suffix]
        for k in sorted(soc_rows):  # prefix match, lowest detail code wins
            if k.startswith(code):
                return soc_rows[k]
        return None

    legacy_labels: dict[str, str] = {}
    try:
        from labels import load_labels
        legacy_labels, _ = load_labels()
        legacy_labels = {k.lower(): v for k, v in legacy_labels.items()}
    except Exception:
        pass

    queries: list[Query] = []
    unmatched: list[str] = []
    with c2s.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            cols = list(row.keys())
            cat = str(row.get(_pick(cols, _CAT_COLS) or "category") or "").strip().upper()
            soc = str(row.get(_pick(cols, ["soc_code", "soc", "onet_soc_code", "onetsoc"]) or "soc_code") or "").strip()
            if not cat or not soc:
                continue
            hit = resolve(soc)
            if hit is None:
                # The mapping file may carry the O*NET text itself, in which case
                # the O*NET bundle is not needed at all.
                dcol = _pick(cols, ["onet_description", "description"])
                tcol = _pick(cols, ["onet_title", "title"])
                desc_inline = str(row.get(dcol) or "").strip() if dcol else ""
                if not desc_inline:
                    unmatched.append(f"{cat}:{soc}")
                    continue
                hit = (str(row.get(tcol) or cat.title()).strip(), desc_inline)
            title, desc = hit
            pw_raw = row.get(_pick(cols, ["pct_women", "percent_women", "women_pct"]) or "")
            try:
                pct_women = float(pw_raw) if str(pw_raw).strip() else None
            except ValueError:
                pct_women = None
            lab = str(row.get(_pick(cols, ["label", "stereotype"]) or "") or "").strip().lower()
            if lab not in ("male", "female", "neutral"):
                lab = _label_from_pct_women(pct_women)
            if lab not in ("male", "female", "neutral"):
                lab = legacy_labels.get(title.lower(), legacy_labels.get(cat.lower(), "unknown"))
            body = desc
            if tasks.get(soc) or tasks.get(soc + ".00"):
                body += " " + " ".join(tasks.get(soc) or tasks.get(soc + ".00", []))
            queries.append(Query(qid=soc, category=cat, title=title,
                                 text=f"{title}. {body}".strip(),
                                 stereotype=lab, pct_women=pct_women))
    if not queries:
        raise SystemExit(f"{c2s}: produced zero queries (unmatched SOC codes: {unmatched[:10]})")
    prov = {
        "onet_file": str(onet) if onet else "not used (descriptions taken from the mapping file)",
        "category_to_soc": str(c2s),
        "n_queries": len(queries), "unmatched_soc": unmatched,
        "include_tasks": include_tasks,
        "stereotype_counts": dict(Counter(q.stereotype for q in queries)),
    }
    return queries, prov


# ---------------------------------------------------------------------------
# SSA name -> gender table
# ---------------------------------------------------------------------------


def load_name_gender(
    path: Path | None = None,
    min_confidence: float = 0.95,
    min_count: int = 200,
    allow_stub: bool = False,
) -> tuple[dict[str, dict], dict]:
    """name(lower) -> {gender, p, count, rank}. SSA national baby-name data."""
    p = Path(path) if path else _first_existing(NAMES_CANDIDATES)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"M": 0, "F": 0})
    source = None

    if p is not None and p.exists() and p.stat().st_size > 1000:
        if p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as z:
                    members = [m for m in z.namelist() if re.match(r"yob\d{4}\.txt$", Path(m).name)]
                    if members:
                        for m in members:
                            for line in io.TextIOWrapper(z.open(m), encoding="utf-8", errors="replace"):
                                parts = line.strip().split(",")
                                if len(parts) == 3:
                                    counts[parts[0].lower()][parts[1].upper()] += int(parts[2])
                        source = f"SSA national baby names, {len(members)} year files from {p.name}"
            except zipfile.BadZipFile:
                source = None
        else:
            with p.open(newline="", encoding="utf-8", errors="replace") as fh:
                sample = fh.read(4096)
                fh.seek(0)
                if "," in sample and re.match(r"^[A-Za-z]+,[MF],\d+\s*$", sample.splitlines()[0] or ""):
                    for line in fh:  # headerless yob-style
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            counts[parts[0].lower()][parts[1].upper()] += int(parts[2])
                    source = f"SSA yob-format table {p.name}"
                else:
                    rdr = csv.DictReader(fh)
                    cols = rdr.fieldnames or []
                    ncol = _pick(cols, ["name", "given_name", "first_name"])
                    gcol = _pick(cols, ["gender", "sex"])
                    fcol = _pick(cols, ["pct_female", "percent_female", "prob_female", "p_female"])
                    ccol = _pick(cols, ["count", "n", "births", "frequency"])
                    rows = list(rdr)
                    if ncol and fcol:
                        # Fraction (0-1) or percent (0-100): decide from the data.
                        vals = []
                        for row in rows:
                            try:
                                vals.append(float(row[fcol]))
                            except (TypeError, ValueError):
                                pass
                        scale = 100.0 if (vals and max(vals) > 1.5) else 1.0
                        for row in rows:
                            try:
                                pf = float(row[fcol]) / scale
                                c = int(float(row.get(ccol) or 1)) if ccol else 1
                            except (TypeError, ValueError):
                                continue
                            nm = str(row[ncol]).strip().lower()
                            counts[nm]["F"] += int(round(pf * c))
                            counts[nm]["M"] += c - int(round(pf * c))
                        source = (f"name table {p.name} (pct_female column, "
                                  f"{'percent' if scale == 100 else 'fraction'} scale)")
                    elif ncol and gcol:
                        for row in rows:
                            g = str(row.get(gcol) or "").strip().upper()[:1]
                            if g in ("M", "F"):
                                try:
                                    c = int(float(row.get(ccol) or 1)) if ccol else 1
                                except ValueError:
                                    c = 1
                                counts[str(row[ncol]).strip().lower()][g] += c
                        source = f"name table {p.name}"

    if not counts:
        if not allow_stub:
            raise MissingData(
                "SSA name->gender table",
                [Path(path)] if path else NAMES_CANDIDATES,
                "https://www.ssa.gov/oact/babynames/limits.html (national data zip, public domain). "
                "A pre-digested CSV with columns name,gender[,count] is also accepted.",
            )
        counts = {n: {"M": c, "F": 0} for n, c in _STUB_NAMES["M"].items()}
        counts.update({n: {"M": 0, "F": c} for n, c in _STUB_NAMES["F"].items()})
        source = "STUB name list built into realistic_pool_experiment.py (demo only)"

    table: dict[str, dict] = {}
    for name, g in counts.items():
        tot = g["M"] + g["F"]
        if tot < min_count:
            continue
        p_m = g["M"] / tot
        gender = "M" if p_m >= min_confidence else ("F" if (1 - p_m) >= min_confidence else None)
        if gender is None:
            continue
        table[name] = {"gender": gender, "p": max(p_m, 1 - p_m), "count": tot}

    for gender in ("M", "F"):
        ranked = sorted((n for n, v in table.items() if v["gender"] == gender),
                        key=lambda n: -table[n]["count"])
        for i, n in enumerate(ranked):
            table[n]["rank"] = i
            table[n]["pct"] = i / max(1, len(ranked) - 1)

    prov = {"source": source, "path": str(p) if p else None,
            "min_confidence": min_confidence, "min_count": min_count,
            "n_names": len(table),
            "n_male": sum(v["gender"] == "M" for v in table.values()),
            "n_female": sum(v["gender"] == "F" for v in table.values())}
    return table, prov


# Only ever used under --demo / allow_stub, and recorded as such in the JSON.
_STUB_NAMES = {
    "M": {"john": 5_000_000, "james": 4_800_000, "robert": 4_700_000, "michael": 4_400_000,
          "william": 4_000_000, "david": 3_600_000, "richard": 2_500_000, "thomas": 2_300_000,
          "daniel": 1_900_000, "matthew": 1_600_000, "andrew": 1_400_000, "kevin": 1_000_000},
    "F": {"mary": 4_100_000, "patricia": 1_600_000, "jennifer": 1_500_000, "linda": 1_500_000,
          "elizabeth": 1_600_000, "barbara": 1_400_000, "susan": 1_100_000, "jessica": 1_100_000,
          "sarah": 1_000_000, "karen": 900_000, "nancy": 800_000, "laura": 700_000},
}


def build_twin_name_map(table: dict[str, dict]) -> dict[str, str]:
    """Match each name to an opposite-gender name of comparable frequency rank.

    Frequency percentile within gender is a proxy for era + commonness, which is
    what REVIEW_TODO.md G2 asks for. Deterministic, so twins are stable across runs.
    """
    by_g = {g: sorted((n for n, v in table.items() if v["gender"] == g),
                      key=lambda n: table[n].get("pct", 0.0)) for g in ("M", "F")}
    out: dict[str, str] = {}
    for g, other in (("M", "F"), ("F", "M")):
        pool = by_g[other]
        if not pool:
            continue
        pcts = [table[n].get("pct", 0.0) for n in pool]
        for n in by_g[g]:
            target = table[n].get("pct", 0.0)
            lo, hi = 0, len(pcts) - 1
            while lo < hi:  # nearest by percentile
                mid = (lo + hi) // 2
                if pcts[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            best = min(range(max(0, lo - 1), min(len(pool), lo + 2)),
                       key=lambda i: abs(pcts[i] - target))
            out[n] = pool[best]
    return out


# ---------------------------------------------------------------------------
# Gender swapping (twin construction). NOT de-identification.
# ---------------------------------------------------------------------------

try:
    from deid_transforms import _POSSESSIVE_STOP as _POSS_STOP
except Exception:  # pragma: no cover
    _POSS_STOP = {"is", "was", "are", "were", "and", "or", "but", "to", "for", "as",
                  "with", "in", "on", "at", "by", "from", "that", "who", "which"}

_M2F = {"he": "she", "him": "her", "his": "her", "himself": "herself"}
_F2M = {"she": "he", "her": "his", "hers": "his", "herself": "himself"}
_PRON_RE = re.compile(r"\b(he|him|his|himself|she|her|hers|herself)\b", re.IGNORECASE)

_FEM_MARK_RE = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)
_MASC_MARK_RE = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _swap_word(w: str, direction: str, rest: str) -> str:
    """One pronoun in the other gender; `rest`, the text after it, resolves the
    his/her and her/his ambiguity by its first token."""
    lw = w.lower()
    if direction == "M2F":
        if lw == "his":
            nxt = re.match(r"[A-Za-z']+", rest.lstrip())
            repl = "hers" if (not nxt or nxt.group(0).lower() in _POSS_STOP) else "her"
        else:
            repl = _M2F.get(lw, lw)
    else:
        if lw == "her":
            nxt = re.match(r"[A-Za-z']+", rest.lstrip())
            repl = "him" if (not nxt or nxt.group(0).lower() in _POSS_STOP) else "his"
        else:
            repl = _F2M.get(lw, lw)
    return _match_case(w, repl)


def swap_pronouns(text: str, direction: str) -> str:
    """direction 'M2F' or 'F2M'. Resolves the his/her and her/his ambiguity by
    the following token, the same heuristic deid_transforms.gram_pos uses."""
    return _PRON_RE.sub(lambda m: _swap_word(m.group(0), direction, text[m.end():]), text)


def pronoun_gender(text: str, min_margin: int = 2) -> str | None:
    """Gender implied by the document's own pronouns; the G4 fallback label."""
    f = len(_FEM_MARK_RE.findall(text))
    m = len(_MASC_MARK_RE.findall(text))
    if m - f >= min_margin:
        return "M"
    if f - m >= min_margin:
        return "F"
    return None


# Held CONSTANT within each twin pair, so the surname -- which carries ethnicity,
# not gender -- cannot affect the counterfactual contrast. It shifts absolute
# scores only. Override with --surnames to supply a sampled list.
DEFAULT_SURNAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]


def detect_carried_gender(
    resumes: Sequence[Resume],
    detector: Callable[[str], list[str]] | None,
    name_table: dict[str, dict],
    head_chars: int = 1500,
    allow_pronoun_gender: bool = True,
    precomputed_names: dict[str, list[str]] | None = None,
) -> dict:
    """What gender marker, if any, does each document ACTUALLY carry?

    This is a measurement, not a transformation. On the livecareer corpus the
    answer is "almost none": the publisher already stripped names and pronouns.
    That fact is a finding in its own right and it is what forces `inject` mode
    below, so it is measured explicitly and recorded in every output JSON.
    """
    stats = Counter()
    for r in resumes:
        if precomputed_names is not None:
            r.names = precomputed_names.get(r.rid, [])
        elif detector:
            r.names = detector(r.text[:head_chars])
        else:
            r.names = []
        stats["with_person_entity"] += bool(r.names)
        for cand in r.names:
            for tok in cand.split():
                key = re.sub(r"[^A-Za-z']", "", tok).lower()
                if key in name_table:
                    r.carried_given = tok
                    r.carried_gender = name_table[key]["gender"]
                    r.carried_source = "name"
                    break
            if r.carried_given:
                break
        if r.carried_gender:
            stats["gender_from_name"] += 1
        elif allow_pronoun_gender:
            g = pronoun_gender(r.text)
            if g:
                r.carried_gender, r.carried_source = g, "pronoun"
                stats["gender_from_pronoun"] += 1
        if r.carried_gender is None:
            stats["no_gender_signal"] += 1
    stats["n"] = len(resumes)
    return dict(stats)


def build_forms(
    resumes: Sequence[Resume],
    name_table: dict[str, dict],
    twin_map: dict[str, str],
    mode: str = "auto",
    seed: int = DEFAULT_SEED,
    surnames: Sequence[str] = DEFAULT_SURNAMES,
    inject_top_n: int = 400,
    use_surname: bool = True,
) -> dict:
    """Give every resume a male form and a female form differing ONLY in gender markers.

    mode:
      swap   -- only documents that already carry a name/pronoun get a pair; the
                two forms are the document and its gender-swapped twin. This is
                the design of REVIEW_TODO.md G2, and it is what the corpus can
                support for the ~5% of documents that still carry a marker.
      inject -- the document carries no marker, so a given name is prepended (as
                a real submitted resume carries one) and the two forms differ in
                that given name alone; the surname is identical in both. Any
                pronouns the document does carry are set to agree with the
                injected name, so each form is internally consistent.
      auto   -- swap where possible, inject otherwise. Per-document mode is
                recorded so flip rates can be broken out by mode.
    """
    rng = random.Random(seed)
    male_pool = sorted((n for n, v in name_table.items() if v["gender"] == "M"),
                       key=lambda n: -name_table[n]["count"])[:inject_top_n]
    stats = Counter()
    for r in resumes:
        want_swap = mode in ("swap", "auto") and r.carried_gender is not None
        if want_swap:
            other = swap_pronouns(r.text, "M2F" if r.carried_gender == "M" else "F2M")
            new_given = None
            if r.carried_given:
                key = re.sub(r"[^A-Za-z']", "", r.carried_given).lower()
                repl = twin_map.get(key)
                if repl:
                    new_given = _match_case(r.carried_given, repl)
                    other = re.sub(rf"\b{re.escape(r.carried_given)}\b", new_given, other)
            if other != r.text:
                if r.carried_gender == "M":
                    r.male_text, r.female_text = r.text, other
                else:
                    r.male_text, r.female_text = other, r.text
                r.mode = "swap"
                r.injected = (r.carried_given or "", new_given or "", "") \
                    if r.carried_gender == "M" else (new_given or "", r.carried_given or "", "")
                stats["swap"] += 1
            else:
                want_swap = False  # nothing changed -> fall through to inject
                stats["swap_produced_no_change"] += 1

        if not want_swap:
            if mode == "swap":
                stats["skipped_no_marker"] += 1
                continue
            m = rng.choice(male_pool) if male_pool else None
            f = twin_map.get(m) if m else None
            if not m or not f:
                stats["inject_failed_no_name_pair"] += 1
                continue
            sur = rng.choice(list(surnames)) if (use_surname and surnames) else ""
            mn, fn = m.capitalize(), f.capitalize()
            head_m = f"{mn} {sur}".strip()
            head_f = f"{fn} {sur}".strip()
            body = r.text
            r.male_text = f"{head_m} {swap_pronouns(body, 'F2M')}"
            r.female_text = f"{head_f} {swap_pronouns(body, 'M2F')}"
            r.mode = "inject"
            r.injected = (mn, fn, sur)
            stats["inject"] += 1

        r.default_form = rng.choice(("male", "female"))
    stats["usable"] = sum(1 for r in resumes if r.male_text and r.female_text)
    stats["n"] = len(resumes)
    return dict(stats)


# ---------------------------------------------------------------------------
# Owner-only markers and document length (October 2026)
# ---------------------------------------------------------------------------
# build_forms swaps every carried name and pronoun, but in this corpus most carried
# names belong to someone else (a supervisor, a book's author, a school named after a
# person) and some pronouns describe a client or a boss. --marker-labels reads
# hand-checked labels of which ones refer to the résumé's owner and rebuilds the forms
# so that only the owner's gender changes. --cut-words and --owner-line then vary how
# long the document is and whether it also carries the owner's pronouns.

# Third-person line with three of the owner's pronouns, as the synthetic profiles have;
# its words are the same in both forms except the pronouns
OWNER_LINE = {"male": "He is a dedicated professional, and his colleagues describe him "
                      "as reliable and thorough.",
              "female": "She is a dedicated professional, and her colleagues describe her "
                        "as reliable and thorough."}
OWNER_LINE_WORDS = len(OWNER_LINE["male"].split())


def load_marker_labels(path: Path) -> tuple[dict, str]:
    """{rid: {"names": [...], "pronouns": [...], "owner_spans": [...]}} and the file's
    sha256. `names` records whether each detected name was the owner's; `owner_spans`
    locate every occurrence of the owner's name, with the template that replaces it."""
    import hashlib
    raw = Path(path).read_bytes()
    labels = json.loads(raw)["labels"]
    for rid, lab in labels.items():
        for h in lab["names"]:
            if h["ref"] not in ("applicant", "other"):
                raise SystemExit(f"{path}: name at {rid}:{h['start']} is {h['ref']!r}")
        for p in lab["pronouns"]:
            if p["ref"] not in ("applicant", "other", "none"):
                raise SystemExit(f"{path}: pronoun at {rid}:{p['start']} is {p['ref']!r}")
    return labels, hashlib.sha256(raw).hexdigest()


def apply_marker_labels(
    resumes: Sequence[Resume],
    labels: dict,
    name_table: dict[str, dict],
    twin_map: dict[str, str],
    seed: int = DEFAULT_SEED,
    surnames: Sequence[str] = DEFAULT_SURNAMES,
    inject_top_n: int = 400,
) -> dict:
    """Rebuild each résumé's two forms so that only its owner's gender differs.

    Every form becomes a prepended name and a body. A résumé that build_forms swapped
    gets a prepended name too, drawn from the same name pools by a generator seeded per
    résumé, so every other résumé keeps its names and filler form. In the body the
    owner's own name (where labelled) becomes the prepended one and the owner's pronouns
    follow the form; other people's names and pronouns stay as written. A résumé with a
    pronoun or a swapped name but no labels is flagged, and run_g3 will not score it.
    """
    male_pool = sorted((n for n, v in name_table.items() if v["gender"] == "M"),
                       key=lambda n: -name_table[n]["count"])[:inject_top_n]
    stats = Counter()
    for r in resumes:
        if not (r.male_text and r.female_text):
            continue
        lab = labels.get(r.rid)
        was_swap = r.mode == "swap"
        if was_swap:
            rng = random.Random(f"{seed}|{r.rid}")
            m = f = None
            for _ in range(1000):
                m = rng.choice(male_pool)
                f = twin_map.get(m)
                if f:
                    break
            if not f:
                raise SystemExit(f"{r.rid}: no male name with a female twin")
            sur = rng.choice(list(surnames)) if surnames else ""
            r.injected = (m.capitalize(), f.capitalize(), sur)
            r.mode = "inject"
            stats["swap_converted_to_inject"] += 1
        mn, fn, sur = r.injected
        text = r.text
        edits = []                                    # (start, end, male, female)
        if lab is None:
            if was_swap or _PRON_RE.search(text):
                r.unlabelled_markers = True
                stats["unlabelled_with_markers"] += 1
        else:
            import hashlib
            for o in lab["owner_spans"]:
                s, e = o["start"], o["end"]
                if hashlib.sha256(text[s:e].encode()).hexdigest() != o["sha256"]:
                    raise SystemExit(f"{r.rid}: the owner's name is not at {s}-{e}")
                edits.append((s, e, *(o["template"].format(
                    given=g, surname=sur, full=f"{g} {sur}".strip(), mr=t)
                    for g, t in ((mn, "Mr"), (fn, "Ms")))))
            for p in lab["pronouns"]:
                s, w = p["start"], p["word"]
                if text[s:s + len(w)] != w:
                    raise SystemExit(f"{r.rid}: label pronoun {w!r} is not at {s}")
                if p["ref"] == "applicant":
                    rest = text[s + len(w):]
                    edits.append((s, s + len(w), _swap_word(w, "F2M", rest),
                                  _swap_word(w, "M2F", rest)))
        edits.sort()
        if any(a[1] > b[0] for a, b in zip(edits, edits[1:])):
            raise SystemExit(f"{r.rid}: overlapping label spans")
        body = {"male": text, "female": text}
        for s, e, vm, vf in reversed(edits):
            body["male"] = body["male"][:s] + vm + body["male"][e:]
            body["female"] = body["female"][:s] + vf + body["female"][e:]
        if len(body["male"].split()) != len(body["female"].split()):
            raise SystemExit(f"{r.rid}: the two bodies differ in length")
        n_names = len((lab or {}).get("owner_spans", []))
        n_own = sum(1 for p in (lab or {}).get("pronouns", []) if p["ref"] == "applicant")
        n_left = sum(1 for p in (lab or {}).get("pronouns", []) if p["ref"] != "applicant")
        r.owner_markers = {"was_swap": was_swap, "owner_names": n_names,
                           "owner_pronouns": n_own, "pronouns_left_as_written": n_left}
        stats["owner_names_replaced"] += n_names
        stats["owner_pronouns_set_to_form"] += n_own
        stats["other_pronouns_left_as_written"] += n_left
        stats["resumes_with_owner_marker_in_body"] += bool(n_names or n_own)
        r.heads = {"male": f"{mn} {sur}".strip(), "female": f"{fn} {sur}".strip()}
        r.bodies = body
        r.male_text = f"{r.heads['male']} {body['male']}"
        r.female_text = f"{r.heads['female']} {body['female']}"
    return dict(stats)


def cut_words(text: str, n: int) -> str:
    """The text up to the end of its n-th whitespace-separated word, as written."""
    for i, m in enumerate(re.finditer(r"\S+", text), 1):
        if i == n:
            return text[:m.end()]
    return text


def shape_forms(resumes: Sequence[Resume], cut: int, owner_line: bool) -> dict:
    """Each form as its prepended name, then (owner_line) OWNER_LINE in the form's gender,
    then the body, cut so that line and body together hold `cut` words (0: whole)."""
    n_cut = 0
    for r in resumes:
        if r.heads is None:
            continue
        keep = cut - OWNER_LINE_WORDS if owner_line else cut
        for f in ("male", "female"):
            body = cut_words(r.bodies[f], keep) if cut else r.bodies[f]
            setattr(r, f"{f}_text", " ".join(
                [r.heads[f]] + ([OWNER_LINE[f]] if owner_line else []) + [body]))
        n_cut += bool(cut) and len(r.bodies["male"].split()) > keep
    return {"cut_words": cut, "owner_line": owner_line, "n_resumes_cut": n_cut}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _patch_xlm_roberta_for_jina() -> bool:
    """Compatibility shim for jinaai/jina-reranker-v2-base-multilingual.

    That model ships custom remote code which imports
    `create_position_ids_from_input_ids` from transformers' XLM-R module. The
    symbol was removed in transformers 5.x, so the model fails to load with an
    ImportError. The function is a three-line, well-defined utility (RoBERTa's
    padding-aware position ids), so it is restored here rather than pinning the
    whole environment to an old transformers. Returns True if a shim was added.
    """
    import torch
    from transformers.models.xlm_roberta import modeling_xlm_roberta as m

    if hasattr(m, "create_position_ids_from_input_ids"):
        return False

    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        inc = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return inc.long() + padding_idx

    m.create_position_ids_from_input_ids = create_position_ids_from_input_ids
    return True


class Scorer:
    """Batched cross-encoder scoring with a de-duplicating cache.

    By default items are scored in insertion order, `batch_size` at a time, as in
    the banked 512-token runs. With `token_budget`, batches are planned by input
    length instead (see _plan_by_length): a long window needs it, since a fixed
    batch pads every input to its longest one; `attention_budget` further caps a
    batch's inputs x padded length squared, the size of its attention matrices,
    which dominate memory for long inputs. With `checkpoint`, every finished
    batch is appended to a JSONL file and `resume` reloads it, so a crash costs
    one batch rather than the model's whole run.
    """

    def __init__(self, model_name: str, device: str = "mps", max_length: int = 512,
                 batch_size: int = 32, token_budget: int = 0, attention_budget: int = 0,
                 checkpoint: Path | None = None, checkpoint_meta: dict | None = None,
                 resume: bool = False, revision: str | None = None, dtype: str | None = None):
        self.shim_applied = _patch_xlm_roberta_for_jina()
        t0 = time.time()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.attention_budget = attention_budget
        self.revision, self.dtype = revision, dtype
        self.model = self._load_model(device)
        self.load_seconds = time.time() - t0
        self.cache: dict[tuple, float] = {}
        self.n_scored = 0
        self.score_seconds = 0.0
        self.n_resumed = 0
        self.n_scored_once = 0
        self._ckpt = None
        if checkpoint:
            meta = json.loads(json.dumps({"model": model_name, "max_length": max_length,
                                          **(checkpoint_meta or {})}))
            self._open_checkpoint(Path(checkpoint), meta, resume)

    def _load_model(self, device: str):
        from sentence_transformers import CrossEncoder
        kw = {}
        if self.dtype:
            import torch
            kw["model_kwargs"] = {"torch_dtype": getattr(torch, self.dtype)}
        model = CrossEncoder(self.model_name, trust_remote_code=True, device=device,
                             max_length=self.max_length, revision=self.revision, **kw)
        # Decoder-based rerankers (Qwen3 backbones, e.g. zerank-2) can ship without a pad
        # token, which a sequence-classification head needs to find each input's last
        # token -- the same fix code/score_open_models.py applies in the audit.
        if model.tokenizer.pad_token is None and hasattr(model, "model"):
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.model.config.pad_token_id = model.tokenizer.eos_token_id
        return model

    def _open_checkpoint(self, path: Path, meta: dict, resume: bool) -> None:
        """Reload the batches a matching earlier attempt finished (with resume);
        otherwise start the file afresh."""
        if resume and path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            try:
                same = bool(lines) and json.loads(lines[0]).get("meta") == meta
            except json.JSONDecodeError:
                same = False
            if same:
                for line in lines[1:]:
                    try:
                        batch = json.loads(line)["scores"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # a batch cut off mid-write is scored again
                    for k, s in batch:
                        self.cache[tuple(k)] = s
                self.n_resumed = len(self.cache)
                print(f"    resuming {self.n_resumed} scores from {path}", flush=True)
            else:
                print(f"    not resuming from {path}: another configuration wrote it",
                      flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.n_resumed:
            path.write_text(json.dumps({"meta": meta}) + "\n", encoding="utf-8")
        self._ckpt = path.open("a", encoding="utf-8")

    def _save(self, scored: list[tuple[tuple, float]]) -> None:
        if self._ckpt:
            self._ckpt.write(json.dumps({"scores": [[list(k), s] for k, s in scored]}) + "\n")
            self._ckpt.flush()
            os.fsync(self._ckpt.fileno())

    def score(self, pairs: Sequence[tuple[str, str]], keys: Sequence[tuple],
              progress: bool = True) -> None:
        if self.token_budget:
            return self._score_by_length(pairs, keys, progress)
        todo = [(k, p) for k, p in zip(keys, pairs) if k not in self.cache]
        uniq: dict[tuple, tuple[str, str]] = {}
        for k, p in todo:
            uniq.setdefault(k, p)
        items = list(uniq.items())
        if not items:
            return
        t0 = time.time()
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            out = self.model.predict([p for _, p in chunk], batch_size=self.batch_size,
                                     show_progress_bar=False)
            for (k, _), s in zip(chunk, out):
                self.cache[k] = float(s)
            self._save([(k, self.cache[k]) for k, _ in chunk])
            if progress and (i // self.batch_size) % 20 == 0:
                done = min(i + self.batch_size, len(items))
                el = time.time() - t0
                rate = done / el if el else 0
                print(f"    {done}/{len(items)} pairs  {rate:6.1f} pairs/s", flush=True)
        self.score_seconds += time.time() - t0
        self.n_scored += len(items)

    def _plan_by_length(self, items: list[tuple[tuple, tuple[str, str]]]) -> list[tuple]:
        """Batches of (items to score, aliases, padded length), longest first.

        Both forms of a (query, resume) unit -- keys differing only in their last
        element -- go into one batch, so their scores come from the same padded
        shape; a batch holds at most `batch_size` inputs, `token_budget` padded
        tokens and, if set, `attention_budget` inputs x padded length squared. A
        unit too long for that is scored one input per batch,
        unpadded, which keeps the memory peak of the longest inputs down. A form
        whose truncated input equals its twin's is not scored again but takes the
        twin's score (an alias), which is what the model returns for identical
        inputs. Going longest first makes a batch too large for memory fail at
        the start.
        """
        tok = self.model.tokenizer
        units: dict[tuple, list] = defaultdict(list)
        for k, p in items:
            units[k[:-1]].append((k, p))
        planned = []
        for members in units.values():
            ids = tok([list(p) for _, p in members], truncation=True)["input_ids"]
            score_these, aliases, lens = [members[0]], [], [len(ids[0])]
            for (k, p), i in zip(members[1:], ids[1:]):
                if i == ids[0]:
                    aliases.append((k, members[0][0]))
                else:
                    score_these.append((k, p))
                    lens.append(len(i))
            planned.append((max(lens), score_these, aliases, lens))
        planned.sort(key=lambda u: -u[0])

        def fits(n_inputs: int, length: int) -> bool:
            return (n_inputs <= self.batch_size and n_inputs * length <= self.token_budget
                    and (not self.attention_budget
                         or n_inputs * length ** 2 <= self.attention_budget))

        batches, cur, cur_alias, cur_len = [], [], [], 0
        for length, score_these, aliases, lens in planned:
            if not fits(len(score_these), length):
                if cur:
                    batches.append((cur, cur_alias, cur_len))
                    cur, cur_alias = [], []
                for j, (item, n_tok) in enumerate(zip(score_these, lens)):
                    batches.append(([item], aliases if j == 0 else [], n_tok))
                continue
            n = len(cur) + len(score_these)
            if cur and not fits(n, cur_len):
                batches.append((cur, cur_alias, cur_len))
                cur, cur_alias = [], []
            if not cur:
                cur_len = length
            cur = cur + score_these
            cur_alias = cur_alias + aliases
        if cur:
            batches.append((cur, cur_alias, cur_len))
        return batches

    def _score_by_length(self, pairs: Sequence[tuple[str, str]], keys: Sequence[tuple],
                         progress: bool) -> None:
        uniq: dict[tuple, tuple[str, str]] = {}
        for k, p in zip(keys, pairs):
            uniq.setdefault(k, p)
        # The plan covers every item, including those a resumed checkpoint already
        # holds, so a resumed run scores each remaining batch exactly as an
        # uninterrupted run would.
        plan = self._plan_by_length(list(uniq.items()))
        self.n_scored_once = sum(len(a) for _, a, _ in plan)
        pending = [not all(k in self.cache for k, _ in b) or
                   not all(a in self.cache for a, _ in al) for b, al, _ in plan]
        # Rough relative cost of a batch: linear layers grow with its padded tokens,
        # attention with their square.
        cost = [len(b) * n * (1 + n / 6144) for b, _, n in plan]
        total = sum(c for c, p in zip(cost, pending) if p) or 1.0
        print(f"    {len(plan)} batches planned (longest input {plan[0][2] if plan else 0} "
              f"tokens), {sum(pending)} to score; {len(uniq) - sum(k in self.cache for k in uniq)}"
              f" of {len(uniq)} pairs left", flush=True)
        done, marks, t0, n_done = 0.0, [25, 50, 75, 100], time.time(), 0
        for bi, ((batch, aliases, length), c, todo) in enumerate(zip(plan, cost, pending)):
            if not todo:
                continue
            out = self.model.predict([p for _, p in batch], batch_size=len(batch),
                                     show_progress_bar=False)
            if length >= 2048 and self.model.device.type == "mps":
                # Return the long inputs' attention buffers to the system, which
                # otherwise keeps them cached and swaps other programs out.
                import torch
                torch.mps.empty_cache()
            scored = [(k, float(s)) for (k, _), s in zip(batch, out)]
            got = dict(scored)
            scored += [(a, got[src]) for a, src in aliases]
            for k, s in scored:
                self.cache.setdefault(k, s)
            self._save(scored)
            self.n_scored += len(batch)
            n_done += len(scored)
            done += c
            el = time.time() - t0
            left = el * (total - done) / done
            if progress and bi % 20 == 0:
                print(f"    batch {bi + 1}/{len(plan)} ({len(batch)} x {length} tokens)  "
                      f"{n_done} pairs this run  {n_done / el:5.2f} pairs/s  "
                      f"~{left / 60:.0f} min left", flush=True)
            while marks and 100 * done / total >= marks[0] - 1e-9:
                print(f"    MILESTONE {marks[0]}% of scoring done: {n_done} pairs this run "
                      f"in {el / 60:.1f} min, ~{left / 60:.0f} min left", flush=True)
                marks.pop(0)
        self.score_seconds += time.time() - t0

    @property
    def pairs_per_second(self) -> float:
        return self.n_scored / self.score_seconds if self.score_seconds else 0.0


class RecipeModel:
    """A cross-encoder's face (predict, tokenizer, device) on the audit's own scoring recipe
    for a reranker that is not a sentence-transformers cross-encoder -- Qwen3-Reranker's
    yes/no logits, NVIDIA Nemotron's classifier -- taken from code/score_open_models.py with
    its input limit raised to `window`, so Scorer can drive it like any other model."""

    def __init__(self, name: str, window: int, revision: str | None, batch: int):
        import torch
        from transformers import AutoTokenizer
        import score_open_models as som
        spec = {**som.MODELS[name], "batch": batch}
        self._score, self.revision = som.load_backend(name, spec, revision, window=window)
        # used only to plan batches; its limit must be the window, or two forms that
        # differ only beyond the tokenizer's own limit would look identical
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=self.revision,
                                                       trust_remote_code=True)
        self.tokenizer.model_max_length = window
        self.device = torch.device(som.device())

    def predict(self, pairs, batch_size: int = 1, show_progress_bar: bool = False):
        out, i = [], 0
        while i < len(pairs):  # one call per run of pairs that share a query
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            out += self._score(pairs[i][0], [d for _, d in pairs[i:j]])
            i = j
        return out


# Open-weight rerankers scored through RecipeModel: short name -> Hugging Face id. Their
# precision is the audit's (bfloat16, code/score_open_models.MODELS).
RECIPE_RERANKERS = {
    "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Reranker-4B",
    "nemotron-1b": "nvidia/llama-nemotron-rerank-1b-v2",
}
RERANKERS.update(RECIPE_RERANKERS)


class RecipeScorer(Scorer):
    """Scorer over a RecipeModel: same batching, checkpoint and resume."""

    def _load_model(self, device: str):
        return RecipeModel(self.model_name, self.max_length, self.revision, self.batch_size)


def bf16_exact(x: float) -> bool:
    """True if x is a float32 whose low 16 bits are zero, i.e. exactly a bfloat16."""
    import struct
    b = struct.pack("<f", x)
    return struct.unpack("<f", b)[0] == x and struct.unpack("<I", b)[0] & 0xFFFF == 0


class GiveUp(BaseException):
    """Stop the run now. A BaseException on purpose: the retry loop retries any
    Exception, and a request cap or a non-retryable HTTP error must not be retried."""


class APIScorer(Scorer):
    """Scores (query, resume) pairs with a hosted reranker, one checked request at a time.

    Each request sends one query with the documents of several (query, resume) units,
    both forms of a unit together and the male form first, as the synthetic audit's
    requests did (`female_first` puts the female form first instead: the second pass
    that counterbalances an order-sensitive listwise model). The whole document is scored: Voyage is called with truncation off,
    so an over-long input fails instead of being cut, and Cohere with
    max_tokens_per_doc raised to the model's window (its default, 4,096, would cut the
    longest resumes); Fireworks and OpenRouter take no such setting, and their windows
    (40,960 and 10,240 tokens) hold every resume -- the ledger keeps the tokens each
    reports reading and the backend that served each request. Every response must hold
    one finite score per document, and not
    one score for all of them, or the request is retried; a non-retryable HTTP error,
    exhausted retries or the request cap stops the run. Finished requests go to the
    checkpoint (see Scorer), and a ledger next to it counts every request sent.

    Voyage answers some requests with every score rounded to a bfloat16 (steps of
    about 0.002 near 0.3) and others at full precision, each repeatably; real-resume
    gaps are often smaller than that step, so a rounded response turns most pairs into
    ties. A rounded response is therefore sent again, up to `coarse_retries` times, and
    the first full-precision one kept; a request that never gets one keeps its rounded
    scores and is flagged (coarse_keys), so the analysis can set it apart.
    """

    NOT_RETRYABLE = {400, 401, 403, 404, 405, 413, 422}

    def __init__(self, short: str, docs_per_request: int = 40,
                 tokens_per_request: int = 60_000, max_requests: int = 0,
                 coarse_retries: int = 8,
                 checkpoint: Path | None = None, checkpoint_meta: dict | None = None,
                 resume: bool = False, female_first: bool = False):
        import order_reversal_test as ort
        ort.load_env()
        self.female_first = female_first
        self.vendor, self.model_name, self.max_length = API_RERANKERS[short]
        self.docs_per_request, self.tokens_per_request = docs_per_request, tokens_per_request
        self.batch_size, self.token_budget = docs_per_request, tokens_per_request
        self.max_requests = max_requests
        self.shim_applied, self.load_seconds = False, 0.0
        self.cache: dict[tuple, float] = {}
        self.n_scored, self.score_seconds, self.n_resumed, self.n_scored_once = 0, 0.0, 0, 0
        self.ledger = {"n_requests": 0, "n_failed_attempts": 0, "billed": 0.0,
                       "warnings": {}, "n_coarse_responses": 0,
                       "n_requests_kept_coarse": 0}
        self.coarse_retries = coarse_retries if self.vendor_of(short) == "voyage" else 0
        self.coarse_keys: set[tuple] = set()
        self._tier_seen = False
        self._retry = ort.call_with_retry
        if self.vendor == "voyage":
            import voyageai
            self.settings = {"truncation": False}
            self.billed_unit = "tokens"
            self.throttle = ort.Throttle(tpm=1_000_000, rpm=120)
            self._vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"], timeout=180)
        elif self.vendor == "azure-cohere":
            self.settings = {"max_tokens_per_doc": self.max_length}
            self.billed_unit = "search units"
            # 80% of the Azure deployment's 250,000 tokens and 250 requests a minute
            self.throttle = ort.Throttle(tpm=200_000, rpm=100)
            self._endpoint = ort.COHERE_ENDPOINT
            self._headers = {"api-key": os.environ["AZURE_API_KEY"],
                             "Content-Type": "application/json"}
        elif self.vendor == "cohere":
            self.settings = {"max_tokens_per_doc": self.max_length}
            # Rerank 5 bills input tokens, earlier models search units
            self.billed_unit = ("input tokens" if self.model_name.startswith("rerank-v5")
                                else "search units")
            # Cohere allows a production key 1,000 rerank calls a minute and documents no
            # token limit; a trial key 10 a minute (and 1,000 a month), which _request
            # reads from the first response's headers and then keeps under. A production
            # key in COHERE_API_KEY_PROD is used when set; both return the same scores.
            self.throttle = ort.Throttle(tpm=3_000_000, rpm=300)
            self._endpoint = COHERE_API
            key = os.environ.get("COHERE_API_KEY_PROD") or os.environ["COHERE_API_KEY"]
            self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        elif self.vendor == "local-listwise":
            import score_open_models as som
            from run_records import pinned_revision
            self._score, revision = som.load_backend(
                self.model_name, som.MODELS[self.model_name],
                pinned_revision(self.model_name, experiment="audit"), window=self.max_length)
            self.settings = {"max_doc_length": self.max_length, "revision": revision}
            self.billed_unit = "none (local model)"
            self.throttle = ort.Throttle(tpm=10 ** 12, rpm=10 ** 9)
        elif self.vendor == "fireworks":
            # No truncation setting: the 40,960-token window holds every resume, and the
            # reported prompt tokens show what was read
            self.settings = {"return_documents": False}
            self.billed_unit = "tokens"
            self.throttle = ort.Throttle(tpm=2_000_000, rpm=60)
            self._endpoint = FIREWORKS_API
            self._headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                             "Content-Type": "application/json"}
        else:  # openrouter
            self.settings = {}
            self.billed_unit = "tokens"
            # OpenRouter's free models allow 20 requests a minute
            self.throttle = ort.Throttle(tpm=2_000_000, rpm=16)
            self._endpoint = OPENROUTER_API
            self._headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                             "Content-Type": "application/json"}
        self._ckpt = None
        self._ledger_path = None
        if checkpoint:
            meta = json.loads(json.dumps({"model": self.model_name, "max_length": self.max_length,
                                          "settings": self.settings, **(checkpoint_meta or {})}))
            self._open_checkpoint(Path(checkpoint), meta, resume)
            self._ledger_path = Path(checkpoint).with_suffix(".requests.json")
            if resume and self.n_resumed and self._ledger_path.exists():
                self.ledger = json.loads(self._ledger_path.read_text())
            if self.n_resumed:
                for line in Path(checkpoint).read_text(encoding="utf-8").splitlines()[1:]:
                    try:
                        batch = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if batch.get("coarse"):
                        self.coarse_keys.update(tuple(k) for k, _ in batch["scores"])

    @staticmethod
    def vendor_of(short: str) -> str:
        return API_RERANKERS[short][0]

    def _save_request(self, scored: list[tuple[tuple, float]], coarse: bool) -> None:
        if self._ckpt:
            self._ckpt.write(json.dumps({"scores": [[list(k), s] for k, s in scored],
                                         "coarse": coarse}) + "\n")
            self._ckpt.flush()
            os.fsync(self._ckpt.fileno())

    def _write_ledger(self) -> None:
        if self._ledger_path:
            tmp = self._ledger_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.ledger, indent=1))
            tmp.replace(self._ledger_path)

    def _request(self, query: str, docs: list[str]) -> list[float]:
        """One API call: a score per document, in input order, or an exception."""
        if self.max_requests and self.ledger["n_requests"] >= self.max_requests:
            raise GiveUp(f"request cap reached ({self.ledger['n_requests']} of "
                         f"{self.max_requests} requests sent)")
        self.throttle.wait(len(query) * len(docs) + sum(map(len, docs)))
        self.ledger["n_requests"] += 1
        self._write_ledger()
        try:
            if self.vendor == "voyage":
                res = self._vo.rerank(query, docs, model=self.model_name, top_k=len(docs),
                                      truncation=False)
                got = {x.index: float(x.relevance_score) for x in res.results}
                self.ledger["billed"] += float(getattr(res, "total_tokens", 0) or 0)
            elif self.vendor == "local-listwise":
                got = dict(enumerate(self._score(query, docs)))
            else:
                import requests
                r = requests.post(self._endpoint, headers=self._headers, timeout=180,
                                  json={"model": self.model_name, "query": query,
                                        "documents": docs, "top_n": len(docs), **self.settings})
                if self.vendor == "cohere" and not self._tier_seen:
                    self._tier_seen = True
                    trial = r.headers.get("x-trial-endpoint-call-limit")
                    tiers = self.ledger.setdefault("key_tiers", [])
                    if ("trial" if trial else "production") not in tiers:
                        tiers.append("trial" if trial else "production")
                    if trial:
                        self.throttle.rpm = min(self.throttle.rpm, max(1, int(trial) - 1))
                        print(f"    trial key: {trial} calls a minute, "
                              f"{r.headers.get('x-endpoint-monthly-call-limit')} a month; "
                              f"throttled to {self.throttle.rpm} a minute", flush=True)
                r.raise_for_status()
                body = r.json()
                got = {x["index"]: float(x["relevance_score"])
                       for x in body.get("results") or body.get("data") or []}
                meta = body.get("meta") or {}
                billed = meta.get("billed_units") or {}
                usage = body.get("usage") or {}
                if self.vendor in ("cohere", "azure-cohere"):
                    n = billed.get("input_tokens" if self.billed_unit == "input tokens"
                                   else "search_units")
                else:
                    n = usage.get("total_tokens") or usage.get("prompt_tokens")
                self.ledger["billed"] += float(n or 0)
                if self.vendor in ("fireworks", "openrouter"):
                    who = "/".join(str(body.get(k)) for k in ("provider", "model") if body.get(k))
                    served = self.ledger.setdefault("served_by", {})
                    served[who] = served.get(who, 0) + 1
                for w in (meta.get("warnings") or []) + (body.get("warnings") or []):
                    w = str(w)[:200]
                    self.ledger["warnings"][w] = self.ledger["warnings"].get(w, 0) + 1
        except Exception as e:  # noqa: BLE001  (vendor SDKs raise assorted types)
            self.ledger["n_failed_attempts"] += 1
            self._write_ledger()
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None) or getattr(e, "http_status", None)
            body = str(getattr(resp, "text", None) or getattr(e, "http_body", None) or e)
            if status == 429 and "trial" in body.lower() and "month" in body.lower():
                raise GiveUp(f"the trial key's monthly call limit is used up: {body[:300]}") from None
            if status == 429 and ("per-day" in body.lower() or "per day" in body.lower()):
                raise GiveUp(f"a daily request limit is used up: {body[:300]}") from None
            if status == 429:
                headers = getattr(resp, "headers", None) or getattr(e, "headers", None) or {}
                try:
                    wait = min(float(headers.get("Retry-After") or 30), 300.0)
                except (TypeError, ValueError):
                    wait = 30.0
                print(f"    rate limited (HTTP 429); waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
            elif status in self.NOT_RETRYABLE:
                raise GiveUp(f"HTTP {status}, not retryable: {body[:300]}") from None
            elif status in (500, 502, 503, 504):
                print(f"    HTTP {status}; waiting 20s", flush=True)
                time.sleep(20)
            raise
        problem = None
        if sorted(got) != list(range(len(docs))):
            problem = f"{len(got)} scores for {len(docs)} documents"
        elif not all(math.isfinite(s) for s in got.values()):
            problem = "a score is not finite"
        elif len(docs) > 2 and len(set(got.values())) == 1:
            problem = f"every document scored {next(iter(got.values()))}"
        if problem:
            self.ledger["n_failed_attempts"] += 1
            self._write_ledger()
            raise ValueError(f"malformed response: {problem}")
        self._write_ledger()
        return [got[i] for i in range(len(docs))]

    def plan_requests(self, items: list[tuple[tuple, tuple[str, str]]]) -> list[list]:
        """Requests in order: one query each, whole units, the male form first, within
        docs_per_request documents and tokens_per_request estimated tokens (characters
        / 4, query counted once per document, as the throttle counts them)."""
        by_query: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
        for k, p in items:
            by_query[k[0]][k[:-1]].append((k, p))
        plan = []
        for units in by_query.values():
            cur, cur_chars = [], 0
            for members in units.values():
                first = "female" if self.female_first else "male"
                members = sorted(members, key=lambda kp: kp[0][-1] != first)
                add = sum(len(q) + len(d) for _, (q, d) in members)
                if cur and (len(cur) + len(members) > self.docs_per_request
                            or (cur_chars + add) / 4 > self.tokens_per_request):
                    plan.append(cur)
                    cur, cur_chars = [], 0
                cur, cur_chars = cur + members, cur_chars + add
            if cur:
                plan.append(cur)
        return plan

    def score(self, pairs: Sequence[tuple[str, str]], keys: Sequence[tuple],
              progress: bool = True) -> None:
        uniq: dict[tuple, tuple[str, str]] = {}
        for k, p in zip(keys, pairs):
            uniq.setdefault(k, p)
        plan = self.plan_requests(list(uniq.items()))
        pending = [req for req in plan if not all(k in self.cache for k, _ in req)]
        if not self.max_requests:
            # Room for retries -- and, for Voyage, for re-sending rounded responses (about
            # 60% of responses in the canary, so 2.5 sends a request on average); a run
            # that needs more is stopped and looked at.
            per_request = 4.0 if self.coarse_retries else 1.3
            self.max_requests = self.ledger["n_requests"] + int(per_request * len(pending)) + 10
        total = sum(len(req) for req in pending) or 1
        print(f"    {len(plan)} requests planned, {len(pending)} to send "
              f"(cap {self.max_requests} requests in all); {total} documents", flush=True)
        done, marks, t0 = 0, [10, 25, 50, 75, 100], time.time()
        for i, req in enumerate(pending):
            query = req[0][1][0]
            docs = [d for _, (_, d) in req]
            for attempt in range(self.coarse_retries + 1):
                scores = self._retry(self._request, query, docs)
                coarse = len(docs) >= 4 and all(bf16_exact(x) for x in scores)
                if not coarse:
                    break
                self.ledger["n_coarse_responses"] += 1
            if coarse:
                self.ledger["n_requests_kept_coarse"] += 1
                self.coarse_keys.update(k for k, _ in req)
            self._write_ledger()
            scored = [(k, s) for (k, _), s in zip(req, scores)]
            for k, s in scored:
                self.cache.setdefault(k, s)
            self._save_request(scored, coarse)
            self.n_scored += len(req)
            done += len(req)
            el = time.time() - t0
            left = el * (total - done) / done
            if progress and i % 10 == 0:
                print(f"    request {i + 1}/{len(pending)}  {done}/{total} documents  "
                      f"~{left / 60:.0f} min left  ({self.ledger['n_requests']} sent, "
                      f"{self.ledger['n_failed_attempts']} failed attempts)", flush=True)
            while marks and 100 * done / total >= marks[0] - 1e-9:
                print(f"    MILESTONE {marks[0]}% of scoring done: {done}/{total} documents "
                      f"in {el / 60:.1f} min, ~{left / 60:.0f} min left; "
                      f"{self.ledger['n_requests']} requests sent, "
                      f"{self.ledger['n_failed_attempts']} failed attempts, "
                      f"billed {self.ledger['billed']:,.0f} {self.billed_unit}", flush=True)
                marks.pop(0)
        self.score_seconds += time.time() - t0


def average_ranks(scores: Sequence[float]) -> list[float]:
    """Descending-score ranks (1 = best) with the average-rank tie convention.

    Ties are never broken by insertion order -- that is the C1 bug, and here it
    would bias the result directly, since the original and the twin would be
    inserted in a fixed order.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def tiebreak_key(seed: int, pool_id: str, rid: str) -> float:
    """Seeded tie-break key in [0, 1) for one member of one pool.

    It depends only on (seed, pool, resume) -- never on the score, the form, or
    the member's position in the pool -- so it is identical in the male-form and
    female-form scorings of a comparison, and a tie cannot be resolved in favour
    of whichever form happens to be scored or inserted first.
    """
    import hashlib
    h = hashlib.sha256(f"{seed}|{pool_id}|{rid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def ordinal_ranks(scores: Sequence[float], tiebreak: Sequence[float]) -> list[int]:
    """Descending-score ranks 1..n, every rank used exactly once.

    Equal scores are ordered by `tiebreak` (ascending). With this ranking,
    "in top-k iff rank <= k" selects exactly k candidates from every scoring.
    """
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], tiebreak[i]))
    ranks = [0] * len(scores)
    for pos, i in enumerate(order):
        ranks[i] = pos + 1
    return ranks


def cutoff_tie(scores: Sequence[float], k: int) -> bool:
    """True iff the k-th and (k+1)-th best scores are equal, i.e. the top-k set
    is not determined by the scores alone (the case the exact-k rule settles)."""
    if k >= len(scores):
        return False
    s = sorted(scores, reverse=True)
    return s[k - 1] == s[k]


# ---------------------------------------------------------------------------
# Synthetic stand-in corpus (validation only)
# ---------------------------------------------------------------------------

_STUB_BANNER = "SYNTHETIC STAND-IN DOCUMENT - NOT A REAL RESUME - GENERATED FOR PIPELINE VALIDATION"

_STUB_SKILLS = {
    "INFORMATION-TECHNOLOGY": ["network administration", "python scripting", "cloud migration",
                               "incident response", "database tuning", "ci/cd pipelines"],
    "HEALTHCARE": ["patient intake", "clinical documentation", "medication administration",
                   "care coordination", "vital sign monitoring", "discharge planning"],
    "TEACHER": ["lesson planning", "classroom management", "curriculum design",
                "student assessment", "parent conferences", "differentiated instruction"],
    "ACCOUNTANT": ["general ledger reconciliation", "accounts payable", "audit preparation",
                   "tax filing", "financial reporting", "budget forecasting"],
    "ENGINEERING": ["cad modelling", "tolerance analysis", "failure investigation",
                    "process improvement", "test fixture design", "supplier qualification"],
    "SALES": ["pipeline management", "territory planning", "contract negotiation",
              "account retention", "quota attainment", "lead qualification"],
}


def make_stub_corpus(n: int, seed: int) -> list[Resume]:
    rng = random.Random(seed)
    cats = list(_STUB_SKILLS)
    male = list(_STUB_NAMES["M"])
    female = list(_STUB_NAMES["F"])
    out: list[Resume] = []
    for i in range(n):
        cat = cats[i % len(cats)]
        g = rng.choice("MF")
        first = (rng.choice(male) if g == "M" else rng.choice(female)).capitalize()
        pr = ("He", "him", "his") if g == "M" else ("She", "her", "her")
        skills = rng.sample(_STUB_SKILLS[cat], k=rng.randint(3, 6))
        years = rng.randint(1, 22)
        text = (
            f"{_STUB_BANNER}. {first} Placeholder-Example, {cat.title().replace('-', ' ')} "
            f"specialist. Summary: {first} Placeholder-Example has {years} years of stand-in "
            f"experience in {cat.lower().replace('-', ' ')}. {pr[0]} lists {pr[2]} core skills as "
            f"{', '.join(skills)}. Colleagues rated {pr[1]} on {pr[2]} delivery record. "
            f"Experience: fictional employer {rng.randint(1, 9)} - performed "
            f"{', '.join(rng.sample(_STUB_SKILLS[cat], k=3))} across {rng.randint(2, 8)} "
            f"invented projects. {pr[0]} reported to a fabricated supervisor and {pr[0].lower()} "
            f"maintained {pr[2]} certifications. Education: Placeholder University, "
            f"{rng.choice(['B.S.', 'B.A.', 'M.S.'])} in a made-up field, "
            f"{2026 - years - rng.randint(0, 4)}. Additional: this document contains no real "
            f"person's information and exists only so the scoring pipeline can be validated. "
            f"Keywords: {', '.join(_STUB_SKILLS[cat])}."
        )
        out.append(Resume(rid=f"stub-{i:04d}", category=cat, text=text))
    return out


def make_stub_queries(seed: int) -> tuple[list[Query], dict]:
    stereo = {"INFORMATION-TECHNOLOGY": "male", "HEALTHCARE": "female", "TEACHER": "female",
              "ACCOUNTANT": "neutral", "ENGINEERING": "male", "SALES": "neutral"}
    qs = []
    for i, (cat, skills) in enumerate(_STUB_SKILLS.items()):
        qs.append(Query(qid=f"00-{1000+i}.00", category=cat,
                        title=cat.title().replace("-", " "),
                        text=(f"{cat.title().replace('-', ' ')}. Stand-in job description for "
                              f"pipeline validation. Responsibilities include "
                              f"{', '.join(skills)}."),
                        stereotype=stereo[cat]))
    return qs, {"onet_file": "STUB", "category_to_soc": "STUB", "n_queries": len(qs),
                "unmatched_soc": [], "include_tasks": 0,
                "stereotype_counts": dict(Counter(q.stereotype for q in qs))}


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------


def sample_pools(queries: Sequence[Query], resumes: Sequence[Resume], k: int,
                 pools_per_query: int, relevant_per_pool: int, seed: int) -> list[dict]:
    """Pools mix on-category and off-category resumes, so relevance and quality
    genuinely vary and the focal candidate sits near a real decision boundary."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Resume]] = defaultdict(list)
    for r in resumes:
        by_cat[r.category].append(r)
    others_all = list(resumes)
    pools = []
    for q in queries:
        on = by_cat.get(q.category, [])
        for p in range(pools_per_query):
            n_on = min(relevant_per_pool, len(on), k)
            chosen = rng.sample(on, n_on) if n_on else []
            picked = {r.rid for r in chosen}
            guard = 0
            while len(chosen) < k and guard < 10000:
                guard += 1
                cand = rng.choice(others_all)
                if cand.rid in picked:
                    continue
                picked.add(cand.rid)
                chosen.append(cand)
            if len(chosen) < k:
                continue
            pools.append({"qid": q.qid, "pool_id": f"{q.qid}#{p}", "members": chosen})
    return pools


def run_g3(args) -> dict:
    t_start = time.time()
    random.seed(args.seed)

    if args.demo:
        resumes = make_stub_corpus(args.demo_n, args.seed)
        queries, qprov = make_stub_queries(args.seed)
        name_table, nprov = load_name_gender(allow_stub=True, min_count=1)
        corpus_prov = {"source": "SYNTHETIC STAND-IN generated by make_stub_corpus()",
                       "n_resumes": len(resumes)}
    else:
        resumes = load_corpus(args.corpus)
        queries, qprov = load_queries(args.onet, args.cat2soc, args.tasks, args.include_tasks)
        name_table, nprov = load_name_gender(args.names)
        corpus_prov = {"source": str(args.corpus or _first_existing(CORPUS_CANDIDATES)),
                       "n_resumes": len(resumes)}
        if args.max_resumes and len(resumes) > args.max_resumes:
            resumes = random.Random(args.seed).sample(resumes, args.max_resumes)
            corpus_prov["subsampled_to"] = len(resumes)
    if args.max_queries:
        queries = queries[:args.max_queries]

    print(f"[G3] {len(resumes)} resumes, {len(queries)} queries", flush=True)

    from ner_coverage import chunked, span_cache_load
    cached_ner = span_cache_load(args.span_cache, args.max_chars).get("ner_base")
    detector = None
    if cached_ner is not None:
        print(f"[G3] reusing cached ner_base spans for {len(cached_ner)} documents",
              flush=True)
    elif not args.no_ner:
        from deid_transforms import make_ner_detector
        print(f"[G3] loading NER {args.ner_model} ...", flush=True)
        detector = chunked(make_ner_detector(args.ner_model, device=args.device))

    twin_map = build_twin_name_map(name_table)
    t_ner = time.time()
    carried = detect_carried_gender(resumes, detector, name_table,
                                    head_chars=args.max_chars,
                                    allow_pronoun_gender=not args.name_gender_only,
                                    precomputed_names=cached_ner)
    ner_seconds = time.time() - t_ner
    print(f"[G3] gender markers actually carried by the corpus: {carried}", flush=True)
    forms = build_forms(resumes, name_table, twin_map, mode=args.twin_mode,
                        seed=args.seed, use_surname=not args.no_surname)
    resumes = [r for r in resumes if r.male_text and r.female_text]
    print(f"[G3] counterfactual-usable resumes: {len(resumes)}  {forms}", flush=True)
    if not resumes:
        raise SystemExit(
            "No resume could be given a male/female counterfactual pair. With "
            "--twin-mode swap this means the corpus carries no gender markers "
            "(the livecareer corpus is pre-anonymised); use --twin-mode auto."
        )
    owner = None
    labels_path = getattr(args, "marker_labels", None)
    if labels_path:
        labels, labels_sha = load_marker_labels(labels_path)
        owner = apply_marker_labels(resumes, labels, name_table, twin_map, seed=args.seed,
                                    surnames=[] if args.no_surname else DEFAULT_SURNAMES)
        owner.update(labels_file=str(labels_path), labels_sha256=labels_sha,
                     shape=shape_forms(resumes, args.cut_words, args.owner_line))
        print(f"[G3] owner-only markers: {owner}", flush=True)

    pools = sample_pools(queries, resumes, args.k, args.pools_per_query,
                         args.relevant_per_pool, args.seed)
    print(f"[G3] {len(pools)} pools of K={args.k}", flush=True)
    unlabelled = sorted({r.rid for p in pools for r in p["members"] if r.unlabelled_markers})
    if unlabelled:
        raise SystemExit(f"[G3] {len(unlabelled)} pooled resumes carry markers the labels "
                         f"do not cover, e.g. {unlabelled[:5]}")

    qtext = {q.qid: q.text for q in queries}
    qmeta = {q.qid: q for q in queries}

    import hashlib
    membership = [(p["qid"], r.rid) for p in pools for r in p["members"]]
    pool_hash = hashlib.sha256(
        json.dumps(membership, separators=(",", ":")).encode()).hexdigest()
    expect = getattr(args, "expect_pool_sha256", None)
    if expect and pool_hash != expect:
        raise SystemExit(f"[G3] pool membership {pool_hash} is not the expected {expect}: "
                         "these pools differ from the run being repeated")
    limit_pools = getattr(args, "limit_pools", 0)
    if limit_pools:
        # Canary: an even spread of the pools. The membership hash above still
        # identifies the full sample they come from.
        pools = pools[::max(1, len(pools) // limit_pools)][:limit_pools]
        membership = [(p["qid"], r.rid) for p in pools for r in p["members"]]
        print(f"[G3] canary: scoring {len(pools)} of the pools", flush=True)
    tb = {p["pool_id"]: [tiebreak_key(args.seed, p["pool_id"], r.rid) for r in p["members"]]
          for p in pools}
    fingerprint = {"script_version": SCRIPT_VERSION, "pool_membership_sha256": pool_hash,
                   "seed": args.seed, "K": args.k, "depths": list(args.depths),
                   "max_length": args.max_length, "twin_mode": args.twin_mode}
    if limit_pools:
        fingerprint["limit_pools"] = limit_pools
    if getattr(args, "female_first", False):
        fingerprint["female_first"] = True
    if owner:
        fingerprint.update(marker_labels_sha256=owner["labels_sha256"],
                           cut_words=args.cut_words, owner_line=args.owner_line)

    # Per-model checkpoint, so a crash does not lose finished models.
    ckpt_path: Path | None = getattr(args, "checkpoint", None)
    ckpt = {"fingerprint": fingerprint, "models": {}, "rows": {}}
    if ckpt_path and args.resume and ckpt_path.exists():
        old = json.loads(ckpt_path.read_text())
        if old.get("fingerprint") == fingerprint:
            ckpt = old
            print(f"[G3] resuming from {ckpt_path}: {sorted(ckpt['models'])} already done",
                  flush=True)
        else:
            print(f"[G3] ignoring checkpoint {ckpt_path}: fingerprint differs", flush=True)

    out_models = {}
    rows_by_model = {}
    for short in args.rerankers:
        model_name = RERANKERS[short]
        if short in ckpt["models"]:
            out_models[short] = ckpt["models"][short]
            rows_by_model[short] = ckpt["rows"][short]
            print(f"[G3] === {short}: taken from checkpoint ===", flush=True)
            continue
        print(f"[G3] === {short} ({model_name}) ===", flush=True)
        score_ckpt = (ckpt_path.with_name(f"{ckpt_path.stem}.{short}.scores.jsonl")
                      if ckpt_path else None)
        if short in API_RERANKERS:
            scorer = APIScorer(short, docs_per_request=getattr(args, "docs_per_request", 40),
                               tokens_per_request=getattr(args, "tokens_per_request", 60_000),
                               max_requests=getattr(args, "max_requests", 0),
                               coarse_retries=getattr(args, "coarse_retries", 8),
                               checkpoint=score_ckpt, checkpoint_meta=fingerprint,
                               resume=args.resume,
                               female_first=getattr(args, "female_first", False))
        else:
            make = RecipeScorer if short in RECIPE_RERANKERS else Scorer
            scorer = make(model_name, device=args.device, max_length=args.max_length,
                          batch_size=args.batch_size,
                          token_budget=getattr(args, "token_budget", 0),
                          attention_budget=getattr(args, "attention_budget", 0),
                          checkpoint=score_ckpt, checkpoint_meta=fingerprint,
                          resume=args.resume, revision=getattr(args, "revision", None),
                          dtype=getattr(args, "model_dtype", None))

        keys, pairs = [], []
        for pool in pools:
            qt = qtext[pool["qid"]]
            for r in pool["members"]:
                for f in ("male", "female"):
                    keys.append((pool["qid"], r.rid, f))
                    pairs.append((qt, r.form(f)))
        print(f"[G3] {len(set(keys))} unique (query, resume, form) pairs to score", flush=True)
        scorer.score(pairs, keys)

        # A cross-encoder truncates at max_length tokens. If a document's only
        # gender marker sits beyond that window, the model sees two IDENTICAL
        # inputs and must tie -- which is a truncation artefact, not model
        # indifference. Flag it so the tie rate can be read correctly.
        window_chars = args.max_length * 4
        in_window = {r.rid: (r.form("male")[:window_chars] != r.form("female")[:window_chars])
                     for pool in pools for r in pool["members"]}
        # The character heuristic above is kept (its summary blocks are cited),
        # but the exact test is on the model's own tokenisation of the
        # (query, resume) pair, truncated as CrossEncoder.predict truncates it:
        # the marker is in the scored window iff the two forms' token ids differ.
        # A hosted model has no local tokenizer, but APIScorer has it read the whole
        # document, so there the marker is in the window wherever the two forms differ.
        tok = None if isinstance(scorer, APIScorer) else scorer.model.tokenizer
        tok_max = int(getattr(tok, "model_max_length", 0) or 0) if tok else scorer.max_length
        in_tok_window: dict[tuple, bool] = {}
        for pool in pools:
            qt = qtext[pool["qid"]]
            for r in pool["members"]:
                key = (pool["qid"], r.rid)
                if key not in in_tok_window:
                    if tok is None:
                        in_tok_window[key] = r.form("male") != r.form("female")
                        continue
                    enc = tok([[qt, r.form("male")], [qt, r.form("female")]], truncation=True)
                    in_tok_window[key] = enc["input_ids"][0] != enc["input_ids"][1]

        rows = []
        filler_scores = {}
        for pool in pools:
            q = qmeta[pool["qid"]]
            members = pool["members"]
            keys_tb = tb[pool["pool_id"]]
            # Non-focal members keep their assigned filler form in BOTH conditions,
            # so the only thing that differs between the two scorings of the pool
            # is the focal candidate's gender markers.
            base = [scorer.cache[(q.qid, r.rid, r.default_form)] for r in members]
            filler_scores[pool["pool_id"]] = base
            for idx, focal in enumerate(members):
                s_m = list(base)
                s_f = list(base)
                s_m[idx] = scorer.cache[(q.qid, focal.rid, "male")]
                s_f[idx] = scorer.cache[(q.qid, focal.rid, "female")]
                avg_m, avg_f = average_ranks(s_m), average_ranks(s_f)
                cut_tie, focal_tie, legacy_n = {}, {}, {}
                for k in args.depths:
                    pair = []
                    for s in (s_m, s_f):
                        srt = sorted(s, reverse=True)
                        t = k < len(s) and srt[k - 1] == srt[k]
                        pair.append((t, t and s[idx] == srt[k - 1]))
                    cut_tie[str(k)] = [pair[0][0], pair[1][0]]
                    focal_tie[str(k)] = [pair[0][1], pair[1][1]]
                    legacy_n[str(k)] = [sum(a <= k for a in avg_m), sum(a <= k for a in avg_f)]
                rows.append({
                    "pool_id": pool["pool_id"], "qid": q.qid, "category": q.category,
                    "stereotype": q.stereotype, "focal_index": idx, "focal_rid": focal.rid,
                    "focal_category": focal.category,
                    "on_category": focal.category == q.category,
                    "mode": focal.mode, "carried_gender": focal.carried_gender,
                    "marker_in_scored_window": in_window[focal.rid],
                    "marker_in_token_window": in_tok_window[(q.qid, focal.rid)],
                    "gender_source": focal.carried_source or focal.mode,
                    # exact-k ordinal ranks (seeded tie-break), used for top-k
                    "rank_male": ordinal_ranks(s_m, keys_tb)[idx],
                    "rank_female": ordinal_ranks(s_f, keys_tb)[idx],
                    # tie-neutral average ranks, used for mean ranks and the legacy rule
                    "avg_rank_male": avg_m[idx], "avg_rank_female": avg_f[idx],
                    "score_male": s_m[idx], "score_female": s_f[idx],
                    "delta": s_m[idx] - s_f[idx],
                    "cutoff_tie": cut_tie, "focal_in_cutoff_tie": focal_tie,
                    "legacy_selected_n": legacy_n,
                })
        summary = summarise_flips(rows, args.depths)
        n_out_tok = sum(not r["marker_in_token_window"] for r in rows)
        out_models[short] = {
            "model": model_name,
            "pairs_per_second": round(scorer.pairs_per_second, 2),
            "n_pairs_scored": scorer.n_scored,
            "n_pairs_resumed_from_checkpoint": scorer.n_resumed,
            "n_pairs_identical_inputs_scored_once": scorer.n_scored_once,
            "score_seconds": round(scorer.score_seconds, 1),
            "load_seconds": round(scorer.load_seconds, 1),
            "xlm_roberta_compat_shim_applied": scorer.shim_applied,
            "summary": summary,
            "legacy_average_rank_rule": {
                "rule": "in top-k iff average rank <= k (can select != k at a tied cutoff)",
                "overall": _flip_block(rows, args.depths, "avg_rank_male", "avg_rank_female"),
            },
            "cutoff_ties": cutoff_tie_stats(rows, args.depths),
            "token_window": {
                "tokenizer_model_max_length": tok_max,
                "n_comparisons": len(rows),
                "n_marker_outside_token_window": n_out_tok,
                "n_marker_outside_token_window_by_mode": dict(Counter(
                    r["mode"] for r in rows if not r["marker_in_token_window"])),
                "n_distinct_query_resume_pairs_outside": sum(not v for v in in_tok_window.values()),
            },
            "rows": None,
            "rows_file": None,  # filled in by main()
        }
        if isinstance(scorer, APIScorer):
            out_models[short]["api"] = {
                "vendor": scorer.vendor, "settings": scorer.settings,
                "documented_context_tokens": scorer.max_length,
                "docs_per_request": scorer.docs_per_request,
                "tokens_per_request": scorer.tokens_per_request,
                "max_requests": scorer.max_requests, "billed_unit": scorer.billed_unit,
                "coarse_retries": scorer.coarse_retries, **scorer.ledger}
            for row in rows:
                row["api_coarse_response"] = any(
                    (row["qid"], row["focal_rid"], f) in scorer.coarse_keys
                    for f in ("male", "female"))
        rows_by_model[short] = {"filler_scores": filler_scores, "rows": rows}
        del scorer
        if ckpt_path:
            ckpt["models"][short] = out_models[short]
            ckpt["rows"][short] = rows_by_model[short]
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ckpt))
            tmp.replace(ckpt_path)
            print(f"[G3] checkpoint -> {ckpt_path}", flush=True)

    rows_payload = {
        "script_version": SCRIPT_VERSION,
        "seed": args.seed,
        "pool_membership_sha256": pool_hash,
        "description": (
            "Comparison-level rows of the G3 companion run. One row per (pool, focal "
            "candidate): the focal candidate's score in its male and female form, with "
            "every other pool member held at its filler form. `pools` gives each pool's "
            "members in order, their filler form and seeded tie-break key; "
            "`models.<m>.filler_scores` gives every member's filler-form score, so any "
            "ranking rule can be recomputed. `resumes` gives each scored resume's "
            "marker-construction mode and the names used."),
        "pools": [{"pool_id": p["pool_id"], "qid": p["qid"],
                   "member_rids": [r.rid for r in p["members"]],
                   "member_filler_form": [r.default_form for r in p["members"]],
                   "member_tiebreak_key": tb[p["pool_id"]]} for p in pools],
        "resumes": {r.rid: {"category": r.category, "mode": r.mode,
                            "male_given": r.injected[0] if r.injected else None,
                            "female_given": r.injected[1] if r.injected else None,
                            "surname": r.injected[2] if r.injected else None,
                            "carried_gender": r.carried_gender,
                            "carried_source": r.carried_source,
                            "filler_form": r.default_form,
                            **({"owner_markers": r.owner_markers} if owner else {})}
                    for p in pools for r in p["members"]},
        "models": rows_by_model,
    }

    return {
        "experiment": "G3 flip rate at the margin",
        "script_version": SCRIPT_VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_only": bool(args.demo),
        "full_corpus": not (args.demo or corpus_prov.get("subsampled_to")
                            or args.max_queries or limit_pools),
        "canary_limit_pools": limit_pools,
        "seed": args.seed,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "K": args.k, "pools_per_query": args.pools_per_query,
            "relevant_per_pool": args.relevant_per_pool, "depths": args.depths,
            "max_length": args.max_length, "batch_size": args.batch_size,
            "token_budget": getattr(args, "token_budget", 0),
            "attention_budget": getattr(args, "attention_budget", 0),
            "revision": getattr(args, "revision", None),
            "model_dtype": getattr(args, "model_dtype", None),
            "female_first": getattr(args, "female_first", False),
            "marker_labels": str(labels_path) if owner else None,
            "cut_words": args.cut_words if owner else None,
            "owner_line": (OWNER_LINE if args.owner_line else None) if owner else None,
            "batching": ("length-sorted, both forms of a (query, resume) in one batch, "
                         "identical truncated inputs scored once"
                         if getattr(args, "token_budget", 0) else
                         "insertion order, batch_size inputs per batch"),
            "device": args.device, "ner_model": None if args.no_ner else args.ner_model,
            "tie_convention": (
                "exact-k: each scoring of a pool selects exactly k candidates; members "
                "are ordered by score, and equal scores by a seeded key "
                "sha256(seed|pool_id|resume_id) that is identical in the male-form and "
                "female-form scorings of a comparison; in-top-k iff ordinal rank <= k. "
                "Mean ranks use average ranks. The earlier average-rank rule is "
                "reported per model under legacy_average_rank_rule."),
            "tie_break": "sha256(f'{seed}|{pool_id}|{resume_id}') -> first 8 bytes / 2**64, ascending",
        },
        "provenance": {
            "corpus": corpus_prov, "queries": qprov, "names": nprov,
            "sampling": {
                "n_distinct_resumes_scored": len({r.rid for p in pools for r in p["members"]}),
                "n_pools": len(pools),
                "n_focal_comparisons": sum(len(p["members"]) for p in pools),
                "n_distinct_query_resume_pairs": len(set(membership)),
                "pool_membership_sha256": pool_hash,
            },
            "deid_module": "code/deid_transforms.py (not used by G3; the two forms "
                           "are gender counterfactuals, not de-identified text)",
            "gender_markers_carried_by_corpus": carried,
            "counterfactual_construction": forms,
            "owner_only_markers": owner,
            "twin_mode": args.twin_mode,
            "n_resumes_usable": len(resumes),
            "ner_seconds": round(ner_seconds, 1),
        },
        "models": out_models,
        "wall_seconds": round(time.time() - t_start, 1),
        "_rows_payload": rows_payload,
    }


def _flip_block(sub: list[dict], depths: Sequence[int], fm: str, ff: str) -> dict:
    """Top-k shortlist flips, deciding membership from rank fields fm / ff."""
    n = len(sub)
    d: dict = {"n": n}
    for k in depths:
        m_in = [r[fm] <= k for r in sub]
        f_in = [r[ff] <= k for r in sub]
        male_only = sum(a and not b for a, b in zip(m_in, f_in))
        female_only = sum(b and not a for a, b in zip(m_in, f_in))
        flips = male_only + female_only
        d[f"top{k}"] = {
            "flip_rate_pct": round(100 * flips / n, 3) if n else None,
            "n_flips": flips,
            "male_form_in_only": male_only,
            "female_form_in_only": female_only,
            "net_male_advantage": male_only - female_only,
            "both_in": sum(a and b for a, b in zip(m_in, f_in)),
            "both_out": sum((not a) and (not b) for a, b in zip(m_in, f_in)),
        }
    return d


def cutoff_tie_stats(rows: list[dict], depths: Sequence[int]) -> dict:
    """How often a score tie straddles the top-k cutoff, and what it changes."""
    out = {}
    n = len(rows)
    for k in depths:
        ks = str(k)
        tie_scorings = sum(r["cutoff_tie"][ks][0] + r["cutoff_tie"][ks][1] for r in rows)
        sizes = Counter(x for r in rows for x in r["legacy_selected_n"][ks])

        def flipped(r, fm, ff):
            return (r[fm] <= k) != (r[ff] <= k)
        changed = sum(flipped(r, "rank_male", "rank_female")
                      != flipped(r, "avg_rank_male", "avg_rank_female") for r in rows)
        out[f"top{k}"] = {
            "n_scorings": 2 * n,
            "n_scorings_with_tie_at_cutoff": tie_scorings,
            "pct_scorings_with_tie_at_cutoff": round(100 * tie_scorings / (2 * n), 3) if n else None,
            "n_comparisons_with_tie_at_cutoff_in_either_form":
                sum(any(r["cutoff_tie"][ks]) for r in rows),
            "n_comparisons_focal_in_cutoff_tie_in_either_form":
                sum(any(r["focal_in_cutoff_tie"][ks]) for r in rows),
            "n_pools_with_any_cutoff_tie":
                len({r["pool_id"] for r in rows if any(r["cutoff_tie"][ks])}),
            "legacy_rule_shortlist_size_counts": {str(s): c for s, c in sorted(sizes.items())},
            "n_comparisons_flip_status_differs_from_legacy": changed,
        }
    return out


def summarise_flips(rows: list[dict], depths: Sequence[int]) -> dict:
    def block(sub: list[dict]) -> dict:
        n = len(sub)
        if not n:
            return {"n": 0}
        d = _flip_block(sub, depths, "rank_male", "rank_female")
        deltas = [r["delta"] for r in sub]
        ties = sum(d0 == 0 for d0 in deltas)
        d["score_gap"] = {
            "mean_delta_male_minus_female": round(sum(deltas) / n, 6),
            "mean_abs_delta": round(sum(abs(x) for x in deltas) / n, 6),
            "pct_male_favored": round(100 * sum(x > 0 for x in deltas) / n, 2),
            "pct_tie": round(100 * ties / n, 2),
            "pct_female_favored": round(100 * sum(x < 0 for x in deltas) / n, 2),
            "tie_aware_pct_male": round(100 * (sum(x > 0 for x in deltas) + 0.5 * ties) / n, 2),
        }
        # Mean ranks are tie-neutral: average ranks, as before the exact-k change.
        d["mean_rank_male"] = round(sum(r["avg_rank_male"] for r in sub) / n, 4)
        d["mean_rank_female"] = round(sum(r["avg_rank_female"] for r in sub) / n, 4)
        return d

    out = {"overall": block(rows)}
    out["by_stereotype"] = {s: block([r for r in rows if r["stereotype"] == s])
                            for s in sorted({r["stereotype"] for r in rows})}
    out["by_category"] = {c: block([r for r in rows if r["category"] == c])
                          for c in sorted({r["category"] for r in rows})}
    out["on_category_focal"] = block([r for r in rows if r["on_category"]])
    out["off_category_focal"] = block([r for r in rows if not r["on_category"]])
    out["by_mode"] = {m: block([r for r in rows if r["mode"] == m])
                      for m in sorted({r["mode"] for r in rows if r["mode"]})}
    # Rows whose gender marker survives the cross-encoder's truncation window.
    # Ties outside this set are an artefact of truncation, not of the model.
    out["marker_in_scored_window"] = block([r for r in rows if r["marker_in_scored_window"]])
    out["marker_outside_scored_window"] = block(
        [r for r in rows if not r["marker_in_scored_window"]])
    out["marker_in_token_window"] = block([r for r in rows if r["marker_in_token_window"]])
    out["marker_outside_token_window"] = block(
        [r for r in rows if not r["marker_in_token_window"]])
    out["by_gender_source"] = {g: block([r for r in rows if r["gender_source"] == g])
                               for g in sorted({r["gender_source"] for r in rows if r["gender_source"]})}
    return out


# ---------------------------------------------------------------------------


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--corpus", type=Path, default=None, help="resume corpus (csv/jsonl/json/parquet)")
    ap.add_argument("--onet", type=Path, default=None, help="O*NET Occupation Data.txt")
    ap.add_argument("--tasks", type=Path, default=None, help="O*NET Task Statements.txt")
    ap.add_argument("--cat2soc", type=Path, default=None, help="category_to_soc.csv")
    ap.add_argument("--names", type=Path, default=None, help="SSA name->gender table (csv or zip)")
    ap.add_argument("--include-tasks", type=int, default=0, help="append N O*NET task statements to each query")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-resumes", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--ner-model", default="dslim/bert-base-NER")
    ap.add_argument("--no-ner", action="store_true", help="skip NER (pronoun-only gender signal)")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="truncate each resume to this many characters before use")
    ap.add_argument("--span-cache", type=Path, default=None,
                    help="JSON of cached detector spans (see ner_coverage.span_cache_load)")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic stand-in corpus")
    ap.add_argument("--demo-n", type=int, default=120)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--rerankers", nargs="+", default=["bge-m3"], choices=sorted(RERANKERS))
    ap.add_argument("--k", type=int, default=20, help="pool size")
    ap.add_argument("--pools-per-query", type=int, default=10)
    ap.add_argument("--relevant-per-pool", type=int, default=5)
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--twin-mode", default="auto", choices=("auto", "swap", "inject"),
                    help="how to build the male/female counterfactual pair (see build_forms)")
    ap.add_argument("--no-surname", action="store_true",
                    help="inject a given name only, no surname")
    ap.add_argument("--name-gender-only", action="store_true",
                    help="only use name-derived gender (no pronoun fallback)")
    ap.add_argument("--dump-rows", action="store_true",
                    help="also write per-focal rows inline into the main JSON "
                         "(they always go to the companion *_rows.json.gz file)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rows-out", type=Path, default=None,
                    help="companion file for comparison-level rows "
                         "(default: <out stem>_rows.json.gz next to --out)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="per-model checkpoint (default: results/partial/<out stem>.checkpoint.json)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished models, and the finished batches of an unfinished "
                         "one, from a matching checkpoint")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="plan batches by input length, at most this many padded tokens each "
                         "(0: insertion order, --batch-size inputs per batch, as the banked run)")
    ap.add_argument("--attention-budget", type=int, default=0,
                    help="with --token-budget, also cap each batch's inputs x padded length "
                         "squared, which bounds its attention memory (0: no cap)")
    ap.add_argument("--revision", default=None,
                    help="model revision to load (pin it to the audit's, see run_records.py)")
    ap.add_argument("--model-dtype", default=None, choices=("bfloat16", "float16", "float32"),
                    help="load the model in this precision, as the audit's recipe did "
                         "(default: the library's, float32)")
    ap.add_argument("--docs-per-request", type=int, default=40,
                    help="hosted rerankers: at most this many documents per request")
    ap.add_argument("--tokens-per-request", type=int, default=60_000,
                    help="hosted rerankers: at most this many estimated tokens per request")
    ap.add_argument("--female-first", action="store_true",
                    help="request layout: each unit's female form first, not its male form "
                         "(the counterbalancing pass for an order-sensitive listwise model)")
    ap.add_argument("--coarse-retries", type=int, default=8,
                    help="Voyage: send a request whose scores all come back rounded to "
                         "bfloat16 again, up to this many times")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="hosted rerankers: stop after this many requests in all "
                         "(0: 1.3 x the planned requests + 10)")
    ap.add_argument("--limit-pools", type=int, default=0,
                    help="canary: score only this many pools, spread evenly over the sample")
    ap.add_argument("--expect-pool-sha256", default=None,
                    help="abort unless the sampled pools have this membership hash")
    ap.add_argument("--marker-labels", type=Path, default=None,
                    help="hand-checked labels of which carried names and pronouns are the "
                         "owner's; only those change between the forms (apply_marker_labels)")
    ap.add_argument("--cut-words", type=int, default=0,
                    help="with --marker-labels: cut each body so that it (with --owner-line, "
                         "line and body together) holds this many words after the name "
                         "(0: whole)")
    ap.add_argument("--owner-line", action="store_true",
                    help="with --marker-labels: put OWNER_LINE, three of the owner's "
                         "pronouns, after the prepended name")
    args = ap.parse_args(argv)
    if (args.cut_words or args.owner_line) and not args.marker_labels:
        ap.error("--cut-words and --owner-line need --marker-labels")
    if args.cut_words < 0 or (args.owner_line and 0 < args.cut_words <= OWNER_LINE_WORDS):
        ap.error(f"--cut-words must be 0 or, with --owner-line, above {OWNER_LINE_WORDS}")

    RESULTS.mkdir(exist_ok=True)
    tag = "demo" if args.demo else "_".join(args.rerankers)
    out = args.out or RESULTS / f"realistic_pool_{tag}.json"
    rows_out = args.rows_out or out.with_name(out.stem + "_rows.json.gz")
    if args.checkpoint is None:
        args.checkpoint = RESULTS / "partial" / f"{out.stem}.checkpoint.json"

    res = run_g3(args)
    rows_payload = res.pop("_rows_payload")
    try:
        rows_rel = str(rows_out.resolve().relative_to(ROOT))
    except ValueError:
        rows_rel = str(rows_out)
    for short, m in res["models"].items():
        m["rows_file"] = rows_rel
        if args.dump_rows:
            m["rows"] = rows_payload["models"][short]["rows"]
    import gzip
    rows_payload["main_results_file"] = out.name
    with gzip.open(rows_out, "wt", encoding="utf-8") as fh:
        json.dump(rows_payload, fh, separators=(",", ":"))
    out.write_text(json.dumps(res, indent=2))
    print(f"[G3] wrote rows -> {rows_out}")

    print(f"\n[G3] wrote {out}")
    for short, m in res["models"].items():
        s = m["summary"]["overall"]
        print(f"  {short:9s} n={s['n']:6d}  " +
              "  ".join(f"flip@{k}={s[f'top{k}']['flip_rate_pct']:5.2f}% "
                        f"(M+{s[f'top{k}']['male_form_in_only']}/F+{s[f'top{k}']['female_form_in_only']})"
                        for k in args.depths) +
              f"  {m['pairs_per_second']:.1f} pairs/s")
    if res["validation_only"]:
        print("\n  *** validation_only: synthetic stand-in corpus. Not evidence. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
