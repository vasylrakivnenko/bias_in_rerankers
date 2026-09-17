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
}


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


def swap_pronouns(text: str, direction: str) -> str:
    """direction 'M2F' or 'F2M'. Resolves the his/her and her/his ambiguity by
    the following token, the same heuristic deid_transforms.gram_pos uses."""
    def sub(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if direction == "M2F":
            if lw == "his":
                rest = text[m.end():].lstrip()
                nxt = re.match(r"[A-Za-z']+", rest)
                repl = "hers" if (not nxt or nxt.group(0).lower() in _POSS_STOP) else "her"
            else:
                repl = _M2F.get(lw, lw)
        else:
            if lw == "her":
                rest = text[m.end():].lstrip()
                nxt = re.match(r"[A-Za-z']+", rest)
                repl = "him" if (not nxt or nxt.group(0).lower() in _POSS_STOP) else "his"
            else:
                repl = _F2M.get(lw, lw)
        return _match_case(w, repl)

    return _PRON_RE.sub(sub, text)


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
    """Batched cross-encoder scoring with a de-duplicating cache."""

    def __init__(self, model_name: str, device: str = "mps", max_length: int = 512,
                 batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.shim_applied = _patch_xlm_roberta_for_jina()
        t0 = time.time()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, trust_remote_code=True, device=device,
                                  max_length=max_length)
        self.load_seconds = time.time() - t0
        self.cache: dict[tuple, float] = {}
        self.n_scored = 0
        self.score_seconds = 0.0

    def score(self, pairs: Sequence[tuple[str, str]], keys: Sequence[tuple],
              progress: bool = True) -> None:
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
            if progress and (i // self.batch_size) % 20 == 0:
                done = min(i + self.batch_size, len(items))
                el = time.time() - t0
                rate = done / el if el else 0
                print(f"    {done}/{len(items)} pairs  {rate:6.1f} pairs/s", flush=True)
        self.score_seconds += time.time() - t0
        self.n_scored += len(items)

    @property
    def pairs_per_second(self) -> float:
        return self.n_scored / self.score_seconds if self.score_seconds else 0.0


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

    pools = sample_pools(queries, resumes, args.k, args.pools_per_query,
                         args.relevant_per_pool, args.seed)
    print(f"[G3] {len(pools)} pools of K={args.k}", flush=True)

    qtext = {q.qid: q.text for q in queries}
    qmeta = {q.qid: q for q in queries}

    out_models = {}
    for short in args.rerankers:
        model_name = RERANKERS[short]
        print(f"[G3] === {short} ({model_name}) ===", flush=True)
        scorer = Scorer(model_name, device=args.device, max_length=args.max_length,
                        batch_size=args.batch_size)

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

        rows = []
        for pool in pools:
            q = qmeta[pool["qid"]]
            members = pool["members"]
            # Non-focal members keep their assigned filler form in BOTH conditions,
            # so the only thing that differs between the two scorings of the pool
            # is the focal candidate's gender markers.
            base = [scorer.cache[(q.qid, r.rid, r.default_form)] for r in members]
            for idx, focal in enumerate(members):
                s_m = list(base)
                s_f = list(base)
                s_m[idx] = scorer.cache[(q.qid, focal.rid, "male")]
                s_f[idx] = scorer.cache[(q.qid, focal.rid, "female")]
                r_m = average_ranks(s_m)[idx]
                r_f = average_ranks(s_f)[idx]
                rows.append({
                    "pool_id": pool["pool_id"], "qid": q.qid, "category": q.category,
                    "stereotype": q.stereotype, "focal_rid": focal.rid,
                    "focal_category": focal.category,
                    "on_category": focal.category == q.category,
                    "mode": focal.mode, "carried_gender": focal.carried_gender,
                    "marker_in_scored_window": in_window[focal.rid],
                    "gender_source": focal.carried_source or focal.mode,
                    "rank_male": r_m, "rank_female": r_f,
                    "score_male": s_m[idx], "score_female": s_f[idx],
                    "delta": s_m[idx] - s_f[idx],
                })
        out_models[short] = {
            "model": model_name,
            "pairs_per_second": round(scorer.pairs_per_second, 2),
            "n_pairs_scored": scorer.n_scored,
            "score_seconds": round(scorer.score_seconds, 1),
            "load_seconds": round(scorer.load_seconds, 1),
            "xlm_roberta_compat_shim_applied": scorer.shim_applied,
            "summary": summarise_flips(rows, args.depths),
            "rows": rows if args.dump_rows else None,
        }
        del scorer

    return {
        "experiment": "G3 flip rate at the margin",
        "validation_only": bool(args.demo),
        "full_corpus": not (args.demo or corpus_prov.get("subsampled_to")
                            or args.max_queries),
        "seed": args.seed,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "K": args.k, "pools_per_query": args.pools_per_query,
            "relevant_per_pool": args.relevant_per_pool, "depths": args.depths,
            "max_length": args.max_length, "batch_size": args.batch_size,
            "device": args.device, "ner_model": None if args.no_ner else args.ner_model,
            "tie_convention": "average ranks; in-top-k iff average rank <= k",
        },
        "provenance": {
            "corpus": corpus_prov, "queries": qprov, "names": nprov,
            "sampling": {
                "n_distinct_resumes_scored": len({r.rid for p in pools for r in p["members"]}),
                "n_pools": len(pools),
                "n_focal_comparisons": sum(len(p["members"]) for p in pools),
            },
            "deid_module": "code/deid_transforms.py (not used by G3; the two forms "
                           "are gender counterfactuals, not de-identified text)",
            "gender_markers_carried_by_corpus": carried,
            "counterfactual_construction": forms,
            "twin_mode": args.twin_mode,
            "n_resumes_usable": len(resumes),
            "ner_seconds": round(ner_seconds, 1),
        },
        "models": out_models,
        "wall_seconds": round(time.time() - t_start, 1),
    }


def summarise_flips(rows: list[dict], depths: Sequence[int]) -> dict:
    def block(sub: list[dict]) -> dict:
        n = len(sub)
        if not n:
            return {"n": 0}
        d = {"n": n}
        for k in depths:
            m_in = [r["rank_male"] <= k for r in sub]
            f_in = [r["rank_female"] <= k for r in sub]
            male_only = sum(a and not b for a, b in zip(m_in, f_in))
            female_only = sum(b and not a for a, b in zip(m_in, f_in))
            flips = male_only + female_only
            d[f"top{k}"] = {
                "flip_rate_pct": round(100 * flips / n, 3),
                "n_flips": flips,
                "male_form_in_only": male_only,
                "female_form_in_only": female_only,
                "net_male_advantage": male_only - female_only,
                "both_in": sum(a and b for a, b in zip(m_in, f_in)),
                "both_out": sum((not a) and (not b) for a, b in zip(m_in, f_in)),
            }
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
        d["mean_rank_male"] = round(sum(r["rank_male"] for r in sub) / n, 4)
        d["mean_rank_female"] = round(sum(r["rank_female"] for r in sub) / n, 4)
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
    ap.add_argument("--dump-rows", action="store_true", help="write per-focal rows into the JSON")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    res = run_g3(args)
    RESULTS.mkdir(exist_ok=True)
    tag = "demo" if args.demo else "_".join(args.rerankers)
    out = args.out or RESULTS / f"realistic_pool_{tag}.json"
    out.write_text(json.dumps(res, indent=2))

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
