"""G6 -- name-detection coverage on real resumes, and a trained-model baseline.

REVIEW_TODO.md section G6 and item B8.

Three questions:

  1. The paper's de-identification step 2 says "replace names". The synthetic
     experiment implements that with the dataset's own 20-name list
     (`synthetic_dataset.NAME_PAIRS`). What fraction of the PERSON entities in a
     real resume corpus would that list have caught? Expected: nearly none. That
     number is what turns "step 2 needs a name dictionary or NER in the field"
     from an aside into a measured limitation.

  2. What is NER's own recall? A hand-checked sample is the only honest answer,
     so this script does two things: it reports agreement between three
     independent detectors (which is NOT ground truth and is labelled as such
     everywhere), and it emits an audit CSV of 100 documents for a human to
     check, which it will read back in and turn into a real recall figure.

  3. `obi/deid_roberta_i2b2` is an off-the-shelf trained de-identification model.
     How does it compare to the cheap three-rule regex? Its detection behaviour
     is measured here; the leakage (G4) and utility (G5) comparisons import
     `make_phi_detector` from this module.

This module is also the home of the detector factories used by leakage_probe.py
and utility_retention.py, so that every script masks names the same way.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realistic_pool_experiment import (  # noqa: E402
    RESULTS, load_corpus, load_name_gender,
    make_stub_corpus, add_common_args,
)

PERSON_PHI_LABELS = {"PATIENT", "STAFF", "DOCTOR", "HCW", "PATORG"}


# ---------------------------------------------------------------------------
# Detector factories -- every script masks names through one of these.
# ---------------------------------------------------------------------------


def chunked(detect_one: Callable[[str], list[str]], window: int = 1200,
            overlap: int = 100, limit: int | None = None) -> Callable[[str], list[str]]:
    """Run a detector over a long document in overlapping windows.

    Transformer token classifiers have a 512-token ceiling; resumes are longer.
    Windows overlap so a name is not cut in half at every boundary.
    """
    def detect(text: str) -> list[str]:
        t = text[:limit] if limit else text
        if len(t) <= window:
            return detect_one(t)
        found: list[str] = []
        step = max(1, window - overlap)
        for i in range(0, len(t), step):
            piece = t[i:i + window]
            if piece.strip():
                found.extend(detect_one(piece))
        return sorted(set(found), key=len, reverse=True)
    return detect


def make_regex_detector(names: Sequence[str]) -> Callable[[str], list[str]]:
    """The paper's own approach: a fixed list of literal names."""
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    def detect(text: str) -> list[str]:
        return sorted(set(pat.findall(text)), key=len, reverse=True)
    return detect


def _phi_backend(model_name: str, device: str | None, threshold: float):
    """One shared, memoised pass of the PHI model over each text chunk.

    The person-only and all-PHI views are two filters over the SAME model output.
    Running the model twice over 2,500 long resumes would cost ~40 wasted minutes.
    """
    from transformers import pipeline
    pipe = pipeline("token-classification", model=model_name,
                    aggregation_strategy="simple", device=device)
    memo: dict[str, list[tuple[str, str]]] = {}

    def raw(chunk: str) -> list[tuple[str, str]]:
        hit = memo.get(chunk)
        if hit is None:
            hit = [(e["word"].strip(), str(e.get("entity_group", "")).upper())
                   for e in pipe(chunk) if e.get("score", 0) >= threshold]
            memo[chunk] = hit
        return hit

    return raw


def _phi_view(raw, person_only: bool) -> Callable[[str], list[str]]:
    def detect_one(text: str) -> list[str]:
        out = [w for w, grp in raw(text)
               if len(w) > 1 and (not person_only or grp in PERSON_PHI_LABELS)]
        parts = {p for s in out for p in s.split() if len(p) > 1}
        return sorted(set(out) | parts, key=len, reverse=True)
    return chunked(detect_one)


def make_phi_detector(model_name: str = "obi/deid_roberta_i2b2",
                      device: str | None = None,
                      person_only: bool = True,
                      threshold: float = 0.80) -> Callable[[str], list[str]]:
    """Trained PHI de-identification model as a name detector.

    person_only=True  -> only PATIENT/STAFF-style spans (comparable to NER PERSON).
    person_only=False -> every PHI span it flags (the realistic "just run the
                         off-the-shelf de-id tool" condition).
    """
    return _phi_view(_phi_backend(model_name, device, threshold), person_only)


def make_detectors(which: Sequence[str], device: str = "mps",
                   paper_names: Sequence[str] | None = None) -> dict[str, Callable]:
    from deid_transforms import make_ner_detector
    out: dict[str, Callable] = {}
    phi_raw = None
    if {"deid_roberta", "deid_roberta_allphi"} & set(which):
        phi_raw = _phi_backend("obi/deid_roberta_i2b2", device, 0.80)
    for name in which:
        if name == "paper_regex":
            out[name] = make_regex_detector(paper_names or [])
        elif name == "ner_base":
            out[name] = chunked(make_ner_detector("dslim/bert-base-NER", device=device))
        elif name == "ner_large":
            out[name] = chunked(make_ner_detector(
                "dbmdz/bert-large-cased-finetuned-conll03-english", device=device))
        elif name == "deid_roberta":
            out[name] = _phi_view(phi_raw, person_only=True)
        elif name == "deid_roberta_allphi":
            out[name] = _phi_view(phi_raw, person_only=False)
        else:
            raise SystemExit(f"unknown detector {name!r}")
    return out


def span_cache_load(path, max_chars: int) -> dict[str, dict[str, list[str]]]:
    """Reuse detector output across runs.

    A full NER pass over 2,500 long resumes costs ~20 minutes and the trained PHI
    model ~20 more. Those spans depend only on (detector, document, truncation),
    never on the analysis, so caching them makes re-running an experiment with
    different probe or ranking settings a seconds-long operation instead of an
    hour. The cache records `max_chars` and refuses to serve a mismatch.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        blob = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("max_chars") != max_chars:
        print(f"[cache] ignoring {p}: built at max_chars={blob.get('max_chars')}, "
              f"need {max_chars}", flush=True)
        return {}
    return blob.get("detectors", {})


def span_cache_save(path, max_chars: int, spans: dict[str, dict[str, list[str]]]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"max_chars": max_chars, "detectors": spans}))


def paper_name_list() -> list[str]:
    from synthetic_dataset import NAME_PAIRS
    return [n for pair in NAME_PAIRS for n in pair]


# ---------------------------------------------------------------------------


def _tokens(spans: Sequence[str]) -> set[str]:
    """Word-level view of a span set, so detectors that split names differently
    are compared fairly."""
    out = set()
    for s in spans:
        for t in re.split(r"\s+", s):
            t = re.sub(r"[^A-Za-z']", "", t)
            if len(t) > 1:
                out.add(t.lower())
    return out


def run_g6(args) -> dict:
    t0 = time.time()
    rng = random.Random(args.seed)

    if args.demo:
        resumes = make_stub_corpus(args.demo_n, args.seed)
        corpus_prov = {"source": "SYNTHETIC STAND-IN", "n_resumes": len(resumes)}
        name_table, nprov = load_name_gender(allow_stub=True, min_count=1)
    else:
        resumes = load_corpus(args.corpus)
        corpus_prov = {"source": str(args.corpus or "auto"), "n_resumes": len(resumes)}
        name_table, nprov = load_name_gender(args.names)
    rng.shuffle(resumes)
    sample = resumes[:args.sample]
    print(f"[G6] {len(sample)} resumes sampled from {len(resumes)}", flush=True)

    papern = paper_name_list()
    cache = span_cache_load(args.span_cache, args.max_chars)
    # Only build the detectors the cache cannot already answer for every sampled
    # document; a cached ner_base or deid_roberta saves a ~20-minute pass each.
    fresh = [d for d in args.detectors
             if not (d in cache and all(r.rid in cache[d] for r in sample))]
    if len(fresh) < len(args.detectors):
        print(f"[G6] cache supplies {sorted(set(args.detectors) - set(fresh))}", flush=True)
    dets = make_detectors(fresh, device=args.device, paper_names=papern)

    per_doc: dict[str, dict[str, list[str]]] = defaultdict(dict)
    timing = {}
    for dname in args.detectors:
        t = time.time()
        if dname in dets:
            for r in sample:
                per_doc[r.rid][dname] = dets[dname](r.text[:args.max_chars])
            cache.setdefault(dname, {}).update(
                {r.rid: per_doc[r.rid][dname] for r in sample})
        else:
            for r in sample:
                per_doc[r.rid][dname] = cache[dname][r.rid]
        timing[dname] = round(time.time() - t, 1)
        n_docs = sum(1 for r in sample if per_doc[r.rid][dname])
        print(f"[G6]   {dname:20s} {n_docs:4d}/{len(sample)} docs with >=1 span  "
              f"({timing[dname]}s{'' if dname in dets else ', cached'})", flush=True)
    span_cache_save(args.span_cache, args.max_chars, cache)

    # -- Q1: what would the paper's 20-name list have caught? --------------
    ref = args.reference
    covered = Counter()
    n_ref_tokens = 0
    n_docs_with_ref = 0
    n_docs_regex_would_mask = 0
    missed_examples = Counter()
    for r in sample:
        ref_tok = _tokens(per_doc[r.rid].get(ref, []))
        reg_tok = _tokens(per_doc[r.rid].get("paper_regex", []))
        if ref_tok:
            n_docs_with_ref += 1
            if ref_tok & reg_tok:
                n_docs_regex_would_mask += 1
        n_ref_tokens += len(ref_tok)
        covered["hit"] += len(ref_tok & reg_tok)
        for t in ref_tok - reg_tok:
            missed_examples[t] += 1

    q1 = {
        "reference_detector": ref,
        "n_docs": len(sample),
        "n_docs_with_person_entity": n_docs_with_ref,
        "pct_docs_with_person_entity": round(100 * n_docs_with_ref / max(1, len(sample)), 2),
        "n_person_name_tokens": n_ref_tokens,
        "n_caught_by_paper_20_name_list": covered["hit"],
        "pct_person_tokens_caught_by_paper_list": round(
            100 * covered["hit"] / max(1, n_ref_tokens), 3),
        "pct_docs_paper_list_would_mask_at_all": round(
            100 * n_docs_regex_would_mask / max(1, len(sample)), 3),
        "paper_name_list": papern,
        "most_common_missed_name_tokens": missed_examples.most_common(40),
    }

    # -- Q2: detector agreement (NOT ground truth) -------------------------
    agree = {}
    names = [d for d in args.detectors if d != "paper_regex"]
    for a in names:
        for b in names:
            if a >= b:
                continue
            tp = fp = fn = 0
            for r in sample:
                ta, tb = _tokens(per_doc[r.rid][a]), _tokens(per_doc[r.rid][b])
                tp += len(ta & tb)
                fp += len(ta - tb)
                fn += len(tb - ta)
            agree[f"{a}_vs_{b}"] = {
                "shared_tokens": tp, f"only_{a}": fp, f"only_{b}": fn,
                "jaccard": round(tp / max(1, tp + fp + fn), 4),
                f"{a}_recall_against_{b}": round(tp / max(1, tp + fn), 4),
                f"{b}_recall_against_{a}": round(tp / max(1, tp + fp), 4),
            }

    # -- Q3: how much does each detector actually remove? ------------------
    removal = {}
    for dname in args.detectors:
        toks = [len(_tokens(per_doc[r.rid][dname])) for r in sample]
        removal[dname] = {
            "mean_masked_tokens_per_doc": round(sum(toks) / max(1, len(toks)), 3),
            "docs_with_any": sum(1 for t in toks if t),
            "seconds": timing.get(dname),
        }

    # -- audit sample for a human ------------------------------------------
    audit_written = None
    if args.emit_audit:
        args.emit_audit.parent.mkdir(parents=True, exist_ok=True)
        audit = resumes[:args.audit_n]
        with args.emit_audit.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rid", "detector_spans", "resume_head_600_chars",
                        "human_true_person_names_semicolon_separated",
                        "human_notes"])
            for r in audit:
                spans = per_doc.get(r.rid, {}).get(ref) or cache.get(ref, {}).get(r.rid, [])
                w.writerow([r.rid, "; ".join(spans),
                            re.sub(r"\s+", " ", r.text[:600]), "", ""])
        audit_written = str(args.emit_audit)
        print(f"[G6] wrote audit sample for hand-checking: {audit_written}", flush=True)

    hand = None
    if args.audit_csv and args.audit_csv.exists():
        hand = score_audit(args.audit_csv, dets.get(ref), args.max_chars)
        print(f"[G6] hand-checked recall: {hand}", flush=True)

    return {
        "experiment": "G6 name-detection coverage and trained-model baseline",
        "validation_only": bool(args.demo),
        "full_corpus": bool(not args.demo and args.sample >= corpus_prov["n_resumes"]),
        "seed": args.seed,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"sample": len(sample), "max_chars": args.max_chars,
                   "detectors": list(args.detectors), "device": args.device,
                   "reference_detector": ref},
        "provenance": {"corpus": corpus_prov, "names": nprov,
                       "paper_name_list_source": "code/synthetic_dataset.py NAME_PAIRS",
                       "models": {"ner_base": "dslim/bert-base-NER",
                                  "ner_large": "dbmdz/bert-large-cased-finetuned-conll03-english",
                                  "deid_roberta": "obi/deid_roberta_i2b2"}},
        "q1_paper_name_list_coverage": q1,
        "q2_detector_agreement_NOT_ground_truth": agree,
        "q3_removal_volume": removal,
        "hand_checked_recall": hand,
        "audit_sample_written": audit_written,
        "wall_seconds": round(time.time() - t0, 1),
    }


def score_audit(path: Path, detector, max_chars: int) -> dict:
    """Turn a hand-filled audit CSV into a real recall/precision figure."""
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    tp = fp = fn = 0
    n_checked = 0
    for r in rows:
        truth_raw = (r.get("human_true_person_names_semicolon_separated") or "").strip()
        if not truth_raw:
            # A blank cell is ambiguous -- it means either "no person name in this
            # document" or "not annotated yet" -- so it is skipped rather than
            # guessed at. `n_documents_hand_checked` reports what was actually used,
            # which is what the recall figure is conditioned on.
            continue
        n_checked += 1
        truth = _tokens([t for t in truth_raw.split(";") if t.strip()])
        pred = _tokens([s for s in (r.get("detector_spans") or "").split(";") if s.strip()])
        tp += len(truth & pred)
        fn += len(truth - pred)
        fp += len(pred - truth)
    return {
        "n_documents_hand_checked": n_checked,
        "n_rows_in_file": len(rows),
        "true_positives": tp, "false_negatives": fn, "false_positives": fp,
        "recall": round(tp / max(1, tp + fn), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "note": "recall/precision at the level of name TOKENS, against a human annotation",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--detectors", nargs="+",
                    default=["paper_regex", "ner_base", "ner_large", "deid_roberta"])
    ap.add_argument("--reference", default="ner_base",
                    help="detector treated as the reference for Q1 coverage")
    ap.add_argument("--emit-audit", type=Path, default=None,
                    help="write a CSV of documents for a human to hand-check")
    ap.add_argument("--audit-n", type=int, default=100)
    ap.add_argument("--audit-csv", type=Path, default=None,
                    help="read back a hand-filled audit CSV and report real recall")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    res = run_g6(args)
    RESULTS.mkdir(exist_ok=True)
    out = args.out or RESULTS / "ner_coverage.json"
    out.write_text(json.dumps(res, indent=2))
    q = res["q1_paper_name_list_coverage"]
    print(f"\n[G6] wrote {out}")
    print(f"  docs with a PERSON entity      : {q['pct_docs_with_person_entity']}% "
          f"({q['n_docs_with_person_entity']}/{q['n_docs']})")
    print(f"  person-name tokens found       : {q['n_person_name_tokens']}")
    print(f"  caught by the paper's 20 names : {q['n_caught_by_paper_20_name_list']} "
          f"({q['pct_person_tokens_caught_by_paper_list']}%)")
    if res["validation_only"]:
        print("\n  *** validation_only: synthetic stand-in corpus. Not evidence. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
