"""
Does de-identification cost ranking quality on whole real résumés?

The paper's full de-identification rule, tested on a ranking task that is not at ceiling.
Each of the corpus's job categories has the title and description of its matched O*NET
occupation as the query, as in the companion experiment (Appendix E). For each query,
SETS_PER_QUERY candidate sets of N_RELEVANT résumés from the query's category and N_OTHER
from the other categories are drawn with a fixed seed; a résumé is relevant iff its category
is the query's. Every résumé first gets a prepended given name and surname
(realistic_pool_experiment.build_forms, mode "inject", as in the companion experiment: the
corpus's publisher removed most names, and a submitted résumé carries one). Each set is then
ranked twice, with every document as written ("original") and after the full rule ("full",
deid_transforms.deidentify): the applicant's own name, which an employer has from the
application, and the other person names that dslim/bert-base-NER finds become "the candidate",
and every gendered pronoun "they", the replacements in lowercase and the rest of the text as
written. The detector groups word pieces into whole words (aggregation "first"; the default
"simple" returns fragments such as "Ke" for "Kelvin", which would then replace unrelated
words), keeps PER entities scored 0.80 or more, and drops pieces under three characters; the
output records how many of the prepended names it found on its own.

Per set: top-1 accuracy (credit shared among tied leaders), MRR and nDCG@10, ties at their
average rank; for each, the change from original to full, with a 95% interval from a
bootstrap that resamples the queries, each with all its sets.

Every input is read whole: the local models with an input limit above the longest input
(the output counts any input over it), the hosted ones through
realistic_pool_experiment.APIScorer (Voyage with truncation off, Cohere with
max_tokens_per_doc at the model's window), one request per candidate set. Finished batches
and requests are checkpointed (--checkpoint, --resume). --canary scores the first two
queries' first two sets of the full design; --prepare only runs the detector, whose spans are
cached by document hash in results/partial/utility_whole_spans.json (names from the corpus;
not released).

Reads:  the résumé corpus (data/PROVENANCE.md), the O*NET, category and SSA inputs that
        realistic_pool_experiment.py finds
Output: results/utility_whole/<model>.json (metrics, settings, the sets as résumé ids; no
        résumé text) and <model>_scores.json.gz (every score); canary_<model>.* for --canary
        --summarize: results/utility_whole/summary.json and results/tex/utility_whole.tex
        from every finished model's output
Run:    code/run_utility_whole.sh prepare|canary|full <model>, then
        venv/bin/python code/utility_whole_resume.py --summarize
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from deid_transforms import deidentify  # noqa: E402
from ner_coverage import chunked  # noqa: E402
from realistic_pool_experiment import (  # noqa: E402
    API_RERANKERS, DEFAULT_SEED, RECIPE_RERANKERS, RERANKERS, RESULTS, APIScorer, RecipeScorer,
    Scorer, average_ranks, build_forms, build_twin_name_map, load_corpus, load_name_gender,
    load_queries)

SCRIPT_VERSION = "utility_whole_resume.py v1.1"
N_RELEVANT, N_OTHER, SETS_PER_QUERY, NDCG_K = 3, 17, 10, 10
CANARY_QUERIES, CANARY_SETS = 2, 2
CONDITIONS = ("original", "full")
PLACEHOLDER = "the candidate"
DETECTOR = "dslim/bert-base-NER"
DETECTOR_SETTINGS = {"aggregation_strategy": "first", "min_score": 0.80, "min_chars": 3}
SPANS = RESULTS / "partial" / "utility_whole_spans.json"
OUT_DIR = RESULTS / "utility_whole"
N_BOOT = 10_000
METRICS = ("top1", "mrr", f"ndcg@{NDCG_K}")
# the reranker whose nDCG@10 interval lies below zero, as the paper names it
NAMED_NDCG_LOSS = ["Alibaba: GTE ModernBERT"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def draw_sets(queries, resumes, seed: int) -> list[dict]:
    """SETS_PER_QUERY sets per query, members shuffled; every set of the design, in order."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in resumes:
        by_cat[r.category].append(r)
    sets = []
    for q in queries:
        rel_pool = by_cat.get(q.category, [])
        other = [r for r in resumes if r.category != q.category]
        if len(rel_pool) < N_RELEVANT:
            raise SystemExit(f"{q.category}: {len(rel_pool)} résumés, fewer than {N_RELEVANT}")
        for s in range(SETS_PER_QUERY):
            members = rng.sample(rel_pool, N_RELEVANT) + rng.sample(other, N_OTHER)
            rng.shuffle(members)
            sets.append({"set_id": f"{q.qid}#{s}", "qid": q.qid, "rids": [m.rid for m in members],
                         "rel": [int(m.category == q.category) for m in members]})
    return sets


def detect_names(texts: dict[str, str], device: str) -> dict[str, list[str]]:
    """The detector's person names for each document, from the cache where its hash matches."""
    cache = json.loads(SPANS.read_text()) if SPANS.exists() else {}
    if cache.get("detector") != DETECTOR or cache.get("settings") != DETECTOR_SETTINGS:
        cache = {"detector": DETECTOR, "settings": DETECTOR_SETTINGS, "spans": {}}
    have = cache["spans"]
    todo = [rid for rid, t in texts.items() if have.get(rid, {}).get("sha") != sha(t)]
    if todo:
        print(f"[util] detecting names in {len(todo)} of {len(texts)} documents", flush=True)
        detect = chunked(person_detector(device))
        t0 = time.time()
        for i, rid in enumerate(todo, 1):
            have[rid] = {"sha": sha(texts[rid]), "names": detect(texts[rid])}
            if i % 200 == 0 or i == len(todo):
                SPANS.parent.mkdir(parents=True, exist_ok=True)
                tmp = SPANS.with_suffix(".tmp")
                tmp.write_text(json.dumps(cache))
                tmp.replace(SPANS)
                print(f"    {i}/{len(todo)} documents, {(time.time() - t0) / 60:.1f} min", flush=True)
    return {rid: have[rid]["names"] for rid in texts}


def person_detector(device: str):
    """detect(text) -> person names, whole words, each name also split into its parts."""
    from transformers import pipeline
    ner = pipeline("token-classification", model=DETECTOR, device=device,
                   aggregation_strategy=DETECTOR_SETTINGS["aggregation_strategy"])

    def detect(text: str) -> list[str]:
        spans = [e["word"].strip() for e in ner(text) if e.get("entity_group") == "PER"
                 and e.get("score", 0) >= DETECTOR_SETTINGS["min_score"]]
        names = set(spans) | {p for s_ in spans for p in s_.split()}
        return sorted((n for n in names if len(n) >= DETECTOR_SETTINGS["min_chars"]
                       and not n.startswith("##")), key=len, reverse=True)
    return detect


def rank_metrics(scores, rel) -> dict:
    """top-1 (shared among tied leaders), MRR and nDCG@k, ties at their average rank."""
    ranks = average_ranks(scores)
    leaders = [i for i, r in enumerate(ranks) if r == min(ranks)]
    top1 = sum(rel[i] for i in leaders) / len(leaders)
    mrr = 1.0 / min(r for r, g in zip(ranks, rel) if g)
    order = sorted(range(len(scores)), key=lambda i: (ranks[i], i))
    dcg = lambda gains: sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = dcg(sorted(rel, reverse=True)[:NDCG_K])
    return {"top1": top1, "mrr": mrr, f"ndcg@{NDCG_K}": dcg([rel[i] for i in order[:NDCG_K]]) / ideal}


def change_with_interval(per_set: list[dict], rng) -> dict:
    """Mean of each metric per condition, and full minus original with a 95% interval
    resampling the queries (each with all its sets)."""
    by_q = defaultdict(list)
    for s in per_set:
        by_q[s["qid"]].append(s)
    qids = sorted(by_q)
    out = {}
    for m in METRICS:
        level = {c: float(np.mean([s[c][m] for s in per_set])) for c in CONDITIONS}
        d = np.array([np.mean([s["full"][m] - s["original"][m] for s in by_q[q]]) for q in qids])
        draws = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        out[m] = {**level, "change": float(d.mean()),
                  "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                  "n_sets_up": sum(s["full"][m] > s["original"][m] + 1e-12 for s in per_set),
                  "n_sets_down": sum(s["full"][m] < s["original"][m] - 1e-12 for s in per_set)}
    return out


def summarize() -> dict:
    """Every finished model's result in one file and one table; the design must be the same."""
    from companion_full_window import SYNTHETIC_KEY
    from model_labels import model_label
    runs = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("*.json"))
            if not f.name.startswith(("canary_", "summary"))]
    if not runs:
        raise SystemExit(f"no finished run in {OUT_DIR}")
    same = ("design", ("provenance", "prepended_name_found_by_detector"), ("provenance", "n_resumes"))
    get = lambda r, k: r[k] if isinstance(k, str) else r[k[0]][k[1]]
    for k in same:
        if any(get(r, k) != get(runs[0], k) for r in runs):
            raise SystemExit(f"the runs differ in {k}")
    if any(json.dumps(r["sets"]) != json.dumps(runs[0]["sets"]) for r in runs):
        raise SystemExit("the runs scored different sets")
    order = ["voyage-rerank-3", "cohere-v5-pro", "bge-m3", "gte-modernbert", "granite-r2"]
    runs.sort(key=lambda r: order.index(r["reranker"]) if r["reranker"] in order else 99)
    p0 = runs[0]["provenance"]
    rows = {r["reranker"]: {"label": model_label(SYNTHETIC_KEY.get(r["model"], r["model"])), "model": r["model"],
                            "metrics": r["metrics"],
                            "ledger": r["provenance"]["scorer"].get("ledger"),
                            "n_coarse_scores": r["provenance"]["scorer"].get("n_coarse_scores")}
            for r in runs}
    changes = {m: [v["metrics"][m]["change"] for v in rows.values()] for m in METRICS}
    out = {"script_version": SCRIPT_VERSION,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "design": runs[0]["design"], "n_resumes": p0["n_resumes"],
           "detector": p0["detector"], "detector_settings": p0["detector_settings"],
           "prepended_name_found_by_detector": p0["prepended_name_found_by_detector"],
           "pct_prepended_given_found": 100 * p0["prepended_name_found_by_detector"]["given"] / p0["n_resumes"],
           "pct_prepended_surname_found": 100 * p0["prepended_name_found_by_detector"]["surname"] / p0["n_resumes"],
           "n_models": len(rows), "models": rows,
           "summary": {m: {"min_change": min(c), "max_change": max(c),
                           "max_abs_change": max(abs(x) for x in c),
                           "n_ci_includes_zero": sum(v["metrics"][m]["ci"][0] <= 0 <= v["metrics"][m]["ci"][1]
                                                     for v in rows.values()),
                           "n_ci_below_zero": sum(v["metrics"][m]["ci"][1] < 0 for v in rows.values()),
                           "n_ci_above_zero": sum(v["metrics"][m]["ci"][0] > 0 for v in rows.values())}
                       for m, c in changes.items()}}
    # what the paper says: no top-1 change distinguishable from zero, and the one reranker whose
    # nDCG@10 interval lies below zero is the one it names
    nd = f"ndcg@{NDCG_K}"
    below = [v["label"] for v in rows.values() if v["metrics"][nd]["ci"][1] < 0]
    t1 = [v["metrics"]["top1"] for v in rows.values()]
    if any(not t["ci"][0] <= 0 <= t["ci"][1] for t in t1) or below != NAMED_NDCG_LOSS \
            or any(v["metrics"][nd]["ci"][0] > 0 for v in rows.values()):
        raise SystemExit(f"the utility results changed (nDCG loss: {below}): reword Section 3")
    loss = next(v for v in rows.values() if v["label"] in NAMED_NDCG_LOSS)["metrics"][nd]
    days = sorted({r["generated"][:10] for r in runs})
    out["paper"] = {"run_dates": {"first": days[0], "last": days[-1]}, "top1_original_pct_min": 100 * min(t["original"] for t in t1),
                    "top1_original_pct_max": 100 * max(t["original"] for t in t1),
                    "top1_change_pts_min": 100 * min(t["change"] for t in t1),
                    "top1_change_pts_max": 100 * max(t["change"] for t in t1),
                    "top1_ci_pts_min": 100 * min(t["ci"][0] for t in t1),
                    "ndcg_other_change_pts_max_abs": 100 * max(
                        abs(v["metrics"][nd]["change"]) for v in rows.values()
                        if v["label"] not in NAMED_NDCG_LOSS),
                    "ndcg_loss": {"label": NAMED_NDCG_LOSS[0], "change_pts": 100 * loss["change"],
                                  "ci_pts": [100 * x for x in loss["ci"]]}}
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    def pct(x):
        return f"{100 * x:.1f}"

    def chg(v):
        f = lambda x: f"{round(100 * x, 1) + 0.0:+.1f}".replace("-", "$-$")
        return f"{f(v['change'])} [{f(v['ci'][0])}, {f(v['ci'][1])}]"
    lines = ["% Generated by code/utility_whole_resume.py --summarize -- do not edit by hand.",
             r"\begin{tabular}{lrrlrrl}", r"\toprule",
             r" & \multicolumn{3}{c}{Top-1 accuracy (\%)} & \multicolumn{3}{c}{nDCG@10 ($\times 100$)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Reranker & Original & Full rule & Change [95\%] & Original & Full rule & Change [95\%] \\",
             r"\midrule"]
    for v in rows.values():
        t, n = v["metrics"]["top1"], v["metrics"][nd]
        lines.append(f"{v['label']} & {pct(t['original'])} & {pct(t['full'])} & {chg(t)} & "
                     f"{pct(n['original'])} & {pct(n['full'])} & {chg(n)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = RESULTS / "tex" / "utility_whole.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"[util] {len(rows)} models -> {OUT_DIR / 'summary.json'}, {tex}")
    for v in rows.values():
        print(f"    {v['label']:28s} " + "  ".join(
            f"{m} {v['metrics'][m]['original']:.3f}->{v['metrics'][m]['full']:.3f} "
            f"[{v['metrics'][m]['ci'][0]:+.3f}, {v['metrics'][m]['ci'][1]:+.3f}]" for m in METRICS))
    return out


def make_scorer(args, fingerprint: dict, checkpoint: Path):
    if args.reranker in API_RERANKERS:
        return APIScorer(args.reranker, docs_per_request=N_RELEVANT + N_OTHER,
                         checkpoint=checkpoint, checkpoint_meta=fingerprint, resume=args.resume)
    make = RecipeScorer if args.reranker in RECIPE_RERANKERS else Scorer
    return make(RERANKERS[args.reranker], device=args.device, max_length=args.max_length,
                batch_size=args.batch_size, token_budget=args.token_budget,
                attention_budget=args.attention_budget, checkpoint=checkpoint,
                checkpoint_meta=fingerprint, resume=args.resume, revision=args.revision,
                dtype=args.model_dtype)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reranker", choices=sorted(RERANKERS))
    ap.add_argument("--prepare", action="store_true", help="only run the name detector")
    ap.add_argument("--canary", action="store_true",
                    help=f"the first {CANARY_QUERIES} queries' first {CANARY_SETS} sets")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--detector-device", default=None, help="default: --device")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--attention-budget", type=int, default=0)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--model-dtype", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="combine the finished runs")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize()
        return 0
    if not (args.prepare or args.reranker):
        ap.error("--reranker is required unless --prepare or --summarize")
    t0 = time.time()

    resumes = load_corpus()
    queries, qprov = load_queries()
    name_table, nprov = load_name_gender()
    # input paths relative to the repository, so the output names no local directory
    root = str(CODE_DIR.parent) + os.sep
    qprov, nprov = ({k: v.replace(root, "") if isinstance(v, str) else v for k, v in d.items()}
                    for d in (qprov, nprov))
    inject = build_forms(resumes, name_table, build_twin_name_map(name_table), mode="inject",
                         seed=args.seed)
    corpus = {r.rid: r for r in resumes}
    sets = draw_sets(queries, resumes, args.seed)
    if args.canary:
        keep = {q.qid for q in queries[:CANARY_QUERIES]}
        sets = [s for s in sets if s["qid"] in keep and int(s["set_id"].split("#")[1]) < CANARY_SETS]
    used = sorted({rid for s in sets for rid in s["rids"]})
    text = {"original": {rid: corpus[rid].form(corpus[rid].default_form) for rid in used}}
    if any(text["original"][rid] == corpus[rid].text for rid in used):
        raise SystemExit("a résumé did not get a prepended name")
    spans = detect_names(text["original"], args.detector_device or args.device)
    if args.prepare:
        print(f"[util] names detected in {len(used)} documents ({time.time() - t0:.0f}s)")
        return 0
    def own_name(rid):
        m_given, f_given, surname = corpus[rid].injected
        return {"given": m_given if corpus[rid].default_form == "male" else f_given,
                "surname": surname}
    # how often the detector alone found the prepended name
    found = {p: sum(own_name(rid)[p].lower() in {n.lower() for n in spans[rid]} for rid in used)
             for p in ("given", "surname")}
    text["full"] = {rid: deidentify(t, variant="full", placeholder=PLACEHOLDER, lowercase=False,
                                    names=[f"{own_name(rid)['given']} {own_name(rid)['surname']}",
                                           *own_name(rid).values(), *spans[rid]])
                    for rid, t in text["original"].items()}
    left = [rid for rid in used for w in own_name(rid).values()
            if re.search(rf"\b{re.escape(w)}\b", text["full"][rid], flags=re.IGNORECASE)]
    if left:
        raise SystemExit(f"the full rule left a prepended name in {len(set(left))} résumés")

    tag = ("canary_" if args.canary else "") + args.reranker
    fingerprint = {"script": SCRIPT_VERSION, "seed": args.seed, "canary": args.canary,
                   "sets_sha256": sha(json.dumps(sets, sort_keys=True)),
                   "texts_sha256": {c: sha(json.dumps(text[c], sort_keys=True)) for c in CONDITIONS}}
    checkpoint = RESULTS / "partial" / f"utility_whole_{tag}.scores.jsonl"
    scorer = make_scorer(args, fingerprint, checkpoint)
    qtext = {q.qid: q.text for q in queries}
    api = args.reranker in API_RERANKERS
    # hosted: one request per set and condition; local: a résumé's two forms side by side
    key = ((lambda c, s, rid: (f"{c}|{s['set_id']}", rid)) if api
           else (lambda c, s, rid: (s["qid"], rid, c)))
    keys, pairs = [], []
    for c in CONDITIONS:
        for s in sets:
            for rid in s["rids"]:
                keys.append(key(c, s, rid))
                pairs.append((qtext[s["qid"]], text[c][rid]))
    print(f"[util] {args.reranker}: {len(sets)} sets, {len(used)} résumés, "
          f"{len(set(keys))} (query, document) pairs to score", flush=True)
    window = {}
    if not api:
        tok = scorer.model.tokenizer
        n_tok = [len(ids) for ids in tok([list(p) for p in dict(zip(keys, pairs)).values()],
                                         truncation=False)["input_ids"]]
        window = {"max_length": args.max_length, "longest_input_tokens": max(n_tok),
                  "n_inputs_over_window": sum(n > args.max_length for n in n_tok)}
        if window["n_inputs_over_window"]:
            raise SystemExit(f"{window['n_inputs_over_window']} inputs exceed {args.max_length} tokens")
    scorer.score(pairs, keys)

    per_set = []
    for s in sets:
        row = {"set_id": s["set_id"], "qid": s["qid"]}
        for c in CONDITIONS:
            row[c] = rank_metrics([scorer.cache[key(c, s, rid)] for rid in s["rids"]], s["rel"])
        per_set.append(row)
    rng = np.random.default_rng([args.seed, int(sha(args.reranker)[:8], 16)])
    stereo = {q.qid: q.stereotype for q in queries}
    out = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "reranker": args.reranker, "model": RERANKERS[args.reranker], "canary": args.canary,
        "design": {"n_queries": len({s["qid"] for s in sets}), "sets_per_query": SETS_PER_QUERY,
                   "n_sets": len(sets), "n_relevant": N_RELEVANT, "n_other": N_OTHER,
                   "conditions": {"original": "as written, with the prepended name",
                                  "full": f"the applicant's own (prepended) name and the person "
                                          f"names {DETECTOR} finds -> '{PLACEHOLDER}', gendered "
                                          "pronouns -> 'they', replacements in lowercase, the rest "
                                          "as written (deid_transforms.deidentify variant 'full', "
                                          "lowercase=False)"},
                   "relevance": "a résumé is relevant iff its corpus category is the query's",
                   "metrics": "per set; ties at their average rank, top-1 credit shared among "
                              "tied leaders", "interval": f"queries resampled ({N_BOOT:,} draws)",
                   "seed": args.seed},
        "provenance": {"queries": qprov, "names": nprov, "injection": inject,
                       "n_resumes": len(used), "detector": DETECTOR,
                       "detector_settings": DETECTOR_SETTINGS,
                       "n_resumes_with_detected_name": sum(bool(spans[r]) for r in used),
                       "prepended_name_found_by_detector": found,
                       "scorer": {"revision": args.revision, "dtype": args.model_dtype,
                                  "device": None if api else args.device, **window,
                                  "settings": getattr(scorer, "settings", None),
                                  "ledger": getattr(scorer, "ledger", None),
                                  "n_coarse_scores": len(getattr(scorer, "coarse_keys", ()))},
                       "n_pairs_scored": scorer.n_scored,
                       "n_scores_from_checkpoint": scorer.n_resumed,
                       "score_seconds": round(scorer.score_seconds, 1)},
        "metrics": change_with_interval(per_set, rng),
        "by_stereotype": {g: change_with_interval([s for s in per_set if stereo[s["qid"]] == g], rng)
                          for g in sorted({stereo[s["qid"]] for s in per_set})},
        "sets": sets,
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    with gzip.open(OUT_DIR / f"{tag}_scores.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"reranker": args.reranker, "scores": [
            [c, s["set_id"], rid, scorer.cache[key(c, s, rid)]]
            for c in CONDITIONS for s in sets for rid in s["rids"]]}, fh)
    print(f"[util] {args.reranker}: the detector alone found the prepended given name in "
          f"{found['given']} and the surname in {found['surname']} of {len(used)} résumés")
    for m, v in out["metrics"].items():
        print(f"    {m:8s} original {v['original']:.4f}  full {v['full']:.4f}  change "
              f"{v['change']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  sets up/down "
              f"{v['n_sets_up']}/{v['n_sets_down']}")
    print(f"[util] wrote {OUT_DIR / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
