"""
The companion experiment on whole résumés: the synthetic-audit contrast against the
whole-résumé one for every reranker that reads one, and the length x pronoun follow-up.

Part 1 takes each of the 24 rerankers of code/companion_full_window.py's overview and its
whole-résumé name-only run with only the résumé owner's own name and pronouns changed
(results/length_pronoun/<model>_wfull_name.json; data/owner_marker_labels.json): the
male-typed minus female-typed tie-aware male-favored share (M-F) on whole résumés, with its
pool-bootstrap interval and its continuous check against the BLS women's share, beside the
synthetic-audit M-F from the overview. jina-reranker-v3 is order-sensitive, so its pairs'
gaps are averaged over a male-first and a female-first run, as in the overview. The
overview's own whole-résumé run swapped every carried name and pronoun, most of them other
people's; each reranker's change from it is reported. Each synthetic M-F gets its
occupation-clustered interval:
the audit's from results/interval_sensitivity.json, and for the models released after the
audit the same bootstrap over their October 2026 scores of the audit's pairs. It counts the
rerankers whose whole-résumé interval lies above zero, below zero, or covers it, overall and
among those whose synthetic interval lies above zero ("the synthetic pattern"), and gives
the rank and linear correlation of the two contrasts across rerankers, with a bootstrap
interval that resamples rerankers.

Part 2 summarises the length x pronoun runs (code/run_length_pronoun.sh). Six rerankers
scored the same 3,840 comparisons, owner-only as in Part 1, in twelve conditions: the résumé cut so that it holds 35, 75,
150, 300 or 600 words after the prepended name, or whole; each with the name only ("name")
or with the name followed by a fourteen-word sentence about the owner carrying three of the
owner's pronouns ("line"; OWNER_LINE in realistic_pool_experiment.py, whose words count
towards the cut). For each condition: M-F with its pool-bootstrap interval, and the paired
line-minus-name difference (the whole name-only condition is Part 1's). Derived: the
35-word line condition, the closest in form to a synthetic profile, against the synthetic
contrast.

Word counts: the scored résumés' bodies (needs the corpus, data/PROVENANCE.md) and the
synthetic audit's documents; without the corpus the previous output's counts are kept.

Reads:  results/companion_full_window.json
        results/length_pronoun/<model>_w<35|75|150|300|600|full>_<name|line>.json (+ rows),
        jina-v3_wfull_name_female_first.json
        data/owner_marker_labels.json, data/owner_marker_handcheck.json (a person's check of 40
        labels, tallied by owner_label_handcheck.py)
        data/resumes/resumes.csv (optional, word counts)
Output: results/companion_whole_resume.json, results/tex/companion_whole_resume.tex
Run:    venv/bin/python code/companion_whole_resume.py
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import stable_seed  # noqa: E402
from companion_full_window import (POST_AUDIT, SYNTHETIC_KEY, bootstrap_m_minus_f,  # noqa: E402
                                   continuous_check, model_rng, winner)
from interval_sensitivity import cell_tables, contrast_draws  # noqa: E402
from labels import load_labels  # noqa: E402
from companion_statistics import shares  # noqa: E402
from model_labels import model_label  # noqa: E402
from owner_label_handcheck import tally as handcheck_tally  # noqa: E402
from realistic_pool_experiment import _PRON_RE, OWNER_LINE, OWNER_LINE_WORDS, load_queries  # noqa: E402

SCRIPT_VERSION = "companion_whole_resume.py v1.2"
RESULTS = ROOT / "results"
OVERVIEW = RESULTS / "companion_full_window.json"
LP_DIR = RESULTS / "length_pronoun"
LABELS = ROOT / "data" / "owner_marker_labels.json"
SENSITIVITY = RESULTS / "interval_sensitivity.json"
# sha256 of the labels files the owner-only runs used (the length x pronoun runs, then the
# other rerankers' runs, after a correction of the file's description), and of its labels
# alone: the description changed again after the runs, the labels never did
RUN_LABEL_FILES = {"2f1e31c3f475224264c5f7ef30afa18a5a97f610fed10ecd8791a6fabc9072e2",
                   "4b03d5d66ea067032691609a57a7dea4fe77a39c669520cbdcca8a91a11dd303"}
RUN_LABELS = "95611e9e7b9cb40c5aa20f03e21066188ed9e1262a056105e84f943e2241b53d"
POOL_SHA = "91687571c4c9dd4f78c5d442c751070ea0efc18b56e5f9713479a4532f80c400"
N_COMPARISONS = 3840
# listwise and order-sensitive: both orders averaged (companion_full_window.COUNTERBALANCED)
ORDER_SENSITIVE = {"jina-v3"}
# reads at most 1,024 tokens, so most résumés are cut
WINDOW_CUT = {"jina"}
# The exceptions the paper names: rerankers with the synthetic pattern whose whole-résumé
# M-F is not smaller, and rerankers with a direction whose continuous check has the other sign
NAMED_IN_PAPER = {"not_smaller": ["Voyage: rerank-2.5-lite"],
                  "rho_sign_disagrees": ["Cohere: v4 fast"]}
OUT = RESULTS / "companion_whole_resume.json"
TEX = RESULTS / "tex" / "companion_whole_resume.tex"
LP_MODELS = ["voyage-rerank-3", "cohere-v5-pro", "qwen3-8b", "voyage-rerank-2", "bge-m3",
             "mxbai-large-v2"]
LENGTHS = [35, 75, 150, 300, 600, 0]          # words after the prepended name; 0 = whole
N_BOOT_MODELS = 10_000
SEED = 20261026


def length_key(n):
    return "full" if n == 0 else str(n)


def label(entry):
    return model_label(SYNTHETIC_KEY.get(entry["model"], entry["model"]))


# ---------------------------------------------------------------------------
# Part 1: the 24-reranker whole-résumé run
# ---------------------------------------------------------------------------

def synthetic_intervals():
    """{model id: occupation-clustered 95% interval of the synthetic M-F}, as in the audit."""
    sens = json.loads(SENSITIVITY.read_text())
    out = {m: v["schemes"]["occupation"]["ci"] for m, v in sens["models"].items()}
    labels, _ = load_labels(strict=True)
    occs = sorted(o for o, c in labels.items() if c in ("male", "female"))
    cat = [labels[o] for o in occs]
    for model, path in POST_AUDIT.items():
        raw = json.loads(path.read_text())["results"]
        W, N = cell_tables(raw, occs, sorted({r["name_pair"] for r in raw}))
        rng = np.random.default_rng(stable_seed(model, "occupation", "contrast"))
        draws = contrast_draws(W, N, cat, dict(enumerate(occs)), sens["bootstrap_B"], rng,
                               resample_names=False)
        out[model] = [float(x) for x in np.percentile(draws, [2.5, 97.5])]
    return out


def direction(ci):
    return "male" if ci[0] > 0 else "female" if ci[1] < 0 else "unclear"


def owner_only_overview(overview):
    """Each reranker's whole-résumé owner-only run in the overview's shape, with the
    overview's own (all-marker) run kept beside it; the synthetic side is the overview's."""
    pct_women = {q.category: q.pct_women for q in load_queries(None, None, None, 0)[0]}
    out, sha = {}, set()
    for short, v in overview.items():
        rng = model_rng(short)
        d, rows = load_lp(short, 0, "name")
        runs = [d]
        entry = {}
        if short in ORDER_SENSITIVE:
            dff, ff = load_lp(short, 0, "name", female_first=True)
            runs.append(dff)
            if [(r["pool_id"], r["focal_index"], r["focal_rid"]) for r in rows] != \
                    [(r["pool_id"], r["focal_index"], r["focal_rid"]) for r in ff]:
                raise SystemExit(f"{short}: the two orders do not hold the same comparisons")
            gap = lambda x: shares(x)["male"] - shares(x)["female"]
            entry["counterbalanced"] = {
                "male_first": gap(rows), "female_first": gap(ff),
                "pct_same_winner_both_orders": 100 * sum(winner(a) == winner(b)
                                                         for a, b in zip(rows, ff)) / len(rows)}
            rows = [{**a, "delta": (a["delta"] + b["delta"]) / 2} for a, b in zip(rows, ff)]
        for run in runs:
            if run["provenance"]["sampling"]["pool_membership_sha256"] != POOL_SHA:
                raise SystemExit(f"{short}: the run used other pools")
            sha.add(run["provenance"]["owner_only_markers"]["labels_sha256"])
        if len(rows) != N_COMPARISONS:
            raise SystemExit(f"{short}: {len(rows)} comparisons, not {N_COMPARISONS}")
        cut = sum(not r["marker_in_token_window"] for r in rows)
        if cut and short not in WINDOW_CUT:
            raise SystemExit(f"{short}: {cut} comparisons not read whole")
        sh = shares(rows)
        out[short] = {"model": v["model"], "post_audit": bool(v.get("post_audit")),
                      "reads_every_resume_whole": short not in WINDOW_CUT,
                      "synthetic": v["synthetic"],
                      "whole_document": sh["male"] - sh["female"],
                      "ci": bootstrap_m_minus_f(rows, rows, rng)["ci_512"],
                      "continuous": continuous_check(rows, pct_women, rng),
                      "all_markers_run": {"whole_document": v["whole_document"], "ci": v["ci"]},
                      **entry}
    return out, sha


def whole_resume(overview):
    syn_ci = synthetic_intervals()
    rows = []
    for short, v in overview.items():
        lo, hi = v["ci"]
        slo, shi = syn_ci[SYNTHETIC_KEY.get(v["model"], v["model"])]
        old = v["all_markers_run"]
        rows.append({"short": short, "model": v["model"], "label": label(v),
                     "post_audit": v["post_audit"],
                     "reads_every_resume_whole": v["reads_every_resume_whole"],
                     "synthetic": v["synthetic"], "synthetic_ci": [slo, shi],
                     "synthetic_pattern": slo > 0,
                     "whole_resume": v["whole_document"],
                     "ci": [lo, hi],
                     "direction": direction(v["ci"]),
                     "rho_women_share": v["continuous"]["spearman_rho"],
                     "rho_p": v["continuous"]["permutation_p_two_sided"],
                     "all_markers_run": {**old, "direction": direction(old["ci"]),
                                         "owner_only_minus_all_markers":
                                             v["whole_document"] - old["whole_document"]}})
    rows.sort(key=lambda r: -r["synthetic"])
    syn = np.array([r["synthetic"] for r in rows])
    whole = np.array([r["whole_resume"] for r in rows])
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT_MODELS):
        i = rng.integers(0, len(rows), len(rows))
        if np.ptp(syn[i]) and np.ptp(whole[i]):
            boot.append(spearmanr(syn[i], whole[i])[0])
    audit = [k for k, r in enumerate(rows) if not r["post_audit"]]
    by_dir = {d: [r for r in rows if r["direction"] == d] for d in ("male", "female", "unclear")}
    v2 = next(r for r in rows if r["short"] == "voyage-rerank-2")
    pat = [r for r in rows if r["synthetic_pattern"]]
    directed = [r for r in rows if r["direction"] != "unclear"]
    change = [r["all_markers_run"]["owner_only_minus_all_markers"] for r in rows]
    not_smaller = [r["label"] for r in pat if r["whole_resume"] >= r["synthetic"]]
    disagrees = [r["label"] for r in directed
                 if (r["rho_women_share"] < 0) != (r["direction"] == "male")]
    # the paper names these exceptions and says every reversal's check differs from zero:
    # fail loudly if that stops holding
    if not_smaller != NAMED_IN_PAPER["not_smaller"] \
            or disagrees != NAMED_IN_PAPER["rho_sign_disagrees"]:
        raise SystemExit(f"exceptions changed: not smaller {not_smaller}, sign {disagrees}: "
                         "reword Section 3 and Appendix E")
    if sum(r["rho_women_share"] > 0 and r["rho_p"] < 0.05 for r in by_dir["female"]) \
            != len(by_dir["female"]):
        raise SystemExit("a reversal's continuous check is not significant: reword Appendix E")
    return {
        "rows": rows,
        "n_models": len(rows),
        "n_audit": len(audit),
        "n_post_audit": len(rows) - len(audit),
        "n_audit_reading_whole": sum(r["reads_every_resume_whole"] and not r["post_audit"]
                                     for r in rows),
        "n_male_direction": len(by_dir["male"]),
        "n_female_direction": len(by_dir["female"]),
        "n_unclear": len(by_dir["unclear"]),
        "female_direction": [r["label"] for r in by_dir["female"]],
        "n_smaller_than_synthetic": int(np.sum(whole < syn)),
        "male_direction_whole_min": min(r["whole_resume"] for r in by_dir["male"]),
        "male_direction_whole_max": max(r["whole_resume"] for r in by_dir["male"]),
        "n_male_direction_rho_negative": sum(r["rho_women_share"] < 0 for r in by_dir["male"]),
        "n_male_direction_rho_negative_p05": sum(r["rho_women_share"] < 0 and r["rho_p"] < 0.05
                                                 for r in by_dir["male"]),
        "n_female_direction_rho_positive_p05": sum(r["rho_women_share"] > 0 and r["rho_p"] < 0.05
                                                   for r in by_dir["female"]),
        "voyage_rerank_2": {"synthetic": v2["synthetic"], "whole_resume": v2["whole_resume"]},
        "synthetic_pattern": {
            "n": len(pat),
            "n_male_direction": sum(r["direction"] == "male" for r in pat),
            "n_female_direction": sum(r["direction"] == "female" for r in pat),
            "n_unclear": sum(r["direction"] == "unclear" for r in pat),
            "n_smaller_than_synthetic": sum(r["whole_resume"] < r["synthetic"] for r in pat),
            "n_male_direction_smaller": sum(r["direction"] == "male"
                                            and r["whole_resume"] < r["synthetic"] for r in pat),
            "not_smaller": {r["label"]: {"synthetic": r["synthetic"], "whole_resume": r["whole_resume"]}
                            for r in pat if r["label"] in not_smaller},
            "male_direction_whole_min": min(r["whole_resume"] for r in pat
                                            if r["direction"] == "male"),
            "without_pattern": {r["label"]: {"synthetic": r["synthetic"],
                                             "synthetic_ci": r["synthetic_ci"],
                                             "whole_resume": r["whole_resume"], "ci": r["ci"]}
                                for r in rows if not r["synthetic_pattern"]}},
        "all_markers_run": {
            "change_min_abs": min(abs(c) for c in change),
            "change_max_abs": max(abs(c) for c in change),
            "change_median_abs": float(np.median(np.abs(change))),
            "n_sign_changes": sum((r["whole_resume"] > 0) != (r["all_markers_run"]["whole_document"] > 0)
                                  for r in rows),
            "direction_changes": {r["label"]: [r["all_markers_run"]["direction"], r["direction"]]
                                  for r in rows if r["direction"] != r["all_markers_run"]["direction"]},
            "note": "the overview's whole-résumé run swapped every carried name and pronoun, "
                    "most of them other people's; change = owner-only minus that run"},
        "n_with_direction": len(directed),
        "n_rho_sign_agrees": sum((r["rho_women_share"] < 0) == (r["direction"] == "male")
                                 for r in directed),
        "rho_sign_disagrees": {r["label"]: {"whole_resume": r["whole_resume"], "ci": r["ci"],
                                            "rho": r["rho_women_share"], "p": r["rho_p"]}
                               for r in directed if r["label"] in disagrees},
        "rho_p_note": "permutation p values are unadjusted for the number of rerankers",
        "correlation": {
            "spearman_rho": float(spearmanr(syn, whole)[0]),
            "spearman_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "pearson_r": float(pearsonr(syn, whole)[0]),
            "audit_only_spearman_rho": float(spearmanr(syn[audit], whole[audit])[0]),
            "n_bootstrap": N_BOOT_MODELS,
            "bootstrap": "rerankers resampled with replacement"},
    }


def tex_table(wr):
    """Rows sorted by the synthetic contrast; marks: post-audit, 1,024-token window, both orders."""
    def num(x, d=1):
        return f"{x:.{d}f}".replace("-", "$-$")
    out = ["% Generated by code/companion_whole_resume.py -- do not edit by hand.",
           r"\begin{tabular}{lrrcr}", r"\toprule",
           r" & Synthetic & Whole-résumé & & $\rho$ with \\",
           r"Reranker & M$-$F & M$-$F & 95\% interval & women's share \\", r"\midrule"]
    for r in wr["rows"]:
        mark = ("$^{a}$" if r["post_audit"] else "") + \
               ("$^{b}$" if not r["reads_every_resume_whole"] else "") + \
               ("$^{c}$" if r["short"] == "jina-v3" else "")
        star = "*" if r["rho_p"] < 0.05 else ""
        out.append(f"{r['label']}{mark} & {num(r['synthetic'])} & {num(r['whole_resume'])} & "
                   f"[{num(r['ci'][0])}, {num(r['ci'][1])}] & {num(r['rho_women_share'], 2)}{star} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Part 2: length x pronoun
# ---------------------------------------------------------------------------

def load_lp(m, n, arm, female_first=False):
    p = LP_DIR / f"{m}_w{length_key(n)}_{arm}{'_female_first' if female_first else ''}.json"
    d = json.loads(p.read_text())
    if d.get("canary_limit_pools") or d["config"]["cut_words"] != n \
            or bool(d["config"]["owner_line"]) != (arm == "line") \
            or bool(d["config"].get("female_first")) != female_first \
            or d["config"]["marker_labels"] != "data/owner_marker_labels.json":
        raise SystemExit(f"{p.name}: not the condition its name says")
    with gzip.open(ROOT / d["models"][m]["rows_file"], "rt", encoding="utf-8") as fh:
        rows = json.load(fh)["models"][m]["rows"]
    return d, sorted(rows, key=lambda r: (r["pool_id"], r["focal_index"]))


def length_pronoun(overview):
    out, sha = {}, set()
    for m in LP_MODELS:
        rng = model_rng(m)
        by = {}
        for n in LENGTHS:
            (da, a), (db, b) = load_lp(m, n, "name"), load_lp(m, n, "line")
            sha |= {da["provenance"]["owner_only_markers"]["labels_sha256"],
                    db["provenance"]["owner_only_markers"]["labels_sha256"]}
            e = {}
            for arm, rows in (("name", a), ("line", b)):
                sh = shares(rows)
                whole_name = n == 0 and arm == "name"   # Part 1's run: its interval
                e[arm] = {"gap": sh["male"] - sh["female"],
                          "ci": (overview[m]["ci"] if whole_name
                                 else bootstrap_m_minus_f(rows, rows, rng)["ci_512"]),
                          "n_resumes_cut": (da if arm == "name" else db)["provenance"]
                          ["owner_only_markers"]["shape"]["n_resumes_cut"]}
            if [(r["pool_id"], r["focal_index"]) for r in a] != \
                    [(r["pool_id"], r["focal_index"]) for r in b]:
                raise SystemExit(f"{m} at {n}: the two arms scored different comparisons")
            e["line_minus_name"] = {"diff": e["line"]["gap"] - e["name"]["gap"],
                                    "ci": bootstrap_m_minus_f(a, b, rng)["ci_change"]}
            by[length_key(n)] = e
        v = overview[m]
        if abs(by["full"]["name"]["gap"] - v["whole_document"]) > 1e-9:
            raise SystemExit(f"{m}: the whole name-only condition differs from Part 1's")
        out[m] = {"label": label(v), "synthetic": v["synthetic"], "by_length": by,
                  "short_line_minus_synthetic": by["35"]["line"]["gap"] - v["synthetic"]}

    def col(f):
        return {m: f(v) for m, v in out.items()}
    short_gap = col(lambda v: abs(v["short_line_minus_synthetic"]))
    worst = max(short_gap, key=short_gap.get)
    rest = [g for m, g in short_gap.items() if m != worst]
    line35 = col(lambda v: v["by_length"]["35"]["line_minus_name"])
    linefull = col(lambda v: v["by_length"]["full"]["line_minus_name"])
    return {
        "models": out,
        "labels_sha256": sorted(sha),
        "lengths": [length_key(n) for n in LENGTHS],
        "derived": {
            "short_line_vs_synthetic_outlier": out[worst]["label"],
            "short_line_vs_synthetic_outlier_abs": short_gap[worst],
            "short_line_vs_synthetic_rest_min": min(rest),
            "short_line_vs_synthetic_rest_max": max(rest),
            "line_effect_35_min": min(x["diff"] for x in line35.values()),
            "line_effect_35_max": max(x["diff"] for x in line35.values()),
            "n_line_effect_35_above_zero": sum(x["ci"][0] > 0 for x in line35.values()),
            "n_line_effect_full_above_zero": sum(x["ci"][0] > 0 for x in linefull.values()),
            "line_effect_full_above_zero": [out[m]["label"] for m, x in linefull.items()
                                            if x["ci"][0] > 0],
            "line_effect_full_rest_max_abs": max(abs(x["diff"]) for x in linefull.values()
                                                 if x["ci"][0] <= 0),
        },
    }


def owner_labels(shas):
    """Tallies of the owner-marker labels every owner-only run used."""
    import hashlib
    raw = LABELS.read_bytes()
    data = json.loads(raw)
    content = hashlib.sha256(json.dumps(data["labels"], sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()
    if content != RUN_LABELS or not set(shas) <= RUN_LABEL_FILES | {hashlib.sha256(raw).hexdigest()}:
        raise SystemExit(f"{LABELS.name} does not hold the labels the runs used")
    labels = data["labels"]
    names = [h["ref"] for v in labels.values() for h in v["names"]]
    prons = [p["ref"] for v in labels.values() for p in v["pronouns"]]
    # the two labelling passes' agreement, as the file's description records it
    import re
    m = re.search(r"agreed on (\d+) of (\d+) names and (\d+) of (\d+) pronouns", data["description"])
    if not m or (int(m[2]), int(m[4])) != (len(names), len(prons)):
        raise SystemExit(f"{LABELS.name}: the description's agreement counts do not match the labels")
    hc = handcheck_tally()
    # the paper says the checker agreed with every applicant label they could decide, and that
    # every disagreement was a word the checker gave to the applicant, settled for the model
    # on a second look
    if (hc["model_applicant"]["agree"] != hc["model_applicant"]["decided"]
            or not hc["disagree"] == hc["disagree_checker_applicant"] == hc["disagree_reread_model"]):
        raise SystemExit("the hand check no longer matches the paper's description of it")
    return {"n_resumes": len(labels),
            "passes_agree_names": int(m[1]),
            "passes_agree_pronouns": int(m[3]),
            "n_name_occurrences_other": len(names) - names.count("applicant"),
            "n_pronouns_not_owner": len(prons) - prons.count("applicant"),
            "n_name_occurrences": len(names),
            "n_name_occurrences_owner": names.count("applicant"),
            "n_resumes_with_owner_name": sum(bool(v["owner_spans"]) for v in labels.values()),
            "n_pronouns": len(prons),
            "n_pronouns_owner": prons.count("applicant"),
            "n_pronouns_other_person": prons.count("other"),
            "n_pronouns_no_person": prons.count("none"),
            "hand_check": hc}


def run_dates():
    """First and last day a whole-résumé or length x pronoun run was written."""
    days = sorted(json.loads(f.read_text())["generated"][:10] for f in
                  [*RESULTS.glob("realistic_pool_full_window_*.json"),
                   *(f for f in LP_DIR.glob("*.json")
                     if not f.name.startswith("canary_") and f.name != "summary.json")])
    return {"first": days[0], "last": days[-1], "n_runs": len(days)}


def word_counts(lp_models):
    """Median and quartiles of the scored résumés' words and of the synthetic documents'."""
    from realistic_pool_experiment import CORPUS_CANDIDATES, _first_existing, load_corpus
    from synthetic_dataset import QUERY_TYPES, generate_dataset
    labels, _ = load_labels(strict=True)
    docs = [t.doc_male for t in generate_dataset(["names_and_pronouns"], list(QUERY_TYPES))
            if t.occupation in labels]
    syn = [len(d.split()) for d in docs]
    pron = [len(_PRON_RE.findall(d)) for d in docs]
    q = lambda w: {"n": len(w), "min": int(min(w)), "q1": float(np.percentile(w, 25)),
                   "median": float(np.median(w)), "q3": float(np.percentile(w, 75)),
                   "max": int(max(w))}
    out = {"synthetic_documents": q(syn),
           "synthetic_pronouns": {"min": min(pron), "max": max(pron)}}
    if _first_existing(CORPUS_CANDIDATES) is None:
        old = json.loads(OUT.read_text())["word_counts"] if OUT.exists() else {}
        out["scored_resumes"] = old.get("scored_resumes")
        out["scored_resumes_source"] = "previous output (corpus not present)"
        return out
    with gzip.open(ROOT / json.loads((LP_DIR / f"{lp_models[0]}_wfull_name.json").read_text())
                   ["models"][lp_models[0]]["rows_file"], "rt", encoding="utf-8") as fh:
        rids = list(json.load(fh)["resumes"])
    text = {r.rid: r.text for r in load_corpus()}
    out["scored_resumes"] = q([len(text[i].split()) for i in rids])
    out["scored_resumes_source"] = "corpus"
    return out


def main():
    fw = json.loads(OVERVIEW.read_text())
    overview, shas = owner_only_overview(fw["overview"])
    wr = whole_resume(overview)
    lp = length_pronoun(overview)
    out = {"script_version": SCRIPT_VERSION,
           "generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "source": {"overview": str(OVERVIEW.relative_to(ROOT)),
                      "overview_script": fw["script_version"],
                      "length_pronoun": str(LP_DIR.relative_to(ROOT))},
           "notes": {"contrast": "M-F = tie-aware male-favored share (%) in the 7 male-typed "
                                 "minus the 3 female-typed job categories, in points",
                     "direction": "male / female: the whole-résumé interval lies above / below "
                                  "zero; unclear: it covers zero",
                     "owner_only": "every whole-résumé and length x pronoun run changes only "
                                   "the résumé owner's own name and pronouns; all_markers_run is "
                                   "the overview's run, which swapped every carried marker",
                     "counterbalanced": "jina-v3: each pair's gap averaged over a male-first "
                                        "and a female-first run (counterbalanced)",
                     "synthetic_ci": "occupation-clustered 95% interval of the synthetic M-F; for "
                                     "jina-v3 the audit's same-order interval, while its synthetic "
                                     "M-F averages both orders"},
           "design": {"cut_words": LENGTHS[:-1],
                      "cut_words_text": ", ".join(map(str, LENGTHS[:-2])) + f", or {LENGTHS[-2]}",
                      "owner_line": OWNER_LINE,
                      "owner_line_words": OWNER_LINE_WORDS,
                      "owner_line_pronouns": len(_PRON_RE.findall(OWNER_LINE["male"]))},
           "owner_labels": owner_labels(shas | set(lp["labels_sha256"])),
           "run_dates": run_dates(),
           "whole_resume": wr, "length_pronoun": lp, "word_counts": word_counts(LP_MODELS)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    TEX.write_text(tex_table(wr))

    c = wr["correlation"]
    print(f"{wr['n_models']} rerankers: {wr['n_male_direction']} male, {wr['n_female_direction']} "
          f"female, {wr['n_unclear']} unclear; rho {c['spearman_rho']:.2f} "
          f"[{c['spearman_ci'][0]:.2f}, {c['spearman_ci'][1]:.2f}], r {c['pearson_r']:.2f}")
    for r in wr["rows"]:
        a = r["all_markers_run"]
        print(f"  {r['short']:24s} {r['synthetic']:+6.1f} -> {r['whole_resume']:+6.1f} "
              f"[{r['ci'][0]:+5.1f}, {r['ci'][1]:+5.1f}] {r['direction']:8s} rho {r['rho_women_share']:+.2f}"
              f" (p {r['rho_p']:.3f}) | all-marker run {a['whole_document']:+6.1f} {a['direction']}")
    print(json.dumps(wr["all_markers_run"], indent=1, ensure_ascii=False))
    print(f"{'model':16s} {'synth':>6s} | " + " | ".join(f"{k:>13s}" for k in lp["lengths"]))
    for m, v in lp["models"].items():
        cells = [f"{v['by_length'][k]['name']['gap']:+5.1f} / {v['by_length'][k]['line']['gap']:+5.1f}"
                 for k in lp["lengths"]]
        print(f"{m:16s} {v['synthetic']:6.1f} | " + " | ".join(cells))
    print("(cells: name only / name + pronoun line)")
    print(json.dumps(lp["derived"], indent=1))
    print("wrote", OUT.relative_to(ROOT), "and", TEX.relative_to(ROOT))


if __name__ == "__main__":
    main()
