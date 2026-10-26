"""
The companion experiment on whole résumés: the synthetic-audit contrast against the
whole-résumé one for every reranker that reads one, and the length x pronoun follow-up.

Part 1 reads the overview of code/companion_full_window.py: for each of its 24 rerankers,
the male-typed minus female-typed tie-aware male-favored share (M-F) in the synthetic audit
and on whole résumés, with the latter's pool-bootstrap interval and its continuous check
against the BLS women's share. Each synthetic M-F gets its occupation-clustered interval:
the audit's from results/interval_sensitivity.json, and for the models released after the
audit the same bootstrap over their October 2026 scores of the audit's pairs. It counts the
rerankers whose whole-résumé interval lies above zero, below zero, or covers it, overall and
among those whose synthetic interval lies above zero ("the synthetic pattern"), and gives
the rank and linear correlation of the two contrasts across rerankers, with a bootstrap
interval that resamples rerankers.

Part 2 summarises the length x pronoun runs (code/run_length_pronoun.sh). Six rerankers
scored the same 3,840 comparisons with only the résumé owner's own name and pronouns changed
(data/owner_marker_labels.json) in twelve conditions: the résumé cut so that it holds 35, 75,
150, 300 or 600 words after the prepended name, or whole; each with the name only ("name")
or with the name followed by a fourteen-word sentence about the owner carrying three of the
owner's pronouns ("line"; OWNER_LINE in realistic_pool_experiment.py, whose words count
towards the cut). For each condition: M-F with its pool-bootstrap interval, and the paired
line-minus-name difference. Derived: the whole-résumé name-only contrast against the
24-reranker run (which swapped every carried marker, most of them other people's), and the
35-word line condition, the closest in form to a synthetic profile, against the synthetic
contrast.

Word counts: the scored résumés' bodies (needs the corpus, data/PROVENANCE.md) and the
synthetic audit's documents; without the corpus the previous output's counts are kept.

Reads:  results/companion_full_window.json
        results/length_pronoun/<model>_w<35|75|150|300|600|full>_<name|line>.json (+ rows)
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
from companion_full_window import POST_AUDIT, SYNTHETIC_KEY, bootstrap_m_minus_f, model_rng  # noqa: E402
from interval_sensitivity import cell_tables, contrast_draws  # noqa: E402
from labels import load_labels  # noqa: E402
from companion_statistics import shares  # noqa: E402
from model_labels import model_label  # noqa: E402
from realistic_pool_experiment import _PRON_RE, OWNER_LINE, OWNER_LINE_WORDS  # noqa: E402

SCRIPT_VERSION = "companion_whole_resume.py v1.0"
RESULTS = ROOT / "results"
OVERVIEW = RESULTS / "companion_full_window.json"
LP_DIR = RESULTS / "length_pronoun"
LABELS = ROOT / "data" / "owner_marker_labels.json"
SENSITIVITY = RESULTS / "interval_sensitivity.json"
# The labels file the runs used (sha256 of the file) and of its labels alone: the file's
# description was corrected afterwards, the labels were not
RUN_LABELS = ("2f1e31c3f475224264c5f7ef30afa18a5a97f610fed10ecd8791a6fabc9072e2",
              "95611e9e7b9cb40c5aa20f03e21066188ed9e1262a056105e84f943e2241b53d")
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


def whole_resume(overview):
    syn_ci = synthetic_intervals()
    rows = []
    for short, v in overview.items():
        lo, hi = v["ci"]
        slo, shi = syn_ci[SYNTHETIC_KEY.get(v["model"], v["model"])]
        rows.append({"short": short, "model": v["model"], "label": label(v),
                     "post_audit": bool(v.get("post_audit")),
                     "reads_every_resume_whole": v["reads_every_resume_whole"],
                     "synthetic": v["synthetic"], "synthetic_ci": [slo, shi],
                     "synthetic_pattern": slo > 0,
                     "whole_resume": v["whole_document"],
                     "ci": [lo, hi],
                     "direction": "male" if lo > 0 else "female" if hi < 0 else "unclear",
                     "rho_women_share": v["continuous"]["spearman_rho"],
                     "rho_p": v["continuous"]["permutation_p_two_sided"]})
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
    # the paper says "all smaller" and "agrees in sign for all": fail loudly if either stops holding
    if any(r["whole_resume"] >= r["synthetic"] for r in pat):
        raise SystemExit("a reranker with the synthetic pattern is not smaller on whole résumés")
    if any((r["rho_women_share"] < 0) != (r["direction"] == "male") for r in directed):
        raise SystemExit("a continuous check disagrees in sign with its reranker's direction")
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
            "male_direction_whole_min": min(r["whole_resume"] for r in pat
                                            if r["direction"] == "male"),
            "without_pattern": {r["label"]: {"synthetic": r["synthetic"],
                                             "synthetic_ci": r["synthetic_ci"],
                                             "whole_resume": r["whole_resume"], "ci": r["ci"]}
                                for r in rows if not r["synthetic_pattern"]}},
        "n_with_direction": len(directed),
        "n_rho_sign_agrees": sum((r["rho_women_share"] < 0) == (r["direction"] == "male")
                                 for r in directed),
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

def load_lp(m, n, arm):
    p = LP_DIR / f"{m}_w{length_key(n)}_{arm}.json"
    d = json.loads(p.read_text())
    if d.get("canary_limit_pools") or d["config"]["cut_words"] != n \
            or bool(d["config"]["owner_line"]) != (arm == "line") \
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
                e[arm] = {"gap": sh["male"] - sh["female"],
                          "ci": bootstrap_m_minus_f(rows, rows, rng)["ci_512"],
                          "n_resumes_cut": (da if arm == "name" else db)["provenance"]
                          ["owner_only_markers"]["shape"]["n_resumes_cut"]}
            if [(r["pool_id"], r["focal_index"]) for r in a] != \
                    [(r["pool_id"], r["focal_index"]) for r in b]:
                raise SystemExit(f"{m} at {n}: the two arms scored different comparisons")
            e["line_minus_name"] = {"diff": e["line"]["gap"] - e["name"]["gap"],
                                    "ci": bootstrap_m_minus_f(a, b, rng)["ci_change"]}
            by[length_key(n)] = e
        v = overview[m]
        out[m] = {"label": label(v), "synthetic": v["synthetic"],
                  "whole_resume_24_model_run": v["whole_document"], "by_length": by,
                  "owner_only_minus_24_model_run": by["full"]["name"]["gap"] - v["whole_document"],
                  "short_line_minus_synthetic": by["35"]["line"]["gap"] - v["synthetic"]}
    if len(sha) != 1:
        raise SystemExit("the runs used different marker labels")

    def col(f):
        return {m: f(v) for m, v in out.items()}
    short_gap = col(lambda v: abs(v["short_line_minus_synthetic"]))
    worst = max(short_gap, key=short_gap.get)
    rest = [g for m, g in short_gap.items() if m != worst]
    line35 = col(lambda v: v["by_length"]["35"]["line_minus_name"])
    linefull = col(lambda v: v["by_length"]["full"]["line_minus_name"])
    change = col(lambda v: v["owner_only_minus_24_model_run"])
    return {
        "models": out,
        "labels_sha256": sha.pop(),
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
            "owner_only_change_min_abs": min(abs(c) for c in change.values()),
            "owner_only_change_max_abs": max(abs(c) for c in change.values()),
            "owner_only_n_sign_changes": sum(
                (v["by_length"]["full"]["name"]["gap"] > 0) != (v["whole_resume_24_model_run"] > 0)
                for v in out.values()),
        },
    }


def owner_labels(sha):
    """Tallies of the hand-checked labels the length x pronoun runs used."""
    import hashlib
    raw = LABELS.read_bytes()
    data = json.loads(raw)
    content = hashlib.sha256(json.dumps(data["labels"], sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()
    if hashlib.sha256(raw).hexdigest() != sha and (sha, content) != RUN_LABELS:
        raise SystemExit(f"{LABELS.name} does not hold the labels the runs used")
    labels = data["labels"]
    names = [h["ref"] for v in labels.values() for h in v["names"]]
    prons = [p["ref"] for v in labels.values() for p in v["pronouns"]]
    # the two labelling passes' agreement, as the file's description records it
    import re
    m = re.search(r"agreed on (\d+) of (\d+) names and (\d+) of (\d+) pronouns", data["description"])
    if not m or (int(m[2]), int(m[4])) != (len(names), len(prons)):
        raise SystemExit(f"{LABELS.name}: the description's agreement counts do not match the labels")
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
            "n_pronouns_no_person": prons.count("none")}


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
    overview = fw["overview"]
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
                     "24_model_run": "swapped every carried name and pronoun, many of which "
                                     "belong to other people; the length x pronoun runs change "
                                     "only the owner's (owner_only_minus_24_model_run)",
                     "synthetic_ci": "occupation-clustered 95% interval of the synthetic M-F; for "
                                     "jina-v3 the audit's same-order interval, while its synthetic "
                                     "M-F averages both orders"},
           "design": {"cut_words": LENGTHS[:-1],
                      "cut_words_text": ", ".join(map(str, LENGTHS[:-2])) + f", or {LENGTHS[-2]}",
                      "owner_line": OWNER_LINE,
                      "owner_line_words": OWNER_LINE_WORDS,
                      "owner_line_pronouns": len(_PRON_RE.findall(OWNER_LINE["male"]))},
           "owner_labels": owner_labels(lp["labels_sha256"]),
           "run_dates": run_dates(),
           "whole_resume": wr, "length_pronoun": lp, "word_counts": word_counts(LP_MODELS)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    TEX.write_text(tex_table(wr))

    c = wr["correlation"]
    print(f"{wr['n_models']} rerankers: {wr['n_male_direction']} male, {wr['n_female_direction']} "
          f"female, {wr['n_unclear']} unclear; rho {c['spearman_rho']:.2f} "
          f"[{c['spearman_ci'][0]:.2f}, {c['spearman_ci'][1]:.2f}], r {c['pearson_r']:.2f}")
    print(f"{'model':16s} {'synth':>6s} {'24run':>6s} | "
          + " | ".join(f"{k:>13s}" for k in lp["lengths"]))
    for m, v in lp["models"].items():
        cells = [f"{v['by_length'][k]['name']['gap']:+5.1f} / {v['by_length'][k]['line']['gap']:+5.1f}"
                 for k in lp["lengths"]]
        print(f"{m:16s} {v['synthetic']:6.1f} {v['whole_resume_24_model_run']:6.1f} | "
              + " | ".join(cells))
    print("(cells: name only / name + pronoun line)")
    print(json.dumps(lp["derived"], indent=1))
    print("wrote", OUT.relative_to(ROOT), "and", TEX.relative_to(ROOT))


if __name__ == "__main__":
    main()
