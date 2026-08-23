"""
De-identification mitigation experiment (synthetic counterfactual corpus).

Tests the proposed policy -- strip gender markers from the document BEFORE
re-ranking -- and, crucially, tests the *weaker* variants the paper's argument
depends on.  All transforms come from the shared module `code/deid_transforms.py`
so the synthetic study and the realistic-pool study provably apply the same rules.

Conditions (see deid_transforms.VARIANTS):
    original        the banked baseline scores (or a local `none` re-score)
    names_only      names -> "the candidate"; pronouns left gendered
    pronouns_only   pronouns -> one uniform token ("they"); names left in place
    full            names + pronouns -> the recommended policy
    gram_her_their  grammatical they/them/their, naive: "her" -> "their"
    gram_her_them   grammatical they/them/their, naive: "her" -> "them"
    gram_pos        POS-aware: "her" -> "their" before a noun, else "them"

What the review (REVIEW_TODO.md) requires this script to settle:

  A4  Names-only does NOT "make the male skew worse".  It raises the male-favoured
      *fraction* while LOWERING the magnitude (mean |score gap|).  Both are
      reported side by side for every condition so the prose can be corrected.

  A5  The paper claims a grammatically faithful mapping cannot equalise the twins,
      because English "her" is both the possessive (= his) and the object (= him).
      That is true only of a NAIVE mapping.  This script measures, on the full
      corpus, the residue (% of pairs whose two texts are still different) for
      gram_her_their and gram_her_them, and confirms that gram_pos leaves none.

  A10 The `full` condition's 100% tie rate is an IDENTITY, not a measurement: the
      transform makes the two twins the same string, and identical (query, doc)
      strings are de-duplicated before scoring, so one float is reused.  Every
      condition therefore carries `pct_pairs_identical_text` and, when that is
      100%, `sanity_check: true` -- it must never be presented as an empirical
      finding.

  A2  Three-way split (male / tie / female) and a tie-aware percentage per
      condition, not "delta > 0 vs everything else".

  B1  Cluster bootstrap by occupation (seeded) on every percentage and on
      mean |score gap|.

  B4  Runs for all four cached local re-rankers (model name is a CLI argument).

Outputs (never touches any *_full_raw.json):
    results/deidentification_<model-slug>.json     one per model
    results/deidentification_summary.json          all models combined
    results/deidentification_bge.json              back-compat superset for bge-v2-m3
    results/tex/deidentification.tex

Run:  venv/bin/python code/deidentification_experiment.py            # all 4 models
      venv/bin/python code/deidentification_experiment.py --model BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import ClusterStat, stable_seed  # noqa: E402
from deid_transforms import DEFAULT_PLACEHOLDER, deidentify  # noqa: E402
from labels import load_labels  # noqa: E402
from synthetic_dataset import NAME_PAIRS, generate_dataset  # noqa: E402

SCRIPT_VERSION = "deidentification_experiment.py v2.0 (REVIEW_TODO A4/A5/A10/A2/B1/B4)"
RESULTS = ROOT / "results"
TEX_DIR = RESULTS / "tex"
SEED = 20260822

LOCAL_RERANKERS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v2-base-multilingual",
]

# Raw banked scores, where they exist, supply the `original` baseline.
RAW_FOR_MODEL = {
    "BAAI/bge-reranker-v2-m3": "bge-reranker-v2-m3_full_raw.json",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "ms-marco-MiniLM-L-12-v2_full_raw.json",
    "jinaai/jina-reranker-v2-base-multilingual": "jina-reranker-v2-base-multilingual_full_raw.json",
}

SCORED_CONDITIONS = ("names_only", "pronouns_only", "full",
                     "gram_her_their", "gram_her_them", "gram_pos")
ALL_CONDITIONS = ("original",) + SCORED_CONDITIONS
CATEGORIES = ("male", "female", "neutral")
TRIPLES_FOR_UTILITY = []

# The synthetic corpus's own name list (the field version needs NER -- see B8/G6).
NAMES = sorted({n for pair in NAME_PAIRS for n in pair} | {"A. Smith", "A.", "Smith"},
               key=len, reverse=True)


def key(occ, qt, pair, tmpl):
    return f"{occ}|{qt}|{pair}|{tmpl}"


def slug(model: str) -> str:
    return model.replace("/", "_")


# ---------------------------------------------------------------------------
# Cluster bootstrap for a plain mean (mean |score gap|)
# ---------------------------------------------------------------------------

class ClusterMean:
    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, occ, v):
        self.s[occ] += v
        self.n[occ] += 1

    def summary(self, seed, B):
        occs = sorted(self.n)
        if not occs:
            return {}
        s = np.array([self.s[o] for o in occs])
        n = np.array([self.n[o] for o in occs], dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(occs), size=(B, len(occs)))
        boot = s[idx].sum(1) / n[idx].sum(1)
        return {"mean_abs_delta": float(s.sum() / n.sum()),
                "mean_abs_delta_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}


# ---------------------------------------------------------------------------
# Build every (query, doc) string the run needs, deduplicated
# ---------------------------------------------------------------------------

def build_texts(triples, labels):
    """Return (rows, needed_pairs).

    rows: one dict per triple with, per condition, the (query, doc_m, doc_f) keys.
    needed_pairs: the deduplicated set of (query, doc) strings to score.
    """
    rows, need = [], {}
    for t in triples:
        if labels.get(t.occupation) is None:
            continue
        rec = {"t": t, "cat": labels[t.occupation], "cond": {}}
        for cond in SCORED_CONDITIONS:
            q = t.query.lower()  # the policy lowercases the query too
            dm = deidentify(t.doc_male, cond, NAMES, DEFAULT_PLACEHOLDER, lowercase=True)
            df = deidentify(t.doc_female, cond, NAMES, DEFAULT_PLACEHOLDER, lowercase=True)
            rec["cond"][cond] = (q, dm, df)
            need[(q, dm)] = None
            need[(q, df)] = None
        rows.append(rec)
    return rows, list(need)


_MODEL_CACHE = {}


def _patch_transformers_for_jina():
    """jina-reranker-v2's remote code calls a helper removed in transformers 5.x.

    Same shim run_experiment.py installs; without it CrossEncoder(...) raises
    ImportError on `create_position_ids_from_input_ids`.
    """
    import torch
    import transformers.models.xlm_roberta.modeling_xlm_roberta as _xlm

    if hasattr(_xlm, "create_position_ids_from_input_ids"):
        return

    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental.long() + padding_idx

    _xlm.create_position_ids_from_input_ids = create_position_ids_from_input_ids


def get_model(model_name, device):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder
        _patch_transformers_for_jina()
        # jina-reranker-v2 ships custom modelling code (same flag run_experiment.py used).
        _MODEL_CACHE[model_name] = CrossEncoder(model_name, device=device, max_length=512,
                                               trust_remote_code=True)
    return _MODEL_CACHE[model_name]


def score_pairs(model_name, pairs, device):
    if not pairs:
        return {}
    model = get_model(model_name, device)
    s = model.predict([[q, d] for q, d in pairs], batch_size=64,
                      show_progress_bar=True, convert_to_numpy=True)
    return {p: float(v) for p, v in zip(pairs, s)}


def baseline_scores(model_name, rows, device):
    """(key -> (score_male, score_female)) for the untransformed documents."""
    raw = RAW_FOR_MODEL.get(model_name)
    if raw and (RESULTS / raw).exists():
        d = json.load(open(RESULTS / raw))
        assert d["model"] == model_name, f"{raw} holds {d['model']!r}, expected {model_name!r}"
        return ({key(r["occupation"], r["query_type"], r["name_pair"], r["template_style"]):
                 (r["score_male"], r["score_female"]) for r in d["results"]},
                f"banked raw scores ({raw})")
    need = {}
    for rec in rows:
        t = rec["t"]
        need[(t.query, t.doc_male)] = None
        need[(t.query, t.doc_female)] = None
    print(f"  no banked baseline for {model_name}; scoring {len(need)} original pairs locally")
    sc = score_pairs(model_name, list(need), device)
    return ({key(rec["t"].occupation, rec["t"].query_type, rec["t"].name_pair,
                 rec["t"].template_style):
             (sc[(rec["t"].query, rec["t"].doc_male)], sc[(rec["t"].query, rec["t"].doc_female)])
             for rec in rows},
            "scored locally in this run (variant `none`, no lowercasing)")


# ---------------------------------------------------------------------------
# B3: utility retention -- does the transform hurt relevance?
# ---------------------------------------------------------------------------
# Legally load-bearing: under Title VII a "less discriminatory alternative" must be
# EQUALLY EFFECTIVE (42 U.S.C. 2000e-2(k)(1)(A)(ii); Watson v. Fort Worth Bank &
# Trust, 487 U.S. 977, 998).  So the transform must be shown not to degrade ranking.
#
# Design: for one query phrasing and each template, the pool is ONE document per
# occupation (a gender-mixed pool, seeded).  The query names one occupation; we ask
# whether that occupation's document still ranks first after the transform.

UTILITY_CONDITIONS = ("original", "names_only", "pronouns_only", "full", "gram_pos")


def utility_test(model_name, triples, labels, device, query_types, conditions):
    docs, queries = {}, {}
    for t in triples:
        if labels.get(t.occupation) is None:
            continue
        docs.setdefault((t.occupation, t.template_style, t.name_pair), (t.doc_male, t.doc_female))
        queries[(t.occupation, t.query_type)] = t.query

    occs = sorted({o for o, _, _ in docs})
    tmpls = sorted({tm for _, tm, _ in docs})
    pairs_list = sorted({p for _, _, p in docs})
    rng = np.random.default_rng(SEED)
    # one fixed, seeded (name pair, gender) per occupation -> a gender-mixed pool
    pick = {o: (pairs_list[i % len(pairs_list)], bool(rng.integers(2)))
            for i, o in enumerate(occs)}

    need, plan = {}, []
    for cond in conditions:
        for tmpl in tmpls:
            pool = {}
            for o in occs:
                pair, is_male = pick[o]
                dm, df = docs[(o, tmpl, pair)]
                raw_doc = dm if is_male else df
                if cond == "original":
                    pool[o] = raw_doc
                else:
                    pool[o] = deidentify(raw_doc, cond, NAMES, DEFAULT_PLACEHOLDER, lowercase=True)
            for qt in query_types:
                for qo in occs:
                    q = queries[(qo, qt)]
                    q = q if cond == "original" else q.lower()
                    for o in occs:
                        need[(q, pool[o])] = None
                    plan.append((cond, tmpl, qt, qo, q, dict(pool)))

    print(f"  utility (B3): scoring {len(need)} unique (query, doc) pairs "
          f"[{len(conditions)} conditions x {len(tmpls)} templates x "
          f"{len(query_types)} query phrasing(s) x {len(occs)} occupations]")
    sc = score_pairs(model_name, list(need), device)

    agg = {c: defaultdict(list) for c in conditions}
    for cond, tmpl, qt, qo, q, pool in plan:
        ranked = sorted(occs, key=lambda o: -sc[(q, pool[o])])
        rank = ranked.index(qo) + 1
        agg[cond]["top1"].append(1.0 if rank == 1 else 0.0)
        agg[cond]["top3"].append(1.0 if rank <= 3 else 0.0)
        agg[cond]["rr"].append(1.0 / rank)

    out = {"config": {"query_types": list(query_types), "templates": tmpls,
                      "pool_size": len(occs), "n_queries": len(plan) // len(conditions),
                      "pool_composition": "one document per occupation, gender-mixed "
                                          "(seeded choice of name pair and gender)",
                      "seed": SEED}}
    for cond in conditions:
        a = agg[cond]
        out[cond] = {"top1_pct": 100.0 * float(np.mean(a["top1"])),
                     "top3_pct": 100.0 * float(np.mean(a["top3"])),
                     "mrr": float(np.mean(a["rr"])),
                     "n_queries": len(a["top1"])}
    base = out[conditions[0]]["top1_pct"]
    for cond in conditions:
        out[cond]["top1_pct_delta_vs_original"] = out[cond]["top1_pct"] - base
    return out


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(model_name, triples, labels, args, device):
    print(f"\n{'='*100}\nMODEL: {model_name}   device={device}\n{'='*100}")
    rows, pairs = build_texts(triples, labels)
    print(f"  triples: {len(rows)}   unique (query, doc) strings to score: {len(pairs)}")

    base, base_src = baseline_scores(model_name, rows, device)
    print(f"  baseline source: {base_src}")
    scores = score_pairs(model_name, pairs, device)

    # ---- accumulate ----
    stat = {c: {"all": ClusterStat(), "abs": ClusterMean(),
                "cat": {k: ClusterStat() for k in CATEGORIES},
                "abs_cat": {k: ClusterMean() for k in CATEGORIES},
                "identical": 0, "n": 0,
                "identical_by_template": defaultdict(lambda: [0, 0]),
                "residue_examples": []}
            for c in ALL_CONDITIONS}

    for rec in rows:
        t, cat = rec["t"], rec["cat"]
        pairs_for_cond = {"original": base[key(t.occupation, t.query_type,
                                               t.name_pair, t.template_style)]}
        for cond in SCORED_CONDITIONS:
            q, dm, df = rec["cond"][cond]
            pairs_for_cond[cond] = (scores[(q, dm)], scores[(q, df)])

        for cond, (sm, sf) in pairs_for_cond.items():
            C = stat[cond]
            delta = sm - sf
            C["n"] += 1
            C["all"].add(t.occupation, delta, cat)
            C["cat"][cat].add(t.occupation, delta, cat)
            C["abs"].add(t.occupation, abs(delta))
            C["abs_cat"][cat].add(t.occupation, abs(delta))
            if cond == "original":
                same = t.doc_male == t.doc_female
            else:
                _, dm, df = rec["cond"][cond]
                same = dm == df
                if not same and len(C["residue_examples"]) < 3:
                    C["residue_examples"].append(
                        {"template": t.template_style, "occupation": t.occupation,
                         "male_form": dm, "female_form": df})
            C["identical"] += int(same)
            tb = C["identical_by_template"][t.template_style]
            tb[0] += int(same); tb[1] += 1

    # ---- summarise ----
    B = args.bootstrap
    out = {"reranker": model_name, "device": device,
           "baseline_source": base_src, "name_placeholder": DEFAULT_PLACEHOLDER,
           "n_triples": len(rows), "conditions": {}}
    for cond in ALL_CONDITIONS:
        C = stat[cond]
        # `overall` holds the new tie-aware three-way statistics; the old flat keys
        # (`pct_male` as a per-category dict, `tie_rate`, `bdi_*`) are preserved
        # below so legal_paper/generate_figures.py keeps working while it is updated.
        overall = C["all"].summary(stable_seed(model_name, cond), B)
        overall.update(C["abs"].summary(stable_seed(model_name, cond, "abs"), B))
        e = {"overall": overall}
        e.update(C["abs"].summary(stable_seed(model_name, cond, "abs"), B))
        e["pct_pairs_identical_text"] = 100.0 * C["identical"] / C["n"]
        e["pct_pairs_with_residue"] = 100.0 - e["pct_pairs_identical_text"]
        e["identical_by_template"] = {k: 100.0 * v[0] / v[1]
                                      for k, v in sorted(C["identical_by_template"].items())}
        e["residue_examples"] = C["residue_examples"]
        if e["pct_pairs_identical_text"] == 100.0:
            e["sanity_check"] = True
            e["sanity_check_note"] = (
                "A10: the transform makes the two twins the same string, so the tie rate "
                "of 100% is an identity of the design, not an empirical measurement. "
                "Identical (query, doc) strings are scored once and the float is reused.")
        else:
            e["sanity_check"] = False
        e["by_category"] = {}
        for c in CATEGORIES:
            d = C["cat"][c].summary(stable_seed(model_name, cond, c), B)
            if d:
                d.update(C["abs_cat"][c].summary(stable_seed(model_name, cond, c, "abs"), B))
                e["by_category"][c] = d
        out["conditions"][cond] = e

    if not args.no_utility:
        out["utility"] = utility_test(model_name, TRIPLES_FOR_UTILITY, labels, device,
                                      args.utility_query_types, UTILITY_CONDITIONS)

    # ---- back-compat keys expected by older readers / figure code ----
    for e in out["conditions"].values():
        e["tie_rate"] = e["overall"]["pct_tie"] / 100.0
        e["pct_male"] = {c: e["by_category"][c]["pct_male"] / 100.0 for c in e["by_category"]}
        e["bdi_male"] = e["by_category"]["male"]["stereotype_match_rate"] / 100.0
        e["bdi_female"] = e["by_category"]["female"]["stereotype_match_rate"] / 100.0
    return out


def add_residue_view(res):
    """Restrict each condition's split to the pairs the transform actually LEFT DIFFERENT.

    A5 needs this.  For the naive grammatical mappings most pairs become identical
    text (hence forced ties), so the pooled `pct_male` is diluted toward zero and
    reads as "the model no longer favours the man" when the truth is "the model
    only had 25% of pairs left to disagree about".  Pure arithmetic on numbers the
    run already produced -- no re-scoring.
    """
    for e in res["conditions"].values():
        ident = e["pct_pairs_identical_text"]
        resid = e["pct_pairs_with_residue"]
        o = e["overall"]
        if resid <= 0:
            e["residue_only"] = {"n_pct_of_pairs": 0.0,
                                 "note": "the transform left no pair different"}
            continue
        # Every identical-text pair is necessarily a tie; the rest of the ties are
        # genuine ties between two DIFFERENT strings.
        tie_resid = max(o["pct_tie"] - ident, 0.0)
        m, f = o["pct_male"], o["pct_female"]
        e["residue_only"] = {
            "n_pct_of_pairs": resid,
            "pct_male": 100.0 * m / resid,
            "pct_tie": 100.0 * tie_resid / resid,
            "pct_female": 100.0 * f / resid,
            "tie_aware_pct_male": 100.0 * (m + 0.5 * tie_resid) / resid,
        }
    return res


def print_model(res):
    print(f"\n{'condition':<16}{'%M':>7}{'%tie':>7}{'%F':>7}{'tie-aware %M':>14}"
          f"{'mean|gap|':>12}{'mean|gap| 95% CI':>22}{'identical text %':>18}")
    for cond in ALL_CONDITIONS:
        e = res["conditions"][cond]
        o = e["overall"]
        ci = f"[{e['mean_abs_delta_ci'][0]:.5f}, {e['mean_abs_delta_ci'][1]:.5f}]"
        print(f"{cond:<16}{o['pct_male']:>7.1f}{o['pct_tie']:>7.1f}{o['pct_female']:>7.1f}"
              f"{o['tie_aware_pct_male']:>14.1f}{e['mean_abs_delta']:>12.5f}{ci:>22}"
              f"{e['pct_pairs_identical_text']:>18.1f}"
              + ("   [SANITY CHECK, not a result]" if e["sanity_check"] else ""))
    print(f"\n{'condition':<16}" + "".join(f"{c + ' jobs %M':>16}" for c in CATEGORIES))
    for cond in ALL_CONDITIONS:
        e = res["conditions"][cond]
        print(f"{cond:<16}" + "".join(
            f"{e['by_category'][c]['pct_male']:>16.1f}" for c in CATEGORIES))
    if "utility" in res:
        u = res["utility"]
        print("\nB3 utility retention (does the transform hurt relevance?)  pool = "
              f"{u['config']['pool_size']} occupations, {u['config']['n_queries']} queries")
        print(f"{'condition':<16}{'top-1 %':>10}{'top-3 %':>10}{'MRR':>8}{'d top-1 vs original':>22}")
        for cond in UTILITY_CONDITIONS:
            c = u[cond]
            print(f"{cond:<16}{c['top1_pct']:>10.2f}{c['top3_pct']:>10.2f}{c['mrr']:>8.3f}"
                  f"{c['top1_pct_delta_vs_original']:>+22.2f}")

    print("\nA5 residue: pairs the transform LEFT DIFFERENT, and how those pairs split")
    print(f"{'condition':<16}{'% of pairs w/ residue':>23}{'of those: %M':>14}{'%tie':>7}{'%F':>7}")
    for cond in SCORED_CONDITIONS:
        r = res["conditions"][cond]["residue_only"]
        if r["n_pct_of_pairs"] <= 0:
            print(f"{cond:<16}{0.0:>23.1f}{'--':>14}{'--':>7}{'--':>7}")
        else:
            print(f"{cond:<16}{r['n_pct_of_pairs']:>23.1f}{r['pct_male']:>14.1f}"
                  f"{r['pct_tie']:>7.1f}{r['pct_female']:>7.1f}")

    print("\nA5 residue by template (% of pairs whose two texts are IDENTICAL after the transform):")
    tmpls = sorted(res["conditions"]["full"]["identical_by_template"])
    print(f"{'condition':<16}" + "".join(f"{t:>20}" for t in tmpls))
    for cond in SCORED_CONDITIONS:
        e = res["conditions"][cond]
        print(f"{cond:<16}" + "".join(
            f"{e['identical_by_template'][t]:>20.1f}" for t in tmpls))


def main():
    ap = argparse.ArgumentParser(description="De-identification mitigation experiment")
    ap.add_argument("--model", action="append", default=None,
                    help="re-ranker to run (repeatable); default: all four cached local models")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--strict-labels", action="store_true")
    ap.add_argument("--no-utility", action="store_true", help="skip the B3 utility test")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing per-model JSON when it was written by this same "
                         "script version against this same label source (skips re-scoring)")
    ap.add_argument("--utility-query-types", nargs="+", default=["bare"],
                    help="query phrasings used by the B3 utility test (cost scales linearly)")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    labels, label_source = load_labels(strict=args.strict_labels)
    print(f"Label source: {label_source} ({len(labels)} occupations)")
    triples = generate_dataset(perturbation_levels=["names_and_pronouns"], query_types=None)
    print(f"Triples: {len(triples)}")
    global TRIPLES_FOR_UTILITY
    TRIPLES_FOR_UTILITY = triples

    models = args.model or LOCAL_RERANKERS
    meta = {"script_version": SCRIPT_VERSION,
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "label_source": label_source, "seed": SEED, "bootstrap_B": args.bootstrap,
            "condition_order": list(ALL_CONDITIONS)}

    n_expected = sum(1 for t in triples if labels.get(t.occupation) is not None)
    allres = {}
    for m in models:
        cached = RESULTS / f"deidentification_{slug(m)}.json"
        res = None
        if args.reuse and cached.exists():
            prev = json.loads(cached.read_text())
            if (prev.get("script_version") == SCRIPT_VERSION
                    and prev.get("label_source") == label_source
                    and prev.get("n_triples") == n_expected
                    and "utility" in prev):
                print(f"\n--- reusing {cached.name} (same script version and label source) ---")
                res = prev
            else:
                print(f"  {cached.name} is stale; recomputing")
        if res is None:
            res = run_model(m, triples, labels, args, device)
            res.update(meta)
        add_residue_view(res)
        allres[m] = res
        print_model(res)
        cached.write_text(json.dumps(res, indent=2))
        print(f"\nSaved -> {cached}")
        if m == "BAAI/bge-reranker-v2-m3":
            legacy = RESULTS / "deidentification_bge.json"
            legacy.write_text(json.dumps(res, indent=2))
            print(f"Saved -> {legacy}  (back-compat superset)")

    combined = dict(meta)
    combined["models"] = allres
    (RESULTS / "deidentification_summary.json").write_text(json.dumps(combined, indent=2))

    # ---- cross-model table (B4) ----
    print("\n" + "=" * 110)
    print("B4  ALL MODELS x CONDITION   (tie-aware % male-favoured / mean |score gap| / % ties)")
    print("=" * 110)
    print(f"{'condition':<16}" + "".join(f"{m.split('/')[-1][:24]:>26}" for m in models))
    for cond in ALL_CONDITIONS:
        cells = []
        for m in models:
            e = allres[m]["conditions"][cond]
            cells.append(f"{e['overall']['tie_aware_pct_male']:.1f} / "
                         f"{e['mean_abs_delta']:.4f} / {e['overall']['pct_tie']:.0f}%")
        print(f"{cond:<16}" + "".join(f"{c:>26}" for c in cells))

    # ---- tex fragment ----
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    esc = lambda s: str(s).replace("_", r"\_")  # noqa: E731
    lines = ["% Generated by code/deidentification_experiment.py -- do not edit by hand.",
             r"\begin{tabular}{l" + "r" * (3 * len(models)) + "}", r"\toprule",
             r"Condition & " + " & ".join(
                 rf"\multicolumn{{3}}{{c}}{{{esc(m.split('/')[-1])}}}" for m in models) + r" \\",
             r" & " + " & ".join([r"\%\,M & tie \% & mean $|$gap$|$"] * len(models)) + r" \\",
             r"\midrule"]
    for cond in ALL_CONDITIONS:
        cells = []
        for m in models:
            e = allres[m]["conditions"][cond]
            cells += [f"{e['overall']['tie_aware_pct_male']:.1f}",
                      f"{e['overall']['pct_tie']:.1f}", f"{e['mean_abs_delta']:.4f}"]
        note = r"$^{\dagger}$" if allres[models[0]]["conditions"][cond]["sanity_check"] else ""
        lines.append(f"{esc(cond)}{note} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $^{\dagger}$ identity of the design (twins become the same string), "
              r"not an empirical measurement -- REVIEW_TODO A10."]
    (TEX_DIR / "deidentification.tex").write_text("\n".join(lines) + "\n")

    # ---- deid_conditions.tex: tabular-only, one row per condition, primary model ----
    primary = "BAAI/bge-reranker-v2-m3" if "BAAI/bge-reranker-v2-m3" in allres else models[0]
    r = allres[primary]
    u = r.get("utility", {})
    have_u = bool(u)
    label = {"original": "Original (no transform)",
             "names_only": "Names only",
             "pronouns_only": "Pronouns only",
             "full": "Full (names + pronouns $\\to$ one token)",
             "gram_her_their": "Grammatical, naive (`her' $\\to$ `their')",
             "gram_her_them": "Grammatical, naive (`her' $\\to$ `them')",
             "gram_pos": "Grammatical, POS-aware"}
    cols = "lrrrr" + ("r" if have_u else "")
    head = (r"Condition & \%\,male-favoured & \%\,ties & mean $|$score gap$|$ & "
            r"twins identical \%")
    if have_u:
        head += r" & top-1 \%"
    lines = ["% Generated by code/deidentification_experiment.py -- do not edit by hand.",
             f"% model: {primary}; label source: {label_source}",
             r"\begin{tabular}{" + cols + "}", r"\toprule", head + r" \\", r"\midrule"]
    for cond in ALL_CONDITIONS:
        e = r["conditions"][cond]
        o = e["overall"]
        dag = r"$^{\dagger}$" if e["sanity_check"] else ""
        row = (f"{label.get(cond, esc(cond))}{dag} & {o['tie_aware_pct_male']:.1f} & "
               f"{o['pct_tie']:.1f} & {e['mean_abs_delta']:.4f} & "
               f"{e['pct_pairs_identical_text']:.1f}")
        if have_u:
            row += f" & {u[cond]['top1_pct']:.1f}" if cond in u else " & --"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $^{\dagger}$ the transform makes the two twins the same string, so the tie",
              r"% rate is an identity of the design, not a measurement (REVIEW_TODO A10)."]
    (TEX_DIR / "deid_conditions.tex").write_text("\n".join(lines) + "\n")

    print(f"\nSaved -> {RESULTS / 'deidentification_summary.json'}")
    print(f"Saved -> {TEX_DIR / 'deidentification.tex'}")
    print(f"Saved -> {TEX_DIR / 'deid_conditions.tex'}")


if __name__ == "__main__":
    main()
