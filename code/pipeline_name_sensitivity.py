"""
Name-set sensitivity and rank-gap baselines for the two-stage pipeline simulation.

Two reviewer points about the pool simulation in code/compounding_experiment.py
(832 pools of 20 documents, retriever keeps 10, re-ranker shortlists 3; the
headline "79% of top-3 slots go to male-associated documents"):

  N1  The 79% interval resamples occupations only, never names.  With ten
      names per gender a few names could carry the result.  This script
      reports which names fill the top 3 (per-name occupancy), drops each
      name pair in turn (leave-one-pair-out), and resamples the name pairs
      (name-pair bootstrap, and a joint occupation x name-pair bootstrap) next
      to the occupation-only interval the paper prints.  The single-stage
      analogue is interval_sensitivity.py's `occupation_x_name` scheme.

  N2  In Table A4's "rank gap" column, positions 11-20 are ordered by the
      retriever (analyse_slate: pipe_rank[dropped] = topk + rankdata(-emb)),
      so a re-ranker with NO gender preference would still show a positive gap
      because the retriever's top 10 is ~60% male.  This script computes that
      baseline: the retriever keeps its top 10, every retrieved document gets
      the expected rank 5.5 (the mean of 1..10), positions 11-20 keep their
      retriever order.  The retriever-alone gap (positions 1-20 by retriever,
      = `emb_gap`) is the other reference point.  Both are also given for the
      top-3 male share, and the 28 re-rankers of the sweep are counted against
      each.

Everything is replayed from banked data: re-ranker scores from
results/*_full_raw.json and retriever scores from
results/compounding_emb_cache.json (compounding_experiment.embedding_scores
returns the cache when it covers every key, so no model weights are loaded;
the script asserts that torch was never imported).

Tie-breaks.  compounding_experiment.run_model draws one random permutation
per pool and per repeat from a seeded generator, iterating K = 5, 10, 15 in
turn.  `replay_tiebreaks` advances an identically seeded generator through
the same calls and keeps the K = 10 draws, so every statistic here uses the
SAME 200 tie-break draws per pool as results/compounding_bge_pipeline.json
(checked: the replay reproduces its headline numbers to float precision, and
the vectorised simulation agrees with analyse_slate on every pool).  The
leave-one-pair-out pools (18 documents) keep the surviving documents' draw
values; the bootstrap pools (20 documents) keep the draws by slot, so two
copies of a repeated pair never share a tie-break value.

Output: results/pipeline_name_sensitivity.json
Run:    cd <repo> && HF_HUB_OFFLINE=1 venv/bin/python code/pipeline_name_sensitivity.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from analyze_single_stage import detect_duplicates, load_raw, stable_seed  # noqa: E402
from compounding_experiment import (  # noqa: E402  (reuse the simulation's own rules)
    CATEGORIES, EMBEDDER_NAME, HEADLINE_SHORTLIST, HEADLINE_TOPK, LISTWISE_MODELS,
    MAIN_RERANKER, OUT_MAIN, OUT_SWEEP, SEED, SHORTLIST_GRID, TOPK_GRID,
    analyse_slate, build_slates, embedding_scores, key,
)
from labels import load_labels  # noqa: E402
from model_labels import model_label  # noqa: E402
from synthetic_dataset import NAME_PAIRS, generate_dataset  # noqa: E402

SCRIPT_VERSION = "pipeline_name_sensitivity.py v1.0 (review round 5, N1/N2)"
OUT = ROOT / "results" / "pipeline_name_sensitivity.json"
CAT_LABEL = {"male": "male-typed", "female": "female-typed", "neutral": "evenly split"}


# ---------------------------------------------------------------------------
# Pools as arrays
# ---------------------------------------------------------------------------

def stack_slates(slates):
    """Pools as (S, n) arrays.  Slot 2j is the male document of name pair j,
    slot 2j+1 its female twin (build_slates appends male-then-female per pair,
    pairs in NAME_PAIRS order); the simulation never relies on that order."""
    keys = list(slates)
    E = np.stack([slates[k]["emb"] for k in keys])
    RR = np.stack([slates[k]["rr"] for k in keys])
    is_male = np.stack([slates[k]["is_male"] for k in keys])
    assert (is_male == is_male[0]).all(), "pools differ in gender layout"
    is_male = is_male[0]
    n = len(is_male)
    assert n == 2 * len(NAME_PAIRS) and (is_male == (np.arange(n) % 2 == 0)).all()
    # The slot -> name map, verified against the triples that built the pools.
    names = [nm for pair in NAME_PAIRS for nm in pair]
    pair_of_slot = np.arange(n) // 2
    cats = np.array([k[3] for k in keys])
    occs = np.array([k[0] for k in keys])
    return keys, E, RR, is_male, names, pair_of_slot, cats, occs


def verify_slot_names(triples, labels, rr_map, keys, names):
    """Check slot j really holds names[j] by rebuilding one pool from the triples."""
    occ, tmpl, qt, _ = keys[0]
    seq = [t.name_pair for t in triples
           if (t.occupation, t.template_style, t.query_type) == (occ, tmpl, qt)
           and labels.get(t.occupation) is not None
           and key(t.occupation, t.query_type, t.name_pair, t.template_style) in rr_map]
    rebuilt = [nm for p in seq for nm in p.split("/")]
    assert rebuilt == names, f"slot/name mismatch: {rebuilt} vs {names}"


def replay_tiebreaks(name: str, n_slates: int, n: int, repeats: int) -> np.ndarray:
    """The exact permutations compounding_experiment.run_model drew for K = HEADLINE_TOPK.

    run_model seeds one generator per model and, for K in TOPK_GRID (in order),
    calls rng.permutation(n) `repeats` times per pool.  Advance through the
    earlier K values, then record the headline block.  (S, repeats, n)."""
    rng = np.random.default_rng(stable_seed(name, "slates") % (2 ** 32 - 1))
    for K in TOPK_GRID:
        if K == HEADLINE_TOPK:
            break
        for _ in range(n_slates * repeats):
            rng.permutation(n)
    tb = np.empty((n_slates, repeats, n), dtype=np.int64)
    for i in range(n_slates):
        for r in range(repeats):
            tb[i, r] = rng.permutation(n)
    return tb


# ---------------------------------------------------------------------------
# Vectorised replica of analyse_slate (all pools x all repeats at once)
# ---------------------------------------------------------------------------

def simulate(E, RR, is_male, TB, topk: int, short: int, occupancy: bool = False) -> dict:
    """Per-pool statistics, averaged over the tie-break draws in TB (S, R, n).

    Same rules as compounding_experiment.analyse_slate: best-first order by
    score with the draw as tie-breaker (np.lexsort), average ranks inside the
    retrieved and dropped blocks for the pipeline rank gap.  Adds
    `neutral_gap`: the pipeline rank gap when the re-ranker has no preference
    (every retrieved document at expected rank (topk+1)/2, dropped documents in
    retriever order)."""
    S, R, n = TB.shape
    F, M = ~is_male, is_male
    emb_rank = rankdata(-E, method="average", axis=1)
    rr_rank = rankdata(-RR, method="average", axis=1)
    out = {
        "emb_gap": emb_rank[:, F].mean(1) - emb_rank[:, M].mean(1),
        "rr_gap": rr_rank[:, F].mean(1) - rr_rank[:, M].mean(1),
    }
    acc = {k: np.zeros(S) for k in ("pipe_gap", "neutral_gap", "retrieved_m",
                                     "pipe_top_m", "rr_top_m", "emb_top_m")}
    occ = {k: np.zeros((S, n)) for k in ("pipe_top", "rr_top", "retrieved")} if occupancy else {}
    rows = np.arange(S)[:, None]
    for r in range(R):
        tb = TB[:, r, :]
        eo = np.lexsort((tb, -E), axis=1)
        ret, drop = eo[:, :topk], eo[:, topk:]
        rr_ret = np.take_along_axis(RR, ret, 1)
        ro = np.take_along_axis(ret, np.lexsort((np.take_along_axis(tb, ret, 1), -rr_ret), axis=1), 1)
        rr_o = np.lexsort((tb, -RR), axis=1)

        pipe_rank = np.empty((S, n))
        pipe_rank[rows, ret] = rankdata(-rr_ret, method="average", axis=1)
        pipe_rank[rows, drop] = topk + rankdata(-np.take_along_axis(E, drop, 1),
                                                method="average", axis=1)
        acc["pipe_gap"] += pipe_rank[:, F].mean(1) - pipe_rank[:, M].mean(1)
        pipe_rank[rows, ret] = (topk + 1) / 2.0
        acc["neutral_gap"] += pipe_rank[:, F].mean(1) - pipe_rank[:, M].mean(1)

        acc["retrieved_m"] += is_male[ret].sum(1)
        acc["pipe_top_m"] += is_male[ro[:, :short]].sum(1)
        acc["rr_top_m"] += is_male[rr_o[:, :short]].sum(1)
        acc["emb_top_m"] += is_male[eo[:, :short]].sum(1)
        if occupancy:
            for k, idx in (("pipe_top", ro[:, :short]), ("rr_top", rr_o[:, :short]), ("retrieved", ret)):
                np.add.at(occ[k], (rows, idx), 1.0)
    out.update({k: v / R for k, v in acc.items()})
    out.update({k: v / R for k, v in occ.items()})
    return out


class _Replay:
    """Stands in for the generator inside analyse_slate: hands back stored draws."""

    def __init__(self, draws):
        self._it = iter(draws)

    def permutation(self, n):
        d = next(self._it)
        assert len(d) == n
        return d


def check_against_analyse_slate(slates, keys, sim, TB, topk, short) -> dict:
    """The vectorised pass must equal analyse_slate on every pool, same draws."""
    worst = 0.0
    fields = {"pipe_gap": "pipe_gap", "retrieved_m": "retrieved_m", "emb_gap": "emb_gap",
              "rr_gap": "rr_gap", f"pipe_top{short}_m": "pipe_top_m",
              f"rr_top{short}_m": "rr_top_m", f"emb_top{short}_m": "emb_top_m"}
    for i, k in enumerate(keys):
        ref = analyse_slate(slates[k], topk, (short,), _Replay(TB[i]), TB.shape[1])
        for f_ref, f_sim in fields.items():
            worst = max(worst, abs(ref[f_ref] - sim[f_sim][i]))
    assert worst < 1e-9, f"vectorised simulation disagrees with analyse_slate by {worst}"
    return {"n_pools_checked": len(keys), "max_abs_difference": float(worst)}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def pct(x: np.ndarray, denom: float, mask=None) -> float:
    v = x if mask is None else x[mask]
    return float(v.sum() * 100.0 / (len(v) * denom))


def mean(x: np.ndarray, mask=None) -> float:
    v = x if mask is None else x[mask]
    return float(v.mean())


def by_cat(fn, cats):
    out = {c: fn(cats == c) for c in CATEGORIES}
    out["ALL"] = fn(None)
    return out


def occupancy_block(counts: np.ndarray, names, mask=None) -> dict:
    """counts: (S, n) expected slot occupancies per pool -> per-name shares."""
    c = counts if mask is None else counts[mask]
    tot = c.sum(0)
    share = tot / tot.sum() * 100.0
    order = np.argsort(-share, kind="stable")
    top3 = [names[j] for j in order[:3]]
    return {
        "share_pct": {names[j]: float(share[j]) for j in range(len(names))},
        "ranked": [[names[j], float(share[j])] for j in order],
        "top3_names": top3,
        "top3_names_share_pct": float(share[order[:3]].sum()),
        "max_name": names[order[0]],
        "max_share_pct": float(share[order[0]]),
        "min_name": names[order[-1]],
        "min_share_pct": float(share[order[-1]]),
        "effective_n_names": float(1.0 / ((share / 100.0) ** 2).sum()),
        "male_names_share_pct": float(share[0::2].sum()),
        "uniform_share_pct": 100.0 / len(names),
    }


def occupation_bootstrap_ci(per_slate: np.ndarray, occs, denom, seed, B) -> list[float]:
    """compounding_experiment.SlateAgg.summary's occupation bootstrap, same seed."""
    occ_list = sorted(set(occs))
    sums = np.array([per_slate[occs == o].sum() for o in occ_list])
    cnt = np.array([(occs == o).sum() for o in occ_list], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(occ_list), size=(B, len(occ_list)))
    boot = sums[idx].sum(1) / cnt[idx].sum(1) * 100.0 / denom
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Name-set sensitivity + rank-gap baselines "
                                             "for the pipeline simulation")
    ap.add_argument("--bootstrap", type=int, default=1000, help="name-pair bootstrap draws")
    ap.add_argument("--repeats", type=int, default=200,
                    help="tie-break draws per pool; must match compounding_bge_pipeline.json")
    ap.add_argument("--skip-slate-check", action="store_true",
                    help="skip the per-pool comparison with analyse_slate (~10 s)")
    args = ap.parse_args()
    t_start = time.time()
    K, S3 = HEADLINE_TOPK, HEADLINE_SHORTLIST

    labels, label_source = load_labels()
    triples = generate_dataset(perturbation_levels=["names_and_pronouns"], query_types=None)
    emb = embedding_scores(triples, refresh=False)
    assert "torch" not in sys.modules and "sentence_transformers" not in sys.modules, \
        "embedder weights were loaded; the cache must cover every key"

    raw = load_raw()
    _, dup, _ = detect_duplicates(raw)
    for d in dup:
        del raw[d]
    for m in [m for m in raw if m in LISTWISE_MODELS]:
        del raw[m]
    rr_map = {key(r["occupation"], r["query_type"], r["name_pair"], r["template_style"]):
              (r["score_male"], r["score_female"]) for r in raw[MAIN_RERANKER]["rows"]}

    main_json = json.load(open(OUT_MAIN))
    sweep = json.load(open(OUT_SWEEP))
    assert main_json["tiebreak_repeats"] == args.repeats and main_json["seed"] == SEED, \
        "repeats/seed differ from compounding_bge_pipeline.json; the draws would not replay"

    slates = build_slates(triples, emb, rr_map, labels)
    keys, E, RR, is_male, names, pair_of_slot, cats, occs = stack_slates(slates)
    verify_slot_names(triples, labels, rr_map, keys, names)
    S, n = E.shape
    n_pairs = len(NAME_PAIRS)
    print(f"{S} pools x {n} documents; label source: {label_source}")

    t0 = time.time()
    TB = replay_tiebreaks(MAIN_RERANKER, S, n, args.repeats)
    emb_ties = int(sum(len(np.unique(row)) < n for row in E))
    print(f"Replayed {args.repeats} tie-break draws per pool in {time.time() - t0:.1f}s; "
          f"pools with tied retriever scores: {emb_ties}")

    # ---- the worked pipeline, replayed ---------------------------------------
    t0 = time.time()
    sim = simulate(E, RR, is_male, TB, K, S3, occupancy=True)
    print(f"Vectorised pass: {time.time() - t0:.2f}s")
    ref = main_json["slate"]
    checks = {}
    for c in list(CATEGORIES) + ["ALL"]:
        m = None if c == "ALL" else cats == c
        got = {
            f"pipe_top{S3}_m": pct(sim["pipe_top_m"], S3, m),
            f"rr_top{S3}_m": pct(sim["rr_top_m"], S3, m),
            f"emb_top{S3}_m": pct(sim["emb_top_m"], S3, m),
            "retrieved_m": pct(sim["retrieved_m"], K, m),
            "pipe_gap": mean(sim["pipe_gap"], m),
            "emb_gap": mean(sim["emb_gap"], m),
            "rr_gap": mean(sim["rr_gap"], m),
        }
        checks[c] = {f: {"replayed": v, "banked": ref[c][f], "abs_diff": abs(v - ref[c][f])}
                     for f, v in got.items()}
    worst = max(d["abs_diff"] for c in checks.values() for d in c.values())
    assert worst < 1e-9, f"replay does not reproduce compounding_bge_pipeline.json ({worst})"
    print(f"Replay matches compounding_bge_pipeline.json (max |diff| {worst:.1e})")
    slate_check = None
    if not args.skip_slate_check:
        t0 = time.time()
        slate_check = check_against_analyse_slate(slates, keys, sim, TB, K, S3)
        print(f"Per-pool agreement with analyse_slate: max |diff| "
              f"{slate_check['max_abs_difference']:.1e} ({time.time() - t0:.1f}s)")

    occ_ci_seed = stable_seed(MAIN_RERANKER, "slate", "ALL", K)
    occ_ci_recomputed = occupation_bootstrap_ci(sim["pipe_top_m"], occs, S3, occ_ci_seed,
                                                main_json["bootstrap_B"])
    occ_ci_banked = ref["ALL"][f"pipe_top{S3}_m_ci"]
    assert np.allclose(occ_ci_recomputed, occ_ci_banked), "occupation bootstrap not reproduced"

    # ---- (a) per-name occupancy ----------------------------------------------
    stages = {"pipeline_top3": "pipe_top", "reranker_top3": "rr_top", "retriever_top10": "retrieved"}
    name_occ = {
        "definition": (
            "Share (%) of the slots at each stage taken by each of the 20 names, over the "
            f"{S} pools and the {args.repeats} tie-break draws: pipeline top-{S3} "
            f"(retriever keeps {K}, re-ranker shortlists {S3}), re-ranker-alone top-{S3}, "
            f"retriever top-{K}. Under no name effect every name would hold "
            f"{100.0 / n:.0f}%. `effective_n_names` is 1 / sum(share^2)."),
        "names": names,
        "name_pairs": [f"{a}/{b}" for a, b in NAME_PAIRS],
    }
    for label, field in stages.items():
        name_occ[label] = occupancy_block(sim[field], names)
    name_occ["by_category"] = {
        c: {label: occupancy_block(sim[field], names, cats == c) for label, field in stages.items()}
        for c in CATEGORIES}

    # ---- (b) leave-one-pair-out ----------------------------------------------
    t0 = time.time()
    loo = {}
    for j, (mn, fn) in enumerate(NAME_PAIRS):
        keep = pair_of_slot != j
        r = simulate(E[:, keep], RR[:, keep], is_male[keep], TB[:, :, keep], K, S3)
        loo[f"{mn}/{fn}"] = {
            f"pipe_top{S3}_m_pct_male": by_cat(lambda m: pct(r["pipe_top_m"], S3, m), cats),
            f"rr_top{S3}_m_pct_male": pct(r["rr_top_m"], S3),
            f"emb_top{S3}_m_pct_male": pct(r["emb_top_m"], S3),
            "retrieved_m_pct_male": pct(r["retrieved_m"], K),
            "pipe_gap": mean(r["pipe_gap"]),
        }
    vals = {p: v[f"pipe_top{S3}_m_pct_male"]["ALL"] for p, v in loo.items()}
    full = pct(sim["pipe_top_m"], S3)
    loo_block = {
        "definition": (
            f"Drop one of the {n_pairs} name pairs (both documents), leaving pools of "
            f"{n - 2}; retriever still keeps {K}, re-ranker still shortlists {S3}; "
            "same tie-break draws restricted to the surviving documents."),
        "full_set_pct_male": full,
        "per_pair": loo,
        "values_pct_male": vals,
        "min": min(vals.values()), "min_pair": min(vals, key=vals.get),
        "max": max(vals.values()), "max_pair": max(vals, key=vals.get),
        "range": max(vals.values()) - min(vals.values()),
        "max_abs_shift_from_full_set": max(abs(v - full) for v in vals.values()),
    }
    print(f"Leave-one-pair-out: {time.time() - t0:.1f}s")

    # ---- (c) name-pair bootstrap ---------------------------------------------
    B = args.bootstrap
    occ_list = sorted(set(occs))
    occ_idx = np.array([occ_list.index(o) for o in occs])
    n_occ = len(occ_list)
    # Occupation clusters per reporting group, as SlateAgg builds them: a
    # category's interval resamples that category's occupations, ALL resamples
    # all 52.
    occ_members = {c: np.array([i for i, o in enumerate(occ_list) if cats[occs == o][0] == c])
                   for c in CATEGORIES}
    occ_members["ALL"] = np.arange(n_occ)

    def pair_draw(rng):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        slots = np.stack([2 * idx, 2 * idx + 1], 1).ravel()      # keeps the M/F slot layout
        return idx, slots

    def pool_share(slots, rng=None):
        """Pipeline % male in the top-S3 for the pools built from `slots`; with
        `rng`, occupations are also resampled (per reporting group)."""
        per = simulate(E[:, slots], RR[:, slots], is_male, TB, K, S3)["pipe_top_m"]
        out = {}
        for c in list(CATEGORIES) + ["ALL"]:
            m = np.ones(S, bool) if c == "ALL" else cats == c
            if rng is None:
                out[c] = per[m].sum() * 100.0 / (m.sum() * S3)
            else:
                mem = occ_members[c]
                w = np.bincount(mem[rng.integers(0, len(mem), size=len(mem))],
                                minlength=n_occ).astype(float)[occ_idx] * m
                out[c] = (per * w).sum() * 100.0 / (w.sum() * S3)
        return out

    t0 = time.time()
    rng_pair = np.random.default_rng(stable_seed(MAIN_RERANKER, "name-pair-bootstrap"))
    draws_pair = {c: np.empty(B) for c in list(CATEGORIES) + ["ALL"]}
    n_distinct = np.empty(B, dtype=int)
    for b in range(B):
        idx, slots = pair_draw(rng_pair)
        n_distinct[b] = len(set(idx))
        for c, v in pool_share(slots).items():
            draws_pair[c][b] = v
    t_pair = time.time() - t0
    print(f"Name-pair bootstrap ({B} draws): {t_pair:.0f}s")

    t0 = time.time()
    rng_joint = np.random.default_rng(stable_seed(MAIN_RERANKER, "occupation-x-name-pair-bootstrap"))
    draws_joint = {c: np.empty(B) for c in list(CATEGORIES) + ["ALL"]}
    for b in range(B):
        _, slots = pair_draw(rng_joint)
        for c, v in pool_share(slots, rng_joint).items():
            draws_joint[c][b] = v
    t_joint = time.time() - t0
    print(f"Joint occupation x name-pair bootstrap ({B} draws): {t_joint:.0f}s")

    def ci_block(draws):
        return {c: {"ci": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                    "sd": float(np.std(d, ddof=1))}
                for c, d in draws.items()}

    boot_block = {
        "definition": (
            f"Percentile 95% intervals for the pipeline % male-associated in the top-{S3}. "
            f"name_pair_only: the {n_pairs} name pairs are resampled with replacement "
            "(a repeated pair appears twice, as separate documents with identical scores), "
            f"pools are rebuilt and the simulation rerun with the same {args.repeats} "
            "tie-break draws per pool. joint_occupation_name_pair: the 52 occupations and "
            f"the {n_pairs} pairs are resampled independently in each draw (a category's "
            "interval resamples that category's occupations, as the paper's does). "
            "occupation_only: the interval the paper prints (occupations resampled, names "
            "fixed), read from compounding_bge_pipeline.json and recomputed here from the "
            "replayed per-pool counts with the same seed."),
        "B": B,
        "point_pct_male": by_cat(lambda m: pct(sim["pipe_top_m"], S3, m), cats),
        "name_pair_only": ci_block(draws_pair),
        "joint_occupation_name_pair": ci_block(draws_joint),
        "occupation_only": {
            "ALL": {"ci": occ_ci_banked, "recomputed_ci": occ_ci_recomputed,
                    "source": f"{OUT_MAIN.relative_to(ROOT)} slate.ALL.pipe_top{S3}_m_ci"},
            **{c: {"ci": ref[c][f"pipe_top{S3}_m_ci"]} for c in CATEGORIES},
        },
        "name_pair_only_mean_distinct_pairs_per_draw": float(n_distinct.mean()),
        "seeds": {"name_pair_only": int(stable_seed(MAIN_RERANKER, "name-pair-bootstrap")),
                  "joint": int(stable_seed(MAIN_RERANKER, "occupation-x-name-pair-bootstrap")),
                  "occupation_only": int(occ_ci_seed)},
    }

    # ---- (d) rank-gap baselines ----------------------------------------------
    gaps = {
        "retriever_alone_gap": by_cat(lambda m: mean(sim["emb_gap"], m), cats),
        "order_neutral_reranker_gap": by_cat(lambda m: mean(sim["neutral_gap"], m), cats),
        "observed_pipeline_gap": by_cat(lambda m: mean(sim["pipe_gap"], m), cats),
        "observed_reranker_alone_gap": by_cat(lambda m: mean(sim["rr_gap"], m), cats),
    }
    gaps["pipeline_gap_minus_order_neutral"] = {
        c: gaps["observed_pipeline_gap"][c] - gaps["order_neutral_reranker_gap"][c]
        for c in gaps["observed_pipeline_gap"]}
    shares = {
        "retriever_alone_top3_pct_male": by_cat(lambda m: pct(sim["emb_top_m"], S3, m), cats),
        "order_neutral_reranker_top3_pct_male": by_cat(lambda m: pct(sim["retrieved_m"], K, m), cats),
        "observed_pipeline_top3_pct_male": by_cat(lambda m: pct(sim["pipe_top_m"], S3, m), cats),
    }
    emb_gap_matches = abs(gaps["retriever_alone_gap"]["ALL"] - ref["ALL"]["emb_gap"]) < 1e-9

    models = sweep["models"]
    g_ret, g_neu = gaps["retriever_alone_gap"]["ALL"], gaps["order_neutral_reranker_gap"]["ALL"]
    s_ret = shares["retriever_alone_top3_pct_male"]["ALL"]
    s_neu = shares["order_neutral_reranker_top3_pct_male"]["ALL"]
    per_model = {}
    for m, v in models.items():
        a = v["slate"]["ALL"]
        assert abs(a["emb_gap"] - g_ret) < 1e-9 and abs(a["retrieved_m"] - s_neu) < 1e-9, \
            f"{m}: different pools or retriever"
        per_model[m] = {
            "label": model_label(m),
            "pipe_gap": a["pipe_gap"],
            "pipe_gap_minus_order_neutral": a["pipe_gap"] - g_neu,
            "rr_gap": a["rr_gap"],
            f"pipe_top{S3}_m_pct_male": a[f"pipe_top{S3}_m_pct_male"],
            "gap_below_retriever_alone": a["pipe_gap"] < g_ret,
            "gap_below_order_neutral": a["pipe_gap"] < g_neu,
            f"top{S3}_below_retriever_alone": a[f"pipe_top{S3}_m_pct_male"] < s_ret,
            f"top{S3}_below_order_neutral": a[f"pipe_top{S3}_m_pct_male"] < s_neu,
        }
    counts = {
        "n_models": len(per_model),
        "n_gap_below_retriever_alone": sum(v["gap_below_retriever_alone"] for v in per_model.values()),
        "n_gap_below_order_neutral": sum(v["gap_below_order_neutral"] for v in per_model.values()),
        f"n_top{S3}_below_retriever_alone": sum(v[f"top{S3}_below_retriever_alone"] for v in per_model.values()),
        f"n_top{S3}_below_order_neutral": sum(v[f"top{S3}_below_order_neutral"] for v in per_model.values()),
        "n_rr_gap_negative": sum(v["rr_gap"] < 0 for v in per_model.values()),
        "models_gap_below_order_neutral": sorted(m for m, v in per_model.items() if v["gap_below_order_neutral"]),
        "models_gap_below_retriever_alone": sorted(m for m, v in per_model.items() if v["gap_below_retriever_alone"]),
        f"models_top{S3}_below_order_neutral": sorted(m for m, v in per_model.items() if v[f"top{S3}_below_order_neutral"]),
    }
    assert counts[f"n_top{S3}_below_retriever_alone"] == sweep["pipeline_spread"]["n_below_retriever_alone"]
    baselines = {
        "definition": (
            "Rank gap = mean rank of female-associated minus male-associated documents in "
            f"the {n}-document pool (rank 1 = top; + means the woman sits lower). "
            "retriever_alone: positions 1-20 by the retriever (= slate.*.emb_gap). "
            f"order_neutral_reranker: the retriever keeps its top {K}; every retrieved "
            f"document gets the expected rank {(K + 1) / 2}; positions {K + 1}-{n} keep their "
            "retriever order. It is what a re-ranker with no gender preference would show "
            "in Table A4's rank-gap column, and it is the same for every re-ranker because "
            "all sit behind the same retriever and pools (the retriever has no tied scores "
            f"in any pool: {emb_ties} pools with ties). For the top-{S3} share, "
            f"retriever_alone = the retriever's own top-{S3} (slate.*.emb_top{S3}_m) and "
            f"order_neutral_reranker = the male share of the retrieved top-{K} "
            f"(slate.*.retrieved_m), since {S3} of {K} are then drawn at random."),
        "rank_gap": gaps,
        "top3_share": shares,
        "retriever_alone_gap_equals_banked_emb_gap": bool(emb_gap_matches),
        "sweep_counts": counts,
        "sweep_per_model": per_model,
    }

    # ---- write ----------------------------------------------------------------
    runtime = time.time() - t_start
    payload = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "label_source": label_source,
        "seed": SEED,
        "bootstrap_B": B,
        "tiebreak_repeats": args.repeats,
        "embedder": EMBEDDER_NAME,
        "reranker": MAIN_RERANKER,
        "topk": K, "shortlist": S3,
        "n_slates": S, "n_occupations": n_occ, "n_name_pairs": n_pairs, "slate_size": n,
        "n_slates_by_category": {c: int((cats == c).sum()) for c in CATEGORIES},
        "tiebreak": (
            "the K=10 tie-break permutations of compounding_experiment.run_model, replayed "
            "from the same seeded generator (stable_seed(reranker, 'slates')), so every "
            "statistic uses the same draws as compounding_bge_pipeline.json"),
        "offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "torch_imported": "torch" in sys.modules,
            "embedder_scores_from_cache": True,
            "reranker_scores_from": os.path.relpath(raw[MAIN_RERANKER]["path"], ROOT),
        },
        "runtime_seconds": runtime,
        "validation": {
            "replay_vs_compounding_bge_pipeline": checks,
            "replay_max_abs_difference": worst,
            "per_pool_vs_analyse_slate": slate_check,
            "occupation_bootstrap_ci": {"banked": occ_ci_banked, "recomputed": occ_ci_recomputed},
        },
        "name_occupancy": name_occ,
        "leave_one_pair_out": loo_block,
        "name_pair_bootstrap": boot_block,
        "rank_gap_baselines": baselines,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    # ---- console summary ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"(a) NAME OCCUPANCY, % of slots  [{MAIN_RERANKER}]")
    print("=" * 78)
    print(f"{'name':<13}{'pipe top3':>11}{'rr top3':>10}{'retr top10':>12}")
    for j, nm in enumerate(names):
        print(f"{nm:<13}{name_occ['pipeline_top3']['share_pct'][nm]:>11.2f}"
              f"{name_occ['reranker_top3']['share_pct'][nm]:>10.2f}"
              f"{name_occ['retriever_top10']['share_pct'][nm]:>12.2f}")
    for label in stages:
        blk = name_occ[label]
        print(f"{label:<16} top-3 names {blk['top3_names']} = {blk['top3_names_share_pct']:.1f}%  "
              f"max {blk['max_name']} {blk['max_share_pct']:.1f}%  eff. names {blk['effective_n_names']:.1f}")

    print("\n(b) LEAVE-ONE-PAIR-OUT, pipeline % male top-3 (full set "
          f"{full:.2f})")
    for p, v in vals.items():
        print(f"  drop {p:<22}{v:>7.2f}")
    print(f"  min {loo_block['min']:.2f} ({loo_block['min_pair']})  max {loo_block['max']:.2f} "
          f"({loo_block['max_pair']})")

    print(f"\n(c) BOOTSTRAP 95% intervals, pipeline % male top-3 (B={B})")
    for c in list(CATEGORIES) + ["ALL"]:
        po = boot_block["point_pct_male"][c]
        a = boot_block["occupation_only"][c]["ci"]
        b_ = boot_block["name_pair_only"][c]["ci"]
        j_ = boot_block["joint_occupation_name_pair"][c]["ci"]
        print(f"  {c:<8} point {po:6.2f}  occupation [{a[0]:.1f}, {a[1]:.1f}]  "
              f"name-pair [{b_[0]:.1f}, {b_[1]:.1f}]  joint [{j_[0]:.1f}, {j_[1]:.1f}]")

    print("\n(d) RANK-GAP BASELINES (woman - man, 20-document pool)")
    print(f"{'cat':<9}{'retriever':>11}{'neutral rr':>12}{'pipeline':>10}{'rr alone':>10}"
          f"{'retr top3%':>12}{'neutral %':>11}{'pipe %':>9}")
    for c in list(CATEGORIES) + ["ALL"]:
        print(f"{c:<9}{gaps['retriever_alone_gap'][c]:>11.3f}"
              f"{gaps['order_neutral_reranker_gap'][c]:>12.3f}"
              f"{gaps['observed_pipeline_gap'][c]:>10.3f}{gaps['observed_reranker_alone_gap'][c]:>10.3f}"
              f"{shares['retriever_alone_top3_pct_male'][c]:>12.2f}"
              f"{shares['order_neutral_reranker_top3_pct_male'][c]:>11.2f}"
              f"{shares['observed_pipeline_top3_pct_male'][c]:>9.2f}")
    print(f"  retriever-alone gap == banked emb_gap: {emb_gap_matches}")
    print(f"  of {counts['n_models']} re-rankers: gap below retriever-alone "
          f"{counts['n_gap_below_retriever_alone']}, below order-neutral "
          f"{counts['n_gap_below_order_neutral']}; top-3 share below retriever-alone "
          f"{counts[f'n_top{S3}_below_retriever_alone']}, below order-neutral "
          f"{counts[f'n_top{S3}_below_order_neutral']}; rr_gap < 0: {counts['n_rr_gap_negative']}")
    print(f"\nSaved -> {OUT}  ({runtime:.0f}s)")


if __name__ == "__main__":
    main()
