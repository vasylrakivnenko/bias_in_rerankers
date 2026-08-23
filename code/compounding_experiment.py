"""
Two-stage pipeline experiment: retriever (bi-encoder) + re-ranker (cross-encoder).

Scenario: a SLATE of 20 equally-qualified candidates for one role -- 10 male-named
+ 10 female-named documents, identical except gender markers.  We measure where
each gender lands under (1) retriever alone, (2) re-ranker alone, (3) the full
pipeline (retriever keeps the top-K, re-ranker reorders the survivors).

Fixes applied from REVIEW_TODO.md:

  C1  Ties used to be broken by insertion order, and the male candidate was ALWAYS
      inserted first -- so every tie silently went to the man.  Ranks are now
      average ranks (deterministic, ties split evenly between the genders) and
      every top-k *selection* uses seeded random tie-breaking averaged over
      `--repeats` draws.  The old insertion-order behaviour is still computed and
      reported as `legacy_male_first_tiebreak` so the size of the bug is on record.

  A6  The claim "they pull in the same direction on 59.4% of pairs (above the 50%
      one would see by chance)" is arithmetically wrong: given each stage's own
      marginals the independence baseline is ~59.1%, i.e. there is NO excess
      agreement.  Both numbers are emitted side by side with an explicit
      `interpretation` field, and the informative statistic -- Spearman rho
      between delta_emb and delta_rr -- is computed instead.

  B1  Cluster bootstrap by occupation (seeded) for every reported percentage.

  B5  The whole slate simulation is run for ALL 14 re-rankers using the scores
      already banked in results/*_full_raw.json (zero API calls; only the
      bge-large embeddings are computed here).  This answers the inevitable
      "is 81% a property of pipelines or of this one re-ranker?".

      Plus sensitivity: retrieval cutoff K in {5, 10, 15} x shortlist size in
      {1, 3, 5}.

  A2  Three-way (male / tie / female) splits and tie-aware percentages
      everywhere, via the shared ClusterStat helper.

Outputs (never touches any *_full_raw.json):
    results/compounding_bge_pipeline.json      main worked example
    results/compounding_all_rerankers.json     B5 sweep + sensitivity
    results/compounding_emb_cache.json         cached bge-large scores (re-run fast)
    results/tex/compounding_sweep.tex

Run:  venv/bin/python code/compounding_experiment.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import (  # noqa: E402  (shared: A1 detection + tie-aware stats)
    ClusterStat, PROVENANCE, detect_duplicates, load_raw, stable_seed,
)
from labels import load_labels  # noqa: E402
from synthetic_dataset import generate_dataset  # noqa: E402

SCRIPT_VERSION = "compounding_experiment.py v2.0 (REVIEW_TODO C1/A6/A2/B1/B5)"
RESULTS = ROOT / "results"
TEX_DIR = RESULTS / "tex"
OUT_MAIN = RESULTS / "compounding_bge_pipeline.json"
OUT_SWEEP = RESULTS / "compounding_all_rerankers.json"
EMB_CACHE = RESULTS / "compounding_emb_cache.json"

EMBEDDER_NAME = "BAAI/bge-large-en-v1.5"
MAIN_RERANKER = "BAAI/bge-reranker-v2-m3"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

SEED = 20260822
TOPK_GRID = (5, 10, 15)
SHORTLIST_GRID = (1, 3, 5)
HEADLINE_TOPK = 10
HEADLINE_SHORTLIST = 3
CATEGORIES = ("male", "female", "neutral")


def key(occ, qt, pair, tmpl):
    return f"{occ}|{qt}|{pair}|{tmpl}"


# ---------------------------------------------------------------------------
# Stage 1: the retriever (only thing that needs a GPU here)
# ---------------------------------------------------------------------------

def embedding_scores(triples, refresh: bool) -> dict[str, tuple[float, float]]:
    """key -> (emb_score_male_doc, emb_score_female_doc). Cached on disk."""
    if EMB_CACHE.exists() and not refresh:
        cached = json.load(open(EMB_CACHE))
        need = {key(t.occupation, t.query_type, t.name_pair, t.template_style) for t in triples}
        if cached.get("embedder") == EMBEDDER_NAME and need <= set(cached["scores"]):
            print(f"Loaded {len(cached['scores'])} cached embedder scores from "
                  f"{EMB_CACHE.name} (covers all {len(need)} needed keys)")
            return {k: tuple(v) for k, v in cached["scores"].items()}

    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    doc_set, query_set = {}, {}
    for t in triples:
        query_set[t.query] = None
        doc_set[t.doc_male] = None
        doc_set[t.doc_female] = None
    docs, queries = list(doc_set), list(query_set)
    print(f"Embedding {len(docs)} docs / {len(queries)} queries on {device} ...")

    model = SentenceTransformer(EMBEDDER_NAME, device=device)
    doc_emb = model.encode(docs, batch_size=128, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    q_emb = model.encode([QUERY_INSTRUCTION + q for q in queries], batch_size=128,
                         normalize_embeddings=True, show_progress_bar=True,
                         convert_to_numpy=True)
    di = {d: i for i, d in enumerate(docs)}
    qi = {q: i for i, q in enumerate(queries)}

    out = {}
    for t in triples:
        qv = q_emb[qi[t.query]]
        out[key(t.occupation, t.query_type, t.name_pair, t.template_style)] = (
            float(qv @ doc_emb[di[t.doc_male]]), float(qv @ doc_emb[di[t.doc_female]]))
    EMB_CACHE.write_text(json.dumps(
        {"embedder": EMBEDDER_NAME, "query_instruction": QUERY_INSTRUCTION,
         "generated": _dt.datetime.now().isoformat(timespec="seconds"),
         "scores": {k: list(v) for k, v in out.items()}}))
    print(f"Cached embedder scores -> {EMB_CACHE}")
    return out


# ---------------------------------------------------------------------------
# Slate construction + ranking (C1: no insertion-order tie-breaks)
# ---------------------------------------------------------------------------

def build_slates(triples, emb, rr_map, labels):
    """(occupation, template, query_type, category) -> dict of numpy arrays."""
    acc = defaultdict(lambda: {"is_male": [], "emb": [], "rr": []})
    for t in triples:
        cat = labels.get(t.occupation)
        if cat is None:                       # dropped by the BLS relabelling
            continue
        k = key(t.occupation, t.query_type, t.name_pair, t.template_style)
        if k not in rr_map:
            continue
        em, ef = emb[k]
        sm, sf = rr_map[k]
        g = acc[(t.occupation, t.template_style, t.query_type, cat)]
        # NB: appended male-then-female only for bookkeeping; NOTHING downstream
        # may depend on this order (that was bug C1).
        g["is_male"].extend([True, False])
        g["emb"].extend([em, ef])
        g["rr"].extend([sm, sf])
    return {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in acc.items()}


def _order(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices best-first; ties resolved by `tiebreak` (a random permutation)."""
    return np.lexsort((tiebreak, -scores))


def analyse_slate(s, topk: int, shortlists, rng, repeats: int) -> dict:
    """Rank gaps (average ranks) + shortlist composition (random tie-breaks)."""
    is_male = s["is_male"]
    emb, rr = s["emb"], s["rr"]
    n = len(is_male)
    M, F = is_male, ~is_male

    # --- rank gaps: average ranks, so a tie contributes equally to both genders
    emb_rank = rankdata(-emb, method="average")
    rr_rank = rankdata(-rr, method="average")

    out = {
        "emb_gap": float(emb_rank[F].mean() - emb_rank[M].mean()),
        "rr_gap": float(rr_rank[F].mean() - rr_rank[M].mean()),
        "pipe_gap": 0.0,
        "retrieved_m": 0.0,
    }
    for S in shortlists:
        out[f"pipe_top{S}_m"] = 0.0
        out[f"rr_top{S}_m"] = 0.0
        out[f"emb_top{S}_m"] = 0.0
        out[f"legacy_pipe_top{S}_m"] = 0.0
        out[f"legacy_rr_top{S}_m"] = 0.0

    for _ in range(repeats):
        tb = rng.permutation(n)
        eo = _order(emb, tb)
        retrieved, dropped = eo[:topk], eo[topk:]
        ro = retrieved[_order(rr[retrieved], tb[retrieved])]

        pipe_rank = np.empty(n)
        # Average ranks inside each block so residual ties are still split fairly.
        pipe_rank[retrieved] = rankdata(-rr[retrieved], method="average")
        pipe_rank[dropped] = topk + rankdata(-emb[dropped], method="average")
        out["pipe_gap"] += float(pipe_rank[F].mean() - pipe_rank[M].mean()) / repeats
        out["retrieved_m"] += float(is_male[retrieved].sum()) / repeats

        rr_o = _order(rr, tb)
        for S in shortlists:
            out[f"pipe_top{S}_m"] += float(is_male[ro[:S]].sum()) / repeats
            out[f"rr_top{S}_m"] += float(is_male[rr_o[:S]].sum()) / repeats
            out[f"emb_top{S}_m"] += float(is_male[eo[:S]].sum()) / repeats

    # --- C1 diagnostic: the OLD behaviour (stable sort, male inserted first) ---
    # Slates are built male-then-female per name pair, so a stable sort sends every
    # tied pair's MALE document ahead of its female twin -- the bug itself.
    legacy_eo = np.argsort(-emb, kind="stable")
    legacy_ret = legacy_eo[:topk]
    legacy_ro = legacy_ret[np.argsort(-rr[legacy_ret], kind="stable")]
    legacy_rr_o = np.argsort(-rr, kind="stable")
    for S in shortlists:
        out[f"legacy_pipe_top{S}_m"] = float(is_male[legacy_ro[:S]].sum())
        out[f"legacy_rr_top{S}_m"] = float(is_male[legacy_rr_o[:S]].sum())
    return out


# ---------------------------------------------------------------------------
# Aggregation with a cluster bootstrap over occupations
# ---------------------------------------------------------------------------

class SlateAgg:
    """Occupation-clustered accumulator for slate-level means."""

    def __init__(self, fields):
        self.fields = list(fields)
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(int)

    def add(self, occ, rec):
        self.counts[occ] += 1
        for f in self.fields:
            self.sums[occ][f] += rec[f]

    @property
    def n_slates(self):
        return sum(self.counts.values())

    def summary(self, seed, B, denom: dict | None = None) -> dict:
        occs = sorted(self.counts)
        if not occs:
            return {}
        c = np.array([self.counts[o] for o in occs], dtype=float)
        arr = {f: np.array([self.sums[o][f] for o in occs], dtype=float) for f in self.fields}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(occs), size=(B, len(occs)))
        cb = c[idx].sum(1)
        out = {"n_slates": int(c.sum()), "n_occupations": len(occs)}
        for f in self.fields:
            point = arr[f].sum() / c.sum()
            boot = arr[f][idx].sum(1) / cb
            d = (denom or {}).get(f)
            if d:
                # Shortlist/retrieval counts are reported as PERCENTAGES on 0-100
                # (the convention legal_paper/numbers.json reads, and what the
                # four-fifths derivation needs).  The raw mean count is kept under
                # `<field>_count`, and `<field>_pct_male` is an explicit alias.
                out[f] = float(point * 100.0 / d)
                out[f + "_ci"] = [float(x) for x in np.percentile(boot * 100.0 / d, [2.5, 97.5])]
                out[f + "_pct_male"] = out[f]
                out[f + "_pct_male_ci"] = out[f + "_ci"]
                out[f + "_count"] = float(point)
                out[f + "_denominator"] = d
            else:
                out[f] = float(point)
                out[f + "_ci"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(name, rr_map, triples, emb, labels, args, want_item_level=False):
    rng = np.random.default_rng(stable_seed(name, "slates") % (2 ** 32 - 1))
    slates = build_slates(triples, emb, rr_map, labels)

    fields = ["emb_gap", "rr_gap", "pipe_gap", "retrieved_m"]
    for S in SHORTLIST_GRID:
        fields += [f"pipe_top{S}_m", f"rr_top{S}_m", f"emb_top{S}_m",
                   f"legacy_pipe_top{S}_m", f"legacy_rr_top{S}_m"]

    sensitivity = {}
    headline = None
    for K in TOPK_GRID:
        aggs = {c: SlateAgg(fields) for c in CATEGORIES}
        aggs["ALL"] = SlateAgg(fields)
        for (occ, tmpl, qt, cat), s in slates.items():
            rec = analyse_slate(s, K, SHORTLIST_GRID, rng, args.repeats)
            aggs[cat].add(occ, rec)
            aggs["ALL"].add(occ, rec)
        denom = {}
        for S in SHORTLIST_GRID:
            for pref in ("pipe", "rr", "emb", "legacy_pipe", "legacy_rr"):
                denom[f"{pref}_top{S}_m"] = S
        denom["retrieved_m"] = K
        B = args.bootstrap if K == HEADLINE_TOPK else min(args.bootstrap, 200)
        res = {c: aggs[c].summary(stable_seed(name, "slate", c, K), B, denom)
               for c in list(CATEGORIES) + ["ALL"]}
        sensitivity[f"topk{K}"] = res
        if K == HEADLINE_TOPK:
            headline = res

    sizes = {len(s["is_male"]) for s in slates.values()}
    assert len(sizes) == 1, f"slates of differing size: {sorted(sizes)}"
    slate_size = sizes.pop()
    n_male = {int(s["is_male"].sum()) for s in slates.values()}
    assert len(n_male) == 1 and 2 * n_male.copy().pop() == slate_size, "slate not gender-balanced"

    out = {
        "reranker": name,
        "vendor": PROVENANCE.get(name, ("unknown", "unknown"))[0],
        "embedder": EMBEDDER_NAME,
        "slate_size": slate_size,          # measured from the simulated slates
        "n_male_per_slate": slate_size // 2,
        "n_female_per_slate": slate_size // 2,
        "n_slates": len(slates),
        "headline_topk": HEADLINE_TOPK,
        "headline_shortlist": HEADLINE_SHORTLIST,
        "slate": headline,
        "sensitivity": sensitivity,
    }

    if want_item_level:
        out["item_level"] = item_level(rr_map, emb, triples, labels, args)
    return out


def item_level(rr_map, emb, triples, labels, args) -> dict:
    """Per-pair directional analysis + A6 (Spearman rho, agreement vs independence)."""
    de_all, dr_all, cats = [], [], []
    stats_e = {c: ClusterStat() for c in CATEGORIES}
    stats_r = {c: ClusterStat() for c in CATEGORIES}
    all_e, all_r = ClusterStat(), ClusterStat()
    agree = ClusterStat()

    for t in triples:
        cat = labels.get(t.occupation)
        k = key(t.occupation, t.query_type, t.name_pair, t.template_style)
        if cat is None or k not in rr_map:
            continue
        em, ef = emb[k]
        sm, sf = rr_map[k]
        de, dr = em - ef, sm - sf
        de_all.append(de); dr_all.append(dr); cats.append(cat)
        stats_e[cat].add(t.occupation, de, cat)
        stats_r[cat].add(t.occupation, dr, cat)
        all_e.add(t.occupation, de, cat)
        all_r.add(t.occupation, dr, cat)
        # encode "agree" as a pseudo-delta so ClusterStat's %male == % agreeing
        agree.add(t.occupation, 1.0 if (de > 0) == (dr > 0) else -1.0, "neutral")

    de_arr, dr_arr = np.array(de_all), np.array(dr_all)
    rho, p = spearmanr(de_arr, dr_arr)
    per_cat_rho = {}
    for c in CATEGORIES:
        m = np.array([x == c for x in cats])
        if m.sum() > 2:
            r_, p_ = spearmanr(de_arr[m], dr_arr[m])
            per_cat_rho[c] = {"spearman_rho": float(r_), "p": float(p_), "n": int(m.sum())}

    # A6: independence baseline given each stage's own marginals, per category.
    obs = agree.point()["pct_male"] / 100.0
    num = den = 0.0
    for c in CATEGORIES:
        pt_e, pt_r = stats_e[c].point(), stats_r[c].point()
        if not pt_e:
            continue
        pe, pr, n = pt_e["pct_male"] / 100.0, pt_r["pct_male"] / 100.0, pt_e["n"]
        num += n * (pe * pr + (1 - pe) * (1 - pr))
        den += n
    baseline = num / den if den else float("nan")

    return {
        "embedder": {c: stats_e[c].summary(stable_seed("emb", c), args.bootstrap)
                     for c in CATEGORIES},
        "reranker": {c: stats_r[c].summary(stable_seed("rr", c), args.bootstrap)
                     for c in CATEGORIES},
        "embedder_overall": all_e.summary(stable_seed("emb", "all"), args.bootstrap),
        "reranker_overall": all_r.summary(stable_seed("rr", "all"), args.bootstrap),
        "spearman_delta_emb_vs_delta_rr": {
            "rho": float(rho), "p": float(p), "n": int(len(de_arr)),
            "by_category": per_cat_rho,
        },
        "directional_agreement": obs,
        "directional_agreement_independence_baseline": baseline,
        "directional_agreement_excess_over_independence": obs - baseline,
        "interpretation": (
            "A6: observed stage agreement is indistinguishable from what the two stages' "
            "own marginal biases already imply (baseline computed per stereotype category). "
            "It carries NO evidence of coupling and must not be described as 'above the 50% "
            "one would see by chance'. Use spearman_delta_emb_vs_delta_rr instead."
        ),
        # Back-compat keys for anything still reading the old schema.
        "embedder_pct_male": {c: stats_e[c].point()["pct_male"] / 100.0 for c in CATEGORIES},
        "reranker_pct_male": {c: stats_r[c].point()["pct_male"] / 100.0 for c in CATEGORIES},
    }


def main():
    ap = argparse.ArgumentParser(description="Two-stage pipeline / compounding experiment")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5,
                    help="random tie-break draws per slate (C1)")
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--strict-labels", action="store_true")
    ap.add_argument("--main-only", action="store_true", help="skip the 14-model sweep (B5)")
    args = ap.parse_args()

    labels, label_source = load_labels(strict=args.strict_labels)
    print(f"Label source: {label_source} ({len(labels)} occupations)")

    triples = generate_dataset(perturbation_levels=["names_and_pronouns"], query_types=None)
    print(f"Regenerated {len(triples)} triples")
    emb = embedding_scores(triples, args.refresh_embeddings)

    raw = load_raw()
    _, dropped, _ = detect_duplicates(raw)
    assert dropped == ["semantic-ranker-default-003"], f"unexpected duplicates: {dropped}"
    for d in dropped:
        del raw[d]
    print(f"Re-rankers after A1 de-duplication: {len(raw)}")

    rr_maps = {
        name: {key(r["occupation"], r["query_type"], r["name_pair"], r["template_style"]):
               (r["score_male"], r["score_female"]) for r in meta["rows"]}
        for name, meta in raw.items()
    }

    meta_block = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "label_source": label_source,
        "seed": SEED, "bootstrap_B": args.bootstrap, "tiebreak_repeats": args.repeats,
        "tiebreak": "average ranks for rank gaps; seeded random tie-breaks for top-k "
                    "selection, averaged over `tiebreak_repeats` draws (fixes C1)",
        "excluded_models": dropped,
    }

    # ---------------- main worked example ----------------
    print(f"\nMain pipeline: {EMBEDDER_NAME} -> {MAIN_RERANKER}")
    main_res = run_model(MAIN_RERANKER, rr_maps[MAIN_RERANKER], triples, emb, labels,
                         args, want_item_level=True)
    main_res.update(meta_block)
    main_res["topk"] = HEADLINE_TOPK  # back-compat
    OUT_MAIN.write_text(json.dumps(main_res, indent=2))

    il = main_res["item_level"]
    print("\n" + "=" * 92)
    print("ITEM-LEVEL, three-way split [% male-favoured / tie / female-favoured], tie-aware %M")
    print("=" * 92)
    print(f"{'stage':<10}{'cat':<9}{'%M':>7}{'%tie':>7}{'%F':>7}{'tie-aware %M':>15}{'95% CI':>18}")
    for stage in ("embedder", "reranker"):
        for c in CATEGORIES:
            d = il[stage][c]
            ci = f"[{d['tie_aware_pct_male_ci'][0]:.1f}, {d['tie_aware_pct_male_ci'][1]:.1f}]"
            print(f"{stage:<10}{c:<9}{d['pct_male']:>7.1f}{d['pct_tie']:>7.1f}{d['pct_female']:>7.1f}"
                  f"{d['tie_aware_pct_male']:>15.1f}{ci:>18}")
    sp = il["spearman_delta_emb_vs_delta_rr"]
    print(f"\nA6  Spearman rho(delta_emb, delta_rr) = {sp['rho']:.3f} (p={sp['p']:.3g}, n={sp['n']:,})")
    for c, v in sp["by_category"].items():
        print(f"      {c:<8} rho={v['spearman_rho']:.3f} (p={v['p']:.3g}, n={v['n']:,})")
    print(f"A6  directional agreement observed {100*il['directional_agreement']:.1f}% vs "
          f"independence baseline {100*il['directional_agreement_independence_baseline']:.1f}% "
          f"(excess {100*il['directional_agreement_excess_over_independence']:+.1f} pp) "
          f"-- NOT evidence of coupling.")

    print("\n" + "=" * 100)
    print(f"SLATE SIMULATION (20 candidates, retrieve top-{HEADLINE_TOPK}, shortlist "
          f"{HEADLINE_SHORTLIST})  [{MAIN_RERANKER}]")
    print("rank gap = mean_rank(F) - mean_rank(M);  > 0 => male candidates rank higher")
    print("=" * 100)
    hdr = (f"{'cat':<9}{'emb gap':>9}{'rr gap':>9}{'pipe gap':>10}"
           f"{'emb %M top3':>13}{'rr %M top3':>12}{'PIPE %M top3':>14}{'pipe CI':>18}"
           f"{'legacy %M':>11}")
    print(hdr)
    S = HEADLINE_SHORTLIST
    P = lambda d, k: d[f"{k}_top{S}_m_pct_male"]  # noqa: E731
    for c in list(CATEGORIES) + ["ALL"]:
        d = main_res["slate"][c]
        lo, hi = d[f"pipe_top{S}_m_pct_male_ci"]
        print(f"{c:<9}{d['emb_gap']:>9.2f}{d['rr_gap']:>9.2f}{d['pipe_gap']:>10.2f}"
              f"{P(d, 'emb'):>13.1f}{P(d, 'rr'):>12.1f}{P(d, 'pipe'):>14.1f}"
              f"  [{lo:.1f}, {hi:.1f}]{P(d, 'legacy_pipe'):>11.1f}")
    print(f"Saved -> {OUT_MAIN}")

    if args.main_only:
        return

    # ---------------- B5: all 14 re-rankers ----------------
    print("\n" + "=" * 100)
    print("B5  PIPELINE SWEEP over all re-rankers (zero API cost; embedder scores reused)")
    print("=" * 100)
    sweep = {}
    for name in sorted(rr_maps):
        sweep[name] = run_model(name, rr_maps[name], triples, emb, labels, args)
        print(f"  done {name}")

    payload = dict(meta_block)
    payload.update({"embedder": EMBEDDER_NAME, "topk_grid": list(TOPK_GRID),
                    "shortlist_grid": list(SHORTLIST_GRID), "models": sweep})
    OUT_SWEEP.write_text(json.dumps(payload, indent=2))

    order = sorted(sweep, key=lambda m: -P(sweep[m]["slate"]["ALL"], "pipe"))
    print(f"\n{'re-ranker':<50}{'PIPE %M top3':>14}{'95% CI':>18}{'rr-alone':>10}"
          f"{'emb-alone':>11}{'pipe gap':>10}{'C1 pipe':>9}{'C1 rr':>8}{'d rr':>7}")
    for m in order:
        d = sweep[m]["slate"]["ALL"]
        lo, hi = d[f"pipe_top{S}_m_pct_male_ci"]
        print(f"{m[:49]:<50}{P(d, 'pipe'):>14.1f}  [{lo:.1f}, {hi:.1f}]"
              f"{P(d, 'rr'):>10.1f}{P(d, 'emb'):>11.1f}"
              f"{d['pipe_gap']:>10.2f}{P(d, 'legacy_pipe'):>9.1f}"
              f"{P(d, 'legacy_rr'):>8.1f}{P(d, 'legacy_rr') - P(d, 'rr'):>+7.1f}")

    print(f"\n{'re-ranker':<50}" + "".join(f"{c:>12}" for c in CATEGORIES)
          + "   (PIPE % male in top-3, by occupation category)")
    for m in order:
        cells = "".join(f"{P(sweep[m]['slate'][c], 'pipe'):>12.1f}" for c in CATEGORIES)
        print(f"{m[:49]:<50}{cells}")

    print("\nSENSITIVITY (PIPE % male, ALL categories): cols = retrieval cutoff K x shortlist S")
    print(f"{'re-ranker':<40}" + "".join(f"{'K=' + str(K) + ',S=' + str(s):>12}"
                                         for K in TOPK_GRID for s in SHORTLIST_GRID))
    for m in order:
        cells = "".join(
            f"{sweep[m]['sensitivity'][f'topk{K}']['ALL'][f'pipe_top{s}_m_pct_male']:>12.1f}"
            for K in TOPK_GRID for s in SHORTLIST_GRID)
        print(f"{m[:39]:<40}{cells}")

    # ---- tex fragment ----
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&")  # noqa: E731
    lines = ["% Generated by code/compounding_experiment.py -- do not edit by hand.",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Re-ranker & Vendor & \multicolumn{4}{c}{Pipeline \% male in top-3} & "
             r"Re-ranker & Mean rank gap \\",
             r" & & all & male & female & neutral & alone & (F$-$M) \\", r"\midrule"]
    for m in order:
        d = sweep[m]["slate"]
        lines.append(
            f"{esc(m)} & {esc(sweep[m]['vendor'])} & "
            + " & ".join(f"{P(d[c], 'pipe'):.1f}" for c in ["ALL"] + list(CATEGORIES))
            + f" & {P(d['ALL'], 'rr'):.1f} & {d['ALL']['pipe_gap']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX_DIR / "compounding_sweep.tex").write_text("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_SWEEP}")
    print(f"Saved -> {TEX_DIR / 'compounding_sweep.tex'}")


if __name__ == "__main__":
    main()
