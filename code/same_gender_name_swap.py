"""
Does a gender swap move a reranker's score more than a same-gender name swap?

The audit reports, per model, which version of a pair scores higher and how
often. Two reviewer objections follow. (a) It never says by how much a score
moves. (b) It has no placebo: swapping John for James might move the score as
much as swapping John for Maria, in which case the "gender" effect is a name
effect. Both are answerable from the banked scores without a new model call:
each (query phrasing, occupation, template) cell holds ten male-associated
documents (ten given names, masculine pronouns) and ten female-associated ones,
all scored in the audit, and the audit compared each male-associated document
only with its own pair.

Per model, on the 52 BLS-labelled occupations (semantic-ranker-default-003 is
the byte-identical duplicate analyze_single_stage.py drops):

  magnitudes     mean and median |score_male - score_female| over the 8,320
                 fixed cross-gender pairs, and over every same-gender pair
                 within a cell (45 male-male and 45 female-female per cell)
  ratio          cross-gender mean |gap| / pooled same-gender mean |gap|, with a
                 percentile interval resampling the 52 occupations (B = 1,000)
  exceedance     share of cross-gender pairs whose |gap| exceeds the median
                 same-gender |gap| of their OWN cell (if the gender swap were
                 just another name swap this would sit near 50%), and the share
                 above the model-level same-gender median
  placebo        each of the 90 ordered same-gender pairings (John->James, ...,
                 Maria->Sarah, ...) run through the audit's own statistic: the
                 tie-aware "first name favoured" share on male-typed minus
                 female-typed occupations (the contrast) and the overall share
                 (the level). Their spread over the 90 pairings is what a name
                 swap alone produces, set against the actual male-female
                 contrast and level. A matched placebo repeats the actual
                 design's arithmetic with same-gender pairs: ten disjoint pairs
                 per cell (five male-male, five female-female, random
                 orientation), averaged, 10,000 draws per model
  pronoun-only   mean |gap| when only the pronouns differ (names replaced by one
                 placeholder), from the pronoun-channel run's per-row scores,
                 so three magnitudes sit side by side: name+pronoun swap,
                 pronoun-only swap, same-gender name swap; the overview repeats
                 the comparison without the models whose re-run did not
                 reproduce every banked winner (the pronoun-channel summary's sets)
  layout         whether two same-gender documents of one cell were scored in
                 the same request (or, for local pointwise models, the same
                 inference batch), from the request layout each route used

Output: results/same_gender_name_swap.json, results/tex/same_gender_name_swap.tex
Run:    venv/bin/python code/same_gender_name_swap.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from analyze_single_stage import (  # noqa: E402
    DEPRECATED_ALIASES, PROVENANCE, RESULTS_DIR, SEED, TEX_DIR, detect_duplicates, load_raw,
    stable_seed)
from labels import filter_rows, load_labels  # noqa: E402
from model_labels import model_label  # noqa: E402
from score_open_models import MODELS as GPU_MODELS  # noqa: E402  (the GPU runs' batch sizes)
from synthetic_dataset import NAME_PAIRS, TEMPLATES  # noqa: E402

SCRIPT_VERSION = "same_gender_name_swap.py v1.0"
OUT = RESULTS_DIR / "same_gender_name_swap.json"
TEX = TEX_DIR / "same_gender_name_swap.tex"
ORDER_TEX = TEX_DIR / "contrast_intervals.tex"
SUMMARY = RESULTS_DIR / "single_stage_summary.json"
PRONOUN_DIR = RESULTS_DIR / "pronoun_channel"
PRONOUN_SUMMARY = RESULTS_DIR / "pronoun_channel_summary.json"
DEID_PAIRS = RESULTS_DIR / "deid_pairs"
RUN_RECORDS = RESULTS_DIR / "run_records.json"
B_BOOT = 1000
LISTWISE = {m for m, (vendor, _) in PROVENANCE.items() if "listwise" in vendor}

# ---------------------------------------------------------------------------
# Request layout. Every route sent one query's 40 pairs as 80 documents, male
# version first in each pair, templates in the generator's order with the ten
# name pairs inside each template (run_experiment.py score_triples_*,
# score_open_models.py layout()); the raw files keep that order, asserted
# below. What differs is the unit a score could depend on besides its own
# (query, document): an API request, the listwise call, or -- for pointwise
# local models -- only the padding of an inference batch. `chunk` is the
# number of documents per unit (None: the whole query); `scope` "flat" is the
# February local runs, one interleaved list over ALL triples cut into batches.
# ---------------------------------------------------------------------------
API = "API request: vendor-side, whether scores depend on the other documents is not observable from the client"
POINTWISE = "inference batch: pointwise scoring, batch membership sets only the padding length"
ROUTES = {
    # run_experiment.py VOYAGE_MODELS: vo.rerank(query, documents, top_k=all)
    "Voyage API": dict(models={"rerank-1", "rerank-2", "rerank-2-lite", "rerank-2.5",
                               "rerank-2.5-lite", "rerank-3", "rerank-3-lite"},
                       unit=API, scope="query", chunk=None, records_route="vendor API"),
    # COHERE_MODELS: Cohere v2 rerank endpoint on Azure AI, top_n=all
    "Cohere on Azure AI": dict(models={"Cohere-rerank-v4.0-pro", "Cohere-rerank-v4.0-fast"},
                               unit=API, scope="query", chunk=None, records_route="Azure"),
    # OPENROUTER_MODELS: one request per query, top_n=all
    "OpenRouter": dict(models={"cohere/rerank-v3.5", "qwen/qwen3-reranker-8b",
                               "nvidia/llama-nemotron-rerank-vl-1b-v2"},
                       unit=API, scope="query", chunk=None, records_route="OpenRouter"),
    # GCP_MODELS: records chunked at 200, so 80 documents go in one request
    "Google ranking API": dict(models={"semantic-ranker-default-002", "semantic-ranker-default-004",
                                       "semantic-ranker-fast-004"},
                               unit=API, scope="query", chunk=200, records_route="vendor API"),
    # AZURE_MODELS: the endpoint takes 32 texts per request -> three requests per query
    "Azure ML": dict(models={"ibm-granite-granite-embedding-reranker-english-r2"},
                     unit=API, scope="query", chunk=32, records_route="Azure"),
    # local CrossEncoder: score_triples(), one flat interleaved list, BATCH_SIZE = 64
    "local Mac (Feb 2026)": dict(models={"BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L-12-v2",
                                         "jinaai/jina-reranker-v2-base-multilingual"},
                                 unit=POINTWISE, scope="flat", chunk=64, records_route="local Mac"),
}


def layout_table(models: set) -> dict:
    """model -> layout, cross-checked against results/run_records.json."""
    out = {}
    for route, spec in ROUTES.items():
        for m in spec["models"]:
            out[m] = {"route": route, "unit": spec["unit"], "scope": spec["scope"],
                      "chunk": spec["chunk"], "records_route": spec["records_route"]}
    # October GPU runs: each query's 80 documents, scored in batches of the
    # model's `batch`; the listwise model takes all 80 in one call.
    for m, spec in GPU_MODELS.items():
        if spec.get("selfhost") or m in out:
            continue
        if spec["backend"] == "jina_listwise":
            out[m] = {"route": "RunPod GPU", "scope": "query", "chunk": None, "records_route": "RunPod GPU",
                      "unit": "listwise call: the model scores a request's documents jointly"}
        else:
            out[m] = {"route": "RunPod GPU", "unit": POINTWISE, "scope": "query",
                      "chunk": spec["batch"], "records_route": "RunPod GPU"}
    missing = sorted(models - set(out))
    if missing:
        raise KeyError(f"no request layout known for {missing}; add the route from run_experiment.py")
    records = {r["model"]: r for r in json.loads(RUN_RECORDS.read_text())["records"]
               if r["experiment"] == "audit"}
    for m in models:
        rec = records.get(m)
        if rec is None or rec["route"] != out[m]["records_route"]:
            raise AssertionError(f"{m}: layout route {out[m]['records_route']!r} does not match "
                                 f"results/run_records.json ({rec and rec['route']!r})")
        out[m]["batch_size_record"] = rec["batch_size"]
    return out


def assert_generator_order(rows, names):
    """Rows are contiguous per query, template-major, ten name pairs in NAME_PAIRS order."""
    templates = list(TEMPLATES)
    seen, q, j = set(), None, 0
    for r in rows:
        if r["query"] != q:
            if r["query"] in seen:
                raise ValueError(f"query {r['query']!r} appears in two separate blocks")
            seen.add(r["query"])
            q, j = r["query"], 0
        if r["template_style"] != templates[(j // len(names)) % len(templates)] \
                or r["name_pair"] != names[j % len(names)]:
            raise ValueError(f"row order differs from the generator's at query {q!r}, position {j}")
        j += 1
    if j != len(names) * len(templates):
        raise ValueError("last query block is incomplete")


def tag_units(rows, lay):
    """Attach to each row the unit of its male and of its female document (small ints)."""
    ids = {}
    q, j = None, 0
    for g, r in enumerate(rows):
        if r["query"] != q:
            q, j = r["query"], 0
        if lay["scope"] == "flat":
            um, uf = (2 * g) // lay["chunk"], (2 * g + 1) // lay["chunk"]
        elif lay["chunk"] is None:
            um = uf = q
        else:
            um, uf = (q, (2 * j) // lay["chunk"]), (q, (2 * j + 1) // lay["chunk"])
        r["_unit_m"] = ids.setdefault(um, len(ids))
        r["_unit_f"] = ids.setdefault(uf, len(ids))
        j += 1


# ---------------------------------------------------------------------------
# Per model
# ---------------------------------------------------------------------------

def cell_arrays(rows, names):
    """Per cell the ten male and ten female scores (and units), names in NAME_PAIRS order."""
    cells = OrderedDict()
    for r in rows:
        cell = cells.setdefault((r["query_type"], r["occupation"], r["template_style"]), {})
        if r["name_pair"] in cell:
            raise ValueError(f"two rows for one name pair in cell {r['occupation']!r}, "
                             f"{r['template_style']!r}, {r['query_type']!r}")
        cell[r["name_pair"]] = r
    keys = sorted(cells)
    M, F = np.empty((len(keys), len(names))), np.empty((len(keys), len(names)))
    UM, UF = np.empty_like(M, dtype=np.int64), np.empty_like(M, dtype=np.int64)
    for c, k in enumerate(keys):
        if sorted(cells[k]) != sorted(names):
            raise ValueError(f"cell {k} does not hold exactly one pair per name pair")
        for i, nm in enumerate(names):
            r = cells[k][nm]
            M[c, i], F[c, i] = r["score_male"], r["score_female"]
            UM[c, i], UF[c, i] = r["_unit_m"], r["_unit_f"]
    return keys, M, F, UM, UF


def gap_stats(g):
    return {"n_pairs": int(g.size), "mean_abs_gap": float(g.mean()),
            "median_abs_gap": float(np.median(g)), "pct_zero_gap": float(100 * (g == 0).mean())}


def half_wins(X):
    """(cells, i, j): 1 if document i outscores j in the cell, 1/2 on a tie, else 0."""
    return (X[:, :, None] > X[:, None, :]) + 0.5 * (X[:, :, None] == X[:, None, :])


def dist(v):
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def analyse(model, rows, names, labels, B):
    keys, M, F, UM, UF = cell_arrays(rows, names)
    n_cells, n = M.shape
    iu, ju = np.triu_indices(n, k=1)             # the 45 unordered pairs, earlier name first
    cat = np.array([labels[k[1]] for k in keys])
    is_m, is_f = cat == "male", cat == "female"

    cross = np.abs(M - F)                        # (cells, 10): the audit's fixed pairs
    mm, ff = np.abs(M[:, iu] - M[:, ju]), np.abs(F[:, iu] - F[:, ju])
    same = np.concatenate([mm, ff], axis=1)      # (cells, 90)
    cell_median = np.median(same, axis=1)
    above_cell = cross > cell_median[:, None]
    at_cell = cross == cell_median[:, None]
    pooled_median = float(np.median(same))

    # Occupation-clustered bootstrap: every statistic is a ratio of sums over
    # cells, so per-occupation sums are resampled (analyze_single_stage.ClusterStat).
    occs = sorted({k[1] for k in keys})
    cell_occ = np.array([occs.index(k[1]) for k in keys])
    K = len(occs)
    agg = lambda per_cell: np.bincount(cell_occ, weights=per_cell, minlength=K)  # noqa: E731
    cross_sum, cross_n = agg(cross.sum(1)), agg(np.full(n_cells, cross.shape[1]))
    same_sum, same_n = agg(same.sum(1)), agg(np.full(n_cells, same.shape[1]))
    above_sum = agg(above_cell.sum(1))
    rng = np.random.default_rng(stable_seed(model, "same_gender_name_swap"))
    idx = rng.integers(0, K, size=(B, K))
    cross_b = cross_sum[idx].sum(1) / cross_n[idx].sum(1)
    same_b = same_sum[idx].sum(1) / same_n[idx].sum(1)
    share_b = 100 * above_sum[idx].sum(1) / cross_n[idx].sum(1)
    ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    same_mean = float(same.mean())
    ratio = float(cross.mean() / same_mean) if same_mean > 0 else None
    ratio_ci = ci(cross_b / same_b) if np.all(same_b > 0) else None

    # Placebo: the audit's statistic for every ordered same-gender pairing.
    W = (M > F) + 0.5 * (M == F)
    actual = {"pct_male_typed": float(100 * W[is_m].mean()),
              "pct_female_typed": float(100 * W[is_f].mean()),
              "overall_tie_aware_pct_male": float(100 * W.mean())}
    actual["contrast"] = actual["pct_male_typed"] - actual["pct_female_typed"]
    actual["abs_level_minus_50"] = abs(actual["overall_tie_aware_pct_male"] - 50)
    pairings, C, L = [], {}, {}
    for kind, H, col in (("male-male", half_wins(M), 0), ("female-female", half_wins(F), 1)):
        s_m, s_f, s_all = 100 * H[is_m].mean(0), 100 * H[is_f].mean(0), 100 * H.mean(0)
        C[kind], L[kind] = s_m - s_f, s_all          # full 10 x 10: C[j, i] = -C[i, j], L[j, i] = 100 - L[i, j]
        for i, j in zip(iu, ju):
            pairings.append({"kind": kind, "first": NAME_PAIRS[i][col], "second": NAME_PAIRS[j][col],
                             "pct_first_male_typed": float(s_m[i, j]),
                             "pct_first_female_typed": float(s_f[i, j]),
                             "contrast": float(s_m[i, j] - s_f[i, j]),
                             "pct_first_overall": float(s_all[i, j]),
                             "abs_level_minus_50": float(abs(s_all[i, j] - 50))})
    abs_c = np.array([abs(p["contrast"]) for p in pairings])
    abs_l = np.array([p["abs_level_minus_50"] for p in pairings])
    worst = lambda v: pairings[int(np.argmax(v))]  # noqa: E731
    name = lambda p: f"{p['first']}->{p['second']}"  # noqa: E731

    # Matched placebo design. The actual statistic averages ten pairs per cell
    # that use each of the twenty documents once; a single pairing is noisier
    # than that average, so its spread overstates what a name swap does. Each
    # draw pairs the ten male names into five disjoint pairs and the ten female
    # names into five, in random orientation, and averages the ten pairings'
    # contrasts and levels -- the same arithmetic as the actual design, with
    # same-gender pairs in place of male-female ones. Both statistics are linear
    # in the pairings, so the draws come from the matrices above.
    rng_p = np.random.default_rng(stable_seed(model, "matched_placebo"))
    N_DRAWS = 10_000
    pm = np.argsort(rng_p.random((N_DRAWS, n)), axis=1)
    pf = np.argsort(rng_p.random((N_DRAWS, n)), axis=1)
    pick = lambda A, p: A[p[:, 0::2], p[:, 1::2]].sum(1)  # noqa: E731  (the 5 pairs of a draw)
    c_draw = (pick(C["male-male"], pm) + pick(C["female-female"], pf)) / n
    l_draw = (pick(L["male-male"], pm) + pick(L["female-female"], pf)) / n
    abs_cd, abs_ld = np.abs(c_draw), np.abs(l_draw - 50)
    matched = {
        "n_draws": N_DRAWS,
        "pairs_per_draw": n,
        "abs_contrast": dist(abs_cd),
        "n_draws_abs_contrast_at_least_abs_actual": int((abs_cd >= abs(actual["contrast"])).sum()),
        "p_abs_contrast": float((1 + (abs_cd >= abs(actual["contrast"])).sum()) / (1 + N_DRAWS)),
        "abs_actual_contrast_above_max_draw": bool(abs(actual["contrast"]) > abs_cd.max()),
        "abs_level_minus_50": dist(abs_ld),
        "n_draws_abs_level_at_least_actual": int((abs_ld >= actual["abs_level_minus_50"]).sum()),
        "p_abs_level": float((1 + (abs_ld >= actual["abs_level_minus_50"]).sum()) / (1 + N_DRAWS)),
        "actual_abs_level_above_max_draw": bool(actual["abs_level_minus_50"] > abs_ld.max())}

    same_unit_same = np.concatenate([UM[:, iu] == UM[:, ju], UF[:, iu] == UF[:, ju]], axis=1)
    return {
        "model": model, "label": model_label(model), "listwise": model in LISTWISE,
        "n_rows": len(rows), "n_cells": n_cells, "n_occupations": K,
        "cross_gender": gap_stats(cross),
        "same_gender": {"male_male": gap_stats(mm), "female_female": gap_stats(ff),
                        "pooled": gap_stats(same)},
        "ratio": {"point": ratio, "ci": ratio_ci,
                  "above_1": bool(ratio is not None and ratio > 1),
                  "ci_above_1": bool(ratio_ci is not None and ratio_ci[0] > 1)},
        "exceedance": {"pct_cross_above_cell_median": float(100 * above_cell.mean()),
                       "pct_cross_above_cell_median_ci": ci(share_b),
                       "tie_aware_pct_cross_above_cell_median":
                           float(100 * (above_cell.mean() + 0.5 * at_cell.mean())),
                       "pooled_same_gender_median": pooled_median,
                       "pct_cross_above_pooled_median": float(100 * (cross > pooled_median).mean())},
        "placebo": {
            "actual": actual,
            "abs_contrast": {**dist(abs_c), "max_pairing": name(worst(abs_c))},
            "n_pairings_abs_contrast_at_least_abs_actual": int((abs_c >= abs(actual["contrast"])).sum()),
            "abs_actual_contrast_above_max_placebo": bool(abs(actual["contrast"]) > abs_c.max()),
            "abs_actual_contrast_over_max_placebo": float(abs(actual["contrast"]) / abs_c.max()),
            "abs_level_minus_50": {**dist(abs_l), "max_pairing": name(worst(abs_l))},
            "n_pairings_abs_level_at_least_actual": int((abs_l >= actual["abs_level_minus_50"]).sum()),
            "actual_abs_level_above_max_placebo": bool(actual["abs_level_minus_50"] > abs_l.max()),
            "actual_abs_level_over_max_placebo": float(actual["abs_level_minus_50"] / abs_l.max()),
            "matched_design": matched,
            "pairings": pairings},
        "_same_unit": {"same_gender": float(100 * same_unit_same.mean()),
                       "cross_gender": float(100 * (UM == UF).mean())},
    }


# ---------------------------------------------------------------------------
# Pronoun-only magnitudes (per-row scores stored by the pronoun-channel run)
# ---------------------------------------------------------------------------

def pronoun_only_sources():
    """audit model id -> (rows, source info). The pronoun-channel run stores per-row
    scores for the models it scored; for the four de-identification rerankers it read
    that run's `names_only` pairs, collapsed exactly as pronoun_channel_experiment.deid_rows."""
    out = {}
    for path in sorted(PRONOUN_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        model = d.get("self_hosted_copy_of") or d.get("model")
        if d.get("status") != "complete" or not d.get("pairs"):
            out[model] = (None, {"reason": f"{path.relative_to(ROOT)}: status {d.get('status')!r}, "
                                           "no per-row scores"})
            continue
        out[model] = (d["pairs"], {"source_file": str(path.relative_to(ROOT)),
                                   "scored_model": d["model"],
                                   "self_hosted_copy_of": d.get("self_hosted_copy_of"),
                                   "access": d.get("access"), "scored_from": d.get("scored_from"),
                                   "request_layout": (d.get("meta") or {}).get("request_layout")})
    for path in sorted(DEID_PAIRS.glob("*.json")):
        d = json.loads(path.read_text())
        model = d["reranker"]
        if model in out or d["score_pair_order"] != ["male", "female"]:
            continue
        comps = OrderedDict()
        for p in d["pairs"]:
            occ, qt, _pair, tmpl = p["key"].split("|")
            sm, sf = p["names_only"]
            if (occ, qt, tmpl) in comps:
                if (comps[occ, qt, tmpl]["score_male"], comps[occ, qt, tmpl]["score_female"]) != (sm, sf):
                    raise ValueError(f"{path.name}: name pairs disagree for {(occ, qt, tmpl)}")
                continue
            comps[occ, qt, tmpl] = {"occupation": occ, "query_type": qt, "template_style": tmpl,
                                    "score_male": sm, "score_female": sf}
        out[model] = (list(comps.values()),
                      {"source_file": str(path.relative_to(ROOT)), "scored_model": model,
                       "self_hosted_copy_of": None, "access": "local",
                       "scored_from": None, "request_layout": None,
                       "note": "de-identification run, `names_only` condition (not re-scored)"})
    return out


def pronoun_only(model, sources, pron_summary, labels):
    rows, info = sources.get(model, (None, {"reason": "no pronoun-only run stores per-row scores "
                                                      "for this model"}))
    if rows is None:
        return {"available": False, **info}
    if {r["occupation"] for r in rows} != set(labels):
        raise ValueError(f"{model}: pronoun-only rows do not cover the labelled occupations")
    gaps = np.abs(np.array([r["score_male"] - r["score_female"] for r in rows]))
    # that run's own check of its scores against the banked ones (hardware drift)
    sc = (pron_summary.get("models", {}).get(info["scored_model"]) or {}).get("scoring") or {}
    return {"available": True, **info, **gap_stats(gaps),
            "banked_score_check": sc.get("drift_check") or sc.get("deid_run_determinism_check")}


def pronoun_only_subset(pron, name, subset):
    """overview.pronoun_only's statistics without the models that one of the
    pronoun-channel overview's sets drops (`models_dropped`, keyed by the scored
    model): those whose re-run did not reproduce every banked winner. The
    pronoun-only gaps come from that re-run, the same-gender gaps from the banked
    scores, so the ratio is cleanest where the re-run reproduced the bank."""
    dropped = set(subset["models_dropped"])
    keep = [v for v in pron.values() if v["scored_model"] not in dropped]
    ratios = [v["pronoun_over_same_gender_mean"] for v in keep]
    return {"rule": subset["rule"],
            "source": f"{PRONOUN_SUMMARY.relative_to(ROOT)} overview.{name}",
            "models_dropped": sorted(m for m, v in pron.items() if v["scored_model"] in dropped),
            "n_models_with_per_row_scores": len(keep),
            "median_pronoun_over_same_gender_mean": float(np.median(ratios)) if ratios else None,
            "n_models_pronoun_mean_above_same_gender_mean": sum(r > 1 for r in ratios)}


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def sig4(x):
    """Four significant figures in fixed notation, so gaps on every score scale read alike."""
    if x == 0:
        return "0"
    d = max(0, 3 - int(math.floor(math.log10(abs(x)))))
    s = f"{x:.{d}f}"
    if d and abs(float(s)) >= 10 ** (4 - d):      # rounding carried into a fifth figure
        s = f"{x:.{d - 1}f}"
    return s


def num(x, nd=1):
    # A true minus sign, as in the other tables; a value rounding to zero prints as 0.0.
    t = f"{x:.{nd}f}"
    if float(t) == 0:
        t = f"{0:.{nd}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def table_order(models):
    """Row order of contrast_intervals.tex when it exists, else contrast descending."""
    by_label = {model_label(m): m for m in models}
    if ORDER_TEX.exists():
        order = []
        for line in ORDER_TEX.read_text().splitlines():
            if " & " not in line or not line.rstrip().endswith(r"\\") or line.lstrip().startswith("&"):
                continue
            label = line.split(" & ")[0].replace(r"$^{\dagger}$", "").strip()
            label = label.replace(r"\_", "_").replace(r"\&", "&").replace(r"\%", "%")
            if label in by_label:
                order.append(by_label[label])
        if sorted(order) == sorted(models):
            return order, str(ORDER_TEX.relative_to(ROOT))
        print(f"  note: {ORDER_TEX.name} does not list exactly these models; sorting by contrast")
    return (sorted(models, key=lambda m: -models[m]["placebo"]["actual"]["contrast"]),
            "actual contrast, descending")


def write_tex(models, order, B, n_cross, n_same):
    kinds = [p["kind"] for p in next(iter(models.values()))["placebo"]["pairings"]]
    lines = [f"% Generated by code/same_gender_name_swap.py (B = {B:,} occupation-clustered draws) "
             "-- do not edit by hand.",
             r"\begin{tabular}{lrrlrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Mean $|\Delta|$} & & Cross $>$ cell & "
             r"\multicolumn{2}{c}{Contrast} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){6-7}",
             r"Model & Cross-gender & Same-gender & Ratio [95\% CI] & median (\%) & Actual & "
             r"Max placebo \\",
             r"\midrule"]
    for m in order:
        e = models[m]
        dag = r"$^{\dagger}$" if e["listwise"] else ""
        r = e["ratio"]
        ratio = (f"{r['point']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
                 if r["point"] is not None and r["ci"] is not None else "--")
        lines.append(
            f"{esc(e['label'])}{dag} & {sig4(e['cross_gender']['mean_abs_gap'])} & "
            f"{sig4(e['same_gender']['pooled']['mean_abs_gap'])} & {ratio} & "
            f"{e['exceedance']['pct_cross_above_cell_median']:.1f} & "
            f"{num(e['placebo']['actual']['contrast'])} & "
            f"{num(e['placebo']['abs_contrast']['max'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip{\footnotesize\raggedright Mean $|\Delta|$ is the mean absolute score "
              r"gap on the model's own score scale: cross-gender over the audit's "
              f"{n_cross:,} fixed name-and-pronoun pairs, same-gender over every pair of the ten "
              r"male-associated and every pair of the ten female-associated documents of a "
              f"(query phrasing, occupation, template) cell ({n_same:,} pairs). "
              f"Max placebo is the largest $|$contrast$|$ over the {len(kinds)} same-gender name "
              f"pairs ({kinds.count('male-male')} male--male, {kinds.count('female-female')} "
              r"female--female), each taken in one fixed orientation; reversing a pair flips the "
              r"sign of its contrast but not its size. "
              r"$^{\dagger}$ Listwise reranker: it scores all 80 documents of a request jointly, so "
              r"each score depends on the other documents in the request; its gaps are comparable "
              r"within the row, not with the pointwise models' score scales.\par}"]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bootstrap", type=int, default=B_BOOT)
    ap.add_argument("--allow-legacy-labels", action="store_true",
                    help="fall back to the pre-BLS labels instead of failing")
    args = ap.parse_args()

    labels, label_source = load_labels(strict=not args.allow_legacy_labels)
    raw = load_raw()
    _groups, dropped, _ = detect_duplicates(raw)
    assert dropped == ["semantic-ranker-default-003"], (
        f"expected to drop exactly ['semantic-ranker-default-003'], got {dropped}")
    for d in dropped:
        del raw[d]
    expected = set(PROVENANCE) - set(DEPRECATED_ALIASES)
    assert set(raw) == expected, (f"raw files do not match the registered models: missing "
                                  f"{sorted(expected - set(raw))}, unexpected {sorted(set(raw) - expected)}")
    summary = json.loads(SUMMARY.read_text())
    layouts = layout_table(set(raw))
    pron_summary = json.loads(PRONOUN_SUMMARY.read_text()) if PRONOUN_SUMMARY.exists() else {}
    pron_sources = pronoun_only_sources()
    names = [f"{m}/{f}" for m, f in NAME_PAIRS]

    models = {}
    for model, meta in sorted(raw.items()):
        rows = meta["rows"]
        assert_generator_order(rows, names)
        tag_units(rows, layouts[model])          # on the full file: batch positions need it
        rows = filter_rows(rows, labels)
        e = analyse(model, rows, names, labels, args.bootstrap)
        e["raw_file"] = Path(meta["path"]).name

        # The recomputed audit statistics must be the published ones.
        ss = summary["models"][model]
        pub = {"contrast": summary["stereotype_tracking_spread"]["per_model"][model],
               "overall_tie_aware_pct_male": ss["overall"]["tie_aware_pct_male"],
               "pct_male_typed": ss["by_category"]["male"]["tie_aware_pct_male"],
               "pct_female_typed": ss["by_category"]["female"]["tie_aware_pct_male"]}
        for k, v in pub.items():
            if abs(e["placebo"]["actual"][k] - v) > 1e-9:
                raise AssertionError(f"{model}: recomputed {k} {e['placebo']['actual'][k]} != "
                                     f"single_stage_summary {v}")
        e["placebo"]["actual"]["matches_single_stage_summary"] = True

        e["pronoun_only"] = pronoun_only(model, pron_sources, pron_summary, labels)
        if e["pronoun_only"]["available"]:
            e["pronoun_only"]["pronoun_over_same_gender_mean"] = (
                e["pronoun_only"]["mean_abs_gap"] / e["same_gender"]["pooled"]["mean_abs_gap"])
            e["pronoun_only"]["pronoun_over_cross_gender_mean"] = (
                e["pronoun_only"]["mean_abs_gap"] / e["cross_gender"]["mean_abs_gap"])

        lay, su = layouts[model], e.pop("_same_unit")
        e["scoring_layout"] = {
            "route": lay["route"], "unit": lay["unit"],
            "documents_per_unit": (lay["chunk"] if lay["chunk"] and lay["chunk"] < 80
                                   else "all 80 of the query"),
            "request_cap": lay["chunk"],
            "batch_size_record": lay["batch_size_record"],
            "pct_same_gender_pairs_in_same_unit": su["same_gender"],
            "pct_cross_gender_pairs_in_same_unit": su["cross_gender"],
            "all_same_gender_pairs_in_same_unit": su["same_gender"] == 100.0}
        models[model] = e
        print(f"{model:<52} cross {e['cross_gender']['mean_abs_gap']:.4g}  same "
              f"{e['same_gender']['pooled']['mean_abs_gap']:.4g}  ratio {e['ratio']['point']:.2f} "
              f"[{e['ratio']['ci'][0]:.2f}, {e['ratio']['ci'][1]:.2f}]  >cell med "
              f"{e['exceedance']['pct_cross_above_cell_median']:.1f}%  contrast "
              f"{e['placebo']['actual']['contrast']:.1f} vs placebo max "
              f"{e['placebo']['abs_contrast']['max']:.1f}  same-unit {su['same_gender']:.1f}%",
              flush=True)

    # ---- overview --------------------------------------------------------
    ratios = {m: e["ratio"]["point"] for m, e in models.items()}
    level_ratio = {m: e["placebo"]["actual_abs_level_over_max_placebo"] for m, e in models.items()}
    contrast_ratio = {m: e["placebo"]["abs_actual_contrast_over_max_placebo"] for m, e in models.items()}
    pooled_c = np.array([abs(p["contrast"]) for e in models.values() for p in e["placebo"]["pairings"]])
    pooled_l = np.array([p["abs_level_minus_50"] for e in models.values() for p in e["placebo"]["pairings"]])
    pron = {m: e["pronoun_only"] for m, e in models.items() if e["pronoun_only"]["available"]}
    first_matched = next(iter(models.values()))["placebo"]["matched_design"]
    lo = min(ratios, key=ratios.get)
    hi = max(ratios, key=ratios.get)
    overview = {
        "n_models": len(models),
        "n_models_ratio_above_1": sum(e["ratio"]["above_1"] for e in models.values()),
        "n_models_ratio_ci_above_1": sum(e["ratio"]["ci_above_1"] for e in models.values()),
        "models_ratio_ci_not_above_1": sorted(m for m, e in models.items() if not e["ratio"]["ci_above_1"]),
        "median_ratio": float(np.median(list(ratios.values()))),
        "min_ratio": {"model": lo, "ratio": ratios[lo], "ci": models[lo]["ratio"]["ci"]},
        "max_ratio": {"model": hi, "ratio": ratios[hi], "ci": models[hi]["ratio"]["ci"]},
        "median_pct_cross_above_cell_median": float(np.median(
            [e["exceedance"]["pct_cross_above_cell_median"] for e in models.values()])),
        "min_pct_cross_above_cell_median": min(
            ({"model": m, "pct": e["exceedance"]["pct_cross_above_cell_median"]} for m, e in models.items()),
            key=lambda d: d["pct"]),
        "n_models_abs_actual_contrast_above_max_placebo": sum(
            e["placebo"]["abs_actual_contrast_above_max_placebo"] for e in models.values()),
        "models_abs_actual_contrast_within_placebo_range": [
            {"model": m, "actual_contrast": e["placebo"]["actual"]["contrast"],
             "max_placebo_abs_contrast": e["placebo"]["abs_contrast"]["max"],
             "max_placebo_pairing": e["placebo"]["abs_contrast"]["max_pairing"]}
            for m, e in sorted(models.items()) if not e["placebo"]["abs_actual_contrast_above_max_placebo"]],
        "median_abs_actual_contrast_over_max_placebo": float(np.median(list(contrast_ratio.values()))),
        "n_models_actual_abs_level_above_max_placebo": sum(
            e["placebo"]["actual_abs_level_above_max_placebo"] for e in models.values()),
        "median_actual_abs_level_over_max_placebo": float(np.median(list(level_ratio.values()))),
        "placebo_pooled_over_models": {"n_pairings": int(pooled_c.size),
                                       "abs_contrast": dist(pooled_c), "abs_level_minus_50": dist(pooled_l)},
        "matched_design": {
            "n_draws_per_model": first_matched["n_draws"],
            "n_models_abs_actual_contrast_above_max_draw": sum(
                e["placebo"]["matched_design"]["abs_actual_contrast_above_max_draw"] for e in models.values()),
            "n_models_p_abs_contrast_below_0_05": sum(
                e["placebo"]["matched_design"]["p_abs_contrast"] < 0.05 for e in models.values()),
            "models_p_abs_contrast_at_least_0_05": {
                m: e["placebo"]["matched_design"]["p_abs_contrast"] for m, e in sorted(models.items())
                if e["placebo"]["matched_design"]["p_abs_contrast"] >= 0.05},
            "median_matched_abs_contrast_p95": float(np.median(
                [e["placebo"]["matched_design"]["abs_contrast"]["p95"] for e in models.values()])),
            "n_models_actual_abs_level_above_max_draw": sum(
                e["placebo"]["matched_design"]["actual_abs_level_above_max_draw"] for e in models.values()),
            "n_models_p_abs_level_below_0_05": sum(
                e["placebo"]["matched_design"]["p_abs_level"] < 0.05 for e in models.values()),
            "models_p_abs_level_at_least_0_05": {
                m: e["placebo"]["matched_design"]["p_abs_level"] for m, e in sorted(models.items())
                if e["placebo"]["matched_design"]["p_abs_level"] >= 0.05}},
        "pronoun_only": {
            "n_models_with_per_row_scores": len(pron),
            "models_without": sorted(set(models) - set(pron)),
            "median_pronoun_over_same_gender_mean": float(np.median(
                [v["pronoun_over_same_gender_mean"] for v in pron.values()])) if pron else None,
            "n_models_pronoun_mean_above_same_gender_mean": sum(
                v["pronoun_over_same_gender_mean"] > 1 for v in pron.values()),
            "median_pronoun_over_cross_gender_mean": float(np.median(
                [v["pronoun_over_cross_gender_mean"] for v in pron.values()])) if pron else None,
            **{k: pronoun_only_subset(pron, k, pron_summary["overview"][k])
               for k in ("excluding_drift_below_100",
                         "excluding_drift_below_100_and_deid_rerun_below_100")
               if k in pron_summary.get("overview", {})}},
        "layout": {
            "n_models_all_same_gender_pairs_in_one_unit": sum(
                e["scoring_layout"]["all_same_gender_pairs_in_same_unit"] for e in models.values()),
            "models_split_across_units": {
                m: {"route": e["scoring_layout"]["route"], "unit": e["scoring_layout"]["unit"],
                    "pct_same_gender_pairs_in_same_unit": e["scoring_layout"]["pct_same_gender_pairs_in_same_unit"]}
                for m, e in sorted(models.items()) if not e["scoring_layout"]["all_same_gender_pairs_in_same_unit"]}},
    }

    first = next(iter(models.values()))
    n_cross, n_same = first["cross_gender"]["n_pairs"], first["same_gender"]["pooled"]["n_pairs"]
    order, order_source = table_order(models)
    out = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "label_source": label_source,
        "seed": SEED,
        "seed_rule": "np.random.default_rng(analyze_single_stage.stable_seed(model, 'same_gender_name_swap'))",
        "bootstrap_B": args.bootstrap,
        "bootstrap_method": "percentile; the 52 occupations resampled with replacement, the "
                            "cross-gender and same-gender means (and the exceedance share) "
                            "recomputed from the resampled occupations",
        "definition": {
            "cell": "one (query phrasing, occupation, template): ten male-associated documents "
                    "(given name + masculine pronouns, score_male) and ten female-associated "
                    "ones (score_female), in NAME_PAIRS order",
            "cross_gender": "|score_male - score_female| over the audit's fixed pairs (John-Maria, ...)",
            "same_gender": "|score_i - score_j| over the 45 male-male and 45 female-female "
                           "unordered pairs within a cell",
            "ratio": "cross-gender mean |gap| / pooled same-gender mean |gap|",
            "exceedance": "share of cross-gender pairs whose |gap| exceeds (strictly) the median "
                          "same-gender |gap| of their own cell; secondary: above the model-level "
                          "pooled same-gender median",
            "placebo": "for each ordered same-gender pairing i->j (earlier name in NAME_PAIRS "
                       "first): tie-aware % 'first name favoured' on male-typed minus female-typed "
                       "occupations (contrast, points) and over all labelled occupations (level); "
                       "'actual' is the audit's male-female statistic recomputed from the same cells "
                       "and checked against single_stage_summary.json",
            "placebo_matched_design": "10,000 seeded draws per model; each pairs the ten male names "
                                      "into five disjoint pairs and the ten female names into five, "
                                      "in random orientation, and averages the ten pairings' contrast "
                                      "and level as the actual design averages its ten male-female "
                                      "pairs; p = (1 + #{|draw| >= |actual|}) / (1 + draws)",
            "pronoun_only": "mean |gap| when the two documents differ only in pronouns (names "
                            "replaced by one placeholder), per-row scores of the pronoun-channel "
                            "run; scored in a separate run, see banked_score_check for how far "
                            "that run's scores drift from the banked ones",
            "scoring_layout": "the unit each document was scored in: API request, listwise call, "
                              "or (pointwise local models) inference batch; "
                              "pct_same_gender_pairs_in_same_unit is the share of the same-gender "
                              "pairs whose two documents were in one unit",
            "gaps_are_on_each_model's_own_score_scale": True},
        "n_models": len(models), "excluded_models": dropped,
        "n_occupations": len(labels), "n_cells_per_model": first["n_cells"],
        "n_cross_gender_pairs_per_model": n_cross, "n_same_gender_pairs_per_model": n_same,
        "n_placebo_pairings": len(first["placebo"]["pairings"]),
        "name_pairs": [list(p) for p in NAME_PAIRS],
        "table_order": order_source,
        "overview": overview,
        "models": models,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    write_tex(models, order, args.bootstrap, n_cross, n_same)

    o = overview
    print(f"\nratio > 1: {o['n_models_ratio_above_1']}/{o['n_models']}; interval above 1: "
          f"{o['n_models_ratio_ci_above_1']}; median ratio {o['median_ratio']:.2f}; "
          f"min {o['min_ratio']['ratio']:.2f} ({o['min_ratio']['model']})")
    print(f"|actual contrast| > max placebo |contrast|: "
          f"{o['n_models_abs_actual_contrast_above_max_placebo']}/{o['n_models']}; "
          f"median |actual level-50| / max placebo: {o['median_actual_abs_level_over_max_placebo']:.2f}")
    md = o["matched_design"]
    print(f"matched design ({md['n_draws_per_model']:,} draws): |actual contrast| above every draw "
          f"{md['n_models_abs_actual_contrast_above_max_draw']}/{o['n_models']}, p < 0.05 "
          f"{md['n_models_p_abs_contrast_below_0_05']}; level p < 0.05 "
          f"{md['n_models_p_abs_level_below_0_05']}; not: {md['models_p_abs_contrast_at_least_0_05']}")
    print(f"pronoun-only per-row scores for {o['pronoun_only']['n_models_with_per_row_scores']} models; "
          f"median pronoun-only / same-gender mean |gap| "
          f"{o['pronoun_only']['median_pronoun_over_same_gender_mean']:.2f}")
    for k, v in o["pronoun_only"].items():
        if isinstance(v, dict):
            print(f"  {k}: {v['n_models_with_per_row_scores']} models, median ratio "
                  f"{v['median_pronoun_over_same_gender_mean']:.2f}, above 1 for "
                  f"{v['n_models_pronoun_mean_above_same_gender_mean']}")
    print(f"all same-gender pairs in one request/call/batch: "
          f"{o['layout']['n_models_all_same_gender_pairs_in_one_unit']}/{o['n_models']} models; "
          f"split: {list(o['layout']['models_split_across_units'])}")
    print(f"Saved -> {OUT}\nSaved -> {TEX}")


if __name__ == "__main__":
    main()
