"""
Run gender bias experiment on cross-encoder re-rankers.

Scores synthetic counterfactual pairs, computes bias metrics, runs statistical tests,
and writes RESULTS.md.

The triple count is NOT fixed: it is
    occupations x query types x templates x (name pairs per perturbation level)
and the occupation list changed when the BLS relabelling landed (82 -> 52 kept;
see code/occupations_bls.csv and REVIEW_TODO.md A8).  Run
`venv/bin/python code/synthetic_dataset.py --quick` to print the current counts.
The banked `results/*_full_raw.json` files were scored in Feb 2026 with
perturbation=names_and_pronouns and all four query types, over the pre-BLS
82-occupation list -> 13,120 pairs per model.  Those files are a historical
record and are never regenerated.

Caveats for anything that reads the metrics below:
  * The Wilcoxon / t / OLS / Kruskal-Wallis tests treat the pairs as independent.
    They are not (they nest in occupations x names x templates x queries), so those
    p-values are pseudo-replicated and must NOT be quoted in the paper.  Use the
    cluster bootstrap in code/analyze_single_stage.py instead (REVIEW_TODO B1).
  * `pct_male_favored` counts only delta > 0.  Ties are a large share of several
    APIs' scores, so read `pct_tie` / `tie_aware_pct_male_favored` alongside it
    (REVIEW_TODO A2/C6).
  * `rbs` and `_bias_label` are no longer used by the paper.

Usage:
    python code/run_experiment.py                    # Full run (all perturbations)
    python code/run_experiment.py --quick            # Quick run (names_and_pronouns, bare)
    python code/run_experiment.py --model MODEL      # Different model
    python code/run_experiment.py --perturbation X   # Specific perturbation level
"""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from scipy.stats import wilcoxon, kruskal
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# Patch for Jina reranker: function removed in transformers 5.x but needed by custom model code
def _create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

import transformers.models.xlm_roberta.modeling_xlm_roberta as _xlm_mod
if not hasattr(_xlm_mod, "create_position_ids_from_input_ids"):
    _xlm_mod.create_position_ids_from_input_ids = _create_position_ids_from_input_ids

from synthetic_dataset import (
    ALL_OCCUPATIONS,
    NAME_PAIRS,
    QUERY_TYPES,
    TEMPLATES,
    Triple,
    generate_dataset,
)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
BATCH_SIZE = 64

# Known API-based models (not local CrossEncoders)
VOYAGE_MODELS = {"rerank-1", "rerank-2", "rerank-2-lite", "rerank-2.5", "rerank-2.5-lite"}
AZURE_MODELS = {"ibm-granite-granite-embedding-reranker-english-r2"}
COHERE_MODELS = {"Cohere-rerank-v4.0-pro", "Cohere-rerank-v4.0-fast"}
GCP_MODELS = {"semantic-ranker-default-004", "semantic-ranker-default-003", "semantic-ranker-fast-004", "semantic-ranker-default-002"}


# ---------------------------------------------------------------------------
# Scoring — local CrossEncoder
# ---------------------------------------------------------------------------

def score_triples(model: CrossEncoder, triples: list[Triple], batch_size: int = BATCH_SIZE) -> list[dict]:
    """Score all triples, interleaving male/female to guard against ordering effects."""
    # Build interleaved batch: [male_0, female_0, male_1, female_1, ...]
    pairs = []
    for t in triples:
        pairs.append((t.query, t.doc_male))
        pairs.append((t.query, t.doc_female))

    # Score in batches with progress bar
    all_scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Scoring", unit="batch"):
        batch = pairs[i : i + batch_size]
        scores = model.predict(batch)
        all_scores.extend(scores)

    # Unpack interleaved scores back to results
    results = []
    for idx, t in enumerate(triples):
        sm = float(all_scores[idx * 2])
        sf = float(all_scores[idx * 2 + 1])
        delta = sm - sf
        results.append({
            "query": t.query,
            "query_type": t.query_type,
            "occupation": t.occupation,
            "stereotype": t.stereotype,
            "name_pair": t.name_pair,
            "template_style": t.template_style,
            "perturbation": t.perturbation,
            "score_male": sm,
            "score_female": sf,
            "delta": delta,
        })

    return results


# ---------------------------------------------------------------------------
# Scoring — Voyage AI API
# ---------------------------------------------------------------------------

def score_triples_voyage(
    triples: list[Triple], model_name: str, api_key: str,
    rpm_limit: int = 3,
) -> list[dict]:
    """Score triples via Voyage AI rerank API, grouped by query for efficiency.

    Includes rate limiting (sleep between requests) and retry on RateLimitError.
    """
    import voyageai
    import voyageai.error

    vo = voyageai.Client(api_key=api_key)

    # Group triples by query — each unique query gets one API call
    from collections import OrderedDict
    query_groups: OrderedDict[str, list[tuple[int, Triple]]] = OrderedDict()
    for i, t in enumerate(triples):
        query_groups.setdefault(t.query, []).append((i, t))

    results: list[dict | None] = [None] * len(triples)
    min_interval = 60.0 / rpm_limit  # seconds between requests
    last_request_time = 0.0

    total_queries = len(query_groups)
    est_minutes = total_queries * min_interval / 60
    print(f"  Rate limit: {rpm_limit} RPM → ~{est_minutes:.0f} min estimated")

    for query, group in tqdm(query_groups.items(), desc="Scoring (Voyage API)", unit="query"):
        # Build document list: [doc_male_0, doc_female_0, doc_male_1, doc_female_1, ...]
        documents = []
        positions = []  # (triple_index, male_doc_pos, female_doc_pos)
        for idx, t in group:
            male_pos = len(documents)
            documents.append(t.doc_male)
            female_pos = len(documents)
            documents.append(t.doc_female)
            positions.append((idx, male_pos, female_pos))

        # Rate limiting: wait if needed
        elapsed_since_last = time.time() - last_request_time
        if elapsed_since_last < min_interval:
            time.sleep(min_interval - elapsed_since_last)

        # Call with retry on rate limit
        max_retries = 5
        for attempt in range(max_retries):
            try:
                last_request_time = time.time()
                reranking = vo.rerank(query, documents, model=model_name, top_k=len(documents))
                break
            except voyageai.error.RateLimitError:
                wait = 30 * (attempt + 1)
                tqdm.write(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed after {max_retries} retries for query: {query!r}")

        # Build score lookup by original document index
        score_map = {r.index: r.relevance_score for r in reranking.results}

        # Map scores back to triples
        for idx, male_pos, female_pos in positions:
            sm = float(score_map[male_pos])
            sf = float(score_map[female_pos])
            t = triples[idx]
            results[idx] = {
                "query": t.query,
                "query_type": t.query_type,
                "occupation": t.occupation,
                "stereotype": t.stereotype,
                "name_pair": t.name_pair,
                "template_style": t.template_style,
                "perturbation": t.perturbation,
                "score_male": sm,
                "score_female": sf,
                "delta": sm - sf,
            }

    return results


# ---------------------------------------------------------------------------
# Scoring — Azure ML hosted models
# ---------------------------------------------------------------------------

def _azure_rerank_request(url, headers, query, texts, max_retries=5):
    """Single rerank API call with retry logic. Returns list of {index, score}."""
    import requests

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json={"query": query, "texts": texts}, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)
            tqdm.write(f"  HTTP error: {e}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries for query: {query!r}")


def score_triples_azure(
    triples: list[Triple], model_name: str,
    endpoint: str, api_key: str,
    max_batch: int = 32,
) -> list[dict]:
    """Score triples via Azure ML rerank endpoint, grouped by query.

    The endpoint enforces a max batch size of 32 texts, so large queries are
    chunked into multiple API calls and scores are merged back.
    """
    url = endpoint.rstrip("/") + "/rerank"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Group triples by query — each unique query gets one (or more) API calls
    from collections import OrderedDict
    query_groups: OrderedDict[str, list[tuple[int, Triple]]] = OrderedDict()
    for i, t in enumerate(triples):
        query_groups.setdefault(t.query, []).append((i, t))

    results: list[dict | None] = [None] * len(triples)

    total_queries = len(query_groups)
    print(f"  {total_queries} unique queries (max {max_batch} texts/request)")

    for query, group in tqdm(query_groups.items(), desc="Scoring (Azure API)", unit="query"):
        # Build document list: [doc_male_0, doc_female_0, doc_male_1, doc_female_1, ...]
        documents = []
        positions = []  # (triple_index, male_doc_pos, female_doc_pos)
        for idx, t in group:
            male_pos = len(documents)
            documents.append(t.doc_male)
            female_pos = len(documents)
            documents.append(t.doc_female)
            positions.append((idx, male_pos, female_pos))

        # Score in chunks if needed, merging into a single score_map
        score_map = {}
        for chunk_start in range(0, len(documents), max_batch):
            chunk = documents[chunk_start : chunk_start + max_batch]
            scores_list = _azure_rerank_request(url, headers, query, chunk)
            for item in scores_list:
                score_map[chunk_start + item["index"]] = item["score"]

        # Map scores back to triples
        for idx, male_pos, female_pos in positions:
            sm = float(score_map[male_pos])
            sf = float(score_map[female_pos])
            t = triples[idx]
            results[idx] = {
                "query": t.query,
                "query_type": t.query_type,
                "occupation": t.occupation,
                "stereotype": t.stereotype,
                "name_pair": t.name_pair,
                "template_style": t.template_style,
                "perturbation": t.perturbation,
                "score_male": sm,
                "score_female": sf,
                "delta": sm - sf,
            }

    return results


# ---------------------------------------------------------------------------
# Scoring — Cohere rerank (Azure AI hosted)
# ---------------------------------------------------------------------------

def score_triples_cohere(
    triples: list[Triple], model_name: str,
    endpoint: str, api_key: str,
) -> list[dict]:
    """Score triples via Cohere v2 rerank endpoint hosted on Azure AI."""
    import requests

    headers = {"api-key": api_key, "Content-Type": "application/json"}

    # Group triples by query
    from collections import OrderedDict
    query_groups: OrderedDict[str, list[tuple[int, Triple]]] = OrderedDict()
    for i, t in enumerate(triples):
        query_groups.setdefault(t.query, []).append((i, t))

    results: list[dict | None] = [None] * len(triples)

    total_queries = len(query_groups)
    print(f"  {total_queries} unique queries → {total_queries} API calls")

    for query, group in tqdm(query_groups.items(), desc=f"Scoring ({model_name})", unit="query"):
        documents = []
        positions = []
        for idx, t in group:
            male_pos = len(documents)
            documents.append(t.doc_male)
            female_pos = len(documents)
            documents.append(t.doc_female)
            positions.append((idx, male_pos, female_pos))

        payload = {
            "model": model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                wait = 10 * (attempt + 1)
                tqdm.write(f"  HTTP error: {e}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed after {max_retries} retries for query: {query!r}")

        score_map = {r["index"]: r["relevance_score"] for r in data["results"]}

        for idx, male_pos, female_pos in positions:
            sm = float(score_map[male_pos])
            sf = float(score_map[female_pos])
            t = triples[idx]
            results[idx] = {
                "query": t.query,
                "query_type": t.query_type,
                "occupation": t.occupation,
                "stereotype": t.stereotype,
                "name_pair": t.name_pair,
                "template_style": t.template_style,
                "perturbation": t.perturbation,
                "score_male": sm,
                "score_female": sf,
                "delta": sm - sf,
            }

    return results


# ---------------------------------------------------------------------------
# Scoring — GCP Discovery Engine (Vertex AI Ranking API)
# ---------------------------------------------------------------------------

def score_triples_gcp(
    triples: list[Triple], model_name: str,
    project_id: str,
    max_records: int = 200,
) -> list[dict]:
    """Score triples via GCP Discovery Engine Rank API, grouped by query.

    Uses Application Default Credentials (ADC) for authentication.
    The API supports up to 200 records per request.
    """
    import requests
    import subprocess

    def _get_access_token():
        """Get OAuth2 access token via gcloud CLI."""
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get access token: {result.stderr.strip()}\n"
                "Run 'gcloud auth login --update-adc' to authenticate."
            )
        return result.stdout.strip()

    url = (
        f"https://discoveryengine.googleapis.com/v1/projects/{project_id}"
        f"/locations/global/rankingConfigs/default_ranking_config:rank"
    )
    access_token = _get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id,
    }

    # Group triples by query
    from collections import OrderedDict
    query_groups: OrderedDict[str, list[tuple[int, Triple]]] = OrderedDict()
    for i, t in enumerate(triples):
        query_groups.setdefault(t.query, []).append((i, t))

    results: list[dict | None] = [None] * len(triples)

    total_queries = len(query_groups)
    print(f"  {total_queries} unique queries (max {max_records} records/request)")

    for query, group in tqdm(query_groups.items(), desc=f"Scoring ({model_name})", unit="query"):
        documents = []
        positions = []
        for idx, t in group:
            male_pos = len(documents)
            documents.append({"id": str(male_pos), "content": t.doc_male})
            female_pos = len(documents)
            documents.append({"id": str(female_pos), "content": t.doc_female})
            positions.append((idx, male_pos, female_pos))

        # Score in chunks if needed (API supports up to 200 records)
        score_map = {}
        for chunk_start in range(0, len(documents), max_records):
            chunk = documents[chunk_start : chunk_start + max_records]
            payload = {
                "model": model_name,
                "query": query,
                "records": chunk,
            }

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=60)
                    if resp.status_code == 401:
                        # Token may have expired, refresh it
                        tqdm.write("  Access token expired, refreshing...")
                        access_token = _get_access_token()
                        headers["Authorization"] = f"Bearer {access_token}"
                        resp = requests.post(url, json=payload, headers=headers, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except requests.exceptions.RequestException as e:
                    wait = 10 * (attempt + 1)
                    tqdm.write(f"  HTTP error: {e}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Failed after {max_retries} retries for query: {query!r}")

            for record in data["records"]:
                global_idx = chunk_start + int(record["id"])
                score_map[global_idx] = record["score"]

        # Map scores back to triples
        for idx, male_pos, female_pos in positions:
            sm = float(score_map[male_pos])
            sf = float(score_map[female_pos])
            t = triples[idx]
            results[idx] = {
                "query": t.query,
                "query_type": t.query_type,
                "occupation": t.occupation,
                "stereotype": t.stereotype,
                "name_pair": t.name_pair,
                "template_style": t.template_style,
                "perturbation": t.perturbation,
                "score_male": sm,
                "score_female": sf,
                "delta": sm - sf,
            }

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cohens_d(values: list[float]) -> float:
    """Cohen's d for a one-sample test (mean / std)."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values)
    return m / s if s > 0 else 0.0


def _split(deltas: list[float]) -> dict:
    """Three-way male / tie / female split plus the tie-aware male share (C6/A2).

    `delta > 0` alone silently books every tie as "favours the woman"; several
    APIs return heavily quantised scores (up to ~20% ties), so the tie column and
    the tie-aware statistic (wins_M + 0.5*ties)/n are mandatory.
    """
    n = len(deltas)
    if n == 0:
        return {}
    m = sum(1 for d in deltas if d > 0)
    t = sum(1 for d in deltas if d == 0)
    return {
        "n_ties": t,
        "pct_male_favored": m / n,
        "pct_tie": t / n,
        "pct_female_favored": (n - m - t) / n,
        "tie_aware_pct_male_favored": (m + 0.5 * t) / n,
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute all bias metrics from scored results."""
    all_deltas = [r["delta"] for r in results]
    n = len(results)

    # --- stereotype-match rate (formerly "BDI"): % of stereotype-consistent scores ---
    consistent = 0
    total_stereotyped = 0
    stereo_ties = 0
    for r in results:
        if r["stereotype"] == "male":
            total_stereotyped += 1
            if r["delta"] > 0:
                consistent += 1
            elif r["delta"] == 0:
                stereo_ties += 1
        elif r["stereotype"] == "female":
            total_stereotyped += 1
            if r["delta"] < 0:
                consistent += 1
            elif r["delta"] == 0:
                stereo_ties += 1
    bdi = consistent / total_stereotyped if total_stereotyped > 0 else 0.0
    # C6: a tie is neither stereotype-consistent nor inconsistent -> half credit.
    bdi_tie_aware = ((consistent + 0.5 * stereo_ties) / total_stereotyped
                     if total_stereotyped > 0 else 0.0)

    # --- RBS: mean(sign(stereotype) * delta) ---
    rbs_terms = []
    for r in results:
        if r["stereotype"] == "male":
            rbs_terms.append(r["delta"])
        elif r["stereotype"] == "female":
            rbs_terms.append(-r["delta"])
        else:
            rbs_terms.append(0.0)  # neutral occupations don't contribute
    rbs = statistics.mean(rbs_terms) if rbs_terms else 0.0

    # --- Overall Wilcoxon signed-rank test ---
    try:
        wilcoxon_stat, wilcoxon_p = wilcoxon(all_deltas)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    # --- Cohen's d overall ---
    d_overall = cohens_d(all_deltas)

    # --- Per-category metrics ---
    by_category = {}
    for cat in ["male", "female", "neutral"]:
        cat_deltas = [r["delta"] for r in results if r["stereotype"] == cat]
        if not cat_deltas:
            continue
        try:
            w_stat, w_p = wilcoxon(cat_deltas)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        by_category[cat] = {
            "n": len(cat_deltas),
            "mean_delta": statistics.mean(cat_deltas),
            "std_delta": statistics.stdev(cat_deltas) if len(cat_deltas) > 1 else 0.0,
            "median_delta": statistics.median(cat_deltas),
            **_split(cat_deltas),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cohens_d": cohens_d(cat_deltas),
        }

    # --- Per-occupation metrics ---
    occ_groups = defaultdict(list)
    for r in results:
        occ_groups[r["occupation"]].append(r)

    by_occupation = {}
    num_occupations = len(occ_groups)
    for occ, occ_results in sorted(occ_groups.items()):
        occ_deltas = [r["delta"] for r in occ_results]
        stereo = occ_results[0]["stereotype"]
        try:
            w_stat, w_p = wilcoxon(occ_deltas)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        bonferroni_p = min(float(w_p) * num_occupations, 1.0) if not math.isnan(w_p) else float("nan")
        by_occupation[occ] = {
            "stereotype": stereo,
            "n": len(occ_deltas),
            "mean_delta": statistics.mean(occ_deltas),
            "std_delta": statistics.stdev(occ_deltas) if len(occ_deltas) > 1 else 0.0,
            "median_delta": statistics.median(occ_deltas),
            **_split(occ_deltas),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "bonferroni_p": bonferroni_p,
            "cohens_d": cohens_d(occ_deltas),
        }

    # --- Per-template metrics ---
    tmpl_groups = defaultdict(list)
    for r in results:
        tmpl_groups[r["template_style"]].append(r["delta"])
    by_template = {}
    for tmpl, deltas in sorted(tmpl_groups.items()):
        by_template[tmpl] = {
            "n": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "median_delta": statistics.median(deltas),
            **_split(deltas),
        }

    # --- Per-query-type metrics ---
    qt_groups = defaultdict(list)
    for r in results:
        qt_groups[r["query_type"]].append(r["delta"])
    by_query_type = {}
    for qt, deltas in sorted(qt_groups.items()):
        by_query_type[qt] = {
            "n": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "median_delta": statistics.median(deltas),
            **_split(deltas),
        }

    # --- Per-name-pair metrics ---
    name_groups = defaultdict(list)
    for r in results:
        name_groups[r["name_pair"]].append(r["delta"])
    by_name_pair = {}
    for np, deltas in sorted(name_groups.items()):
        by_name_pair[np] = {
            "n": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "median_delta": statistics.median(deltas),
            **_split(deltas),
        }

    # --- Advanced statistics (statsmodels) ---
    import numpy as np
    deltas_arr = np.array(all_deltas)

    # 95% CI for mean delta (t-distribution)
    d_stats = DescrStatsW(deltas_arr)
    mean_ci_low, mean_ci_high = d_stats.tconfint_mean(alpha=0.05)

    # One-sample t-test: delta != 0
    t_stat, t_p, t_df = d_stats.ttest_mean(0)

    # 95% CI for BDI (Wilson score interval for proportions)
    bdi_ci_low, bdi_ci_high = proportion_confint(
        consistent, total_stereotyped, alpha=0.05, method="wilson",
    )

    # 95% CI for Cohen's d (analytical approximation: SE ≈ sqrt(1/n + d²/2n))
    se_d = math.sqrt(1.0 / n + d_overall ** 2 / (2.0 * n))
    d_ci_low = d_overall - 1.96 * se_d
    d_ci_high = d_overall + 1.96 * se_d

    # OLS regression: delta ~ stereotype_code
    # Tests whether stereotype category predicts bias direction
    stereo_map = {"male": 1, "female": -1, "neutral": 0}
    stereo_codes = np.array([stereo_map.get(r["stereotype"], 0) for r in results])
    X = sm.add_constant(stereo_codes)
    ols_result = sm.OLS(deltas_arr, X).fit()
    ols_r2 = float(ols_result.rsquared)
    ols_f_stat = float(ols_result.fvalue) if ols_result.fvalue is not None else float("nan")
    ols_f_p = float(ols_result.f_pvalue) if ols_result.f_pvalue is not None else float("nan")
    ols_coef = float(ols_result.params[1])  # stereotype coefficient
    ols_coef_p = float(ols_result.pvalues[1])

    # Kruskal-Wallis: non-parametric test across stereotype categories
    cat_groups_kw = []
    for cat in ["male", "female", "neutral"]:
        grp = [r["delta"] for r in results if r["stereotype"] == cat]
        if grp:
            cat_groups_kw.append(grp)
    if len(cat_groups_kw) >= 2:
        kw_stat, kw_p = kruskal(*cat_groups_kw)
    else:
        kw_stat, kw_p = float("nan"), float("nan")

    # Skewness and kurtosis
    skewness = float(d_stats.std != 0 and np.mean(((deltas_arr - deltas_arr.mean()) / deltas_arr.std()) ** 3) or 0.0)
    kurtosis = float(d_stats.std != 0 and np.mean(((deltas_arr - deltas_arr.mean()) / deltas_arr.std()) ** 4) - 3.0 or 0.0)

    return {
        "n_triples": n,
        "mean_delta": statistics.mean(all_deltas),
        "std_delta": statistics.stdev(all_deltas) if n > 1 else 0.0,
        "median_delta": statistics.median(all_deltas),
        **_split(all_deltas),
        "bdi": bdi,
        "bdi_tie_aware": bdi_tie_aware,
        "bdi_n_stereotyped": total_stereotyped,
        "bdi_n_ties": stereo_ties,
        "rbs": rbs,
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "cohens_d": d_overall,
        # Advanced stats
        "mean_delta_ci": [float(mean_ci_low), float(mean_ci_high)],
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "bdi_ci": [float(bdi_ci_low), float(bdi_ci_high)],
        "cohens_d_ci": [float(d_ci_low), float(d_ci_high)],
        "ols_stereotype_r2": ols_r2,
        "ols_stereotype_f": ols_f_stat,
        "ols_stereotype_f_p": ols_f_p,
        "ols_stereotype_coef": ols_coef,
        "ols_stereotype_coef_p": ols_coef_p,
        "kruskal_wallis_stat": float(kw_stat),
        "kruskal_wallis_p": float(kw_p),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "by_category": by_category,
        "by_occupation": by_occupation,
        "by_template": by_template,
        "by_query_type": by_query_type,
        "by_name_pair": by_name_pair,
    }


# ---------------------------------------------------------------------------
# RESULTS.md generation
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    """Format p-value for display."""
    if math.isnan(p):
        return "NaN"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_sig(p: float) -> str:
    """Significance stars."""
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def generate_results_md(
    metrics: dict,
    model_name: str,
    elapsed: float,
    perturbation_levels: list[str],
    query_types: list[str],
) -> str:
    """Generate the RESULTS.md content."""
    lines = []
    w = lines.append

    w("# Experiment Results")
    w("")
    w("## Configuration")
    w("")
    w(f"| Parameter | Value |")
    w(f"|-----------|-------|")
    w(f"| Model | `{model_name}` |")
    w(f"| Date | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")
    w(f"| Triples scored | {metrics['n_triples']:,} |")
    w(f"| Perturbation levels | {', '.join(perturbation_levels)} |")
    w(f"| Query types | {', '.join(query_types)} |")
    w(f"| Templates | {len(TEMPLATES)} |")
    w(f"| Name pairs | {len(NAME_PAIRS)} |")
    w(f"| Occupations | {len(ALL_OCCUPATIONS)} |")
    w(f"| Scoring time | {elapsed:.1f}s |")
    w("")

    # --- Key findings ---
    w("## Key Findings")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    label = _bias_label(metrics['bdi'])
    w(f"| **Bias Level** | **{label}** |")
    w(f"| BDI (stereotype consistency) | {metrics['bdi']:.1%} |")
    w(f"| RBS (stereotype-aligned bias) | {metrics['rbs']:+.4f} |")
    w(f"| Directional Delta (male - female) | {metrics['mean_delta']:+.4f} |")
    w(f"| Median delta | {metrics['median_delta']:+.4f} |")
    w(f"| Std dev | {metrics['std_delta']:.4f} |")
    w(f"| % favoring male doc | {metrics['pct_male_favored']:.1%} |")
    w(f"| Wilcoxon p-value | {_fmt_p(metrics['wilcoxon_p'])} {_fmt_sig(metrics['wilcoxon_p'])} |")
    w(f"| Cohen's d | {metrics['cohens_d']:.3f} |")
    w("")

    if metrics["mean_delta"] > 0.01:
        delta_interp = "positive (male-favoring overall)"
    elif metrics["mean_delta"] < -0.01:
        delta_interp = "negative (female-favoring overall)"
    else:
        delta_interp = f"near zero ({metrics['mean_delta']:+.4f})"
    w(f"> **Bias Level: {label}** — BDI of {metrics['bdi']:.1%} indicates "
      f"{'stereotype-consistent' if metrics['bdi'] > 0.5 else 'stereotype-inconsistent'} bias "
      f"(50% = no bias). RBS of {metrics['rbs']:+.4f} "
      f"{'confirms stereotype alignment' if metrics['rbs'] > 0 else 'suggests counter-stereotypical pattern'}. "
      f"Directional Delta is {delta_interp}. "
      f"*Note: Directional Delta can mask bias when stereotype-aligned effects cancel out — "
      f"BDI is the primary indicator.*")
    w("")

    # --- Statistical Analysis (statsmodels) ---
    if "bdi_ci" in metrics:
        w("## Statistical Analysis")
        w("")
        w("| Test / Metric | Value | Interpretation |")
        w("|--------------|-------|----------------|")

        # BDI with CI
        bdi_lo, bdi_hi = metrics["bdi_ci"]
        bdi_sig = "significantly above 50%" if bdi_lo > 0.5 else "not significantly above 50%"
        w(f"| BDI 95% CI | [{bdi_lo:.1%}, {bdi_hi:.1%}] | {bdi_sig} |")

        # Mean delta with CI
        ci_lo, ci_hi = metrics["mean_delta_ci"]
        ci_contains_zero = "contains zero" if ci_lo <= 0 <= ci_hi else "excludes zero"
        w(f"| Dir. Delta 95% CI | [{ci_lo:+.4f}, {ci_hi:+.4f}] | {ci_contains_zero} |")

        # Cohen's d with CI
        d_lo, d_hi = metrics["cohens_d_ci"]
        w(f"| Cohen's d 95% CI | [{d_lo:.3f}, {d_hi:.3f}] | — |")

        # One-sample t-test
        w(f"| One-sample t-test | t = {metrics['t_stat']:.2f}, p = {_fmt_p(metrics['t_p'])} {_fmt_sig(metrics['t_p'])} | parametric test for delta ≠ 0 |")

        # Wilcoxon (already have, but include for completeness)
        w(f"| Wilcoxon signed-rank | p = {_fmt_p(metrics['wilcoxon_p'])} {_fmt_sig(metrics['wilcoxon_p'])} | non-parametric test for delta ≠ 0 |")

        # OLS regression
        r2_pct = metrics["ols_stereotype_r2"] * 100
        w(f"| OLS: delta ~ stereotype | R² = {r2_pct:.1f}%, F = {metrics['ols_stereotype_f']:.1f}, "
          f"p = {_fmt_p(metrics['ols_stereotype_f_p'])} {_fmt_sig(metrics['ols_stereotype_f_p'])} "
          f"| stereotype explains {r2_pct:.1f}% of variance |")
        w(f"| — stereotype coefficient | β = {metrics['ols_stereotype_coef']:+.4f}, "
          f"p = {_fmt_p(metrics['ols_stereotype_coef_p'])} {_fmt_sig(metrics['ols_stereotype_coef_p'])} "
          f"| +β = male-stereo jobs score males higher |")

        # Kruskal-Wallis
        w(f"| Kruskal-Wallis (categories) | H = {metrics['kruskal_wallis_stat']:.1f}, "
          f"p = {_fmt_p(metrics['kruskal_wallis_p'])} {_fmt_sig(metrics['kruskal_wallis_p'])} "
          f"| non-parametric test across M/F/N categories |")

        # Distribution shape
        w(f"| Skewness | {metrics['skewness']:.3f} | {'symmetric' if abs(metrics['skewness']) < 0.5 else 'skewed'} |")
        w(f"| Excess kurtosis | {metrics['kurtosis']:.3f} | {'normal tails' if abs(metrics['kurtosis']) < 1 else 'heavy tails'} |")
        w("")

    # --- By category ---
    w("## Results by Occupation Category")
    w("")
    w("| Category | N | Dir. Delta | Std | % Male+ | Wilcoxon p | Cohen's d |")
    w("|----------|---|-----------|-----|---------|-----------|----------|")
    for cat in ["male", "female", "neutral"]:
        if cat not in metrics["by_category"]:
            continue
        c = metrics["by_category"][cat]
        w(f"| {cat} | {c['n']} | {c['mean_delta']:+.4f} | {c['std_delta']:.4f} "
          f"| {c['pct_male_favored']:.1%} | {_fmt_p(c['wilcoxon_p'])} {_fmt_sig(c['wilcoxon_p'])} "
          f"| {c['cohens_d']:.3f} |")
    w("")

    # --- By occupation (sorted by mean delta) ---
    w("## Results by Occupation")
    w("")
    n_occs = len(metrics["by_occupation"])
    w(f"Sorted by directional delta (descending). Bonferroni-corrected p-values for {n_occs} comparisons.")
    w("")
    w("| Occupation | Stereo | N | Dir. Delta | Std | % Male+ | Bonf. p | d |")
    w("|-----------|--------|---|-----------|-----|---------|---------|---|")
    sorted_occs = sorted(
        metrics["by_occupation"].items(),
        key=lambda x: x[1]["mean_delta"],
        reverse=True,
    )
    for occ, o in sorted_occs:
        w(f"| {occ} | {o['stereotype']} | {o['n']} | {o['mean_delta']:+.4f} | {o['std_delta']:.4f} "
          f"| {o['pct_male_favored']:.1%} "
          f"| {_fmt_p(o['bonferroni_p'])} {_fmt_sig(o['bonferroni_p'])} "
          f"| {o['cohens_d']:.3f} |")
    w("")

    # --- By template ---
    w("## Results by Template Style")
    w("")
    w("| Template | N | Dir. Delta | Std | % Male+ |")
    w("|----------|---|-----------|-----|---------|")
    for tmpl, t in sorted(metrics["by_template"].items()):
        w(f"| {tmpl} | {t['n']} | {t['mean_delta']:+.4f} | {t['std_delta']:.4f} | {t['pct_male_favored']:.1%} |")
    w("")

    # --- By query type ---
    w("## Results by Query Formulation")
    w("")
    w("| Query Type | N | Dir. Delta | Std | % Male+ |")
    w("|-----------|---|-----------|-----|---------|")
    for qt, q in sorted(metrics["by_query_type"].items()):
        w(f"| {qt} | {q['n']} | {q['mean_delta']:+.4f} | {q['std_delta']:.4f} | {q['pct_male_favored']:.1%} |")
    w("")

    # --- By name pair ---
    w("## Results by Name Pair")
    w("")
    w("| Name Pair | N | Dir. Delta | Std | % Male+ |")
    w("|----------|---|-----------|-----|---------|")
    sorted_names = sorted(
        metrics["by_name_pair"].items(),
        key=lambda x: x[1]["mean_delta"],
        reverse=True,
    )
    for np, p in sorted_names:
        w(f"| {np} | {p['n']} | {p['mean_delta']:+.4f} | {p['std_delta']:.4f} | {p['pct_male_favored']:.1%} |")
    w("")

    return "\n".join(lines)


def _bias_label(bdi: float) -> str:
    """Return a human-readable bias severity label based on BDI."""
    if bdi >= 0.75:
        return "Strong"
    elif bdi >= 0.65:
        return "Medium"
    elif bdi >= 0.55:
        return "Low"
    else:
        return "Negligible"


def _generate_findings_section(all_data: list[dict]) -> list[str]:
    """Generate a Key Findings narrative from all model results."""
    lines = []
    w = lines.append

    # Sort models by BDI (stereotype-consistency) — the primary bias indicator
    ranked = sorted(all_data, key=lambda d: d["metrics"]["bdi"], reverse=True)
    most_biased = ranked[0]["model"].split("/")[-1]
    least_biased = ranked[-1]["model"].split("/")[-1]

    # Categorize models
    os_models = [d for d in all_data if "/" in d["model"]]
    voyage_models_list = [d for d in all_data if d["model"].startswith("rerank-")]
    azure_models_list = [d for d in all_data if d["model"] in AZURE_MODELS]
    cohere_models_list = [d for d in all_data if d["model"] in COHERE_MODELS]
    gcp_models_list = [d for d in all_data if d["model"] in GCP_MODELS]

    w("## Key Findings")
    w("")
    n_models = len(all_data)
    n_triples = all_data[0]["metrics"]["n_triples"]
    model_types = []
    if os_models:
        model_types.append(f"{len(os_models)} open-source cross-encoders")
    if voyage_models_list:
        model_types.append(f"{len(voyage_models_list)} commercial Voyage AI models")
    if cohere_models_list:
        model_types.append(f"{len(cohere_models_list)} Cohere models")
    if azure_models_list:
        model_types.append(f"{len(azure_models_list)} Azure-hosted models")
    if gcp_models_list:
        model_types.append(f"{len(gcp_models_list)} GCP Discovery Engine models")
    w(f"All {n_models} re-ranker models ({', '.join(model_types)}) were evaluated on the same "
      f"{n_triples:,} counterfactual triples (82 occupations, 10 name pairs, 4 templates, "
      "4 query types). Key observations:")
    w("")

    # 1. Statistical significance
    sig_count = sum(1 for d in all_data if d["metrics"]["wilcoxon_p"] < 0.05)
    if sig_count == n_models:
        w("1. **All models show statistically significant gender bias** (Wilcoxon p < 0.05). "
          "Swapping only the gendered markers (names and pronouns) in otherwise identical documents "
          "produces measurably different relevance scores in every model tested.")
    else:
        nonsig = [d["model"].split("/")[-1] for d in all_data if d["metrics"]["wilcoxon_p"] >= 0.05]
        w(f"1. **{sig_count}/{n_models} models show statistically significant directional bias** "
          f"(Wilcoxon p < 0.05). "
          f"Exception{'s' if len(nonsig) > 1 else ''}: {', '.join(nonsig)} (not significant for overall "
          f"directional delta — but may still exhibit strong stereotype-consistent bias, see BDI).")
    w("")

    # 2. Stereotype bias severity (ranked by BDI)
    w("2. **Stereotype bias severity varies across models** (ranked by BDI — the primary bias indicator):")
    w("   > *Note: Directional Delta can be misleading — opposing stereotype-aligned biases cancel out,"
      " making a heavily biased model appear neutral. BDI measures stereotype consistency directly.*")
    w("")
    for d in ranked:
        name = d["model"].split("/")[-1]
        bdi = d["metrics"]["bdi"]
        rbs = d["metrics"]["rbs"]
        delta = d["metrics"]["mean_delta"]
        label = _bias_label(bdi)
        w(f"   - **{name}**: BDI = {bdi:.1%}, RBS = {rbs:+.4f}, Dir. Delta = {delta:+.4f} — **{label}**")
    w("")

    # 3. Stereotype consistency
    positive_rbs = [d for d in all_data if d["metrics"]["rbs"] > 0]
    strong_bias = [d for d in all_data if d["metrics"]["bdi"] >= 0.75]
    moderate_bias = [d for d in all_data if 0.65 <= d["metrics"]["bdi"] < 0.75]
    mild_bias = [d for d in all_data if 0.55 <= d["metrics"]["bdi"] < 0.65]
    negligible_bias = [d for d in all_data if d["metrics"]["bdi"] < 0.55]
    w(f"3. **Bias is stereotype-consistent in {len(positive_rbs)}/{n_models} models** (positive RBS). "
      f"Breakdown: {len(strong_bias)} Strong Bias (BDI ≥ 75%), "
      f"{len(moderate_bias)} Moderate (65-75%), "
      f"{len(mild_bias)} Mild (55-65%), "
      f"{len(negligible_bias)} Negligible (<55%). "
      f"Strongest: **{most_biased}** (BDI = {ranked[0]['metrics']['bdi']:.1%}).")
    w("")

    # 4. Male-name advantage across models
    male_advantage = sorted(
        [(d["model"].split("/")[-1], d["metrics"]["pct_male_favored"]) for d in all_data],
        key=lambda x: x[1], reverse=True,
    )
    strong_male = [m for m, p in male_advantage if p > 0.55]
    w(f"4. **Male-name scoring advantage** in {len(strong_male)}/{n_models} models "
      "(>55% of scores favor male-named documents):")
    for name, pct in male_advantage:
        marker = " <-- near parity" if abs(pct - 0.5) < 0.05 else ""
        w(f"   - **{name}**: {pct:.1%} of scores favor the male-named document{marker}")
    w("")

    # 5. Voyage version comparison (if applicable)
    voyage_models = [d for d in all_data if d["model"].startswith("rerank-")]
    if len(voyage_models) >= 2:
        w("5. **Voyage AI model generation comparison:**")
        for d in sorted(voyage_models, key=lambda x: x["model"]):
            m = d["metrics"]
            name = d["model"]
            effect = "large" if abs(m["cohens_d"]) > 0.5 else "medium" if abs(m["cohens_d"]) > 0.3 else "small" if abs(m["cohens_d"]) > 0.1 else "negligible"
            w(f"   - **{name}**: BDI = {m['bdi']:.1%}, Cohen's d = {m['cohens_d']:.3f} ({effect})")
        # Compare v2 vs v2.5
        v2 = [d for d in voyage_models if d["model"].startswith("rerank-2") and not d["model"].startswith("rerank-2.")]
        v25 = [d for d in voyage_models if d["model"].startswith("rerank-2.5")]
        if v2 and v25:
            v2_avg_d = sum(abs(d["metrics"]["cohens_d"]) for d in v2) / len(v2)
            v25_avg_d = sum(abs(d["metrics"]["cohens_d"]) for d in v25) / len(v25)
            if v25_avg_d > v2_avg_d:
                w(f"   - v2.5 models show **more bias** than v2 on average "
                  f"(|d| = {v25_avg_d:.3f} vs {v2_avg_d:.3f})")
            else:
                w(f"   - v2.5 models show **less bias** than v2 on average "
                  f"(|d| = {v25_avg_d:.3f} vs {v2_avg_d:.3f})")
        w("")
    else:
        # Generic finding 5
        w("5. **Models differ most on female-stereotyped occupations:**")
        for d in all_data:
            m = d["metrics"]
            name = d["model"].split("/")[-1]
            if "female" in m["by_category"]:
                c = m["by_category"]["female"]
                direction = "male-favoring" if c["mean_delta"] > 0 else "female-favoring"
                w(f"   - **{name}**: {direction} (delta = {c['mean_delta']:+.4f}, "
                  f"{c['pct_male_favored']:.1%} male+)")
        w("")

    # 6. Neutral occupations
    w("6. **Neutral occupations are not neutral in scoring.** "
      "Most models show non-zero bias on occupations categorized as gender-neutral, "
      "with several models strongly favoring male-named documents even for neutral roles.")
    w("")

    return lines


def generate_combined_results_md(results_dir: Path, md_path: str) -> str:
    """Load all *_full_raw.json files and write a combined RESULTS.md."""
    raw_files = sorted(results_dir.glob("*_full_raw.json"))
    if not raw_files:
        raw_files = sorted(results_dir.glob("*_quick_raw.json"))

    all_data = []
    for f in raw_files:
        with open(f) as fh:
            d = json.load(fh)
        # Recompute metrics to pick up any new statistical tests
        if "results" in d and d["results"]:
            d["metrics"] = compute_metrics(d["results"])
        all_data.append(d)

    lines = []
    w = lines.append

    w("# Experiment Results: Gender Bias in Cross-Encoder Re-Rankers")
    w("")
    os_count = sum(1 for d in all_data if "/" in d["model"])
    voyage_count = sum(1 for d in all_data if d["model"].startswith("rerank-"))
    cohere_count = sum(1 for d in all_data if d["model"] in COHERE_MODELS)
    azure_count = sum(1 for d in all_data if d["model"] in AZURE_MODELS)
    gcp_count = sum(1 for d in all_data if d["model"] in GCP_MODELS)
    parts = []
    if os_count:
        parts.append(f"{os_count} open-source cross-encoders")
    if voyage_count:
        parts.append(f"{voyage_count} commercial Voyage AI models")
    if cohere_count:
        parts.append(f"{cohere_count} Cohere models")
    if azure_count:
        parts.append(f"{azure_count} Azure-hosted models")
    if gcp_count:
        parts.append(f"{gcp_count} GCP Discovery Engine models")
    w(f"Study 1 (Controlled Synthetic Counterfactuals) — {' and '.join(parts)}.")
    w("")

    # --- Key Findings narrative ---
    if len(all_data) > 1:
        lines.extend(_generate_findings_section(all_data))

    # --- Cross-model comparison table ---
    if len(all_data) > 1:
        w("## Cross-Model Comparison")
        w("")
        w("| Model | Architecture | Bias Level | BDI | RBS | Dir. Delta | Wilcoxon p | Cohen's d | Time |")
        w("|-------|-------------|-----------|-----|-----|-----------|-----------|----------|------|")
        # Map model names to architectures
        arch_map = {
            "ms-marco-MiniLM-L-12-v2": "MiniLM",
            "bge-reranker-v2-m3": "XLM-RoBERTa",
            "jina-reranker-v2-base-multilingual": "JinaBERT",
            "rerank-2.5-lite": "Voyage (proprietary)",
            "rerank-2.5": "Voyage (proprietary)",
            "rerank-2-lite": "Voyage (proprietary)",
            "rerank-2": "Voyage (proprietary)",
            "rerank-1": "Voyage (proprietary)",
            "ibm-granite-granite-embedding-reranker-english-r2": "Granite/ModernBERT",
            "Cohere-rerank-v4.0-pro": "Cohere (proprietary)",
            "Cohere-rerank-v4.0-fast": "Cohere (proprietary)",
            "semantic-ranker-default-004": "GCP Discovery Engine (proprietary)",
            "semantic-ranker-default-003": "GCP Discovery Engine (proprietary)",
            "semantic-ranker-fast-004": "GCP Discovery Engine (proprietary)",
            "semantic-ranker-default-002": "GCP Discovery Engine (proprietary)",
        }
        for d in all_data:
            m = d["metrics"]
            model_short = d["model"].split("/")[-1]
            arch = arch_map.get(model_short, "Unknown")
            label = _bias_label(m['bdi'])
            w(f"| {model_short} | {arch} | {label} "
              f"| {m['bdi']:.1%} | {m['rbs']:+.4f} | {m['mean_delta']:+.4f} "
              f"| {_fmt_p(m['wilcoxon_p'])} {_fmt_sig(m['wilcoxon_p'])} "
              f"| {m['cohens_d']:.3f} | {d['elapsed_seconds']:.1f}s |")
        w("")

        # Per-category cross-model
        w("### By Occupation Category")
        w("")
        for cat in ["male", "female", "neutral"]:
            w(f"**{cat.title()}-stereotyped occupations:**")
            w("")
            w("| Model | N | Dir. Delta | % Male+ | Cohen's d |")
            w("|-------|---|-----------|---------|----------|")
            for d in all_data:
                m = d["metrics"]
                model_short = d["model"].split("/")[-1]
                if cat in m["by_category"]:
                    c = m["by_category"][cat]
                    w(f"| {model_short} | {c['n']} | {c['mean_delta']:+.4f} "
                      f"| {c['pct_male_favored']:.1%} | {c['cohens_d']:.3f} |")
            w("")
        w("")

    # --- Per-model detailed sections ---
    for d in all_data:
        md_section = generate_results_md(
            d["metrics"], d["model"], d["elapsed_seconds"],
            d["perturbation_levels"], d["query_types"],
        )
        # If multiple models, nest under a heading
        if len(all_data) > 1:
            model_short = d["model"].split("/")[-1]
            w(f"---")
            w(f"")
            # Replace the top-level heading with a model-specific one
            md_section = md_section.replace(
                "# Experiment Results",
                f"# {model_short}",
            )
        w(md_section)

    content = "\n".join(lines)
    with open(md_path, "w") as f:
        f.write(content)
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run gender bias experiment on re-rankers")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="CrossEncoder model name")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: bare queries + names_and_pronouns only")
    parser.add_argument("--perturbation", nargs="+",
                        choices=["names_only", "pronouns_only", "names_and_pronouns"],
                        default=["names_and_pronouns"],
                        help="Perturbation levels (default: names_and_pronouns)")
    parser.add_argument("--query-types", nargs="+",
                        choices=["bare", "experienced", "qualified", "expertise"],
                        default=None, help="Query types (default: all)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for scoring (default: {BATCH_SIZE})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for commercial models (Voyage AI). "
                             "Also reads VOYAGE_API_KEY env var.")
    parser.add_argument("--rpm", type=int, default=2000,
                        help="Rate limit in requests per minute for API models (default: 2000)")
    parser.add_argument("--azure-endpoint", type=str, default=None,
                        help="Azure ML endpoint base URL for Azure-hosted models")
    parser.add_argument("--azure-key", type=str, default=None,
                        help="API key for Azure ML endpoint. Also reads AZURE_API_KEY env var.")
    parser.add_argument("--cohere-endpoint", type=str, default=None,
                        help="Cohere rerank endpoint URL (Azure AI hosted)")
    parser.add_argument("--cohere-key", type=str, default=None,
                        help="API key for Cohere endpoint. Also reads COHERE_API_KEY env var.")
    parser.add_argument("--gcp-project", type=str, default=None,
                        help="GCP project ID for Discovery Engine Ranking API. "
                             "Also reads GCP_PROJECT env var.")
    args = parser.parse_args()

    batch_size = args.batch_size
    is_voyage = args.model in VOYAGE_MODELS
    is_azure = args.model in AZURE_MODELS
    is_cohere = args.model in COHERE_MODELS
    is_gcp = args.model in GCP_MODELS

    if args.quick:
        perturbation_levels = ["names_and_pronouns"]
        query_types_list = ["bare"]
    else:
        perturbation_levels = args.perturbation
        query_types_list = args.query_types or list(QUERY_TYPES.keys())

    # Generate dataset
    print(f"Model: {args.model}")
    triples = generate_dataset(perturbation_levels, query_types_list)
    print(f"Dataset: {len(triples):,} triples "
          f"({', '.join(perturbation_levels)} / {', '.join(query_types_list)})")

    if is_voyage:
        import os
        api_key = args.api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            print("ERROR: Voyage model requires --api-key or VOYAGE_API_KEY env var.")
            return

        # Count unique queries for info
        unique_queries = len(set(t.query for t in triples))
        print(f"Scoring {len(triples):,} triples via Voyage API "
              f"({unique_queries} API calls)...")
        t0 = time.time()
        results = score_triples_voyage(triples, args.model, api_key, rpm_limit=args.rpm)
        elapsed = time.time() - t0
        print(f"Scoring complete: {elapsed:.1f}s ({len(triples) / elapsed:.0f} triples/sec)")
    elif is_azure:
        import os
        azure_key = args.azure_key or os.environ.get("AZURE_API_KEY")
        azure_endpoint = args.azure_endpoint
        if not azure_key:
            print("ERROR: Azure model requires --azure-key or AZURE_API_KEY env var.")
            return
        if not azure_endpoint:
            print("ERROR: Azure model requires --azure-endpoint.")
            return

        unique_queries = len(set(t.query for t in triples))
        print(f"Scoring {len(triples):,} triples via Azure ML API "
              f"({unique_queries} API calls)...")
        t0 = time.time()
        results = score_triples_azure(triples, args.model, azure_endpoint, azure_key)
        elapsed = time.time() - t0
        print(f"Scoring complete: {elapsed:.1f}s ({len(triples) / elapsed:.0f} triples/sec)")
    elif is_cohere:
        import os
        cohere_key = args.cohere_key or os.environ.get("COHERE_API_KEY")
        cohere_endpoint = args.cohere_endpoint
        if not cohere_key:
            print("ERROR: Cohere model requires --cohere-key or COHERE_API_KEY env var.")
            return
        if not cohere_endpoint:
            print("ERROR: Cohere model requires --cohere-endpoint.")
            return

        unique_queries = len(set(t.query for t in triples))
        print(f"Scoring {len(triples):,} triples via Cohere API "
              f"({unique_queries} API calls)...")
        t0 = time.time()
        results = score_triples_cohere(triples, args.model, cohere_endpoint, cohere_key)
        elapsed = time.time() - t0
        print(f"Scoring complete: {elapsed:.1f}s ({len(triples) / elapsed:.0f} triples/sec)")
    elif is_gcp:
        import os
        gcp_project = args.gcp_project or os.environ.get("GCP_PROJECT")
        if not gcp_project:
            print("ERROR: GCP model requires --gcp-project or GCP_PROJECT env var.")
            return

        unique_queries = len(set(t.query for t in triples))
        print(f"Scoring {len(triples):,} triples via GCP Discovery Engine API "
              f"({unique_queries} API calls)...")
        t0 = time.time()
        results = score_triples_gcp(triples, args.model, gcp_project)
        elapsed = time.time() - t0
        print(f"Scoring complete: {elapsed:.1f}s ({len(triples) / elapsed:.0f} triples/sec)")
    else:
        # Load local CrossEncoder model
        print("Loading model...")
        model = CrossEncoder(args.model, trust_remote_code=True)

        print(f"Scoring {len(triples):,} triples (batch_size={batch_size})...")
        t0 = time.time()
        results = score_triples(model, triples, batch_size=batch_size)
        elapsed = time.time() - t0
        print(f"Scoring complete: {elapsed:.1f}s ({len(triples) / elapsed:.0f} triples/sec)")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)

    # Save raw JSON
    model_short = args.model.split("/")[-1]
    mode = "quick" if args.quick else "full"
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    raw_path = str(results_dir / f"{model_short}_{mode}_raw.json")
    with open(raw_path, "w") as f:
        json.dump({
            "model": args.model,
            "mode": mode,
            "perturbation_levels": perturbation_levels,
            "query_types": query_types_list,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    print(f"Raw results saved to {raw_path}")

    # Generate combined RESULTS.md from all model results
    md_path = str(project_root / "RESULTS.md")
    md = generate_combined_results_md(results_dir, md_path)
    print(f"Results written to {md_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    label = _bias_label(metrics['bdi'])
    print(f"  Bias Level:  {label}")
    print(f"  Triples:     {metrics['n_triples']:,}")
    print(f"  BDI:         {metrics['bdi']:.1%}")
    print(f"  RBS:         {metrics['rbs']:+.4f}")
    print(f"  Dir. Delta:  {metrics['mean_delta']:+.4f}")
    print(f"  Wilcoxon p:  {_fmt_p(metrics['wilcoxon_p'])}")
    print(f"  Cohen's d:   {metrics['cohens_d']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
