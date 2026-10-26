"""Exploratory counterfactual audit using racialized first-name cues.

This script is deliberately separate from the manuscript build.  It neither
reads nor writes either paper source.  It scores otherwise identical synthetic
professional documents after changing only the first name; surname, pronouns,
qualifications, occupation, template, and query stay fixed.

Terminology matters: the names below are *Black-associated* and
*White-associated* experimental cues.  They do not reveal anyone's race.

Name provenance
---------------
The candidate names and their published racial-distinctiveness and DOLMA
frequency measurements come from Tables 4 and 6 of:

    Wilson, K. and Caliskan, A. (2024), "Gender, Race, and Intersectional
    Bias in Resume Screening via Language Model Retrieval," AIES 2024.
    https://arxiv.org/abs/2407.20371

Wilson and Caliskan selected the names from the perception database introduced
by:

    Elder, E. M. and Hayes, M. (2023), "Signaling Race, Ethnicity, and Gender
    with Names: Challenges and Recommendations," Journal of Politics 85(2),
    764--770. https://doi.org/10.1086/723820

The fixed surname Williams follows Wilson and Caliskan.  They chose it because
it is common and approximately evenly associated with Black and White people
in Census surname data.  Their frequency-matched White-name lists are used.

Pre-score selection rule
------------------------
The ten pairs below were fixed before any reranker score from this experiment
was inspected.  Within gender, we required both names to have published racial
distinctiveness >= 0.75, equal subword counts (including "Williams") under
both audited rerankers, and close DOLMA log-frequency.  We then selected five
disjoint pairs with the smallest total frequency distance.  For the male set,
the exact minimum had tied solutions; the tie was resolved by the larger sum
of the weaker distinctiveness value in each pair.  This is an exploratory,
post-hoc extension to the existing project, not a preregistered confirmatory
study.

Default outputs:
    results/racialized_name_pilot_<model-slug>.json  (raw scores + summary)
    results/racialized_name_pilot_summary.json       (cross-model summary)

Run offline with cached weights:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      venv/bin/python code/racialized_name_experiment.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(CODE_DIR))

from deidentification_experiment import get_model  # noqa: E402
from synthetic_dataset import (  # noqa: E402
    ALL_OCCUPATIONS,
    LABEL_SOURCE,
    PRONOUNS,
    QUERY_TYPES,
    TEMPLATES,
    _fill_template,
)


SCRIPT_VERSION = "racialized_name_experiment.py v1.0"
SEED = 20260917
SURNAME = "Williams"
DEFAULT_MODELS = (
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
)

SOURCE = {
    "name_list": {
        "citation": (
            "Wilson, Kyra, and Aylin Caliskan. 2024. Gender, Race, and "
            "Intersectional Bias in Resume Screening via Language Model "
            "Retrieval. AIES 2024."
        ),
        "url": "https://arxiv.org/abs/2407.20371",
        "location": "Appendix Tables 4 and 6 (frequency-matched lists)",
    },
    "underlying_perception_database": {
        "citation": (
            "Elder, Elizabeth Mitchell, and Matthew Hayes. 2023. Signaling "
            "Race, Ethnicity, and Gender with Names: Challenges and "
            "Recommendations. The Journal of Politics 85(2): 764-770."
        ),
        "doi": "https://doi.org/10.1086/723820",
    },
    "surname_rationale": (
        "Constant surname Williams, following Wilson and Caliskan (2024): "
        "common and approximately balanced between Black and White Census "
        "surname categories."
    ),
}


@dataclass(frozen=True)
class NamePair:
    pair_id: str
    gender: str
    black_first: str
    white_first: str
    black_log_frequency: float
    white_log_frequency: float
    black_distinctiveness: float
    white_distinctiveness: float

    @property
    def black_name(self) -> str:
        return f"{self.black_first} {SURNAME}"

    @property
    def white_name(self) -> str:
        return f"{self.white_first} {SURNAME}"

    @property
    def log_frequency_gap(self) -> float:
        return abs(self.black_log_frequency - self.white_log_frequency)


# Locked before scoring.  Values are transcribed from Wilson & Caliskan's
# Appendix Tables 4 and 6.  "Distinctiveness" is their difference between the
# most likely and second-most-likely perceived racial group (1--5 scale).
NAME_PAIRS = (
    NamePair("F01", "female", "Ebony", "Brittany", 15.18, 15.20, 1.29, 1.46),
    NamePair("F02", "female", "Keisha", "Aileen", 12.93, 13.12, 1.30, 0.87),
    NamePair("F03", "female", "Monique", "Ethel", 14.14, 14.34, 0.87, 1.14),
    NamePair("F04", "female", "Chandra", "Terri", 14.75, 14.48, 0.88, 0.82),
    NamePair("F05", "female", "Latoya", "Rebeca", 11.68, 11.97, 1.09, 1.30),
    NamePair("M01", "male", "Tyrone", "Mathew", 14.32, 14.32, 1.72, 1.24),
    NamePair("M02", "male", "Abdul", "Spencer", 16.18, 16.18, 0.82, 0.90),
    NamePair("M03", "male", "Jerome", "Ricky", 15.30, 15.30, 0.82, 0.89),
    NamePair("M04", "male", "Darius", "Sylvester", 14.47, 14.48, 1.13, 0.82),
    NamePair("M05", "male", "Lamar", "Randall", 15.21, 15.20, 1.12, 0.87),
)


def slug(model: str) -> str:
    return model.replace("/", "_")


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def cached_revision(model_name: str) -> str | None:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    ref = cache / ("models--" + model_name.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_design(models: list[str]) -> dict:
    """Validate pre-score matching constraints and record tokenizer pieces."""
    from transformers import AutoTokenizer

    tokenization: dict[str, dict] = {}
    for model_name in models:
        tok = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True
        )
        tokenization[model_name] = {}
        for p in NAME_PAIRS:
            black_pieces = tok.tokenize(p.black_name)
            white_pieces = tok.tokenize(p.white_name)
            if len(black_pieces) != len(white_pieces):
                raise AssertionError(
                    f"token-count match failed for {model_name}, {p.pair_id}: "
                    f"{black_pieces} vs {white_pieces}"
                )
            tokenization[model_name][p.pair_id] = {
                "black_pieces": black_pieces,
                "white_pieces": white_pieces,
                "n_pieces_each": len(black_pieces),
            }

    assert len(NAME_PAIRS) == 10
    assert sum(p.gender == "female" for p in NAME_PAIRS) == 5
    assert sum(p.gender == "male" for p in NAME_PAIRS) == 5
    assert len({p.black_first for p in NAME_PAIRS}) == len(NAME_PAIRS)
    assert len({p.white_first for p in NAME_PAIRS}) == len(NAME_PAIRS)
    assert min(min(p.black_distinctiveness, p.white_distinctiveness) for p in NAME_PAIRS) >= 0.75
    return tokenization


def generate_rows() -> list[dict]:
    rows: list[dict] = []
    for occupation, occupation_group in ALL_OCCUPATIONS:
        for query_type, query_template in QUERY_TYPES.items():
            query = query_template.format(occupation=occupation)
            for template_style, template in TEMPLATES.items():
                for pair in NAME_PAIRS:
                    pronouns = PRONOUNS[pair.gender]
                    black_doc = _fill_template(
                        template, pair.black_name, occupation, pronouns
                    )
                    white_doc = _fill_template(
                        template, pair.white_name, occupation, pronouns
                    )
                    # This audit changes the first name only.
                    expected_black = black_doc.replace(pair.black_first, pair.white_first, 1)
                    if expected_black != white_doc:
                        raise AssertionError(
                            f"documents differ by more than first name for {pair.pair_id}"
                        )
                    rows.append(
                        {
                            "occupation": occupation,
                            "occupation_group": occupation_group,
                            "query_type": query_type,
                            "query": query,
                            "template_style": template_style,
                            "pair_id": pair.pair_id,
                            "gender": pair.gender,
                            "black_first": pair.black_first,
                            "white_first": pair.white_first,
                            "black_doc": black_doc,
                            "white_doc": white_doc,
                        }
                    )
    expected = len(ALL_OCCUPATIONS) * len(QUERY_TYPES) * len(TEMPLATES) * len(NAME_PAIRS)
    assert len(rows) == expected
    return rows


def score_rows(model_name: str, rows: list[dict], device: str, batch_size: int) -> list[dict]:
    model = get_model(model_name, device)
    needed: dict[tuple[str, str], None] = {}
    for row in rows:
        needed[(row["query"], row["black_doc"])] = None
        needed[(row["query"], row["white_doc"])] = None
    pairs = list(needed)
    print(f"  scoring {len(pairs):,} unique (query, document) inputs", flush=True)
    values = model.predict(
        [[q, d] for q, d in pairs],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    scores = {pair: float(value) for pair, value in zip(pairs, values)}

    out: list[dict] = []
    for row in rows:
        score_black = scores[(row["query"], row["black_doc"])]
        score_white = scores[(row["query"], row["white_doc"])]
        out.append(
            {
                k: row[k]
                for k in (
                    "occupation",
                    "occupation_group",
                    "query_type",
                    "template_style",
                    "pair_id",
                    "gender",
                    "black_first",
                    "white_first",
                )
            }
            | {
                "score_black": score_black,
                "score_white": score_white,
                "delta_white_minus_black": score_white - score_black,
            }
        )
    return out


def _point(records: list[dict]) -> dict:
    delta = np.asarray([r["delta_white_minus_black"] for r in records], dtype=float)
    white = int(np.sum(delta > 0))
    tie = int(np.sum(delta == 0))
    black = int(np.sum(delta < 0))
    n = len(records)
    return {
        "n_comparisons": n,
        "n_occupations": len({r["occupation"] for r in records}),
        "n_name_pairs": len({r["pair_id"] for r in records}),
        "pct_white_higher": 100.0 * white / n,
        "pct_tie": 100.0 * tie / n,
        "pct_black_higher": 100.0 * black / n,
        "tie_aware_pct_white": 100.0 * (white + 0.5 * tie) / n,
        "mean_delta_white_minus_black": float(np.mean(delta)),
        "median_delta_white_minus_black": float(np.median(delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
    }


def _bootstrap(records: list[dict], seed: int, B: int) -> dict:
    """Two-way cluster bootstrap over occupations and names, by gender stratum."""
    occupations = sorted({r["occupation"] for r in records})
    genders = sorted({r["gender"] for r in records})
    oi = {o: i for i, o in enumerate(occupations)}

    matrices: dict[str, dict[str, np.ndarray]] = {}
    for gender in genders:
        pair_ids = sorted({r["pair_id"] for r in records if r["gender"] == gender})
        pi = {p: i for i, p in enumerate(pair_ids)}
        shape = (len(occupations), len(pair_ids))
        mats = {
            "white": np.zeros(shape, dtype=float),
            "tie": np.zeros(shape, dtype=float),
            "n": np.zeros(shape, dtype=float),
            "delta": np.zeros(shape, dtype=float),
            "abs_delta": np.zeros(shape, dtype=float),
        }
        for r in records:
            if r["gender"] != gender:
                continue
            i, j = oi[r["occupation"]], pi[r["pair_id"]]
            d = r["delta_white_minus_black"]
            mats["white"][i, j] += d > 0
            mats["tie"][i, j] += d == 0
            mats["n"][i, j] += 1
            mats["delta"][i, j] += d
            mats["abs_delta"][i, j] += abs(d)
        matrices[gender] = mats | {"pair_ids": pair_ids}  # type: ignore[dict-item]

    rng = np.random.default_rng(seed)
    boot_white = np.empty(B)
    boot_mean = np.empty(B)
    boot_abs = np.empty(B)
    for b in range(B):
        occ_draw = rng.integers(0, len(occupations), len(occupations))
        totals = defaultdict(float)
        for gender, mats in matrices.items():
            n_pairs = len(mats["pair_ids"])
            pair_draw = rng.integers(0, n_pairs, n_pairs)
            ix = np.ix_(occ_draw, pair_draw)
            for key in ("white", "tie", "n", "delta", "abs_delta"):
                totals[key] += float(mats[key][ix].sum())
        boot_white[b] = 100.0 * (totals["white"] + 0.5 * totals["tie"]) / totals["n"]
        boot_mean[b] = totals["delta"] / totals["n"]
        boot_abs[b] = totals["abs_delta"] / totals["n"]

    def ci(a: np.ndarray) -> list[float]:
        return [float(x) for x in np.percentile(a, [2.5, 97.5])]

    return {
        "method": (
            "percentile bootstrap; occupations and name pairs resampled "
            "independently with replacement; name-pair resampling stratified by gender"
        ),
        "B": B,
        "seed": seed,
        "tie_aware_pct_white_ci": ci(boot_white),
        "mean_delta_white_minus_black_ci": ci(boot_mean),
        "mean_abs_delta_ci": ci(boot_abs),
    }


def summarize(records: list[dict], seed: int, B: int) -> dict:
    point = _point(records)
    point["bootstrap"] = _bootstrap(records, seed, B)
    return point


def _plain_breakdown(records: list[dict], field: str) -> dict:
    out = {}
    for value in sorted({r[field] for r in records}):
        subset = [r for r in records if r[field] == value]
        out[value] = _point(subset)
    return out


def analyze(records: list[dict], B: int) -> dict:
    summary = {
        "overall": summarize(records, SEED, B),
        "by_gender": {},
        "by_occupation_group": {},
        "by_gender_and_occupation_group": {},
        "by_name_pair": _plain_breakdown(records, "pair_id"),
        "by_template": _plain_breakdown(records, "template_style"),
        "by_query_type": _plain_breakdown(records, "query_type"),
    }
    for i, gender in enumerate(("female", "male")):
        sub = [r for r in records if r["gender"] == gender]
        summary["by_gender"][gender] = summarize(sub, SEED + 100 + i, B)
    for i, group in enumerate(("male", "female", "neutral")):
        sub = [r for r in records if r["occupation_group"] == group]
        summary["by_occupation_group"][group] = summarize(sub, SEED + 200 + i, B)
    for gi, gender in enumerate(("female", "male")):
        summary["by_gender_and_occupation_group"][gender] = {}
        for oi, group in enumerate(("male", "female", "neutral")):
            sub = [
                r
                for r in records
                if r["gender"] == gender and r["occupation_group"] == group
            ]
            summary["by_gender_and_occupation_group"][gender][group] = summarize(
                sub, SEED + 300 + gi * 10 + oi, B
            )

    pair_mean = {
        p: d["mean_delta_white_minus_black"]
        for p, d in summary["by_name_pair"].items()
    }
    occ_breakdown = _plain_breakdown(records, "occupation")
    occ_mean = {
        o: d["mean_delta_white_minus_black"] for o, d in occ_breakdown.items()
    }
    summary["directional_consistency"] = {
        "name_pairs_white_positive": sum(v > 0 for v in pair_mean.values()),
        "name_pairs_tied": sum(v == 0 for v in pair_mean.values()),
        "name_pairs_black_positive": sum(v < 0 for v in pair_mean.values()),
        "name_pairs_total": len(pair_mean),
        "occupations_white_positive": sum(v > 0 for v in occ_mean.values()),
        "occupations_tied": sum(v == 0 for v in occ_mean.values()),
        "occupations_black_positive": sum(v < 0 for v in occ_mean.values()),
        "occupations_total": len(occ_mean),
    }

    # Leave-one-name-pair-out sensitivity exposes whether one cue drives the mean.
    loo = {}
    for pair_id in sorted(pair_mean):
        loo[pair_id] = _point([r for r in records if r["pair_id"] != pair_id])
    summary["leave_one_name_pair_out"] = loo
    return summary


def model_metadata(model_name: str, device: str) -> dict:
    return {
        "model": model_name,
        "cached_revision": cached_revision(model_name),
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "sentence-transformers": package_version("sentence-transformers"),
            "numpy": package_version("numpy"),
        },
    }


def print_summary(model_name: str, summary: dict) -> None:
    print("\n" + "=" * 100)
    print(model_name)
    print("=" * 100)
    print(
        f"{'scope':<18}{'n':>8}{'% White':>10}{'% tie':>9}{'% Black':>10}"
        f"{'tie-aware W':>14}{'mean W-B':>14}{'95% CI mean':>25}"
    )
    scopes = [("overall", summary["overall"])] + [
        (g, summary["by_gender"][g]) for g in ("female", "male")
    ]
    for label, d in scopes:
        ci = d["bootstrap"]["mean_delta_white_minus_black_ci"]
        print(
            f"{label:<18}{d['n_comparisons']:>8,d}{d['pct_white_higher']:>10.1f}"
            f"{d['pct_tie']:>9.1f}{d['pct_black_higher']:>10.1f}"
            f"{d['tie_aware_pct_white']:>14.1f}"
            f"{d['mean_delta_white_minus_black']:>14.6f}"
            f"  [{ci[0]:.6f}, {ci[1]:.6f}]"
        )
    c = summary["directional_consistency"]
    print(
        f"name-pair mean direction: White-associated {c['name_pairs_white_positive']}/"
        f"{c['name_pairs_total']}, Black-associated {c['name_pairs_black_positive']}/"
        f"{c['name_pairs_total']}"
    )
    print(
        f"occupation mean direction: White-associated {c['occupations_white_positive']}/"
        f"{c['occupations_total']}, Black-associated {c['occupations_black_positive']}/"
        f"{c['occupations_total']}"
    )
    print("\nPer-name-pair mean differences (White minus Black):")
    for pair in NAME_PAIRS:
        d = summary["by_name_pair"][pair.pair_id]
        print(
            f"  {pair.pair_id} {pair.white_first:10} - {pair.black_first:10}: "
            f"mean={d['mean_delta_white_minus_black']:+.6f}, "
            f"tie-aware White={d['tie_aware_pct_white']:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="cached local reranker (repeatable; default: two prespecified models)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse a per-model output only when its design and script hash match",
    )
    args = parser.parse_args()

    import torch

    models = args.model or list(DEFAULT_MODELS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenization = validate_design(models)
    rows = generate_rows()
    print(
        f"Design locked: {len(NAME_PAIRS)} pairs / {len(rows):,} comparisons per model / "
        f"{len(rows) * 2:,} score inputs per model"
    )
    print(f"Occupations: {len(ALL_OCCUPATIONS)}; label source: {LABEL_SOURCE}")
    print(f"Device: {device}; models: {', '.join(models)}")

    design = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256(),
        "seed": SEED,
        "status": "exploratory post-hoc extension; selection fixed before scores were inspected",
        "causal_contrast": (
            "within each comparison only the first name changes; surname, pronouns, "
            "qualifications, occupation, template, and query are identical"
        ),
        "interpretation": (
            "sensitivity to Black-associated versus White-associated name strings; "
            "not observed candidate race and not a race-only causal effect"
        ),
        "surname": SURNAME,
        "sources": SOURCE,
        "selection_rule": (
            "Within gender: published racial distinctiveness >= 0.75, equal subword "
            "counts under both audited rerankers, then five disjoint pairs minimizing "
            "total absolute DOLMA log-frequency distance; male minimum-cost tie broken "
            "by larger sum of within-pair minimum distinctiveness."
        ),
        "name_pairs": [
            asdict(p)
            | {
                "black_full": p.black_name,
                "white_full": p.white_name,
                "absolute_log_frequency_gap": p.log_frequency_gap,
            }
            for p in NAME_PAIRS
        ],
        "tokenization": tokenization,
        "n_occupations": len(ALL_OCCUPATIONS),
        "n_templates": len(TEMPLATES),
        "n_query_types": len(QUERY_TYPES),
        "n_comparisons_per_model": len(rows),
        "n_score_inputs_per_model": 2 * len(rows),
        "label_source": LABEL_SOURCE,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    all_models: dict[str, dict] = {}
    for model_name in models:
        output_path = RESULTS / f"racialized_name_pilot_{slug(model_name)}.json"
        result = None
        if args.reuse and output_path.exists():
            previous = json.loads(output_path.read_text())
            if (
                previous.get("design", {}).get("script_sha256") == design["script_sha256"]
                and previous.get("model_metadata", {}).get("model") == model_name
                and previous.get("design", {}).get("n_comparisons_per_model") == len(rows)
            ):
                print(f"Reusing {output_path.name}")
                result = previous

        if result is None:
            print(f"\nLoading and scoring {model_name} ...", flush=True)
            raw = score_rows(model_name, rows, device, args.batch_size)
            summary = analyze(raw, args.bootstrap)
            result = {
                "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "design": design,
                "model_metadata": model_metadata(model_name, device),
                "bootstrap_B": args.bootstrap,
                "summary": summary,
                "raw_results": raw,
            }
            output_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Saved -> {output_path}")
        print_summary(model_name, result["summary"])
        all_models[model_name] = {
            "model_metadata": result["model_metadata"],
            "summary": result["summary"],
            "per_model_file": str(output_path.relative_to(ROOT)),
        }

    combined = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "design": design,
        "bootstrap_B": args.bootstrap,
        "models": all_models,
    }
    combined_path = RESULTS / "racialized_name_pilot_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"\nSaved -> {combined_path}")


if __name__ == "__main__":
    main()
