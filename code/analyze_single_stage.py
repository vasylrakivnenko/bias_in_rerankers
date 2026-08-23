"""
Single-stage (re-ranker only) analysis of the banked counterfactual scores.

Reads every `results/*_full_raw.json` (paid API scores -- NEVER rewritten by this
script) and produces the corrected numbers the paper needs:

  A1  Duplicate-model detection.  `semantic-ranker-default-003` returns scores
      byte-identical to `-004` (Google serves the deprecated name with the -004
      model).  The duplicate group is found PROGRAMMATICALLY by hashing each
      model's canonicalised (occupation, query_type, name_pair, template) ->
      (score_male, score_female) map; the deprecated member of each group is
      dropped and the exclusion is asserted, so a *new* duplicate cannot slip
      through unnoticed.  Result: 14 distinct models, not 15.

  A2  Ties.  `delta > 0` = male and "everything else" = female miscounts ties
      (up to ~21% of pairs for rerank-2-lite) as favouring women.  Every
      percentage here is reported as a three-way split (male / tie / female)
      plus a tie-aware statistic  (wins_M + 0.5*ties) / n.  The
      stereotype-match rate (formerly "BDI") is made tie-aware the same way.

  A3  Neutral-role summary reports the MEDIAN across models and the count of
      models leaning each way, not only the outlier-driven mean.

  B1  Cluster bootstrap by occupation (B=1000, seeded) for every percentage --
      the 13,120 pairs nest in 82 occupations x 10 names x 4 templates x 4
      queries, so naive binomial CIs are far too narrow.  Also per model:
      "k of N male-stereotyped occupations lean male" / "k of N female-
      stereotyped lean female", which is far more convincing than a pooled %.

  B2  Breakdowns by template, query phrasing and name pair (heterogeneity is
      large and currently hidden).

  B7  Model provenance table: vendor, access type, score range, tie rate, n,
      date scored (from raw-file mtimes).

Outputs (never overwrites any *_full_raw.json):
    results/single_stage_summary.json
    results/tex/single_stage_main.tex
    results/tex/provenance.tex
    results/tex/occupation_consistency.tex
    results/tex/robustness_template.tex
    results/tex/robustness_query.tex
    results/tex/robustness_namepair.tex

Run:  venv/bin/python code/analyze_single_stage.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from labels import load_labels  # noqa: E402  (shared contract module)

SCRIPT_VERSION = "analyze_single_stage.py v1.0 (REVIEW_TODO A1/A2/A3/B1/B2/B7)"
RESULTS_DIR = ROOT / "results"
TEX_DIR = RESULTS_DIR / "tex"
OUT_JSON = RESULTS_DIR / "single_stage_summary.json"
SEED = 20260822
B_BOOT = 1000

CATEGORIES = ("male", "female", "neutral")

# ---------------------------------------------------------------------------
# B7: provenance.  Keyed by the `model` string inside each raw JSON.
# ---------------------------------------------------------------------------
PROVENANCE = {
    "rerank-1": ("Voyage AI", "API"),
    "rerank-2": ("Voyage AI", "API"),
    "rerank-2-lite": ("Voyage AI", "API"),
    "rerank-2.5": ("Voyage AI", "API"),
    "rerank-2.5-lite": ("Voyage AI", "API"),
    "Cohere-rerank-v4.0-pro": ("Cohere", "API"),
    "Cohere-rerank-v4.0-fast": ("Cohere", "API"),
    "semantic-ranker-default-002": ("Google Vertex AI Ranking", "API"),
    "semantic-ranker-default-003": ("Google Vertex AI Ranking", "API"),
    "semantic-ranker-default-004": ("Google Vertex AI Ranking", "API"),
    "semantic-ranker-fast-004": ("Google Vertex AI Ranking", "API"),
    "ibm-granite-granite-embedding-reranker-english-r2": ("IBM (via Azure AI Foundry)", "API"),
    "BAAI/bge-reranker-v2-m3": ("BAAI (open weights)", "local"),
    "cross-encoder/ms-marco-MiniLM-L-12-v2": ("Microsoft / ms-marco (open weights)", "local"),
    "jinaai/jina-reranker-v2-base-multilingual": ("Jina AI (open weights)", "local"),
}

# Deprecated aliases: if a model is found to be score-identical to another, the
# member listed here is the one dropped (Google documents -003 as superseded).
DEPRECATED_ALIASES = {"semantic-ranker-default-003": "semantic-ranker-default-004"}


# ---------------------------------------------------------------------------
# Loading + A1 duplicate detection
# ---------------------------------------------------------------------------

def _row_key(r: dict) -> tuple:
    return (r["occupation"], r["query_type"], r["name_pair"], r["template_style"])


def load_raw() -> dict[str, dict]:
    """model_name -> {"rows": [...], "path": str, "mtime": date}"""
    out = {}
    for path in sorted(glob.glob(str(RESULTS_DIR / "*_full_raw.json"))):
        d = json.load(open(path))
        name = d["model"]
        if name in out:
            raise RuntimeError(f"two raw files claim model {name!r}")
        out[name] = {
            "rows": d["results"],
            "path": path,
            "mtime": _dt.datetime.fromtimestamp(os.stat(path).st_mtime),
        }
    return out


def score_fingerprint(rows: list[dict]) -> str:
    """Order-independent hash of the full (key -> score_male, score_female) map."""
    items = sorted((_row_key(r), r["score_male"], r["score_female"]) for r in rows)
    h = hashlib.sha256()
    for k, sm, sf in items:
        h.update(repr(k).encode())
        h.update(repr((sm, sf)).encode())
    return h.hexdigest()


def detect_duplicates(raw: dict[str, dict]) -> tuple[list[list[str]], list[str], dict]:
    """Return (duplicate_groups, dropped_models, evidence)."""
    by_fp = defaultdict(list)
    for name, d in raw.items():
        by_fp[score_fingerprint(d["rows"])].append(name)

    groups = [sorted(v) for v in by_fp.values() if len(v) > 1]
    dropped, evidence = [], {}
    for g in groups:
        # Row-level evidence, recomputed rather than asserted.
        a, b = g[0], g[1]
        amap = {_row_key(r): (r["score_male"], r["score_female"]) for r in raw[a]["rows"]}
        identical = sum(
            1 for r in raw[b]["rows"] if amap.get(_row_key(r)) == (r["score_male"], r["score_female"])
        )
        evidence["+".join(g)] = {
            "n_rows": len(raw[b]["rows"]),
            "n_identical_rows": identical,
            "n_distinct_score_values": len({s for r in raw[a]["rows"]
                                            for s in (r["score_male"], r["score_female"])}),
        }
        keep = None
        for m in g:
            if m in DEPRECATED_ALIASES and DEPRECATED_ALIASES[m] in g:
                dropped.append(m)
            else:
                keep = m
        if keep is None or len([m for m in g if m not in dropped]) != 1:
            raise AssertionError(
                f"duplicate score group {g} has no single documented survivor; "
                "extend DEPRECATED_ALIASES after checking the vendor docs "
                "(REVIEW_TODO.md A1)."
            )
    return groups, sorted(dropped), evidence


# ---------------------------------------------------------------------------
# A2 + B1: tie-aware statistics with a cluster bootstrap by occupation
# ---------------------------------------------------------------------------

class ClusterStat:
    """Per-occupation aggregate counts -> exact bootstrap over occupation clusters.

    Every statistic here is a ratio of sums over rows, so resampling occupations
    and re-summing their PRE-AGGREGATED counts is identical to resampling and
    re-scanning the rows, and is ~1000x faster.
    """

    def __init__(self):
        self._n = defaultdict(int)      # occupation -> rows
        self._m = defaultdict(int)      # occupation -> delta > 0
        self._t = defaultdict(int)      # occupation -> delta == 0
        self._c = defaultdict(int)      # occupation -> stereotype-consistent (non-tie)
        self._s = defaultdict(int)      # occupation -> rows with a stereotype direction
        self._st = defaultdict(int)     # occupation -> ties among stereotyped rows

    def add(self, occ: str, delta: float, stereotype: str) -> None:
        self._n[occ] += 1
        if delta > 0:
            self._m[occ] += 1
        elif delta == 0:
            self._t[occ] += 1
        if stereotype in ("male", "female"):
            self._s[occ] += 1
            if delta == 0:
                self._st[occ] += 1
            elif (stereotype == "male" and delta > 0) or (stereotype == "female" and delta < 0):
                self._c[occ] += 1

    @property
    def n(self) -> int:
        return sum(self._n.values())

    @property
    def n_occupations(self) -> int:
        return len(self._n)

    def _arrays(self):
        occs = sorted(self._n)
        a = lambda d: np.array([d[o] for o in occs], dtype=np.float64)  # noqa: E731
        return occs, a(self._n), a(self._m), a(self._t), a(self._c), a(self._s), a(self._st)

    def point(self) -> dict:
        n, m, t = self.n, sum(self._m.values()), sum(self._t.values())
        if n == 0:
            return {}
        s, c, st = sum(self._s.values()), sum(self._c.values()), sum(self._st.values())
        out = {
            "n": n,
            "n_occupations": self.n_occupations,
            "pct_male": 100.0 * m / n,
            "pct_tie": 100.0 * t / n,
            "pct_female": 100.0 * (n - m - t) / n,
            "tie_aware_pct_male": 100.0 * (m + 0.5 * t) / n,
        }
        if s:
            out["n_stereotyped"] = s
            out["stereotype_match_rate"] = 100.0 * c / s
            out["tie_aware_stereotype_match_rate"] = 100.0 * (c + 0.5 * st) / s
        return out

    def bootstrap(self, seed: int, B: int = B_BOOT) -> dict:
        occs, n, m, t, c, s, st = self._arrays()
        K = len(occs)
        if K == 0:
            return {}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, K, size=(B, K))
        N = n[idx].sum(1)
        M = m[idx].sum(1)
        T = t[idx].sum(1)
        ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
        out = {
            "pct_male_ci": ci(100.0 * M / N),
            "pct_tie_ci": ci(100.0 * T / N),
            "pct_female_ci": ci(100.0 * (N - M - T) / N),
            "tie_aware_pct_male_ci": ci(100.0 * (M + 0.5 * T) / N),
        }
        S = s[idx].sum(1)
        if S.sum() > 0:
            C = c[idx].sum(1)
            ST = st[idx].sum(1)
            ok = S > 0
            out["stereotype_match_rate_ci"] = ci(100.0 * C[ok] / S[ok])
            out["tie_aware_stereotype_match_rate_ci"] = ci(100.0 * (C[ok] + 0.5 * ST[ok]) / S[ok])
        return out

    def summary(self, seed: int, B: int = B_BOOT) -> dict:
        out = self.point()
        if out:
            out.update(self.bootstrap(seed, B))
        return out

    def per_occupation(self) -> dict:
        """occupation -> tie-aware % male + the three-way split (for B1 counts)."""
        res = {}
        for o in sorted(self._n):
            n, m, t = self._n[o], self._m[o], self._t[o]
            res[o] = {
                "n": n,
                "pct_male": 100.0 * m / n,
                "pct_tie": 100.0 * t / n,
                "pct_female": 100.0 * (n - m - t) / n,
                "tie_aware_pct_male": 100.0 * (m + 0.5 * t) / n,
            }
        return res


def build_stat(rows) -> ClusterStat:
    cs = ClusterStat()
    for r in rows:
        cs.add(r["occupation"], r["delta"], r["stereotype"])
    return cs


def stable_seed(*parts) -> int:
    """Deterministic per-group seed so a model's CI does not depend on iteration order."""
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
    return (SEED + int(h[:12], 16)) % (2 ** 32 - 1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_model(name: str, rows: list[dict], meta: dict, boot: int) -> dict:
    vendor, access = PROVENANCE.get(name, ("unknown", "unknown"))
    scores = [s for r in rows for s in (r["score_male"], r["score_female"])]
    ties = sum(1 for r in rows if r["delta"] == 0)

    entry = {
        "model": name,
        "vendor": vendor,
        "access": access,
        "n_pairs": len(rows),
        "score_min": min(scores),
        "score_max": max(scores),
        "n_distinct_scores": len(set(scores)),
        "tie_rate_pct": 100.0 * ties / len(rows),
        "date_scored": meta["mtime"].date().isoformat(),
        "raw_file": os.path.basename(meta["path"]),
        "overall": build_stat(rows).summary(stable_seed(name, "overall"), boot),
        "by_category": {},
    }

    for cat in CATEGORIES:
        sub = [r for r in rows if r["stereotype"] == cat]
        if sub:
            entry["by_category"][cat] = build_stat(sub).summary(stable_seed(name, cat), boot)

    # B1: occupation-level consistency counts (far more convincing than a pooled %).
    consistency = {}
    for cat in ("male", "female"):
        sub = [r for r in rows if r["stereotype"] == cat]
        if not sub:
            continue
        per_occ = build_stat(sub).per_occupation()
        want_male = cat == "male"
        k = sum(1 for v in per_occ.values()
                if (v["tie_aware_pct_male"] > 50.0) == want_male and v["tie_aware_pct_male"] != 50.0)
        exact = sum(1 for v in per_occ.values() if v["tie_aware_pct_male"] == 50.0)
        consistency[cat] = {
            "n_occupations": len(per_occ),
            "k_leaning_stereotype_direction": k,
            "k_exactly_balanced": exact,
            "per_occupation": per_occ,
        }
    # Neutral: how many neutral occupations lean male at all (A3 at occupation level).
    sub = [r for r in rows if r["stereotype"] == "neutral"]
    if sub:
        per_occ = build_stat(sub).per_occupation()
        consistency["neutral"] = {
            "n_occupations": len(per_occ),
            "k_leaning_male": sum(1 for v in per_occ.values() if v["tie_aware_pct_male"] > 50.0),
            "k_exactly_balanced": sum(1 for v in per_occ.values() if v["tie_aware_pct_male"] == 50.0),
            "per_occupation": per_occ,
        }
    entry["occupation_consistency"] = consistency

    # B2: heterogeneity breakdowns.
    for field, out_key in (("template_style", "by_template"),
                           ("query_type", "by_query_type"),
                           ("name_pair", "by_name_pair")):
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        entry[out_key] = {
            g: build_stat(rs).summary(stable_seed(name, field, g), boot)
            for g, rs in sorted(groups.items())
        }
    return entry


# ---------------------------------------------------------------------------
# A8 follow-up: does the BLS relabelling change the picture?
# ---------------------------------------------------------------------------

def _legacy_labels() -> dict[str, str]:
    """The pre-BLS hand/embedding labels, kept in synthetic_dataset for transparency."""
    import synthetic_dataset as sd
    src = getattr(sd, "LEGACY_OCCUPATIONS", None) or sd.OCCUPATIONS
    return {occ: stereo for stereo, occs in src.items() for occ in occs}


def _cat_pcts(rows, labelmap) -> dict:
    stats = {c: ClusterStat() for c in CATEGORIES}
    for r in rows:
        lab = labelmap.get(r["occupation"])
        if lab in stats:
            stats[lab].add(r["occupation"], r["delta"], lab)
    out = {}
    for c in CATEGORIES:
        p = stats[c].point()
        if p:
            out[c] = {k: p[k] for k in ("n", "n_occupations", "pct_male", "pct_tie",
                                        "pct_female", "tie_aware_pct_male")}
    return out


def label_set_comparison(raw: dict, primary: dict[str, str], primary_source: str) -> dict:
    """Per-model per-category % male-favoured under three label views.

    legacy_all82   the pre-BLS labels on every occupation (what the paper reported)
    legacy_kept    the pre-BLS labels restricted to the occupations BLS keeps
                   (isolates the RELABELLING from the change of sample)
    primary        whatever load_labels() returned for this run
    """
    legacy = _legacy_labels()
    legacy_kept = {o: l for o, l in legacy.items() if o in primary}
    views = {"legacy_all82": legacy, "legacy_kept": legacy_kept, "primary": primary}

    changed = sorted(o for o in set(legacy) & set(primary) if legacy[o] != primary[o])
    out = {
        "primary_source": primary_source,
        "n_occupations": {k: len(v) for k, v in views.items()},
        "label_counts": {k: {c: sum(1 for x in v.values() if x == c) for c in CATEGORIES}
                         for k, v in views.items()},
        "occupations_dropped_by_bls": sorted(set(legacy) - set(primary)),
        "occupations_relabelled": [{"occupation": o, "legacy": legacy[o], "primary": primary[o]}
                                   for o in changed],
        "models": {},
    }
    for name, meta in sorted(raw.items()):
        out["models"][name] = {k: _cat_pcts(meta["rows"], v) for k, v in views.items()}

    # spread = male-jobs % minus female-jobs % (tie-aware), per view
    out["spread_by_view"] = {}
    for k in views:
        sp = {m: v[k]["male"]["tie_aware_pct_male"] - v[k]["female"]["tie_aware_pct_male"]
              for m, v in out["models"].items()
              if "male" in v[k] and "female" in v[k]}
        out["spread_by_view"][k] = {
            "per_model": sp, "median": statistics.median(sp.values()),
            "mean": statistics.mean(sp.values()),
            "min": min(sp.values()), "max": max(sp.values()),
            "n_models_positive": sum(1 for x in sp.values() if x > 0), "n_models": len(sp),
        }
    return out


# ---------------------------------------------------------------------------
# LaTeX fragments
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _ci(d: dict, key: str) -> str:
    v = d.get(key)
    return f"[{v[0]:.1f}, {v[1]:.1f}]" if v else "--"


def write_tex(summary: dict) -> list[str]:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    models = summary["models"]
    order = sorted(models, key=lambda m: -models[m]["by_category"]["male"]["tie_aware_pct_male"])
    written = []

    # --- Table A1 replacement: three-way split + tie-aware % with cluster CIs ---
    lines = [
        "% Generated by code/analyze_single_stage.py -- do not edit by hand.",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model (vendor) & Category & \%\,M & \%\,tie & \%\,F & Tie-aware \%\,M [95\% CI] \\",
        r"\midrule",
    ]
    for m in order:
        e = models[m]
        for i, cat in enumerate(CATEGORIES):
            c = e["by_category"][cat]
            label = f"{_esc(m)} ({_esc(e['vendor'])})" if i == 0 else ""
            lines.append(
                f"{label} & {cat} & {c['pct_male']:.1f} & {c['pct_tie']:.1f} & "
                f"{c['pct_female']:.1f} & {c['tie_aware_pct_male']:.1f} "
                f"{_ci(c, 'tie_aware_pct_male_ci')} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX_DIR / "single_stage_main.tex").write_text("\n".join(lines) + "\n")
    written.append(str(TEX_DIR / "single_stage_main.tex"))

    # --- B7 provenance ---
    lines = [
        "% Generated by code/analyze_single_stage.py",
        r"\begin{tabular}{llrrrl}", r"\toprule",
        r"Model & Vendor / access & $n$ & Score range & Ties \% & Scored \\",
        r"\midrule",
    ]
    for m in order:
        e = models[m]
        lines.append(
            f"{_esc(m)} & {_esc(e['vendor'])} ({e['access']}) & {e['n_pairs']:,} & "
            f"{e['score_min']:.3f}--{e['score_max']:.3f} & {e['tie_rate_pct']:.1f} & "
            f"{e['date_scored']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX_DIR / "provenance.tex").write_text("\n".join(lines) + "\n")
    written.append(str(TEX_DIR / "provenance.tex"))

    # --- B1 occupation-level consistency ---
    lines = [
        "% Generated by code/analyze_single_stage.py",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & male-stereo.\ leaning male & female-stereo.\ leaning female & neutral leaning male \\",
        r"\midrule",
    ]
    for m in order:
        oc = models[m]["occupation_consistency"]
        lines.append(
            f"{_esc(m)} & {oc['male']['k_leaning_stereotype_direction']}/{oc['male']['n_occupations']}"
            f" & {oc['female']['k_leaning_stereotype_direction']}/{oc['female']['n_occupations']}"
            f" & {oc['neutral']['k_leaning_male']}/{oc['neutral']['n_occupations']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX_DIR / "occupation_consistency.tex").write_text("\n".join(lines) + "\n")
    written.append(str(TEX_DIR / "occupation_consistency.tex"))

    # --- B2 robustness tables ---
    for key, fname, head in (("by_template", "robustness_template.tex", "Template"),
                             ("by_query_type", "robustness_query.tex", "Query phrasing"),
                             ("by_name_pair", "robustness_namepair.tex", "Name pair")):
        groups = sorted({g for m in order for g in models[m][key]})
        lines = [
            "% Generated by code/analyze_single_stage.py -- tie-aware \\% male-favoured.",
            r"\begin{tabular}{l" + "r" * len(groups) + "}", r"\toprule",
            "Model & " + " & ".join(_esc(g) for g in groups) + r" \\", r"\midrule",
        ]
        for m in order:
            cells = [f"{models[m][key][g]['tie_aware_pct_male']:.1f}" if g in models[m][key] else "--"
                     for g in groups]
            lines.append(f"{_esc(m)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TEX_DIR / fname).write_text("\n".join(lines) + "\n")
        written.append(str(TEX_DIR / fname))
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bootstrap", type=int, default=B_BOOT)
    ap.add_argument("--strict-labels", action="store_true",
                    help="fail instead of falling back to the pre-BLS labels")
    args = ap.parse_args()

    labels, label_source = load_labels(strict=args.strict_labels)
    print(f"Label source: {label_source}  ({len(labels)} occupations, "
          f"{dict((c, sum(1 for v in labels.values() if v == c)) for c in CATEGORIES)})")

    raw = load_raw()
    print(f"Raw files loaded: {len(raw)}")

    # ---- A1 -------------------------------------------------------------
    groups, dropped, evidence = detect_duplicates(raw)
    print("\n--- A1 duplicate-model detection ---")
    if not groups:
        print("  no duplicate score sets found")
    for g in groups:
        ev = evidence["+".join(g)]
        print(f"  identical scores: {g}  "
              f"({ev['n_identical_rows']}/{ev['n_rows']} rows byte-identical, "
              f"{ev['n_distinct_score_values']} distinct score values)")
    assert dropped == ["semantic-ranker-default-003"], (
        f"expected to drop exactly ['semantic-ranker-default-003'], got {dropped}. "
        "A NEW duplicate model appeared -- investigate before publishing (A1)."
    )
    for d in dropped:
        del raw[d]
    print(f"  dropped: {dropped}  -> {len(raw)} distinct models")
    assert len(raw) == 14, f"expected 14 distinct models after A1, got {len(raw)}"

    # ---- per-model analysis --------------------------------------------
    models = {}
    dropped_occ_counts = {}
    for name, meta in sorted(raw.items()):
        rows = []
        n_dropped = 0
        for r in meta["rows"]:
            lab = labels.get(r["occupation"])
            if lab is None:
                n_dropped += 1
                continue
            rr = dict(r)
            rr["stereotype"] = lab
            rows.append(rr)
        dropped_occ_counts[name] = n_dropped
        models[name] = analyse_model(name, rows, meta, args.bootstrap)
        e = models[name]
        print(f"  {name:<52} n={e['n_pairs']:>6} ties={e['tie_rate_pct']:>5.1f}% "
              f"(dropped {n_dropped} rows w/o a label)")

    # ---- A3 cross-model summaries (all three categories) ----------------
    def across(d):
        v = list(d.values())
        return {
            "n_models": len(v),
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "min": min(v), "max": max(v),
            "n_models_above_50": sum(1 for x in v if x > 50.0),
            "n_models_below_50": sum(1 for x in v if x < 50.0),
            "n_models_at_50": sum(1 for x in v if x == 50.0),
            "per_model": d,
        }

    def across_category(cat):
        return {stat: across({m: e["by_category"][cat][stat] for m, e in models.items()
                              if cat in e["by_category"]})
                for stat in ("pct_male", "tie_aware_pct_male")}

    across_models = {cat: across_category(cat) for cat in CATEGORIES}
    a3 = across_models["neutral"]  # back-compat: `neutral_across_models` keeps its shape

    # Stereotype-tracking spread: male-jobs % minus female-jobs % (tie-aware).
    spread = {m: (e["by_category"]["male"]["tie_aware_pct_male"]
                  - e["by_category"]["female"]["tie_aware_pct_male"])
              for m, e in models.items()}
    # A stronger claim than "male% > female%": does the model order all THREE
    # categories male > neutral > female?  The paper must not say "all models put
    # the three categories in the stereotype order" unless this count says so.
    def _ta(m, c):
        return models[m]["by_category"][c]["tie_aware_pct_male"]

    full_order = {m: (_ta(m, "male") > _ta(m, "neutral") > _ta(m, "female")) for m in models}

    spread_summary = {
        "per_model": spread,
        "n_models_positive": sum(1 for v in spread.values() if v > 0),
        "n_models": len(spread),
        "min": min(spread.values()), "max": max(spread.values()),
        "median": statistics.median(spread.values()),
        "definition": "tie-aware % male-favoured on male-typed jobs minus the same on "
                      "female-typed jobs",
        "n_models_full_order": sum(full_order.values()),
        "full_order_exceptions": [
            {"model": m,
             "male": _ta(m, "male"), "neutral": _ta(m, "neutral"), "female": _ta(m, "female")}
            for m, ok in full_order.items() if not ok],
    }

    # ---- label-set comparison: BLS relabelling vs the legacy labels ------
    label_cmp = label_set_comparison(raw, labels, label_source)

    summary = {
        "script_version": SCRIPT_VERSION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "label_source": label_source,
        "n_labelled_occupations": len(labels),
        "label_counts": {c: sum(1 for v in labels.values() if v == c) for c in CATEGORIES},
        "seed": SEED,
        "bootstrap_B": args.bootstrap,
        "bootstrap_method": "cluster bootstrap resampling occupations with replacement",
        "n_models": len(models),
        "excluded_models": dropped,
        "duplicate_detection": {"groups": groups, "evidence": evidence},
        "rows_dropped_no_label": dropped_occ_counts,
        "neutral_across_models": a3,
        "across_models": across_models,
        "stereotype_tracking_spread": spread_summary,
        "label_set_comparison": label_cmp,
        "models": models,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    tex = write_tex(summary)

    # ---- console report -------------------------------------------------
    print("\n" + "=" * 108)
    print("A2/B1  THREE-WAY SPLIT PER CATEGORY  (M / tie / F), tie-aware %M with cluster-bootstrap 95% CI")
    print("=" * 108)
    order = sorted(models, key=lambda m: -models[m]["by_category"]["male"]["tie_aware_pct_male"])
    print(f"{'model':<50}{'cat':<8}{'%M':>7}{'%tie':>7}{'%F':>7}{'tie-aware %M':>15}{'95% CI':>18}")
    for m in order:
        for cat in CATEGORIES:
            c = models[m]["by_category"][cat]
            print(f"{m[:49]:<50}{cat:<8}{c['pct_male']:>7.1f}{c['pct_tie']:>7.1f}"
                  f"{c['pct_female']:>7.1f}{c['tie_aware_pct_male']:>15.1f}"
                  f"{_ci(c, 'tie_aware_pct_male_ci'):>18}")

    print("\n" + "=" * 108)
    print("A2/C6  TIE-AWARE STEREOTYPE-MATCH RATE (formerly BDI), male+female-stereotyped jobs")
    print("=" * 108)
    print(f"{'model':<50}{'naive':>10}{'tie-aware':>12}{'95% CI':>18}")
    for m in order:
        o = models[m]["overall"]
        print(f"{m[:49]:<50}{o['stereotype_match_rate']:>10.1f}"
              f"{o['tie_aware_stereotype_match_rate']:>12.1f}"
              f"{_ci(o, 'tie_aware_stereotype_match_rate_ci'):>18}")

    print("\n" + "=" * 108)
    print("B1  OCCUPATION-LEVEL CONSISTENCY (tie-aware lean per occupation)")
    print("=" * 108)
    print(f"{'model':<50}{'male-jobs lean M':>20}{'female-jobs lean F':>22}{'neutral lean M':>18}")
    for m in order:
        oc = models[m]["occupation_consistency"]
        print(f"{m[:49]:<50}"
              f"{oc['male']['k_leaning_stereotype_direction']:>13}/{oc['male']['n_occupations']:<6}"
              f"{oc['female']['k_leaning_stereotype_direction']:>15}/{oc['female']['n_occupations']:<6}"
              f"{oc['neutral']['k_leaning_male']:>11}/{oc['neutral']['n_occupations']:<6}")

    print("\n" + "=" * 108)
    print("A3  NEUTRAL OCCUPATIONS ACROSS MODELS")
    print("=" * 108)
    for key in ("pct_male", "tie_aware_pct_male"):
        a = a3[key]
        print(f"  {key:<22} n={a['n_models']}  mean={a['mean']:.1f}  median={a['median']:.1f}  "
              f"range=[{a['min']:.1f},{a['max']:.1f}]  "
              f">50: {a['n_models_above_50']}  <50: {a['n_models_below_50']}  =50: {a['n_models_at_50']}")
    print(f"  stereotype-tracking spread (male%-female%, tie-aware): "
          f"{spread_summary['n_models_positive']}/{spread_summary['n_models']} models positive, "
          f"median {spread_summary['median']:.1f}, range "
          f"[{spread_summary['min']:.1f}, {spread_summary['max']:.1f}]")

    print("\n" + "=" * 108)
    print("A8  LABEL-SET COMPARISON: pre-BLS labels vs BLS CPS Table 11 labels")
    print("=" * 108)
    lc = summary["label_set_comparison"]
    print(f"  occupations: " + "  ".join(f"{k}={v}" for k, v in lc["n_occupations"].items()))
    print(f"  dropped by BLS ({len(lc['occupations_dropped_by_bls'])}): "
          f"{', '.join(lc['occupations_dropped_by_bls'])}")
    print(f"  relabelled ({len(lc['occupations_relabelled'])}): "
          + ", ".join(f"{d['occupation']} {d['legacy']}->{d['primary']}"
                      for d in lc["occupations_relabelled"]))
    print(f"\n{'model':<50}" + "".join(f"{v[:14]:>36}" for v in ("legacy_all82", "legacy_kept", "primary"))
          + "\n" + " " * 50 + "".join(f"{'M / F / N %male':>36}" for _ in range(3)))
    for m in order:
        cells = []
        for v in ("legacy_all82", "legacy_kept", "primary"):
            d = lc["models"][m][v]
            cells.append(" / ".join(f"{d[c]['pct_male']:.1f}" for c in CATEGORIES))
        print(f"{m[:49]:<50}" + "".join(f"{c:>36}" for c in cells))
    print("\n  stereotype-tracking spread (tie-aware male% - female%), by label view:")
    for v, d in lc["spread_by_view"].items():
        print(f"    {v:<14} median {d['median']:6.1f}  mean {d['mean']:6.1f}  "
              f"range [{d['min']:.1f}, {d['max']:.1f}]  positive {d['n_models_positive']}/{d['n_models']}")

    print("\n" + "=" * 108)
    print("B2  HETEROGENEITY (tie-aware % male-favoured, min -> max across levels)")
    print("=" * 108)
    print(f"{'model':<50}{'template':>22}{'query':>22}{'name pair':>22}")
    for m in order:
        cells = []
        for key in ("by_template", "by_query_type", "by_name_pair"):
            vals = [v["tie_aware_pct_male"] for v in models[m][key].values()]
            cells.append(f"{min(vals):.1f} -> {max(vals):.1f}")
        print(f"{m[:49]:<50}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

    print("\n" + "=" * 108)
    print("B7  PROVENANCE")
    print("=" * 108)
    print(f"{'model':<50}{'vendor':<34}{'access':<8}{'n':>7}{'ties%':>7}{'range':>22}{'scored':>12}")
    for m in order:
        e = models[m]
        rng_s = f"{e['score_min']:.3f}..{e['score_max']:.3f}"
        print(f"{m[:49]:<50}{e['vendor'][:33]:<34}{e['access']:<8}{e['n_pairs']:>7}"
              f"{e['tie_rate_pct']:>7.1f}{rng_s:>22}{e['date_scored']:>12}")

    print(f"\nSaved -> {OUT_JSON}")
    for t in tex:
        print(f"Saved -> {t}")


if __name__ == "__main__":
    main()
