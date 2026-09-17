"""
Vendor-documentation audit for the re-rankers scored in this paper.

The empirical sections show that re-rankers score identical men and women
differently, and that the direction of the lean is a property of the individual
product.  Section 4.2 of the paper draws the legal consequence: a deployer
cannot know which sex a tool disadvantages without testing it.  That claim needs
evidence about what a deployer could learn *without* testing -- that is, from the
vendor's own documentation.  This script supplies it.

The population is closed and is not chosen by us: it is exactly the models in
`results/single_stage_summary.json`, the 14 re-rankers the paper already scores.
There is therefore no question of which vendors were sampled, and no vendor can
be said to have been singled out.

Four questions are coded per vendor in `code/vendor_docs_audit.json`, from
first-party sources only (model card, product documentation, technical report,
terms or usage policy), each with a URL, an access date, and a verbatim quote or
an explicit finding of absence:

  Q1  Is hiring / recruitment / resume screening named as an intended use?
  Q2  Is any fairness or bias evaluation of the RANKING model published?
  Q3  Is any gender-related limitation disclosed?
  Q4  Is there a use restriction covering high-stakes or employment decisions?

Q4 is recorded together with two further facts, because they carry the legal
weight: whether the restriction lives in the model's own documentation or in a
platform-wide policy, and whether the model documentation links to it.  A
restriction a deployer would never encounter while reading the product docs is a
different thing from a warning on the product itself -- and under Art.\\ 26(1) of
the EU AI Act, which obliges a deployer to use a system in accordance with its
instructions for use, the difference is the whole question.

This file computes; it never asserts.  Every count in the paper comes from the
evidence file, and a model that is scored but not audited (or audited but not
scored) fails the run rather than being silently dropped.

Outputs:
    results/vendor_docs_audit_summary.json
    results/tex/vendor_docs_audit.tex

Run:  venv/bin/python code/vendor_docs_audit.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIT_FILE = ROOT / "code" / "vendor_docs_audit.json"
SUMMARY_FILE = ROOT / "results" / "single_stage_summary.json"
OUT_JSON = ROOT / "results" / "vendor_docs_audit_summary.json"
TEX_DIR = ROOT / "results" / "tex"

SCRIPT_VERSION = 2

# The four coded questions, in the order they are reported.
QUESTIONS = [
    ("q1_hiring_named", "Hiring named as an intended use"),
    ("q2_fairness_eval", "Fairness evaluation of the ranking model"),
    ("q3_gender_disclosure", "Gender-related limitation disclosed"),
    ("q4_highstakes_limit", "Restriction on high-stakes / employment use"),
]


def _esc(s: str) -> str:
    """Escape the LaTeX specials that occur in vendor and model names."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def load() -> tuple[dict, dict]:
    audit = json.loads(AUDIT_FILE.read_text())
    summary = json.loads(SUMMARY_FILE.read_text())
    return audit, summary


def check_population(audit: dict, summary: dict) -> dict[str, list[str]]:
    """Join the audit to the scored models, failing loudly on any mismatch.

    The join is on the vendor string recorded in single_stage_summary.json, so
    the audit cannot drift out of step with the model set: adding a re-ranker to
    the paper without auditing its vendor breaks this run.
    """
    audited = set(audit["vendors"])
    by_vendor: dict[str, list[str]] = defaultdict(list)
    for name, entry in summary["models"].items():
        by_vendor[entry["vendor"]].append(name)

    scored = set(by_vendor)
    if scored - audited:
        sys.exit(f"scored but not audited: {sorted(scored - audited)}")
    if audited - scored:
        sys.exit(f"audited but not scored: {sorted(audited - scored)}")

    n_models = sum(len(v) for v in by_vendor.values())
    if n_models != summary["n_models"]:
        sys.exit(f"model count mismatch: joined {n_models}, summary says "
                 f"{summary['n_models']}")
    weights = audit["weight_availability"]
    if set(weights) != set(summary["models"]):
        raise ValueError("Weight-availability evidence must cover exactly the scored models")
    for model, entry in weights.items():
        if type(entry.get("public_weights")) is not bool or not entry.get("source") or not entry.get("checked"):
            raise ValueError(f"Incomplete weight-availability evidence: {model}")
    return {v: sorted(m) for v, m in sorted(by_vendor.items())}


def tally(audit: dict, by_vendor: dict[str, list[str]]) -> dict:
    """Count vendors and models by answer, for each of the four questions."""
    out: dict[str, dict] = {}
    for key, label in QUESTIONS:
        vendors = Counter()
        models = Counter()
        for vendor, model_names in by_vendor.items():
            answer = audit["vendors"][vendor][key]
            vendors[answer] += 1
            models[answer] += len(model_names)
        out[key] = {
            "label": label,
            "vendors": dict(vendors),
            "models": dict(models),
        }
    return out


def main() -> None:
    audit, summary = load()
    by_vendor = check_population(audit, summary)
    counts = tally(audit, by_vendor)

    n_vendors = len(by_vendor)
    n_models = sum(len(v) for v in by_vendor.values())

    # The headline facts, all derived.  "yes_other_model" is deliberately NOT
    # folded into "yes": Cohere publishes a gender-bias evaluation for its
    # generative model and none for its re-ranker, which is a finding about the
    # ranking layer specifically, not an absence of effort by the vendor.
    def n_models_with(key: str, *answers: str) -> int:
        return sum(counts[key]["models"].get(a, 0) for a in answers)

    def n_vendors_with(key: str, *answers: str) -> int:
        return sum(counts[key]["vendors"].get(a, 0) for a in answers)

    # Of the vendors that DO restrict employment use, how many state that
    # restriction in the model's own documentation, and how many link to it?
    restricting = [v for v in by_vendor
                   if audit["vendors"][v]["q4_highstakes_limit"] == "yes"]
    in_product_doc = [v for v in restricting
                      if audit["vendors"][v]["q4_scope"] == "product_doc"]
    linked = [v for v in restricting
              if audit["vendors"][v]["q4_linked_from_model_doc"]]

    n_sources = sum(len(audit["vendors"][v]["sources"]) for v in by_vendor)

    # Access route is provenance, not evidence about weight availability.
    n_no_public_weights = sum(not e["public_weights"]
                              for e in audit["weight_availability"].values())

    headline = {
        "n_models": n_models,
        "n_vendors": n_vendors,
        "n_sources_checked": n_sources,
        "models_without_public_weights": n_no_public_weights,
        "models_accessed_via_api": sum(e["access"] == "API" for e in summary["models"].values()),
        "models_hiring_named": n_models_with("q1_hiring_named", "yes"),
        "models_fairness_eval": n_models_with("q2_fairness_eval", "yes"),
        "models_gender_disclosure": n_models_with("q3_gender_disclosure", "yes"),
        "models_highstakes_restricted": n_models_with("q4_highstakes_limit", "yes"),
        "vendors_highstakes_restricted": n_vendors_with("q4_highstakes_limit", "yes"),
        "models_no_restriction": n_models_with("q4_highstakes_limit", "no"),
        "vendors_restricting_in_product_doc": len(in_product_doc),
        "vendors_restricting_linked_from_model_doc": len(linked),
        "vendors_fairness_eval_other_model_only":
            n_vendors_with("q2_fairness_eval", "yes_other_model"),
        "vendors_with_limitations_section":
            sum(1 for v in by_vendor
                if audit["vendors"][v]["has_limitations_section"]),
    }

    result = {
        "script_version": SCRIPT_VERSION,
        "audit_version": audit["audit_version"],
        "accessed": audit["accessed"],
        "population_source": str(SUMMARY_FILE.relative_to(ROOT)),
        "headline": headline,
        "counts_by_question": counts,
        "models_by_vendor": by_vendor,
        "weight_availability": audit["weight_availability"],
        "vendors": {
            v: {k: audit["vendors"][v][k] for k in
                ("license", "q1_hiring_named", "q2_fairness_eval",
                 "q3_gender_disclosure", "q4_highstakes_limit", "q4_scope",
                 "q4_linked_from_model_doc", "has_limitations_section")}
            | {"n_models": len(by_vendor[v]),
               "n_sources": len(audit["vendors"][v]["sources"])}
            for v in by_vendor
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    # --- appendix table -----------------------------------------------------
    # One row per vendor, with the model count, so 14 models compress to 7 rows
    # without hiding that Voyage contributes five of them.
    glyph = {"yes": r"yes", "no": r"--", "yes_other_model": r"other model"}
    lines = [
        "% Generated by code/vendor_docs_audit.py",
        r"\begin{tabular}{lcccccl}", r"\toprule",
        r"Vendor & Models & Hiring & Fairness & Gender & High-stakes & Where stated \\",
        r"\midrule",
    ]
    for vendor in by_vendor:
        e = audit["vendors"][vendor]
        if e["q4_highstakes_limit"] == "yes":
            where = ("product documentation" if e["q4_scope"] == "product_doc"
                     else ("platform policy; navigation link" if e["q4_linked_from_model_doc"]
                           else "platform policy; no direct link found"))
        else:
            where = "--"
        lines.append(
            f"{_esc(vendor)} & {len(by_vendor[vendor])} & "
            f"{glyph[e['q1_hiring_named']]} & {glyph[e['q2_fairness_eval']]} & "
            f"{glyph[e['q3_gender_disclosure']]} & "
            f"{glyph[e['q4_highstakes_limit']]} & {where} \\\\"
        )
    lines += [
        r"\midrule",
        f"Total & {n_models} & {headline['models_hiring_named']} & "
        f"{headline['models_fairness_eval']} & "
        f"{headline['models_gender_disclosure']} & "
        f"{headline['models_highstakes_restricted']} & \\\\",
        r"\bottomrule", r"\end{tabular}",
    ]
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    (TEX_DIR / "vendor_docs_audit.tex").write_text("\n".join(lines) + "\n")

    print(f"{n_models} models, {n_vendors} vendors, "
          f"{n_sources} first-party sources, accessed {audit['accessed']}")
    for key, label in QUESTIONS:
        c = counts[key]
        parts = ", ".join(f"{a}: {n} models" for a, n in sorted(c["models"].items()))
        print(f"  {label:<45} {parts}")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {(TEX_DIR / 'vendor_docs_audit.tex').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
