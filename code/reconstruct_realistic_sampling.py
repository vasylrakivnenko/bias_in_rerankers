"""Reconstruct the banked companion run's sampling without loading any model.

Run after rebuilding data/resumes/resumes.csv from data/PROVENANCE.md.
This only recovers sampling provenance; it does not reproduce model scores.
"""
import hashlib
import json
from pathlib import Path

from realistic_pool_experiment import load_corpus, load_queries, sample_pools

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "results/realistic_pool_four_rerankers.json"
OUT = ROOT / "results/realistic_pool_sampling.json"


def sampling_summary(pools):
    members = [(p["qid"], r.rid) for p in pools for r in p["members"]]
    return {
        "n_pools": len(pools),
        "n_focal_comparisons": len(members),
        "n_distinct_resumes_scored": len({rid for _, rid in members}),
        "n_distinct_query_resume_pairs": len(set(members)),
        "pool_membership_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode()).hexdigest(),
    }


def main():
    bank = json.loads(BANK.read_text())
    prov, config = bank["provenance"], bank["config"]
    resumes = load_corpus()
    queries, _ = load_queries(include_tasks=prov["queries"]["include_tasks"])
    # No membership filtering can be reconstructed without the original forms.
    # In this bank all corpus documents were usable; fail for any other design.
    if not (len(resumes) == prov["corpus"]["n_resumes"] == prov["n_resumes_usable"]):
        raise ValueError("Corpus changed or original run filtered documents; cannot reconstruct pools")
    if len(queries) != prov["queries"]["n_queries"]:
        raise ValueError("Query population differs from banked run")
    pools = sample_pools(queries, resumes, config["K"], config["pools_per_query"],
                         config["relevant_per_pool"], bank["seed"])
    summary = sampling_summary(pools)
    for name, model in bank["models"].items():
        if model["n_pairs_scored"] != 2 * summary["n_distinct_query_resume_pairs"]:
            raise ValueError(f"Scored-pair count differs for {name}")
        if model["summary"]["overall"]["n"] != summary["n_focal_comparisons"]:
            raise ValueError(f"Focal-comparison count differs for {name}")
    summary.update({
        "source": str(BANK.relative_to(ROOT)),
        "method": "Seeded sampling reconstructed from corpus and queries; banked counts cross-checked; no model rescoring.",
        "n_corpus_resumes": len(resumes),
        "seed": bank["seed"],
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (BANK, ROOT / "data/resumes/resumes.csv",
                                   ROOT / "data/category_to_soc.csv",
                                   ROOT / "data/onet/Occupation Data.txt")},
    })
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
