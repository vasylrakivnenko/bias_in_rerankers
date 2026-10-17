"""Regression checks for the September review corrections; no model loading."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import vendor_docs_audit as audit
from deid_transforms import deidentify
from reconstruct_realistic_sampling import sampling_summary
from synthetic_dataset import NAME_PAIRS, generate_dataset


class ReviewCorrections(unittest.TestCase):
    def test_audit_distinguishes_weights_from_api_access_and_renders_link(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(audit, "ROOT", output), \
                 patch.object(audit, "SUMMARY_FILE", output / "single_stage_summary.json"), \
                 patch.object(audit, "OUT_JSON", output / "summary.json"), \
                 patch.object(audit, "TEX_DIR", output), \
                 patch.object(audit, "load", return_value=(
                     json.loads(audit.AUDIT_FILE.read_text()),
                     json.loads((ROOT / "results/single_stage_summary.json").read_text()))), \
                 contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["headline"]["models_without_public_weights"], 10)
            self.assertEqual(result["headline"]["models_accessed_via_api"], 11)
            self.assertTrue(result["weight_availability"][
                "ibm-granite-granite-embedding-reranker-english-r2"]["public_weights"])
            cohere = next(line for line in (output / "vendor_docs_audit.tex").read_text().splitlines()
                          if line.startswith("Cohere &"))
            self.assertIn("navigation link", cohere)

    def test_missing_weight_evidence_fails(self):
        evidence, summary = audit.load()
        evidence = copy.deepcopy(evidence)
        evidence["weight_availability"].pop(next(iter(audit.scoped(evidence, summary)["models"])))
        with self.assertRaises(ValueError):
            audit.check_population(evidence, summary)

    def test_sampling_counts_unique_resumes_across_queries_and_repeats(self):
        a, b = SimpleNamespace(rid="a"), SimpleNamespace(rid="b")
        result = sampling_summary([
            {"qid": "one", "members": [a, b]},
            {"qid": "one", "members": [a, b]},
            {"qid": "two", "members": [a, b]},
        ])
        self.assertEqual(result["n_distinct_resumes_scored"], 2)
        self.assertEqual(result["n_focal_comparisons"], 6)
        self.assertEqual(result["n_distinct_query_resume_pairs"], 4)

    def test_full_transform_equalises_study_templates(self):
        names = [name for pair in NAME_PAIRS for name in pair]
        for t in generate_dataset(["names_and_pronouns"]):
            for variant in ("full", "gram_pos"):
                self.assertEqual(deidentify(t.doc_male, variant, names, lowercase=False),
                                 deidentify(t.doc_female, variant, names, lowercase=False))

    def test_documented_heuristic_counterexamples(self):
        # Preserve these examples as explicit limits of the cached condition.
        for male, female in [("John praised him yesterday.", "Mary praised her yesterday."),
                             ("John said his was selected.", "Mary said hers was selected.")]:
            self.assertNotEqual(deidentify(male, "gram_pos", ["John", "Mary"], lowercase=False),
                                deidentify(female, "gram_pos", ["John", "Mary"], lowercase=False))
            self.assertEqual(deidentify(male, "full", ["John", "Mary"], lowercase=False),
                             deidentify(female, "full", ["John", "Mary"], lowercase=False))


if __name__ == "__main__":
    unittest.main()
