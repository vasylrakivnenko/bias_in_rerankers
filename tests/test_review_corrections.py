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
from run_records import pinned_revision
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


class RunRecords(unittest.TestCase):
    """Reviewer finding 5: every reported run has a record and no field is null."""

    @staticmethod
    def _nulls(value, path=""):
        if value is None:
            yield path
        elif isinstance(value, dict):
            for k, v in value.items():
                yield from RunRecords._nulls(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from RunRecords._nulls(v, f"{path}[{i}]")

    def test_every_reported_run_has_a_complete_record(self):
        data = json.loads((ROOT / "results/run_records.json").read_text())
        recs = [r for r in data["records"] if r["reported"]]
        count = lambda e: sum(r["experiment"] == e for r in recs)  # noqa: E731
        self.assertEqual(count("audit"), 29)
        self.assertEqual(count("selfhost_check"), 2)
        self.assertEqual(count("deidentification"), 4)
        self.assertEqual(count("pipeline_retriever"), 1)
        self.assertEqual(count("companion"), 4)
        self.assertEqual(list(self._nulls(data)), [])
        for r in recs:
            if r["recoverable"]:
                self.assertRegex(r["revision"], r"^[0-9a-f]{40}$", r["id"])
                self.assertEqual(r["tokenizer_revision"], r["revision"], r["id"])
            if r["route"] == "local Mac" and r["scored_at"] < "2026-10-01":
                self.assertFalse(r["recoverable"], r["id"])
                self.assertEqual(r["revision"], "not recorded", r["id"])

    def test_manifest_has_no_null_fields(self):
        manifest = json.loads((ROOT / "results/model_manifest.json").read_text())
        self.assertEqual(list(self._nulls(manifest)), [])

    def test_pinned_revision_prefers_same_experiment_then_falls_back(self):
        sha_a, sha_b = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "run_records.json"
            manifest = Path(directory) / "model_manifest.json"
            records.write_text(json.dumps({"records": [
                {"model": "m", "experiment": "audit", "revision": "not recorded",
                 "scored_at": "2026-02-24"},
                {"model": "m", "experiment": "deidentification", "revision": sha_a,
                 "scored_at": "2026-10-16"}]}))
            manifest.write_text(json.dumps({"models": {"x": {"revision": sha_b}}}))
            warnings = []
            pin = lambda *a: pinned_revision(*a, records_path=records,  # noqa: E731
                                             manifest_path=manifest, warn=warnings.append)
            self.assertEqual(pin("m", "deidentification"), sha_a)
            self.assertEqual(warnings, [])
            self.assertEqual(pin("m", "audit"), sha_a)
            self.assertEqual(pin("x", "audit"), sha_b)
            self.assertIsNone(pin("y", "audit"))
            self.assertEqual(len(warnings), 3)


if __name__ == "__main__":
    unittest.main()
