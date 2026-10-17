"""
Pin the exact local model revisions and runtime behind the reported results.

Model identifiers alone are not immutable: a Hugging Face repository can be
updated in place, and the paper documents one local model whose scores drifted
between runs (reviewer item 4). This records the commit each local model
resolves to in the Hugging Face cache, the scoring configuration, and the
library versions, and the actual precision of every local and GPU-scored model.
API models cannot be pinned; their scoring dates are in
results/single_stage_summary.json. Per-run records (including runs whose
settings were never recorded) are in results/run_records.json
(code/run_records.py).

Output: results/model_manifest.json
Run:    venv/bin/python code/model_manifest.py
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
OUT = ROOT / "results" / "model_manifest.json"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

from run_records import (POD_PINS, POD_PINS_SOURCE, ST6_ENV_SOURCE, cached_main,  # noqa: E402
                         gpu_specs, precision, stored_dtype)

LOCAL_MODELS = {
    "BAAI/bge-large-en-v1.5": "retriever (pipeline simulation)",
    "BAAI/bge-reranker-v2-m3": "reranker (audit, pipeline, de-identification)",
    "BAAI/bge-reranker-base": "reranker (de-identification, companion)",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "reranker (audit, de-identification, companion)",
    "jinaai/jina-reranker-v2-base-multilingual": "reranker (audit, de-identification, companion)",
}


def revision(model: str) -> str | None:
    return cached_main(model)


def main():
    import sentence_transformers
    import torch
    import transformers

    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "note": "Local models: the revision is the commit the local Hugging Face cache "
                   "resolves to; it is the one the October 2026 de-identification and order-test "
                   "runs recorded and loaded. The February 2026 local audit runs and the "
                   "August 2026 companion and retriever runs recorded no revision or library "
                   "versions; those settings cannot be recovered (results/run_records.json "
                   "gives per-run records and the circumstantial evidence). GPU-scored models: "
                   "revision, precision and torch come from each raw file's self_hosted meta; "
                   "the other library versions are recorded there for two models and are the "
                   "stated pod pins for the rest (see runtime_source). `precision` is the dtype "
                   "the weights actually ran in: transformers 5.x loads with dtype='auto' when "
                   "no dtype is given.",
           "scoring": {"max_length": 512, "batch_size": 64,
                       "loader": "sentence_transformers.CrossEncoder (rerankers), "
                                 "SentenceTransformer (retriever)",
                       "device": "Apple MPS (Metal) where available, else CPU",
                       "note": "max_length=512 is set by the October de-identification runs; the "
                               "February local audit runs (run_experiment.py) set none and "
                               "truncated at the tokenizer limit (no pair comes close)"},
           "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": transformers.__version__,
                       "sentence_transformers": sentence_transformers.__version__,
                       "platform": platform.platform()},
           "models": {}}
    for m, role in LOCAL_MODELS.items():
        rev = revision(m)
        stored, stored_src = stored_dtype(m, rev) if rev else (None, "not in cache")
        prec, prec_src = precision(None, m, rev, transformers.__version__)
        out["models"][m] = {"role": role, "revision": rev, "stored_dtype": stored,
                            "precision": prec, "precision_source": prec_src,
                            "runtime_source": "this manifest's `runtime` (project venv)"}
    # Open-weight models scored on a rented GPU record their own revision,
    # precision and runtime in their raw files (code/score_open_models.py).
    specs = gpu_specs()
    for f in sorted((ROOT / "results").glob("**/*_full_raw.json")):
        meta = json.loads(f.read_text()).get("self_hosted")
        if not meta:
            continue
        m, rev = meta["model"], meta["revision"]
        st6 = "transformers" in meta
        runtime = {"torch": meta["torch"]}
        if st6:
            runtime.update({"transformers": meta["transformers"],
                            "sentence_transformers": meta["sentence_transformers"]})
        else:
            runtime.update({k: v for k, v in POD_PINS.items() if k != "torch"})
        stored, _ = stored_dtype(m, rev)
        prec, prec_src = precision(specs[m]["dtype"], m, rev, runtime["transformers"])
        entry = {"role": "reranker (audit, scored on " + meta["gpu"] + ")"
                         + (" -- self-hosted check" if "selfhost" in f.parts else ""),
                 "revision": rev, "stored_dtype": stored or "not in local cache",
                 "precision": prec, "precision_source": prec_src,
                 "loader_dtype": meta.get("dtype") or "none (library default)",
                 "runtime": runtime,
                 "runtime_source": ST6_ENV_SOURCE if st6 else
                                   "torch: raw-file self_hosted meta; others: " + POD_PINS_SOURCE}
        if m in out["models"]:
            # also run locally (e.g. bge-reranker-base): keep both, same weights required
            assert out["models"][m]["revision"] == rev, f"{m}: local and GPU revisions differ"
            out["models"][m]["role"] += "; " + entry["role"]
            out["models"][m]["gpu_run"] = entry
        else:
            out["models"][m] = entry
    missing = [m for m, v in out["models"].items() if v["revision"] is None]
    assert not missing, f"not in the local Hugging Face cache: {missing}"

    def nulls(d, path=""):
        for k, v in d.items():
            if v is None:
                yield path + k
            elif isinstance(v, dict):
                yield from nulls(v, path + k + ".")
    bad = list(nulls(out))
    assert not bad, f"null fields: {bad}"
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["models"], indent=1))


if __name__ == "__main__":
    main()
