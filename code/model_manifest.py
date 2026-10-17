"""
Pin the exact local model revisions and runtime behind the reported results.

Model identifiers alone are not immutable: a Hugging Face repository can be
updated in place, and the paper documents one local model whose scores drifted
between runs (reviewer item 4). This records the commit each local model
resolves to in the Hugging Face cache, the scoring configuration, and the
library versions. API models cannot be pinned; their scoring dates are in
results/single_stage_summary.json.

Output: results/model_manifest.json
Run:    venv/bin/python code/model_manifest.py
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "model_manifest.json"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

LOCAL_MODELS = {
    "BAAI/bge-large-en-v1.5": "retriever (pipeline simulation)",
    "BAAI/bge-reranker-v2-m3": "reranker (audit, pipeline, de-identification)",
    "BAAI/bge-reranker-base": "reranker (de-identification, companion)",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "reranker (audit, de-identification, companion)",
    "jinaai/jina-reranker-v2-base-multilingual": "reranker (audit, de-identification, companion)",
}


def revision(model: str) -> str | None:
    ref = HF_HUB / f"models--{model.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def dtype(model: str) -> str | None:
    rev = revision(model)
    cfg = HF_HUB / f"models--{model.replace('/', '--')}" / "snapshots" / str(rev) / "config.json"
    return json.loads(cfg.read_text()).get("torch_dtype") if cfg.exists() else None


def main():
    import sentence_transformers
    import torch
    import transformers

    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "note": "Revisions are those cached when this manifest was generated; they pin "
                   "the de-identification and order-test runs of October 2026. The banked "
                   "single-stage scores (February 2026) may predate them.",
           "scoring": {"max_length": 512, "batch_size": 64,
                       "loader": "sentence_transformers.CrossEncoder (rerankers), "
                                 "SentenceTransformer (retriever)",
                       "device": "Apple MPS (Metal) where available, else CPU"},
           "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": transformers.__version__,
                       "sentence_transformers": sentence_transformers.__version__,
                       "platform": platform.platform()},
           "models": {m: {"role": role, "revision": revision(m), "config_dtype": dtype(m)}
                      for m, role in LOCAL_MODELS.items()}}
    # Open-weight models scored on a rented GPU record their own revision,
    # precision and runtime in their raw files (code/score_open_models.py).
    for f in sorted((ROOT / "results").glob("**/*_full_raw.json")):
        meta = json.loads(f.read_text()).get("self_hosted")
        if meta:
            out["models"][meta["model"]] = {
                "role": "reranker (audit, scored on " + meta["gpu"] + ")"
                        + (" -- self-hosted check" if "selfhost" in f.parts else ""),
                "revision": meta["revision"], "config_dtype": meta.get("dtype"),
                "runtime": {k: meta.get(k) for k in ("torch", "transformers",
                                                     "sentence_transformers")}}
    missing = [m for m, v in out["models"].items() if v["revision"] is None]
    assert not missing, f"not in the local Hugging Face cache: {missing}"
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["models"], indent=1))


if __name__ == "__main__":
    main()
