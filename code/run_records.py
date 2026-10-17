"""
One record per reported scoring run: what was run, where, and how far it can be
reproduced (reviewer finding 5).

results/model_manifest.json pins the local Hugging Face cache; this file goes one
step further and describes every *run* the paper reports -- the 29 audit models,
the two self-hosted consistency checks, the four de-identification models, the
pipeline retriever and the companion experiment's four rerankers -- with its
route, date, model and tokenizer revision, actual precision, software versions,
scoring recipe and truncation length. Every value is derived from a results file,
a scoring script or the local Hugging Face cache; each carries a `*_source`
saying which. Where a value was never recorded it says so, and `recoverable` is
false: such historical settings cannot be recovered after the fact, only
bounded by circumstantial evidence, which is kept apart in `inferred`.

Two loaders consume these pins when re-scoring (see `pinned_revision`):
code/deidentification_experiment.py and the local path of
code/run_experiment.py; code/score_open_models.py uses them for GPU re-runs.

Precision. transformers 5.x loads with dtype="auto" when no dtype is given, i.e.
in the dtype the checkpoint's config.json declares (else that of its weights) --
not in fp32. A sentence-transformers CrossEncoder without a dtype therefore runs
jina-reranker-v2-base-multilingual (config: bfloat16) in bf16 under
transformers 5.2.0, and every float32 checkpoint in fp32.

Output: results/run_records.json
Run:    venv/bin/python code/run_records.py [--fetch-configs]
        (--fetch-configs downloads config.json, and reads the safetensors header,
        for any pinned revision missing from the local cache; nothing is scored)
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import re
import struct
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "run_records.json"
MANIFEST = RESULTS / "model_manifest.json"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
VENV_SITE = next((ROOT / "venv" / "lib").glob("python3*/site-packages"), None) \
    if (ROOT / "venv" / "lib").exists() else None
SHA = re.compile(r"^[0-9a-f]{40}$")
NOT_RECORDED = "not recorded"

# The RunPod pod's environment, pinned by its launch script (the launch script is
# not in this repository; these are the stated pins). The raw files record torch
# and the GPU themselves, and the last two models record their own library versions.
POD_PINS = {"transformers": "5.2.0", "sentence_transformers": "5.2.3",
            "huggingface_hub": "1.4.1", "tokenizers": "0.22.2", "torch": "2.4.1+cu124"}
POD_PINS_SOURCE = ("stated pod pins (RunPod launch script: transformers==5.2.0, "
                   "sentence-transformers==5.2.3, huggingface_hub==1.4.1, tokenizers==0.22.2, "
                   "torch 2.4.1+cu124); the launch script is not in the repository")
ST6_ENV_SOURCE = ("raw-file self_hosted meta (sentence_transformers, transformers, torch); "
                  "the separate sentence-transformers 6 environment's huggingface_hub, "
                  "tokenizers and python were not recorded")

ROUTE_LABEL = {"vendor": "vendor API", "openrouter": "OpenRouter", "azure": "Azure",
               "mac": "local Mac", "runpod": "RunPod GPU"}

BACKEND_RECIPE = {
    "crossencoder": "sentence_transformers.CrossEncoder.predict on (query, document) pairs, "
                    "the model's default activation",
    "qwen3_yesno": "official Qwen3-Reranker recipe: instruction prompt (task 'Given a web search "
                   "query, retrieve relevant passages that answer the query'), score = "
                   "P('yes') / (P('yes') + P('no')) at the final position",
    "nemotron": "model-card template 'question:{query} \\n \\n passage:{doc}', "
                "AutoModelForSequenceClassification logit",
    "jina_listwise": "listwise model.rerank(query, docs): all of a query's documents in one call "
                     "(plus reversed / shuffled / repeated layouts as order checks)",
}


# ---------------------------------------------------------------------------
# Reading code without importing it (the scoring scripts import torch)
# ---------------------------------------------------------------------------

def module_constants(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.literal_eval(node.value)
    return out


def api_routes() -> dict:
    c = module_constants(CODE_DIR / "run_experiment.py",
                         {"VOYAGE_MODELS", "AZURE_MODELS", "COHERE_MODELS", "GCP_MODELS",
                          "OPENROUTER_MODELS", "BATCH_SIZE"})
    routes = {}
    for m in c["VOYAGE_MODELS"]:
        routes[m] = ("vendor", "Voyage AI rerank API (voyageai client 0.2.4 per requirements.txt)",
                     "per-query request: the query's 40 counterfactual pairs, male/female "
                     "interleaved (80 documents), top_k = all")
    for m in c["GCP_MODELS"]:
        routes[m] = ("vendor", "Google Cloud Discovery Engine Ranking API (HTTPS via requests)",
                     "per-query request: 80 interleaved documents as records (chunked at 200)")
    for m in c["COHERE_MODELS"]:
        routes[m] = ("azure", "Cohere v2 rerank endpoint hosted on Azure AI (HTTPS via requests)",
                     "per-query request: 80 interleaved documents, top_n = all")
    for m in c["AZURE_MODELS"]:
        routes[m] = ("azure", "Azure ML managed rerank endpoint (HTTPS via requests)",
                     "per-query request: 80 interleaved documents, chunked at 32 texts")
    for m, or_id in c["OPENROUTER_MODELS"].items():
        routes[m] = ("openrouter", f"OpenRouter rerank endpoint (model id {or_id})",
                     "per-query request: 80 interleaved documents, top_n = all")
    return routes, c["BATCH_SIZE"]


def gpu_specs() -> dict:
    return module_constants(CODE_DIR / "score_open_models.py", {"MODELS"})["MODELS"]


# ---------------------------------------------------------------------------
# Hugging Face cache: revisions, stored dtypes, tokenizer limits
# ---------------------------------------------------------------------------

def repo_dir(model: str) -> Path:
    return HF_HUB / f"models--{model.replace('/', '--')}"


def cached_main(model: str) -> str | None:
    ref = repo_dir(model) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def snapshot_file(model: str, rev: str, name: str) -> Path | None:
    p = repo_dir(model) / "snapshots" / rev / name
    return p if p.exists() else None


def _json(model, rev, name):
    p = snapshot_file(model, rev, name)
    return json.loads(p.read_text()) if p else None


def _safetensors_dtype(path: Path) -> str | None:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    floats = [v["dtype"] for k, v in header.items()
              if k != "__metadata__" and v["dtype"] in ("F32", "F16", "BF16", "F64")]
    return floats[0] if floats else None


SHORT = {"float32": "fp32", "F32": "fp32", "bfloat16": "bf16", "BF16": "bf16",
         "float16": "fp16", "F16": "fp16", "F64": "fp64", "float64": "fp64"}


def stored_dtype(model: str, rev: str, fetch: bool = False) -> tuple[str | None, str]:
    """The dtype transformers' dtype='auto' resolves to: config.json, else the weights."""
    cfg = _json(model, rev, "config.json")
    if cfg is None and fetch:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model, "config.json", revision=rev)
        cfg = _json(model, rev, "config.json")
    if cfg is None:
        return None, f"config.json for {rev[:12]} not in the local cache (run with --fetch-configs)"
    d = cfg.get("torch_dtype") or cfg.get("dtype")
    if d:
        return SHORT.get(d, d), f"config.json at {rev[:12]} declares {d}"
    st = snapshot_file(model, rev, "model.safetensors")
    if st:
        w = _safetensors_dtype(st)
        return SHORT.get(w, w), f"config.json at {rev[:12]} sets no dtype; checkpoint weights are {w}"
    if fetch:
        from huggingface_hub import get_safetensors_metadata
        md = get_safetensors_metadata(model, revision=rev)
        w = next(iter(next(iter(md.files_metadata.values())).parameter_count))
        return SHORT.get(w, w), f"config.json at {rev[:12]} sets no dtype; checkpoint weights are {w}"
    return None, f"config.json at {rev[:12]} sets no dtype and no local weights to inspect"


def precision(explicit, model, rev, transformers_version, fetch=False) -> tuple[str, str]:
    """The precision a loader actually ran at."""
    if explicit not in (None, "auto"):
        return SHORT.get(explicit, explicit), f"explicit dtype={explicit} in the loader"
    if explicit is None and transformers_version and int(transformers_version.split(".")[0]) < 5:
        return "fp32", f"transformers {transformers_version} default (torch default dtype)"
    stored, src = stored_dtype(model, rev, fetch) if rev and SHA.match(rev) else (None, "no revision")
    how = ("loader passes dtype='auto'" if explicit == "auto"
           else f"no dtype given; transformers {transformers_version} defaults to dtype='auto'")
    return (stored or "unverified"), f"{how}; {src}"


def tokenizer_limit(model: str, rev: str) -> int | None:
    """What a CrossEncoder without max_length truncates at (sentence-transformers 5.2)."""
    tc, cfg = _json(model, rev, "tokenizer_config.json"), _json(model, rev, "config.json")
    if not tc or "model_max_length" not in tc:
        return None
    lim = int(tc["model_max_length"])
    if cfg and cfg.get("max_position_embeddings"):
        lim = min(lim, int(cfg["max_position_embeddings"]))
    return lim


def snapshot_evidence(model: str) -> tuple[str | None, str]:
    """Circumstantial evidence for which snapshot an unrecorded run used."""
    d = repo_dir(model) / "snapshots"
    snaps = sorted(p.name for p in d.iterdir()) if d.exists() else []
    if not snaps:
        return None, "model not in the local cache"
    blobs = sorted((repo_dir(model) / "blobs").iterdir(), key=lambda p: p.stat().st_mtime)
    fmt = lambda p: _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")  # noqa: E731
    note = (f"{len(snaps)} snapshot(s) in the local cache ({', '.join(s[:12] for s in snaps)}); "
            f"files downloaded {fmt(blobs[0])} to {fmt(blobs[-1])}")
    return (snaps[0] if len(snaps) == 1 else None), note


def venv_evidence() -> tuple[dict, str]:
    """Versions in the project venv and when they were installed."""
    pkgs = {"torch": "torch", "transformers": "transformers",
            "sentence_transformers": "sentence_transformers", "tokenizers": "tokenizers",
            "huggingface_hub": "huggingface_hub"}
    out, dates = {}, set()
    if VENV_SITE is None:
        return {}, "no project venv"
    for key, dist in pkgs.items():
        hit = sorted(VENV_SITE.glob(f"{dist}-*.dist-info"))
        if hit:
            out[key] = hit[0].name[len(dist) + 1:-len(".dist-info")]
            dates.add(_dt.datetime.fromtimestamp(hit[0].stat().st_mtime).isoformat(timespec="minutes"))
    return out, ("project venv dist-info for these packages dated " + ", ".join(sorted(dates))
                 + " and not reinstalled since")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def base(model, experiment, role, route, **kw) -> dict:
    rec = {"id": f"{experiment}:{model}", "model": model, "experiment": experiment, "role": role,
           "route": ROUTE_LABEL[route]}
    rec.update(kw)
    return rec


def api_record(raw, d, routes, feb_date):
    model = d["model"]
    route, provider, layout = routes[model]
    scored = d.get("scored_at")
    served = d.get("served_by")
    excluded = model in EXCLUDED
    return base(
        model, "audit",
        "reranker (single-stage audit)" + (" -- excluded from the audit as a duplicate of "
                                           "semantic-ranker-default-004" if excluded else ""),
        route, reported=not excluded, provider=provider,
        served_by=served if served else ("not recorded" if route == "openrouter" else "not applicable"),
        hardware="vendor-side", script="code/run_experiment.py", output=str(raw.relative_to(ROOT)),
        scored_at=scored or feb_date,
        scored_at_source=("raw file `scored_at`" if scored else
                          "results/single_stage_summary.json raw_scored_date (raw file carries no date)"),
        revision="not applicable (vendor-hosted; the API exposes only the model name)",
        revision_source="vendor API", tokenizer_revision="not applicable (vendor-side)",
        precision="unknown (vendor-side)", precision_source="vendor API",
        software={"client": provider}, software_source="code/run_experiment.py; requirements.txt",
        recipe=layout + "; scores as returned by the API",
        batch_size="one request per query", truncation_max_length="vendor default (not controllable)",
        truncation_note="vendor-side", n_pairs=d["metrics"]["n_triples"], pinning="none", recoverable=False,
        note=("Vendor-hosted model: the model behind the name can be updated by the vendor, so "
              "re-scoring reproduces these scores only while the vendor serves the same model. "
              "The banked raw scores are the record." +
              (" OpenRouter records the backend that served each request (`served_by`)."
               if route == "openrouter" else "")))


def gpu_record(raw, d, specs, fetch):
    meta = d["self_hosted"]
    model = meta["model"]
    spec = specs[model]
    st6 = "transformers" in meta
    sw = {"torch": meta["torch"]}
    if st6:
        sw.update({"transformers": meta["transformers"],
                   "sentence_transformers": meta["sentence_transformers"],
                   "huggingface_hub": NOT_RECORDED, "tokenizers": NOT_RECORDED,
                   "python": NOT_RECORDED})
        sw_src = ST6_ENV_SOURCE
    else:
        sw.update({k: v for k, v in POD_PINS.items() if k != "torch"})
        sw["python"] = NOT_RECORDED
        sw_src = "torch: raw-file self_hosted meta; others: " + POD_PINS_SOURCE
    prec, prec_src = precision(spec["dtype"], model, meta["revision"],
                               sw["transformers"], fetch)
    backend = meta["backend"]
    if backend == "qwen3_yesno":
        trunc, tnote = 8192, "explicit max_length=8192 (prompt + document), qwen3_yesno recipe"
    elif backend == "jina_listwise":
        lim = tokenizer_limit(model, meta["revision"])
        trunc = 2048
        tnote = ("model.rerank defaults: max_doc_length=2048, max_query_length=512, context "
                 f"{lim or 'tokenizer model_max_length'} tokens, blocks of 125 documents")
    else:
        trunc, tnote = 512, "explicit max_length=512"
    selfhost = spec.get("selfhost")
    return base(
        model, "selfhost_check" if selfhost else "audit",
        (f"self-hosted consistency check of the API model {selfhost} (not in the audit)"
         if selfhost else "reranker (single-stage audit)"),
        "runpod", reported=True, provider="open weights from the Hugging Face Hub",
        served_by="not applicable", hardware=meta["gpu"], script=meta["script_version"],
        output=str(raw.relative_to(ROOT)),
        scored_at=d["scored_at"], scored_at_source="raw file `scored_at` (= self_hosted.started, pod clock)",
        revision=meta["revision"], revision_source="raw-file self_hosted meta (snapshot_download commit)",
        tokenizer_revision=meta["revision"],
        tokenizer_revision_source="same snapshot: every backend loads model, tokenizer and "
                                  "remote code with revision=<that commit>",
        precision=prec, precision_source=prec_src, software=sw, software_source=sw_src,
        recipe=BACKEND_RECIPE[backend], batch_size=spec["batch"] or "one call per query",
        truncation_max_length=trunc, truncation_note=tnote, n_pairs=d["metrics"]["n_triples"],
        pinning="full", recoverable=True,
        note=("Fully pinned: revision, precision, recipe and truncation are in the raw file or "
              "score_open_models.py; " + ("library versions recorded in the raw file."
                                          if st6 else "library versions are the stated pod pins.")
              + " Re-run with the recorded revision: score_open_models.py reads it from this file."))


def feb_local_record(raw, d, feb_date, batch_default, venv_sw, venv_note):
    model = d["model"]
    snap, snap_note = snapshot_evidence(model)
    inferred_prec, prec_src = precision(None, model, snap, venv_sw.get("transformers"))
    lim = tokenizer_limit(model, snap) if snap else None
    return base(
        model, "audit", "reranker (single-stage audit)", "mac", reported=True,
        provider="open weights from the Hugging Face Hub", served_by="not applicable",
        hardware="local Mac (device not recorded; sentence-transformers selects MPS when available)",
        script="code/run_experiment.py (local CrossEncoder path)", output=str(raw.relative_to(ROOT)),
        scored_at=feb_date,
        scored_at_source="results/single_stage_summary.json raw_scored_date (raw file carries no date)",
        revision=NOT_RECORDED, revision_source="the February run recorded no revision",
        tokenizer_revision=NOT_RECORDED,
        precision=NOT_RECORDED, precision_source="the February run recorded no dtype or library versions",
        software={k: NOT_RECORDED for k in ("python", "torch", "transformers",
                                             "sentence_transformers", "tokenizers")},
        software_source="the February run recorded no library versions",
        recipe=("CrossEncoder(model, trust_remote_code=True).predict on interleaved "
                "(query, male doc), (query, female doc) pairs; default activation"),
        batch_size=f"{batch_default} (CLI default; the value used was not recorded)",
        truncation_max_length="not set: tokenizer model_max_length",
        truncation_note=(f"{lim} tokens under the inferred snapshot; the longest query-document "
                         "pair of the 52-occupation dataset tokenizes to 55-60 tokens (measured "
                         "2026-10-16), so truncation does not bind" if lim else "unknown"),
        n_pairs=d["metrics"]["n_triples"], pinning="none", recoverable=False,
        inferred={"revision": snap or "ambiguous", "revision_evidence": snap_note,
                  "software": venv_sw, "software_evidence": venv_note,
                  "precision": inferred_prec,
                  "precision_evidence": prec_src + " (under the inferred revision and versions)"},
        note=("Historical settings that cannot be recovered: the February 2026 local run "
              "recorded neither the model revision nor the library versions or precision. "
              "`inferred` gives circumstantial evidence only (cache snapshot and venv install "
              "dates on the scoring day); see also the October de-identification run's "
              "determinism check against these banked scores. Re-scoring now uses the October "
              "pinned revision and may not reproduce these numbers exactly."))


EXCLUDED: set = set()


def audit_records(fetch):
    summary = json.loads((RESULTS / "single_stage_summary.json").read_text())
    feb_date = summary["raw_scored_date"]
    EXCLUDED.update(summary.get("excluded_models", []))
    routes, batch_default = api_routes()
    specs = gpu_specs()
    venv_sw, venv_note = venv_evidence()
    recs = []
    for raw in sorted(RESULTS.glob("*_full_raw.json")) + sorted((RESULTS / "selfhost").glob("*_full_raw.json")):
        d = json.loads(raw.read_text())
        if d.get("self_hosted"):
            recs.append(gpu_record(raw, d, specs, fetch))
        elif d["model"] in routes:
            recs.append(api_record(raw, d, routes, feb_date))
        else:
            recs.append(feb_local_record(raw, d, feb_date, batch_default, venv_sw, venv_note))
    # determinism evidence from the October de-identification runs
    for r in recs:
        if r["route"] == ROUTE_LABEL["mac"]:
            f = RESULTS / f"deidentification_{r['model'].replace('/', '_')}.json"
            if f.exists():
                dc = json.loads(f.read_text()).get("determinism_check", {})
                if dc.get("status") == "checked":
                    r["inferred"]["determinism_check_oct2026"] = {
                        k: dc[k] for k in ("max_abs_score_diff", "mean_abs_score_diff", "passed")}
    return recs, venv_sw, venv_note


def deid_records(fetch):
    rc = json.loads((RESULTS / "reproducibility_checks.json").read_text())
    mac_hw = next((v["b"] for v in rc.get("cross_hardware", {}).values()), "Apple Mac (MPS)")
    recs = []
    for f in sorted(RESULTS.glob("deidentification_*.json")):
        d = json.loads(f.read_text())
        if "reranker" not in d or f.name == "deidentification_bge.json":   # bge.json = legacy copy of v2-m3
            continue
        rt = d["runtime"]
        rev = d["model_revision"]
        prec, prec_src = precision(None, d["reranker"], rev, rt["transformers"], fetch)
        pairs = RESULTS / "deid_pairs" / f"{d['reranker'].replace('/', '_')}.json"
        recs.append(base(
            d["reranker"], "deidentification", "reranker (de-identification experiment)", "mac",
            reported=True, provider="open weights from the Hugging Face Hub",
            served_by="not applicable", hardware=f"{mac_hw}, device={d['device']}",
            script=d["script_version"],
            output=str(f.relative_to(ROOT)) + (f"; {pairs.relative_to(ROOT)}" if pairs.exists() else ""),
            scored_at=rt["date"], scored_at_source="output `runtime.date`",
            revision=rev, revision_source="output `model_revision` (local cache commit at run time)",
            tokenizer_revision=rev,
            tokenizer_revision_source="same snapshot: CrossEncoder loads config, weights, tokenizer "
                                      "and remote code from one revision",
            precision=prec, precision_source=prec_src,
            software={"torch": rt["torch"], "transformers": rt["transformers"],
                      "sentence_transformers": rt["sentence_transformers"],
                      "tokenizers": venv_tok(), "python": NOT_RECORDED},
            software_source="output `runtime` (torch, transformers, sentence_transformers); "
                            "tokenizers: project venv; python patch level not recorded",
            recipe="CrossEncoder(max_length=512, trust_remote_code=True).predict on de-duplicated "
                   "(query, document) pairs; default activation",
            batch_size=rt["batch_size"], truncation_max_length=rt["max_length"],
            truncation_note="explicit max_length", pinning="full", recoverable=True,
            n_pairs=d["n_triples"],
            note=("Fully pinned. Baseline (untransformed) scores: " + d["baseline_source"] + ".")))
    return recs


def venv_tok():
    sw, _ = venv_evidence()
    return sw.get("tokenizers", NOT_RECORDED)


def retriever_record(venv_sw, venv_note):
    cache = json.loads((RESULTS / "compounding_emb_cache.json").read_text())
    model = cache["embedder"]
    snap, snap_note = snapshot_evidence(model)
    sbc = _json(model, snap, "sentence_bert_config.json") if snap else None
    prec, prec_src = precision(None, model, snap, venv_sw.get("transformers"))
    return base(
        model, "pipeline_retriever", "first-stage dense retriever (two-stage pipeline simulation)",
        "mac", reported=True, provider="open weights from the Hugging Face Hub",
        served_by="not applicable",
        hardware="local Mac (code selects MPS when available; device not recorded)",
        script="code/compounding_experiment.py (embedder scores cached)",
        output="results/compounding_emb_cache.json",
        scored_at=cache["generated"][:10], scored_at_source="cache `generated`",
        revision=NOT_RECORDED, revision_source="the embedder cache records no revision",
        tokenizer_revision=NOT_RECORDED, precision=NOT_RECORDED,
        precision_source="not recorded", software={k: NOT_RECORDED for k in
                                                    ("python", "torch", "transformers",
                                                     "sentence_transformers")},
        software_source="not recorded",
        recipe=("SentenceTransformer.encode, normalize_embeddings=True, batch 128; query prefixed "
                f"with '{cache['query_instruction']}'; score = cosine (dot of unit vectors)"),
        batch_size=128,
        truncation_max_length=(sbc or {}).get("max_seq_length", "not set: model default"),
        truncation_note="sentence_bert_config.json max_seq_length of the inferred snapshot",
        pinning="none", recoverable=False,
        inferred={"revision": snap or "ambiguous", "revision_evidence": snap_note,
                  "software": venv_sw, "software_evidence": venv_note, "precision": prec,
                  "precision_evidence": prec_src},
        note=("Revision and library versions were not recorded and cannot be recovered; the "
              "embedder scores are banked in results/compounding_emb_cache.json and every "
              "pipeline analysis reads them from there."))


def companion_records(venv_sw, venv_note):
    d = json.loads((RESULTS / "realistic_pool_four_rerankers.json").read_text())
    cfg = d["config"]
    recs = []
    for short, m in d["models"].items():
        model = m["model"]
        snap, snap_note = snapshot_evidence(model)
        prec, prec_src = precision(None, model, snap, venv_sw.get("transformers"))
        recs.append(base(
            model, "companion", "reranker (companion experiment on published resume examples)",
            "mac", reported=True, provider="open weights from the Hugging Face Hub",
            served_by="not applicable", hardware=f"local Mac, device={cfg['device']}",
            script="code/realistic_pool_experiment.py",
            output="results/realistic_pool_four_rerankers.json",
            scored_at=d["generated"][:10], scored_at_source="output `generated`",
            revision=NOT_RECORDED, revision_source="the companion output records no revision",
            tokenizer_revision=NOT_RECORDED, precision=NOT_RECORDED,
            precision_source="not recorded",
            software={k: NOT_RECORDED for k in ("python", "torch", "transformers",
                                                 "sentence_transformers")},
            software_source="not recorded",
            recipe=("CrossEncoder(max_length=512, trust_remote_code=True) on (query, resume) pairs "
                    "in pools of K=20; documents longer than the window are handled by the "
                    "script's windowing"),
            batch_size=cfg["batch_size"], truncation_max_length=cfg["max_length"],
            truncation_note="explicit max_length (binds for long resumes)",
            pinning="partial", recoverable=False,
            inferred={"revision": snap or "ambiguous", "revision_evidence": snap_note,
                      "software": venv_sw, "software_evidence": venv_note, "precision": prec,
                      "precision_evidence": prec_src},
            note=("Recipe, truncation, batch and device are recorded; the model revision and "
                  "library versions were not and cannot be recovered. The summary is banked.")))
    return recs


def build(fetch=False) -> dict:
    audit, venv_sw, venv_note = audit_records(fetch)
    recs = audit + deid_records(fetch) + [retriever_record(venv_sw, venv_note)] \
        + companion_records(venv_sw, venv_note)
    reported_audit = [r for r in recs if r["experiment"] == "audit" and r["reported"]]
    assert len(reported_audit) == 29, len(reported_audit)
    counts = {}
    for r in recs:
        if not r["reported"]:
            continue
        k = "pinned" if r["recoverable"] else "not recoverable"
        counts.setdefault(r["experiment"], {}).setdefault(k, 0)
        counts[r["experiment"]][k] += 1
    return {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": "code/run_records.py",
        "definitions": {
            "recoverable": "true when every setting that determines the scores (revision, "
                           "precision, library versions, recipe, truncation) is recorded or "
                           "pinned, so the run can be repeated on the same weights; false for "
                           "vendor APIs and for local runs that recorded no revision",
            "pinning": "full / partial (recipe recorded, weights or versions not) / none",
            "inferred": "circumstantial evidence for an unrecorded setting; never a record",
            "precision": "the dtype the weights ran in. transformers 5.x loads with "
                         "dtype='auto' when none is given, so a CrossEncoder without a dtype "
                         "runs in the checkpoint's declared dtype (bf16 for "
                         "jina-reranker-v2-base-multilingual), not necessarily fp32",
        },
        "counts_reported_runs": counts,
        "excluded_runs": sorted(r["id"] for r in recs if not r["reported"]),
        "records": recs,
    }


# ---------------------------------------------------------------------------
# Consuming the pins
# ---------------------------------------------------------------------------

def pinned_revision(model: str, experiment: str | None = None, records_path: Path = OUT,
                    manifest_path: Path = MANIFEST, warn=print) -> str | None:
    """The commit to load `model` at, or None (load the current revision) with a warning.

    Preference: the record of the same experiment with a recorded revision; else the
    most recent record of the model with one (a February run that recorded none
    falls back to the October pin); else the manifest's cache revision.
    """
    recs = []
    if records_path.exists():
        recs = [r for r in json.loads(records_path.read_text())["records"]
                if r["model"] == model and SHA.match(str(r.get("revision", "")))]
    same = [r for r in recs if experiment and r["experiment"] == experiment]
    if same:
        return same[0]["revision"]
    if recs:
        r = max(recs, key=lambda r: r["scored_at"])
        if experiment:
            warn(f"WARNING: no revision recorded for the {experiment} run of {model}; "
                 f"loading the revision pinned by its {r['experiment']} run of {r['scored_at']} "
                 f"({r['revision'][:12]}). Scores may differ from the banked run.")
        return r["revision"]
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text()).get("models", {}).get(model, {})
        if SHA.match(str(m.get("revision", ""))):
            warn(f"WARNING: {model} has no run record with a revision; loading the revision "
                 f"in {manifest_path.name} ({m['revision'][:12]}).")
            return m["revision"]
    warn(f"WARNING: no pinned revision for {model} in {records_path.name} or "
         f"{manifest_path.name}; loading the current Hugging Face revision.")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch-configs", action="store_true",
                    help="download config.json for pinned revisions missing from the cache")
    args = ap.parse_args()
    out = build(fetch=args.fetch_configs)
    nulls = [(r["id"], k) for r in out["records"] for k, v in r.items() if v is None]
    assert not nulls, f"null fields: {nulls}"
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    w = max(len(r["id"]) for r in out["records"])
    for r in out["records"]:
        print(f"{r['id']:<{w}}  {r['route']:<10} {r['scored_at']}  "
              f"{str(r['revision'])[:12]:<14} {str(r['precision']):<22} "
              f"{'pinned' if r['recoverable'] else 'NOT recoverable'}")
    print(json.dumps(out["counts_reported_runs"]), "excluded:", out["excluded_runs"])
    print(f"Saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
