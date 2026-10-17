"""
Checkpointed scoring of open-weight rerankers on a GPU (RunPod) for the audit.

Built for long unattended runs:

  * Canary first. Before a model's full run, a short occupation-matching check
    (one query per sampled occupation against one document from each of the
    sampled occupations) must place the matching document first often enough,
    and scores must be finite and non-constant. A model that fails is skipped,
    so a wrong recipe cannot burn hours producing meaningless numbers.
  * Checkpoints. Every finished query (40 counterfactual pairs, the audit's
    request unit) is appended to results/partial/<slug>.jsonl. A restart, on
    the same pod or a fresh one, skips every query already on disk.
  * Status. results/partial/status.json is rewritten after every query with
    progress, rate, ETA and the running male-typed minus female-typed contrast,
    so a monitor can report milestones without reading the raw rows.
  * Parallel workers. `--worker big` and `--worker small` take disjoint model
    lists and fit on one 24 GB GPU together; status writes are atomic and
    per-worker, so the two never clobber each other.

Listwise models (jina-reranker-v3) score all of a query's documents in one
call, so each query is also scored in reversed, shuffled and repeated layouts,
the same arms as code/order_reversal_test.py.

`--finalize` (run locally after syncing) turns each completed jsonl into the
audit's results/<model>_full_raw.json schema. Self-hosted copies of models the
audit already reaches through an API are written to results/selfhost/ instead,
so they never enter the main audit.

Revisions. A re-run loads each model at the commit recorded for its run in
results/run_records.json (a resumed checkpoint keeps its own); --unpinned loads
the current Hugging Face revision instead.

Run on the pod:   python code/score_open_models.py --worker big
Finalize locally: venv/bin/python code/score_open_models.py --finalize
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
import zlib
from collections import OrderedDict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from labels import load_labels  # noqa: E402
from synthetic_dataset import generate_dataset  # noqa: E402

SCRIPT_VERSION = "score_open_models.py v1.0"
RESULTS = ROOT / "results"
PARTIAL = RESULTS / "partial"
SELFHOST = RESULTS / "selfhost"
SEED = 20260822
ARMS = ("same", "repeat", "reversed", "shuffled")

# name -> recipe. `selfhost` marks a copy of a model the audit already scores
# through an API; it is a consistency check, not a new model.
MODELS = {
    # big worker: LLM-sized models, bf16, one at a time
    "Qwen/Qwen3-Reranker-4B": {"worker": "big", "backend": "qwen3_yesno", "dtype": "bfloat16", "batch": 16},
    "zeroentropy/zerank-2-reranker": {"worker": "big", "backend": "crossencoder", "dtype": "bfloat16", "batch": 16},
    "mixedbread-ai/mxbai-rerank-large-v2": {"worker": "big", "backend": "crossencoder", "dtype": "bfloat16", "batch": 32},
    "nvidia/llama-nemotron-rerank-1b-v2": {"worker": "big", "backend": "nemotron", "dtype": "bfloat16", "batch": 32},
    "jinaai/jina-reranker-v3": {"worker": "big", "backend": "jina_listwise", "dtype": "auto", "batch": 0},
    "Qwen/Qwen3-Reranker-8B": {"worker": "big", "backend": "qwen3_yesno", "dtype": "bfloat16", "batch": 8,
                               "selfhost": "qwen/qwen3-reranker-8b"},
    # small worker: encoder-sized cross-encoders, full precision as in the audit
    "cross-encoder/ms-marco-MiniLM-L6-v2": {"worker": "small", "backend": "crossencoder", "dtype": None, "batch": 64},
    "BAAI/bge-reranker-base": {"worker": "small", "backend": "crossencoder", "dtype": None, "batch": 64},
    "BAAI/bge-reranker-large": {"worker": "small", "backend": "crossencoder", "dtype": None, "batch": 64},
    "Alibaba-NLP/gte-reranker-modernbert-base": {"worker": "small", "backend": "crossencoder", "dtype": None, "batch": 64},
    "Qwen/Qwen3-Reranker-0.6B": {"worker": "small", "backend": "qwen3_yesno", "dtype": "bfloat16", "batch": 32},
    "ibm-granite/granite-embedding-reranker-english-r2": {"worker": "small", "backend": "crossencoder", "dtype": None,
                                                          "batch": 64, "selfhost": "ibm-granite-granite-embedding-reranker-english-r2"},
}


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def free_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def slug(model: str) -> str:
    return model.replace("/", "__")


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Work units: the audit's per-query groups, in the audit's layout
# ---------------------------------------------------------------------------

def work_units():
    labels, _ = load_labels()
    triples = generate_dataset(perturbation_levels=["names_and_pronouns"], query_types=None)
    groups: OrderedDict[str, list] = OrderedDict()
    for t in triples:
        if labels.get(t.occupation) is not None:
            groups.setdefault(t.query, []).append(t)
    return groups, labels


def canary_set(groups, labels, n_occ=12):
    """One query per sampled occupation vs one document from each sampled occupation."""
    rng = random.Random(SEED)
    by_occ = {}
    for q, ts in groups.items():
        t = ts[0]
        if t.query_type == "bare" and t.occupation not in by_occ:
            by_occ[t.occupation] = (q, t.doc_male)
    occs = sorted(by_occ)
    rng.shuffle(occs)
    occs = sorted(occs[:n_occ])
    docs = [by_occ[o][1] for o in occs]
    return [(by_occ[o][0], docs, i) for i, o in enumerate(occs)]


# ---------------------------------------------------------------------------
# Backends: score(query, docs) -> list of floats, in input order
# ---------------------------------------------------------------------------

def pinned_or_current(name, spec, checkpoint_revision=None):
    """The commit to load: a resumed checkpoint's own revision, else the revision
    results/run_records.json records for this model's GPU run, else None (the
    current Hugging Face revision) with a printed warning. --unpinned skips the pins."""
    if checkpoint_revision:
        return checkpoint_revision
    if UNPINNED:
        print(f"  {name}: --unpinned, resolving the current Hugging Face revision", flush=True)
        return None
    from run_records import pinned_revision
    return pinned_revision(name, experiment="selfhost_check" if spec.get("selfhost") else "audit")


UNPINNED = False


def load_backend(name, spec, revision=None):
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(name, revision=revision)
    revision = Path(path).name
    dtype = getattr(torch, spec["dtype"]) if spec["dtype"] not in (None, "auto") else None

    if spec["backend"] == "crossencoder":
        from sentence_transformers import CrossEncoder
        kw = {"trust_remote_code": True, "max_length": 512}
        if dtype is not None:
            kw["model_kwargs"] = {"torch_dtype": dtype}
        model = CrossEncoder(name, revision=revision, device=device(), **kw)
        # Decoder-based rerankers (Qwen2/3 backbones) ship without a pad token; a
        # sequence-classification head needs one to find each sequence's last token.
        # (sentence-transformers 6 handles this itself for its LogitScore modules)
        if model.tokenizer.pad_token is None and hasattr(model, "model"):
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.model.config.pad_token_id = model.tokenizer.eos_token_id

        def score(query, docs):
            return [float(x) for x in model.predict([(query, d) for d in docs],
                                                    batch_size=spec["batch"],
                                                    show_progress_bar=False)]
        return score, revision

    if spec["backend"] == "qwen3_yesno":
        # The official recipe from the Qwen3-Reranker model card: an instruction
        # prompt, scored as P("yes") against P("no") at the final position.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype).eval().to(device())
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based "
                  "on the Query and the Instruct provided. Note that the answer can only be "
                  "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = tok.encode(suffix, add_special_tokens=False)
        task = "Given a web search query, retrieve relevant passages that answer the query"
        max_length = 8192

        def score(query, docs):
            out = []
            texts = [f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {d}" for d in docs]
            for i in range(0, len(texts), spec["batch"]):
                enc = tok(texts[i:i + spec["batch"]], padding=False, truncation="longest_first",
                          return_attention_mask=False,
                          max_length=max_length - len(pre) - len(suf))
                enc["input_ids"] = [pre + ids + suf for ids in enc["input_ids"]]
                b = tok.pad(enc, padding=True, return_tensors="pt").to(device())
                with torch.inference_mode():
                    last = model(**b).logits[:, -1, :]
                    pair = torch.stack([last[:, no], last[:, yes]], dim=1).float()
                    out += torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()
            return out
        return score, revision

    if spec["backend"] == "nemotron":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True,
                                            padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            name, revision=revision, trust_remote_code=True, torch_dtype=dtype).eval().to(device())
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id

        def score(query, docs):
            out = []
            texts = [f"question:{query} \n \n passage:{d}" for d in docs]  # model card template
            for i in range(0, len(texts), spec["batch"]):
                b = tok(texts[i:i + spec["batch"]], padding=True, truncation=True,
                        return_tensors="pt", max_length=512).to(device())
                with torch.inference_mode():
                    out += model(**b).logits.view(-1).float().cpu().tolist()
            return out
        return score, revision

    if spec["backend"] == "jina_listwise":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name, revision=revision, dtype="auto",
                                          trust_remote_code=True).eval().to(device())

        def score(query, docs):
            res = model.rerank(query, docs)
            got = {r["index"]: float(r["relevance_score"]) for r in res}
            return [got[i] for i in range(len(docs))]
        return score, revision

    raise ValueError(spec["backend"])


def layout(pairs, arm, seed):
    """Documents for one call plus (male index, female index) per pair."""
    docs, pos = [], []
    for t in pairs:
        if arm == "reversed":
            docs += [t.doc_female, t.doc_male]
            pos.append((len(docs) - 1, len(docs) - 2))
        else:
            docs += [t.doc_male, t.doc_female]
            pos.append((len(docs) - 2, len(docs) - 1))
    if arm == "shuffled":
        perm = list(range(len(docs)))
        random.Random(seed).shuffle(perm)
        where = {old: new for new, old in enumerate(perm)}
        docs = [docs[i] for i in perm]
        pos = [(where[m], where[f]) for m, f in pos]
    return docs, pos


# ---------------------------------------------------------------------------
# Status (atomic, per worker)
# ---------------------------------------------------------------------------

def write_status(worker, state):
    PARTIAL.mkdir(parents=True, exist_ok=True)
    tmp = PARTIAL / f".status_{worker}.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, PARTIAL / f"status_{worker}.json")


def running_contrast(rows, labels):
    share = {}
    for c in ("male", "female"):
        d = [r["score_male"] - r["score_female"] for r in rows if labels[r["occupation"]] == c]
        if not d:
            return None
        share[c] = sum(1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in d) / len(d) * 100
    return share["male"] - share["female"]


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------

def run_model(name, spec, groups, labels, worker, state, canary):
    import torch
    PARTIAL.mkdir(parents=True, exist_ok=True)
    ck = PARTIAL / f"{slug(name)}.jsonl"
    done, rows, ck_revision = set(), [], None
    if ck.exists():
        for line in ck.read_text().splitlines():
            rec = json.loads(line)
            if "_meta" in rec:
                ck_revision = rec["_meta"].get("revision")
            if "query" in rec:
                done.add(rec["query"])
                rows += rec["rows"]
    st = state["models"].setdefault(name, {})
    st.update({"total": len(groups), "done": len(done), "status": "loading", "updated": now()})
    write_status(worker, state)
    if len(done) == len(groups):
        st["status"] = "complete"
        write_status(worker, state)
        return

    t_load = time.time()
    score, revision = load_backend(name, spec, pinned_or_current(name, spec, ck_revision))
    if ck_revision and revision != ck_revision:
        raise RuntimeError(f"{name}: loaded {revision}, checkpoint was scored at {ck_revision}")
    st["load_seconds"] = round(time.time() - t_load, 1)

    # ---- canary ----
    hits, flat = 0, []
    for q, docs, target in canary:
        s = score(q, docs)
        flat += s
        hits += int(max(range(len(s)), key=s.__getitem__) == target)
    n_canary = len(canary)
    finite = all(x == x and abs(x) != float("inf") for x in flat)
    varied = len(set(round(x, 6) for x in flat)) > 1
    st["canary"] = {"top1": f"{hits}/{n_canary}", "finite": finite, "varied": varied}
    if not (finite and varied and hits >= 0.75 * n_canary):
        st["status"] = "FAILED canary"
        write_status(worker, state)
        print(f"[{worker}] {name}: canary FAILED {st['canary']}", flush=True)
        del score
        free_memory()
        return

    if not done:   # header line, once per model
        with ck.open("a") as fh:
            fh.write(json.dumps({"_meta": {
                "model": name, "revision": revision, "backend": spec["backend"],
                "dtype": spec["dtype"],
                "sentence_transformers": __import__("sentence_transformers").__version__,
                "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device(),
                "torch": torch.__version__, "script_version": SCRIPT_VERSION,
                "started": now(), "canary": st["canary"]}}) + "\n")

    st["status"] = "running"
    t0, n_new = time.time(), 0
    listwise = spec["backend"] == "jina_listwise"
    for q, pairs in groups.items():
        if q in done:
            continue
        out = []
        arms = ARMS if listwise else ("same",)
        scored = {}
        for arm in arms:
            docs, pos = layout(pairs, arm, zlib.crc32(f"{name}|{q}".encode()))
            s = score(q, docs)
            scored[arm] = [(s[m], s[f]) for m, f in pos]
        for i, t in enumerate(pairs):
            sm, sf = scored["same"][i]
            r = {"query": t.query, "query_type": t.query_type, "occupation": t.occupation,
                 "stereotype": t.stereotype, "name_pair": t.name_pair,
                 "template_style": t.template_style, "perturbation": t.perturbation,
                 "score_male": sm, "score_female": sf, "delta": sm - sf}
            if listwise:
                r["arms"] = {a: scored[a][i] for a in arms if a != "same"}
            out.append(r)
        with ck.open("a") as fh:
            fh.write(json.dumps({"query": q, "rows": out}) + "\n")
        rows += out
        done.add(q)
        n_new += 1
        rate = n_new / (time.time() - t0)
        st.update({"done": len(done), "rate_q_per_min": round(rate * 60, 1),
                   "eta_min": round((len(groups) - len(done)) / rate / 60, 1),
                   "contrast_so_far": running_contrast(rows, labels),
                   "tie_pct_so_far": round(100 * sum(r["delta"] == 0 for r in rows) / len(rows), 2),
                   "updated": now()})
        write_status(worker, state)

    st["status"] = "complete"
    st["finished"] = now()
    write_status(worker, state)
    del score
    free_memory()


# ---------------------------------------------------------------------------
# Finalize locally into the audit's raw-file schema
# ---------------------------------------------------------------------------

def finalize():
    from run_experiment import compute_metrics
    groups, _ = work_units()
    SELFHOST.mkdir(parents=True, exist_ok=True)
    for name, spec in MODELS.items():
        ck = PARTIAL / f"{slug(name)}.jsonl"
        if not ck.exists():
            print(f"  {name}: no checkpoint")
            continue
        meta, rows, queries = {}, [], set()
        for line in ck.read_text().splitlines():
            rec = json.loads(line)
            if "_meta" in rec:
                meta = rec["_meta"]
            else:
                queries.add(rec["query"])
                rows += rec["rows"]
        if len(queries) != len(groups):
            print(f"  {name}: incomplete ({len(queries)}/{len(groups)} queries)")
            continue
        out_dir = SELFHOST if spec.get("selfhost") else RESULTS
        out = out_dir / f"{name.split('/')[-1]}_full_raw.json"
        payload = {"model": name, "mode": "full", "perturbation_levels": ["names_and_pronouns"],
                   "query_types": sorted({r["query_type"] for r in rows}),
                   "scored_at": meta.get("started", "")[:10], "served_by": None,
                   "self_hosted": meta, "selfhost_of": spec.get("selfhost"),
                   "metrics": compute_metrics([{k: v for k, v in r.items() if k != "arms"}
                                               for r in rows]),
                   "results": rows}
        out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"  {name}: {len(rows)} rows -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Checkpointed scoring of open-weight rerankers")
    ap.add_argument("--worker", choices=("big", "small"))
    ap.add_argument("--model", action="append", help="restrict to these models")
    ap.add_argument("--status-name", help="name for this run's status file (default: the worker); "
                                          "lets an extra run share a worker's model list")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--unpinned", action="store_true",
                    help="ignore the revisions recorded in results/run_records.json and load "
                         "each model's current Hugging Face revision")
    ap.add_argument("--smoke", type=int, default=0,
                    help="score only this many queries, into results/partial_smoke/ (a test run "
                         "that never touches the real checkpoints)")
    args = ap.parse_args()
    if args.finalize:
        return finalize()
    if not args.worker:
        ap.error("--worker is required unless --finalize is given")
    global UNPINNED
    UNPINNED = args.unpinned

    groups, labels = work_units()
    canary = canary_set(groups, labels)       # always from the full set
    if args.smoke:
        global PARTIAL
        PARTIAL = RESULTS / "partial_smoke"
        groups = OrderedDict(list(groups.items())[:args.smoke])
    run_name = args.status_name or args.worker
    state = {"worker": run_name, "started": now(), "models": {}}
    todo = [m for m, s in MODELS.items()
            if s["worker"] == args.worker and (not args.model or m in args.model)]
    for name in todo:
        try:
            run_model(name, MODELS[name], groups, labels, run_name, state, canary)
        except Exception as e:  # noqa: BLE001  (one bad model must not stop the queue)
            state["models"].setdefault(name, {}).update(
                {"status": f"ERROR: {type(e).__name__}: {str(e)[:300]}", "updated": now()})
            write_status(run_name, state)
            print(f"[{run_name}] {name}: {type(e).__name__}: {e}", flush=True)
    state["finished"] = now()
    write_status(run_name, state)


if __name__ == "__main__":
    main()
