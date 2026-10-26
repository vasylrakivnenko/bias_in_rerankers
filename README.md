# Gender bias in neural re-rankers — code and data

Research code for an empirical-legal paper on gender bias in the *re-ranking*
stage of two-stage retrieval pipelines used for candidate search, and on cheap
de-identification as a mitigation.

This is a **broad audit** of commercial and open re-rankers, scored on synthetic
counterfactual pairs — two documents identical except the candidate's name
and pronouns. These were scored once against paid APIs; the raw scores are
banked in `results/*_full_raw.json` and **are never re-run**. Every table and
figure in the paper is computed from those files, locally and for free. On top
of that audit, one representative retriever-plus-re-ranker pipeline and a
de-identification mitigation are each tested more deeply — also entirely on
locally cached model weights, no API credentials involved.

The paper itself lives in `legal_paper/` and has its own README.

---

## Setup

```bash
python3.14 -m venv venv
venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions the reported numbers were produced with
(Python 3.14.4, macOS/Apple Silicon; torch uses the `mps` device — pass
`--device cpu` or `--device cuda` where a script takes `--device`).

Local model weights are read from `~/.cache/huggingface`. Once they are cached,
prefix any command with `HF_HUB_OFFLINE=1` to guarantee no network access.

---

## Getting the data

Tidy, derived data files are committed. The raw third-party downloads under
`data/raw/` are not (≈218 MB, all re-downloadable). **`data/PROVENANCE.md` is
the authority**: for every file it gives the URL, the licence, the retrieval
date, and — for the two sites that block scripted access — the exact manual
step.

To rebuild everything under `data/` from scratch:

```bash
# 1. O*NET 30.3 (CC BY 4.0) — job descriptions and SOC codes
curl -L -o data/raw/onet_db_text.zip https://www.onetcenter.org/dl_files/database/db_30_3_text.zip
unzip -q data/raw/onet_db_text.zip -d data/raw/onet_extract
cp "data/raw/onet_extract/db_30_3_text/Occupation Data.txt" "data/raw/onet_extract/db_30_3_text/Task Statements.txt" data/onet/

# 2. Résumé corpus (CC0) — 2,484 livecareer résumé examples
curl -L -o data/raw/livecareer_Resume_full.csv \
  https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv
venv/bin/python data/build_resumes_csv.py

# 3. SSA baby names (public domain) — name -> gender table
#    www.ssa.gov 403s a plain curl; see PROVENANCE §3 for the header set and
#    the browser fallback.
unzip -q data/raw/ssa_names.zip -d data/raw/ssa_names
venv/bin/python data/build_name_gender.py

# 4. BLS CPS Table 11 (public domain) — % women by occupation
#    www.bls.gov 403s ALL scripted requests. PROVENANCE §1 gives the archived
#    copy used here and the browser download step.
venv/bin/python data/build_bls_table11.py

# 5. The two hand-made mapping tables
venv/bin/python data/build_occupations_bls.py   # -> code/occupations_bls.csv
venv/bin/python data/build_category_to_soc.py   # -> data/category_to_soc.csv
```

### Occupation labels

`code/labels.py` is the single source of truth for the occupation →
male/female/neutral labels. It reads `code/occupations_bls.csv`, which is
derived from **BLS CPS Table 11, 2025 annual averages** (% women): male if
< 30 % women, female if > 70 %, neutral if 40–60 %, dropped otherwise. Of the
82 occupations scored, 52 have a usable share and are kept; the other 30 (no
matching BLS occupation, a suppressed estimate, or a share in the 30–40 % /
60–70 % ambiguous bands) are dropped, with the reason for each recorded in
`code/occupations_bls.csv` and released as `results/tex/occupations_table.tex`.

```bash
venv/bin/python code/labels.py
# source: BLS CPS Table 11 via occupations_bls.csv
# occupations: 52  {'male': 18, 'neutral': 13, 'female': 21}
```

Analysis scripts must get labels from `labels.load_labels()` and record the
returned source string in their output JSON, so a reader can always tell which
taxonomy produced a number. Scripts fail if `code/occupations_bls.csv` is missing
or unreadable; pass `--allow-legacy-labels` only to reproduce the superseded
hand-made labels deliberately.

---

## Reproducing every table and figure

The short empirical PDF builds directly from released results with
`cd legal_paper && python3 build.py`; add `--paper legal` for the longer legal
manuscript. Neither build needs a corpus download, model weights, or credentials.
To regenerate the derived outputs, run the commands below from the repository root. Step 3 uses
banked local results with `--reuse`; without that flag it loads model weights.

| # | Command | Produces |
|---|---|---|
| 1 | `venv/bin/python code/analyze_single_stage.py` | `results/single_stage_summary.json`; `results/tex/single_stage_main.tex`, `occupation_consistency.tex`, `provenance.tex`, `robustness_template.tex`, `robustness_query.tex`, `robustness_namepair.tex` |
| 2 | `venv/bin/python code/compounding_experiment.py` | `results/compounding_bge_pipeline.json`, `results/compounding_all_rerankers.json`, `results/tex/compounding_sweep.tex` |
| 3 | `venv/bin/python code/deidentification_experiment.py --reuse` | `results/deidentification_<model>.json` (one per local re-ranker), `results/deidentification_summary.json`, `results/tex/deid_conditions.tex`, `results/tex/deidentification.tex` |
| 4 | `venv/bin/python code/generate_appendix_tables.py` | `results/tex/documents_appendix.tex`, `results/tex/occupations_table.tex` |
| 4a | `venv/bin/python code/vendor_docs_audit.py` | `results/vendor_docs_audit_summary.json`, `results/tex/vendor_docs_audit.tex` |
| 4b | `venv/bin/python code/reconstruct_realistic_sampling.py` | `results/realistic_pool_sampling.json` (requires the downloaded résumé corpus; no scoring) |
| 4c | `venv/bin/python code/companion_whole_resume.py` | `results/companion_whole_resume.json`, `results/tex/companion_whole_resume.tex` (from `results/companion_full_window.json` and `results/length_pronoun/`; the résumé word counts need the corpus and are otherwise kept from the previous output) |
| 5 | `venv/bin/python legal_paper/generate_figures.py` | `legal_paper/figures/fig_single_stage.pdf`, `fig_pipeline.pdf`, `fig_deid.pdf`, `fig_length_pronoun.pdf` |
| 6 | `cd legal_paper && python3 build.py` | `legal_paper/main.tex`, `main.pdf`, `../empirical_paper_overleaf.zip`, `../empirical_paper_arxiv.zip` (use `--paper legal` for the legal versions) |

Useful flags: `--bootstrap N` (cluster-bootstrap replicates, default 1000) on
1–3; `--main-only` on 2 to skip the 14-model sweep; `--model NAME` (repeatable)
on 3, which by default scores all four locally cached re-rankers. Step 3 also
runs a utility-retention test across all four query phrasings by default
(`--no-utility` skips it; `--utility-query-types` narrows it).
`python3 build.py --figures` runs step 5 for you before building.

**Numbers are never typed into the prose.** Every statistic in either manuscript is a
`{{PLACEHOLDER}}` resolved through `legal_paper/numbers.json` from a path in a
`results/*.json`; an unresolved placeholder fails the build (`--draft` renders
the gaps in red instead, so the paper can still be read while data lands).
Regenerate the JSON before rebuilding the PDF.

### Re-scoring the commercial APIs (costs money — normally skip)

`code/run_experiment.py` is the only script that calls a paid API. It produced
`results/*_full_raw.json` in February 2026 and should not need to run again.
Voyage AI credentials come from the environment; Google Vertex AI and Cohere
are both called over plain HTTPS with `requests` (Google via a `gcloud auth
print-access-token` subprocess call) — neither needs a vendor SDK installed.

---

## Layout

```
code/                     analysis and experiment scripts
  labels.py               occupation -> label, single source of truth
  occupations_bls.csv     BLS-derived labels (82 rows; 30 marked `drop`)
  synthetic_dataset.py    counterfactual triples; legacy labels kept for transparency
  deid_transforms.py      the de-identification variants under test
  run_experiment.py       API scoring (already run; do not re-run)
  analyze_single_stage.py, compounding_experiment.py,
  deidentification_experiment.py,
  generate_appendix_tables.py                         the audit + appendix data
data/                     tidy third-party data + the build_*.py that make it
  PROVENANCE.md           URLs, licences, dates, manual steps  <- read this
  bls/, onet/, resumes/, name_gender.csv, category_to_soc.csv
  raw/                    downloads (git-ignored, re-downloadable)
results/                  scored pairs, summary JSON, .tex table fragments
legal_paper/              both papers (.md -> main.pdf via build.py)
```

The companion real-résumé experiment's first run, which cut every input at 512 tokens, is
released in `results/realistic_pool_four_rerankers.json`, with the experiment and its
`ner_coverage.py` helper; the paper takes its pools and design counts from it and its results
from the whole-résumé runs described below. `code/reconstruct_realistic_sampling.py` reconstructs
sampling without scoring: 1,952 distinct résumés entered the pools from a corpus
of 2,483. The source corpus can be rebuilt using `data/PROVENANCE.md`.

To rerun the companion experiment locally after obtaining the data and model
weights: `venv/bin/python code/realistic_pool_experiment.py --rerankers bge-m3
bge-base ms-marco jina --pools-per-query 8 --span-cache results/.span_cache_6000.json
--out results/realistic_pool_replication.json` (add `--resume` to reuse models
finished before a crash). Shortlist membership uses an exact-k rule: every
scoring selects exactly k candidates, with score ties broken by a seeded
per-(pool, résumé) key that is the same in both counterfactual forms
(`script_version` in the JSON). Comparison-level rows, pool membership and
filler scores are written to `results/realistic_pool_four_rerankers_rows.json.gz`.
The banked run used cached name detections; a fresh detector/model run may differ.
Sampling reconstruction reproduces membership and counts, not the model scores.
The padding and same-sex placebo claims were removed because their supporting
artifacts could not be recovered. Separate leakage and utility extensions remain
outside this release.

The banked run truncated every (query, résumé) pair to 512 tokens. A full-window
re-run (October 2026) scored the same pools and pairs at each model's own input
limit, writing `results/realistic_pool_full_window_<model>.json` and its rows file:
`bge-m3` at `--max-length 8192` (every résumé whole) and `jina` at
`--max-length 1024`, both with `--token-budget 8192 --attention-budget 25000000`,
which plan batches by length so the longest inputs fit in memory; and two hosted
rerankers on whole documents, `voyage-rerank-3` (truncation off) and
`cohere-v4-pro` (Cohere rerank-v4.0-pro on Azure, `max_tokens_per_doc` 32000),
with `--max-length 32000`. Each run added `--expect-pool-sha256` with the banked
membership hash and `--resume`, which reuses every finished batch or request after
a crash; `--limit-pools N` runs a canary on N pools. Voyage returns some responses
with every score rounded to bfloat16; the script detects and re-sends those (see
`APIScorer`). Four more rerankers were scored on whole documents only:
`voyage-rerank-2.5`; `gte-modernbert` and `granite-r2` on the Mac at 8,192 tokens with
the audit's pinned `--revision`; and `zerank-2` on an RTX 4090 (`code/runpod/`) at
8,192 tokens with `--model-dtype bfloat16 --batch-size 1 --device cuda`, one input per
batch because its bfloat16 scores shift with batch composition. Cohere's Rerank 5
(`cohere-v5-pro`, `cohere-v5-fast`), released on 2026-10-21 after the audit, was scored the
same way on Cohere's own API (`COHERE_API_KEY`; a trial key's 10 calls a minute are read from
the response and respected), and on the audit's 8,320 retained synthetic pairs by
`code/post_audit_synthetic.py`. That script regenerates the pairs, checks them and their
request layout against the banked Cohere v4.0 files, and writes
`results/post_audit/<model>_synthetic.json` -- deliberately not a `*_full_raw.json`, so the
29-model audit is unchanged.

The rest of the audit's rerankers that read a whole résumé were added the same way (the
longest (query, résumé) input is about 4,700 tokens). Hosted, each on the backend that served
it in the audit: `voyage-rerank-2`, `voyage-rerank-3-lite`, `voyage-rerank-2.5-lite`,
`voyage-rerank-1`, `voyage-rerank-2-lite`, `cohere-v4-fast` (Azure), `cohere-v3.5` (Cohere's
API), `qwen3-8b` (Fireworks) and `nemotron-vl` (OpenRouter). On an RTX 4090, at the audit's
pinned revisions: `qwen3-0.6b`, `qwen3-4b` and `nemotron-1b` through the audit's own scoring
recipes in `code/score_open_models.py` with the input limit raised (`RecipeScorer`), one
résumé per batch; `mxbai-large-v2` as a cross-encoder (sentence-transformers 6.1); and
`jina-v3`, which is listwise and so takes the hosted models' request layout, with documents
cut at 8,192 tokens instead of its default 2,048. Each passed a two-pool canary and a check
that the end of a long résumé moves its score; each GPU model first reproduced its banked
synthetic scores exactly. jina-v3 is order-sensitive -- with each pair's female form first,
0.7% of its whole-résumé winners survive -- so it was also run with `--female-first`
(`results/realistic_pool_full_window_jina-v3_female_first.json`), and
`code/companion_full_window.py` averages each pair's gap over the two orders, on the synthetic
side too (the audit's reversed arm). Google's three rankers cut every record at 512 or 1,024 tokens, and
bge-reranker-base and -large and the two MiniLM cross-encoders at 512, so they have no
whole-résumé run.
`code/companion_full_window.py` compares the local models' full-window runs with the
512-token run, reports the others beside their synthetic-audit contrasts, and tests
every run against the BLS women's share across all 21 job categories that have one
(`results/companion_full_window.json`).

Those runs swap every name and pronoun a résumé carries, but many of them belong to someone
else (a supervisor, a book's author, a school named after a person).
`data/owner_marker_labels.json` records, for each of the 295 scored résumés with a detected
name or a gendered pronoun, which occurrences refer to the résumé's owner. Two independent
labelling passes (Claude subagents, October 2026) agreed on 333 of 333 names and 270 of 273
pronouns; a third reading by the same model settled the disagreements, and no person has
checked the labels. The file stores character offsets and a sha256 of the text there, never the names,
and the labelling passes' working files, which held the names, are not released.
With `--marker-labels`, `realistic_pool_experiment.py` gives every résumé a prepended name,
puts it wherever the owner is named, and changes only the owner's pronouns; `--cut-words N`
keeps the first N words after the name, and `--owner-line` inserts a fourteen-word sentence
with three of the owner's pronouns (`OWNER_LINE`). `code/run_length_pronoun.sh` runs the
twelve length x pronoun conditions for one reranker (its header lists the settings used for
Voyage rerank-3 and rerank-2, Cohere Rerank 5 Pro, Qwen3-Reranker-8B, bge-reranker-v2-m3 and
mxbai-rerank-large-v2) into `results/length_pronoun/`. `code/companion_whole_resume.py` then
writes `results/companion_whole_resume.json` and `results/tex/companion_whole_resume.tex`
(the whole-résumé table and the numbers of the paper's whole-résumé subsection and Appendix E), and
`legal_paper/generate_figures.py` draws `fig_length_pronoun.pdf` from it. Fireworks' Qwen
scores vary slightly between identical calls, so a re-run of that model will not match exactly.

### What is and is not tracked

`data/raw/` and the derived `data/resumes/*.csv` are git-ignored: they are large
and every one of them is re-downloadable from `data/PROVENANCE.md`.

`results/*_full_raw.json` (~72 MB) **is** tracked, deliberately. Those files are
the output of paid commercial APIs and cannot be regenerated without spending
money again; no single file approaches GitHub's 50 MB warning threshold. If the
repository ever has to slim down, attach them to a release — or
`tar -czf results_raw.tgz results/*_full_raw.json` — *before* ignoring them.

## Licences of included data

O*NET 30.3 is **CC BY 4.0** and requires attribution in the paper. The résumé
corpus is **CC0 1.0**. BLS and SSA data are U.S. Government works in the public
domain. Full citations are in `data/PROVENANCE.md`.
