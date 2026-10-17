# Papers and reproducible build

This directory contains a short empirical paper for arXiv and a longer paper
with a legal angle. One build pipeline generates either manuscript from the
same machine-readable results.

## Editing workflow (the reusable pipeline)
**The default source is `empirical_paper.md`; everything else is generated.**

```
# edit prose in empirical_paper.md, then:
python3 build.py              # empirical PDF + Overleaf and arXiv source zips
python3 build.py --paper legal  # paper.md -> main.tex/main.pdf + legal zip
python3 build.py --figures    # also regenerate the charts from data first
python3 build.py --no-zip     # build the PDF only, skip the zip
python3 build.py --draft      # build even with unresolved {{PLACEHOLDER}}s (shown in red)
python3 build.py --data-root D   # look up numbers.json under D (fixture testing)
```

`build.py` converts the selected Markdown source into `main.tex`, compiles it
with `pdflatex` twice, and writes an Overleaf bundle plus a minimal arXiv source
bundle. The arXiv zip contains only `main.tex` and the three PDF figures. No
third-party Python packages are needed.

**What you edit vs. what is automatic**
- Prose, headings, lists, emphasis, quotes, citations → edit in the selected Markdown source.
  Conventions: `*italic*`, `**bold**`, `"quotes"`, `` `code` ``, `[@citekey]`, `$math$`;
  `#`/`##`/`###` for section / subsection / run-in heading.
- Tables, figures, and the reference list live in ` ```latex ` raw blocks inside
  the selected source — they pass through untouched. Leave them alone unless you know LaTeX.
- **Every statistic is a `{{PLACEHOLDER}}`**, never a typed number. Each one is
  defined in `numbers.json` with the exact path into a `results/*.json` file (or
  a small arithmetic expression over two such paths, or a `.tex` fragment to
  insert verbatim). An unresolved placeholder fails the build and prints every
  missing name with what it needs — this is deliberate: it is the mechanism
  that keeps the prose from drifting away from the data.
- The LaTeX preamble (packages, title) lives in `build.py` (the `PREAMBLE` string).

## Files
- `empirical_paper.md` — **the default short empirical paper.**
- `paper.md` — the longer empirical and legal paper (`--paper legal`).
- `numbers.json` — every placeholder's data source; read its own `_README` entry.
- `build.py` — the shared build pipeline (selected source + numbers → PDF + zip).
- `main.tex` — generated artifact (do not hand-edit; `build.py` overwrites it).
- `main.pdf` — compiled draft.
- `../empirical_paper_arxiv.zip` — minimal source archive ready for arXiv upload.
- `generate_figures.py` — regenerates the three charts from `../results/*.json`.

## Empirical backbone (all reproducible from `../code` + `../results`)
| Claim | Script | Output |
|---|---|---|
| Single-stage re-ranker bias, 29 models (Tables A1–A3) | `../code/analyze_single_stage.py`, over the banked `../code/run_experiment.py` scores | `../results/single_stage_summary.json` |
| Pipeline shortlist, one pair and swept over the 28 pointwise re-rankers (Figure 2, Table A7) | `../code/compounding_experiment.py` | `../results/compounding_bge_pipeline.json`, `../results/compounding_all_rerankers.json` |
| De-identification mitigation, one model and all four local re-rankers (Figure 3, Tables A8–A9) | `../code/deidentification_experiment.py` | `../results/deidentification_bge.json`, `../results/deidentification_summary.json` |
| The documents, query forms, name pairs, and full occupation list (Appendix A; the occupation list is released, not printed) | `../code/generate_appendix_tables.py` | `../results/tex/documents_appendix.tex`, `../results/tex/occupations_table.tex` |
| Vendor documentation and public weight availability | `../code/vendor_docs_audit.py` | `../results/vendor_docs_audit_summary.json` |
| Open-weight rerankers scored on a rented GPU (checkpointed, canary-gated) | `../code/score_open_models.py` | `../results/*_full_raw.json`, `../results/selfhost/` |
| Request-order test for API rerankers and listwise order arms | `../code/order_reversal_test.py` | `../results/order_reversal_test.json`, `../results/order_test/` |
| Cross-hardware and self-hosted vs API reproducibility | `../code/reproducibility_checks.py` | `../results/reproducibility_checks.json` |
| Interval sensitivity (BLS-category and occupation × name resampling) | `../code/interval_sensitivity.py` | `../results/interval_sensitivity.json` |
| Paired de-identification statistics | `../code/deid_statistics.py` | `../results/deid_statistics_bge.json` |
| Pinned local model revisions and runtime | `../code/model_manifest.py` | `../results/model_manifest.json` |
| Companion results table (Appendix F) | `../code/companion_table.py` | `../results/tex/companion_table.tex` |
| Real-résumé companion run and sampled population | `../code/realistic_pool_experiment.py`, `../code/reconstruct_realistic_sampling.py` | `../results/realistic_pool_four_rerankers.json`, `../results/realistic_pool_sampling.json` |

All inputs for `python3 build.py` are included in the release; building the PDF
needs neither downloaded résumés nor model weights. Reconstructing the companion
sampling requires the corpus described in `../data/PROVENANCE.md`. It reproduces
membership and counts without rescoring; see the root README for the full run.

Run experiments with the project venv, e.g.:
```
HF_HUB_OFFLINE=1 ../venv/bin/python ../code/deidentification_experiment.py
```
(retriever and re-ranker weights are all cached locally; no API keys needed
for anything the paper actually reports — the twelve commercial-API endpoints
were scored once and are banked in `../results/*_full_raw.json` (eleven distinct
models after dropping the duplicate `semantic-ranker-default-003`); the three
open-weight rerankers run locally for free.)

## Headline numbers

**Do not hand-copy these into a paper.** Every statistic in either manuscript is a
`{{PLACEHOLDER}}` resolved from `../results/*.json` via `numbers.json`, and the build fails
on any unresolved one. These are a sanity check for a human reader only.

- **27 of 29 rerankers** favor the man more often on male-typed than on
  female-typed jobs, with an occupation-clustered interval above zero (tie-aware;
  median within-model spread 31.4 pts). The exceptions are jina-reranker-v3
  (listwise, +0.8) and gte-reranker-modernbert-base (−6.7, interval touching zero).
  The largest gaps come from zerank-2 (+65) and Voyage rerank-3 (+61).
- **The default lean on evenly split occupations is model-specific and not
  small.** Tie-aware median 64% favor-male, but 21 of 29 lean male and 8 lean
  female, from 31% to 91%. A buyer cannot tell which way a given product leans
  without testing it.
- **Ties are counted as ties** (up to 28.5% of pairs for the hosted Qwen3-8B) and
  split evenly (half a win each side) wherever a single share is reported.
- Full-pipeline top-3 shortlist: **79% male overall, 77% on evenly split
  roles.** Across the 28 pointwise rerankers sharing the same retriever it spans
  44% (rerank-3-lite) to 85% (ibm-granite-…-r2); 11 of 28 push the shortlist below
  retrieval alone. The listwise jina-reranker-v3 is excluded, because its scores
  depend on the other documents in a request.
- **Request order** was tested on all 15 API rerankers: 14 unmoved, rerank-2 +2.3
  pts on a 57-pt contrast. Listwise jina-reranker-v3 flips its winner in 35% of
  pairs when each pair is reversed.
- **De-identification, run correctly (case preserved).** Deleting names but
  leaving pronouns does *not* shrink the average score gap — it is slightly
  *larger* (0.0358 → 0.0380) — while pushing the direction to near-total
  consistency (87.0% → 98.6% on male-typed jobs). This reverses the
  conclusion of an earlier, lowercasing-confounded run of this experiment;
  see `code/deidentification_experiment.py`'s module docstring for the fix.
  Neutralising pronouns but keeping names does the opposite: gap down to
  0.0236. Which channel dominates is **not the same on every re-ranker** —
  see Table A8 in the paper.
- **Full de-id → 100% ties by construction** (a check on the symmetry
  argument, not a finding). The following-word heuristic also equalises the
  tested templates. It is not a POS tagger, fails on ordinary sentences and
  does not repair verb agreement; it is not a validated grammatical alternative.
- **Synthetic occupation matching showed no loss.** Top-1 accuracy across all
  four query phrasings: 99.4% before, 99.8% after the full transform. This does
  not establish equal effectiveness in selecting qualified applicants within a job.
- Occupation labels come from **BLS CPS Table 11** (% women), not hand
  judgement: 82 occupations scored, 52 kept (18 male / 21 female / 13
  neutral); the other 30 (no matching BLS occupation, a suppressed estimate,
  or an ambiguous 30–40%/60–70% share) are recorded with their reason in
  `../code/occupations_bls.csv` and released as `../results/tex/occupations_table.tex`.
- One of the four locally-run re-rankers (jina-reranker-v2) did not exactly
  reproduce its own banked scores on a plain local re-run six months later
  (max score difference 0.012 on its own ~0.15–0.84 range) — flagged in the
  rendered note under Table A8, not hidden.


## References
The longer `paper.md` bibliography was checked individually (title,
venue, DOI/URL) and are current as of August 2026 — including that EU AI Act
deployer duties (Arts. 26–27) are deferred to 2 December 2027 by Regulation
(EU) 2026/1744, which the paper's footnotes and bibliography already reflect.
