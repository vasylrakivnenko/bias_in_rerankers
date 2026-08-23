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
`code/occupations_bls.csv` and rendered in the paper's appendix.

```bash
venv/bin/python code/labels.py
# source: BLS CPS Table 11 via occupations_bls.csv
# occupations: 52  {'male': 18, 'neutral': 13, 'female': 21}
```

Analysis scripts must get labels from `labels.load_labels()` and record the
returned source string in their output JSON, so a reader can always tell which
taxonomy produced a number. Pass `--strict-labels` to make a script fail rather
than silently fall back to the superseded hand-made labels.

---

## Reproducing every table and figure

All of these read banked results and need no credentials. Run them from the
repository root, in this order (each reads the previous step's output).

| # | Command | Produces |
|---|---|---|
| 1 | `venv/bin/python code/analyze_single_stage.py` | `results/single_stage_summary.json`; `results/tex/single_stage_main.tex`, `occupation_consistency.tex`, `provenance.tex`, `robustness_template.tex`, `robustness_query.tex`, `robustness_namepair.tex` |
| 2 | `venv/bin/python code/compounding_experiment.py` | `results/compounding_bge_pipeline.json`, `results/compounding_all_rerankers.json`, `results/tex/compounding_sweep.tex` |
| 3 | `venv/bin/python code/deidentification_experiment.py` | `results/deidentification_<model>.json` (one per local re-ranker), `results/deidentification_summary.json`, `results/tex/deid_conditions.tex`, `results/tex/deidentification.tex` |
| 4 | `venv/bin/python code/generate_appendix_tables.py` | `results/tex/documents_appendix.tex`, `results/tex/occupations_table.tex` |
| 5 | `venv/bin/python legal_paper/generate_figures.py` | `legal_paper/figures/fig_single_stage.pdf`, `fig_pipeline.pdf`, `fig_deid.pdf` |
| 6 | `cd legal_paper && python3 build.py` | `legal_paper/main.tex`, `main.pdf`, `../legal_paper_overleaf.zip` |

Useful flags: `--bootstrap N` (cluster-bootstrap replicates, default 1000) on
1–3; `--main-only` on 2 to skip the 14-model sweep; `--model NAME` (repeatable)
on 3, which by default scores all four locally cached re-rankers. Step 3 also
runs a utility-retention test across all four query phrasings by default
(`--no-utility` skips it; `--utility-query-types` narrows it).
`python3 build.py --figures` runs step 5 for you before building.

**Numbers are never typed into the prose.** Every statistic in `paper.md` is a
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
legal_paper/              the paper (paper.md -> main.pdf via build.py)
```

A deeper, unreleased extension of this work — real résumés, a name-detection
coverage test, a residual-gender-leakage probe — exists privately and is not
part of this repository; nothing in the paper depends on it.

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
