# Legal paper: Gender Bias in Automated Hiring — the Case for De-Identification

First draft of the empirical-with-legal-angle paper arguing for regex gender
de-identification in two-stage neural ranking pipelines (retriever + re-ranker).

## Editing workflow (the reusable pipeline)
**You edit `paper.md`; everything else is generated.**

```
# edit prose in paper.md (in VS Code or any editor), then:
python3 build.py              # paper.md -> main.tex -> main.pdf + the Overleaf zip
python3 build.py --figures    # also regenerate the charts from data first
python3 build.py --no-zip     # build the PDF only, skip the zip
```

`build.py` converts the Markdown in `paper.md` into `main.tex`, compiles it with
`pdflatex` (twice), and writes `../legal_paper_overleaf.zip`. No third-party packages
are needed — just `python3` and `pdflatex`.

**What you edit vs. what is automatic**
- Prose, headings, lists, emphasis, quotes, citations → edit in `paper.md`.
  Conventions: `*italic*`, `**bold**`, `"quotes"`, `` `code` ``, `[@citekey]`, `$math$`;
  `#`/`##`/`###` for section / subsection / run-in heading.
- Tables, figures, and the reference list live in ` ```latex ` raw blocks inside
  `paper.md` — they pass through untouched. Leave them alone unless you know LaTeX.
- The LaTeX preamble (packages, title) lives in `build.py` (the `PREAMBLE` string).

## Files
- `paper.md` — **the source you edit.**
- `build.py` — the build pipeline (paper.md → main.tex → PDF + zip).
- `main.tex` — generated artifact (do not hand-edit; `build.py` overwrites it).
- `main.pdf` — compiled draft (currently 10 pp.).
- `generate_figures.py` — regenerates the three charts from `../results/*.json`.

## Empirical backbone (all reproducible from `../code` + `../results`)
| Claim | Script | Output |
|---|---|---|
| Single-stage re-ranker bias, 15 models (Table 1) | scoring done in `../code/run_experiment.py` | `../results/*_full_raw.json` |
| Pipeline compounding / shortlist (Figure 2) | `../code/compounding_experiment.py` | `../results/compounding_bge_pipeline.json` |
| De-identification mitigation (Figure 3) | `../code/deidentification_experiment.py` | `../results/deidentification_bge.json` |

Run experiments with the project venv, e.g.:
```
HF_HUB_OFFLINE=1 ../venv/bin/python ../code/deidentification_experiment.py
```
(embedding + reranker models are cached locally; no API keys needed for the bge stack.)

## Headline numbers

**Do not hand-copy these into the paper.** Every statistic in `paper.md` is a
`{{PLACEHOLDER}}` resolved from `../results/*.json` via `numbers.json`, and the build fails
on any unresolved one. These are a sanity check for a human reader only.

- **14 of 14 models** favor the man more often on male-typed
  than on female-typed jobs (median spread 31.7 pts). Only
  13 order all three categories male > neutral > female.
- **The default lean is model-specific.** On evenly split occupations the median model sits at
  56.1% favor-male, but 8 models lean male and
  6 lean female (range 26.9-87.8%).
- **Ties are counted as ties** (up to 20.6% of pairs for some APIs), never credited to
  either side. The old `delta > 0` arithmetic silently gave every tie to the woman.
- Full-pipeline top-3 shortlist: **79% male overall**,
  77% on evenly split roles. Across all 14
  re-rankers sharing the same retriever it spans 44%
  (semantic-ranker-default-004) to 85% (ibm-granite-granite-embedding-reranker-english-r2) -- the shortlist is a
  property of the re-ranker, not of two-stage pipelines as such.
- **Full de-id -> 100% ties by construction** (a check on the symmetry argument, not a finding).
  Names-only makes the direction near-deterministic
  (99.7% on male-typed jobs) while the mean
  score gap *falls* (0.0358 -> 0.0334).
- **A part-of-speech-aware they/them/their mapping ties every pair**, exactly like the one-token
  rule, and still reads as English. Only the naive mapping fails, leaving a residue on
  25% of pairs.
- Occupation labels come from **BLS CPS Table 11** (% women), not hand judgement: 82 occupations
  -> 52 kept. See `../code/occupations_bls.csv`, `../data/PROVENANCE.md`.


## References
All 20 entries verified August 2026. See `references_and_legal.md` for the entry-by-entry check,
the legal corrections, and the finding that EU AI Act deployer duties are deferred to
2 December 2027 by Regulation (EU) 2026/1744.
