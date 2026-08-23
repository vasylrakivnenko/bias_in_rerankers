# Legal paper: Gender Bias in Automated Hiring — the Case for De-Identification

The empirical-with-legal-angle paper arguing for gender de-identification in
two-stage neural ranking pipelines (retriever + re-ranker), and a build
pipeline that generates it from data rather than from hand-typed numbers.

## Editing workflow (the reusable pipeline)
**You edit `paper.md`; everything else is generated.**

```
# edit prose in paper.md (in VS Code or any editor), then:
python3 build.py              # paper.md -> main.tex -> main.pdf + the Overleaf zip
python3 build.py --figures    # also regenerate the charts from data first
python3 build.py --no-zip     # build the PDF only, skip the zip
python3 build.py --draft      # build even with unresolved {{PLACEHOLDER}}s (shown in red)
python3 build.py --data-root D   # look up numbers.json under D (fixture testing)
```

`build.py` converts the Markdown in `paper.md` into `main.tex`, compiles it with
`pdflatex` (twice), and writes `../legal_paper_overleaf.zip`. No third-party
LaTeX packages are needed beyond a standard TeX Live install — just `python3`
and `pdflatex`.

**What you edit vs. what is automatic**
- Prose, headings, lists, emphasis, quotes, citations → edit in `paper.md`.
  Conventions: `*italic*`, `**bold**`, `"quotes"`, `` `code` ``, `[@citekey]`, `$math$`;
  `#`/`##`/`###` for section / subsection / run-in heading.
- Tables, figures, and the reference list live in ` ```latex ` raw blocks inside
  `paper.md` — they pass through untouched. Leave them alone unless you know LaTeX.
- **Every statistic is a `{{PLACEHOLDER}}`**, never a typed number. Each one is
  defined in `numbers.json` with the exact path into a `results/*.json` file (or
  a small arithmetic expression over two such paths, or a `.tex` fragment to
  insert verbatim). An unresolved placeholder fails the build and prints every
  missing name with what it needs — this is deliberate: it is the mechanism
  that keeps the prose from drifting away from the data.
- The LaTeX preamble (packages, title) lives in `build.py` (the `PREAMBLE` string).

## Files
- `paper.md` — **the source you edit.**
- `numbers.json` — every placeholder's data source; read its own `_README` entry.
- `build.py` — the build pipeline (paper.md + numbers.json → main.tex → PDF + zip).
- `main.tex` — generated artifact (do not hand-edit; `build.py` overwrites it).
- `main.pdf` — compiled draft.
- `generate_figures.py` — regenerates the three charts from `../results/*.json`.

## Empirical backbone (all reproducible from `../code` + `../results`)
| Claim | Script | Output |
|---|---|---|
| Single-stage re-ranker bias, 14 models (Tables A1–A3) | `../code/analyze_single_stage.py`, over the banked `../code/run_experiment.py` scores | `../results/single_stage_summary.json` |
| Pipeline shortlist, one pair and swept over all 14 re-rankers (Figure 2, Table A7) | `../code/compounding_experiment.py` | `../results/compounding_bge_pipeline.json`, `../results/compounding_all_rerankers.json` |
| De-identification mitigation, one model and all four local re-rankers (Figure 3, Tables A8–A9) | `../code/deidentification_experiment.py` | `../results/deidentification_bge.json`, `../results/deidentification_summary.json` |
| The documents, query forms, name pairs, and full occupation list (Appendix B) | `../code/generate_appendix_tables.py` | `../results/tex/documents_appendix.tex`, `../results/tex/occupations_table.tex` |

Run experiments with the project venv, e.g.:
```
HF_HUB_OFFLINE=1 ../venv/bin/python ../code/deidentification_experiment.py
```
(retriever and re-ranker weights are all cached locally; no API keys needed
for anything the paper actually reports — the fifteen commercial-API models'
scores were bought once and are banked in `../results/*_full_raw.json`.)

## Headline numbers

**Do not hand-copy these into the paper.** Every statistic in `paper.md` is a
`{{PLACEHOLDER}}` resolved from `../results/*.json` via `numbers.json`, and the build fails
on any unresolved one. These are a sanity check for a human reader only.

- **14 of 14 models** favor the man more often on male-typed than on
  female-typed jobs (tie-aware, median spread 31.6 pts). 13 of 14 order all
  three categories male > neutral > female; the one exception (ms-marco)
  favors male-typed jobs over neutral ones, but neutral over female-typed.
- **The default lean on evenly split occupations is model-specific and not
  small.** Tie-aware median 63.0% favor-male, but only because 10 of 14 lean
  male and 4 lean female — the range runs 32.8% to 87.8%. A buyer cannot tell
  which way a given product leans without testing it.
- **Ties are counted as ties** (up to 20.6% of pairs for some APIs) and split
  evenly (half a win each side) wherever a single share is reported, matching
  Figure 1 / Table A2 / the body text. The old `delta > 0` arithmetic silently
  gave every tie to the woman.
- Full-pipeline top-3 shortlist: **79% male overall, 77% on evenly split
  roles.** Across all 14 re-rankers sharing the same retriever it spans 44%
  (semantic-ranker-default-004) to 85% (ibm-granite-…-r2) — the shortlist is
  a property of the re-ranker, not of two-stage pipelines as such.
- **De-identification, run correctly (case preserved).** Deleting names but
  leaving pronouns does *not* shrink the average score gap — it is slightly
  *larger* (0.0358 → 0.0380) — while pushing the direction to near-total
  consistency (87.0% → 98.6% on male-typed jobs). This reverses the
  conclusion of an earlier, lowercasing-confounded run of this experiment;
  see `code/deidentification_experiment.py`'s module docstring for the fix.
  Neutralising pronouns but keeping names does the opposite: gap down to
  0.0236. Which channel dominates is **not the same on every re-ranker** —
  see Table A9 in the paper.
- **Full de-id → 100% ties by construction** (a check on the symmetry
  argument, not a finding). A part-of-speech-aware they/them/their mapping
  ties every pair too, exactly like the one-token rule, and reads as normal
  English. Only the *naive* her→their mapping fails, leaving 25% of pairs
  different — and where it fails, 96% of those residual pairs favor the
  *woman's* document, not the man's.
- **De-identification costs nothing measurable.** Top-1 accuracy across all
  four query phrasings: 99.4% before, 99.8% after the full transform.
- Occupation labels come from **BLS CPS Table 11** (% women), not hand
  judgement: 82 occupations scored, 52 kept (18 male / 21 female / 13
  neutral); the other 30 (no matching BLS occupation, a suppressed estimate,
  or an ambiguous 30–40%/60–70% share) are listed with their reason in the
  paper's appendix and in `../code/occupations_bls.csv`.
- One of the four locally-run re-rankers (jina-reranker-v2) did not exactly
  reproduce its own banked scores on a plain local re-run six months later
  (max score difference 0.012 on its own ~0.15–0.84 range) — flagged in the
  paper's Limits section and in Table A9's caption, not hidden.


## References
All 20 entries in `paper.md`'s bibliography were checked individually (title,
venue, DOI/URL) and are current as of August 2026 — including that EU AI Act
deployer duties (Arts. 26–27) are deferred to 2 December 2027 by Regulation
(EU) 2026/1744, which the paper's footnotes and bibliography already reflect.
