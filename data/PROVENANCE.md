# Data provenance

Every third-party file used by this project, where it came from, what licence it
carries, when it was retrieved, and how to get it again. All retrievals below
were made on **2026-08-22** unless stated otherwise.

Downloads land in `data/raw/`, which is **not** tracked by git (see
`.gitignore`) — it is ~218 MB of re-downloadable third-party data. The tidy,
derived files that the analysis actually reads *are* tracked. Each derived file
is produced by a `data/build_*.py` script, so nothing in this directory is typed
by hand except the two hand-made mapping tables, which are themselves Python
source (`data/build_occupations_bls.py`, `data/build_category_to_soc.py`) and
therefore reviewable line by line.

---

## 1. BLS — Current Population Survey, Table 11

**Used for:** the occupation labels (REVIEW_TODO.md A8). This is the file that
makes the paper's phrase "jobs the labor market does not gender" true and
citable.

| | |
|---|---|
| Source | U.S. Bureau of Labor Statistics, *Labor Force Statistics from the Current Population Survey*, Table 11: "Employed people by detailed occupation, sex, race, and Hispanic or Latino ethnicity" |
| Canonical URL | <https://www.bls.gov/cps/cpsaat11.htm> |
| Reference period | **2025 annual averages** (the table's own column header) |
| Licence | U.S. Government work, public domain (17 U.S.C. § 105) |
| Retrieved | 2026-08-22 |
| Raw file | `data/raw/cpsaat11_bls_2025.html` (790 KB) |
| Tidy file | `data/bls/cpsaat11_2025.csv` — **595 rows**, 575 of them detailed occupations |

### How it was actually retrieved — read this before re-running

`www.bls.gov` returns **HTTP 403 to every scripted request**, for the whole
domain, not just the data files. This was confirmed against `/`, `/cps/`,
`/cps/cpsaat11.htm`, `/cps/cpsaat11.xlsx` and `/cps/aa2024/cpsaat11.htm`, with a
full Chrome header set, HTTP/1.1, and a cookie jar seeded from the site's own
pages. It is an Akamai edge block on the client fingerprint; no header
combination gets past it.

The HTML was therefore taken from the **Internet Archive's snapshot of the
official BLS page**, which is byte-for-byte the page BLS served:

```
curl -L --compressed -o data/raw/cpsaat11_bls_2025.html \
  "https://web.archive.org/web/20260802082242id_/https://www.bls.gov/cps/cpsaat11.htm"
venv/bin/python data/build_bls_table11.py
```

Snapshot timestamp `20260802082242` = 2026-08-02. Sanity check printed by the
parser: *Total, 16 years and over — 163,493 thousand employed, 47.1 % women*,
which matches the published CPS 2025 annual average.

**Manual refresh (do this to update to a newer annual average).** Open
<https://www.bls.gov/cps/cpsaat11.htm> in a normal browser and either
(a) *File → Save Page As → Web Page, HTML Only* to
`data/raw/cpsaat11_bls_<year>.html`, or (b) click the **XLSX** link on that page
to download `cpsaat11.xlsx`. Then point `data/build_bls_table11.py` at the new
file (it parses HTML; for XLSX you will need a small openpyxl reader instead)
and re-run `data/build_occupations_bls.py`. Older snapshots are listed at
`https://web.archive.org/cdx/search/cdx?url=bls.gov/cps/cpsaat11.htm&output=text&fl=timestamp,statuscode`.

### Suppressed estimates are never imputed

199 of the 595 rows carry `-` in the `% women` column: BLS suppresses estimates
whose base is below its publication threshold. These are written through as `-`
and the occupation is labelled `drop`. **No percentage in this repository is
estimated, interpolated, or guessed.**

### Occupation → BLS row mapping

Hand-made in `data/build_occupations_bls.py`; the percentages and the year are
read from the parsed table, never typed. Each of the 82 occupations in
`synthetic_dataset.OCCUPATIONS` is mapped by one of these routes:

| Route | Meaning | Example |
|---|---|---|
| (a) exact | a BLS row title *is* the occupation | `electrician` → "Electricians" |
| (b) residual | a BLS row is the catch-all for it | `engineer` → "Engineers, all other" |
| (c) largest kind | the largest-employment BLS row naming a *kind* of it | `teacher` → "Elementary and middle school teachers" |
| (d) synonym | a documented common synonym | `sheriff` → "Police officers" |
| (e) none | BLS has no occupation of that name → `drop`, `pct_women` blank | `warrior`, `boss` |

Route (e) covers three distinct situations, all flagged in the source file:

* the word names a **rank or function, not an occupation** — `boss`, `chief`,
  `supervisor`, `assistant`, `attendant`, `researcher`, `financier`, `protege`.
  BLS has no such occupation; the rows sharing the word are different jobs that
  span the whole range (e.g. "First-line supervisors of …" runs from 1.3 % women
  to 68.6 %), so no single row is the referent;
* the word is **not a civilian occupation** — `warrior`, `philosopher`,
  `maestro`, `socialite`, `homemaker`, `magician`, `fighter pilot` (CPS covers
  civilian employment only);
* BLS **suppressed** the estimate — `captain` and `skipper` ("Ship and boat
  captains and operators"), `midwife` ("Nurse midwives"), `broadcaster`
  ("Broadcast announcers and radio disc jockeys").

**Judgement calls worth a reviewer's attention.** For these the label happens to
be robust to which row is chosen, but the choice is a judgement:

| Occupation | Chosen row | Alternatives considered (% women) | Label robust? |
|---|---|---|---|
| `engineer` | Engineers, all other (17.9) | every detailed engineering row runs 7.0–25.1 | yes — all male |
| `mechanic` | Automotive service technicians and mechanics (1.8) | other mechanic rows 1.8–8.1 | yes — all male |
| `scientist` | Physical scientists, all other (42.6) | chemists 31.1, biological 47.8, medical 55.6, environmental 46.1 | mostly — one row (31.1) falls in the ambiguous band |
| `driver` | Driver/sales workers and truck drivers (7.7) | taxi 13.9, transit bus 36.7, school bus 51.8 | **no** — chosen row is 4× the next by employment |
| `clerk` | Office clerks, general (80.5) | shipping/receiving 33.7, billing 84.5, financial 41.9 | **no** — largest row chosen |
| `salesperson` | Retail salespersons (47.0) | parts salespersons 16.2 | **no** — largest row chosen |
| `laborer`, `mover` | Laborers and freight, stock, and material movers, hand (24.6) | construction laborers 4.7 | yes — both male |
| `consultant` | Management analysts (40.7) | — (the CPS category that covers management consultants) | n/a |
| `professor` | Postsecondary teachers (52.3) | — | n/a |

The three marked **no** (`driver`, `clerk`, `salesperson`) are the ones to drop
first if a reviewer objects; doing so removes 1 male, 1 female and 1 neutral
occupation and changes no headline conclusion.

### Label rule and result

`male` if % women < 30, `female` if > 70, `neutral` if 40 ≤ % women ≤ 60,
otherwise `drop` (the 30–40 and 60–70 bands are deliberately excluded as
ambiguous). Occupations with no BLS row → `drop`.

Output: `code/occupations_bls.csv`, 82 rows, schema documented in
`code/labels.py`. Of the 82 occupations, **52 survive**: 18 male, 21 female,
13 neutral; **30 are dropped**.

---

## 2. O*NET database — occupation titles and descriptions

**Used for:** SOC codes in `code/occupations_bls.csv`, and the job-description
queries in the realistic-pool study (REVIEW_TODO.md G1/G2).

| | |
|---|---|
| Source | O*NET 30.3 Database, May 2026 Release, National Center for O*NET Development for the U.S. Department of Labor, Employment and Training Administration |
| URL | <https://www.onetcenter.org/database.html> → `https://www.onetcenter.org/dl_files/database/db_30_3_text.zip` |
| Licence | **CC BY 4.0** — <https://www.onetcenter.org/license_db.html>. Attribution is required in the paper. |
| Retrieved | 2026-08-22 (direct download, no blocking, 13,222,549 bytes) |
| Raw file | `data/raw/onet_db_text.zip`, extracted to `data/raw/onet_extract/db_30_3_text/` (40 files) |
| Tracked files | `data/onet/Occupation Data.txt` — **1,016 occupations** (SOC code, title, description)<br>`data/onet/Task Statements.txt` — **18,796 task statements**<br>`data/onet/README_onet.txt` — the bundle's own version/licence notice |

Suggested citation for the paper: *National Center for O*NET Development.
O\*NET 30.3 Database. O\*NET Resource Center. Retrieved 22 August 2026 from
<https://www.onetcenter.org/database.html>. Used under CC BY 4.0.*

Re-download:
```
curl -L -o data/raw/onet_db_text.zip https://www.onetcenter.org/dl_files/database/db_30_3_text.zip
unzip -q data/raw/onet_db_text.zip -d data/raw/onet_extract
cp "data/raw/onet_extract/db_30_3_text/Occupation Data.txt"  data/onet/
cp "data/raw/onet_extract/db_30_3_text/Task Statements.txt"  data/onet/
```
Check <https://www.onetcenter.org/database.html> for the current version number
before assuming `db_30_3` is still the latest.

---

## 3. SSA baby-name national data — name → gender table

**Used for:** inferring a résumé-holder's gender from the name it carries
(G4 leakage probe), and choosing a matched substitute name for the gender-swapped
twin (G2). Also supports the honest description of the paper's own name list
(REVIEW_TODO.md A9).

| | |
|---|---|
| Source | U.S. Social Security Administration, *Beyond the Top 1000 Names* — national data |
| URL | <https://www.ssa.gov/oact/babynames/limits.html> → `https://www.ssa.gov/oact/babynames/names.zip` |
| Licence | U.S. Government work, public domain |
| Retrieved | 2026-08-22 (7,860,026 bytes) |
| Raw file | `data/raw/ssa_names.zip` → `data/raw/ssa_names/` (**146** `yob<year>.txt` files, 1880–2025) |
| Tidy file | `data/name_gender.csv` — **23,618 names** (15,434 female-typed, 8,184 male-typed) |

`www.ssa.gov` sits behind the same Akamai edge as BLS and returns HTTP 403 to a
plain `curl`. Unlike BLS it *can* be reached with a complete browser header set;
it is also rate-limited, so a request may 403 even with correct headers — wait
and retry. The command that worked:

```
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
curl -L --http1.1 -A "$UA" \
  -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8" \
  -H "Accept-Language: en-US,en;q=0.9" -H "Accept-Encoding: gzip, deflate, br" \
  -H "Sec-Fetch-Dest: document" -H "Sec-Fetch-Mode: navigate" \
  -H "Sec-Fetch-Site: same-origin" -H "Sec-Fetch-User: ?1" \
  -H "Upgrade-Insecure-Requests: 1" \
  -o data/raw/ssa_names.zip "https://www.ssa.gov/oact/babynames/names.zip"
unzip -q data/raw/ssa_names.zip -d data/raw/ssa_names
venv/bin/python data/build_name_gender.py
```

**Manual fallback:** open <https://www.ssa.gov/oact/babynames/limits.html> in a
browser and click "National data"; save the zip to `data/raw/ssa_names.zip`.

### How `data/name_gender.csv` is built

Restricted to **birth years 1940–2005** — essentially the whole current
working-age population, so a name's gender association is the one it had for the
cohort now writing résumés rather than the one it has for newborns. Kept only
names with **≥ 95 % one-gender usage** and **≥ 100 births** in that window.

Columns: `name`, `pct_female`, `n`, `peak_decade`. `peak_decade` (the decade of
the name's peak births) is there so a substituted name can be matched on era as
well as on frequency, as G2 requires.

All 20 names in the paper's own name list are present and unambiguous (male
names 0.34–0.51 % female, female names 99.24–99.77 % female), which supports
rewriting the "validated lists" claim in A9 as "the most frequent U.S. given
names per gender, SSA".

---

## 4. Résumé corpus (livecareer)

**Used for:** the realistic-pool study (G2–G6). Same corpus as Wilson &
Caliskan, which makes the results directly comparable.

| | |
|---|---|
| Source | `opensporks/resumes` on HuggingFace, a conversion of Kaggle `snehaanbhawal/resume-dataset`; résumé *examples* scraped from livecareer.com |
| URL | <https://huggingface.co/datasets/opensporks/resumes> → `https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv` |
| Licence | **CC0 1.0** (declared on the dataset card) |
| Retrieved | 2026-08-22 (56,273,235 bytes, no account or token needed) |
| Raw file | `data/raw/livecareer_Resume_full.csv` |
| Tidy file | `data/resumes/resumes.csv` — **2,484 résumés**, 24 categories, columns `id, category, resume_text` |

```
curl -L -o data/raw/livecareer_Resume_full.csv \
  https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv
venv/bin/python data/build_resumes_csv.py
```

The tidy file drops the unused `Resume_html` column (~70 % of the bytes) and
collapses whitespace. The dataset also ships per-résumé PDFs under `data/data/`
in the repo; they are not used here.

**Ethics / validity, for the Limitations section.** These are published résumé
*examples*, not real applicants' documents — an ethics advantage (no personal
data about identifiable job seekers) and a validity caveat (idealised, template-y
text). Both should be stated.

Category counts: ACCOUNTANT 118, ADVOCATE 118, AGRICULTURE 63, APPAREL 97,
ARTS 103, AUTOMOBILE 36, AVIATION 117, BANKING 115, BPO 22,
BUSINESS-DEVELOPMENT 120, CHEF 118, CONSTRUCTION 112, CONSULTANT 115,
DESIGNER 107, DIGITAL-MEDIA 96, ENGINEERING 118, FINANCE 118, FITNESS 117,
HEALTHCARE 115, HR 110, INFORMATION-TECHNOLOGY 120, PUBLIC-RELATIONS 111,
SALES 116, TEACHER 102.

---

## 5. `data/category_to_soc.csv` — résumé category → SOC

Hand-made in `data/build_category_to_soc.py` (24 rows). The SOC code and the
confidence judgement are typed there; the O*NET title and description and the
BLS % women are looked up, not transcribed.

`match_confidence` records how well the category name corresponds to an
occupation: **high** (5 categories) — the category names an occupation and the
résumés are that job; **medium** (15) — names an occupation, résumés vary around
it; **low** (4: APPAREL, ARTS, AVIATION, BANKING) — the category names an
*industry*, and the SOC is only the modal job among those résumés.

The corpus labels are noisy in an ordinary way: a résumé filed under CHEF may be
a nursing student's, one under AUTOMOBILE an ETL developer's. That is a property
of the corpus, not of the mapping. **Category should be treated as a weak
relevance label, not ground truth**, and the realistic-pool study should say so.

`data/names_gender.csv` is a symlink to `data/name_gender.csv`; the
realistic-pool script looks for the plural spelling.

---

## 6. Nothing else was downloaded

No other external data source is used. No dataset was reconstructed, inferred or
synthesised. Where a source could not be fetched (BLS direct), the route actually
used is named above and the manual step is written out; where a value is
unavailable (BLS-suppressed estimates), the cell is left blank and the row is
dropped rather than filled in.
