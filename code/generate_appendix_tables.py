"""
Generate two LaTeX appendix table fragments, read straight from the live data
sources so they cannot drift out of sync with the code/data they describe.

See REVIEW_ROUND2.md item A4, bullets 3 and 5.

Reads:
    code/synthetic_dataset.py   (TEMPLATES, QUERY_TYPES, NAME_PAIRS)
    data/name_gender.csv        (name, pct_female, n, peak_decade)
    code/occupations_bls.csv    (occupation, bls_title, soc_code, pct_women, label, year)

Writes (bare LaTeX fragments -- no \\begin{table} float wrapper, no caption,
no label; a wrapping appendix section supplies those):
    results/tex/documents_appendix.tex   -- the four templates, the four query
                                             forms, the ten name pairs + SSA shares
    results/tex/occupations_table.tex    -- Table A (52 kept occupations) and
                                             Table B (30 dropped occupations,
                                             with a mechanically derived reason)

No ML dependencies -- pure data formatting. Safe to re-run any time either
source file changes; never edits synthetic_dataset.py, occupations_bls.csv,
or anything under legal_paper/.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = ROOT / "results"
TEX_DIR = RESULTS_DIR / "tex"
NAME_GENDER_CSV = ROOT / "data" / "name_gender.csv"
OCC_BLS_CSV = CODE_DIR / "occupations_bls.csv"

import synthetic_dataset as sd  # noqa: E402  (needs CODE_DIR on sys.path)

sys.path.insert(0, str(ROOT / "data"))
from build_name_gender import YEAR_MAX, YEAR_MIN  # noqa: E402  (birth window behind name_gender.csv)


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc(s: str) -> str:
    """Escape LaTeX special characters in plain text (char-by-char, so the
    backslashes this inserts are never themselves re-escaped)."""
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in s)


def tabular(colspec: str, header: list[str], rows: list[list[str]],
            note: str | None = None) -> str:
    """Render a bare booktabs tabular block. Cells are assumed pre-escaped.

    `note` is an italic line set flush left ABOVE the top rule and INSIDE the
    tabular box, which is the only place a page break cannot get between it and
    its table. (As its own paragraph it is stranded at the foot of the previous
    page, and `\\\\*` does not save it: the 4pt of glue the optional argument
    adds sits before the no-break penalty and is a legal breakpoint itself.)
    """
    lines = [f"\\begin{{tabular}}{{{colspec}}}"]
    if note is not None:
        lines.append(f"\\multicolumn{{{len(header)}}}{{@{{}}l}}{{\\emph{{{note}}}}} \\\\[4pt]")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page-safe tables
# ---------------------------------------------------------------------------
#
# BUG FOUND IN REVIEW: a `tabular` is ONE unbreakable box. When it is taller
# than the text block TeX does not paginate it -- it lets it run off the bottom
# of the page without an error a reader would ever see. The overflowing rows
# stay in the PDF's content stream (grep and text extraction still find them,
# which is exactly why this survived several proof-reads) but they are painted
# below the paper edge and are invisible.
#
# Table A did that: 52 kept occupations, six of them wrapping onto a second
# line, is ~59 text lines against the ~47 that fit in a 9in text block. In the
# built paper the last seven occupations (receptionist ... veterinarian) were
# printed off-page, the page folio was overprinted by the `nanny` row, and the
# row straddling the page edge was sliced through the middle. That slice is
# what a reviewer reported as receptionist printing "89 6%": the cell holds a
# perfectly well-formed `89.6\%`, but the decimal point sits on the baseline,
# in the half of the row the page edge cut away, so only the digits survived.
# Nothing was wrong with the percent formatting, the escaping, or the column
# spec -- the table was simply taller than the page.
#
# Fix: emit the rows in page-sized chunks, each a complete booktabs tabular
# with the header repeated. Two tabulars separated by a blank line are two
# paragraphs, so TeX is free to break the page between them, and each chunk is
# short enough to fit whatever room is left. `longtable` would do this
# natively, but it needs \usepackage{longtable}: this script does not own the
# paper's preamble, and a generated fragment must never require a package the
# document does not load.

# One 9in text block holds about 47 lines at 11pt on 13.6pt leading, before the
# booktabs rules and any surrounding prose. 40 is deliberately short of that:
# over-estimating what a chunk needs costs a little whitespace at a page break,
# while under-estimating brings back rows nobody can see.
LINES_PER_CHUNK = 40

# Characters that fit on one line of a p{} column, per inch of column width, in
# 11pt Computer Modern. Calibrated against the built PDF -- "Automotive service
# technicians and mechanics" (44 characters) wraps inside the 2.9in column --
# and rounded DOWN, so a row is more often judged taller than it really is.
CHARS_PER_INCH = 14

_COLSPEC_RE = re.compile(r"p\{([0-9.]+)in\}|([lrc])")


def _column_char_budgets(colspec: str) -> list[int | None]:
    """Per column: how many characters fit on one line, or None where the
    column cannot wrap at all (`l`/`r`/`c` grow to fit their widest cell)."""
    budgets: list[int | None] = []
    for m in _COLSPEC_RE.finditer(colspec):
        budgets.append(int(float(m.group(1)) * CHARS_PER_INCH) if m.group(1) else None)
    return budgets


def _row_lines(row: list[str], budgets: list[int | None]) -> int:
    """Height of one row in text lines: the tallest cell wins. Cell strings are
    already LaTeX-escaped, so a backslash or two is counted that will not be
    printed -- another over-estimate, and over-estimating is the safe way to be
    wrong here."""
    height = 1
    for cell, budget in zip(row, budgets):
        if budget:
            height = max(height, -(-len(cell) // budget))   # ceil division
    return height


def tabular_paged(colspec: str, header: list[str], rows: list[list[str]],
                  continued_note: str, lines_per_chunk: int = LINES_PER_CHUNK) -> str:
    """Render `rows` as one or more booktabs tabulars, none taller than a page.

    Every chunk repeats the header, and every chunk after the first carries
    `continued_note` inside its own box, so a reader meeting a repeated header
    mid-appendix does not mistake it for a new table. A table that already fits
    comes back as a single tabular, byte-identical to `tabular()`, and no note.
    """
    budgets = _column_char_budgets(colspec)
    if len(budgets) != len(header):
        raise ValueError(
            f"colspec {colspec!r} parses to {len(budgets)} columns but the header "
            f"has {len(header)}. The height estimate below lines cells up with "
            "columns by position, so a mis-parse would silently mis-measure the "
            "table -- fix the column spec (or _COLSPEC_RE) first."
        )

    # +1 for the "(continued)" note row every chunk but the first carries. The
    # first chunk is charged for a line it does not use; one line of slack costs
    # nothing and keeps the budget arithmetic in one place.
    head_lines = _row_lines(header, budgets) + 1
    heights = [_row_lines(r, budgets) for r in rows]
    tallest = max(heights, default=1)
    if head_lines + tallest > lines_per_chunk:
        raise ValueError(
            f"a single row needs {tallest} lines and the header {head_lines}, which "
            f"will not fit in {lines_per_chunk}. No split can save this table; "
            "narrow a column or shorten the cell."
        )

    def pack(budget: int) -> list[list[list[str]]]:
        chunks: list[list[list[str]]] = []
        current: list[list[str]] = []
        used = head_lines
        for row, height in zip(rows, heights):
            if current and used + height > budget:
                chunks.append(current)
                current, used = [], head_lines
            current.append(row)
            used += height
        if current or not chunks:      # an empty table is still one (empty) chunk
            chunks.append(current)
        return chunks

    # First find how many pages the table needs, then BALANCE across exactly
    # that many. Greedy packing at the full page budget fills each page to the
    # brim and leaves whatever is left over alone on the last one (Table B came
    # out 27 rows + 3), which reads as a mistake. Re-packing at the smallest
    # budget that still needs the same number of chunks splits the rows evenly.
    chunks = pack(lines_per_chunk)
    if len(chunks) > 1:
        even = head_lines + -(-sum(heights) // len(chunks))
        for budget in range(even, lines_per_chunk):
            trial = pack(budget)
            if len(trial) == len(chunks):
                chunks = trial
                break

    parts = [tabular(colspec, header, chunks[0])]
    for chunk in chunks[1:]:
        parts.append("")                                  # blank line = \par
        parts.append(tabular(colspec, header, chunk, note=continued_note))
    print(f"  {len(rows)} rows ({sum(heights)} est. lines) -> {len(chunks)} "
          "page-sized chunk(s) of "
          + ", ".join(str(len(c)) for c in chunks) + " rows")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deliverable 1: results/tex/documents_appendix.tex
# ---------------------------------------------------------------------------

# {slot} -> bracketed plain-English placeholder, applied via the templates'
# own str.format() so the rendering is mechanically tied to TEMPLATES itself
# (a template using an unknown slot would raise KeyError here, not drift
# silently). Pronoun slots are spelled from sd.PRONOUNS, so the appendix shows
# the scored capitalization: the subject pronoun is "He"/"She" in every
# document.
def _pronoun_slot(form: str) -> str:
    return f"[{sd.PRONOUNS['male'][form]}/{sd.PRONOUNS['female'][form]}]"


SLOT_PLACEHOLDERS = {
    "name": "[name]",
    "occupation": "[occupation]",
    "occupation_title": "[Occupation]",
    "pronoun_subject": _pronoun_slot("subject"),
    "pronoun_object": _pronoun_slot("object"),
    "pronoun_possessive": _pronoun_slot("possessive"),
}

TEMPLATE_LABELS = {
    "professional_bio": "Short bio",
    "resume_summary": "Resume summary",
    "news_article": "News mention",
    "linkedin_style": "LinkedIn-style profile",
}

QUERY_LABELS = {
    "bare": "Bare",
    "experienced": "With ``experienced''",
    "qualified": "With ``qualified ... professional''",
    "expertise": "With ``expertise''",
}

EXAMPLE_OCCUPATION = "engineer"


def render_template(template: str) -> str:
    """Fill a raw TEMPLATES value with bracketed placeholders instead of real
    values -- this literally calls the same .format() the real pipeline uses,
    just with placeholder strings, so it can never silently diverge from the
    real template text."""
    return template.format(**SLOT_PLACEHOLDERS)


def build_documents_appendix() -> str:
    parts: list[str] = []
    parts.append("% Generated by code/generate_appendix_tables.py -- do not edit by hand.")
    parts.append("% Source: code/synthetic_dataset.py (TEMPLATES, QUERY_TYPES, NAME_PAIRS)")
    parts.append("%         and data/name_gender.csv (SSA female-share lookup).")
    parts.append("")

    # --- 1. The four templates -------------------------------------------------
    parts.append("\\textbf{Document templates}\\\\[4pt]")
    parts.append("\\begin{description}")
    for key, template in sd.TEMPLATES.items():
        if key not in TEMPLATE_LABELS:
            raise KeyError(
                f"synthetic_dataset.TEMPLATES has a key {key!r} with no readable "
                "label in TEMPLATE_LABELS -- add one so this appendix cannot drift silently."
            )
        label = TEMPLATE_LABELS[key]
        rendered = esc(render_template(template))
        parts.append(f"\\item[{label}] ``{rendered}''")
    parts.append("\\end{description}")
    parts.append("")

    # --- 2. The four query forms -------------------------------------------------
    parts.append("\\bigskip")
    # Each heading shares a minipage with its table so a page break cannot
    # strand the heading at the foot of a page.
    parts.append("\\noindent\\begin{minipage}{\\linewidth}")
    parts.append("\\textbf{Query phrasings}\\\\[4pt]")
    qf_header = ["Label", "Example"]
    qf_rows = []
    for key, pattern in sd.QUERY_TYPES.items():
        if key not in QUERY_LABELS:
            raise KeyError(
                f"synthetic_dataset.QUERY_TYPES has a key {key!r} with no readable "
                "label in QUERY_LABELS -- add one so this appendix cannot drift silently."
            )
        label = QUERY_LABELS[key]
        example = esc(pattern.format(occupation=EXAMPLE_OCCUPATION))
        qf_rows.append([label, example])
    parts.append(tabular("ll", qf_header, qf_rows))
    parts.append("\\end{minipage}")
    parts.append("")

    # --- 3. The ten name pairs ---------------------------------------------------
    parts.append("\\bigskip")
    parts.append("\\noindent\\begin{minipage}{\\linewidth}")
    parts.append("\\textbf{Name pairs}\\\\[4pt]")
    name_pct = load_name_gender(NAME_GENDER_CSV)
    missing: list[str] = []
    np_header = ["Male name", "Female name", "\\% female (male name)", "\\% female (female name)"]
    np_rows = []
    for male_name, female_name in sd.NAME_PAIRS:
        male_pct = lookup_pct_female(name_pct, male_name)
        female_pct = lookup_pct_female(name_pct, female_name)
        if male_pct == "n/a":
            missing.append(male_name)
        if female_pct == "n/a":
            missing.append(female_name)
        np_rows.append([esc(male_name), esc(female_name), male_pct, female_pct])
    parts.append(tabular("llrr", np_header, np_rows))
    parts.append("")
    # FACTCHECK_ROUND3 S20: the shares come from a birth-year window, not the
    # whole SSA archive the bibliography points to, so say which window.
    parts.append(f"\\par\\smallskip{{\\footnotesize Female shares are computed from SSA "
                 f"national birth-name records \\cite{{ssa2026}}, restricted to births "
                 f"{YEAR_MIN}--{YEAR_MAX} (approximately the current working-age cohort).}}")
    parts.append("\\end{minipage}")
    parts.append("")

    if missing:
        print(
            f"WARNING: {len(missing)} name(s) in synthetic_dataset.NAME_PAIRS did not "
            f"resolve to an SSA percentage and were rendered as 'n/a': {missing}",
            file=sys.stderr,
        )

    return "\n".join(parts) + "\n"


def load_name_gender(path: Path) -> dict[str, float]:
    """name.lower() -> pct_female, from data/name_gender.csv."""
    table: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            pct = (row.get("pct_female") or "").strip()
            if not name or not pct:
                continue
            key = name.lower()
            table.setdefault(key, float(pct))
    return table


def lookup_pct_female(table: dict[str, float], name: str) -> str:
    val = table.get(name.strip().lower())
    if val is None:
        return "n/a"
    return f"{val:.2f}\\%"


# ---------------------------------------------------------------------------
# Deliverable 2: results/tex/occupations_table.tex
# ---------------------------------------------------------------------------

LABEL_ORDER = {"male": 0, "neutral": 1, "female": 2}


def fmt_pct_1dp(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "---"
    return f"{float(raw):.1f}\\%"


def fmt_field(raw: str) -> str:
    raw = (raw or "").strip()
    return esc(raw) if raw else "---"


def drop_reason(row: dict) -> str:
    """Derived mechanically from the CSV columns themselves (never from
    data/PROVENANCE.md prose), per REVIEW_ROUND2.md A4 bullet 5.

    The final branch ASSERTS the ambiguous-band membership rather than assuming
    it. Falling through to "ambiguous share" for a row dropped for some other
    reason would publish a fabricated justification, which is worse than
    crashing: the reader cannot tell the difference.
    """
    bls_title = (row.get("bls_title") or "").strip()
    pct_women = (row.get("pct_women") or "").strip()
    if not bls_title:
        return "no matching BLS occupation"
    if not pct_women:
        return "BLS estimate suppressed"
    v = float(pct_women)
    if not (30 <= v < 40 or 60 < v <= 70):
        raise ValueError(
            f"{row['occupation']!r} is labelled drop with pct_women={v}, which is "
            "outside the 30-40 / 60-70 ambiguous bands and outside the male/female/"
            "neutral cutoffs. This script cannot state why it was dropped without "
            "guessing -- add an explicit reason column or fix the label rule."
        )
    return "ambiguous share (30--40\\% or 60--70\\% women)"


def load_occupations(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_occupations_table() -> tuple[str, int, int]:
    rows = load_occupations(OCC_BLS_CSV)

    kept = [r for r in rows if r.get("label") in ("male", "female", "neutral")]
    dropped = [r for r in rows if r.get("label") == "drop"]

    unaccounted = len(rows) - len(kept) - len(dropped)
    if unaccounted:
        raise ValueError(
            f"{unaccounted} row(s) in {OCC_BLS_CSV.name} have a label outside "
            "{male, female, neutral, drop} -- fix the CSV or this script's "
            "assumptions before regenerating the appendix."
        )

    parts: list[str] = []
    parts.append("% Generated by code/generate_appendix_tables.py -- do not edit by hand.")
    parts.append("% Source: code/occupations_bls.csv.")
    parts.append("")

    # --- Table A: kept occupations ------------------------------------------
    kept_sorted = sorted(kept, key=lambda r: (LABEL_ORDER[r["label"]], r["occupation"].lower()))
    a_header = ["Occupation", "BLS occupation title", "\\% women (BLS)", "Label"]
    a_rows = []
    for r in kept_sorted:
        pct_women = (r.get("pct_women") or "").strip()
        if not pct_women:
            raise ValueError(
                f"kept occupation {r['occupation']!r} has label={r['label']!r} but no "
                "pct_women -- labels.py's rule requires a numeric share for any "
                "non-drop label; check occupations_bls.csv."
            )
        a_rows.append([
            fmt_field(r["occupation"]),
            fmt_field(r["bls_title"]),
            fmt_pct_1dp(pct_women),
            esc(r["label"]),
        ])
    # REVIEW_ROUND2 A11 bug found in review: natural-width `l` columns for
    # Occupation and BLS occupation title ran past the 6.5in text block on
    # several rows (long BLS titles like "Secretaries and administrative
    # assistants, except legal, medical, and executive"), spilling text off
    # the page edge. Fixed-width wrapping p{} columns for the two long fields;
    # % women and Label are always short, so they stay natural.
    #
    # ...and that fixed the width while leaving the height: 52 rows is taller
    # than a page, so the last seven were printed off the bottom of it. See
    # `tabular_paged` for the whole story; it splits the rows across pages.
    parts.append(tabular_paged("p{1.3in}p{2.9in}rl", a_header, a_rows,
                               "Occupations kept for analysis (continued)"))
    parts.append("")

    # --- Table B: dropped occupations ---------------------------------------
    parts.append("\\bigskip")
    parts.append("\\noindent\\textbf{Dropped occupations}\\\\[4pt]")
    dropped_sorted = sorted(dropped, key=lambda r: r["occupation"].lower())
    b_header = ["Occupation", "BLS occupation title", "\\% women (BLS)", "Reason"]
    b_rows = []
    for r in dropped_sorted:
        b_rows.append([
            fmt_field(r["occupation"]),
            fmt_field(r["bls_title"]),
            fmt_pct_1dp(r.get("pct_women", "")),
            drop_reason(r),
        ])
    # Same width fix as Table A; Reason is also long free text. Table B fits on
    # one page today (30 rows, half of them two lines tall) but sits within a
    # few rows of the same cliff, so it goes through the paged renderer too --
    # which leaves it as a single tabular until it actually needs splitting.
    parts.append(tabular_paged("p{1.1in}p{1.8in}rp{1.9in}", b_header, b_rows,
                               "Dropped occupations (continued)"))
    parts.append("")

    return "\n".join(parts) + "\n", len(kept), len(dropped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    # --- Deliverable 1 ---
    doc_tex = build_documents_appendix()
    doc_path = TEX_DIR / "documents_appendix.tex"
    doc_path.write_text(doc_tex)
    print(f"Wrote {doc_path} ({len(doc_tex.splitlines())} lines)")

    # Spot-check print: professional_bio, original vs rendered.
    original = sd.TEMPLATES["professional_bio"]
    rendered = render_template(original)
    print("\n--- spot check: professional_bio ---")
    print(f"original: {original!r}")
    print(f"rendered: {rendered!r}")

    # --- Deliverable 2 ---
    # Build and CHECK before writing. These counts are a tripwire for an
    # unintended edit to occupations_bls.csv, but if they fire after the writes
    # the tree is left half-updated and looks like a clean regeneration, which is
    # the worst of both. A legitimate change to the CSV should update the
    # expected counts here in the same commit.
    occ_tex, n_kept, n_dropped = build_occupations_table()

    print(f"\nTable A (kept) rows: {n_kept}")
    print(f"Table B (dropped) rows: {n_dropped}")
    print(f"Sum: {n_kept + n_dropped}")
    assert n_kept == 52, f"expected 52 kept occupations, got {n_kept}"
    assert n_dropped == 30, f"expected 30 dropped occupations, got {n_dropped}"
    assert n_kept + n_dropped == 82, f"expected 82 total, got {n_kept + n_dropped}"

    occ_path = TEX_DIR / "occupations_table.tex"
    occ_path.write_text(occ_tex)
    print(f"\nWrote {occ_path} ({len(occ_tex.splitlines())} lines)")
    print("\nRow-count assertions passed (52 / 30 / 82).")


if __name__ == "__main__":
    main()
