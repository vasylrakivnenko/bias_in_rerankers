#!/usr/bin/env python3
"""
build.py --- reusable build pipeline for the empirical and legal papers.

WORKFLOW
  1. Edit the prose in empirical_paper.md (the default) or paper.md (the longer
     legal paper). Both are plain Markdown; you never touch LaTeX unless you
     want to.
  2. Run:   python3 build.py
     -> regenerates main.tex from empirical_paper.md, compiles main.pdf, and
        writes Overleaf and minimal arXiv source bundles.
     Run:   python3 build.py --paper legal
     -> builds paper.md and writes ../legal_paper_overleaf.zip instead.

OPTIONS
  --paper NAME    source manuscript: empirical (default) or legal.
  --figures       regenerate the charts from the experiment data first
                  (runs generate_figures.py with the project venv) before building.
  --no-zip        build the PDF but skip the zip.
  --draft         do not fail on unresolved {{PLACEHOLDER}} tokens; render them in
                  red in the PDF instead. Never use for a version you send out.
  --data-root D   look up numbers.json paths under D instead of the repo root
                  (used to test the pipeline against fixture data).

NUMBERS
  Every statistic in either manuscript is a {{PLACEHOLDER}} resolved from results/*.json
  through numbers.json. A placeholder that cannot be resolved FAILS the build and
  prints the full list. No number is ever typed into the prose by hand.

WHAT YOU EDIT vs WHAT IS AUTOMATIC
  - Prose, section headings, lists, emphasis, quotes, citations: edit in the
    selected Markdown source.
  - Tables / figures / bibliography: kept as ```latex raw blocks in that source.
    Leave them alone unless you know LaTeX; they pass through verbatim.
  - Statistics and generated tables: {{PLACEHOLDER}} + an entry in numbers.json.
  - The LaTeX preamble (packages, title formatting) lives in this file (PREAMBLE).

No third-party packages required: only python3 (standard library) and pdflatex.
"""
import argparse
import ast
import datetime as dt
import json
import re
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
NUMBERS_JSON = ROOT / "numbers.json"
MAIN_TEX = ROOT / "main.tex"
VENV_PY = ROOT.parent / "venv" / "bin" / "python"

PAPERS = {
    "empirical": {
        "source": ROOT / "empirical_paper.md",
        "slug": "gender-bias-neural-rerankers_empirical",
        "zip": REPO_ROOT / "empirical_paper_overleaf.zip",
        "arxiv_zip": REPO_ROOT / "empirical_paper_arxiv.zip",
    },
    "legal": {
        "source": ROOT / "paper.md",
        "slug": "gender-bias-automated-hiring_deidentification",
        "zip": REPO_ROOT / "legal_paper_overleaf.zip",
        "arxiv_zip": REPO_ROOT / "legal_paper_arxiv.zip",
    },
}
DEFAULT_PAPER = "empirical"

# Files placed into the Overleaf bundle (main.tex is added after it is generated).
BUNDLE = ["main.tex", "figures", "README.md", "generate_figures.py",
          "numbers.json", "build.py", "main.pdf"]

PREAMBLE = r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{float}
%% lmodern: scalable Type1 faces at every size. Without it microtype's font
%% expansion aborts with "auto expansion is only possible with scalable fonts"
%% as soon as the document uses a size that ships only as a bitmap (footnotes).
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}

\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}

%% Sections are numbered; the short subsection titles inside "What we found"
%% are not (the paper deliberately avoids 4.1/4.2-style cross-references).
\setcounter{secnumdepth}{1}

%% Marker used by `build.py --draft` for a statistic whose data has not landed.
\newcommand{\missingnum}[1]{\textcolor{red}{\textbf{[[#1]]}}}
\newcommand{\doi}[1]{doi:\href{https://doi.org/#1}{#1}}

\title{\bfseries %(title)s}
\author{%(author)s}
\date{%(date)s}

\begin{document}
\maketitle
"""

POSTAMBLE = "\n\\end{document}\n"


# --------------------------------------------------------------------------
# Front matter: --- key: value --- block at the top of the selected source
# --------------------------------------------------------------------------
def parse_front_matter(text):
    meta = {"title": "Untitled", "author": "", "date": "Draft"}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end]
            for line in block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4:]
    return meta, text.lstrip("\n")


# --------------------------------------------------------------------------
# {{PLACEHOLDER}} substitution (selected source + numbers.json -> results/*.json)
#
# Rule of the house: no statistic is ever typed into the prose. Each one is a
# {{PLACEHOLDER}} whose value is fetched from a results JSON at build time, so
# the paper cannot drift away from the data. An unresolved placeholder stops
# the build and prints every missing name with what it needs.
# --------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

_NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
_GLOSS_DENOMS = (2, 3, 4, 5, 10)          # denominators a reader says out loud
_FRACTION_WORDS = [
    (0.10, "about a tenth"), (0.125, "about an eighth"), (1 / 6, "about a sixth"),
    (0.20, "about a fifth"), (0.25, "about a quarter"), (1 / 3, "about a third"),
    (0.40, "about two-fifths"), (0.50, "about half"), (0.60, "about three-fifths"),
    (2 / 3, "about two-thirds"), (0.75, "about three-quarters"),
    (0.80, "about four-fifths"),
]


def gloss_x_in_n(pct):
    """A percentage in plain English: 81.1 -> 'about four in every five'."""
    p = float(pct)
    if p >= 97:
        return "almost every time"
    if p <= 3:
        return "almost never"
    if abs(p - 50) <= 4:
        return "about half the time"
    best = min(((abs(p / 100 - n / d) + 0.0015 * d, n, d)
                for d in _GLOSS_DENOMS for n in range(1, d)))
    _, n, d = best
    # Never let the plain-English gloss misstate the number by more than 3 points:
    # 56% is nearest to three-in-five, but "three in every five" reads as 60%.
    if abs(100 * n / d - p) > 3.0:
        if 50 < p < 60:
            return "just over half the time"
        if 40 < p < 50:
            return "just under half the time"
        return f"about {p:.0f}% of the time"
    return f"about {_NUM_WORD[n]} in every {_NUM_WORD[d]}"


def gloss_fraction_word(x):
    """A ratio in plain English: 0.23 -> 'about a quarter'."""
    x = float(x)
    if x > 1.5:                                    # tolerate a 0-100 percentage
        x /= 100.0
    return min(_FRACTION_WORDS, key=lambda t: abs(t[0] - x))[1]


GLOSSERS = {"x_in_n": gloss_x_in_n, "fraction_word": gloss_fraction_word}

_SAFE_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
               ast.Name, ast.Load)


def _safe_eval(expr, env):
    """Evaluate a small arithmetic expression over named reference values."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"disallowed element {type(node).__name__} in expr")
        if isinstance(node, ast.Name) and node.id not in env:
            raise ValueError(f"unknown name '{node.id}' in expr")
    return eval(compile(tree, "<numbers.json>", "eval"), {"__builtins__": {}}, env)


def _latex_escape(s):
    out = []
    for ch in str(s):
        if ch in "&%#$":
            out.append("\\" + ch)
        elif ch == "_":
            out.append(r"\_")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


class Unresolved(Exception):
    pass


class NumberStore:
    """Resolves numbers.json entries against the results tree."""

    def __init__(self, spec_path, data_root):
        self.data_root = Path(data_root)
        self.specs = {k: v for k, v in json.loads(spec_path.read_text()).items()
                      if not k.startswith("_")}
        self._json_cache = {}

    # -- low level -------------------------------------------------------
    def _load(self, rel):
        if rel not in self._json_cache:
            p = self.data_root / rel
            if not p.exists():
                raise Unresolved(f"file not found: {rel}")
            try:
                self._json_cache[rel] = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                raise Unresolved(f"{rel} is not valid JSON ({e})")
        return self._json_cache[rel]

    def _dig(self, rel, path):
        parts = path.split(".") if isinstance(path, str) else list(path)
        cur = self._load(rel)
        for i, part in enumerate(parts):
            here = ".".join(str(x) for x in parts[:i + 1])
            if isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except (ValueError, IndexError):
                    raise Unresolved(f"{rel}: no element at '{here}'")
            elif isinstance(cur, dict):
                if part not in cur:
                    raise Unresolved(f"{rel}: no key '{here}'")
                cur = cur[part]
            else:
                raise Unresolved(f"{rel}: '{here}' is not a container")
        return cur

    # -- formatting ------------------------------------------------------
    @staticmethod
    def _format(val, spec):
        if spec.get("len"):
            try:
                val = len(val)
            except TypeError:
                raise Unresolved("value has no length but 'len' was requested")
        if spec.get("scale") is not None:
            val = val * spec["scale"]
        if spec.get("gloss"):
            gl = GLOSSERS.get(spec["gloss"])
            if gl is None:
                raise Unresolved(f"unknown gloss mode '{spec['gloss']}'")
            return gl(val)
        fmt = spec.get("format")
        if fmt:
            if fmt.endswith("d"):
                val = int(round(float(val)))
            txt = format(val, fmt)
        else:
            txt = str(val)
        return txt + spec.get("unit", "")

    # -- public ----------------------------------------------------------
    def value(self, name):
        """LaTeX-ready string for one placeholder, or raise Unresolved."""
        spec = self.specs.get(name)
        if spec is None:
            raise Unresolved("no entry in numbers.json")

        if "tex_file" in spec:                       # verbatim LaTeX fragment
            p = self.data_root / spec["tex_file"]
            if not p.exists():
                raise Unresolved(f"table fragment not found: {spec['tex_file']}")
            return p.read_text().rstrip("\n")

        if "literal" in spec:
            if spec["literal"] is None:
                raise Unresolved("literal is null -- fill it in in numbers.json")
            return _latex_escape(spec["literal"])

        if "expr" in spec:
            env = {}
            for ref, r in (spec.get("refs") or {}).items():
                env[ref] = self._dig(r["file"], r["path"])
            try:
                val = _safe_eval(spec["expr"], env)
            except ZeroDivisionError:
                raise Unresolved("expression divided by zero")
            except ValueError as e:
                raise Unresolved(str(e))
            return _latex_escape(self._format(val, spec))

        if "file" in spec and "path" in spec:
            return _latex_escape(self._format(self._dig(spec["file"], spec["path"]), spec))

        raise Unresolved("entry has none of: file+path, expr, tex_file, literal")


def substitute_numbers(tex, store, draft=False):
    """Replace every {{PLACEHOLDER}} in `tex`. Returns (tex, problems, used)."""
    problems = {}
    used = set()

    def repl(m):
        name = m.group(1)
        used.add(name)
        try:
            return store.value(name)
        except Unresolved as e:
            problems[name] = str(e)
            return r"\missingnum{" + name.replace("_", r"\_") + "}" if draft else m.group(0)

    return PLACEHOLDER_RE.sub(repl, tex), problems, used


# --------------------------------------------------------------------------
# Inline markup:  **bold**  *italic*  "quotes"  `code`  $math$  [@cite; @key]
# Raw LaTeX (backslash commands, braces) passes through untouched.
# --------------------------------------------------------------------------
def _escape_text(s):
    # Escape the LaTeX specials that commonly appear in prose. We deliberately
    # leave \ { } alone so raw LaTeX macros (e.g. \bdi) keep working inline.
    for ch in ("&", "%", "#", "_"):
        s = s.replace(ch, "\\" + ch)
    return s


def convert_inline(s):
    placeholders = []

    def stash(latex):
        placeholders.append(latex)
        return f"\x00{len(placeholders) - 1}\x00"

    # protect {{PLACEHOLDER}} tokens so the underscores in their names survive
    # the LaTeX escaping below; they are substituted after conversion.
    s = PLACEHOLDER_RE.sub(lambda m: stash(m.group(0)), s)

    # protect $math$ (verbatim) and `code` (-> \texttt, escaped)
    s = re.sub(r"\$([^$]+)\$", lambda m: stash(f"${m.group(1)}$"), s)
    s = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{" + _escape_text(m.group(1)) + "}"), s)

    s = _escape_text(s)

    # citations: [@key] or [@k1; @k2]
    def cite(m):
        keys = [k.strip().lstrip("@").strip() for k in m.group(1).split(";")]
        return r"\cite{" + ",".join(keys) + "}"
    s = re.sub(r"\[@([^\]]+)\]", cite, s)

    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)          # bold
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)  # italic
    s = re.sub(r'"([^"]*)"', r"``\1''", s)                    # quotes

    for i, latex in enumerate(placeholders):                  # restore
        s = s.replace(f"\x00{i}\x00", latex)
    return s


# --------------------------------------------------------------------------
# Block-level conversion
# --------------------------------------------------------------------------
def convert_body(body):
    lines = body.split("\n")
    out = []
    i = 0
    n = len(lines)
    para = []
    list_items = []

    def flush_para():
        if para:
            out.append(convert_inline(" ".join(para).strip()))
            out.append("")
            para.clear()

    def flush_list():
        if list_items:
            out.append(r"\begin{enumerate}")
            for it in list_items:
                out.append(r"\item " + convert_inline(it))
            out.append(r"\end{enumerate}")
            out.append("")
            list_items.clear()

    in_abstract = False

    while i < n:
        line = lines[i]

        # raw LaTeX block:  ```  or  ```latex  ... ```
        if line.strip().startswith("```"):
            flush_para(); flush_list()
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                out.append(lines[i])
                i += 1
            i += 1
            out.append("")
            continue

        # headings
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            flush_para(); flush_list()
            if in_abstract:
                out.append(r"\end{abstract}")
                out.append("")
                in_abstract = False
            level, title = len(m.group(1)), m.group(2).strip()
            if title.lower() == "abstract":
                out.append(r"\begin{abstract}")
                out.append(r"\noindent")
                in_abstract = True
            elif level == 1:
                out.append(r"\section{" + convert_inline(title) + "}")
            elif level == 2:
                out.append(r"\subsection{" + convert_inline(title) + "}")
            else:  # level 3 -> run-in \paragraph{...} ending in a period
                inner = convert_inline(title)
                if inner.endswith("''"):           # period goes inside a closing quote
                    if not inner[:-2].rstrip().endswith((".", "?", "!")):
                        inner = inner[:-2].rstrip() + ".''"
                elif not inner.endswith((".", "?", "!")):
                    inner = inner + "."
                out.append(r"\paragraph{" + inner + "}")
            out.append("")
            i += 1
            continue

        # ordered list item
        lm = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if lm:
            flush_para()
            list_items.append(lm.group(1).strip())
            i += 1
            continue

        # blank line: paragraph / list boundary
        if line.strip() == "":
            flush_para(); flush_list()
            i += 1
            continue

        # continuation of a list item (indented wrap)
        if list_items and line.startswith(("    ", "\t")):
            list_items[-1] += " " + line.strip()
            i += 1
            continue

        flush_list()
        para.append(line.strip())
        i += 1

    flush_para(); flush_list()
    if in_abstract:
        out.append(r"\end{abstract}")
    return "\n".join(out)


# --------------------------------------------------------------------------
def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=ROOT, **kw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=sorted(PAPERS), default=DEFAULT_PAPER,
                        help="manuscript to build (default: empirical)")
    parser.add_argument("--figures", action="store_true",
                        help="regenerate figures before building")
    parser.add_argument("--no-zip", action="store_true",
                        help="skip the Overleaf bundle")
    parser.add_argument("--draft", action="store_true",
                        help="render unresolved placeholders in red")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT,
                        help="root used to resolve paths in numbers.json")
    args = parser.parse_args()
    selected = PAPERS[args.paper]
    paper_md = selected["source"]
    slug = selected["slug"]
    zip_path = selected["zip"]
    arxiv_zip_path = selected["arxiv_zip"]
    draft = args.draft
    data_root = args.data_root.resolve()

    if args.figures:
        if VENV_PY.exists():
            print("== regenerating figures ==")
            run([str(VENV_PY), "generate_figures.py"])
        else:
            print("!! venv python not found; skipping --figures", file=sys.stderr)

    print(f"== converting {paper_md.name} -> main.tex ==")
    raw = HTML_COMMENT_RE.sub("", paper_md.read_text())   # editorial notes
    meta, body = parse_front_matter(raw)
    tex = (PREAMBLE % {"title": meta["title"], "author": meta["author"],
                       "date": meta["date"]}
           + "\n" + convert_body(body) + POSTAMBLE)

    print(f"== resolving {{{{PLACEHOLDER}}}} numbers (data root: {data_root}) ==")
    store = NumberStore(NUMBERS_JSON, data_root)
    tex, problems, used = substitute_numbers(tex, store, draft=draft)
    print(f"   {len(used) - len(problems)}/{len(used)} placeholders resolved")
    unused = sorted(set(store.specs) - used)
    if unused:
        print(f"   !! defined in numbers.json but never used in {paper_md.name}: "
              + ", ".join(unused))
    if problems:
        head = ("!! %d UNRESOLVED PLACEHOLDER%s -- the paper will not be built."
                % (len(problems), "" if len(problems) == 1 else "S"))
        print("\n" + "=" * 78, file=sys.stderr)
        print(head if not draft else head.replace(
            "the paper will not be built.", "rendered in red (--draft)."),
            file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        for name in sorted(problems):
            print(f"  {{{{{name}}}}}\n      problem: {problems[name]}", file=sys.stderr)
            need = (store.specs.get(name) or {}).get("needs")
            if need:
                print(f"      needs:   {need}", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        if not draft:
            print("Fix the producing script or numbers.json, or re-run with "
                  "--draft to see the paper with the gaps marked.", file=sys.stderr)
            sys.exit(1)

    MAIN_TEX.write_text(tex)

    print("== compiling (pdflatex x2) ==")
    for _ in range(2):
        r = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-25:])
        print("!! pdflatex FAILED:\n" + tail, file=sys.stderr)
        sys.exit(1)

    # clean aux files
    for ext in (".aux", ".log", ".out"):
        (ROOT / ("main" + ext)).unlink(missing_ok=True)

    pages = "?"
    try:
        info = subprocess.run(["pdfinfo", "main.pdf"], cwd=ROOT,
                              capture_output=True, text=True).stdout
        for ln in info.splitlines():
            if ln.startswith("Pages:"):
                pages = ln.split(":")[1].strip()
    except FileNotFoundError:
        pass
    print(f"== main.pdf built ({pages} pages) ==")

    # A second, self-identifying copy.  `main.pdf` is what LaTeX and Overleaf
    # need, but it is a terrible name to have on disk or in someone's Downloads
    # folder: this repo once held two unrelated manuscripts both called
    # main.pdf, one of them superseded and wrong.  Send THIS file to anyone.
    # It is rewritten on every build, so it can never drift from main.pdf.
    stamped = ROOT / f"{slug}_{dt.date.today().isoformat()}.pdf"
    for old in ROOT.glob(f"{slug}_*.pdf"):
        if old != stamped:
            old.unlink()          # keep exactly one stamped copy, today's
    shutil.copy2(ROOT / "main.pdf", stamped)
    print(f"   -> {stamped.name}   (send this one, not main.pdf)")

    if not args.no_zip:
        print("== writing Overleaf bundle ==")
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for item in BUNDLE + [paper_md.name]:
                p = ROOT / item
                if p.is_dir():
                    for f in sorted(p.rglob("*")):
                        if f.is_file():
                            z.write(f, f.relative_to(ROOT))
                elif p.exists():
                    z.write(p, p.name)
        print(f"   -> {zip_path}")

        # arXiv asks TeX submitters to omit generated PDFs, auxiliary files,
        # build scripts, and other material not needed to compile the paper.
        print("== writing minimal arXiv source bundle ==")
        arxiv_zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(arxiv_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(MAIN_TEX, MAIN_TEX.name)
            figures = ROOT / "figures"
            for f in sorted(figures.glob("*.pdf")):
                z.write(f, f.relative_to(ROOT))
        print(f"   -> {arxiv_zip_path}")

    print("done.")


if __name__ == "__main__":
    main()
