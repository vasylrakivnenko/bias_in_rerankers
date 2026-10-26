"""
A blind check by a person of the owner-marker labels (data/owner_marker_labels.json), which a
language model made.

Each round draws 20 labels, one per résumé and from résumés no earlier round drew: 10 names
(5 the model gave to the applicant, 5 to someone else) and 10 pronouns (5 applicant, 3
another person, 2 no person), with a fixed seed. `sheet` writes a page that shows each case
highlighted in 320 characters of context either side, with the résumé's category and
opening, and asks who the word refers to; the model's label is not shown. The page holds
résumé text, so it goes to a directory you name, outside the repository, and is never
released. The checker's answers are in data/owner_marker_handcheck.json (offsets only, no
text); `tally` checks that its cases are the ones the draw gives and counts the agreement.
companion_whole_resume.py reports the tally.

Reads:  data/owner_marker_labels.json, data/owner_marker_handcheck.json,
        data/resumes/resumes.csv (sheet only)
Run:    venv/bin/python code/owner_label_handcheck.py sheet ROUND --out-dir DIR
        venv/bin/python code/owner_label_handcheck.py tally
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

LABELS = ROOT / "data" / "owner_marker_labels.json"
HANDCHECK = ROOT / "data" / "owner_marker_handcheck.json"
SEED = 20261026
PER_ROUND = 20
STRATA = [("name", "applicant", 5), ("name", "other", 5), ("pronoun", "applicant", 5),
          ("pronoun", "other", 3), ("pronoun", "none", 2)]
CONTEXT = 320
ANSWERS = {"A": "applicant", "O": "other", "N": "none", "?": "unsure"}


def draw(labels: dict, rounds: int) -> list[dict]:
    """The cases of rounds 1..rounds, numbered 1..20*rounds."""
    items = [{"rid": rid, "kind": kind, "start": h["start"], "ref": h["ref"]}
             for rid, v in labels.items()
             for kind, key in (("name", "names"), ("pronoun", "pronouns")) for h in v[key]]
    out, used = [], set()
    for r in range(1, rounds + 1):
        rng = random.Random(SEED)
        pool = items[:]
        rng.shuffle(pool)
        picked = []
        for kind, ref, n in STRATA:
            for it in pool:
                if n and it["kind"] == kind and it["ref"] == ref and it["rid"] not in used:
                    picked.append(it)
                    used.add(it["rid"])
                    n -= 1
            assert n == 0, (r, kind, ref)
        rng.shuffle(picked)
        out += [{"n": PER_ROUND * (r - 1) + i, "round": r, **it} for i, it in enumerate(picked, 1)]
    return out


def sheet(rnd: int, out_dir: Path) -> Path:
    from realistic_pool_experiment import load_corpus
    corpus = {r.rid: r for r in load_corpus()}
    labels = json.loads(LABELS.read_text())["labels"]
    cards = []
    for it in draw(labels, rnd)[-PER_ROUND:]:
        r = corpus[it["rid"]]
        t, s = r.text, it["start"]
        word = (re.match(r"[A-Za-z][A-Za-z'\-]*", t[s:]).group(0) if it["kind"] == "name" else
                next(p["word"] for p in labels[it["rid"]]["pronouns"] if p["start"] == s))
        e = s + len(word)
        a, b = max(0, s - CONTEXT), min(len(t), e + CONTEXT)
        ctx = (("… " if a else "") + html.escape(t[a:s]) + "<mark>" + html.escape(t[s:e]) + "</mark>"
               + html.escape(t[e:b]) + (" …" if b < len(t) else ""))
        ctx = re.sub(r"\s+", " ", ctx)
        head = html.escape(re.sub(r"\s+", " ", t[:140]).strip())
        opts = "".join(f'<button data-v="{k}">{lbl}</button>' for k, lbl in (
            ("A", "The applicant"), ("O", "Someone else"), ("N", "No one / not a person"), ("?", "Can't tell")))
        cards.append(f"""<section class="card" data-n="{it["n"]}"><h2>{it["n"]}. {it["kind"].title()}: <span class="w">{html.escape(word)}</span></h2>
<p class="meta">Résumé category: {html.escape(r.category.title())} · begins: “{head}…”</p>
<p class="ctx">{ctx}</p>
<div class="opts">{opts}</div></section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label Check {rnd}</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--line:#e7e5e4;--mark:#fde68a;--on:#1d4ed8;--onfg:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--mark:#a16207;--on:#60a5fa;--onfg:#0c0a09}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;margin:0;padding:24px 16px 120px}}
main{{max-width:760px;margin:auto}} h1{{font-size:1.4rem;margin:0 0 4px}} .lead{{color:var(--muted);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}
.card h2{{font-size:1.05rem;margin:0 0 4px}} .w{{background:var(--mark);padding:0 4px;border-radius:3px}}
.meta{{color:var(--muted);font-size:.85rem;margin:0 0 8px}} .ctx{{margin:0 0 10px;overflow-wrap:anywhere}}
mark{{background:var(--mark);color:inherit;font-weight:600;padding:0 2px}}
.opts{{display:flex;flex-wrap:wrap;gap:8px}} button{{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}}
button.on{{background:var(--on);color:var(--onfg);border-color:var(--on)}}
footer{{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px}}
footer div{{max-width:760px;margin:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}} #out{{flex:1;min-width:200px;font:14px ui-monospace,monospace}}
</style></head><body><main><h1>Who does the highlighted word refer to? (round {rnd})</h1>
<p class="lead">{PER_ROUND} cases from the résumé corpus. For each, choose whether the highlighted name or pronoun refers to the résumé's owner (the applicant), to someone else (a supervisor, an author, a client…), or to no one. When done, press Copy and paste the line into data/owner_marker_handcheck.json's answers.</p>
{"".join(cards)}</main>
<footer><div><span id="count">0 / {PER_ROUND} answered</span><input id="out" readonly><button id="copy">Copy</button></div></footer>
<script>
const ans={{}};
function render(){{const s=Object.keys(ans).sort((a,b)=>a-b).map(n=>n+ans[n]).join(' ');document.getElementById('out').value=s;document.getElementById('count').textContent=Object.keys(ans).length+' / {PER_ROUND} answered';}}
document.querySelectorAll('.card').forEach(c=>c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{c.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');ans[c.dataset.n]=b.dataset.v;render();}}));
document.getElementById('copy').onclick=()=>{{const o=document.getElementById('out');o.select();try{{navigator.clipboard.writeText(o.value)}}catch(e){{document.execCommand('copy')}}}};
</script></body></html>"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"label_check_round{rnd}.html"
    out.write_text(page)
    return out


def tally(path: Path = HANDCHECK, labels_path: Path = LABELS) -> dict:
    """Agreement between the checker and the model, overall, by the model's label and by kind."""
    data = json.loads(path.read_text())
    labels = json.loads(labels_path.read_text())["labels"]
    cases = data["cases"]
    if len(cases) % PER_ROUND:
        raise SystemExit(f"{path.name}: {len(cases)} cases is not whole rounds of {PER_ROUND}")
    drawn = draw(labels, len(cases) // PER_ROUND)
    key = ("n", "round", "rid", "kind", "start")
    if ([[c[k] for k in key] + [c["model"]] for c in cases]
            != [[d[k] for k in key] + [d["ref"]] for d in drawn]):
        raise SystemExit(f"{path.name}: its cases are not the draw from {labels_path.name}")
    if not all(c["checker"] in ANSWERS.values() for c in cases):
        raise SystemExit(f"{path.name}: an answer is not one of {sorted(ANSWERS.values())}")

    def count(sub):
        dec = [c for c in sub if c["checker"] != "unsure"]
        return {"n": len(sub), "decided": len(dec), "agree": sum(c["checker"] == c["model"] for c in dec)}

    dis = [c for c in cases if c["checker"] not in ("unsure", c["model"])]
    return {"n_rounds": len(cases) // PER_ROUND, "seed": SEED, **count(cases),
            "unsure": sum(c["checker"] == "unsure" for c in cases),
            "names": count([c for c in cases if c["kind"] == "name"]),
            "pronouns": count([c for c in cases if c["kind"] == "pronoun"]),
            "model_applicant": count([c for c in cases if c["model"] == "applicant"]),
            "model_not_applicant": count([c for c in cases if c["model"] != "applicant"]),
            "disagree": len(dis),
            "disagree_checker_applicant": sum(c["checker"] == "applicant" for c in dis),
            "disagree_reread_model": sum(c.get("reread") == c["model"] for c in dis),
            "disagree_cases": [c["n"] for c in dis]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="write round ROUND's page (holds résumé text)")
    s.add_argument("round", type=int)
    s.add_argument("--out-dir", type=Path, required=True, help="outside the repository")
    sub.add_parser("tally", help="agreement between the checker and the model")
    a = ap.parse_args()
    if a.cmd == "sheet":
        print("wrote", sheet(a.round, a.out_dir))
    else:
        print(json.dumps(tally(), indent=1))


if __name__ == "__main__":
    main()
