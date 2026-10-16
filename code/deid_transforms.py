"""Canonical gender de-identification transforms (single source of truth).

Every experiment that de-identifies text must import from here, so that the
synthetic study (code/deidentification_experiment.py) and the realistic-pool
study (code/realistic_pool_experiment.py, code/leakage_probe.py) are provably
applying the *same* rules -- which is the paper's whole auditability claim.

Variants (REVIEW_TODO.md A4, A5, B8):

    none            passthrough (baseline)
    names_only      names -> placeholder; pronouns left gendered
    pronouns_only   pronouns -> one uniform token; names left in place
    full            names + pronouns (the paper's recommended policy)
    gram_her_their  grammatical they/them/their, "her" -> "their" (possessive)
    gram_her_them   grammatical they/them/their, "her" -> "them" (object)
    gram_pos        following-word heuristic for "her" (not a POS tagger)

`gram_*` are pronoun-rewrite ablations, not complete grammatical rewriters.
English "her" is both possessive (counterpart of "his") and object ("him").
The following-word heuristic equalises these templates, but it can misclassify
ordinary prose and none of the variants repairs subject-verb agreement.

`lowercase` is a SEPARATE flag, not baked into a variant, so B8 can ablate it:
it contributes nothing to the symmetry guarantee (both twins get identical
substitutions either way) and may cost retrieval signal.

Name handling: `names` may be
  - an iterable of literal names (the synthetic study passes the dataset's own
    name list), or
  - a callable `detect(text) -> list[str]` (the realistic-pool study passes an
    NER-backed detector, since a 20-name regex covers ~nothing of a real corpus;
    see REVIEW_TODO.md G6).
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

VARIANTS = (
    "none",
    "names_only",
    "pronouns_only",
    "full",
    "gram_her_their",
    "gram_her_them",
    "gram_pos",
)

DEFAULT_PLACEHOLDER = "the candidate"

# One display name per experimental condition, shared by Figure 3 and the two
# de-identification tables so a reader can line the three up by name
# (FACTCHECK_ROUND3 S6). `$\to$` renders in both LaTeX and matplotlib mathtext.
CONDITION_LABELS = {
    "original": "Original (no transform)",
    "lowercase_only": "Lowercase only (control)",
    "names_only": "Names only",
    "pronouns_only": "Pronouns only",
    "gram_her_their": "Naive rewrite: her $\\to$ their",
    "gram_her_them": "Naive rewrite: her $\\to$ them",
    "gram_pos": "Following-word rule for her",
    "full": "Full rule (names + pronouns)",
}

UNIFORM_PRONOUN = "they"

# Longest-first so that "himself"/"herself"/"hers" are consumed before "him"/"her"/"he".
GENDERED_PRONOUNS = ["himself", "herself", "hers", "his", "him", "her", "she", "he"]

_GRAM_BASE = {
    "he": "they",
    "she": "they",
    "him": "them",
    "his": "their",
    "hers": "theirs",
    "himself": "themselves",
    "herself": "themselves",
}

# Words that, following "her", indicate the possessive reading ("her role").
# Deliberately a heuristic, not a tagger. A grammatical deployment alternative
# would need pronoun disambiguation, verb agreement and independent validation.
_POSSESSIVE_STOP = {
    "is", "was", "are", "were", "and", "or", "but", "to", "for", "as", "with",
    "in", "on", "at", "by", "from", "that", "who", "which", "than", "then",
    "both", "very", "so", "too", "not", "well", "again", "also",
}


def _word_re(words: Iterable[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def _replace_names(text: str, names, placeholder: str) -> str:
    if names is None:
        return text
    found = names(text) if callable(names) else list(names)
    found = [n for n in dict.fromkeys(found) if n and n.strip()]
    if not found:
        return text
    # Longest first so "A. Smith" is consumed before "Smith".
    for name in sorted(found, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", placeholder, text, flags=re.IGNORECASE)
    return text


def _uniform_pronouns(text: str) -> str:
    return _word_re(GENDERED_PRONOUNS).sub(UNIFORM_PRONOUN, text)


def _grammatical(text: str, her: str) -> str:
    mapping = dict(_GRAM_BASE, her=her)

    def sub(m: re.Match) -> str:
        return mapping[m.group(0).lower()]

    return _word_re(mapping).sub(sub, text)


def _grammatical_pos(text: str) -> str:
    """Experimental following-word heuristic; not a grammatical rewriter.

    The legacy key ``gram_pos`` is retained for cached-result compatibility.
    It equalises the study templates but fails, for example, on "her yesterday"
    and does not repair verb agreement ("they specializes").
    """
    mapping = dict(_GRAM_BASE)

    def sub(m: re.Match) -> str:
        word = m.group(0).lower()
        if word != "her":
            return mapping[word]
        rest = text[m.end():].lstrip()
        nxt = re.match(r"[A-Za-z']+", rest)
        if not nxt:
            return "them"  # sentence-final -> object ("praised her.")
        return "them" if nxt.group(0).lower() in _POSSESSIVE_STOP else "their"

    return _word_re(list(mapping) + ["her"]).sub(sub, text)


def deidentify(
    text: str,
    variant: str = "full",
    names: Iterable[str] | Callable[[str], list[str]] | None = None,
    placeholder: str = DEFAULT_PLACEHOLDER,
    lowercase: bool = True,
) -> str:
    """Apply one de-identification variant. See module docstring for variants."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    out = text
    # The gram_* variants replace names too: they are alternatives to `full`
    # that differ ONLY in how pronouns are mapped, so that A5 isolates the
    # pronoun channel. Without this the twins would differ by name regardless
    # and the comparison would be vacuous.
    if variant in ("names_only", "full", "gram_her_their", "gram_her_them", "gram_pos"):
        out = _replace_names(out, names, placeholder)

    if variant in ("pronouns_only", "full"):
        out = _uniform_pronouns(out)
    elif variant == "gram_her_their":
        out = _grammatical(out, "their")
    elif variant == "gram_her_them":
        out = _grammatical(out, "them")
    elif variant == "gram_pos":
        out = _grammatical_pos(out)

    return out.lower() if lowercase else out


def make_ner_detector(model_name: str = "dslim/bert-base-NER", device: str | None = None):
    """Return a `detect(text) -> list[str]` backed by a cached NER model.

    Used by the realistic-pool study; the synthetic study passes its own literal
    name list instead. Kept lazy so importing this module needs no torch.
    """
    from transformers import pipeline

    ner = pipeline("token-classification", model=model_name,
                   aggregation_strategy="simple", device=device)

    def detect(text: str) -> list[str]:
        spans = [e["word"].strip() for e in ner(text)
                 if e.get("entity_group") == "PER" and e.get("score", 0) >= 0.80]
        # Split multi-token names so "John Smith" also masks a bare "Smith".
        parts = {p for s in spans for p in s.split() if len(p) > 1}
        return sorted(set(spans) | parts, key=len, reverse=True)

    return detect


if __name__ == "__main__":
    m = ("Daniel is a dedicated nurse. He specializes in his area of expertise, "
         "and colleagues praised him for his work; they describe him as capable.")
    f = ("Susan is a dedicated nurse. She specializes in her area of expertise, "
         "and colleagues praised her for her work; they describe her as capable.")
    names = ["Daniel", "Susan"]
    print(f"{'variant':<16}{'twins identical?':<18} female-form output")
    for v in VARIANTS:
        a = deidentify(m, v, names)
        b = deidentify(f, v, names)
        print(f"{v:<16}{str(a == b):<18}{b[:78]}")
