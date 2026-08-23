"""Canonical occupation -> stereotype-label lookup (single source of truth).

Every analysis script must get its occupation labels from here, never from a
hard-coded dict, so that the BLS relabelling (REVIEW_TODO.md A8) propagates
everywhere at once.

Preference order:
  1. `code/occupations_bls.csv` -- labels derived from BLS CPS Table 11 (% women).
  2. The original hand/embedding labels baked into `synthetic_dataset.py`.

The fallback exists so that every script keeps running while the relabelling
lands; `LABEL_SOURCE` records which one was used and MUST be written into every
results JSON so a reader can tell which taxonomy produced a number.

CSV schema (`code/occupations_bls.csv`), one row per occupation:
    occupation,bls_title,soc_code,pct_women,label,year
      occupation : the string used in synthetic_dataset.OCCUPATIONS
      bls_title  : the matching BLS CPS Table 11 row title (verbatim)
      soc_code   : SOC code if available, else empty
      pct_women  : float 0-100 from BLS, else empty
      label      : male | female | neutral | drop
      year       : BLS annual-average year, e.g. 2025

Label rule (A8): male if pct_women < 30, female if pct_women > 70,
neutral if 40 <= pct_women <= 60, otherwise `drop` (the 30-40 / 60-70 bands are
deliberately excluded as ambiguous). Occupations with no BLS row are `drop`.
"""

from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "occupations_bls.csv"

VALID_LABELS = {"male", "female", "neutral"}


def _fallback_labels() -> dict[str, str]:
    from synthetic_dataset import OCCUPATIONS

    return {occ: stereo for stereo, occs in OCCUPATIONS.items() for occ in occs}


def load_labels(strict: bool = False) -> tuple[dict[str, str], str]:
    """Return (occupation -> label, source_description).

    Occupations labelled `drop` in the CSV are excluded from the returned map;
    callers should skip any occupation missing from it.

    If `strict` is True, raise when the BLS CSV is absent instead of falling
    back -- use that in final paper-number scripts so a missing relabelling
    cannot silently produce old numbers.
    """
    if CSV_PATH.exists():
        labels: dict[str, str] = {}
        with CSV_PATH.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                label = (row.get("label") or "").strip().lower()
                occ = (row.get("occupation") or "").strip()
                if occ and label in VALID_LABELS:
                    labels[occ] = label
        if labels:
            return labels, f"BLS CPS Table 11 via {CSV_PATH.name}"

    if strict:
        raise FileNotFoundError(
            f"{CSV_PATH} not found (or empty); refusing to fall back to the "
            "pre-BLS labels. See REVIEW_TODO.md A8."
        )
    return _fallback_labels(), "legacy hand/embedding labels (synthetic_dataset.py)"


def filter_rows(rows: list[dict], labels: dict[str, str], key: str = "occupation") -> list[dict]:
    """Keep only rows whose occupation survives the label map, and overwrite
    their `stereotype` field with the canonical label."""
    out = []
    for r in rows:
        lab = labels.get(r.get(key))
        if lab is not None:
            r = dict(r)
            r["stereotype"] = lab
            out.append(r)
    return out


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    labels, source = load_labels()
    print(f"source: {source}")
    print(f"occupations: {len(labels)}  {dict(Counter(labels.values()))}")
