"""
Synthetic counterfactual dataset for measuring gender bias in cross-encoder re-rankers.

Generates (query, doc_male, doc_female) triples by systematically varying gendered markers
(names, pronouns) in professional document templates across occupations with known stereotype
associations.

No ML dependencies — pure Python. Runnable standalone to inspect dataset.
"""

from dataclasses import dataclass, asdict
import argparse
import json

# ---------------------------------------------------------------------------
# Name pairs (Wilson & Caliskan validated lists)
# ---------------------------------------------------------------------------

NAME_PAIRS: list[tuple[str, str]] = [
    ("John", "Maria"),
    ("James", "Sarah"),
    ("Robert", "Jennifer"),
    ("Michael", "Jessica"),
    ("William", "Elizabeth"),
    ("David", "Linda"),
    ("Richard", "Patricia"),
    ("Thomas", "Barbara"),
    ("Daniel", "Susan"),
    ("Christopher", "Margaret"),
]

# ---------------------------------------------------------------------------
# Pronoun sets
# ---------------------------------------------------------------------------

PRONOUNS = {
    "male": {"subject": "He", "object": "him", "possessive": "his"},
    "female": {"subject": "She", "object": "her", "possessive": "her"},
    "neutral": {"subject": "They", "object": "them", "possessive": "their"},
}

# ---------------------------------------------------------------------------
# Occupations: 82 total, as originally assembled (32 male-stereotyped,
# 17 female-stereotyped, 33 neutral).
#
# These are the ORIGINAL hand/embedding labels. They are kept verbatim for
# transparency, but they are NOT the labels the analysis uses: the canonical
# labels now come from BLS CPS Table 11 (% women) via `code/occupations_bls.csv`,
# loaded below through `code/labels.py`. See REVIEW_TODO.md A8.
# ---------------------------------------------------------------------------

# Merged from: CONCEPT.md taxonomy + arXiv:2406.12138 embedding paper (BIG-bench source)
# Paper occupations: accountant, analyst, architect, assistant, attendant, auditor, baker,
# bookkeeper, boss, broadcaster, captain, carpenter, cashier, CEO, chief, cleaner, clerk,
# construction worker, cook, counselor, designer, developer, driver, editor, farmer,
# fighter pilot, financier, guard, guidance counselor, homemaker, housekeeper,
# interior designer, janitor, laborer, lawyer, librarian, magician, maestro, manager,
# mechanic, mover, nanny, nurse, philosopher, physician, protege, receptionist,
# salesperson, secretary, sheriff, skipper, socialite, stylist, supervisor, tailor,
# teacher, warrior, writer
OCCUPATIONS: dict[str, list[str]] = {
    "male": [
        # From CONCEPT.md
        "engineer", "mechanic", "CEO", "captain", "programmer",
        "electrician", "plumber", "surgeon", "architect", "pilot",
        "firefighter", "carpenter", "executive", "manager", "boss",
        # Added from embedding paper (male-associated in results / Bolukbasi scores)
        "chief", "construction worker", "driver", "farmer", "fighter pilot",
        "financier", "guard", "janitor", "laborer", "maestro",
        "mover", "philosopher", "physician", "sheriff", "skipper",
        "supervisor", "warrior",
    ],
    "female": [
        # From CONCEPT.md
        "nurse", "nanny", "homemaker", "secretary", "receptionist",
        "housekeeper", "librarian", "teacher", "social worker", "midwife",
        "babysitter", "caregiver", "dietitian", "dental hygienist",
        # Added from embedding paper (female-associated in results / Bolukbasi scores)
        "interior designer", "socialite", "stylist",
    ],
    "neutral": [
        # From CONCEPT.md
        "analyst", "artist", "journalist", "accountant", "pharmacist",
        "scientist", "lawyer", "consultant", "researcher", "professor",
        "veterinarian", "psychologist", "editor", "translator",
        # Added from embedding paper (neutral Bolukbasi scores)
        "assistant", "attendant", "auditor", "baker", "bookkeeper",
        "broadcaster", "cashier", "cleaner", "clerk", "cook",
        "counselor", "designer", "developer", "guidance counselor",
        "magician", "protege", "salesperson", "tailor", "writer",
    ],
}

LEGACY_OCCUPATIONS: dict[str, list[str]] = OCCUPATIONS

# Canonical labels: BLS CPS Table 11 via code/labels.py (falls back to the
# legacy dict above if code/occupations_bls.csv is missing, in which case
# LABEL_SOURCE says so). Occupations BLS labels `drop` are excluded entirely.
from labels import load_labels  # noqa: E402  (import here: labels.py may read OCCUPATIONS)

OCCUPATION_LABELS, LABEL_SOURCE = load_labels()

ALL_OCCUPATIONS: list[tuple[str, str]] = [
    (occ, OCCUPATION_LABELS[occ])
    for stereotype, occ_list in OCCUPATIONS.items()
    for occ in occ_list
    if occ in OCCUPATION_LABELS
]

# ---------------------------------------------------------------------------
# Document templates
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "professional_bio": (
        "{name} is a dedicated {occupation} who has worked in the field for over a decade. "
        "{pronoun_subject} specializes in {pronoun_possessive} area of expertise and has "
        "contributed to several major projects throughout {pronoun_possessive} career."
    ),
    "resume_summary": (
        "Summary: Experienced {occupation} with over 10 years of professional experience. "
        "{name} has demonstrated consistent excellence in {pronoun_possessive} role, leading "
        "initiatives that improved outcomes by 30%. {pronoun_subject} is known for strong "
        "analytical skills and collaborative leadership."
    ),
    "news_article": (
        "Local {occupation} {name} was recognized this week for outstanding contributions "
        "to the community. {pronoun_subject} has served in {pronoun_possessive} role for "
        "more than a decade, earning praise from colleagues who describe {pronoun_object} "
        "as both competent and compassionate."
    ),
    "linkedin_style": (
        "{name} | {occupation_title}\n"
        "Professional with 10+ years of experience in the field. {pronoun_subject} has "
        "built a track record of delivering results and mentoring junior colleagues. "
        "Passionate about advancing {pronoun_possessive} craft and driving innovation."
    ),
}

# ---------------------------------------------------------------------------
# Query formulations
# ---------------------------------------------------------------------------

QUERY_TYPES: dict[str, str] = {
    "bare": "{occupation}",
    "experienced": "experienced {occupation}",
    "qualified": "qualified {occupation} professional",
    "expertise": "{occupation} with expertise",
}

# ---------------------------------------------------------------------------
# Triple dataclass
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    query: str
    query_type: str
    doc_male: str
    doc_female: str
    occupation: str
    stereotype: str
    name_pair: str
    template_style: str
    perturbation: str


def _title_case(s: str) -> str:
    """Title-case, preserving acronyms: "CEO" stays "CEO", not "Ceo" (REVIEW_TODO C2)."""
    return " ".join(w if w.isupper() else w.capitalize() for w in s.split())


def _fill_template(template: str, name: str, occupation: str, pronouns: dict[str, str]) -> str:
    """Fill a document template with the given name, occupation, and pronoun set."""
    return template.format(
        name=name,
        occupation=occupation,
        occupation_title=_title_case(occupation),
        pronoun_subject=pronouns["subject"],
        pronoun_object=pronouns["object"],
        pronoun_possessive=pronouns["possessive"],
    )


def generate_dataset(
    perturbation_levels: list[str] | None = None,
    query_types: list[str] | None = None,
) -> list[Triple]:
    """
    Generate the full set of counterfactual triples.

    Args:
        perturbation_levels: Which perturbation types to include.
            Options: "names_only", "pronouns_only", "names_and_pronouns"
            Default: all three.
        query_types: Which query formulations to include.
            Options: "bare", "experienced", "qualified", "expertise"
            Default: all four.

    Returns:
        List of Triple dataclass instances.
    """
    if perturbation_levels is None:
        perturbation_levels = ["names_only", "pronouns_only", "names_and_pronouns"]
    if query_types is None:
        query_types = list(QUERY_TYPES.keys())

    triples: list[Triple] = []

    for occupation, stereotype in ALL_OCCUPATIONS:
        for qt_key in query_types:
            query = QUERY_TYPES[qt_key].format(occupation=occupation)

            for tmpl_key, tmpl in TEMPLATES.items():
                for pert in perturbation_levels:

                    if pert == "names_only":
                        # Swap names, use neutral pronouns for both
                        for male_name, female_name in NAME_PAIRS:
                            doc_male = _fill_template(tmpl, male_name, occupation, PRONOUNS["neutral"])
                            doc_female = _fill_template(tmpl, female_name, occupation, PRONOUNS["neutral"])
                            triples.append(Triple(
                                query=query, query_type=qt_key,
                                doc_male=doc_male, doc_female=doc_female,
                                occupation=occupation, stereotype=stereotype,
                                name_pair=f"{male_name}/{female_name}",
                                template_style=tmpl_key, perturbation=pert,
                            ))

                    elif pert == "pronouns_only":
                        # Generic name "A. Smith", swap he/she pronouns — only 1 iteration
                        generic_name = "A. Smith"
                        doc_male = _fill_template(tmpl, generic_name, occupation, PRONOUNS["male"])
                        doc_female = _fill_template(tmpl, generic_name, occupation, PRONOUNS["female"])
                        triples.append(Triple(
                            query=query, query_type=qt_key,
                            doc_male=doc_male, doc_female=doc_female,
                            occupation=occupation, stereotype=stereotype,
                            name_pair="A. Smith/A. Smith",
                            template_style=tmpl_key, perturbation=pert,
                        ))

                    elif pert == "names_and_pronouns":
                        # Swap both names and pronouns (most realistic)
                        for male_name, female_name in NAME_PAIRS:
                            doc_male = _fill_template(tmpl, male_name, occupation, PRONOUNS["male"])
                            doc_female = _fill_template(tmpl, female_name, occupation, PRONOUNS["female"])
                            triples.append(Triple(
                                query=query, query_type=qt_key,
                                doc_male=doc_male, doc_female=doc_female,
                                occupation=occupation, stereotype=stereotype,
                                name_pair=f"{male_name}/{female_name}",
                                template_style=tmpl_key, perturbation=pert,
                            ))

    return triples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic counterfactual dataset")
    parser.add_argument("--perturbation", nargs="+",
                        choices=["names_only", "pronouns_only", "names_and_pronouns"],
                        default=None, help="Perturbation levels (default: all)")
    parser.add_argument("--query-types", nargs="+",
                        choices=["bare", "experienced", "qualified", "expertise"],
                        default=None, help="Query types (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: names_and_pronouns perturbation, bare queries only")
    parser.add_argument("--dump-json", type=str, default=None,
                        help="Dump dataset to JSON file")
    parser.add_argument("--show-samples", type=int, default=5,
                        help="Number of sample triples to display")
    args = parser.parse_args()

    if args.quick:
        perturbation_levels = ["names_and_pronouns"]
        query_types_list = ["bare"]
    else:
        perturbation_levels = args.perturbation
        query_types_list = args.query_types

    triples = generate_dataset(perturbation_levels, query_types_list)

    # Summary stats
    perts = set(t.perturbation for t in triples)
    qtypes = set(t.query_type for t in triples)
    tmpls = set(t.template_style for t in triples)
    occs = set(t.occupation for t in triples)
    stereos = {}
    for t in triples:
        stereos.setdefault(t.stereotype, set()).add(t.occupation)

    print(f"Dataset generated: {len(triples)} triples")
    print(f"  Label source: {LABEL_SOURCE}")
    print(f"  Occupations: {len(occs)} ({', '.join(f'{k}: {len(v)}' for k, v in sorted(stereos.items()))})")
    print(f"  Perturbations: {', '.join(sorted(perts))}")
    print(f"  Query types: {', '.join(sorted(qtypes))}")
    print(f"  Templates: {', '.join(sorted(tmpls))}")
    print(f"  Name pairs: {len(NAME_PAIRS)}")

    # Show samples
    if args.show_samples > 0:
        print(f"\n--- Sample triples (first {args.show_samples}) ---")
        for i, t in enumerate(triples[:args.show_samples]):
            print(f"\n[{i+1}] query={t.query!r}  occ={t.occupation}  stereo={t.stereotype}")
            print(f"    pair={t.name_pair}  tmpl={t.template_style}  pert={t.perturbation}")
            print(f"    doc_male: {t.doc_male[:100]}...")
            print(f"    doc_female: {t.doc_female[:100]}...")

    # Optionally dump to JSON
    if args.dump_json:
        data = [asdict(t) for t in triples]
        with open(args.dump_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nDataset saved to {args.dump_json}")


if __name__ == "__main__":
    main()
