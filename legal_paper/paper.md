---
title: Gender Bias in Automated Hiring\\ --- the Case for De-Identification
author: Vasyl Rakivnenko
date: Draft
---

```latex
% ============================================================================
% This .md file is the SOURCE of the paper. Edit the prose freely.
% Build it with:   python3 build.py     (writes main.tex, main.pdf, and the zip)
% Conventions:  *italic*  **bold**  "quotes"  `code`  [@citekey]
%               # Section   ## Subsection   ### Run-in heading
% Blocks fenced as ```latex (tables, figures, references) are raw LaTeX:
% they pass through untouched -- leave them alone unless you know LaTeX.
% <!-- html comments --> are editorial notes; build.py strips them.
%
% NUMBERS ARE NEVER TYPED BY HAND. Every statistic is a \{\{PLACEHOLDER\}\} that
% build.py substitutes from results/*.json via numbers.json. An unresolved
% placeholder FAILS the build (use `python3 build.py --draft` to see the PDF
% anyway, with the gaps marked in red). To add a number: put the placeholder
% here and an entry in numbers.json. Never paste a value into the prose.
%
% HOUSE STYLE (the author's brief): readable by a non-technical reader.
% Sentences <= 25 words. One idea per paragraph, <= 6 sentences. No formulas
% in the body. Every number gets its plain meaning inline.
% ============================================================================
```

# Abstract

Hiring tools increasingly sort applicants automatically. A retriever pulls plausible resumes from a large pool. A re-ranker then puts them in final order. We audited {{N_MODELS}} commercial and open-source re-rankers with counterfactual pairs: texts identical except for the candidate's name and pronouns. All {{N_MODELS_STEREOTYPE_ORDER}} favored men more often for male-typed jobs than for female-typed ones. Which way a model leans by default is a property of that product, and cannot be known without testing it. In a simulated pool of {{SLATE_SIZE}} equally qualified candidates, one common pipeline gave men {{SHORTLIST_PCT_MALE_PIPELINE}} of shortlist places, {{SHORTLIST_PCT_MALE_PIPELINE_GLOSS}}. Removing names and pronouns before scoring removes the difference, because it removes the only thing that differs. Half-measures do not: deleting names alone leaves the direction almost perfectly consistent. We argue that de-identification is a cheap, auditable, less discriminatory alternative that deployers should be expected to adopt.

# Why this matters

Software that sorts job applicants usually works in two steps [@nogueira2019]. A *retriever* scans a large pool and pulls out a few dozen plausible candidates. A *re-ranker* then reads each one against the job description and fixes the final order. The retriever is the clerk who pulls a stack of files; the re-ranker is the decision-maker who reads them and arranges them best-first. The people at the top of that order get looked at. The rest, in practice, do not.

This is not hypothetical. LinkedIn Recruiter returns a ranked list of candidates [@linkedin]; the same ranking technology runs the job-seeker side of the market at Indeed and Google [@indeed; @google]. Published resume-matching systems score a job description and a candidate document together, using the same family of neural rankers we audit [@shpjf; @confit].

Researchers have shown that the embedding models behind the retriever carry gender bias [@caliskan2017; @rakivnenko2024; @wilson2024]. The re-ranker has been studied far less, and in a different way: no prior audit scores identical candidates across commercial ranking products. That is the wrong way round. The re-ranker has the last word, and it decides who appears at the top of the recruiter's screen.

Gender bias in neural rankers is not new. Rekabsaz and Schedl showed that BERT rankers surface more male-oriented passages than keyword search for neutral queries [@rekabsaz2020], and later added an adversarial fix [@rekabsaz2021]. Both papers ask what the retrieved text is about, not how two candidates who differ only by sex are scored. Audits of language models do ask that, and they disagree on direction. One study of large language models scoring resumes finds an advantage for women [@an2025]; an embedding-based resume screener favors male-associated names [@wilson2024]. We run the counterfactual test on the re-rankers that commercial hiring stacks actually use, and follow it through to who makes the shortlist.

Two legal regimes already cover this ground. In the United States, a neutral-looking practice that screens out one sex at a disproportionate rate is unlawful unless the employer shows business necessity. Even then it fails if a less discriminatory alternative exists and the employer declines it. In the European Union, systems that filter job applications are classed as high-risk. From December 2027, the organisations that deploy them owe duties to control that risk. Both regimes ask the same practical question: is there a cheaper, fairer way to do the same job?

# What we did

Our test is a counterfactual pair: two versions of the same professional document. They are identical word for word, except for the candidate's name and pronouns. Any difference in score must therefore come from gender, because nothing else differs. We built these documents from {{N_TEMPLATES}} templates across {{N_OCCUPATIONS}} occupations. We varied {{N_NAME_PAIRS}} common American given names of each sex, taken from national birth-name records [@ssa2026]. We used {{N_QUERIES}} ways of phrasing the search. That yields {{N_PAIRS}} counterfactual pairs per model. The appendix shows the templates, the query forms, and the name pairs in full.

We scored {{N_OCC_SCORED}} occupations, then labeled each by the share of women employed in it, from the federal labor-force survey [@bls2025]. {{N_OCC_DROPPED}} occupations had no usable labor-force figure -- no matching category, a suppressed estimate, or a share too close to the 30% and 70% cutoffs to call -- and are excluded from every number in this paper; the appendix lists all of them, with the reason for each. Of the remaining {{N_OCCUPATIONS}}, {{N_OCC_MALE}} are male-typed, {{N_OCC_FEMALE}} are female-typed, and {{N_OCC_NEUTRAL}} are close to evenly split. The evenly split group carries most of the legal weight, because nobody can claim those jobs require a particular sex.

We scored every counterfactual pair with {{N_MODELS}} re-rankers, both commercial and open-source. Some models return coarse scores that give the two documents exactly the same number. We count those ties as ties. The tables report them in their own column; where we need a single share, a tie counts as half a win for each document. The appendix lists every model, its vendor, and its tie rate.

To see what this does in practice, we simulated a pool of candidates for one job. It holds {{SLATE_SIZE}} documents, half with men's names and half with women's, identical in every other respect. Because the candidates are by construction equally qualified, a fair system should treat them interchangeably. We ran the pool through a common open-source pipeline -- the retriever {{RETRIEVER_MODEL}} feeding its top {{TOPK}} candidates to the re-ranker {{RERANKER_MODEL}} -- and recorded who made the top three. Finally, we re-scored everything after stripping gender markers, at several strengths, to see which strengths actually work.

# What we found

## Identical candidates are not scored equally

All {{N_MODELS_STEREOTYPE_ORDER}} models favored the man more often for male-typed jobs than for female-typed ones. The size of that gap varies; its direction never does. For male-typed jobs the median model favored the man in {{MEDIAN_PCT_MALE_MALEJOBS}} of comparisons, {{MEDIAN_PCT_MALE_MALEJOBS_GLOSS}}. For female-typed jobs the median model favored the man in only {{MEDIAN_PCT_MALE_FEMALEJOBS}}. Nothing changed between the two documents but the name and the pronouns. The tilt is not random. It follows the same stereotype that would be unlawful in a human screener.

Where a model sits by default is a different matter. On jobs the labor market splits evenly, the median model favored the man in {{MEDIAN_PCT_MALE_NEUTRAL}} of comparisons, {{MEDIAN_PCT_MALE_NEUTRAL_GLOSS}}. But the models themselves disagree: {{N_MODELS_NEUTRAL_LEAN_MALE}} of the {{N_MODELS}} lean toward men on those jobs, and {{N_MODELS_NEUTRAL_LEAN_FEMALE}} lean the other way. A buyer therefore cannot guess which sex a given product disadvantages. The direction is invisible from the outside, and knowable only by testing.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_single_stage.pdf}
\caption{Each point is one re-ranker: how often it favored the man's document
for male-typed jobs (horizontal) against how often it favored the man's document
for female-typed jobs (vertical). A model that treated the pair alike would sit
at 50\% on both axes. Points are numbered as in the appendix tables, and the marker
shape gives the vendor. Where a model scored the two documents exactly equal we split the
tie evenly rather than crediting either side. Bars show the 95\% range across
occupations.}
\label{fig:single}
\end{figure}
```

## The re-ranker has the last word

A pool of {{SLATE_SIZE}} interchangeable candidates should produce a shortlist that is half men and half women. It does not. We expected the two stages to compound. A candidate must clear both, so two filters leaning the same way should skew the shortlist further than either alone. That is what happens where they do lean together. On male-typed jobs the retriever alone gave men {{SHORTLIST_PCT_MALE_SEARCH_MALEJOBS}} of the top three places, the re-ranker alone {{SHORTLIST_PCT_MALE_RANKER_MALEJOBS}}, and the two in sequence {{SHORTLIST_PCT_MALE_PIPELINE_MALEJOBS}} -- worse than either stage on its own.

Where the two stages disagree, the effect reverses. On female-typed jobs the retriever leans the other way, giving men {{SHORTLIST_PCT_MALE_SEARCH_FEMJOBS}} of shortlist places. That pulls the pipeline back from the re-ranker's {{SHORTLIST_PCT_MALE_RANKER_FEMJOBS}} to {{SHORTLIST_PCT_MALE_PIPELINE_FEMJOBS}}. Across all jobs the two effects cancel, and the pipeline comes to resemble the re-ranker alone: {{SHORTLIST_PCT_MALE_PIPELINE}} against {{SHORTLIST_PCT_MALE_RANKER}}. For the evenly split jobs it is {{SHORTLIST_PCT_MALE_PIPELINE_NEUTRAL}}.

Favoring either sex among interchangeable candidates is a disparity, so none of these figures is the right answer. Two lessons follow. The pipeline cannot be read off either stage alone; what matters is whether the two happen to agree. And fixing the retriever achieves little on its own, because the re-ranker has the last word.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.82\linewidth]{figures/fig_pipeline.pdf}
\caption{Share of top-three shortlist places going to men's names, in a pool of
equally qualified candidates: under the retriever alone, the re-ranker
alone, and the two combined.}
\label{fig:pipeline}
\end{figure}
```

## Removing the markers removes the gap, if you remove all of them

If both documents become the same text, any model must score them the same. Our full transform replaces the name and neutralises the pronouns, and every pair then ties. That is a check on the argument rather than a discovery, because identical texts cannot be scored differently. Half-measures are the interesting case, and they cut the other way from what you might guess.

Deleting names but leaving pronouns does not shrink the score gap -- if anything it is slightly larger: {{DEID_NAMESONLY_MEAN_GAP}} against {{DEID_ORIGINAL_MEAN_GAP}}. What changes is consistency: the man's document wins {{DEID_NAMESONLY_PCT_MALE_MALEJOBS}} of male-typed comparisons, up from {{DEID_ORIGINAL_PCT_MALE_MALEJOBS}}. Removing the name does not take away a large signal and leave a small one behind. It takes away the noisier signal and leaves the more consistent one. Neutralising pronouns but keeping the name shows the same thing from the other side: the gap falls to {{DEID_PRONOUNSONLY_MEAN_GAP}}. This pattern is not identical on every re-ranker we tested; the appendix repeats it on three more.

So a half-transform is not half a fix. The obvious repair, rewriting pronouns as *they*, *them*, and *their*, works only if the rewriter understands grammar. English *her* is both possessive and object, so mapping it to one word leaves the pair differing by a word. That happens on {{DEID_GRAM_NAIVE_PCT_RESIDUE}} of pairs, and where it happens {{DEID_GRAM_NAIVE_RESIDUE_PCT_FEMALE}} of the leftovers favor the woman's document, not the man's. A rewriter that checks each word's part of speech avoids the mistake and keeps the text readable. Neither route costs accuracy: the right candidate came out on top {{UTILITY_TOP1_AFTER}} of the time, slightly more often than the {{UTILITY_TOP1_BEFORE}} before any transform.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.82\linewidth]{figures/fig_deid.pdf}
\caption{Who scores higher under each de-identification rule, across all counterfactual
pairs. Each bar splits into the share favoring the man's document, the share
scored exactly equal, and the share favoring the woman's document. The last two
rules make the two documents the same text, so every pair ties; the simple
\emph{her $\rightarrow$ their} rewrite leaves a quarter of pairs still different.
Job-category breakdowns are in the appendix.}
\label{fig:deid}
\end{figure}
```

# What the law makes of it

### Judged by the result, not the mechanism

United States law does not ask how a hiring tool reaches its decision. It asks whether a neutral-looking practice selects one sex at a disproportionate rate [@titlevii; @griggs].\footnote{The rule is codified at 42 U.S.C. \S\,2000e-2(k)(1)(A). Clause (i) makes the practice unlawful unless the employer shows it is job related for the position in question and consistent with business necessity. Clause (ii) makes it unlawful anyway where the complaining party identifies an alternative employment practice and the employer refuses to adopt it.} So the number that matters is not the small score difference, but the shortlist. In our pool of interchangeable candidates, women took {{SHORTLIST_PCT_FEMALE_PIPELINE}} of shortlist places. Men were shortlisted at a rate of {{FF_MALE_RATE}} and women at {{FF_FEMALE_RATE}}, a selection ratio of {{FF_SELECTION_RATIO}}. That is {{FF_SELECTION_RATIO_GLOSS}} of the men's rate, far under the four-fifths level federal enforcement agencies treat as a warning line [@uniformguidelines].

Our pool is a stylised illustration of that metric, not an applicant-flow analysis of a real employer's hiring. It shows what ranking does to a group of people the system itself cannot tell apart. Small, consistent nudges in score become a hard line between making the cut and missing it.

### An employer cannot know the direction without testing

The direction of a model's default lean is a property of the individual product. A deployer therefore cannot know which sex a tool disadvantages until it is measured. Two vendors' models, both marketed as accurate, can push opposite ways on the same job. Buying on trust is not a defence, and a counterfactual test of the kind we run costs an afternoon.

### The evenly split jobs are the sharpest case

For a job the labor market divides evenly between men and women, no business-necessity story is available. An employer cannot argue that the work requires men. Yet the shortlist for those jobs was {{SHORTLIST_PCT_MALE_PIPELINE_NEUTRAL}} male. The skew is not a feature of the work. It is a property of the software, and a different re-ranker skews the other way (the appendix repeats the pipeline behind every audited re-ranker).

### De-identification is a textbook less discriminatory alternative

Even a justified practice is unlawful if a less discriminatory alternative exists and the employer refuses it. De-identification qualifies. It costs almost nothing and can be read and checked by hand. Within the channels it covers, it drives the disparity to zero rather than merely shrinking it. The alternative must also be equally effective at the underlying job.\footnote{42 U.S.C. \S\,2000e-2(k)(1)(A)(ii), read with \S\,2000e-2(k)(1)(C), which fixes the standard as the law stood on 4 June 1989. Under that law, cost and other burdens bear on whether a proposed alternative would be "equally as effective" in serving the employer's legitimate business goals [@watson].} That is why we measured accuracy, and found it unchanged.

### Europe arrives at the same place

The EU AI Act classes systems used to filter job applications as high-risk. It warns that such systems "may perpetuate historical patterns of discrimination, for example against women" [@euaiact]. The Act's design duties fall on the vendor, but an employer that only deploys the system still owes duties of its own. It must run the system as instructed, oversee it, and keep the input data relevant to the system's purpose.\footnote{Annex III, point 4(a) and Recital 57 classify recruitment systems as high-risk. The requirements addressed to the system itself --- risk management (Art.\ 9), data governance (Art.\ 10) and human oversight (Art.\ 14) --- bind the provider, through Art.\ 16(a). The deployer's own duties are in Art.\ 26: to use the system in accordance with the instructions for use (Art.\ 26(1)); to assign human oversight to competent staff (Art.\ 26(2)); to ensure input data is relevant and sufficiently representative so far as the deployer controls it (Art.\ 26(4)); and, for an employer, to inform workers and their representatives before use (Art.\ 26(7)). Under Art.\ 113 as amended by Regulation (EU) 2026/1744, these obligations apply from 2 December 2027 [@omnibus].} A further duty to assess the effect on fundamental rights before use reaches public bodies and public-service providers, not ordinary private employers.\footnote{Art.\ 27(1), which is limited to deployers that are bodies governed by public law or private entities providing public services, together with deployers of the credit and insurance systems in Annex III points 5(b) and 5(c). Regulation (EU) 2026/1744 amended only Art.\ 27(4) and (5); the scope in Art.\ 27(1) is unchanged.} De-identification is a measure the deployer controls, applied at exactly that input boundary.

# What to do about it

Anyone deploying two-step ranking over documents that describe people should remove gender markers from the documents and from the search queries, before either step runs. The specification below is short enough for a court, a regulator, or a compliance team to require and to verify.

1. **Replace names.** Substitute every personal name with a fixed placeholder such as "the candidate", or delete it. A real system needs a name detector rather than a word list, and that detector's coverage is itself an audit item.
2. **Neutralise pronouns, by one of two routes.** The simplest is to replace every gendered pronoun with a single token. That is trivial to implement, but it produces ungrammatical text. The alternative is to rewrite them as *they*, *them*, and *their* with a tool that checks each word's part of speech. Both make the two documents identical, and the second reads normally. Do not use a plain find-and-replace for *they/them/their*: English *her* is both possessive and object, and ignoring that leaves a gender signal behind.
3. **Apply it at both steps.** The re-ranker can override the retriever, so covering one step is not enough. Apply the same rules to the query as well as to the documents.
4. **Audit the transform, not only the model.** The substitution list is short and readable, so someone should read it. Someone should also spot-check the output for gendered words that slipped through: school names, awards, sports, honorifics.

The guarantee is simple, and it does not require trusting the model. Once the only gender-differing words are gone, the two documents are the same text, and any ranker must score the same text the same way.

# Limits

Our transform removes gender only from the channels it covers. Real resumes signal gender in other ways: single-sex schools, gendered awards, sports, writing style. Removing names and pronouns is therefore necessary but not sufficient in the field. Our documents are short synthetic templates rather than real applications.

The pipeline and de-identification results use one open-source pair as the worked example. The appendix repeats the pipeline behind every audited re-ranker and the de-identification test on four open-weight re-rankers, and the pattern is not identical across them. Our names are among the most common American given names and are mostly associated with White Americans, so we cannot separate gender from race. We treat gender as binary, which real applicants are not. The commercial models are black boxes, so we report the version strings and query dates in the appendix. One of the four open-weight re-rankers did not exactly reproduce its own earlier scores on a plain re-run, a caveat noted where that table appears.

One structural point deserves care in any extension to real documents. The two stages need not read the same text. Our retriever accepts 512 tokens; our re-ranker accepts 8,192. On a long resume the two stages would therefore score different amounts of the same candidate, and some of the divergence we attribute to the re-ranker would instead be an artifact of what each stage could see. Our documents are short enough that neither model truncates anything, so the comparison here is clean. It would not be on real resumes.

# Conclusion

Re-rankers score identical men and women differently, and the difference tracks job stereotypes. Which sex a given model favors by default cannot be known without testing it. In a pool of interchangeable candidates, one common pipeline gave men {{SHORTLIST_PCT_MALE_PIPELINE}} of shortlist places, and the re-ranker overrode the retriever. Removing names and pronouns before scoring removes the difference, because it removes the only thing that differs. The fix is cheap, checkable, and available today, and deployers should be expected to use it.

### Data and code availability

The dataset generator, the model scores, the pipeline simulation, and the de-identification code are at {{REPO_URL}}. Every number in this paper is produced by a script in that repository.

```latex
\appendix
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{table}{0}
```

# How to read the numbers

**Counterfactual pair.** One male-named document and its female-named counterpart, scored against the same query. The two texts are identical apart from the name and the gendered pronouns.

**The score gap.** The man's score minus the woman's score for one counterfactual pair. Models report scores on different, incomparable scales, so we never average the gap across models and report it only within a model.

**Favoring the man.** The share of counterfactual pairs in which the man's score is strictly higher. Equal scores are counted as ties and reported in their own column, never credited to either side.

**Tie-aware share.** The share favoring the man's document with each tie counted as half a win for each side. Figure 1, the last column of Table A2, and every single share reported in the main text use this convention; the tables that break results down further also show the plain three-way split.

**Shortlist.** The top {{SHORTLIST_SIZE}} of a pool of {{SLATE_SIZE}} equally qualified candidates for one job.

**Occupation labels.** Male-typed, female-typed, or evenly split, assigned from federal labor-force statistics on the share of women employed in each occupation.

**Confidence intervals.** The counterfactual pairs are not independent, because they are built from the same occupations, names, and templates. All intervals resample whole occupations rather than individual comparisons. They do not additionally resample name pairs or templates, whose effects are large (see Robustness); the true uncertainty is therefore somewhat wider than the reported ranges.

# The documents and the occupations

The four templates below are shown with their name and pronoun slots left as brackets, filled in for each of the ten name pairs and each of the eighty-two occupations we scored. The query forms and the name pairs, with each name's share female in national birth records, are the complete lists actually used.

```latex
{{TABLE_DOCUMENTS}}
```

The table below lists the fifty-two occupations kept for analysis, with the federal labor-force share of women behind each label, and the thirty scored but dropped, with the reason for each.

```latex
{{TABLE_OCCUPATIONS}}
```

# Models audited

Table~\ref{tab:models} lists every re-ranker in the audit with its vendor, how it was accessed, the date it was queried, its score range, and its tie rate. Requests to one deprecated Google model returned scores byte-identical to those of its successor, so the two are reported once, as one model. Table~\ref{tab:single} gives the three-way split for each model and job category, and Table~\ref{tab:consistency} counts how many individual occupations lean each way.

```latex
\begin{table}[H]
\centering
\caption{The re-rankers audited: vendor, access route, number of counterfactual
pairs, score range, share of comparisons that tied, and date scored.}
\label{tab:models}
\resizebox{\textwidth}{!}{%
{{TABLE_MODELS}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Share of counterfactual pairs favoring the man's document (M), tied, and
favoring the woman's document (F), by model and job category. The last column is
tie-aware: ties count as half a win each. An even-handed model would show equal
M and F shares in every row.}
\label{tab:single}
\resizebox{\textwidth}{!}{%
{{TABLE_SINGLE_STAGE}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Occupation-level consistency: how many individual occupations in each
category lean in the stereotype-consistent direction, out of the number of
occupations in that category.}
\label{tab:consistency}
\resizebox{\textwidth}{!}{%
{{TABLE_CONSISTENCY}}%
}
\end{table}
```

# Robustness

The lean varies with the wording of the document and of the search, and with which pair of names is used. Tables~\ref{tab:robtemplate}--\ref{tab:robname} break the tie-aware share favoring the man's document down three ways. The spread across name pairs is the largest of the three: that variation is exactly what removing names takes out, which is why deleting names changes how *consistent* the disparity is rather than how *large* it is (see the third finding).

```latex
\begin{table}[H]
\centering
\caption{Tie-aware share favoring the man's document, by document template.}
\label{tab:robtemplate}
\resizebox{\textwidth}{!}{%
{{TABLE_ROBUST_TEMPLATE}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Tie-aware share favoring the man's document, by phrasing of the search.}
\label{tab:robquery}
\resizebox{\textwidth}{!}{%
{{TABLE_ROBUST_QUERY}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Tie-aware share favoring the man's document, by name pair, split into
two tables of five pairs each for legibility.}
\label{tab:robname}
{{TABLE_ROBUST_NAMEPAIR}}
\end{table}
```

# The same pipeline with every re-ranker

The pipeline result in the body uses one open-source pair. Table~\ref{tab:sweep} repeats the simulation with each of the audited re-rankers behind the same retriever. The shortlist skew follows the re-ranker, in both size and direction, which is the point of the second finding.

The headline pipeline also uses a retrieval cutoff of {{TOPK}} -- how many candidates the retriever passes on before the re-ranker sees them. That choice is not neutral, and it is not symmetric. Narrowing the cut to 5 lowers the shortlist to {{PIPE_PCT_MALE_K5}} male, because the re-ranker is handed fewer candidates to reorder. Widening it to 15 changes almost nothing ({{PIPE_PCT_MALE_K15}}), since the re-ranker already dominates the ordering by that point. A deployer who tightens that first cut changes the disparity without touching either model.

```latex
\begin{table}[H]
\centering
\caption{Share of shortlist places going to men's names, by re-ranker, with
the same retriever and the same pool of equally qualified candidates.}
\label{tab:sweep}
\resizebox{\textwidth}{!}{%
{{TABLE_SWEEP}}%
}
\end{table}
```

# De-identification conditions

Table~\ref{tab:deid} reports every de-identification rule tested on the same re-ranker, including each half of the transform on its own and the grammatical rewrites. The full transform makes the two documents byte-identical, so its result follows by construction; it is reported as a check on the symmetry argument, not as a measurement.

```latex
\begin{table}[H]
\centering
\caption{Every de-identification rule tested on {{RERANKER_MODEL}}: the share of
counterfactual pairs favoring the man's document (M), tied, and favoring the
woman's document (F); the average size of the score gap on this model's own
scale; the share of pairs the rule leaves textually identical; and top-1
accuracy where measured. A dagger marks a rule that makes the two documents
identical by construction, so its row is a check on that fact rather than an
independent measurement.}
\label{tab:deid}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID}}%
}
\end{table}
```

Table~\ref{tab:deidall} repeats the same eight rules on three more re-rankers. The direction of the half-measures is not the same on every model: on {{RERANKER_MODEL}} and one other, removing names leaves the stronger and more one-directional signal; on a third, removing pronouns does; on the fourth, the two half-measures push toward opposite net directions entirely. What holds on all four is the part that matters for the policy: the full transform and the part-of-speech-aware rewrite tie every pair, on every model.

```latex
\begin{table}[H]
\centering
\caption{The same eight de-identification conditions on all four locally run
re-rankers: the tie-aware share favoring the man's document and the average
size of the score gap, on each model's own scale. One model's untransformed
scores did not exactly reproduce themselves on a plain re-run six months
later (Limits); its other seven rows were scored fresh in this same run and
are internally consistent.}
\label{tab:deidall}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID_ALLMODELS}}%
}
\end{table}
```

```latex
\begin{thebibliography}{99}

\bibitem{an2025}
J.~An, D.~Huang, C.~Lin, and M.~Tai.
\newblock Measuring gender and racial biases in large language models: Intersectional
evidence from automated resume evaluation.
\newblock \emph{PNAS Nexus}, 4(3):pgaf089, 2025. \doi{10.1093/pnasnexus/pgaf089}.

\bibitem{bls2025}
Bureau of Labor Statistics, U.S. Department of Labor.
\newblock Employed people by detailed occupation, sex, race, and Hispanic or Latino
ethnicity.
\newblock Current Population Survey, Household Data Annual Averages, Table 11,
2025 annual averages.
\url{https://www.bls.gov/cps/cpsaat11.htm} (accessed August 2026).

\bibitem{caliskan2017}
A.~Caliskan, J.~J.~Bryson, and A.~Narayanan.
\newblock Semantics derived automatically from language corpora contain human-like
biases.
\newblock \emph{Science}, 356(6334):183--186, 2017. \doi{10.1126/science.aal4230}.

\bibitem{titlevii}
Civil Rights Act of 1964, Title VII, 42 U.S.C. \S\S\,2000e \emph{et seq.}
\newblock Disparate impact codified at 42 U.S.C. \S\,2000e-2(k)(1)(A), added by the
Civil Rights Act of 1991, Pub.\ L.\ No.\ 102-166, \S\,105(a), 105 Stat.\ 1074.

\bibitem{google}
Google Cloud.
\newblock Cloud Talent Solution: Job Search overview.
\newblock \url{https://docs.cloud.google.com/talent-solution/job-search/docs}
(accessed August 2026).

\bibitem{griggs}
\emph{Griggs v.\ Duke Power Co.}, 401 U.S. 424 (1971).

\bibitem{shpjf}
Y.~Hou, X.~Pan, W.~X.~Zhao, S.~Bian, Y.~Song, T.~Zhang, and J.-R.~Wen.
\newblock Leveraging Search History for Improving Person-Job Fit.
\newblock In \emph{Database Systems for Advanced Applications (DASFAA 2022)}, Lecture
Notes in Computer Science 13245, pages 38--54. Springer, 2022.
\doi{10.1007/978-3-031-00123-9\_3}. arXiv:2203.14232.

\bibitem{indeed}
Indeed Engineering.
\newblock Distilling Long-Tail User Behavior into Scalable Embeddings for Job Search.
\newblock \url{https://engineering.indeedblog.com/blog/2026/06/distilling-long-tail-user-behavior-into-scalable-embeddings-for-job-search/}
(accessed June 2026).

\bibitem{linkedin}
LinkedIn Engineering.
\newblock The AI Behind LinkedIn Recruiter search and recommendation systems.
\newblock Q.~Guo, S.~C.~Geyik, C.~Ozcaglar, K.~Thakkar, N.~Anjum, and K.~Kenthapadi,
22 April 2019.
\url{https://www.linkedin.com/blog/engineering/recommendations/ai-behind-linkedin-recruiter-search-and-recommendation-systems}
(accessed August 2026).

\bibitem{nogueira2019}
R.~Nogueira and K.~Cho.
\newblock Passage Re-ranking with BERT.
\newblock arXiv preprint arXiv:1901.04085, 2019.

\bibitem{rakivnenko2024}
V.~Rakivnenko, N.~Maslej, J.~Cervi, and V.~Zhukov.
\newblock Bias in Text Embedding Models.
\newblock arXiv preprint arXiv:2406.12138, 2024.

\bibitem{euaiact}
Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024
(Artificial Intelligence Act), OJ L, 2024/1689, 12.7.2024.
\newblock Annex~III, point 4(a); Recital 57; Arts.\ 16(a), 26 and 27.

\bibitem{omnibus}
Regulation (EU) 2026/1744 of the European Parliament and of the Council of 8 July 2026
(Digital Omnibus on AI), OJ L, 2026/1744, 24.7.2026.
\newblock Amending Art.\ 113 of Regulation (EU) 2024/1689: Chapter~III, Sections~1--3
apply from 2 December 2027 to high-risk AI systems under Art.\ 6(2) and Annex~III.

\bibitem{rekabsaz2020}
N.~Rekabsaz and M.~Schedl.
\newblock Do Neural Ranking Models Intensify Gender Bias?
\newblock In \emph{Proceedings of the 43rd International ACM SIGIR Conference on
Research and Development in Information Retrieval}, pages 2065--2068, 2020.
\doi{10.1145/3397271.3401280}. arXiv:2005.00372.

\bibitem{rekabsaz2021}
N.~Rekabsaz, S.~Kopeinik, and M.~Schedl.
\newblock Societal Biases in Retrieved Contents: Measurement Framework and Adversarial
Mitigation of BERT Rankers.
\newblock In \emph{Proceedings of the 44th International ACM SIGIR Conference on
Research and Development in Information Retrieval}, pages 306--316, 2021.
\doi{10.1145/3404835.3462949}. arXiv:2104.13640.

\bibitem{ssa2026}
Social Security Administration.
\newblock National data on the relative frequency of given names in the population of
U.S. births.
\newblock Tabulated from Social Security records as of 1 March 2026.
\url{https://www.ssa.gov/oact/babynames/limits.html} (accessed August 2026).

\bibitem{uniformguidelines}
Uniform Guidelines on Employee Selection Procedures (1978), 29 C.F.R. pt.~1607.
\newblock Four-fifths rule at \S\,1607.4(D).

\bibitem{watson}
\emph{Watson v.\ Fort Worth Bank \& Trust}, 487 U.S. 977, 998 (1988) (plurality opinion).

\bibitem{wilson2024}
K.~Wilson and A.~Caliskan.
\newblock Gender, Race, and Intersectional Bias in Resume Screening via Language Model
Retrieval.
\newblock In \emph{Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society
(AIES)}, 7(1):1578--1590, 2024. \doi{10.1609/aies.v7i1.31748}.

\bibitem{confit}
X.~Yu, R.~Xu, C.~Xue, J.~Zhang, X.~Ma, and Z.~Yu.
\newblock ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding
and Runner-Up Hard-Negative Mining.
\newblock In \emph{Findings of the Association for Computational Linguistics: ACL 2025},
pages 12775--12790, 2025. \doi{10.18653/v1/2025.findings-acl.661}. arXiv:2502.12361.

\end{thebibliography}
```
