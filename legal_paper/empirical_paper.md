---
title: Gender-Associated Bias in Neural Rerankers\\ --- An Empirical Audit and De-Identification Test
author: Vasyl Rakivnenko
date:
---

```latex
% This is the source of the short empirical paper.
% Build the default paper with: python3 build.py
% Build the longer legal paper with: python3 build.py --paper legal
% Every reported result uses a named placeholder resolved through numbers.json.
```

# Abstract

Automated hiring pipelines can retrieve candidate documents with an embedding model and then reorder the leading candidates with a neural reranker. Prior audits have studied gender-associated differences in retrieval, while evidence about the reranking stage remains limited. We audit {{N_MODELS}} open and proprietary rerankers with {{N_PAIRS}} counterfactual document pairs per model. Within each pair, qualifications are unchanged and only the selected name and pronouns differ. All {{N_MODELS_STEREOTYPE_ORDER}} models favor the male-associated document more often for male-typed occupations than for female-typed occupations. In a synthetic pool of {{SLATE_SIZE}} equally qualified candidates, one retrieval-and-reranking pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE}} of top-{{SHORTLIST_SIZE}} positions to male-associated documents. Results vary materially across models, occupations, names, templates, and query wordings. We then test a simple input transform that replaces names with one placeholder and gendered pronouns with one uniform token before ranking. The transform makes each counterfactual pair textually identical and therefore removes score differences attributable to those markers by construction. On a synthetic occupation-matching task, top-rank accuracy changes from {{UTILITY_TOP1_BEFORE}} before transformation to {{UTILITY_TOP1_AFTER}} after transformation. These findings establish sensitivity to the tested names and pronouns, rather than discrimination in deployed hiring or a complete solution to demographic bias.

# Introduction

Modern search systems commonly retrieve a candidate set with an embedding model and then use a more expensive model to rerank that set [@nogueira2019]. The same architecture can be used for job and resume matching. A candidate must therefore pass two learned ordering stages before appearing near the top of a recruiter's list.

Evidence of social bias in word representations and retrieval is well established [@caliskan2017; @rakivnenko2024; @wilson2024]. Neural rankers can also amplify gendered patterns in retrieved content [@rekabsaz2020; @rekabsaz2021]. Resume-screening studies report effects in different directions across tasks and models [@an2025; @wilson2024]. Yet a broad counterfactual audit of current rerankers is missing, even though a reranker can reverse the order produced by retrieval.

We ask three narrow empirical questions. First, do current rerankers score otherwise identical professional documents differently when selected names and pronouns change? Second, how do such differences affect a two-stage shortlist? Third, which simple de-identification rules remove the measured difference without reducing performance on a basic relevance check?

Our contribution is a dated, reproducible audit rather than a claim about any employer's deployed system. We compare {{N_MODELS}} rerankers on a controlled synthetic corpus, propagate the scores through a retrieval-and-reranking simulation, and test de-identification on four locally runnable rerankers. The design isolates sensitivity to the changed text. It does not isolate gender from every other association carried by a name, and it does not measure actual hiring decisions.

# Experimental design

## Counterfactual documents

Each observation is a pair of short professional documents. The two versions describe the same occupation, experience, skills, and accomplishments. One version uses a male-associated given name and masculine pronouns; the other uses a female-associated given name and feminine pronouns. Thus, any within-pair score difference is caused by the changed name and pronoun tokens. Because names can also signal age, ethnicity, nationality, and individual identity, the design cannot attribute the whole difference to gender alone.

We combine {{N_TEMPLATES}} document templates, {{N_QUERIES}} query phrasings, and {{N_NAME_PAIRS}} pairs of common American given names. Names come from national birth-name records [@ssa2026]. The complete templates, queries, and names appear in Appendix A.

We initially score {{N_OCC_SCORED}} occupations. We label occupations with the share of women employed in the corresponding federal labor-force category [@bls2025]. We retain {{N_OCCUPATIONS}} occupations with a clear label: {{N_OCC_MALE}} male-typed, {{N_OCC_FEMALE}} female-typed, and {{N_OCC_NEUTRAL}} close to evenly split. The remaining {{N_OCC_DROPPED}} lack a usable match, have a suppressed estimate, or fall in an intentionally ambiguous range. Appendix A reports every inclusion and exclusion.

## Rerankers and analysis

We evaluate {{N_MODELS}} rerankers from commercial APIs and open-weight releases. The scores were collected on {{RAW_SCORED_DATE}}. Model identifiers, access routes, score ranges, tie rates, and collection dates appear in Appendix B.

For each model and occupation group, we report whether the male-associated version scores higher, the versions tie, or the female-associated version scores higher. When a single directional share is needed, a tie contributes one half to each side. Confidence intervals use a seeded cluster bootstrap that resamples occupations, because comparisons within an occupation are not independent. Scores from different models have different scales, so we compare directions and within-model gaps rather than raw magnitudes across models.

## Two-stage shortlist simulation

We form synthetic pools of {{SLATE_SIZE}} equally qualified documents, split evenly between male- and female-associated versions. The retriever {{RETRIEVER_MODEL}} selects the top {{TOPK}} candidates. The reranker {{RERANKER_MODEL}} then produces a top-{{SHORTLIST_SIZE}} shortlist. We also evaluate the retriever and reranker separately and repeat the pipeline with every audited reranker behind the same retriever. Because candidate qualifications are identical by construction, departures from an even shortlist reflect ordering effects from the changed markers and tie handling.

## De-identification

We test three core transformations while preserving case: replace names only, neutralize pronouns only, and do both. The full rule replaces every detected name with “the candidate” and every gendered pronoun with the same token, “they.” This deliberately simple normalization is applied before scoring. The two counterfactual versions then become the same string, so any deterministic text-only embedding model or reranker must assign them the same representation or score. This is a symmetry guarantee, not an empirical discovery.

We run the transformations on four open-weight rerankers. We also test relevance on a synthetic occupation-matching task. For each query, a model ranks one document from every retained occupation, and success means placing the document that names the queried occupation first. This check tests whether occupation signal survives; it does not test whether a system selects the best-qualified person within an occupation.

# Results

## Reranker scores track occupational gender composition

All {{N_MODELS_STEREOTYPE_ORDER}} audited models favor the male-associated document more often for male-typed occupations than for female-typed occupations. For the median model, the tie-aware male-favored share is {{MEDIAN_PCT_MALE_MALEJOBS}} on male-typed occupations and {{MEDIAN_PCT_MALE_FEMALEJOBS}} on female-typed occupations. The size varies across models, but the direction of this difference is consistent.

Evenly split occupations reveal substantial model variation. The median male-favored share is {{MEDIAN_PCT_MALE_NEUTRAL}}, but {{N_MODELS_NEUTRAL_LEAN_MALE}} models lie above an even split and {{N_MODELS_NEUTRAL_LEAN_FEMALE}} lie below it. For {{N_NEUTRAL_INDETERMINATE}} models, the occupation-clustered confidence interval includes an even split. A deployer therefore cannot infer either the size or direction of a model's behavior without testing that model.

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_single_stage.pdf}
\caption{\textbf{Reranker preferences across female- and male-typed
occupations.} Each row shows one reranker. Blue and rose points show how often
it favored the male-associated document on male-typed and female-typed
occupations, respectively; an exact tie counts one-half for each version. Rows
run from the largest change at the top to the smallest at the bottom. Every
blue point is to the right of its rose counterpart. Exact values and
occupation-clustered intervals are in Appendix B.}
\label{fig:single}
\end{figure}
```

The result is not invariant to design choices. Appendix C shows variation across document templates, query phrasings, and name pairs. Variation across names is especially large. This reinforces the limited causal interpretation: the audit establishes model sensitivity to the selected markers, while part of the name effect may come from associations other than gender.

## The reranker can dominate the pipeline

In the worked pipeline, the retriever alone assigns {{SHORTLIST_PCT_MALE_SEARCH}} of top-{{SHORTLIST_SIZE}} positions to male-associated documents. The reranker alone assigns {{SHORTLIST_PCT_MALE_RANKER}}, and the combined pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE}}. For occupations close to evenly split in the labor market, the combined share is {{SHORTLIST_PCT_MALE_PIPELINE_NEUTRAL}}.

The interaction depends on occupation type. On male-typed occupations, the male-associated share rises from {{SHORTLIST_PCT_MALE_SEARCH_MALEJOBS}} under retrieval to {{SHORTLIST_PCT_MALE_RANKER_MALEJOBS}} under reranking and {{SHORTLIST_PCT_MALE_PIPELINE_MALEJOBS}} in sequence. On female-typed occupations, retrieval assigns {{SHORTLIST_PCT_MALE_SEARCH_FEMJOBS}} to male-associated documents, while reranking assigns {{SHORTLIST_PCT_MALE_RANKER_FEMJOBS}} and the combined pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE_FEMJOBS}}. The two stages can therefore compound or counteract one another.

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_pipeline.pdf}
\caption{\textbf{Shortlist composition by pipeline stage and occupation group.}
Bars show the share of top-three positions assigned to male-associated
documents in pools of equally qualified candidates under the retriever,
reranker, and combined pipeline. Whiskers are 95\% occupation-clustered
bootstrap intervals.}
\label{fig:pipeline}
\end{figure}
```

Appendix D repeats the simulation with every audited reranker behind the same retriever. The resulting shortlist share changes substantially with the reranker. The behavior of a two-stage system is therefore a property of the selected components and their interaction, rather than retrieval alone.

## Full marker removal eliminates the measured pathway

Removing only names does not remove the difference. On the worked reranker, the mean absolute gap changes from {{DEID_ORIGINAL_MEAN_GAP}} to {{DEID_NAMESONLY_MEAN_GAP}}, while the male-associated version wins {{DEID_NAMESONLY_PCT_MALE_MALEJOBS}} of comparisons on male-typed occupations, compared with {{DEID_ORIGINAL_PCT_MALE_MALEJOBS}} before transformation. Neutralizing pronouns while retaining names reduces the mean absolute gap to {{DEID_PRONOUNSONLY_MEAN_GAP}}. These partial effects differ across the four rerankers in Appendix E.

The full rule makes every tested counterfactual pair textually identical. All pairs therefore tie under each tested reranker. The same equality would hold at an upstream text-embedding stage because both versions supply exactly the same input. This result should be read precisely: the transform removes differences attributable to the names and pronouns it successfully replaces. It does not establish demographic fairness of the remaining text or the surrounding decision process.

Synthetic occupation-matching top-rank accuracy is {{UTILITY_TOP1_BEFORE}} before the full transform and {{UTILITY_TOP1_AFTER}} after it. This provides evidence that the occupation signal in our templates survives the transformation. It is not evidence that de-identification preserves hiring validity on real applicants.

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_deid.pdf}
\caption{\textbf{Pairwise balance under de-identification rules.} Exact ties are
divided equally between the two sides for display. Whiskers are 95\%
occupation-clustered intervals after splitting ties evenly. The final two
50/50 bars are 100\% exact ties because those rules make the paired texts
identical; they are not observed wins for either document.}
\label{fig:deid}
\end{figure}
```

# Implications and future work

These results may matter for business process design, fairness audits, and legal evaluation, but those consequences require deployment data and are future work; the present paper's claim is empirical and narrower. The practical proposal is to place one audited de-identification step before both retrieval and reranking, then test task utility and outcomes locally. Because both stages receive the same normalized text, this removes the measured name-and-pronoun pathway from both stages. It may also reduce one source of repeated exclusion when many employers reuse the same ranking components, connecting our study to the “algorithmic monoculture” described by Bommasani and colleagues [@bommasani2026]. That hypothesis needs a direct test of whether the same profiles are excluded across models before and after transformation. Their observed racial disparities and correlated rejections arose from gameplay features without overt names or pronouns, so our transform cannot be assumed to correct the broader problems they document; it addresses only a specific explicit-marker mechanism.

# Limitations

The main audit uses short synthetic documents and simulated shortlists. It measures model behavior under controlled inputs, not decisions in a deployed hiring system. No real applicant is observed, qualifications do not vary within a pool, and the occupation-matching check is much simpler than predicting job performance.

Names carry several social signals at once. Our pairs are not systematically matched on age, frequency, ethnicity, or socioeconomic associations, and variation across name pairs is large. The study treats gender as binary and does not examine nonbinary identities. It also does not audit race or intersectional effects.

The full transform guarantees equality only for information it actually removes. Real resumes may signal gender through schools, affiliations, activities, job histories, and writing patterns [@parasurama2022]. Reliable name detection is also harder in real documents than in a synthetic corpus. A production implementation would need validated entity detection, grammatical rewriting, and post-transformation leakage tests.

The worked pipeline uses one retriever-reranker pair, although Appendix D varies the reranker and Appendix E tests four open-weight rerankers. Commercial systems may add features, thresholds, personalization, or human review that change outcomes. The results are a dated snapshot of model versions queried on {{RAW_SCORED_DATE}}.

A companion experiment sampled {{N_REAL_RESUMES}} distinct resumes from a corpus of {{N_REAL_CORPUS}}. Its model directions and magnitudes differ from the synthetic results, and its construction combines inserted names with swaps of sparse existing gender markers. Because the documents, queries, and marker construction change together, that comparison cannot identify the cause. We retain it as a warning against generalizing the synthetic effect size to real resumes.

# Conclusion

Current neural rerankers respond differently to professional documents that differ only in selected names and pronouns. Across {{N_MODELS}} models, the direction of that response consistently changes with occupational gender composition, while the baseline direction varies by product. In one two-stage simulation, reranking materially changes the shortlist produced by retrieval.

A simple normalization of names and gendered pronouns eliminates this particular pairwise difference because it makes the compared inputs identical. Synthetic occupation matching is retained, but real hiring utility and residual demographic leakage remain untested. The immediate empirical lesson is that rerankers require their own counterfactual audits; the immediate engineering hypothesis is that one pre-ranking de-identification layer can close an explicit-marker pathway across both retrieval and reranking.

### Data and code availability

The generated documents, banked model scores, analysis scripts, pipeline simulation, and de-identification code are available in the \href{{{REPO_URL}}}{project repository}. Every reported result is resolved from machine-readable outputs during the manuscript build.

```latex
\appendix
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{table}{0}
```

# Documents and occupations

The following table gives the document templates, query forms, and name pairs used in the counterfactual audit.

```latex
{{TABLE_DOCUMENTS}}
```

The next table reports retained occupations with their federal labor-force category and excluded occupations with the reason for exclusion.

```latex
{{TABLE_OCCUPATIONS}}
```

# Models and single-stage results

```latex
\begin{table}[H]
\centering
\caption{Audited rerankers, access routes, counterfactual comparisons, score
ranges, tie rates, and scoring dates.}
\label{tab:models}
\resizebox{\textwidth}{!}{%
{{TABLE_MODELS}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Counterfactual outcomes by reranker and occupation category. M and F
denote the male- and female-associated documents. The final column counts each
tie as half for each version and reports occupation-clustered intervals.}
\label{tab:single}
\resizebox{\textwidth}{!}{%
{{TABLE_SINGLE_STAGE}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Occupation-level directional consistency by model and category.}
\label{tab:consistency}
\resizebox{\textwidth}{!}{%
{{TABLE_CONSISTENCY}}%
}
\end{table}
```

# Robustness across inputs

```latex
\begin{table}[H]
\centering
\caption{Tie-aware male-favored share by document template.}
\label{tab:robtemplate}
\resizebox{\textwidth}{!}{%
{{TABLE_ROBUST_TEMPLATE}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Tie-aware male-favored share by query phrasing.}
\label{tab:robquery}
\resizebox{\textwidth}{!}{%
{{TABLE_ROBUST_QUERY}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Tie-aware male-favored share by name pair.}
\label{tab:robname}
{{TABLE_ROBUST_NAMEPAIR}}
\end{table}
```

# Pipeline results across rerankers

```latex
\begin{table}[H]
\centering
\caption{Male-associated share of shortlist positions for each reranker behind
the same retriever and the same pools of equally qualified documents.}
\label{tab:sweep}
\resizebox{\textwidth}{!}{%
{{TABLE_SWEEP}}%
}
\end{table}
```

# De-identification results

```latex
\begin{table}[H]
\centering
\caption{De-identification conditions on {{RERANKER_MODEL}}. A dagger marks a
condition that makes paired documents identical by construction. The utility
column is synthetic occupation-matching top-rank accuracy.}
\label{tab:deid}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{De-identification conditions across four locally run rerankers. One
model did not exactly reproduce its earlier untransformed scores on a later
local run; its transformed conditions were scored together and remain
internally comparable.}
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
\newblock Measuring gender and racial biases in large language models: Intersectional evidence from automated resume evaluation.
\newblock \emph{PNAS Nexus}, 4(3):pgaf089, 2025. \doi{10.1093/pnasnexus/pgaf089}.

\bibitem{bls2025}
Bureau of Labor Statistics, U.S. Department of Labor.
\newblock Employed people by detailed occupation, sex, race, and Hispanic or Latino ethnicity.
\newblock Current Population Survey, Household Data Annual Averages, Table 11, 2025 annual averages.
\url{https://www.bls.gov/cps/cpsaat11.htm}.

\bibitem{bommasani2026}
R.~Bommasani, S.~H.~Bana, K.~A.~Creel, D.~Jurafsky, and P.~Liang.
\newblock Algorithmic Monocultures in Hiring.
\newblock In \emph{Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26)}, 2026.
\doi{10.1145/3805689.3812400}. arXiv:2605.27371.

\bibitem{caliskan2017}
A.~Caliskan, J.~J.~Bryson, and A.~Narayanan.
\newblock Semantics derived automatically from language corpora contain human-like biases.
\newblock \emph{Science}, 356(6334):183--186, 2017. \doi{10.1126/science.aal4230}.

\bibitem{nogueira2019}
R.~Nogueira and K.~Cho.
\newblock Passage Re-ranking with BERT.
\newblock arXiv preprint arXiv:1901.04085, 2019.

\bibitem{parasurama2022}
P.~Parasurama and J.~Sedoc.
\newblock Gendered language in resumes and its implications for algorithmic bias in hiring.
\newblock In \emph{Proceedings of the 4th Workshop on Gender Bias in Natural Language Processing}, page 74, 2022.
\doi{10.18653/v1/2022.gebnlp-1.7}.

\bibitem{rakivnenko2024}
V.~Rakivnenko, N.~Maslej, J.~Cervi, and V.~Zhukov.
\newblock Bias in Text Embedding Models.
\newblock arXiv preprint arXiv:2406.12138, 2024.

\bibitem{rekabsaz2020}
N.~Rekabsaz and M.~Schedl.
\newblock Do Neural Ranking Models Intensify Gender Bias?
\newblock In \emph{Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval}, pages 2065--2068, 2020.
\doi{10.1145/3397271.3401280}.

\bibitem{rekabsaz2021}
N.~Rekabsaz, S.~Kopeinik, and M.~Schedl.
\newblock Societal Biases in Retrieved Contents: Measurement Framework and Adversarial Mitigation of BERT Rankers.
\newblock In \emph{Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval}, pages 306--316, 2021.
\doi{10.1145/3404835.3462949}.

\bibitem{ssa2026}
Social Security Administration.
\newblock National data on the relative frequency of given names in the population of U.S. births.
\newblock Tabulated from Social Security records as of 1 March 2026.
\url{https://www.ssa.gov/oact/babynames/limits.html}.

\bibitem{wilson2024}
K.~Wilson and A.~Caliskan.
\newblock Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval.
\newblock In \emph{Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society}, 7(1):1578--1590, 2024.
\doi{10.1609/aies.v7i1.31748}.

\end{thebibliography}
```
