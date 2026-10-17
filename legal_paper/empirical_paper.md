---
title: Gender-Associated Bias in Neural Rerankers\\ --- An Empirical Audit and De-Identification Test
author: Vasyl Rakivnenko\\ Stanford University\\ \texttt{vasyl@stanford.edu}
date:
---

```latex
% This is the source of the short empirical paper.
% Build the default paper with: python3 build.py
% Build the longer legal paper with: python3 build.py --paper legal
% Every reported result uses a named placeholder resolved through numbers.json.
```

# Abstract

Automated hiring pipelines can retrieve candidate documents with an embedding model and then reorder the leading candidates with a neural reranker. Bias in embedding models is well documented, but rerankers are a different class of model and have received little counterfactual auditing. We present, to our knowledge, the first counterfactual audit of current rerankers: {{N_MODELS}} commercial and open-weight models, each scoring {{N_PAIRS}} document pairs. Within each pair, qualifications are unchanged and only the selected name and pronouns differ. In {{N_CONTRAST_CI_POS}} of the {{N_MODELS}} models, the male-associated document is favored more often for male-typed occupations than for female-typed occupations; the two exceptions are recent models whose difference is not distinguishable from zero. In synthetic pools of {{SLATE_SIZE}} equally qualified candidates, one retrieval-and-reranking pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE}} of top-{{SHORTLIST_SIZE}} positions to male-associated documents; behind the same retriever, the audited pointwise rerankers range from {{PIPELINE_SWEEP_MIN}} to {{PIPELINE_SWEEP_MAX}}, against {{SHORTLIST_PCT_MALE_SEARCH}} for retrieval alone. Results vary materially across models, occupations, names, templates, and query wordings. We then test a simple input transform that replaces names with one placeholder and gendered pronouns with one uniform token before ranking. The transform makes each counterfactual pair textually identical and therefore removes score differences attributable to those markers by construction. On a near-ceiling synthetic occupation-matching check, the transform did not reduce top-rank accuracy ({{UTILITY_CORRECT_BEFORE}} of {{UTILITY_N_EVENTS}} correct before, {{UTILITY_CORRECT_AFTER}} after). These findings establish sensitivity to the tested names and pronouns, rather than discrimination in deployed hiring or a complete solution to demographic bias.

# Introduction

Modern search systems commonly retrieve a candidate set with a fast first-stage method, such as lexical matching or a dense embedding model [@karpukhin2020], and then use a more expensive model to rerank that set [@nogueira2019]. The same architecture can be used for job and resume matching. A candidate must therefore pass two learned ordering stages before appearing near the top of a recruiter's list.

Prior work on social bias spans several other model types. Word and text embeddings encode social stereotypes [@caliskan2017; @rakivnenko2024]. Large language models that grade résumés directly show gender and racial effects whose direction differs across models [@an2025]. Embedding-based résumé retrieval favors some demographic groups when only the name changes [@wilson2024]. Rerankers are a different type of model. Rather than comparing precomputed vectors, a reranker typically reads the query and a candidate document together and outputs a relevance score, and in a retrieval pipeline it sits on top of the embedding stage, reordering the candidates that retrieval returns. Findings about embedding models therefore cannot be assumed to carry over to rerankers. Work on rerankers themselves is sparse. A recent comparison of six rerankers measures group exposure on a general search benchmark, without varying candidate names or pronouns and outside hiring [@samuel2026]. Closest to our setting, early BERT rankers were shown to return more male-skewed passages for gender-neutral queries [@rekabsaz2020; @rekabsaz2021]. That work measures the gender content of search results rather than how a reranker treats the same candidate under different names and pronouns, and it predates the commercial rerankers sold today.

We ask three narrow empirical questions. First, do current rerankers score otherwise identical professional documents differently when selected names and pronouns change? Second, how do such differences affect a two-stage shortlist? Third, which simple de-identification rules remove the measured difference without reducing performance on a basic relevance check?

Our contribution is, to our knowledge, the first counterfactual audit of current rerankers, in three parts. First, {{N_CONTRAST_CI_POS}} of the {{N_MODELS}} audited rerankers favor the male-associated version more often on male-typed than on female-typed occupations, and newer releases are not less affected. Second, in a retrieval-and-reranking simulation, the reranker can override the shortlist that retrieval produces. Third, on four locally runnable rerankers, replacing names and pronouns together removes the measured difference, while replacing names alone does not. The audit is dated and reproducible, and it makes no claim about any employer's deployed system. The design isolates sensitivity to the changed text. It does not isolate gender from every other association carried by a name, and it does not measure actual hiring decisions.

# Experimental design

## Counterfactual documents

Each observation is a pair of short professional documents. The two versions describe the same occupation, experience, skills, and accomplishments. One version uses a male-associated given name and masculine pronouns; the other uses a female-associated given name and feminine pronouns. For a reranker that scores each document independently, any within-pair score difference is therefore caused by the changed name and pronoun tokens. The API rerankers received both versions of a pair in the same request, always with the male-associated version first, and vendors do not guarantee that a document's score is independent of its position in a request. We therefore re-sent the original requests for a category-balanced subset of {{ORDER_N_PAIRS}} pairs to each of the {{ORDER_N_API}} API rerankers in three layouts: unchanged, with each pair reversed, and fully shuffled, together with an identical repeat that measures run-to-run noise. Results are reported in Section 3. Because names can also signal age, ethnicity, nationality, and individual identity, the design cannot attribute the whole difference to gender alone.

We combine {{N_TEMPLATES}} document templates, {{N_QUERIES}} query phrasings, and {{N_NAME_PAIRS}} pairs of common American given names. Names come from national birth-name records [@ssa2026]. The complete templates, queries, and names appear in Appendix A.

We initially score {{N_OCC_SCORED}} occupations. We label occupations with the share of women employed in the corresponding federal labor-force category [@bls2025]. An occupation is male-typed if women hold under 30% of its jobs, female-typed if they hold over 70%, and evenly split if they hold 40% to 60%; the bands in between are excluded as ambiguous. We retain {{N_OCCUPATIONS}} occupations with a clear label: {{N_OCC_MALE}} male-typed, {{N_OCC_FEMALE}} female-typed, and {{N_OCC_NEUTRAL}} close to evenly split. The remaining {{N_OCC_DROPPED}} lack a usable match, have a suppressed estimate, or fall in an intentionally ambiguous range. The released repository records every inclusion and exclusion, its federal category, and the exclusion reason.

## Rerankers and analysis

We evaluate {{N_MODELS}} rerankers from commercial APIs and open-weight releases. Scores for {{N_SCORED_DATE_1}} models were collected on {{SCORED_DATE_1}}. The remaining {{LATER_N}} were added from {{LATER_FIRST}} to {{LATER_LAST}} with the same documents, queries, and request layout: commercial models through their vendors' APIs or the OpenRouter routing service, and open-weight models from their public weights, at pinned revisions, on a rented NVIDIA RTX 4090. Each open-weight model scored on the rented GPU passed a short occupation-matching check before its full run. A further endpoint, {{EXCLUDED_DUPLICATE_MODEL}}, returned scores identical to {{EXCLUDED_DUPLICATE_OF}} on every row; we treat the two as one model and exclude the duplicate from every reported figure. Appendix B lists each model's identifier, access route, score range, overall tie rate, and collection date, with its results.

For each model and occupation group, we report whether the male-associated version scores higher, the versions tie, or the female-associated version scores higher. When a single directional share is needed, a tie contributes one half to each side. Ties largely reflect score resolution rather than indifference: several models return coarsely rounded scores, and one open-weight reranker, run in bfloat16, emits only {{JINA_N_DISTINCT_SCORES}} distinct values across {{JINA_N_SCORES}} scored documents. Confidence intervals are percentile intervals from a cluster bootstrap that resamples occupations ({{BOOTSTRAP_B}} replications, seed {{SEED}}), because comparisons within an occupation are not independent. They describe uncertainty over occupations for this benchmark, holding the names, templates, and query phrasings fixed. As a sensitivity check, we also resample the {{N_BLS_CATEGORIES}} federal labor-force categories instead of the {{N_OCCUPATIONS}} occupations, so that occupations sharing a category move together, and we resample occupations and name pairs jointly; the interval for the contrast between male-typed and female-typed occupations stays above zero for {{N_EXCL_ZERO_BLS}} of the {{N_MODELS}} models under the first check and {{N_EXCL_ZERO_TWOWAY}} under the second. Scores from different models have different scales, so we compare directions and within-model gaps rather than raw magnitudes across models.

## Two-stage shortlist simulation

We form synthetic pools of {{SLATE_SIZE}} equally qualified documents, split evenly between male- and female-associated versions. The retriever {{RETRIEVER_MODEL}} selects the top {{TOPK}} candidates. The reranker {{RERANKER_MODEL}} then produces a top-{{SHORTLIST_SIZE}} shortlist. We also evaluate the retriever and reranker separately and repeat the pipeline with every audited reranker behind the same retriever. The simulation reuses the reranker scores from the single-stage audit rather than rescoring each pool. This is exact for a reranker that scores documents independently; for the API rerankers, which scored each document inside a larger per-query request, it assumes the scores do not depend on the other documents in the request, which the order test in Section 3 supports for {{ORDER_N_UNMOVED}} of the {{ORDER_N_API}} API models. Exact score ties at a selection cutoff are broken at random, and each of the {{N_SLATES}} pools is averaged over {{TIEBREAK_REPEATS}} such draws. Because candidate qualifications are identical by construction, departures from an even shortlist reflect ordering effects from the changed markers.

## De-identification

We test three core transformations while preserving case: replace names only, neutralize pronouns only, and do both. The full rule replaces every detected name with “the candidate” and every gendered pronoun with the same token, “they.” Appendix E adds four comparison conditions: lowercasing alone, a control that removes no marker; two naive grammatical rewrites that also replace names and map “she/he” to “they,” “him” to “them,” “his” to “their,” and “her” to either “their” or “them”; and a following-word rule, otherwise identical, that maps “her” to “their” or “them” depending on the next word. This deliberately simple normalization is applied before scoring. The two counterfactual versions then become the same string, so any deterministic text-only embedding model or reranker must assign them the same representation or score. This is a symmetry guarantee, not an empirical discovery. Because rules that replace names map all ten name pairs to the same placeholder, those conditions contain only {{N_DISTINCT_NAME_REPLACED}} distinct comparisons, each repeated across the {{N_PAIRS}} rows; shares are unaffected, but the name dimension is gone.

We run the transformations on four open-weight rerankers. On the worked reranker we also test relevance on a synthetic occupation-matching task: in each of {{UTILITY_N_EVENTS}} ranking events, the model ranks a pool of {{UTILITY_POOL_SIZE}} documents, one per retained occupation, and success means placing the document that names the queried occupation first. Every document states its occupation, and the transforms never touch that term, so the task is near ceiling by design. It checks that the transform does not destroy the occupation signal; it cannot measure retained ranking quality, and it does not test whether a system selects the best-qualified person within an occupation.

# Results

## Reranker scores track occupational gender composition

In {{N_CONTRAST_CI_POS}} of the {{N_MODELS}} audited models, the male-associated document is favored more often for male-typed occupations than for female-typed occupations, with an occupation-clustered interval above zero. The two exceptions are recent releases: the listwise jina-reranker-v3 ({{JINA3_CONTRAST}} points, interval {{JINA3_CI_LO}} to {{JINA3_CI_HI}}) and gte-reranker-modernbert-base ({{GTE_CONTRAST}} points, interval {{GTE_CI_LO}} to {{GTE_CI_HI}}). Newer releases are not less affected: the largest differences come from two of the most recent models, zerank-2 ({{ZERANK_CONTRAST}} points) and Voyage rerank-3 ({{VOYAGE3_CONTRAST}} points). Across models, the median tie-aware male-favored share is {{MEDIAN_PCT_MALE_MALEJOBS}} on male-typed occupations and {{MEDIAN_PCT_MALE_FEMALEJOBS}} on female-typed occupations; the two medians are taken separately and need not come from the same model. Within a single model, the median gap between the two categories is {{SPREAD_MEDIAN}} percentage points (range {{SPREAD_MIN}} to {{SPREAD_MAX}}). The size varies widely across models.

Evenly split occupations reveal substantial model variation. The median male-favored share is {{MEDIAN_PCT_MALE_NEUTRAL}}, but {{N_MODELS_NEUTRAL_LEAN_MALE}} models lie above an even split and {{N_MODELS_NEUTRAL_LEAN_FEMALE}} lie below it. For {{N_NEUTRAL_INDETERMINATE}} models, the occupation-clustered confidence interval includes an even split. A deployer therefore cannot infer either the size or direction of a model's behavior without testing that model.

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_single_stage.pdf}
\caption{\textbf{Reranker preferences across female- and male-typed
occupations.} Each row shows one reranker. Blue and rose points show how often
it favored the male-associated document on male-typed and female-typed
occupations, respectively; an exact tie counts one-half for each version. Rows
run from the largest change at the top to the smallest at the bottom. In all but
the bottom rows, the blue point lies to the right of the rose one. Exact values and
occupation-clustered intervals are in Appendix B.}
\label{fig:single}
\end{figure}
```

The result is not invariant to design choices. Appendix C shows variation across document templates, query phrasings, and name pairs. Variation across names is especially large. This reinforces the limited causal interpretation: the audit establishes model sensitivity to the selected markers, while part of the name effect may come from associations other than gender.

## Request order and reproducibility

Request order does not explain the main result. For {{ORDER_N_UNMOVED}} of the {{ORDER_N_API}} API rerankers, neither reversing each pair nor shuffling the request moves the male-typed versus female-typed contrast beyond run-to-run noise: every 95% interval includes zero. For Voyage rerank-2, reversing each pair raises the contrast by {{ORDER_R2_CHANGE}} points (interval {{ORDER_R2_LO}} to {{ORDER_R2_HI}}), small against its full-audit contrast of {{ORDER_R2_BASE}} points. Listwise rerankers, which read all candidates in one pass, behave differently. jina-reranker-v3 returns identical scores for an identical request, but when each pair is reversed it keeps the same winner in only {{JINA3_REV_SAME_WINNER}} of pairs, and its male-favored share moves from {{JINA3_PCT_MALE_SAME}} to {{JINA3_PCT_MALE_REV}}. Its overall level therefore reflects document position rather than the changed markers, while its contrast stays small in every layout.

Open-weight scores reproduce across hardware. The same bge-reranker-base revision scored on an NVIDIA RTX 4090 and on an Apple M4 Pro picks the same winner in {{XHW_SAME_WINNER}} of pairs, with scores differing by at most {{XHW_MAX_DIFF}}. A self-hosted copy of IBM's Granite reranker agrees with the vendor-hosted API in {{SELF_GRANITE_SAME_WINNER}} of pairs. The Qwen3-Reranker-8B served through OpenRouter does not reproduce its public weights: the two agree in only {{SELF_QWEN_SAME_WINNER}} of pairs, and the contrast is {{SELF_QWEN_CONTRAST_API}} points through OpenRouter against {{SELF_QWEN_CONTRAST_SELF}} points from the public weights. We cannot tell whether serving precision or prompt formatting causes the difference, but an audit of a hosted open model measures the hosted version, not the published weights.

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

Appendix D repeats the simulation with each of the {{N_SWEEP}} pointwise rerankers behind the same retriever; the listwise jina-reranker-v3 is left out because its scores depend on the other documents in a request, so scores from the audit's larger requests cannot be replayed in a smaller pool. The resulting shortlist share ranges from {{PIPELINE_SWEEP_MIN}} to {{PIPELINE_SWEEP_MAX}}, and {{N_PIPELINE_BELOW_RETRIEVER}} rerankers produce a shortlist with a lower male-associated share than retrieval alone ({{SHORTLIST_PCT_MALE_SEARCH}}). The worked example sits near the top of that range rather than being typical. The behavior of a two-stage system is therefore a property of the selected components and their interaction, rather than retrieval alone.

## Full marker removal eliminates the measured pathway

Removing only names does not remove the difference. On the worked reranker, the male-associated version wins {{DEID_NAMESONLY_PCT_MALE_MALEJOBS}} of comparisons on male-typed occupations after names are removed, compared with {{DEID_ORIGINAL_PCT_MALE_MALEJOBS}} before. The mean absolute gap is essentially unchanged ({{DEID_ORIGINAL_MEAN_GAP}} to {{DEID_NAMESONLY_MEAN_GAP}}; paired occupation-clustered 95% interval on the difference [{{DEID_NAMESONLY_GAPDIFF_LO}}, {{DEID_NAMESONLY_GAPDIFF_HI}}]). Neutralizing pronouns while retaining names reduces the mean absolute gap to {{DEID_PRONOUNSONLY_MEAN_GAP}} (interval on the difference [{{DEID_PRONOUNSONLY_GAPDIFF_LO}}, {{DEID_PRONOUNSONLY_GAPDIFF_HI}}]). Partial removal is not consistent across the four rerankers in Appendix E: removing names alone leaves the mean absolute gap unchanged or higher on two models and reduces it on the other two, and on one model it lowers the male-favored share where the worked reranker's rises. Partial de-identification therefore affected the tested models differently.

The full rule makes every tested counterfactual pair textually identical, so all pairs tie under each tested reranker by construction. Identical strings are scored once and the score is reused, so this tie rate is not an observation; it holds for any reranker whose scores are deterministic. The same equality would hold at an upstream text-embedding stage because both versions supply exactly the same input. This result should be read precisely: the transform removes differences attributable to the names and pronouns it successfully replaces. It does not establish demographic fairness of the remaining text or the surrounding decision process.

On the worked reranker, synthetic occupation-matching top-rank accuracy is {{UTILITY_CORRECT_BEFORE}} of {{UTILITY_N_EVENTS}} ({{UTILITY_TOP1_BEFORE}}) before the full transform and {{UTILITY_CORRECT_AFTER}} of {{UTILITY_N_EVENTS}} ({{UTILITY_TOP1_AFTER}}) after it: {{UTILITY_IMPROVED}} ranking events improved, {{UTILITY_WORSENED}} worsened, and accuracy changed by {{UTILITY_DELTA_PTS}} percentage points. Top-three accuracy is {{UTILITY_TOP3}} in both conditions. The ranking events repeat across templates and query phrasings within each occupation, so we report these counts descriptively rather than as a test of equivalence. Because the task is near ceiling, they show only that the transform leaves the occupation signal intact on this check. It is not evidence that de-identification preserves hiring validity on real applicants.

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_deid.pdf}
\caption{\textbf{Pairwise balance under de-identification rules on the worked
reranker.} Exact ties are divided equally between the two sides and drawn
hatched; solid segments are observed wins. Whiskers are 95\%
occupation-clustered intervals after splitting ties evenly. The final two bars
are 100\% exact ties, and the naive her$\to$their rewrite is 75\% exact ties,
because those rules make the paired texts identical; ties are not wins for
either document. Conditions are defined in Section~2.}
\label{fig:deid}
\end{figure}
```

# Implications and future work

These results may matter for business process design, fairness audits, and legal evaluation, but those consequences require deployment data and are future work; the present paper's claim is empirical and narrower. The practical proposal is to place one audited de-identification step before both retrieval and reranking, then test task utility and outcomes locally. Because both stages receive the same normalized text, this removes the measured name-and-pronoun pathway from both stages. It may also reduce one source of repeated exclusion when many employers reuse the same ranking components, connecting our study to the “algorithmic monoculture” described by Bommasani and colleagues [@bommasani2026]. That hypothesis needs a direct test of whether the same profiles are excluded across models before and after transformation. Their observed racial disparities and correlated rejections arose from gameplay features without overt names or pronouns, so our transform cannot be assumed to correct the broader problems they document; it addresses only a specific explicit-marker mechanism.

# Limitations

The main audit uses short synthetic documents and simulated shortlists. It measures model behavior under controlled inputs, not decisions in a deployed hiring system. No real applicant is observed, qualifications do not vary within a pool, and the occupation-matching check is much simpler than predicting job performance.

Names carry several social signals at once. Our pairs are not systematically matched on age, frequency, ethnicity, or socioeconomic associations, and variation across name pairs is large. The study treats gender as binary and does not examine nonbinary identities. It also does not audit race or intersectional effects.

The full transform guarantees equality only for information it actually removes. Real resumes may signal gender through schools, affiliations, activities, job histories, and writing patterns [@parasurama2022full]. Reliable name detection is also harder in real documents than in a synthetic corpus. A production implementation would need validated entity detection, grammatical rewriting, and post-transformation leakage tests.

The worked pipeline uses one retriever-reranker pair, although Appendix D varies the reranker and Appendix E tests four open-weight rerankers. Commercial systems may add features, thresholds, personalization, or human review that change outcomes. The results are a dated snapshot of model versions queried on {{SCORED_DATE_1}} and from {{LATER_FIRST}} to {{LATER_LAST}}.

A companion experiment sampled {{N_REAL_RESUMES}} distinct documents from a public corpus of {{N_REAL_CORPUS}} publicly posted résumé examples scraped from a career website (released under CC0), with no observed hiring decisions or applicant outcomes. The publisher had already removed most names: across the corpus, {{N_COMPANION_INJECTED}} of the {{N_REAL_CORPUS}} counterfactual pairs had to be built by inserting a name, and among the {{N_COMPANION_COMPARISONS}} scored comparisons per model, {{N_COMPANION_INJECTED_SCORED}} were built by inserting a name rather than swapping an existing marker. Its directions and magnitudes differ from the synthetic results (Appendix F); the worked reranker, for example, favors the male-associated version in {{COMPANION_BGE_PCT_MALE}} of comparisons, against {{SYNTH_BGE_PCT_MALE}} in the synthetic audit. Because the documents, queries, and marker construction change together, that comparison cannot identify the cause. We retain it as a warning against generalizing the synthetic effect size to applicant résumés in deployment. Its results predate the released version of its script, and its corpus must be rebuilt from the repository's provenance notes to reproduce it.

# Conclusion

Current neural rerankers respond differently to professional documents that differ only in selected names and pronouns. In {{N_CONTRAST_CI_POS}} of {{N_MODELS}} models, the direction of that response changes with occupational gender composition, while the baseline direction varies by product, and the size of the effect varies within vendors and model families and does not shrink in newer releases. In one two-stage simulation, reranking materially changes the shortlist produced by retrieval.

A simple normalization of names and gendered pronouns eliminates this particular pairwise difference because it makes the compared inputs identical. No loss was observed on a near-ceiling occupation-matching check, but real hiring utility and residual demographic leakage remain untested. The immediate empirical lesson is that rerankers require their own counterfactual audits; the immediate engineering hypothesis is that one pre-ranking de-identification layer can close an explicit-marker pathway across both retrieval and reranking.

### Data and code availability

The generated documents, banked model scores, analysis scripts, pipeline simulation, and de-identification code are available in the \href{{{REPO_URL}}}{project repository}. Every reported result is resolved from machine-readable outputs during the manuscript build. The release includes the pair-level scores for every de-identification condition, the per-event occupation-matching outcomes, and a manifest pinning each local model's exact revision, numeric precision, and library versions; this manuscript corresponds to the repository tag `arxiv-v1`. Rebuilding the tables and figures needs no model inference; rerunning inference needs the local models, and the API rerankers' scores cannot be regenerated once vendors update their models. Occupation titles and SOC codes are derived from O\*NET 30.3 (National Center for O\*NET Development for the U.S. Department of Labor, Employment and Training Administration), used under CC BY 4.0 [@onet2026]. The companion corpus is released under CC0 1.0, and the BLS and SSA inputs are U.S. Government works. Full provenance is in `data/PROVENANCE.md`.

```latex
\appendix
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{table}{0}
```

# Audit inputs

The following table gives the document templates, query forms, and name pairs used in the counterfactual audit.

```latex
{{TABLE_DOCUMENTS}}
```

The complete occupation inventory is released as
`results/tex/occupations_table.tex` rather than reproduced here. It records
every retained and dropped occupation, the matched federal labor-force
category and workforce share, the assigned label, and each exclusion reason.

# Models and single-stage results

Tables use the short labels in Table~\ref{tab:models}, which maps each label to
the model's full identifier, vendor and access route, and records the number of
comparisons, score range, overall tie rate, and scoring date.

```latex
\begin{table}[H]
\centering
\caption{Model labels, identifiers, and provenance.}
\label{tab:models}
\resizebox{\textwidth}{!}{%
{{TABLE_MODELS}}%
}
\end{table}
```

```latex
\newcommand{\SingleStageCaption}{Counterfactual outcomes by reranker and occupation
category. M and F denote the male- and female-associated documents. The final
column counts each tie as half for each version and reports occupation-clustered
intervals.}
{\small
{{TABLE_SINGLE_STAGE}}
}

\begin{table}[H]
\centering
\caption{Occupation-level directional consistency by model and category. Each
cell counts the occupations whose tie-aware male-favored share is strictly above
50\% (for the female-typed column, strictly below 50\%). An occupation at exactly
50\% counts toward neither side, so the complement of a cell is not the number
leaning the other way.}
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
\caption{Male-associated share of shortlist positions for each pointwise reranker
(the listwise jina-reranker-v3 is excluded; see Section 3) behind the same retriever and the same pools of equally qualified documents. The last
column is the combined pipeline's mean rank of female-associated documents minus
that of male-associated documents in the 20-document pool; a positive value
means female-associated documents rank lower.}
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
column is synthetic occupation-matching top-rank accuracy; a dash means that
check was not run for the condition.}
\label{tab:deid}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{De-identification conditions across four locally run rerankers.}
\label{tab:deidall}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID_ALLMODELS}}%
}
{{TABLE_DEID_ALLMODELS_NOTES}}
\end{table}
```

# Companion experiment

```latex
\begin{table}[H]
\centering
\caption{Companion experiment on publicly posted résumé examples: tie-aware
share (\%) of comparisons in which the male-associated version scores higher,
overall and by occupation group, and the share of comparisons in which the two
versions land on opposite sides of the top-three cutoff. Groups follow the
corpus's own job categories, mapped to the same labels where possible; the last
row gives each group's denominator. These documents, queries, and marker
constructions differ from the synthetic audit, so differences between the two
cannot be attributed to any single cause.}
\label{tab:companion}
\resizebox{\textwidth}{!}{%
{{TABLE_COMPANION}}%
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

\bibitem{karpukhin2020}
V.~Karpukhin, B.~O\u{g}uz, S.~Min, P.~Lewis, L.~Wu, S.~Edunov, D.~Chen, and W.~Yih.
\newblock Dense Passage Retrieval for Open-Domain Question Answering.
\newblock In \emph{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, pages 6769--6781, 2020.
\doi{10.18653/v1/2020.emnlp-main.550}.

\bibitem{nogueira2019}
R.~Nogueira and K.~Cho.
\newblock Passage Re-ranking with BERT.
\newblock arXiv preprint arXiv:1901.04085, 2019.

\bibitem{parasurama2022full}
P.~Parasurama and J.~Sedoc.
\newblock Degendering Resumes for Fair Algorithmic Resume Screening.
\newblock arXiv preprint arXiv:2112.08910, 2022.

\bibitem{onet2026}
National Center for O*NET Development.
\newblock O*NET 30.3 Database.
\newblock O*NET Resource Center, U.S. Department of Labor, Employment and Training Administration. Retrieved 22 August 2026. Used under CC BY 4.0.
\url{https://www.onetcenter.org/database.html}.

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

\bibitem{samuel2026}
S.~Samuel, B.~Van~Durme, and E.~Yang.
\newblock Does Reasoning Make Search More Fair? Comparing Fairness in Reasoning and Non-Reasoning Rerankers.
\newblock arXiv preprint arXiv:2603.10332, 2026.

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
