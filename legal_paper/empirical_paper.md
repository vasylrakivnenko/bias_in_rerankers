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

Automated hiring pipelines can retrieve candidate documents with an embedding model and then reorder the leading candidates with a neural reranker. Bias in embedding models is well documented, but rerankers are a different class of model, and few studies have tested how they respond to a candidate's name and pronouns. We audit {{N_MODELS}} current commercial and open-weight rerankers with controlled name-and-pronoun comparisons; to our knowledge, no earlier bias study of rerankers covers this many models or includes the commercial reranking services sold today. Each model scores the same {{N_PAIRS}} document pairs, and within each pair, qualifications are unchanged and only the selected name and pronouns differ. In {{N_CONTRAST_CI_POS}} of the {{N_MODELS}} models, the male-associated document is favored more often for male-typed occupations than for female-typed occupations, and in each of these the difference is larger than all {{SG_MATCHED_DRAWS}} placebo differences built from random pairs of same-gender names; of the two recent exceptions, one shows no difference and the other a reverse pattern that does not survive adjustment for multiple comparisons. When both names are replaced by one placeholder, so that the two documents differ only in their pronouns, the pattern holds in {{PRON_N_ABOVE}} of {{PRON_N_MODELS}} re-scored models. On whole real résumés, {{WR_N_PAT_KEEP}} of the {{WR_N_PATTERN}} tested rerankers that show the pattern keep it, all but one at a smaller size, and {{WR_N_PAT_REVERSE}} reverse it, so how much of it carries over depends on the model. In synthetic pools of {{SLATE_SIZE}} candidates whose documents differ only in name and pronouns, one retrieval-and-reranking pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE}} of top-{{SHORTLIST_SIZE}} positions to male-associated documents; behind the same retriever, the audited pointwise rerankers range from {{PIPELINE_SWEEP_MIN}} to {{PIPELINE_SWEEP_MAX}} (median {{PIPELINE_SWEEP_MEDIAN}}), against {{SHORTLIST_PCT_MALE_SEARCH}} for retrieval alone. Results vary materially across models, occupations, names, templates, and query wordings. We then test a simple input transform that replaces names with one placeholder and gendered pronouns with one uniform token before ranking. The transform makes each counterfactual pair textually identical and therefore removes score differences attributable to those markers by construction. On a near-ceiling synthetic occupation-matching check, the transform did not reduce top-rank accuracy ({{UTILITY_CORRECT_BEFORE}} of {{UTILITY_N_EVENTS}} correct before, {{UTILITY_CORRECT_AFTER}} after). These findings establish sensitivity to the tested names and pronouns, rather than discrimination in deployed hiring or a complete solution to demographic bias.

# Introduction

Modern search systems commonly retrieve a candidate set with a fast first-stage method, such as lexical matching or a dense embedding model [@karpukhin2020], and then use a more expensive model to rerank that set [@nogueira2019]. The same architecture can be used for job and resume matching. A candidate must therefore pass two learned ordering stages before appearing near the top of a recruiter's list.

Prior work on social bias spans several other model types. Word and text embeddings encode social stereotypes [@caliskan2017; @rakivnenko2024]. Large language models that grade résumés directly show gender and racial effects whose direction differs across models [@an2025]. Embedding-based résumé retrieval favors some demographic groups when only the name changes [@wilson2024]. A hiring study of four commercial embedding models finds that swapping a résumé's first name changes which candidates reach the top of the ranking, but that swapping between two names of the same group, or adding typographical errors, does so as well [@seshadri2025]. Rerankers are a different type of model. Rather than comparing precomputed vectors, a reranker typically reads the query and a candidate document together and outputs a relevance score, and in a retrieval pipeline it sits on top of the embedding stage, reordering the candidates that retrieval returns. Findings about embedding models therefore cannot be assumed to carry over to rerankers. Work on rerankers themselves is less extensive and mostly concerns general search. Counterfactual inputs have probed rerankers for lexical sensitivity: rewriting passages to remove keyword overlap with the query changes the relevance judgments of LLM-based rerankers [@mao2025]. LLMs prompted to rank documents have been evaluated for gender and geographic fairness [@wang2024], and six recent rerankers have been compared on group exposure [@samuel2026], both on TREC Fair Ranking data and with the documents left unchanged. BERT rankers were shown to return more male-skewed passages for gender-neutral queries [@rekabsaz2020; @rekabsaz2021], and later work reduced this skew through bias-aware query revision, bias-aware sampling of training documents, or a neutrality penalty during training [@bigdeli2021; @bigdeli2022; @zerveas2022]. A related dataset pairs gender-neutral queries with male, female, and neutral versions of each document that differ only in gendered words [@krieg2023]. Closest to our setting, Kumar and colleagues swapped a placeholder name, a title, and pronouns in candidate biographies, scored both versions with a cross-encoder they had trained, and used the pairs to find the job-advertisement words behind the score difference [@kumar2023]. A follow-up reuses that counterfactual, with the candidate as the query and job advertisements as the ranked documents, for two such cross-encoders [@kumar2025]. None of these studies tests a commercial reranking service or compares released rerankers on the same counterfactual candidate documents.

We ask three narrow empirical questions. First, do current rerankers score otherwise identical professional documents differently when selected names and pronouns change? Second, how do such differences affect a two-stage shortlist? Third, which simple de-identification rules remove the measured difference without reducing performance on a basic relevance check?

Our contribution is an audit of {{N_MODELS}} current rerankers, commercial and open-weight, in which every model scores the same documents with only the name and pronouns changed. To our knowledge, no earlier bias study of rerankers covers this many models or includes commercial reranking services. Earlier counterfactual tests of candidate text used one placeholder name per gender and one or two cross-encoders that the authors trained [@kumar2023; @kumar2025]; this audit uses {{N_NAME_PAIRS}} pairs of common given names and rerankers as their developers release them. The audit has three parts. First, {{N_CONTRAST_CI_POS}} of the {{N_MODELS}} audited rerankers favor the male-associated version more often on male-typed than on female-typed occupations, including some of the most recently released models. The pattern holds in {{PRON_N_ABOVE}} of {{PRON_N_MODELS}} re-scored models when the two documents differ only in their pronouns, so it does not depend on the names. On whole real résumés it persists in {{WR_N_PAT_KEEP}} of the {{WR_N_PATTERN}} tested rerankers that show it, all but one at a smaller size, and reverses in {{WR_N_PAT_REVERSE}}. Second, in a retrieval-and-reranking simulation, the reranker can override the shortlist that retrieval produces. Third, on four locally runnable rerankers, replacing names and pronouns together removes the measured difference, while replacing names alone does not. The audit is dated and reproducible, and it makes no claim about any employer's deployed system. The design isolates sensitivity to the changed text. Names carry associations beyond gender, but the pronoun-only comparison reproduces the pattern without them. The audit does not measure actual hiring decisions.

# Experimental design

## Counterfactual documents

Each observation is a pair of short professional documents. The two versions describe the same occupation, experience, skills, and accomplishments. One version uses a male-associated given name and masculine pronouns; the other uses a female-associated given name and feminine pronouns. Like correspondence studies of hiring, which randomly assign different names to comparable job applications [@bertrand2004], and like template-based audits of language systems in which only a name or a gendered word changes [@kiritchenko2018], the design varies the name, and here also the pronouns, while holding the rest of the document fixed. For a deterministic reranker that scores each document independently, any within-pair score difference is therefore caused by the changed name and pronoun tokens. Each API request held all {{DOCS_PER_REQUEST}} documents for one query, that is, one occupation in one query phrasing ({{N_TEMPLATES}} templates $\times$ {{N_NAME_PAIRS}} name pairs $\times$ 2 versions), ordered template by template with the male-associated version of each pair first. IBM's Granite endpoint accepted at most {{GRANITE_MAX_TEXTS}} documents per request, so its requests were split; the listwise jina-reranker-v3, run from its public weights, received the same {{DOCS_PER_REQUEST}}-document lists.

Vendors do not guarantee that a document's score is independent of its position in a request. We therefore re-sent the original requests for a category-balanced subset of {{ORDER_N_PAIRS}} pairs ({{ORDER_PER_CATEGORY}} occupations per category, drawn with a fixed seed) to {{ORDER_N_API}} of the {{N_API_TOTAL}} rerankers we accessed through an API (all except IBM's Granite reranker, whose endpoint could not be re-queried) in three layouts: unchanged; with each pair reversed, so that the female-associated version comes first; and with all of a request's documents in a seeded random order. An identical repeat of the unchanged layout measures run-to-run variability. Results are reported in Section 3.

Because names can also signal age, ethnicity, nationality, and individual identity, the name-and-pronoun comparison cannot attribute the whole difference to gender alone. We therefore also score a pronoun-only version of every comparison, in which both names are replaced by the same placeholder, so that the two documents differ only in their pronouns; this is the names-only de-identification rule below. We scored it on {{PRON_DATE}} for every audited model except the listwise jina-reranker-v3, whose scores depend on the other documents in a request. The API models were re-queried, the open-weight models were re-run on an Apple M4 Pro, and IBM's Granite reranker, whose endpoint could not be re-queried, was scored from its public weights. Because every model scored ten names of each gender for each occupation, template, and query phrasing, swaps between two names of the same gender are also available from the original scores; Section 3 uses them as a yardstick for the size of the gender swap and as a placebo for the contrast.

We combine {{N_TEMPLATES}} document templates, {{N_QUERIES}} query phrasings, and {{N_NAME_PAIRS}} pairs of common American given names. Names come from national birth-name records [@ssa2026]. Each pair joins one male-associated and one female-associated name; the pairing itself is arbitrary, and Section 3 checks that it does not drive the results. The complete templates, queries, and names appear in Appendix A.

The {{N_OCC_SCORED}} occupation words were assembled by hand before any labeling, from occupations chosen for this study and from the occupation list used in earlier work on embedding bias [@rakivnenko2024], which draws on the BIG-bench gender-occupation data. They are everyday job words, not a sample of the labor market, and some (for example, boss, warrior, or socialite) name a rank or role rather than an occupation; the labeling step below removes these. Statements about how many models or occupations show an effect therefore describe this benchmark, not the distribution of occupations in employment or job postings. We label occupations with the share of women employed in the corresponding federal labor-force category [@bls2025]. An occupation is male-typed if women hold under 30% of its jobs, female-typed if they hold over 70%, and evenly split if they hold 40% to 60%; the bands in between are excluded as ambiguous. We retain {{N_OCCUPATIONS}} occupations with a clear label: {{N_OCC_MALE}} male-typed, {{N_OCC_FEMALE}} female-typed, and {{N_OCC_NEUTRAL}} close to evenly split. The remaining {{N_OCC_DROPPED}} lack a usable match, have a suppressed estimate, or fall in an intentionally ambiguous range. The released repository records every inclusion and exclusion, its federal category, and the exclusion reason.

## Rerankers and analysis

We evaluate {{N_MODELS}} rerankers from commercial APIs and open-weight releases. The model set is a convenience sample, not a market sample: it comprises commercial rerankers we could query at scoring time through Voyage AI's and Google Cloud's APIs, Microsoft Azure, or the OpenRouter routing service, and open-weight rerankers published on Hugging Face by major model developers that run on a single 24 GB GPU and are licensed for research use. Scores for {{N_SCORED_DATE_1}} models were collected on {{SCORED_DATE_1}}. The remaining {{LATER_N}} were added from {{LATER_FIRST}} to {{LATER_LAST}} with the same documents, queries, and request layout: through vendors' APIs or the OpenRouter routing service, or from public weights at pinned revisions on a rented NVIDIA RTX 4090. The first {{N_SCORED_DATE_1}} models were scored on all {{N_OCC_SCORED}} occupation words and the later {{LATER_N}} only on the {{N_OCCUPATIONS}} retained ones; only one robustness check in Section 3, which moves the cut-points, uses the others. Each open-weight model scored on the rented GPU passed a short occupation-matching check before its full run. A further endpoint, {{EXCLUDED_DUPLICATE_MODEL}}, returned scores identical to {{EXCLUDED_DUPLICATE_OF}} on every row; we treat the two as one model and exclude the duplicate from every reported figure. Appendix B lists each model's identifier, access route, score range, overall tie rate, and collection date, with its results.

For each model and occupation group, we report whether the male-associated version scores higher, the versions tie, or the female-associated version scores higher. When a single directional share is needed, a tie contributes one half to each side; we call this the tie-aware share. Ties largely reflect score resolution rather than indifference: several models return coarsely rounded scores, and one open-weight reranker, run in bfloat16, emits only {{JINA_N_DISTINCT_SCORES}} distinct values across {{JINA_N_SCORES}} scored documents. Confidence intervals are percentile intervals from a cluster bootstrap that resamples occupations, because comparisons within an occupation are not independent: {{BOOTSTRAP_B}} replications for per-category shares and {{CONTRAST_B}} for the male-typed versus female-typed contrasts and the order test (seed {{SEED}}); a few robustness checks use fewer, as their tables state. They describe uncertainty over occupations for this benchmark, holding the names, templates, and query phrasings fixed; with {{N_OCC_MALE_CLUSTERS}} and {{N_OCC_FEMALE_CLUSTERS}} clusters in the two categories, the percentile intervals are approximate. As a sensitivity check, we also resample the {{N_BLS_CATEGORIES}} federal labor-force categories instead of the {{N_OCCUPATIONS}} occupations, so that occupations sharing a category move together, and we resample occupations and name pairs jointly; the interval for the contrast between male-typed and female-typed occupations stays above zero for {{N_EXCL_ZERO_BLS}} of the {{N_MODELS}} models under the first check and {{N_EXCL_ZERO_TWOWAY}} under the second. Because these intervals are approximate and unadjusted for the number of models, we also test each contrast with a two-sided permutation test that randomly relabels which occupations are male-typed and which are female-typed ({{PERM_N}} relabelings), with a Holm adjustment across the {{N_MODELS}} models. Because the 30% and 70% cut-points are a choice, we also treat the share of women as a continuous variable and repeat the contrast at other cut-points. Scores from different models have different scales, so we compare directions and within-model gaps rather than raw magnitudes across models.

## Two-stage shortlist simulation

We form synthetic pools of {{SLATE_SIZE}} equally qualified documents: one document, with its occupation, template, and query fixed, in its {{SLATE_SIZE}} name-and-pronoun versions, half male- and half female-associated, so that every candidate in a pool is identical except for those markers. The retriever {{RETRIEVER_MODEL}} selects the top {{TOPK}} candidates. The reranker {{RERANKER_MODEL}} then produces a top-{{SHORTLIST_SIZE}} shortlist. We use this pair as the worked example because both are open-weight models from the same developer that can be re-run locally; Appendix C shows where the reranker falls among all audited models. We also evaluate the retriever and reranker separately and repeat the pipeline with every audited reranker behind the same retriever. The simulation reuses the reranker scores from the single-stage audit rather than rescoring each pool. This is exact for a reranker that scores documents independently; for the API rerankers, which scored each document inside a larger per-query request, it assumes the scores do not depend on the other documents in the request. We test that assumption directly by sending the retriever's actual ten documents for {{REPLAY_N_POOLS}} sampled pools to each reachable API reranker and comparing the resulting shortlists with the replayed ones (Section 3). The sample takes {{REPLAY_PER_OCC}} pools from every retained occupation, drawing their templates and query phrasings at random without repeats, and uses the same pools for every model. Exact score ties at a selection cutoff are broken at random, and each of the {{N_SLATES}} pools is averaged over {{TIEBREAK_REPEATS}} such draws. Because candidate qualifications are identical by construction, departures from an even shortlist reflect ordering effects from the changed markers. For the same reason, a consistent score difference of any size fills the shortlist exactly as a large one would, so the simulation measures the direction and consistency of a model's preference rather than its size, and it is a stress test rather than an estimate for pools of genuinely different candidates.

## De-identification

We test three rules: replace names only, neutralize pronouns only, and both. Each rule changes only the words it replaces, and it inserts the replacement in lowercase, even at the start of a sentence. The full rule replaces every detected name with “the candidate” and every gendered pronoun (he, she, him, his, her, hers, himself, herself) with the single word “they”, which avoids deciding whether a given “her” is possessive or an object; Appendix D adds four comparison conditions, including rewrites that map “her” to “their”, to “them”, or according to the next word. Under the full rule the two counterfactual versions become the same string, so any deterministic pointwise reranker, whose only inputs are the query and the document, must score them identically; this is a symmetry guarantee, not a discovery. A listwise reranker, which also reads the other candidates and their positions, is guaranteed this only for a candidate held at the same position in an otherwise unchanged list, and ties at a selection cutoff still need a tie-breaking rule that ignores the replaced markers. Rules that replace names collapse the ten name pairs into one text, leaving {{N_DISTINCT_NAME_REPLACED}} distinct comparisons repeated across the {{N_PAIRS}} rows.

We apply all rules to four open-weight rerankers, and the names-only rule, as the pronoun-only comparison, to every model we could re-score. On the worked reranker, a synthetic occupation-matching check builds one pool per document template, each holding one document for every retained occupation ({{UTILITY_POOL_SIZE}} documents), and ranks each pool for every occupation's query in each of the {{N_QUERIES}} phrasings: {{UTILITY_N_EVENTS}} ranking events in all. Success means ranking the queried occupation's document first. Because every document names its occupation and the rules never touch that word, the check is near ceiling by design: it confirms that the occupation signal survives, not that ranking quality or selection within an occupation is preserved.

# Results

## Reranker scores track occupational gender composition

In {{N_CONTRAST_CI_POS}} of the {{N_MODELS}} audited models, the male-associated document is favored more often for male-typed occupations than for female-typed occupations, with an occupation-clustered interval above zero. The two exceptions are recent releases. The listwise jina-reranker-v3 shows no difference ({{JINA3_CONTRAST}} points, interval {{JINA3_CI_LO}} to {{JINA3_CI_HI}}). gte-reranker-modernbert-base shows the reverse pattern ({{GTE_CONTRAST}} points, interval {{GTE_CI_LO}} to {{GTE_CI_HI}}), although its interval includes zero when name pairs are resampled as well ({{GTE_TWOWAY_LO}} to {{GTE_TWOWAY_HI}}). Appendix B lists every model's contrast with its intervals; these are individual 95% intervals, not adjusted for the {{N_MODELS}} comparisons. The permutation test gives the same {{PERM_N_POS_HOLM}} models a positive contrast that survives the Holm adjustment, while the reverse pattern of gte-reranker-modernbert-base does not (adjusted p = {{GTE_PERM_HOLM}}). The cut-points do not drive this result. Treating the federal share of women as a continuous variable, a model's occupation-level male-favored share falls as the share of women rises in {{CONT_N_RHO_NEG}} of the {{N_MODELS}} models (median Spearman $\rho$ {{CONT_MEDIAN_RHO}}; {{CONT_N_NEG_HOLM_SIG}} remain significant after a Holm adjustment of a {{CONT_N_PERM}}-draw permutation test), by a median {{CONT_MEDIAN_SLOPE10_ABS}} points per 10 points of women's share; the exception is again gte-reranker-modernbert-base ($\rho$ = {{CONT_GTE_RHO}}). Moving the cut-points to 25% and 75% leaves {{CUT2575_N_POS}} models positive and {{CUT2575_N_CI_POS}} intervals above zero, and for the {{CUT3565_N_MODELS}} models scored on all {{N_OCC_SCORED}} occupations, cut-points of 35% and 65%, which add {{CUT3565_N_ADDED}} occupations from the excluded bands, leave all {{CUT3565_N_POS}} positive (Table~\ref{tab:continuous}). All {{N_MODELS}} models score the same documents, so these counts describe one benchmark seen through {{N_MODELS}} models, not independent replications. Some recently released models show the largest differences, including zerank-2 ({{ZERANK_CONTRAST}} points) and Voyage rerank-3 ({{VOYAGE3_CONTRAST}} points). Within Voyage's two product lines, the only ones with three or more versions in the audit, the contrast rises with each version: {{VOYAGE1_CONTRAST}}, {{VOYAGE2_CONTRAST}}, {{VOYAGE25_CONTRAST}}, and {{VOYAGE3_CONTRAST}} points for Voyage rerank-1, rerank-2, rerank-2.5, and rerank-3, and {{VOYAGE2LITE_CONTRAST}}, {{VOYAGE25LITE_CONTRAST}}, and {{VOYAGE3LITE_CONTRAST}} points along the lite line, where part of the rise coincides with a tie rate that falls from {{VOYAGE2LITE_TIES}} to {{VOYAGE3LITE_TIES}} (Table~\ref{tab:models}). These are descriptive comparisons of three or four products; the audit does not test for a trend over release dates. Across models, the median tie-aware male-favored share is {{MEDIAN_PCT_MALE_MALEJOBS}} on male-typed occupations and {{MEDIAN_PCT_MALE_FEMALEJOBS}} on female-typed occupations; the two medians are taken separately and need not come from the same model. Within a single model, the median gap between the two categories is {{SPREAD_MEDIAN}} percentage points (range {{SPREAD_MIN}} to {{SPREAD_MAX}}). The size varies widely across models.

Names are not needed for this pattern. When both names are replaced by the same placeholder, so that the two documents differ only in their pronouns, the contrast has an occupation-clustered interval above zero for {{PRON_N_ABOVE}} of the {{PRON_N_MODELS}} re-scored models (Table~\ref{tab:pronoun}); the exception is again gte-reranker-modernbert-base ({{PRON_GTE}} points, interval {{PRON_GTE_LO}} to {{PRON_GTE_HI}}). Pronouns, which carry no ethnic or age association, are therefore enough to produce the stereotype-aligned pattern.

The pronoun-only contrast is also larger, with a median of {{PRON_MEDIAN}} points against {{PRON_ORIG_MEDIAN}} with names, but this does not show that pronouns matter more than names. Most of the gap comes from how the comparisons are grouped. With one placeholder, the ten name pairs that share an occupation, template, and query phrasing become a single comparison; with names, each pair is its own comparison, and a model's preferences among particular names turn some pairs the other way. Averaging each group's ten male-version and ten female-version scores before comparing raises the median with names to {{PRON_ORIG_NAMEAVG_MEDIAN}} (Table~\ref{tab:pronoun}). Two smaller differences remain: the pronoun-only texts were scored in a later run, for the open-weight models on another machine (see the reproducibility checks below), and the placeholder changes the surface of the text, since it is inserted in lowercase (Appendix D). Without the {{PRON_N_DRIFT_BELOW_100}} models whose re-run changed some checked winners, the medians are {{PRON_STABLE_NAMEAVG_MEDIAN}} with names averaged and {{PRON_STABLE_MEDIAN}} with pronouns only, and {{PRON_STRICT_NAMEAVG_MEDIAN}} and {{PRON_STRICT_MEDIAN}} if jina-reranker-v2-base-multilingual, whose full re-score changed {{JINA_RERUN_CHANGED}} of its winners, is also left out.

Evenly split occupations reveal substantial model variation. The median male-favored share is {{MEDIAN_PCT_MALE_NEUTRAL}}, but {{N_MODELS_NEUTRAL_LEAN_MALE}} models lie above an even split and {{N_MODELS_NEUTRAL_LEAN_FEMALE}} lie below it. For {{N_NEUTRAL_INDETERMINATE}} models, the occupation-clustered confidence interval includes an even split, and for {{N_NEUTRAL_INDETERMINATE_TWOWAY}} when name pairs are resampled as well ({{NEUTRAL_TWOWAY_B}} draws). A deployer therefore cannot infer either the size or direction of a model's behavior without testing that model. Part of any overall lean, however, reflects a model's preferences among particular names rather than their gender; the same-gender comparison below separates the two.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_single_stage.pdf}
\caption{\textbf{Reranker preferences across female- and male-typed
occupations.} Each row shows one reranker. Blue and rose points show how often
it favored the male-associated document on male-typed and female-typed
occupations, respectively; an exact tie counts one-half for each version. Rows
are ordered by the blue value minus the rose value, largest at the top. In every row but
the last, the blue point lies to the right of the rose one; in the second-to-last
the two nearly coincide. Thin lines are 95\% occupation-clustered bootstrap
intervals, drawn just above each blue point and just below each rose point.
Table~\ref{tab:contrast} lists both values for every model; the intervals of
each point are released with the data.}
\label{fig:single}
\end{figure}
```

The level is not invariant to design choices. Across document templates, query phrasings, and name pairs, the median model's male-favored share ranges over {{ROB_TEMPLATE}}, {{ROB_QUERY}}, and {{ROB_NAMEPAIR}} percentage points respectively, so variation across names is especially large; the full tables are released with the data. Template-based bias measurements are known to shift considerably even under semantically equivalent rewrites of the templates [@seshadri2022]. The direction of the contrast is steadier. It is positive in all four templates for {{WT_N_ALL}} models and in all four query phrasings for {{WQ_N_ALL}} ({{WT_CELLS_POS}} and {{WQ_CELLS_POS}} of {{WT_CELLS}} model-level cells), and comparing every male-associated name with every female-associated name, rather than only the fixed pairs, leaves it positive for {{CROSS_N_POS}} models. The audit establishes model sensitivity to the selected markers; part of the name effect on the level may still come from associations other than gender.

Swapping two names of the same gender gives a yardstick for the size of these differences and a placebo for their specificity. Within each cell of the design (one occupation, template, and query phrasing) every model scored ten male-associated and ten female-associated documents, so the {{SG_N_SAME_PAIRS}} pairs that exchange one male-associated name for another, or one female-associated name for another, hold gender fixed and change only the name. Scores are on each model's own scale, so the comparison is within models (Table~\ref{tab:samegender}). The name-and-pronoun swap moves a model's score further than a same-gender name swap in {{SG_N_RATIO_GT1}} of the {{N_MODELS}} models (median ratio of mean absolute gaps {{SG_MEDIAN_RATIO}}; interval above one for {{SG_N_RATIO_CI_GT1}}), and the median model's cross-gender gap exceeds the same-gender median of its own cell in {{SG_MEDIAN_PCT_ABOVE_CELL_MEDIAN}} of pairs. The exceptions are the listwise jina-reranker-v3 ({{SG_JINA3_RATIO}}) and Cohere rerank v4 pro ({{SG_COHERE_PRO_RATIO}}, interval {{SG_COHERE_PRO_CI_LO}} to {{SG_COHERE_PRO_CI_HI}}), whose gender swap moves its score no further than a change of first name although its contrast of {{COHERE_PRO_CONTRAST}} points is among the larger ones: its difference is consistent in direction rather than large. The same swaps serve as a placebo. Pairing the ten male-associated names at random into five same-gender pairs, the ten female-associated names likewise, and computing the male-typed minus female-typed contrast from those ten pairs exactly as for the real ones, {{SG_MATCHED_DRAWS}} times per model, never reaches the size of the observed contrast for {{SG_MATCHED_N_CONTRAST_ABOVE_ALL}} of the {{N_MODELS}} models (for gte-reranker-modernbert-base, the size of its reversed contrast); the median model's 95th placebo percentile is {{SG_MATCHED_MEDIAN_P95}} points, and the exception is again jina-reranker-v3 (p = {{SG_MATCHED_JINA3_P}}). A harder benchmark is the largest contrast that any single same-gender name pair produces on its own ({{SG_N_PLACEBO_PER_GENDER}} male–male and {{SG_N_PLACEBO_PER_GENDER}} female–female pairs); {{SG_N_CONTRAST_ABOVE_MAX_PLACEBO}} models still exceed it. The overall male-favored share does not pass the same test. Models hold strong preferences among names of the same gender, and the observed share differs from an even split by more than the matched placebo, at the 5% level, for only {{SG_MATCHED_N_LEVEL_P05}} models. The contrast between occupation groups, rather than a model's overall lean, is therefore the statistic this audit can attribute to the gender markers. For the {{SG_PRON_N}} models with pronoun-only scores, the pronoun-only swap moves the score further than a same-gender name swap in {{SG_PRON_N_ABOVE_SAME}} (median ratio {{SG_PRON_MEDIAN_OVER_SAME}}), although that comparison spans two scoring runs; without the {{PRON_N_DRIFT_BELOW_100}} models whose re-run changed some checked winners, it does so in {{SG_PRON_STABLE_N_ABOVE}} of {{SG_PRON_STABLE_N}} (median ratio {{SG_PRON_STABLE_MEDIAN}}).

## The pattern on whole real résumés

The audit's documents are short, {{SYN_WORDS_MIN}} to {{SYN_WORDS_MAX}} words, whereas a real résumé in the corpus used here has a median of {{REAL_WORDS_MEDIAN}} words (interquartile range {{REAL_WORDS_Q1}} to {{REAL_WORDS_Q3}}). A companion experiment therefore drew {{N_REAL_RESUMES}} documents from a public corpus of {{N_REAL_CORPUS}} résumé examples posted on a career website (released under CC0) and built {{N_COMPANION_COMPARISONS}} comparisons. In each, one résumé in a pool of {{COMP_POOL_SIZE}} switches between a male- and a female-associated version, and the job query is an O\*NET occupation description (Appendix E). The publisher had already removed most names, so most pairs differ by an added given name, male or female, the two equally common. Only the applicant's own name and pronouns change; those of other people a résumé mentions, such as a supervisor, stay as written. Every audited reranker that can read a whole résumé ({{WR_N_AUDIT_WHOLE}}) scored these comparisons, together with jina-reranker-v2-base-multilingual at its {{WR_JINA_WINDOW}}-token limit and Cohere rerank v5 pro and fast, released after the audit: {{WR_N_MODELS}} rerankers in all.

The pattern carries over to real résumés for most models, but how much of it survives depends on the model (Table~\ref{tab:wholeresume}). Of the {{WR_N_PATTERN}} rerankers whose synthetic contrast has an occupation-clustered interval above zero, {{WR_N_PAT_KEEP}} keep the pattern on whole résumés ({{WR_PAT_KEEP_MIN}} to {{WR_MALE_MAX}} points, each interval above zero), and for all but Voyage rerank-2.5-lite, whose contrast is about the same ({{WR_V25L_SYN}} points synthetic, {{WR_V25L_WHOLE}} whole), the contrast is smaller. The other {{WR_N_PAT_REVERSE}} reverse it, among them Voyage rerank-2, whose contrast falls from {{WR_V2_SYN}} points in the synthetic audit, one of the largest, to {{WR_V2_WHOLE}}. The two rerankers without the synthetic pattern show no clear contrast here either: jina-reranker-v3 shows none in either setting, and gte-reranker-modernbert-base, which showed the reverse, now has an interval that includes zero ({{WR_GTE_WHOLE}} points, {{WR_GTE_LO}} to {{WR_GTE_HI}}). These intervals resample résumé pools within the fixed job categories. A check against each category's share of women agrees in sign for {{WR_N_RHO_AGREE}} of the {{WR_N_DIRECTED}} rerankers whose interval excludes zero; the exception, Cohere rerank v4 fast, keeps the pattern only marginally, with an interval from {{WR_C4F_LO}} to {{WR_C4F_HI}} (Appendix E). Across rerankers, the synthetic and whole-résumé contrasts have a rank correlation of {{WR_SPEARMAN}} (95% interval {{WR_SPEARMAN_LO}} to {{WR_SPEARMAN_HI}}, resampling rerankers). The synthetic audit therefore ranks rerankers roughly as they rank on whole résumés, but it does not give the size of the contrast there, and for {{WR_N_PAT_REVERSE}} rerankers not its direction.

Length alone does not explain these differences, and its effect differs by reranker. For {{LP_N_MODELS}} of the rerankers, we also scored each résumé cut to its first {{LP_CUT_TEXT}} words after the name (Figure~\ref{fig:lengthpronoun}). The closest match to a synthetic document is a 35-word excerpt that begins with the name and a sentence of {{LP_LINE_WORDS}} words carrying {{LP_LINE_PRONOUNS}} of the applicant's pronouns; the synthetic documents carry {{SYN_PRON_MIN}} or {{SYN_PRON_MAX}}. This excerpt comes within {{LP_SHORT_REST_MIN}} to {{LP_SHORT_REST_MAX}} points of the synthetic contrast for every reranker except Voyage rerank-2, which falls {{LP_SHORT_OUTLIER_ABS}} points short with only {{LP_SHORT_RESUME_WORDS}} words of résumé text. With the name alone, as the résumé lengthens, the contrast stays level for Voyage rerank-3 ({{LP_V3_NAME_35}} points at 35 words, {{LP_V3_NAME_FULL}} whole), grows for Qwen3-Reranker-8B ({{LP_QWEN_NAME_35}} to {{LP_QWEN_NAME_FULL}}), holds to 300 words and then falls for Cohere rerank v5 pro ({{LP_C5_NAME_35}} to {{LP_C5_NAME_FULL}}), and falls below zero for mxbai-rerank-large-v2 ({{LP_MXBAI_NAME_35}} to {{LP_MXBAI_NAME_FULL}}). For Voyage rerank-2 it is small at 35 words ({{LP_V2_NAME_35}} points, interval {{LP_V2_NAME_35_LO}} to {{LP_V2_NAME_35_HI}}) and negative from 150 words on. The pronoun sentence raises the contrast by {{LP_LINE35_MIN}} to {{LP_LINE35_MAX}} points at 35 words, with an interval above zero for {{LP_N_LINE35_POS}} of the {{LP_N_MODELS}} rerankers. On whole résumés it changes the contrast by at most {{LP_LINEFULL_REST_MAX}} points, except for the two reversals, which it moves back toward zero (mxbai-rerank-large-v2 by {{LP_MXBAI_LINEFULL}} points, Voyage rerank-2 by {{LP_V2_LINEFULL}}).

## Request order and reproducibility

Measured against an identical repeat of each request, so that call-to-call variability is not mistaken for an order effect, reversing pairs or shuffling the request shifted the contrast beyond that variability in {{ORDER_N_MOVED_BEYOND_REPEAT}} of the {{ORDER_N_API}} tested API rerankers. These are individual 95% intervals that detect only aggregate shifts of about one to two points, not proof that order has no effect. The repeat itself moves Voyage rerank-2 by {{R2_REPEAT_CHANGE}} points ({{R2_REPEAT_LO}} to {{R2_REPEAT_HI}}), the only one of the {{ORDER_N_API}} repeat intervals to exclude zero, which is about what chance alone produces at the 95% level. Re-sending the original requests also reproduced the banked contrasts closely. For the {{DRIFT_FEB_N}} API models first scored in February, {{DRIFT_FEB_DAYS}} days earlier, the contrast moved by at most {{DRIFT_FEB_MAX}} points, {{DRIFT_FEB_N_EXCL}} of the {{DRIFT_FEB_N}} intervals on the change exclude zero, and the two runs pick the same winner in {{DRIFT_FEB_SAMEWIN_MIN}} to {{DRIFT_FEB_SAMEWIN_MAX}} of pairs. The listwise jina-reranker-v3 is deterministic but position-dependent: reversing each pair keeps the same winner in only {{JINA3_REV_SAME_WINNER}} of pairs, and its overall male-favored share is {{JINA3_PCT_MALE_SAME}} in the original layout, {{JINA3_PCT_MALE_REV}} with each pair reversed, and {{JINA3_SHUF_LEVEL}} with the request shuffled, while its male-typed minus female-typed contrast stays small in all three ({{JINA3_SAME_CONTRAST}}, {{JINA3_REV_CONTRAST}}, and {{JINA3_SHUF_CONTRAST}} points). Its level in Figure~\ref{fig:single} therefore reflects the request layout as much as the model, whereas its near-zero contrast does not.

Open-weight models do not always give identical scores on a second machine. Scored in full on both an NVIDIA RTX 4090 and an Apple M4 Pro, bge-reranker-base picks the same winner in {{XHW_SAME_WINNER}} of pairs. For the pronoun-only comparison we re-ran {{PRON_N_DRIFT}} other open-weight models on the M4 Pro and checked each against its saved scores on the {{DRIFT_N_PAIRS}} pairs of two sampled queries: {{DRIFT_N_100}} reproduced all {{DRIFT_N_PAIRS}} saved winners, and the other {{PRON_N_DRIFT_BELOW_100}} did not (mxbai-rerank-large-v2 {{DRIFT_MXBAI}}, Qwen3-Reranker-0.6B {{DRIFT_QWEN06}}, Qwen3-Reranker-4B {{DRIFT_QWEN4}}, llama-nemotron-rerank-1b-v2 {{DRIFT_NEMO1B}}, zerank-2 {{DRIFT_ZERANK}}). The lowest agreement comes from a model whose two versions of a pair score almost the same: on these pairs, mxbai-rerank-large-v2's saved scores for the two versions differ by {{DRIFT_MXBAI_GAP}} on average, and the second machine moved each score by {{DRIFT_MXBAI_CHANGE}}, enough to change the outcome of many pairs. The four models of the de-identification run were instead re-scored on all {{N_PAIRS}} original pairs (Table~\ref{tab:pronoun}), which shows that a check of {{DRIFT_N_PAIRS}} pairs can miss rare changes: jina-reranker-v2-base-multilingual reproduced every sampled winner but only {{JINA_RERUN_SAME_WINNER}} of all winners, a change that moved its contrast by {{JINA_RERUN_CONTRAST_CHANGE}} points. The pronoun-only results above are therefore also reported without the {{PRON_N_DRIFT_BELOW_100}} models whose re-run changed some checked winners; re-querying the API models, which stay in, moved their contrasts by at most {{API_DRIFT_MAX}} points. A self-hosted copy of IBM's Granite reranker matches its API in {{SELF_GRANITE_SAME_WINNER}} of pairs. Qwen3-Reranker-8B served through OpenRouter agrees with its public weights in {{SELF_QWEN_SAME_WINNER}} of pairs overall, but the disagreement is mostly score resolution: the hosted service returns exact ties in {{SELF_QWEN_HOSTED_TIES}} of pairs against {{SELF_QWEN_SELF_TIES}} from the weights, and where neither ties the two agree in {{SELF_QWEN_SAME_WINNER_NOTIE}}. Because ties are split evenly, the contrast still differs ({{SELF_QWEN_CONTRAST_API}} against {{SELF_QWEN_CONTRAST_SELF}} points), so an audit of a hosted open model measures the hosted version, including its score resolution. The audit's Qwen3-Reranker-8B row is the hosted version; the self-hosted scores are released separately.

## The reranker can dominate the pipeline

In the worked pipeline, the retriever alone assigns {{SHORTLIST_PCT_MALE_SEARCH}} of top-{{SHORTLIST_SIZE}} positions to male-associated documents. The reranker alone assigns {{SHORTLIST_PCT_MALE_RANKER}}, and the combined pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE}}. For occupations close to evenly split in the labor market, the combined share is {{SHORTLIST_PCT_MALE_PIPELINE_NEUTRAL}}.

The interaction depends on occupation type. On male-typed occupations, the male-associated share rises from {{SHORTLIST_PCT_MALE_SEARCH_MALEJOBS}} under retrieval to {{SHORTLIST_PCT_MALE_RANKER_MALEJOBS}} under reranking and {{SHORTLIST_PCT_MALE_PIPELINE_MALEJOBS}} in sequence. On female-typed occupations, retrieval assigns {{SHORTLIST_PCT_MALE_SEARCH_FEMJOBS}} to male-associated documents, while reranking assigns {{SHORTLIST_PCT_MALE_RANKER_FEMJOBS}} and the combined pipeline assigns {{SHORTLIST_PCT_MALE_PIPELINE_FEMJOBS}}. The two stages can therefore compound or counteract one another.

The combined share also depends on which names are in the pool. Resampling the ten name pairs instead of the occupations gives an interval of {{NAME_BOOT_CI_LO}} to {{NAME_BOOT_CI_HI}} for the combined share, and resampling both {{JOINT_BOOT_CI_LO}} to {{JOINT_BOOT_CI_HI}} ({{NAME_BOOT_B}} draws), against {{PIPE_CI_LO}} to {{PIPE_CI_HI}} over occupations alone; dropping any one pair moves the share by at most {{NAME_LOO_MAX_SHIFT}} points. The reranker also concentrates the shortlist on particular names: the three most frequent names take {{NAME_TOP3_NAMES_SHARE_PIPELINE}} of the combined pipeline's top-{{SHORTLIST_SIZE}} slots, against 15% if the twenty names were drawn evenly, and {{NAME_MAX_NAME_PIPELINE}} alone takes {{NAME_MAX_SHARE_PIPELINE}}, whereas the retriever spreads its top ten almost evenly: an effective {{NAME_EFFECTIVE_N_RETRIEVER}} of the twenty names, against {{NAME_EFFECTIVE_N_PIPELINE}} for the pipeline. The effective number of names is $1/\sum_i p_i^2$, where $p_i$ is name $i$'s share of all selected slots across the {{N_SLATES}} pools and {{TIEBREAK_REPEATS}} tie-break draws; it equals twenty when every name is chosen equally often. Name concentration is thus a property of the reranking stage, consistent with the same-gender comparison earlier in this section.

```latex
\begin{figure}[tbp]
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

Appendix C repeats the simulation with each of the {{N_SWEEP}} pointwise rerankers behind the same retriever; the listwise jina-reranker-v3 is left out because its scores depend on the other documents in a request, so scores from the audit's larger requests cannot be replayed in a smaller pool. The resulting shortlist share ranges from {{PIPELINE_SWEEP_MIN}} to {{PIPELINE_SWEEP_MAX}}, and {{N_PIPELINE_BELOW_RETRIEVER}} rerankers produce a shortlist with a lower male-associated share than retrieval alone ({{SHORTLIST_PCT_MALE_SEARCH_1DP}}). The worked example sits near the top of that range, whose median is {{PIPELINE_SWEEP_MEDIAN}}, rather than being typical. Replayed scores match actual ten-document requests: for {{REPLAY_N_SUPPORTED}} of the {{REPLAY_N_MODELS}} API rerankers we re-queried on {{REPLAY_N_POOLS}} pools, the difference between the actual and replayed male-associated shortlist share has a 95% interval that includes zero; for Cohere v4 pro it is {{REPLAY_COHEREPRO_DIFF}} points (interval {{REPLAY_COHEREPRO_LO}} to {{REPLAY_COHEREPRO_HI}}). The upper end of the range comes from IBM's Granite reranker, whose API endpoint we could not re-query. It is a pointwise cross-encoder, for which replay is exact, and its self-hosted copy agrees with the API in {{SELF_GRANITE_SAME_WINNER}} of pairs; without it, the range ends at {{SWEEP_MAX_WITHOUT_GRANITE}}. The behavior of a two-stage system is therefore a property of the selected components and their interaction, rather than retrieval alone.

## Full marker removal eliminates the measured pathway

Lowercasing alone, which removes no marker, moves the worked reranker's overall male-favored share from {{SYNTH_BGE_PCT_MALE}} to {{DEID_LOWERCASE_PCT_MALE}}, so the level is sensitive to surface form. It also moves the contrast between male-typed and female-typed occupations, from {{DEID_ORIGINAL_CONTRAST}} to {{DEID_LOWERCASE_CONTRAST}} points, although the contrast stays positive on all four tested rerankers (Table~\ref{tab:deidall}). Removing only names does not remove the difference. On the worked reranker, the male-associated version wins {{DEID_NAMESONLY_PCT_MALE}} of all comparisons after names are removed, against {{SYNTH_BGE_PCT_MALE}} before, and the contrast between male-typed and female-typed occupations is {{DEID_NAMESONLY_CONTRAST}} points, against {{DEID_ORIGINAL_CONTRAST}} before (Table~\ref{tab:deid}). The mean absolute gap is essentially unchanged ({{DEID_ORIGINAL_MEAN_GAP}} to {{DEID_NAMESONLY_MEAN_GAP}}; paired occupation-clustered 95% interval on the difference [{{DEID_NAMESONLY_GAPDIFF_LO}}, {{DEID_NAMESONLY_GAPDIFF_HI}}]). Neutralizing pronouns while retaining names reduces the mean absolute gap to {{DEID_PRONOUNSONLY_MEAN_GAP}} (interval on the difference [{{DEID_PRONOUNSONLY_GAPDIFF_LO}}, {{DEID_PRONOUNSONLY_GAPDIFF_HI}}]). These partial-rule figures do not isolate the markers: each rule inserts its replacement in lowercase, even at the start of a sentence, and none repairs verb agreement (Appendix D), so the transformed documents also differ from the originals in surface form, and surface form alone moves the overall male-favored share, as the lowercasing control shows. Partial removal is not consistent across the four rerankers in Appendix D: removing names alone leaves the mean absolute gap unchanged or higher on two models and reduces it on the other two, and on one model it lowers the male-favored share where the worked reranker's rises. Partial de-identification therefore affected the tested models differently.

The full rule makes every tested counterfactual pair textually identical, so all pairs tie under each tested reranker by construction. Identical strings are scored once and the score is reused, so this tie rate is not an observation; it holds for any deterministic pointwise reranker, which all four tested models are. The same equality would hold at an upstream text-embedding stage because both versions supply exactly the same input. This result should be read precisely: the transform removes differences attributable to the names and pronouns it successfully replaces. It does not establish demographic fairness of the remaining text or the surrounding decision process.

On the worked reranker, synthetic occupation-matching top-rank accuracy is {{UTILITY_CORRECT_BEFORE}} of {{UTILITY_N_EVENTS}} ({{UTILITY_TOP1_BEFORE}}) before the full transform and {{UTILITY_CORRECT_AFTER}} of {{UTILITY_N_EVENTS}} ({{UTILITY_TOP1_AFTER}}) after it: {{UTILITY_IMPROVED}} ranking events improved, {{UTILITY_WORSENED}} worsened, and accuracy changed by {{UTILITY_DELTA_PTS}} percentage points. Top-three accuracy is {{UTILITY_TOP3}} in both conditions. The ranking events repeat across templates and query phrasings within each occupation, so we report these counts descriptively rather than as a test of equivalence. Because the task is near ceiling, they show only that the transform leaves the occupation signal intact on this check. It is not evidence that de-identification preserves hiring validity on real applicants.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_deid.pdf}
\caption{\textbf{Pairwise balance under de-identification rules on the worked
reranker.} Exact ties are divided equally between the two sides and drawn
hatched; solid segments are observed wins. Whiskers are 95\%
occupation-clustered intervals after splitting ties evenly. The final two bars
are 100\% exact ties, and the naive her$\to$their rewrite is 75\% exact ties,
because those rules make the paired texts identical; ties are not wins for
either document. The partial rules also change capitalization and verb
agreement (Appendix D), so their bars reflect more than the removed markers.
Conditions are defined in Section~2 and Appendix D.}
\label{fig:deid}
\end{figure}
```

# Implications and future work

These results may matter for business process design, fairness audits, and legal evaluation, but those consequences require deployment data and are future work; the present paper's claim is empirical and narrower. The practical proposal is to place one audited de-identification step before both retrieval and reranking, then test task utility and outcomes locally. We test an input-side transform rather than retraining or a fairness-aware reranking stage because many of the audited rerankers, including every Voyage, Google, and Cohere model, are closed services whose weights a deployer cannot change; replacing identity tokens with one placeholder is the “blindness” baseline of the counterfactual-fairness literature, exact for the tokens it covers and silent about everything else [@garg2019]. Anonymization can backfire. In a French field experiment, volunteer firms randomly assigned anonymized résumés interviewed minority candidates less often than volunteer firms that saw standard résumés [@behaghel2015]. A de-identification step should therefore be judged by its measured effect on outcomes, not by the markers it removes. Because both stages receive the same normalized text, this removes the measured name-and-pronoun pathway from both stages when they are pointwise scorers. It may also reduce one source of repeated exclusion when many employers reuse the same ranking components, connecting our study to the “algorithmic monoculture” described by Bommasani and colleagues [@bommasani2026]. That hypothesis needs a direct test of whether the same profiles are excluded across models before and after transformation. Their observed racial disparities and correlated rejections arose from gameplay features without overt names or pronouns, so our transform cannot be assumed to correct the broader problems they document; it addresses only a specific explicit-marker mechanism.

# Limitations

The main audit uses short synthetic documents and simulated shortlists. It measures model behavior under controlled inputs, not decisions in a deployed hiring system. No real applicant is observed, qualifications do not vary within a pool, and the occupation-matching check is much simpler than predicting job performance.

Names carry several social signals at once. Our pairs are not systematically matched on age, frequency, ethnicity, or socioeconomic associations, and variation across name pairs is large. The pronoun-only comparison avoids these confounds but measures the pronoun channel alone. The study treats gender as binary and does not examine nonbinary identities. It also does not audit race or intersectional effects.

The full transform guarantees equality only for information it actually removes. Real resumes may signal gender through schools, affiliations, activities, job histories, and writing patterns [@parasurama2022full]. On a large corpus of real online biographies, removing names, pronouns, and honorifics reduced but did not eliminate the gender gap in occupation-classification recall, because other words still signal gender [@dearteaga2019]. Reliable name detection is also harder in real documents than in a synthetic corpus; in the companion corpus, most detected names belong to someone other than the résumé's owner (Appendix E). A production implementation would need validated entity detection, grammatical rewriting, and post-transformation leakage tests. This paper does not test the transform on real résumés, where names must be found automatically, or on a ranking task that is not at ceiling; we leave both, together with a measure of the gender signal that remains after the transform, to a follow-up study.

The worked pipeline uses one retriever-reranker pair, although Appendix C varies the reranker and Appendix D tests four open-weight rerankers. Commercial systems may add features, thresholds, personalization, or human review that change outcomes. The results are a dated snapshot of model versions queried on {{SCORED_DATE_1}} and from {{LATER_FIRST}} to {{LATER_LAST}}, and for the companion experiment from {{COMP_DATE_FIRST}} to {{COMP_DATE_LAST}}.

The companion experiment on real résumés (Section 3, Appendix E) shows that synthetic effect sizes should not be read as sizes on real résumés, but it has limits of its own. It uses publicly posted résumé examples rather than applications, with no hiring decisions or applicant outcomes, and occupation descriptions rather than job advertisements as queries. Most of its pairs differ by a name we added rather than one the applicant wrote. Which names and pronouns belong to the applicant was labelled by a language model, and no person has checked the labels. Its female-typed group rests on {{COMP_N_FEMALE_CATS}} job categories, although a check against every category's share of women agrees in sign for {{WR_N_RHO_AGREE}} of the {{WR_N_DIRECTED}} rerankers with a clear direction (Appendix E). Its corpus must be rebuilt from the repository's provenance notes to reproduce it.

# Conclusion

Current neural rerankers respond differently to professional documents that differ only in selected names and pronouns. In {{N_CONTRAST_CI_POS}} of {{N_MODELS}} models, the male-favored share is higher for male-typed than for female-typed occupations, and the same holds in {{PRON_N_ABOVE}} of {{PRON_N_MODELS}} re-scored models when only the pronouns differ. The baseline direction varies by product, and the size of the difference varies within vendors and model families. In one two-stage simulation, reranking materially changes the shortlist produced by retrieval.

For pointwise rerankers, a simple normalization of names and gendered pronouns eliminates this particular pairwise difference because it makes the compared inputs identical. No loss was observed on a near-ceiling occupation-matching check, but real hiring utility and residual demographic leakage remain untested. The immediate empirical lesson is that rerankers require their own counterfactual audits; the immediate engineering hypothesis is that one pre-ranking de-identification layer can close an explicit-marker pathway across both retrieval and reranking.

### Data and code availability

The generated documents, banked model scores, analysis scripts, pipeline simulation, and de-identification code are available in the project repository at \href{{{REPO_URL}}}{\texttt{github.com/\allowbreak vasylrakivnenko/\allowbreak bias_in_rerankers}}. Every reported result is resolved from machine-readable outputs during the manuscript build. Every table and figure can be regenerated exactly from the scores saved in the repository, with no model or API access; the release includes the pair-level scores for every de-identification condition and the per-event occupation-matching outcomes, and this manuscript corresponds to the repository tag {{RELEASE_TAG}}. Re-running the scoring is another matter. The October 2026 local and GPU runs recorded each model's revision, numeric precision, library versions, and scoring recipe, and the scripts reload those exact revisions. The February 2026 local runs did not record the model revision or library versions, and those settings cannot be recovered; the API models' scores may not reproduce either, since vendors can change the model behind a name, so for them the saved scores are the record, although an October 2026 re-query reproduced the February API models' contrasts within {{DRIFT_FEB_MAX}} points (Section 3). The benchmark's occupation words were assembled by hand and labeled with BLS shares; O\*NET 30.3 (National Center for O\*NET Development for the U.S. Department of Labor, Employment and Training Administration), used under CC BY 4.0 [@onet2026], supplies the occupation titles and descriptions that serve as job queries in the companion experiment, and the SOC codes recorded beside the BLS categories in the repository's label table. The companion corpus is released under CC0 1.0, and the BLS and SSA inputs are U.S. Government works. Full provenance is in `data/PROVENANCE.md`.

```latex
\appendix
\setlength{\belowcaptionskip}{6pt}
\renewcommand{\thetable}{A\arabic{table}}
\setcounter{table}{0}
```

# Audit inputs

The document templates, query phrasings, and name pairs used in the counterfactual audit follow.

```latex
{{TABLE_DOCUMENTS}}
```

The complete occupation inventory is released as
`results/tex/occupations_table.tex` rather than reproduced here. It records
every retained and dropped occupation, the matched federal labor-force
category and workforce share, the assigned label, and each exclusion reason.

# Models and single-stage results

The three-way split by occupation category for every model, occupation-level consistency, and the robustness tables by document template, query phrasing, and name pair are released in the repository's `results/tex` folder. Tables use the short labels in Table~\ref{tab:models}, which maps each label to
the model's full identifier, vendor and access route, and records the number of
comparisons, score range, overall tie rate, and scoring date.

```latex
\newcommand{\ModelsCaption}{Model labels, identifiers, and provenance, in descending order of the tie-aware male-favored share on male-typed occupations, which Table~\ref{tab:contrast} lists. Identifiers are printed exactly as loaded and may wrap after a hyphen; no hyphen is added at a line break. Dates are as each run recorded them: UTC on the rented GPU, Pacific time otherwise.}
{\footnotesize
{{TABLE_MODELS_LONG}}
}
```

```latex

\begin{table}[H]
\centering
\caption{How often each model favored the male-associated document on male-typed
and on female-typed occupations (tie-aware \%, the points in
Figure~\ref{fig:single}), and their difference, the male-typed minus female-typed
contrast in percentage points, with individual 95\% intervals for the contrast
under three resampling schemes: occupations; federal labor-force categories; and
occupations and name pairs jointly. Intervals are not adjusted for multiple comparisons. The last column is
the two-sided p-value of an occupation-label permutation test of the contrast
({{PERM_N}} permutations), Holm-adjusted across the {{N_MODELS}} models.}
\label{tab:contrast}
{\small
{{TABLE_CONTRAST_INTERVALS}}
}
\end{table}

\begin{table}[H]
\centering
\caption{Pronoun-only comparison. Male-typed minus female-typed contrast
(tie-aware percentage points) with names and pronouns, and when both names are
replaced by the same placeholder so that the two documents differ only in their
pronouns, with occupation-clustered 95\% intervals for the pronoun-only contrast
({{PRON_B}} draws). Original is the contrast of Table~\ref{tab:contrast}, which
counts each of a cell's ten name pairs separately; Name-avg., defined below the
table, gives one comparison per cell, like the pronoun-only design. The last
column shows how often the run that scored the pronoun-only texts reproduced the
saved winners of the original pairs (defined below the table). Every open-weight
row compares two scoring runs; for the {{PRON_N_DRIFT_BELOW_100}} models below
100\% on the {{DRIFT_N_PAIRS}} checked pairs, the change of machine alone moved
some winners. Scored on {{PRON_DATE}} (Pacific time); the
listwise jina-reranker-v3 is excluded. $^{\dagger}$Scored from IBM's public
weights, which agree with its API on {{SELF_GRANITE_SAME_WINNER}} of pairs.}
\label{tab:pronoun}
{\small
{{TABLE_PRONOUN}}
}
\end{table}

\begin{table}[H]
\centering
\caption{Continuous check of the occupation labels. For each model: Spearman's
$\rho$ between the occupation-level tie-aware male-favored share and the federal
share of women across the {{N_OCCUPATIONS}} retained occupations; the two-sided
permutation $p$-value ({{CONT_N_PERM}} relabelings), Holm-adjusted across the
{{N_MODELS}} models; the least-squares slope in points of male-favored share per
10 points of women's share; and the male-typed minus female-typed contrast under
the 30/70 cut-points of Section~2, under 25/75 with its 95\% interval, and under
35/65 (see the note below the table).}
\label{tab:continuous}
{\small
{{TABLE_CONTINUOUS}}
}
\end{table}

\begin{table}[H]
\centering
\caption{Same-gender name swaps as a yardstick and placebo. For each model: the
mean absolute score gap of the name-and-pronoun swap and of same-gender name
swaps, their ratio with an occupation-clustered 95\% interval ({{SG_B}} draws),
the share of name-and-pronoun pairs whose gap exceeds the same-gender median of
their cell, the observed male-typed minus female-typed contrast, and the largest
contrast from any single same-gender name pair; the note below the table defines
each. The matched-design placebo of Section~3 is released with the data.}
\label{tab:samegender}
{\footnotesize\setlength{\tabcolsep}{3.5pt}
{{TABLE_SAMEGENDER}}
}
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
means female-associated documents rank lower. Positions 11--20 keep the
retriever's order, so this column also reflects the retriever. A reranker with no
preference among the ten documents it receives would give each of them the
average rank, {{TOPK_MEAN_RANK}}, and a gap of {{GAP_ORDER_NEUTRAL_RERANKER}}, the
same to two decimals as the retriever's own gap over all 20, because the
retriever's gap comes from which documents reach its top ten, not from their
order within it; {{N_RERANKERS_GAP_BELOW_NEUTRAL}} rerankers fall below
that value.}
\label{tab:sweep}
\resizebox{\textwidth}{!}{%
{{TABLE_SWEEP}}%
}
\end{table}
```

# De-identification results

Besides the three core rules, Table~\ref{tab:deid} includes a lowercasing-only control that removes no marker; two naive grammatical rewrites that also replace names and map “she/he” to “they,” “him” to “them,” “his” to “their,” and “her” to either “their” or “them”; and a following-word rule, otherwise identical, that maps “her” to “their” or “them” depending on the next word. Every rule inserts its replacement tokens in lowercase, including at the start of a sentence, and no rule repairs verb agreement, so the full rule turns a short bio into “the candidate is a dedicated nurse … they specializes in they area of expertise” [sic], and the grammatical rewrites into “… they specializes in their area of expertise.” The partial rules therefore change the surface form beyond the markers they remove, which is one more reason to read their levels as the effect of the rule rather than of the markers alone; the full rule applies the same change to both versions of a pair, so its ties are unaffected.

```latex
\begin{table}[H]
\centering
\caption{De-identification conditions on {{RERANKER_MODEL}}. A dagger marks a
condition that makes paired documents identical by construction. The
contrast column is the tie-aware male-favored share on male-typed minus
female-typed occupations, in percentage points; the three share columns before
it count exact ties separately. The utility column is synthetic
occupation-matching top-rank accuracy; a dash means that check was not run for
the condition.}
\label{tab:deid}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID}}%
}
\end{table}

\begin{table}[H]
\centering
\caption{De-identification conditions across four locally run rerankers. For
each model the columns give the tie-aware male-favored share, the male-typed
minus female-typed contrast in percentage points, and the mean absolute score
gap.}
\label{tab:deidall}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID_ALLMODELS}}%
}
{{TABLE_DEID_ALLMODELS_NOTES}}
\end{table}
```

# Companion experiment

For each of the corpus's {{COMP_N_QUERIES}} job categories, the title and description of the matched O\*NET occupation served as the job query. We drew {{COMP_POOLS_PER_QUERY}} pools of {{COMP_POOL_SIZE}} résumés per query ({{COMP_N_POOLS}} pools), each combining {{COMP_RELEVANT}} résumés from the query's category with {{COMP_OTHER}} drawn at random from the whole corpus. Each pool member served once as the focal candidate, giving {{N_COMPANION_COMPARISONS}} comparisons per model, in which only the focal résumé switched between its male- and female-associated forms while the other members stayed in a fixed, randomly assigned form. Pools were drawn independently, so the {{N_REAL_RESUMES}} distinct résumés appear in one or several pools.

Every reranker except jina-reranker-v2-base-multilingual, which reads at most {{WR_JINA_WINDOW}} tokens and so cuts most résumés, read each query and résumé whole: the hosted services with truncation turned off, and the open-weight models at the audit's pinned revisions with an input limit above the longest input. The audit's other {{WR_N_NOT_RUN}} rerankers cut every input at 512 or 1,024 tokens and were not run. jina-reranker-v3 is listwise and sensitive to the order of the documents in a request, so its row averages each comparison over the two orders, as its synthetic contrast does. Cohere rerank v5 pro and fast were released after the audit; their synthetic contrasts and intervals come from scoring the audit's pairs in October 2026.

Every document was paired by prepending a male or female given name, the two equally common, with an identical surname. The same name also replaces the applicant's own name wherever the résumé gives it, and the applicant's own pronouns take its gender. Names and pronouns that belong to other people, such as a supervisor, a book's author, or a school's namesake, stay as written. Of the {{N_REAL_RESUMES}} scored résumés, {{OL_N_LABELLED}} carry a detected name or a gendered pronoun; across them, {{OL_N_NAMES_OTHER}} of {{OL_N_NAMES}} detected-name occurrences belong to other people, and {{OL_N_PRON_NOT_OWNER}} of {{OL_N_PRONOUNS}} pronouns do not refer to the résumé's owner. Two passes by the same language model (Claude), run independently, labelled every occurrence. They agreed on {{OL_AGREE_NAMES}} of {{OL_N_NAMES}} names and {{OL_AGREE_PRON}} of {{OL_N_PRONOUNS}} pronouns, which shows consistency rather than accuracy; a third reading by the model settled the disagreements and corrected one name, and no person has checked the labels. The labels, stored as character offsets without the names, are in the repository. An earlier run of all {{WR_N_MODELS}} rerankers swapped every detected name and pronoun instead, including other people's. Its contrasts differ from those in Table~\ref{tab:wholeresume} by {{OL_CHANGE_MIN}} to {{OL_CHANGE_MAX}} points (median {{OL_CHANGE_MEDIAN}}), and none has the other sign, but {{OL_N_DIR_CHANGES}} rerankers whose intervals lie near zero change category: Cohere rerank v4 fast, Qwen3-Reranker-0.6B, and granite-embedding-reranker-english-r2 move from no clear direction to male-favoring, and gte-reranker-modernbert-base the other way.

The {{COMP_N_QUERIES}} job categories fall into {{COMP_N_MALE_CATS}} male-typed, {{COMP_N_FEMALE_CATS}} female-typed, {{COMP_N_EVEN_CATS}} evenly split, and {{COMP_N_UNLABELED_CATS}} unlabeled groups. Each category takes the women's share of its matched federal labor-force category and the thresholds of Section 2; categories with no match or a share in an excluded band stay unlabeled. Because the female-typed group has only {{COMP_N_FEMALE_CATS}} categories, each run is also tested against the women's share of all {{WR_N_CONT_CATS}} categories that have one (last column of Table~\ref{tab:wholeresume}). For {{WR_N_RHO_AGREE}} of the {{WR_N_DIRECTED}} rerankers whose whole-résumé interval excludes zero, the sign of this correlation matches the direction of the contrast: the male-favored share falls as the share of women rises for {{WR_N_MALE_RHO_NEG}} of the {{WR_N_MALE}} with a positive contrast and rises for all {{WR_N_FEMALE}} reversals. The exception is Cohere rerank v4 fast, whose interval only just excludes zero and whose correlation is {{WR_C4F_RHO}} (permutation p = {{WR_C4F_P}}). The correlation differs from zero for {{WR_N_MALE_RHO}} of the {{WR_N_MALE}} and for all {{WR_N_FEMALE}} reversals (permutation p < 0.05 with categories as the units, not adjusted for the number of rerankers).

```latex
\begin{table}[H]
\centering
\caption{Companion experiment on whole résumés. M$-$F is the tie-aware share
(\%) of comparisons in which the male-associated version scores higher on
male-typed occupations minus that share on female-typed occupations, in the
synthetic audit and on whole résumés; the interval resamples pools within each
job category. $\rho$ is Spearman's correlation between each job category's
male-favored share and its share of women, across the
{{WR_N_CONT_CATS}} categories with one (* permutation p $<$ 0.05). Rows are
sorted by the synthetic M$-$F. $^{a}$Released after the audit; synthetic M$-$F
scored in October 2026 on the audit's pairs. $^{b}$Reads the first
{{WR_JINA_WINDOW}} tokens. $^{c}$Listwise; both columns average the two orders
in which a pair's versions can be sent. Only the applicant's own name and
pronouns change between a pair's versions.}
\label{tab:wholeresume}
{\small
{{TABLE_WHOLE_RESUME}}
}
\end{table}
```

We ran {{LP_N_MODELS}} rerankers again to separate length from content: Voyage rerank-3, Cohere rerank v5 pro, and Qwen3-Reranker-8B, which keep the pattern on whole résumés; Voyage rerank-2 and mxbai-rerank-large-v2, which reverse it; and bge-reranker-v2-m3, the worked reranker, whose contrast is small in both settings. Each scored the same {{N_COMPANION_COMPARISONS}} comparisons with the text after the prepended name cut to its first {{LP_CUT_TEXT}} words or left whole, each with the name alone or with the name followed by the sentence “He is a dedicated professional, and his colleagues describe him as reliable and thorough” (or its female form), whose {{LP_LINE_WORDS}} words count toward the cut. The 35-word condition with the sentence resembles a synthetic document in length and in carrying a name and pronouns. Figure~\ref{fig:lengthpronoun} shows every condition.

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_length_pronoun.pdf}
\caption{\textbf{Companion experiment by résumé length.} Each panel shows one
reranker's male-typed minus female-typed contrast (points) as more of each
résumé is kept, with the prepended name alone (solid) or with the name and a
sentence carrying {{LP_LINE_PRONOUNS}} of the applicant's pronouns (dashed), whose words count
toward the length; ``all'' is the whole résumé. Only the applicant's own name and
pronouns change. Bands are 95\% intervals resampling pools within each job
category; the dashed gray line is the reranker's contrast in the synthetic
audit.}
\label{fig:lengthpronoun}
\end{figure}
```

```latex
\begin{thebibliography}{99}

\bibitem{an2025}
J.~An, D.~Huang, C.~Lin, and M.~Tai.
\newblock Measuring gender and racial biases in large language models: Intersectional evidence from automated resume evaluation.
\newblock \emph{PNAS Nexus}, 4(3):pgaf089, 2025. \doi{10.1093/pnasnexus/pgaf089}.

\bibitem{behaghel2015}
L.~Behaghel, B.~Cr\'{e}pon, and T.~Le~Barbanchon.
\newblock Unintended Effects of Anonymous R\'{e}sum\'{e}s.
\newblock \emph{American Economic Journal: Applied Economics}, 7(3):1--27, 2015. \doi{10.1257/app.20140185}.

\bibitem{bertrand2004}
M.~Bertrand and S.~Mullainathan.
\newblock Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination.
\newblock \emph{American Economic Review}, 94(4):991--1013, 2004. \doi{10.1257/0002828042002561}.

\bibitem{bigdeli2021}
A.~Bigdeli, N.~Arabzadeh, S.~Seyedsalehi, M.~Zihayat, and E.~Bagheri.
\newblock On the Orthogonality of Bias and Utility in Ad hoc Retrieval.
\newblock In \emph{Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval}, pages 1748--1752, 2021.
\doi{10.1145/3404835.3463110}.

\bibitem{bigdeli2022}
A.~Bigdeli, N.~Arabzadeh, S.~Seyedsalehi, M.~Zihayat, and E.~Bagheri.
\newblock A Light-Weight Strategy for Restraining Gender Biases in Neural Rankers.
\newblock In \emph{Advances in Information Retrieval: 44th European Conference on IR Research (ECIR 2022), Part II}, volume 13186 of \emph{Lecture Notes in Computer Science}, pages 47--55, 2022.
\doi{10.1007/978-3-030-99739-7\_6}.

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

\bibitem{dearteaga2019}
M.~De-Arteaga, A.~Romanov, H.~Wallach, J.~Chayes, C.~Borgs, A.~Chouldechova, S.~Geyik, K.~Kenthapadi, and A.~T.~Kalai.
\newblock Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting.
\newblock In \emph{Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)}, pages 120--128, 2019.
\doi{10.1145/3287560.3287572}.

\bibitem{garg2019}
S.~Garg, V.~Perot, N.~Limtiaco, A.~Taly, E.~H.~Chi, and A.~Beutel.
\newblock Counterfactual Fairness in Text Classification through Robustness.
\newblock In \emph{Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (AIES)}, pages 219--226, 2019.
\doi{10.1145/3306618.3317950}.

\bibitem{karpukhin2020}
V.~Karpukhin, B.~O\u{g}uz, S.~Min, P.~Lewis, L.~Wu, S.~Edunov, D.~Chen, and W.~Yih.
\newblock Dense Passage Retrieval for Open-Domain Question Answering.
\newblock In \emph{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, pages 6769--6781, 2020.
\doi{10.18653/v1/2020.emnlp-main.550}.

\bibitem{kiritchenko2018}
S.~Kiritchenko and S.~Mohammad.
\newblock Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems.
\newblock In \emph{Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics (*SEM)}, pages 43--53, 2018.
\doi{10.18653/v1/S18-2005}.

\bibitem{krieg2023}
K.~Krieg, E.~Parada-Cabaleiro, G.~Medicus, O.~Lesota, M.~Schedl, and N.~Rekabsaz.
\newblock Grep-BiasIR: A Dataset for Investigating Gender Representation Bias in Information Retrieval Results.
\newblock In \emph{Proceedings of the 2023 Conference on Human Information Interaction and Retrieval (CHIIR '23)}, pages 444--448, 2023.
\doi{10.1145/3576840.3578295}.

\bibitem{kumar2023}
D.~Kumar, T.~Grosz, E.~Greif, N.~Rekabsaz, and M.~Schedl.
\newblock Identifying Words in Job Advertisements Responsible for Gender Bias in Candidate Ranking Systems via Counterfactual Learning.
\newblock In \emph{Proceedings of the 3rd Workshop on Recommender Systems for Human Resources (RecSys in HR 2023)}, volume 3490 of \emph{CEUR Workshop Proceedings}, 2023.
\url{https://ceur-ws.org/Vol-3490/RecSysHR2023-paper_7.pdf}.

\bibitem{kumar2025}
D.~Kumar, S.~Masoudian, A.~B.~Melchiorre, and M.~Schedl.
\newblock Mitigating Gender Bias in Job Ranking Systems Using Job Advertisement Neutrality.
\newblock In \emph{Proceedings of the Fourth Workshop on NLP for Positive Impact (NLP4PI)}, pages 264--271, 2025.
\doi{10.18653/v1/2025.nlp4pi-1.23}.

\bibitem{mao2025}
Y.~Mao, B.~Plank, and R.~Litschko.
\newblock Relevant for the Right Reasons? Investigating Lexical Biases in Zero-Shot and Instruction-Tuned Rerankers.
\newblock In \emph{Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025)}, pages 468--482, 2025.
\doi{10.18653/v1/2025.mrl-main.31}.

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

\bibitem{seshadri2022}
P.~Seshadri, P.~Pezeshkpour, and S.~Singh.
\newblock Quantifying Social Biases Using Templates is Unreliable.
\newblock arXiv preprint arXiv:2210.04337, 2022.

\bibitem{seshadri2025}
P.~Seshadri, H.~Chen, S.~Singh, and S.~Goldfarb-Tarrant.
\newblock Small Changes, Large Consequences: Analyzing the Allocational Fairness of LLMs in Hiring Contexts.
\newblock In \emph{Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL)}, pages 2645--2665, 2025.
\doi{10.18653/v1/2025.ijcnlp-long.143}.

\bibitem{ssa2026}
Social Security Administration.
\newblock National data on the relative frequency of given names in the population of U.S. births.
\newblock Tabulated from Social Security records as of 1 March 2026.
\url{https://www.ssa.gov/oact/babynames/limits.html}.

\bibitem{wang2024}
Y.~Wang, X.~Wu, H.-T.~Wu, Z.~Tao, and Y.~Fang.
\newblock Do Large Language Models Rank Fairly? An Empirical Study on the Fairness of LLMs as Rankers.
\newblock In \emph{Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)}, pages 5712--5724, 2024.
\doi{10.18653/v1/2024.naacl-long.319}.

\bibitem{wilson2024}
K.~Wilson and A.~Caliskan.
\newblock Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval.
\newblock In \emph{Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society}, 7(1):1578--1590, 2024.
\doi{10.1609/aies.v7i1.31748}.

\bibitem{zerveas2022}
G.~Zerveas, N.~Rekabsaz, D.~Cohen, and C.~Eickhoff.
\newblock Mitigating Bias in Search Results Through Contextual Document Reranking and Neutrality Regularization.
\newblock In \emph{Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval}, pages 2532--2538, 2022.
\doi{10.1145/3477495.3531891}.

\end{thebibliography}
```
