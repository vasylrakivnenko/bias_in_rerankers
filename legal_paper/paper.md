---
title: Gender Bias in Automated Hiring\\ --- the Case for De-Identification
author: Vasyl Rakivnenko
date:
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

As job applications are increasingly screened by algorithm before any human assesses the candidate [@banks2026], the significant's of bias present in hiring algorithms is becoming increasingly important. Typical automated hiring pipeline includes two core stages - text embedding model first orders the pool of resumes by semantic similarity to the given job description, and then a neural reranker re-scores the top candidates to fix the final order. Prior work has documented gender bias in the embedding models underlying retrieval [@rakivnenko2024] and in language models used for resume screening, but no study has audited identical candidates across open source and proprietary neural rerankers - the component with the last word on who reaches the top of the list. We address this gap by auditing {{N_MODELS}} proprietary and open-source rerankers using counterfactual pairs: resumes identical except for the candidate's name and pronouns, though names also carry associations beyond gender that this design does not separate out. All {{N_MODELS_STEREOTYPE_ORDER}} models favored the male-named candidate more often for male-typed jobs than for female-typed ones, a pattern that tracks occupational stereotypes; which way an individual model leans by default, however, is a property of that product and cannot be known without testing it. In a simulated pool of {{SLATE_SIZE}} equally qualified candidates, this bias compounds through a common retrieval-and-reranking pipeline: men receive {{SHORTLIST_PCT_MALE_PIPELINE}} of shortlist places ({{SHORTLIST_PCT_MALE_PIPELINE_GLOSS}}), and women's shortlist selection rate falls to {{FF_SELECTION_RATIO_GLOSS}} of men's - well under the four-fifths threshold federal enforcement agencies use to flag disparate impact, though courts treat it as a starting point rather than proof [@uniformguidelines]. We first tried removing only the candidate's name, the marker that prior resume audits isolate [@wilson2024] and that anonymised-application programmes strip [@behaghel2015], but it still left the direction of the bias almost entirely intact; removing gendered pronouns as well, however, eliminates the disparity. Another important observation that we document is that in most cases a buyer won't learn these risks from the vendor: of the {{N_MODELS}} rerankers we audited, {{N_MODELS_FAIRNESS_EVAL}} had a published fairness evaluation of the ranking model and {{N_MODELS_GENDER_DISCLOSURE}} disclosed a gender-related limitation. We argue that technology providers should add desclaimers about gender bias or limitations of their rerankers for hiring automation purposes and that full (full) de-identification (including pronouns) is a testable and auditable mitigation approach that hiring-technology deployers should adopt and validate against their own hiring outcomes.

# Introduction

AI-powered software that sorts job applicants usually works in two steps [@nogueira2019]. An *embedding* model scans a large pool of resumes and sorts the most semantically similar to the job descriptions candidates to the top. A *reranker* model (typically, more intelligent model for this purpose, yet with a smaller context window) then takes several top candidates, reads each one against the job description and fixes the final order. The embedding model is the clerk who pulls a stack of files in an initial order; the reranker is the decision-maker who reads them and arranges them best-first. The people at the top of that order get looked at. The rest, in practice, do not.

This is not hypothetical. "Job applications are being screened by an algorithm before any human assesses a candidate's qualifications" [@banks2026] and some applicants now sit a first-round interview with a conversational agent before any recruiter is involved [@abril2025]. Our subject is the quieter step before that one: the ordering of the pile. LinkedIn Recruiter returns a ranked list of candidates [@linkedin]; the same kind of ranking technology runs the job-seeker side of the market at Indeed and Google [@indeed; @google]. Published resume-matching systems score a job description and a candidate document together, using the same family of neural rankers we audit [@shpjf; @confit].

Researchers have shown that the embedding models behind the retriever carry gender bias [@caliskan2017; @rakivnenko2024]. The closest study to ours inserts names into otherwise equivalent resumes and measures what the retrieval stage returns [@wilson2024]. The reranker has been studied far less, and we found no prior audit scores identical candidates across commercial neural rerankers. We believe, that is the wrong way round - the stage we know most about (bias embedding model) is the one the next stage (rerankers) can overrule. The reranker has the last word, and it decides who appears at the top of the recruiter's screen.

Gender bias in neural rankers is not new. Rekabsaz and Schedl showed that BERT rankers surface more male-oriented passages than keyword search for neutral queries [@rekabsaz2020], and later added an adversarial fix [@rekabsaz2021]. Both papers ask what the retrieved text is about, not how documents differing in names and pronouns are scored. Audits of language models do ask that, and they disagree on direction. One study of large language models scoring resumes finds an advantage for women [@an2025]; an embedding-based resume screener favors male-associated names [@wilson2024]. We run the counterfactual test on the general-purpose rerankers that published resume-matching systems build on, and follow it through to who makes the shortlist. We do not claim any particular employer runs these products: the vendors document them for search, not for hiring, and our own audit of their documentation finds no exception. What we claim is narrower, and enough. These components can be assembled into such a pipeline. Some vendors restrict employment decisions through company-wide policies, but their ranking documentation does not report demographic evaluations of these models.

Legal scholars have made the same argument from the other direction. Title VII's disparate-impact framework, they argue, already extends to algorithmic hiring decisions [@barocas2016], and automated hiring systems specifically call for a mandatory auditing regime [@ajunwa2021]. Yet, those arguments have had little audit evidence to draw on for the ranking stage specifically. In this work we contribute that kind of audit.

Houser argues that responsible AI can reduce human hiring bias through anonymisation and auditing \cite[pp.~325--326, 344]{houser2019}. We add contemporary evidence on {{N_MODELS}} proprietary and open-source neural rerankers, connecting their score differences to shortlist outcomes and testing a few de-identification approaches at that stage. The model scores were collected on {{RAW_SCORED_DATE}}; the documentation audit covers {{VENDOR_AUDIT_DATE}}. Together, these provide a dated account of model behaviour and the information available to buyers. Houser also argues that AI should be compared with imperfect human decision-making \cite[p.~344]{houser2019}. Our experiment identifies a disparity within these models; it does not compare their performance with human recruiters.

Two legal regimes already cover this ground. In the United States, a neutral-looking practice that screens out one sex at a disproportionate rate is unlawful unless the employer shows business necessity. Even then it fails if a less discriminatory alternative exists and the employer declines it. In the European Union, systems that filter job applications are classed as high-risk. From December 2027, the organisations that deploy this kind of technology owe duties to control that risk. Both regimes ask the same practical question: is there a more efficient and fairer way to do the same job?

# Methodology

Our test is a counterfactual pair: two versions of the same professional document, identical word for word except for the candidate's name and pronouns. The design follows resume audits that swap names in otherwise equivalent documents [@wilson2024]; we apply it one stage downstream and extend it to pronouns. Any score difference therefore comes from those changed markers. Names also carry associations with age, ethnicity and individual identity, which this design does not separate from gender. We built these documents from {{N_TEMPLATES}} templates across {{N_OCCUPATIONS}} occupations. We varied {{N_NAME_PAIRS}} common American given names of each sex, taken from national birth-name records [@ssa2026]. We used {{N_QUERIES}} ways of phrasing the search. That yields {{N_PAIRS}} counterfactual pairs per model. The appendix shows the templates, the query forms, and the name pairs in full.

We scored {{N_OCC_SCORED}} occupations, then labeled each by the share of women employed in it, from the federal labor-force survey [@bls2025]. {{N_OCC_DROPPED}} occupations had no usable labor-force figure -- no matching category, a suppressed estimate, or a share too close to the 30% and 70% cutoffs to call -- and are excluded from the synthetic analyses; the appendix lists all of them, with the reason for each. Of the remaining {{N_OCCUPATIONS}}, {{N_OCC_MALE}} are male-typed, {{N_OCC_FEMALE}} are female-typed, and {{N_OCC_NEUTRAL}} are close to evenly split. The evenly split group carries most of the legal weight, because nobody can claim those jobs require a particular sex.

We scored every counterfactual pair with {{N_MODELS}} rerankers, both commercial and open-source. Some models return coarse scores that give the two documents exactly the same number. We count those ties as ties. The tables report them in their own column; where we need a single share, a tie counts as half a win for each document. The appendix lists every model, its vendor, and its tie rate.

We also read what the vendors themselves publish. For every reranker in the audit we went through the model card, the product documentation, any technical report, and the terms and usage policies. That is {{N_VENDOR_SOURCES}} first-party sources across {{N_RERANKER_VENDORS}} vendors, each recorded with the date we read it and the wording we relied on. We asked four questions of each. Is hiring named as an intended use? Is any fairness evaluation of the ranking model published? Is any gender-related limitation disclosed? Is there a restriction covering high-stakes or employment decisions? This is an audit of what a buyer can read, not of what a vendor does internally.

To see what this does in practice, we simulated a pool of candidates for one job. It holds {{SLATE_SIZE}} documents, half with men's names and half with women's, identical in every other respect. Because the candidates are by construction equally qualified, a fair system should treat them interchangeably. We ran the pool through a common open-source pipeline -- the retriever {{RETRIEVER_MODEL}} feeding its top {{TOPK}} candidates to the reranker {{RERANKER_MODEL}} -- and recorded who made the top three. Finally, we re-scored everything after stripping gender markers, at several strengths, to see which strengths actually work.

# Findings

## Identical candidates are not scored equally

All {{N_MODELS_STEREOTYPE_ORDER}} models favored the man more often for male-typed jobs than for female-typed ones. The size of that gap varies; its direction never does. For male-typed jobs the median model favored the man in {{MEDIAN_PCT_MALE_MALEJOBS}} of comparisons, {{MEDIAN_PCT_MALE_MALEJOBS_GLOSS}}. For female-typed jobs the median model favored the man in only {{MEDIAN_PCT_MALE_FEMALEJOBS}}. Nothing changed between the two documents but the name and the pronouns. The aggregate pattern is consistent with job stereotypes. It does not establish that gender, rather than other associations of the chosen names, explains the entire effect.

Where a model sits by default is a different matter. On jobs the labor market splits evenly, the median model favored the man in {{MEDIAN_PCT_MALE_NEUTRAL}} of comparisons, {{MEDIAN_PCT_MALE_NEUTRAL_GLOSS}}. But the models themselves disagree: {{N_MODELS_NEUTRAL_LEAN_MALE}} of the {{N_MODELS}} lean toward men on those jobs, and {{N_MODELS_NEUTRAL_LEAN_FEMALE}} lean the other way. For {{N_NEUTRAL_INDETERMINATE}} of them the lean is too small to call: their confidence interval includes an even split, so we count the direction but do not claim it. A buyer therefore cannot guess which sex a given product disadvantages. The direction is invisible from the outside, and knowable only by testing.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_single_stage.pdf}
\caption{\textbf{Reranker preferences across female- and male-typed
occupations.} Each row shows one reranker. Blue and rose points show how often
it favored the man's document on male-typed and female-typed occupations,
respectively; an exact tie counts one-half for each version. Rows run from the
largest change at the top to the smallest at the bottom. Every blue point is to
the right of its rose counterpart. Exact values and occupation-clustered
intervals are in the appendix.}
\label{fig:single}
\end{figure}
```

## The reranker has the last word

A pool of {{SLATE_SIZE}} interchangeable candidates should produce a shortlist that is half men and half women. It does not. We expected the two stages to compound. A candidate must clear both, so two filters leaning the same way should skew the shortlist further than either alone. That is what happens where they do lean together. On male-typed jobs the retriever alone gave men {{SHORTLIST_PCT_MALE_SEARCH_MALEJOBS}} of the top three places, the reranker alone {{SHORTLIST_PCT_MALE_RANKER_MALEJOBS}}, and the two in sequence {{SHORTLIST_PCT_MALE_PIPELINE_MALEJOBS}} -- worse than either stage on its own.

Where the two stages disagree, the effect reverses. On female-typed jobs the retriever leans the other way, giving men {{SHORTLIST_PCT_MALE_SEARCH_FEMJOBS}} of shortlist places. That pulls the pipeline back from the reranker's {{SHORTLIST_PCT_MALE_RANKER_FEMJOBS}} to {{SHORTLIST_PCT_MALE_PIPELINE_FEMJOBS}}. Across all jobs the two effects cancel, and the pipeline comes to resemble the reranker alone: {{SHORTLIST_PCT_MALE_PIPELINE}} against {{SHORTLIST_PCT_MALE_RANKER}}. For the evenly split jobs it is {{SHORTLIST_PCT_MALE_PIPELINE_NEUTRAL}}.

Favoring either sex among interchangeable candidates is a disparity, so none of these figures is the right answer. Two lessons follow. The pipeline cannot be read off either stage alone; what matters is whether the two happen to agree. And fixing the retriever achieves little on its own, because the reranker has the last word.

This also locates the existing evidence. The retrieval stage, where resume audits have looked [@wilson2024], gave men {{SHORTLIST_PCT_MALE_SEARCH}} of shortlist places in our pool. The reranker took the same pool to {{SHORTLIST_PCT_MALE_PIPELINE}}. An intervention aimed only at retrieval would leave most of that in place.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_pipeline.pdf}
\caption{\textbf{Shortlist composition by pipeline stage and occupation group.}
Bars show the share of top-three shortlist places going to men's names in a
pool of equally qualified candidates under the retriever alone, the reranker
alone, and the two combined. Whiskers are 95\% occupation-clustered bootstrap
intervals.}
\label{fig:pipeline}
\end{figure}
```

## Removing the markers removes the gap, if you remove all of them

If both documents become the same text, any model must score them the same. Our full transform replaces the name and neutralises the pronouns, and every pair then ties. That is a check on the argument rather than a discovery, because identical texts cannot be scored differently. Half-measures are the interesting case, and they cut the other way from what you might guess.

Deleting names but leaving pronouns does not shrink the score gap -- if anything it is slightly larger: {{DEID_NAMESONLY_MEAN_GAP}} against {{DEID_ORIGINAL_MEAN_GAP}}. What changes is consistency: the man's document wins {{DEID_NAMESONLY_PCT_MALE_MALEJOBS}} of male-typed comparisons, up from {{DEID_ORIGINAL_PCT_MALE_MALEJOBS}}. Removing the name does not take away a large signal and leave a small one behind. It takes away the noisier signal and leaves the more consistent one. Neutralising pronouns but keeping the name shows the same thing from the other side: the gap falls to {{DEID_PRONOUNSONLY_MEAN_GAP}}. This pattern is not identical on every reranker we tested; the appendix repeats it on three more.

So a half-transform is not half a fix. English *her* is both possessive and object, so replacing it with *their* leaves some pairs differing by a word. That happens on {{DEID_GRAM_NAIVE_PCT_RESIDUE}} of pairs, and where it happens {{DEID_GRAM_NAIVE_RESIDUE_PCT_FEMALE}} of the leftovers favor the woman's document. A heuristic using the following word equalises all our templates, but it does not analyse grammar or reliably handle ordinary prose. It can turn "praised her yesterday" into "praised their yesterday" and leaves verb agreement errors such as "they specializes".

The full transform showed no loss on our synthetic occupation-matching test: top-1 accuracy was {{UTILITY_TOP1_AFTER}}, against {{UTILITY_TOP1_BEFORE}} before transformation. Each query sought the document naming its occupation among documents naming other occupations. This does not establish equal effectiveness at choosing better-qualified applicants within the same occupation.

```latex
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\linewidth]{figures/fig_deid.pdf}
\caption{\textbf{Pairwise balance under de-identification rules.} Exact ties are
divided equally between the two sides for display. Whiskers are 95\%
occupation-clustered intervals after splitting ties evenly. The final two
50/50 bars are 100\% exact ties because those rules make the paired texts
identical; they are not observed wins for either document. The simple
\emph{her $\rightarrow$ their} rewrite leaves a quarter of pairs different.
Job-category breakdowns are in the appendix.}
\label{fig:deid}
\end{figure}
```

# Implications

### Judged by the result, not the mechanism

United States law does not ask how a hiring tool reaches its decision. It asks whether a neutral-looking practice selects one sex at a disproportionate rate [@titlevii; @griggs].\footnote{The rule is codified at 42 U.S.C. \S\,2000e-2(k)(1)(A). Clause (i) makes the practice unlawful unless the employer shows it is job related for the position in question and consistent with business necessity. Clause (ii) makes it unlawful anyway where the complaining party identifies an alternative employment practice and the employer refuses to adopt it.} So the number that matters is not the small score difference, but the shortlist. In our pool of interchangeable candidates, women took {{SHORTLIST_PCT_FEMALE_PIPELINE}} of shortlist places. Men were shortlisted at a rate of {{FF_MALE_RATE}} and women at {{FF_FEMALE_RATE}}, a selection ratio of {{FF_SELECTION_RATIO}}. That is {{FF_SELECTION_RATIO_GLOSS}} of the men's rate, far under the four-fifths level federal enforcement agencies treat as a warning line [@uniformguidelines].

Courts have repeatedly declined to treat that ratio as decisive on its own. Passing it has not stopped a finding of adverse impact, and clearing it by a wide margin has not defeated one either.\footnote{Adverse impact found despite a pass rate that satisfied the four-fifths ratio [@isabel2005]; a ratio above 0.80 did not defeat a prima facie case otherwise supported by a standard-deviation analysis [@waisome1991].} The ratio is a screening trigger for enforcement agencies, not itself proof of unlawful disparate impact.

Our pool is a stylised illustration of that metric, not an applicant-flow analysis of a real employer's hiring. It shows what ranking does to a group of people the system itself cannot tell apart. Small, consistent nudges in score become a hard line between making the cut and missing it.

### The federal retreat does not remove the risk

Federal enforcement of this theory has receded. The Commission has withdrawn its guidance on algorithmic bias, and a 2025 executive order directs agencies to deprioritize disparate-impact enforcement generally, though the statute itself is unchanged [@eo14281]. Private litigation is proceeding regardless. A federal court has allowed disparate-impact claims to reach discovery against a hiring-technology vendor, not only the employer that deployed it, on the theory that the vendor acts as the employer's agent, and the claims have since reached collective certification [@mobley]. A weaker enforcement posture at the agency does not remove the underlying legal risk; if anything, it shifts more of the work onto private suits like this one.

### An employer cannot know the direction without testing

The direction of a model's default lean is a property of the individual product. A deployer therefore cannot know which sex a tool disadvantages until it is measured. Two vendors' models, both marketed as accurate, can push opposite ways on the same job. Buying on trust is not a defence, and a counterfactual test of the kind we run costs an afternoon.

Nor can the buyer learn the direction from the seller. Across all {{N_MODELS}} rerankers we found {{N_MODELS_FAIRNESS_EVAL}} published fairness evaluations of the ranking model, {{N_MODELS_GENDER_DISCLOSURE}} disclosures of a gender-related limitation, and {{N_MODELS_HIRING_NAMED}} mentions of hiring as an intended use. These are not vendors who ignore fairness. Cohere publishes a gender-bias benchmark and an explicit employment warning for its text-generating model, but no corresponding demographic benchmark for its reranker. Its company-wide employment restriction nevertheless covers the reranker.

A minority do restrict the use. {{N_MODELS_HIGHSTAKES}} of the {{N_MODELS}} models carry a restriction reaching employment decisions, and they come from just {{N_VENDORS_HIGHSTAKES}} vendors. Cohere's usage policy bars using its services for "automated determinations about individuals" in domains including employment [@cohereusage]. Google's prohibited-use policy bars automated decisions in employment that have "a material detrimental impact on individual rights without human supervision" [@googleaup]. The remaining {{N_MODELS_NO_RESTRICTION}} say nothing on the subject at all.

The location and prominence of the restriction matter. Both restrictions are stated in company-wide policies; {{N_VENDORS_RESTRICTION_IN_PRODUCT_DOC}} are stated in the ranking product text itself. Cohere's reranker documentation links directly to its Usage Policy under "Responsible Use" in the navigation. We found no direct link to Google's prohibited-use policy on the ranking page or the product's responsible-AI page. These observations distinguish a navigation link from an explicit product warning; they do not establish that a buyer could never discover the policies.

### The evenly split jobs are the sharpest case

For a job the labor market divides evenly between men and women, a business-necessity defence is hard to build on the work itself. Necessity turns on the particular practice and the evidence validating it, not on how an occupation happens to be staffed, so occupational balance does not decide the question. It does remove the most familiar answer: an employer cannot argue that work done by women and men in equal numbers requires men. Yet the shortlist for those jobs was {{SHORTLIST_PCT_MALE_PIPELINE_NEUTRAL}} male. The skew is not a feature of the work. It is a property of the software, and a different reranker skews the other way (the appendix repeats the pipeline behind every audited reranker).

### De-identification is a candidate less discriminatory alternative

Even a justified practice is unlawful if a less discriminatory alternative exists and the employer refuses it. De-identification is a candidate alternative. Our substitution rules are inexpensive to apply and can be inspected by hand. In our synthetic pairs, they remove every textual difference and therefore the score disparity. The alternative must also be equally effective at the underlying job.\footnote{42 U.S.C. \S\,2000e-2(k)(1)(A)(ii), read with \S\,2000e-2(k)(1)(C), which fixes the standard as the law stood on 4 June 1989. Under that law, cost and other burdens bear on whether a proposed alternative would be "equally as effective" in serving the employer's legitimate business goals [@watson].} Our synthetic occupation-matching test showed no loss, but it does not establish equal effectiveness in hiring. That requires validation on applicants for the same job, using independent judgments of relevant qualifications.

It is also not a technology anyone has to build. Hiring platforms already ship de-identification, and the objection that a proposed alternative would be costly or disruptive is weaker for a component the industry has already deployed. Eightfold's published responsible-AI paper is the one vendor description we found that names pronouns at all: it lists "Gender, Gender Pronouns and Titles" first among the fields it can mask [@eightfold]. So the hard part, a masker that finds gendered words in ordinary prose, exists and runs in production today.

But read where that masking is applied. Eightfold's list governs what is hidden "when displaying a candidate to the recruiter". The same paper describes the input to its own models as stripped of "names, contact information and address information". Pronouns are removed for the human reader, and the enumeration for the model stops at the name. That is the names-only condition reported above, and on our models it does not close the gap; it removes the noisier signal and leaves the more consistent one.

So the answer to "the industry already does this" is that the industry does the half we found does not work, at the stage where a human was going to see the resume anyway. The marginal ask is correspondingly small. Run the removal the vendor already performs for the recruiter on the text the models score instead, and run it at both stages. We are not asking an employer to invent a safeguard. We are asking it to move one it already owns to the point in the pipeline where the ordering is actually decided.

### Europe arrives at the same place

The EU AI Act classes systems used to filter job applications as high-risk. It warns that such systems "may perpetuate historical patterns of discrimination, for example against women" [@euaiact]. The Act's design duties fall on the vendor, but an employer that only deploys the system still owes duties of its own. It must run the system as instructed, oversee it, and keep the input data relevant to the system's purpose.\footnote{Annex III, point 4(a) and Recital 57 classify recruitment systems as high-risk. The requirements addressed to the system itself --- risk management (Art.\ 9), data governance (Art.\ 10) and human oversight (Art.\ 14) --- bind the provider, through Art.\ 16(a). The deployer's own duties are in Art.\ 26: to use the system in accordance with the instructions for use (Art.\ 26(1)); to assign human oversight to competent staff (Art.\ 26(2)); to ensure input data is relevant and sufficiently representative so far as the deployer controls it (Art.\ 26(4)); and, for an employer, to inform workers and their representatives before use (Art.\ 26(7)). Under Art.\ 113 as amended by Regulation (EU) 2026/1744, these obligations apply from 2 December 2027 [@omnibus].} A further duty to assess the effect on fundamental rights before use reaches public bodies and public-service providers, not ordinary private employers.\footnote{Art.\ 27(1), which is limited to deployers that are bodies governed by public law or private entities providing public services, together with deployers of the credit and insurance systems in Annex III points 5(b) and 5(c). Regulation (EU) 2026/1744 amended only Art.\ 27(4) and (5); the scope in Art.\ 27(1) is unchanged.} De-identification is a measure the deployer controls, applied at exactly that input boundary.

The duty to follow the instructions for use has more bite here than it first appears. For {{N_MODELS_HIGHSTAKES}} of the models, the vendor's policy restricts employment decisions. Whether a deployment conflicts with it depends on the policy's conditions, including Google's qualification about human supervision. Whether a given policy is part of the instructions for use in the Act's sense is a question we raise rather than settle. For the other {{N_MODELS_NO_RESTRICTION}}, our audit found no explicit employment restriction in the checked sources. That finding does not remove the deployer's other obligations or establish that no relevant instructions exist.

# Recommendations

Anyone deploying two-step ranking over documents that describe people should remove gender markers from the documents and from the search queries, before either step runs. The specification below is short enough for a court, a regulator, or a compliance team to require and to verify.

1. **Replace names.** Substitute every personal name with a fixed placeholder such as "the candidate", or delete it. A real system needs a name detector rather than a word list, and that detector's coverage is itself an audit item.
2. **Neutralise pronouns.** Our demonstrated method replaces every covered gendered pronoun with a single token, producing identical text for our pairs but sometimes ungrammatical prose. A grammatical alternative would need to resolve pronoun roles and repair verb agreement, then be validated on varied sentences. The heuristic tested here is not that implementation: it succeeds on our templates but can leave differences in ordinary prose.
3. **Apply it at both steps.** The reranker can override the retriever, so covering one step is not enough. Apply the same rules to the query as well as to the documents.
4. **Audit the transform, not only the model.** The substitution list is short and readable, so someone should read it. Someone should also spot-check the output for gendered words that slipped through: school names, awards, sports, honorifics.
5. **Re-audit the model's output, not only the transform's completeness.** The step above catches gendered words the substitution missed. Depending on the industry, role, and other specifics, a resume can carry gender signal through elements like school names, awards, or sports. These can often be added to the same substitution approach once identified. A smaller residual runs deeper: after removing names, pronouns, and hobbies, one study of real resumes still predicted sex about eight times in ten from writing style alone, a signal no fixed word list fully captures [@parasurama2022]. A model can learn to use whatever is left as a proxy for the attribute it can no longer see directly [@gerards2022]. Score a sample of de-identified real documents for residual skew, not only the substitution list, and repeat the check when the job, the model, or the candidate pool changes.

The guarantee is simple, and it does not require trusting the model. Once the only gender-differing words are gone, the two documents are the same text, and any ranker must score the same text the same way. That guarantee is exact for pairs built to differ only in name and pronoun, as ours are. Real resumes differ in many other ways, some of them gender-correlated, which is exactly why the fifth step above matters.

We are not claiming this is the only route. A second family of fixes changes the model rather than the text it reads. Adversarial training suppresses a protected signal while the model learns [@rekabsaz2021]. Machine unlearning removes what a model has already absorbed: one method scrubs an entire category from a trained image classifier by pruning the channels that carry it, without retraining from scratch, and its authors offer it as a step toward "compliance with legal and ethical criteria" [@wang2022]. That ambition is the same as ours, and for a firm that trains its own ranker these methods are worth having.

These methods require access to the model's weights and fresh validation after modification. {{N_MODELS_NO_PUBLIC_WEIGHTS}} of the {{N_MODELS}} audited rerankers have no publicly released weights in the vendor sources we checked. A buyer of those hosted services cannot directly modify the served model. Others do release weights: IBM Granite is one, although we accessed it through an API. Whether a modified model still carries the signal remains an empirical question. De-identification can instead be applied to the input of either a hosted or locally run model.

The two families of fixes also sit differently under the law. De-identification never encodes sex as an input; it deletes the marker before scoring happens. Adversarial training and unlearning do the opposite: they use sex as a signal precisely to suppress its effect on the output. Title VII treats a remedial step that is conscious of a protected characteristic with more suspicion than one that simply removes it from view. An employer that changes a decision because of its demographic pattern can itself face a disparate-treatment claim on that basis [@ricci2009]. That does not make adversarial fixes unlawful, but it does mean de-identification carries a legal-risk profile the alternatives do not.

### Providers should state hiring risks and restrictions

Houser discusses documenting algorithm design and auditing outcomes as safeguards against bias \cite[pp.~339, 344]{houser2019}. We extend that argument to the information a provider should give buyers of a general-purpose reranker.

Our documentation audit supports a further proposal: providers should state hiring risks and restrictions directly in the documentation for each reranker. That statement should explain whether hiring use is permitted, any conditions on that use, and the model's known demographic risks and limitations. It should link to the applicable usage policy and any relevant fairness evaluations, describing what those evaluations cover. If the model's suitability for hiring has not been evaluated, the provider should say so explicitly.

This is not a new regulatory concept. New York City already requires employers using automated hiring tools to commission an independent bias audit and publish a summary of the results [@nyc144]. A December 2025 state audit found enforcement of that requirement weak [@nyccomptroller2025], which is itself a reason to pair outcome audits with the kind of upstream disclosure we propose. Our ask is narrower and further upstream than New York's. We are not asking for an audit of outcomes after deployment, but a disclosure of what the vendor already knows before a buyer decides to deploy at all.

This disclosure would help buyers distinguish permission to use a model from evidence that it is suitable for hiring. Providers should update it when the model, evidence or policy changes. Employers would still need to validate the system on their own hiring task.

# Limitations

Our transform removes gender only from the channels it covers. Real resumes signal gender in other ways: single-sex schools, gendered awards, sports, writing style. Parasurama and Sedoc put a number on that residue. Across 709,000 real resumes, a model still told a man's resume from a woman's about eight times in ten after gender identifiers, hobbies and the gender direction in the embeddings had all been removed, and they report that what predicts sex is far more than the words that sound masculine or feminine [@parasurama2022]. Removing names and pronouns is therefore necessary but not sufficient in the field. Our documents are short synthetic templates rather than real applications.

A synthetic audit like ours is not itself a discrimination charge, and no applicant in our pool was actually harmed. Title VII litigation has nonetheless made room for evidence built this way: courts have recognized that a person who applies only to test for discriminatory treatment, without intending to accept the job, can still have standing to sue over what the test reveals [@kyles2000]. Whether a counterfactual audit of a ranking model would be treated the same way is untested. We flag the question rather than answer it.

A companion run on real résumés complicates the picture. From a corpus of {{N_REAL_CORPUS}} résumés, the sampled pools included {{N_REAL_RESUMES}} distinct résumés, scored in male and female forms. Construction combined injected names with swaps of existing gender markers, rather than injecting a name into every document. The four models' directions and magnitudes differed from the synthetic results. The documents, names and query design also differed, so this comparison cannot identify what caused the change. The released results and sampling reconstruction record the corpus size separately from the number actually scored.

Our headline figure averages over the chosen name pairs, and the robustness tables show substantial variation between pairs. A different set could give a materially different number. The intervals resample occupations, not names, and therefore do not capture uncertainty from choosing the names. A replication matching names for birth cohort, frequency and ethnicity, together with same-sex name substitutions as a control, would help separate these effects.

That residue is a legal problem as much as a technical one. Algorithms "can generate new categories of people based on seemingly innocuous characteristics", and such categories "could evade non-discrimination law" because the characteristic they key on is not itself protected [@gerards2022]. A preference for *cared* over *repaired* is exactly that kind of characteristic. De-identification closes the channels a court can read and verify, which is why we propose it; it does not close that one. An employer that adopts it should still test the outcome, not only the input.

The pipeline and de-identification results use one open-source pair as the worked example. The appendix repeats the pipeline behind every audited reranker and the de-identification test on four open-weight rerankers, and the pattern is not identical across them. Our chosen names are common American given names, but the pairs were not systematically matched for frequency, birth cohort or ethnicity. We cannot separate those name associations from gender. Since the spread across name pairs is the largest source of variation we measure, an age or ethnicity story is a live competing explanation for part of what we report, and a cohort-matched replication is the way to settle it. We treat gender as binary, which real applicants are not. The commercial models are black boxes, so we report the version strings and query dates in the appendix. One of the four open-weight rerankers did not exactly reproduce its own earlier scores on a plain re-run, a caveat noted where that table appears.

The name channel also carries race, and we do not test it. In resume retrieval, names associated with race shift outcomes about as much as names associated with sex [@wilson2024]. That is part of why our own name effects cannot be read as gender alone. It also means the reranking stage remains unaudited for race and for intersectional effects. That is the obvious next study.

The documentation audit has a matching limit. It records what {{N_RERANKER_VENDORS}} vendors publish, not what they do. A vendor may test its ranking models for demographic bias and not say so, and an absence in our table is an absence from the public record on {{VENDOR_AUDIT_DATE}}, nothing more. That is the right measure for the legal question we ask, which is what a deployer can learn before buying, but it is not a finding about anyone's internal practice. The same caution applies to the hiring platform we quote: we report the scope its own paper gives its masking, not what its systems do.

One structural point deserves care in any extension to real documents. The two stages need not read the same text. Our retriever accepts 512 tokens; our reranker accepts 8,192. On a long resume the two stages would therefore score different amounts of the same candidate, and some of the divergence we attribute to the reranker would instead be an artifact of what each stage could see. Our documents are short enough that neither model truncates anything, so the comparison here is clean. It would not be on real resumes.

# Conclusion

Rerankers give different scores to documents differing only in the chosen names and pronouns, with a pattern consistent with job stereotypes. This design does not separate gender associations from the other associations carried by names. In our synthetic pool, one common pipeline gave male-named candidates {{SHORTLIST_PCT_MALE_PIPELINE}} of shortlist places, and the reranker overrode the retriever. Removing the differing markers equalises the text and eliminates that score difference. The transform showed no loss on synthetic occupation matching. Deployers should evaluate it as an auditable mitigation and validate effectiveness on their actual hiring task.

Providers also have a role: each reranker's documentation should clearly state its hiring restrictions, known risks, and whether its suitability for hiring has been evaluated.

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

**Confidence intervals.** The counterfactual pairs are not independent, because they are built from the same occupations, names, and templates. All intervals resample whole occupations rather than individual comparisons. They do not additionally resample name pairs or templates, whose effects are large (see Robustness); the true uncertainty is therefore somewhat wider than the reported ranges. The clusters are also fewer than they look. Our {{N_OCCUPATIONS}} occupations map onto {{N_BLS_CATEGORIES}} distinct federal categories, because four pairs of them share one -- laborer and mover, nanny and babysitter, analyst and consultant, accountant and auditor. Resampling treats those pairs as two draws when they are one, which narrows the intervals further than it should.

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

Table~\ref{tab:models} lists every reranker in the audit with its vendor, how it was accessed, the date it was queried, its score range, and its tie rate. Requests to one deprecated Google model returned scores byte-identical to those of its successor, so the two are reported once, as one model. Table~\ref{tab:single} gives the three-way split for each model and job category, and Table~\ref{tab:consistency} counts how many individual occupations lean each way.

```latex
\begin{table}[H]
\centering
\caption{The rerankers audited: vendor, access route, number of counterfactual
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

# The same pipeline with every reranker

The pipeline result in the body uses one open-source pair. Table~\ref{tab:sweep} repeats the simulation with each of the audited rerankers behind the same retriever. The shortlist skew follows the reranker, in both size and direction, which is the point of the second finding.

The headline pipeline also uses a retrieval cutoff of {{TOPK}} -- how many candidates the retriever passes on before the reranker sees them. That choice is not neutral, and it is not symmetric. Narrowing the cut to 5 lowers the shortlist to {{PIPE_PCT_MALE_K5}} male, because the reranker is handed fewer candidates to reorder. Widening it to 15 changes almost nothing ({{PIPE_PCT_MALE_K15}}), since the reranker already dominates the ordering by that point. A deployer who tightens that first cut changes the disparity without touching either model.

```latex
\begin{table}[H]
\centering
\caption{Share of shortlist places going to men's names, by reranker, with
the same retriever and the same pool of equally qualified candidates.}
\label{tab:sweep}
\resizebox{\textwidth}{!}{%
{{TABLE_SWEEP}}%
}
\end{table}
```

# What the vendors publish

Table~\ref{tab:vendordocs} records the documentation audit described in §2. The population is closed: it is exactly the {{N_MODELS}} rerankers scored in this paper, grouped by their {{N_RERANKER_VENDORS}} vendors, so no vendor was selected by us for inclusion or omission. Every entry rests on a first-party source -- the vendor's own model card, product documentation, technical report, or terms and usage policy. The initial audit and follow-up corrections span {{VENDOR_AUDIT_DATE}}; individual access dates and follow-up evidence are recorded in the repository. Trade-press articles, analyst notes and review sites were not used. The repository holds the {{N_VENDOR_SOURCES}} sources with the quoted wording for each.

Two entries need their coding explained. Cohere is marked "other model" for the fairness and gender columns because it does publish a gender and race bias evaluation, on the BOLD benchmark, in the model card for its Command R text-generating models; that card does not mention the rerankers or the embedding models. IBM's reranker carries the only "Ethical Considerations and Limitations" section among the open-weight models, but it discloses training-data content filtering and the context window, not demographic behaviour; its technical report contains no fairness section, and the only occurrences of the word "fair" concern fair comparison between models.

```latex
\begin{table}[H]
\centering
\caption{What the vendors of the audited rerankers publish, as of
{{VENDOR_AUDIT_DATE}}. \emph{Hiring}: is hiring or recruitment named as an
intended use? \emph{Fairness}: is a fairness evaluation of the ranking model
published? \emph{Gender}: is a gender-related limitation disclosed?
\emph{High-stakes}: is there a restriction reaching employment decisions, and
where is it stated? "Other model" marks a vendor that publishes the item for a
different model family than the reranker audited here.}
\label{tab:vendordocs}
\resizebox{\textwidth}{!}{%
{{TABLE_VENDOR_AUDIT}}%
}
\end{table}
```

# De-identification conditions

Table~\ref{tab:deid} reports every de-identification rule tested on the same reranker, including each half of the transform on its own and the pronoun-rewrite heuristics. The full transform makes the two documents byte-identical, so its result follows by construction; it is reported as a check on the symmetry argument, not as a measurement.

```latex
\begin{table}[H]
\centering
\caption{Every de-identification rule tested on {{RERANKER_MODEL}}: the share of
counterfactual pairs favoring the man's document (M), tied, and favoring the
woman's document (F); the average size of the score gap on this model's own
scale; the share of pairs the rule leaves textually identical; and synthetic occupation-matching top-1
accuracy where measured. A dagger marks a rule that makes the two documents
identical by construction, so its row is a check on that fact rather than an
independent measurement.}
\label{tab:deid}
\resizebox{\textwidth}{!}{%
{{TABLE_DEID}}%
}
\end{table}
```

Table~\ref{tab:deidall} repeats the same eight rules on three more rerankers. The direction of the half-measures is not the same on every model: on {{RERANKER_MODEL}} and one other, removing names leaves the stronger and more one-directional signal; on a third, removing pronouns does; on the fourth, the two half-measures push toward opposite net directions entirely. What holds on all four is the part that matters for the policy: the full transform and the following-word heuristic tie every pair in these templates, on every model. This does not establish that the heuristic equalises ordinary prose.

```latex
\begin{table}[H]
\centering
\caption{The same eight de-identification conditions on all four locally run
rerankers: the tie-aware share favoring the man's document and the average
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

\bibitem{abril2025}
D.~Abril.
\newblock AI is now screening job candidates before humans ever see them.
\newblock \emph{The Washington Post}, 1 July 2025.
\url{https://www.washingtonpost.com/business/2025/06/30/virtual-recruiters-ai-jobs/};
archived at \url{https://perma.cc/D7JB-PFWF}.

\bibitem{ajunwa2021}
I.~Ajunwa.
\newblock An Auditing Imperative for Automated Hiring Systems.
\newblock \emph{Harvard Journal of Law \& Technology}, 34(2):621--700, 2021.
\url{https://jolt.law.harvard.edu/assets/articlePDFs/v34/5.-Ajunwa-An-Auditing-Imperative-for-Automated-Hiring-Systems.pdf}
(accessed 5 September 2026).

\bibitem{an2025}
J.~An, D.~Huang, C.~Lin, and M.~Tai.
\newblock Measuring gender and racial biases in large language models: Intersectional
evidence from automated resume evaluation.
\newblock \emph{PNAS Nexus}, 4(3):pgaf089, 2025. \doi{10.1093/pnasnexus/pgaf089}.

\bibitem{banks2026}
R.~R.~Banks, G.-U.~Charles, S.~Overton, and V.~Y.~Wu.
\newblock Antidiscrimination law in an AI-pervaded world.
\newblock \emph{Stanford Law Review Online}, vol.~79, September 2026.
\url{https://www.stanfordlawreview.org/online/antidiscrimination-law-in-an-ai-pervaded-world/}
(accessed 3 September 2026).

\bibitem{barocas2016}
S.~Barocas and A.~D.~Selbst.
\newblock Big Data's Disparate Impact.
\newblock \emph{California Law Review}, 104(3):671--732, 2016. \doi{10.15779/Z38BG31}.

\bibitem{behaghel2015}
L.~Behaghel, B.~Cr\'epon, and T.~Le~Barbanchon.
\newblock Unintended Effects of Anonymous R\'esum\'es.
\newblock \emph{American Economic Journal: Applied Economics}, 7(3):1--27, 2015.
\doi{10.1257/app.20140185}.

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

\bibitem{cohereusage}
Cohere.
\newblock Usage Policy, dated 21 November 2024.
\newblock \url{https://docs.cohere.com/docs/usage-policy}
(accessed 3 September 2026).

\bibitem{eightfold}
Eightfold AI.
\newblock \emph{Responsible AI at Eightfold}.
\newblock Table 1 (information capable of being masked during candidate masking) and
the "Masked Features" section.
\url{https://eightfold.ai/wp-content/uploads/Responsible_AI_at_Eightfold.pdf}
(accessed 3 September 2026).

\bibitem{eo14281}
Exec.\ Order No.\ 14,281.
\newblock Restoring Equality of Opportunity and Meritocracy.
\newblock 90 Fed.\ Reg.\ 17537 (Apr.\ 23, 2025).

\bibitem{gerards2022}
J.~Gerards and F.~Zuiderveen Borgesius.
\newblock Protected grounds and the system of non-discrimination law in the context
of algorithmic decision-making and artificial intelligence.
\newblock \emph{Colorado Technology Law Journal}, 20(1):1--55, 2022.

\bibitem{googleaup}
Google.
\newblock Generative AI Prohibited Use Policy, \S\,1.1.8.
\newblock \url{https://policies.google.com/terms/generative-ai/use-policy}
(accessed 3 September 2026).

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

\bibitem{houser2019}
K.~A.~Houser.
\newblock Can AI Solve the Diversity Problem in the Tech Industry? Mitigating
Noise and Bias in Employment Decision-Making.
\newblock \emph{Stanford Technology Law Review}, 22:290--354, 2019.
\url{https://law.stanford.edu/wp-content/uploads/2019/08/Houser_20190830_test.pdf}
(accessed 4 September 2026).

\bibitem{isabel2005}
\emph{Isabel v.\ City of Memphis}, 404 F.3d 404 (6th Cir.\ 2005).

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

\bibitem{kyles2000}
\emph{Kyles v.\ J.K.\ Guardian Sec.\ Servs., Inc.}, 222 F.3d 289 (7th Cir.\ 2000).

\bibitem{mobley}
\emph{Mobley v.\ Workday, Inc.}, No.\ 3:23-cv-00770-RFL (N.D.\ Cal.).
\newblock Order of 12 July 2024 (denying the motion to dismiss the Title~VII, ADEA, and ADA
disparate-impact claims on an agency theory); order of 16 May 2025 (conditionally certifying
a collective action on the ADEA claim).

\bibitem{nogueira2019}
R.~Nogueira and K.~Cho.
\newblock Passage Re-ranking with BERT.
\newblock arXiv preprint arXiv:1901.04085, 2019.

\bibitem{nyc144}
New York City, Local Law 144 of 2021.
\newblock Codified at N.Y.C.\ Admin.\ Code \S\S\,20-870 to 20-874.

\bibitem{nyccomptroller2025}
Office of the New York State Comptroller.
\newblock Enforcement of Local Law 144 -- Automated Employment Decision Tools.
\newblock 2 December 2025.
\url{https://www.osc.ny.gov/state-agencies/audits/2025/12/02/enforcement-local-law-144-automated-employment-decision-tools}
(accessed 5 September 2026).

\bibitem{parasurama2022}
P.~Parasurama and J.~Sedoc.
\newblock Gendered language in resumes and its implications for algorithmic bias
in hiring.
\newblock In \emph{Proceedings of the 4th Workshop on Gender Bias in Natural Language
Processing (GeBNLP)}, page 74. Association for Computational Linguistics, 2022.
\doi{10.18653/v1/2022.gebnlp-1.7}.

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

\bibitem{ricci2009}
\emph{Ricci v.\ DeStefano}, 557 U.S. 557 (2009).

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

\bibitem{waisome1991}
\emph{Waisome v.\ Port Auth.\ of N.Y.\ \& N.J.}, 948 F.2d 1370 (2d Cir.\ 1991).

\bibitem{wang2022}
J.~Wang, S.~Guo, X.~Xie, and H.~Qi.
\newblock Federated unlearning via class-discriminative pruning.
\newblock In \emph{Proceedings of the ACM Web Conference 2022 (WWW '22)},
pages 622--632. ACM, 2022. \doi{10.1145/3485447.3512222}. arXiv:2110.11794.

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
