# Vendor audit follow-up, 4 September 2026

These dated observations correct the 3 September coding. They are evidence
notes, not archived copies of the complete pages. The JSON retains each source's
individual access date; the manuscript reports the audit date range.

- [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank): the
  navigation contains **Responsible Use → Usage Policy**, directly linking to
  <https://docs.cohere.com/docs/usage-policy>. The link was followed successfully.
  The restriction is in the platform policy, not the ranking product text.
  The navigation also links the Command R model card. The previous finding of
  no policy or model-card link was incorrect.
- [Google ranking documentation](https://docs.cloud.google.com/generative-ai-app-builder/docs/ranking)
  and [responsible-AI documentation](https://docs.cloud.google.com/generative-ai-app-builder/docs/responsible-ai):
  no direct link to the Generative AI Prohibited Use Policy was found in the
  checked pages. General terms links exist. This observation says nothing about
  whether the policy can be reached by following additional links.
- [IBM Granite](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2):
  Apache-2.0 model repository with downloadable weights and local CrossEncoder
  and Transformers examples. Our API access route does not make it proprietary.
- Public model repositories for
  [BAAI](https://huggingface.co/BAAI/bge-reranker-v2-m3),
  [MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2), and
  [Jina](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
  supply weights for local inference. Jina's commercial-use licence restriction
  is separate from weight availability.
- The checked [Voyage](https://docs.voyageai.com/docs/reranker), Cohere and
  Google ranking documentation describes hosted services. No public weights
  for those audited models were found in those sources. This is a bounded
  documentation finding, not a claim that every private deployment arrangement
  is impossible.

Per-model weight availability, evidence URLs and check dates are stored in
`code/vendor_docs_audit.json`. The audit generator validates coverage against
the scored model population and counts weight availability separately from
the access route used in the experiment.
