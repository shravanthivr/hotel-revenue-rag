# Hotel Revenue RAG: Chunking and Retrieval Report

## Summary

This project uses a hybrid RAG pipeline for hotel revenue optimization. The corpus contains structured hotel records, local events, campaign history, and guest personas. The current implementation uses semantic record-level chunks for ingestion, dense retrieval through Pinecone, BM25 keyword retrieval over the local corpus, reciprocal rank fusion, and cross-encoder reranking.

## Chunking Strategies Compared

### Strategy 1: Fixed-size chunking

Fixed-size chunking splits text into word windows with overlap. This is useful for long narrative documents because each chunk stays within a predictable token budget. In this project, fixed-size chunking is available in `ingestion/loader.py` through `fixed_size_chunks`.

Expected strengths:

* Predictable chunk length.
* Works well for long documents.
* Easy to compare across datasets.

Expected weaknesses:

* Can split a hotel, event, or pricing fact away from its related context.
* Adds little value for short structured records.

### Strategy 2: Semantic record-level chunking

Semantic chunking keeps each structured business record together. A hotel record includes occupancy, room counts, price constraints, date range, guest segment, and nearby events in the same chunk. Campaign records keep offer, lift, discount, and price-band metadata together.

Expected strengths:

* Keeps structured revenue facts together.
* Better fit for the selected `BAAI/bge-small-en-v1.5` 384-dim embedding model.
* Reduces noisy fragments for reranking.

Expected weaknesses:

* Very long records would need additional splitting.
* Requires clean source data boundaries.

## Chunk Size and Embedding Model Choice

The project uses `BAAI/bge-small-en-v1.5`, a compact 384-dimensional embedding model. Because the model is small and the records are structured, record-level chunks are intentionally concise. Large 2,000-token chunks would dilute important signals such as hotel names, dates, occupancy, and pricing constraints. Tiny fragments would lose business context. The current semantic chunks are a practical middle ground for this dataset.

## Hybrid Retrieval Design

The retriever combines:

* Dense retrieval from Pinecone for semantic matches.
* BM25 retrieval over the local JSON corpus for exact matches such as hotel names, event names, cities, dates, and prices.
* Reciprocal rank fusion to merge dense and sparse rankings.
* Cross-encoder reranking to improve precision before generation.

This is stronger than pure dense retrieval because exact business terms matter in revenue workflows. It is also stronger than pure BM25 because users may ask semantic questions such as "what should we offer conference guests?" without matching exact wording in the source data.

## Reranking Impact

The first retrieval stage optimizes for recall by collecting broad dense and sparse candidates. Reranking then evaluates the query and candidate text together, selecting the best evidence for the final answer. This improves answer grounding because the Nebius generation call receives fewer, more relevant chunks.

Recommended manual evaluation queries:

* What campaign should Marriott Tysons Corner run for the Tech Conference?
* Which campaign had the highest occupancy lift?
* What pricing constraints apply to Westin Copley Place?
* What should Chicago offer National Sales Expo attendees?
* What amenities matter most to business travelers?

Measure before and after reranking:

* Whether the top source contains the correct hotel or event.
* Whether pricing metadata is present in the final context.
* Whether the answer cites the retrieved source.
* Whether unrelated questions trigger the confidence gate.

## Current Recommendation

Use semantic record-level chunking as the default for this project. Keep fixed-size chunking available for comparison and for future expansion if the project adds longer narrative documents such as call transcripts, contracts, or monthly revenue reports.
