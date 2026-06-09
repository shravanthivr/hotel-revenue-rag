Build a LangChain RAG application for hotel occupancy optimization.

Requirements:

- Python
- LangChain
- Pinecone
- Local BGE-small embeddings
- Pinecone dense retrieval
- BM25 sparse retrieval
- Cross-encoder reranking
- Nebius Token Factory chat model

Data Sources:
- data/rag_data.json

Features:

1. Load JSON documents
2. Build semantic record-level chunks
3. Create dimension-matched embeddings
4. Store in Pinecone
5. Create hybrid retriever
6. Rerank retrieved candidates
7. Generate campaign recommendations with Nebius
8. Provide a Streamlit query UI
9. Include a chunking and retrieval comparison report

Project Structure:

hotel-revenue-rag/
├── data/
├── ingestion/
├── retrievers/
├── chains/
├── prompts/
├── app.py
├── requirements.txt
└── README.md
