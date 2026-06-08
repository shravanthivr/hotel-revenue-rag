Build a LangChain RAG application for hotel occupancy optimization.

Requirements:

- Python
- LangChain
- Pinecone
- OpenAI Embeddings
- OpenAI Chat Models

Data Sources:
- campaigns.json
- events.json
- personas.json
- playbooks.json

Features:

1. Load JSON documents
2. Chunk documents
3. Create embeddings
4. Store in Pinecone
5. Create retriever
6. Build RAG chain
7. Generate campaign recommendations

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