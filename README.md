
Story Arc-HOtel RAG
<img width="3420" height="1776" alt="HotelPage1-RAG" src="https://github.com/user-attachments/assets/ba13b6b8-e5f9-467f-b690-7022215292fd" />
<img width="3420" height="1776" alt="HotelPage2-RAG" src="https://github.com/user-attachments/assets/3154da5c-7abd-4e72-a90c-5fb3fa78ed59" />
# hotel-revenue-rag
Demo: Graph RAG for Organizational Knowledge - Langchain/Graph

# Hotel Revenue Optimization RAG

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) application for hotel revenue optimization. The system analyzes hotel occupancy forecasts, local events, guest personas, historical campaign performance, and marketing playbooks to recommend targeted promotional campaigns that can help increase hotel occupancy.

The application uses Pinecone Vector Database, a local dimension-matched embedding model, hybrid retrieval, reranking, and Nebius Token Factory for grounded answer generation.

---

## Business Problem

Hotels frequently experience periods of low occupancy and must determine which marketing campaigns to launch to increase bookings.

Revenue managers typically analyze:

* Occupancy forecasts
* Local events and conferences
* Guest segments and preferences
* Historical campaign performance
* Marketing best practices

This project uses RAG to retrieve relevant business knowledge and generate evidence-based campaign recommendations.

---

## Learning Objectives

This project demonstrates:

* Document ingestion
* Text chunking
* Embedding generation
* Vector database storage
* Hybrid dense and BM25 retrieval
* Cross-encoder reranking
* Retrieval-Augmented Generation (RAG)
* Business-focused AI recommendations
* Nebius Token Factory integration

---

## Architecture

```text
Synthetic Business Data
      │
      ▼
Document Ingestion
      │
      ▼
Semantic record-level chunking
      │
      ▼
Local BGE-small embeddings
      │
      ▼
Pinecone Vector Database
      │
      ▼
Hybrid Retriever + Reranker
      │
      ▼
Context Augmentation
      │
      ▼
Nebius LLM
      │
      ▼
Campaign Recommendation
```

---

## Data Sources

The knowledge base consists of synthetic hotel marketing data:

### Campaign History

Historical promotional campaigns and occupancy lift metrics.

### Local Events

Conferences, concerts, festivals, and sporting events.

### Guest Personas

Traveler segments, preferences, and booking behaviors.

### Marketing Playbooks

Best practices and campaign strategies.

### Occupancy Forecasts

Projected occupancy levels used as input for recommendations.

---

## Technology Stack

### AI & RAG

* Pinecone Vector Database
* SentenceTransformers embeddings and reranker
* BM25 keyword retrieval
* Nebius Token Factory generation model
* OpenAI-compatible API integration (via Nebius)
* Streamlit UI

### Development

* Python 3.9+
* VS Code
* Git

---

## Retrieval Workflow

### Ingestion

1. Load JSON business documents
2. Convert to LangChain Documents
3. Build semantic record-level chunks
4. Generate local BGE-small embeddings
5. Store vectors in Pinecone

### Query

1. User submits occupancy scenario
2. Retrieve relevant documents from Pinecone
3. Retrieve dense candidates from Pinecone
4. Retrieve sparse candidates with BM25
5. Merge results with reciprocal rank fusion
6. Rerank top candidates with a cross-encoder
7. Generate campaign recommendation using Nebius LLM
8. Return recommendation with supporting evidence

---

## Project Structure

```text
hotel-revenue-rag/
│
├── data/
│   └── rag_data.json
│
├── ingestion/
│   ├── loader.py
│   └── query.py
│
├── retrievers/
│   └── retriever.py
│
├── chains/
│   └── rag_chain.py
│
├── prompts/
│   └── campaigns_prompt.txt
├── reports/
│   └── chunking_retrieval_report.md
├── app.py
├── requirements.txt
└── README.md
```

---

## Environment Variables

Create a `.env` file:

```env
NEBIUS_API_KEY=your_nebius_api_key
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/
NEBIUS_GEN_MODEL=meta-llama/Llama-3.3-70B-Instruct
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=hospitality-rag
```

---

## Sample Query

```text
Hotel: Marriott Tysons Corner

Forecast Occupancy: 62%

Location: Northern Virginia

Question:
What marketing campaign should we launch next weekend to improve occupancy?
```

---

## Sample Response

```text
Recommended Campaign:
Tech Conference Stay Package

Target Segment:
Business Travelers

Offer:
- Complimentary breakfast
- Late checkout
- Shuttle service

Expected Occupancy Lift:
10-15%

Supporting Evidence:
Retrieved from conference marketing playbooks,
historical campaign data, and business traveler personas.
```
---

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Load the Pinecone index:

```bash
python -m ingestion.loader
```

Start the UI:

```bash
streamlit run app.py
```

## Future Enhancements

* Hybrid RAG (Graph + Vector)
* Neo4j Knowledge Graph
* LangGraph Agent Workflows
* LangSmith Evaluation
* Real-Time Event Ingestion
* Revenue Optimization Dashboard
