# hotel-revenue-rag
Demo: Graph RAG for Organizational Knowledge - Langchain/Graph

# Hotel Revenue Optimization RAG

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) application for hotel revenue optimization. The system analyzes hotel occupancy forecasts, local events, guest personas, historical campaign performance, and marketing playbooks to recommend targeted promotional campaigns that can help increase hotel occupancy.

The application uses LangChain, Pinecone Vector Database, and Nebius Token Factory models for embeddings and/or generation.

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
* Semantic retrieval
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
Chunking
      │
      ▼
Nebius Embeddings
      │
      ▼
Pinecone Vector Database
      │
      ▼
Retriever
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

* LangChain
* Pinecone Vector Database
* Nebius Token Factory Models
* OpenAI-compatible API integration (via Nebius)

### Development

* Python 3.9+
* VS Code
* Git

---

## Retrieval Workflow

### Ingestion

1. Load JSON business documents
2. Convert to LangChain Documents
3. Chunk text
4. Generate embeddings using Nebius models
5. Store vectors in Pinecone

### Query

1. User submits occupancy scenario
2. Retrieve relevant documents from Pinecone
3. Augment prompt with retrieved context
4. Generate campaign recommendation using Nebius LLM
5. Return recommendation with supporting evidence

---

## Project Structure

```text
hotel-revenue-rag/
│
├── data/
│   ├── campaigns.json
│   ├── events.json
│   ├── personas.json
│   ├── playbooks.json
│   └── occupancy_forecast.json
│
├── ingestion/
│   ├── load_documents.py
│   ├── chunk_documents.py
│   └── load_to_pinecone.py
│
├── retrievers/
│   └── retriever.py
│
├── chains/
│   └── rag_chain.py
│
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

Create a `.env` file:

```env
NEBIUS_API_KEY=your_nebius_api_key
NEBIUS_BASE_URL=your_nebius_endpoint
PINECONE_API_KEY=your_pinecone_api_key
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

## Cohort Requirement

This project satisfies the Week 2 requirement by using Nebius Token Factory for at least one model call (embedding generation and/or response generation) while implementing a complete Retrieval-Augmented Generation workflow.

---

## Future Enhancements

* Hybrid RAG (Graph + Vector)
* Neo4j Knowledge Graph
* LangGraph Agent Workflows
* LangSmith Evaluation
* Real-Time Event Ingestion
* Revenue Optimization Dashboard
