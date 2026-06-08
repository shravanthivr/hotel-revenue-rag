# hotel-revenue-rag
Demo: Graph RAG for Organizational Knowledge - Langchain/Graph

# Hotel Revenue Optimization RAG

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) application for hotel revenue optimization. The system analyzes hotel occupancy forecasts, local events, guest personas, historical campaign performance, and marketing playbooks to recommend targeted promotional campaigns that can help increase hotel occupancy.

The project is built using LangChain, OpenAI Embeddings, and Pinecone Vector Database.

---

## Business Problem

Hotels frequently experience periods of low occupancy and must decide which marketing campaigns to launch to increase bookings.

Traditionally, revenue managers manually analyze:

* Occupancy forecasts
* Local events
* Guest segments
* Historical campaign performance
* Marketing best practices

This project uses RAG to retrieve relevant business knowledge and generate campaign recommendations grounded in historical data and marketing playbooks.

---

## Architecture

```text
Business Query
      │
      ▼
Retriever (Pinecone)
      │
      ▼
Relevant Campaigns
Relevant Events
Guest Personas
Marketing Playbooks
      │
      ▼
Prompt Augmentation
      │
      ▼
OpenAI LLM
      │
      ▼
Campaign Recommendation
```

---

## Data Sources

The knowledge base contains synthetic hotel marketing data:

* `campaigns.json` – Historical marketing campaigns
* `events.json` – Local events and conferences
* `personas.json` – Guest personas and preferences
* `playbooks.json` – Marketing best practices
* `occupancy_forecast.json` – Hotel occupancy forecasts

---

## Technology Stack

* Python 3.9+
* LangChain
* OpenAI Embeddings
* Pinecone Vector Database
* dotenv
* VS Code

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

## Setup

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## Running the Project

### Load Data into Pinecone

```bash
python ingestion/load_to_pinecone.py
```

### Run the Application

```bash
python app.py
```

---

## Sample Query

```text
Hotel: Marriott Tysons Corner

Forecast Occupancy: 62%

Location: Northern Virginia

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

Reasoning:
Similar campaigns performed well during large technology conferences and generated measurable occupancy improvements.
```

---

## Future Enhancements

* Hybrid RAG (Vector + Knowledge Graph)
* Neo4j Integration
* LangGraph Agent Workflows
* LangSmith Evaluation
* Real-Time Event Data Ingestion
* Campaign Performance Analytics Dashboard

---

## Learning Objectives

This project demonstrates:

* Document ingestion
* Text chunking
* Embeddings generation
* Vector storage
* Semantic retrieval
* Retrieval-Augmented Generation (RAG)
* Business-focused AI applications
