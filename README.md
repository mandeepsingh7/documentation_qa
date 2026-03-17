# CUDA Runtime API RAG Assistant

A **Retrieval-Augmented Generation (RAG)** system that answers questions about the **NVIDIA CUDA Runtime API** using official documentation.

The system crawls the CUDA Runtime API documentation, builds a hybrid retrieval index, and serves answers through a **FastAPI backend**.

## Overview

This project builds a **domain-specific question answering system** over the official CUDA Runtime documentation.

The pipeline includes:

- CUDA Runtime documentation crawling using **Scrapy**
- Document chunking and embedding
    - **Dense:** OpenAI Embeddings stored in ChromaDB
    - **Sparse:** BM25
- **Hybrid retrieval** (Dense + BM25)
- **Reciprocal Rank Fusion (RRF)**
- **Cohere reranking**
- **LLM-based answer generation**

The system is exposed through a **FastAPI backend** and accessed via a **web frontend**.

## Project Structure 

```text
documentation-qa/
|
├── cuda_scraper/              # Scrapes CUDA Runtime documentation 
├── data/ 
│   └── cuda_docs.json         # Scraped data       
│
├── notebooks/                 # Experimentation / testing
├── storage/
│   ├── chroma_db/             
│   └── bm25/                
│
└── src/
    ├── create_db.py           # Embeddings + BM25 index generation 
    ├── global_settings.py     
    ├── query.py               # Retrieval pipeline
    └── schemas.py             # Pydantic schemas

```




## Tech Stack

### Web Scraping 

-   Scrapy

### LLM & RAG

-   Langchain
-   ChromaDB
-   BM25 (rank_bm25)
-   Cohere Rerank
-   OpenAI API 

### Backend

-   FastAPI
-   Docker

### Infrastructure

-   AWS EC2
-   Vercel

## Setup

### Install dependencies
```sh
pip install -r requirements.txt
```
### Create `.env`
```
OPENAI_API_KEY=your_key
COHERE_API_KEY=your_key
```
### Crawl documentation
```sh
cd cuda_scraper
scrapy crawl cuda_docs -O ../data/cuda_docs.json
```
### Build Retriever Index
```sh
cd .. 
python src/create_db.py

```

## Usage 
This repository handles data collection and retrieval pipeline setup for the CUDA Runtime RAG system. 

API and frontend code are maintained in separate repositories that serve as centralized backend and frontend for all projects.



### Backend ─ Projects Hub

https://github.com/mandeepsingh7/projects-hub

- Centralized backend for all ML/AI projects 
- Contains FastAPI endpoints for multiple projects, including the CUDA Runtime RAG system 
- Handles :
  - Inference pipelines
  - Routing 

**Deployment:** AWS EC2

### Frontend ─ Portfolio

https://github.com/mandeepsingh7/portfolio

- Unified frontend for all projects
- Provides UI to interact with different APIs from Projects Hub 
- Handles :
  - User Interface 
  - API Integration 

**Deployment:** Vercel 

## API Example

### Get Random Sample

**Endpoint**

```
GET /cuda-runtime-rag-assistant/random-sample
```

**Response**
```json
{
  "query": "What information is stored in a cudaChannelFormatDesc structure?"
}
```

### RAG Query

**Endpoint**

```
POST /cuda-runtime-rag-assistant/ask/
```
**Request**
```json
{
  "query": "What information is stored in a cudaChannelFormatDesc structure?"
}
```
**Response**
```json
{
  "response": "The cudaChannelFormatDesc structure stores the following information:\n\n- int x\n- int y\n- int z\n- int w\n- enum cudaChannelFormatKind f\n\nWhere cudaChannelFormatKind is one of cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or cudaChannelFormatKindFloat.",
  "urls": [
    "https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaChannelFormatDesc.html",
    "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html",
    "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html",
    "https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaEglPlaneDesc.html",
    "https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaEglFrame.html"
  ]
}
```


## Demo

Accessible through the portfolio:

https://mandeeps.in
