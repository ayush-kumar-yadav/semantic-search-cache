# Semantic Search System

## Features

- Vector embeddings using SentenceTransformers
- FAISS vector database
- Fuzzy clustering using Gaussian Mixture Models
- Custom semantic cache (no Redis)
- FastAPI API service

## Run the project

1. Create virtual environment
2. Install dependencies

pip install -r requirements.txt

3. Start server

uvicorn main:app --reload

## Cluster Analysis

The Gaussian Mixture Model produced semantically meaningful clusters
from the 20 Newsgroups dataset.

Example clusters discovered:

Cluster 6 – Sports (Hockey discussions)
Example posts mention NHL teams such as the Penguins, Oilers, and Bruins,
and discussions about playoff coverage and game broadcasts.

Cluster 0 – Computer Hardware
Posts include discussions about video cards, motherboards, VESA local bus,
and CAD workstation setups.

Cluster 7 – Computer Peripherals
Topics include SCSI controllers, communication drivers, and printing
between Macintosh systems.

Cluster 1 – Religion
Posts include Christian theology, biblical references, and discussions
about scripture interpretation.

Cluster 10 – Politics / International Affairs
Posts reference geopolitical conflicts and international political topics.

These clusters demonstrate that the embedding + GMM pipeline successfully
captures semantic relationships within the dataset.

Query
 ↓
Embedding Model (MiniLM)
 ↓
Cluster Detection (GMM)
 ↓
Semantic Cache
 ↓
Vector Search (FAISS)
 ↓
Response