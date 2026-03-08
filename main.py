from fastapi import FastAPI

from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder
from vector_store.faiss_store import VectorStore
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache


app = FastAPI()

print("Loading dataset...")
documents = load_dataset()

print("Embedding documents...")
embedder = Embedder()

doc_embeddings = embedder.embed_documents(documents)

dimension = len(doc_embeddings[0])

vector_store = VectorStore(dimension)
vector_store.add(doc_embeddings, documents)

cluster_model = FuzzyCluster(n_clusters=20)
cluster_model.fit(doc_embeddings)

cache = SemanticCache()


@app.post("/query")
def query_api(body: dict):

    query = body["query"]

    query_embedding = embedder.embed_query(query)

    hit, entry, similarity = cache.lookup(query_embedding)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(similarity),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    results = vector_store.search(query_embedding, k=1)

    result = results[0]

    cluster, _ = cluster_model.get_cluster_distribution(query_embedding)

    cache.add(query, query_embedding, result, cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}