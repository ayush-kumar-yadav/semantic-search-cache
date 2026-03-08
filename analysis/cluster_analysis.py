from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder
from clustering.fuzzy_cluster import FuzzyCluster
import numpy as np


documents = load_dataset()

embedder = Embedder()
embeddings = embedder.embed_documents(documents)

cluster_model = FuzzyCluster(n_clusters=20)
cluster_model.fit(embeddings)

clusters = {}

for i, emb in enumerate(embeddings):

    dominant, probs = cluster_model.get_cluster_distribution(emb)

    if dominant not in clusters:
        clusters[dominant] = []

    clusters[dominant].append((documents[i], probs[dominant]))


print("\nCLUSTER ANALYSIS\n")

for cluster_id in clusters:

    print(f"\nCluster {cluster_id}")

    samples = clusters[cluster_id][:5]

    for doc, prob in samples:
        print(f"Probability: {prob:.3f}")
        print(doc[:200])
        print("------")