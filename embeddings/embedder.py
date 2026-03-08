from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, documents):

        embeddings = self.model.encode(
            documents,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, query):

        embedding = self.model.encode([query])[0]

        return embedding