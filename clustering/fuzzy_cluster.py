from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, n_clusters=20):

        print("Initializing GMM clustering...")

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42
        )

    def fit(self, embeddings):

        print("Training clustering model...")

        self.model.fit(embeddings)

    def get_cluster_distribution(self, embedding):

        probs = self.model.predict_proba([embedding])[0]

        dominant_cluster = int(np.argmax(probs))

        return dominant_cluster, probs