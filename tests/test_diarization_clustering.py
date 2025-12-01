
import unittest
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

class TestDiarizationClustering(unittest.TestCase):
    def generate_embeddings(self, num_speakers=5, segments_per_speaker=10, noise_level=0.1):
        """
        Generate synthetic embeddings for testing clustering.
        """
        np.random.seed(42)
        embeddings = []
        true_labels = []
        
        # Generate random centroids on a unit sphere
        centroids = np.random.randn(num_speakers, 192)
        centroids = normalize(centroids)
        
        for i in range(num_speakers):
            # Generate points around the centroid
            for _ in range(segments_per_speaker):
                noise = np.random.randn(192) * noise_level
                point = centroids[i] + noise
                embeddings.append(point)
                true_labels.append(i)
                
        embeddings = np.array(embeddings)
        embeddings = normalize(embeddings) # SpeechBrain embeddings are usually normalized
        return embeddings, true_labels

    def test_default_clustering_logic(self):
        """
        Test the new default logic: Cosine + Average + Threshold 0.8
        """
        X, true_labels = self.generate_embeddings(num_speakers=5, segments_per_speaker=20, noise_level=0.1)
        
        # This mimics what app/diarization.py now does by default
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0.8, 
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(X)
        n_clusters = len(set(labels))
        
        try:
            self.assertEqual(n_clusters, 5, f"Default Logic Failed: Expected 5 clusters, got {n_clusters}")
        except AssertionError as e:
            with open("test_output.txt", "w") as f:
                f.write(str(e))
            raise e

if __name__ == "__main__":
    unittest.main()
