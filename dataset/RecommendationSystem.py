import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecommendationSystem:
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns
    

    def compute_similarity(self, target_row, chunkSize=None):
        """Computes similarity between songs based on selected features."""
        
        features = self.data[self.feature_columns].head(chunkSize)

        # Normalize the data
        normalized_data = normalize(features)

        # Convert features to sparse matrix (if necessary)
        sparse_matrix = csr_matrix(normalized_data)

        # Ensure we are using float32 for memory efficiency
        data = sparse_matrix.astype(np.float32)

        # Ensure chunk_data is always defined
        chunk_data = data  # Default to the full dataset in case chunkSize condition is not met

        if chunkSize and data.shape[0] > chunkSize:
            start_idx = (target_row // chunkSize) * chunkSize
            end_idx = min(start_idx + chunkSize, data.shape[0])
            chunk_data = data[start_idx:end_idx]

            target_vector = data[target_row]  # Extract the target row

            # Compute similarity only for the chunk
            similarity_scores = cosine_similarity(chunk_data, target_vector, dense_output=False).toarray().flatten()
        else:
            # Compute similarity for the entire dataset
            target_vector = data[target_row]
            similarity_scores = cosine_similarity(chunk_data, target_vector, dense_output=False).toarray().flatten()

        return similarity_scores

    
    def recommend(self, song_index, top_n=5):
        """Recommends top N similar songs."""
        similarity_matrix = self.compute_similarity(song_index)

        top_n_indices = np.argpartition(similarity_matrix, -top_n)[-top_n:]
        top_n_indices = top_n_indices[np.argsort(similarity_matrix[top_n_indices])[::-1]]
        similar_songs = np.argsort(similarity_matrix[top_n_indices])[::-1]
        return self.data.iloc[top_n_indices]