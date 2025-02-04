import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import random

class SongClustering:
    def __init__(self, df, feature_columns, name_column="artists", n_clusters=3):
        """
        Initialize the SongClustering class.

        :param df: Pandas DataFrame containing song data.
        :param feature_columns: List of feature columns to use for clustering.
        :param name_column: Column name for artist names (default: 'artists').
        :param n_clusters: Number of clusters for KMeans (default: 3).
        """
        self.df = df.copy()  # Keep full dataset for recommendations
        self.data = df[feature_columns].copy()  # Only feature columns for clustering
        self.name_column = name_column
        self.n_clusters = n_clusters

    def split_features(self, test_size=0.2, random_state=42):
        """
        Splits feature data into training and testing sets.
        """
        return train_test_split(self.data, test_size=test_size, random_state=random_state)

    def create_clusters(self):
        """
        Applies KMeans clustering using selected features.
        Assigns clusters to the full dataset (`self.df`).
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=100, n_init=10)
        self.df["cluster"] = kmeans.fit_predict(self.data)  # Use feature data for clustering
        self.data["cluster"] = self.df["cluster"]  # Add cluster labels to feature DataFrame

    def visualize_clusters(self, based_on=["popularity", "tempo"]):
        """
        Visualizes clusters using a scatter plot.

        :param based_on: List containing two feature names for visualization.
        """
        if len(based_on) != 2:
            raise ValueError("based_on should contain exactly two feature names.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[based_on[0]], y=self.df[based_on[1]], 
                        hue=self.df['cluster'], palette='viridis', alpha=0.7)
        plt.title("K-Means Clustering Visualization")
        plt.show()

    def get_recommendations_by_cluster(self, artist_name, n=5):
        """
        Recommends songs from the same cluster as the given artist.

        :param artist_name: Artist name for whom to find recommendations.
        :param n: Number of recommendations to return (default: 5).
        """
        target_index = self.get_random_artist_index_by_name(artist_name)
       
        
        if target_index is None:
            return None

        target_cluster = self.df.iloc[target_index]["cluster"]
        recommended_songs = self.df[self.df["cluster"] == target_cluster]

        # Filter based on popularity range
        target_popularity = self.df.iloc[target_index]["popularity"]
        filtered = recommended_songs[
            (recommended_songs["popularity"] >= target_popularity - 10) &
            (recommended_songs["popularity"] <= target_popularity + 10)
        ]

        # If not enough recommendations, add extra
        if len(filtered) < n:
            additional = recommended_songs[~recommended_songs.index.isin(filtered.index)]
            filtered = pd.concat([filtered, additional.head(n - len(filtered))])
            
        return filtered.head(n)
    
    def get_random_artist_index_by_name(self, artist_name):
        """
        Finds an index of a random song by a given artist.
        """
        # Use self.df since self.data does not contain the 'artists' column
        artists=  self.df[self.name_column].tolist()

        indexes = []

        # find all indexes of artists that contain the name
        for x in range(len(artists)):
            if artist_name.lower() in str(artists[x]).lower():
                indexes.append(x)

        # if no matches found, return None
        if len(indexes) == 0:
            return None
        else:
            # return a random index from the found indexes
            return indexes[random.randint(0, len(indexes))]

