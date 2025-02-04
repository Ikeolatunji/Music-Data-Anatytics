import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class DataPreprocessor:
    def __init__(self, data, genres_data=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data.copy()
        self.genres_data = genres_data.copy() if genres_data is not None else None
    
    def handle_missing_values(self, columns=None):
        """Handles missing values by filling numeric columns with their mean (or specified columns)."""
        if columns:
            self.data[columns] = self.data[columns].fillna(self.data[columns].mean())
        else:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        return self.data
    
    def convert_dtypes(self):
        """Converts data types for consistency."""
        self.data['year'] = self.data['year'].astype(int)
        self.data['explicit'] = self.data['explicit'].astype(bool)
        return self.data
    
    def merge_genres(self):
        """Merges the main dataset with the genres dataset on 'id', if available."""
        if self.genres_data is not None and 'id' in self.data.columns and 'id' in self.genres_data.columns:
            self.data = self.data.merge(self.genres_data, on='id', how='left')
        return self.data
    
    def split_data(self, target_column, test_size=0.2):
        """Splits the dataset into training and testing sets."""
        if target_column not in self.data.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)
