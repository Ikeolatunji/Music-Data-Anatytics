import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class DataExplorer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        """Loads the dataset from a CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def _check_data_loaded(self):
        """Helper method to ensure data is loaded before processing."""
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

    def check_missing_values(self):
        """Checks for missing values in the dataset."""
        self._check_data_loaded()
        return self.data.isnull().sum()
    
    def describe_data(self):
        """Returns statistical description of the dataset."""
        self._check_data_loaded()
        return self.data.describe()
    
    def visualize_distribution(self, column):
        """Plots the distribution of a given column."""
        self._check_data_loaded()
        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")
        
        plt.hist(self.data[column].dropna(), bins=20, edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
    
    def bar_plot(self, column):
        """Creates a bar plot for categorical variables."""
        self._check_data_loaded()
        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")
        
        plt.figure(figsize=(10, 5))
        sns.countplot(x=self.data[column], palette='viridis')
        plt.title(f'Bar Plot of {column}')
        plt.xticks(rotation=45)
        plt.show()
    
    def box_plot(self, column):
        """Creates a box plot for numerical variables."""
        self._check_data_loaded()
        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not found in dataset.")
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(y=self.data[column], palette='coolwarm')
        plt.title(f'Box Plot of {column}')
        plt.show()
    
    def scatter_plot(self, x_col, y_col):
        """Creates a scatter plot to find relationships between two numerical variables."""
        self._check_data_loaded()
        if x_col not in self.data.columns or y_col not in self.data.columns:
            raise KeyError(f"Columns '{x_col}' or '{y_col}' not found in dataset.")
        
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.data[x_col], y=self.data[y_col], alpha=0.7)
        plt.title(f'Scatter Plot of {x_col} vs {y_col}')
        plt.show()

class AdditionalStatistics:
    """Provides additional statistical measures for a given dataset."""
    
    def __init__(self, df):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not loaded.")
        self.df = df

    def _check_column_exists(self, column_name):
        """Helper method to check if column exists."""
        if column_name not in self.df.columns:
            raise KeyError(f"Column '{column_name}' not found in dataset.")
    
    def mean(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].mean()

    def median(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].median()

    def std(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].std()

    def variance(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].var()

    def minimum(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].min()

    def maximum(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].max()

    def skewness(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].skew()

    def kurtosis(self, column_name):
        self._check_column_exists(column_name)
        return self.df[column_name].kurtosis()
