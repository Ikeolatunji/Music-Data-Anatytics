**Project Overview**

This project implements a music recommendation system using data analysis, preprocessing, clustering, and similarity-based recommendations. The system is built using Python and consists of four modules:

Data Explorer – Performs data loading, visualization, and statistical analysis.

Data Preprocessor – Handles missing values, data type conversions, and data merging.

Recommendation System – Computes similarity between songs and recommends tracks.

Song Clustering – Groups songs based on features using K-Means clustering.

A main.ipynb Jupyter Notebook is included to facilitate interaction with these modules.

**Features**

Data cleaning and preprocessing

Statistical data analysis

Data visualization using Matplotlib and Seaborn

Recommendation system using cosine similarity

K-Means clustering for song grouping

****Installation**

**Prerequisites****

Ensure you have Python installed (recommended: Python 3.8 or later). You will also need the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn scipy

**Execution Instructions**

Clone this repository:

git clone <repository-url>
cd <repository-folder>

Ensure the dataset (CSV file) is placed in the correct location.

Run the Jupyter Notebook (main.ipynb) to interact with the modules:

jupyter notebook

Alternatively, execute individual Python modules as needed.

**Project Structure**

|-- main.ipynb               # Jupyter Notebook to run and test functions
|-- data_explorer.py         # Data loading, visualization, and statistics
|-- data_preprocessor.py     # Data cleaning and preprocessing
|-- recommendation_system.py # Similarity-based song recommendations
|-- song_clustering.py       # K-Means clustering for song grouping
|-- data.csv              # Music dataset (ensure correct placement)
|-- data_genre.csv        # Music dataset (ensure correct placement)

**Future Improvements**

Improve recommendation accuracy with deep learning techniques

Implement a user interface for better interaction

Expand dataset with more music features

**References**

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 56–61.

Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95.



License
This project is for educational purposes only.


**Author**
**Ikeoluwa Olatunji**

