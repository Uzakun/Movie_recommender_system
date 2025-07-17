import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import zipfile
import os
import warnings

warnings.filterwarnings('ignore')

def download_and_load_movielens_data():
    """Download and extract MovieLens dataset and return movies DataFrame."""
    if not os.path.exists('ml-latest-small'):
        print("Downloading MovieLens dataset...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        response = requests.get(url)
        with open('ml-latest-small.zip', 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove('ml-latest-small.zip')
        print("MovieLens dataset downloaded and extracted.")
    
    movies = pd.read_csv('ml-latest-small/movies.csv')
    return movies

def main():
    movies = download_and_load_movielens_data()

    print("Preparing content features and calculating similarity matrix...")
    movies_processed = movies.copy() # Work on a copy to ensure original movies DF for display is clean if needed
    movies_processed['genres'] = movies_processed['genres'].fillna('(no genres listed)')
    movies_processed['genres'] = movies_processed['genres'].str.replace('|', ' ', regex=False)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_processed['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype('float32') # Crucial for memory

    # Save the pre-computed data
    np.save('cosine_sim_matrix.npy', cosine_sim)
    movies_processed.to_pickle('movies_df_processed.pkl') # Save processed DataFrame

    print("Pre-computation complete!")
    print(f"Saved 'cosine_sim_matrix.npy' (shape: {cosine_sim.shape}, dtype: {cosine_sim.dtype})")
    print(f"Saved 'movies_df_processed.pkl' (shape: {movies_processed.shape})")

if __name__ == "__main__":
    main()