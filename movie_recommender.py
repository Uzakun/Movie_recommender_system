import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # Still needed if you want to inspect genres
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import zipfile
import os
import warnings
import numpy as np # Import numpy for loading .npy
warnings.filterwarnings('ignore')

# Download MovieLens dataset (still needed for movies.csv, but not for its processing if precomputed)
@st.cache_data
def download_movielens_data():
    """Download and extract MovieLens dataset"""
    if not os.path.exists('ml-latest-small'):
        with st.spinner('Downloading MovieLens dataset...'):
            url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
            response = requests.get(url)
            with open('ml-latest-small.zip', 'wb') as f:
                f.write(response.content)
            
            # Extract zip file
            with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove('ml-latest-small.zip')
    
    # Load data
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    return movies, ratings

# Cached function to load pre-computed data
@st.cache_resource
def load_precomputed_data():
    """Load pre-computed cosine similarity matrix and movies DataFrame."""
    try:
        # Explicitly set allow_pickle=True for np.load()
        cosine_sim = np.load('cosine_sim_matrix.npy')
        movies_df = pd.read_pickle('movies_df_processed.pkl') # This line is fine
        return cosine_sim, movies_df
    except FileNotFoundError:
        st.error("Pre-computed data files not found. Please run the precomputation script (`precompute_data.py`).")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        st.error(f"Error loading pre-computed data: {e}")
        st.stop()

# Content-based recommender (simplified)
class ContentBasedRecommender:
    def __init__(self, movies_df, cosine_sim_matrix):
        self.movies_df = movies_df
        self.cosine_sim = cosine_sim_matrix
        # No need to call _prepare_content_features anymore

    def get_recommendations(self, movie_title, n=5):
        try:
            # The movies_df passed to init should already have its genres processed ('|' to ' ')
            idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Filter out movies with no genres and the movie itself
            valid_recommendations = []
            for movie_idx, score in sim_scores[1:]:  
                if self.movies_df.iloc[movie_idx]['genres'] != '(no genres listed)' and score > 0:
                    valid_recommendations.append(movie_idx)
                if len(valid_recommendations) >= n:
                    break
            
            # If we don't have enough similar movies, add popular ones
            if len(valid_recommendations) < n:
                movies_with_genres = self.movies_df[
                    (self.movies_df['genres'] != '(no genres listed)') & 
                    (self.movies_df.index != idx)
                ]
                num_to_add = min(n - len(valid_recommendations), len(movies_with_genres))
                if num_to_add > 0:
                    popular_movies = movies_with_genres.sample(n=num_to_add)
                    valid_recommendations.extend(popular_movies.index.tolist())
            
            recommendations = self.movies_df.iloc[valid_recommendations[:n]][['movieId', 'title', 'genres']]
            return recommendations, None
            
        except IndexError: # More specific exception for movie not found
            movies_with_genres = self.movies_df[self.movies_df['genres'] != '(no genres listed)']
            if len(movies_with_genres) >= n:
                return movies_with_genres.sample(n=n)[['movieId', 'title', 'genres']], "Movie not found. Showing popular movie recommendations."
            else:
                return pd.DataFrame(), "Not enough movies with genre information."
        except Exception as e:
            # Catch other potential errors gracefully
            st.error(f"An unexpected error occurred: {e}. Showing popular movie recommendations as a fallback.")
            movies_with_genres = self.movies_df[self.movies_df['genres'] != '(no genres listed)']
            if len(movies_with_genres) >= n:
                return movies_with_genres.sample(n=n)[['movieId', 'title', 'genres']], None
            else:
                return pd.DataFrame(), "Not enough movies with genre information."


# Streamlit UI
def main():
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")
    
    # Download raw MovieLens data (for initial display and metrics, not for recommender processing)
    try:
        raw_movies, ratings = download_movielens_data()
        st.success("✅ MovieLens dataset (raw) loaded successfully for metrics!")
    except Exception as e:
        st.error(f"Error loading raw MovieLens dataset: {str(e)}")
        return
    
    # Load pre-computed data
    st.write("Loading pre-computed model data...")
    cosine_sim, precomputed_movies_df = load_precomputed_data()
    st.success("✅ Pre-computed model data loaded!")

    # Initialize recommender
    content_recommender = ContentBasedRecommender(precomputed_movies_df, cosine_sim)
    
    st.header("📚 Content-Based Movie Recommendations")
    st.write("Get recommendations based on movie genres and features")
    
    # Movie selection uses the titles from the precomputed_movies_df
    movie_title = st.selectbox(
        "Select or type a movie title:",
        options=precomputed_movies_df['title'].tolist(), # Use precomputed titles
    )
    
    if st.button("Get Recommendations", key="content"):
        with st.spinner("Finding similar movies..."):
            recommendations, message = content_recommender.get_recommendations(movie_title, n=5)
            
            if message:
                st.info(message)
            
            if not recommendations.empty:
                st.subheader(f"Top 5 movie recommendations:")
                
                # Show selected movie info
                selected_movie = precomputed_movies_df[precomputed_movies_df['title'] == movie_title].iloc[0]
                st.write(f"**Based on:** {movie_title}")
                st.write(f"**Genres:** {selected_movie['genres']}")
                st.markdown("---")
                
                # Show recommendations
                for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{idx}. **{row['title']}**")
                    with col2:
                        st.write(f"*{row['genres']}*")
            else:
                st.error("Unable to generate recommendations. Please try another movie or ensure pre-computed data is valid.")
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies", len(raw_movies)) # Use raw_movies for total count
        with col2:
            st.metric("Total Users", ratings['userId'].nunique())
        with col3:
            st.metric("Total Ratings", len(ratings))
        
        st.write("**Genres Distribution:**")
        # Ensure 'genres' column is processed before splitting for distribution
        # The precomputed_movies_df already has genres processed
        all_genres = precomputed_movies_df['genres'].str.split(' ').explode() # Use space as delimiter after replace
        genre_counts = all_genres[all_genres != '(no genres listed)'].value_counts().head(10) # Exclude 'no genres listed' for chart
        st.bar_chart(genre_counts)

if __name__ == "__main__":
    main()