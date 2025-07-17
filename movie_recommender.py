import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# Download MovieLens dataset
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

# Content-based recommender
class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._prepare_content_features()
    
    def _prepare_content_features(self):
        """Prepare content features for recommendation"""
        # Clean and prepare genres
        self.movies_df['genres'] = self.movies_df['genres'].fillna('(no genres listed)')
        self.movies_df['genres'] = self.movies_df['genres'].str.replace('|', ' ', regex=False) # Use regex=False for literal replacement
        
        # Create TF-IDF matrix based on genres
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    
    def get_recommendations(self, movie_title, n=5):
        """Get movie recommendations based on content similarity"""
        try:
            # Find the exact movie
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
                # Get popular movies with genres
                movies_with_genres = self.movies_df[
                    (self.movies_df['genres'] != '(no genres listed)') & 
                    (self.movies_df.index != idx)
                ]
                # To avoid errors if there are not enough popular movies, sample at most available
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


# Initialize recommender (cached)
@st.cache_resource
def initialize_recommender(movies_df):
    return ContentBasedRecommender(movies_df)

# Streamlit UI
def main():
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")
    
    # Download data
    try:
        movies, ratings = download_movielens_data()
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return
    
    # Initialize recommender
    content_recommender = initialize_recommender(movies)
    
    st.header("📚 Content-Based Movie Recommendations")
    st.write("Get recommendations based on movie genres and features")
    
    # Movie selection
    movie_title = st.selectbox(
        "Select or type a movie title:",
        options=movies['title'].tolist(),
    )
    
    if st.button("Get Recommendations", key="content"):
        with st.spinner("Finding similar movies..."):
            recommendations, message = content_recommender.get_recommendations(movie_title, n=5)
            
            if message:
                st.info(message)
            
            if not recommendations.empty:
                st.subheader(f"Top 5 movie recommendations:")
                
                # Show selected movie info
                selected_movie = movies[movies['title'] == movie_title].iloc[0]
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
                st.error("Unable to generate recommendations. Please try another movie or check dataset.")
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies", len(movies))
        with col2:
            st.metric("Total Users", ratings['userId'].nunique())
        with col3:
            st.metric("Total Ratings", len(ratings))
        
        st.write("**Genres Distribution:**")
        # Ensure 'genres' column is processed before splitting for distribution
        movies['genres'] = movies['genres'].fillna('(no genres listed)')
        movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
        all_genres = movies['genres'].str.split(' ').explode() # Use space as delimiter after replace
        genre_counts = all_genres[all_genres != '(no genres listed)'].value_counts().head(10) # Exclude 'no genres listed' for chart
        st.bar_chart(genre_counts)

if __name__ == "__main__":
    main()