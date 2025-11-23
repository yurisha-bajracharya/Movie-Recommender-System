"""
Movie Recommender using Word Embeddings
========================================
This project implements a movie recommendation system using word embeddings 
from gensim and cosine similarity metrics.

Author: Student Assignment
Dataset: The Movies Dataset from Kaggle
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import time

# Download required NLTK resources
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)

# ==================== DATA LOADING & PREPROCESSING ====================

def load_and_prepare_data(metadata_path, keywords_path):
    """Load and merge movie metadata and keywords datasets."""
    print("Loading datasets...")
    df1 = pd.read_csv(metadata_path)
    df2 = pd.read_csv(keywords_path)
    
    # Remove rows with invalid IDs
    invalid_indices = df1[df1.id.str.contains(r'\d{4}-\d{2}-\d{2}', 
                                              regex=True, na=False)].index.tolist()
    df1 = df1.drop(invalid_indices)
    df2 = df2.drop(invalid_indices)
    
    # Convert id to int64
    df1['id'] = df1['id'].astype('int64')
    
    # Merge dataframes
    df = df1.merge(df2, on='id')
    
    # Drop unnecessary columns
    columns_to_drop = ['adult', 'belongs_to_collection', 'budget', 'homepage',
                      'imdb_id', 'id', 'original_title', 'release_date',
                      'poster_path', 'production_countries', 'popularity',
                      'revenue', 'runtime', 'spoken_languages', 'status',
                      'video', 'vote_average', 'vote_count']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    print(f"Dataset prepared. Total movies: {len(df)}")
    return df

# ==================== TEXT PREPROCESSING FUNCTIONS ====================

def make_lower_case(text):
    """Convert text to lowercase."""
    return text.lower()

def remove_stop_words(text):
    """Remove English stop words from text."""
    words = text.split()
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in words if word not in stop_words])

def remove_blacklist_words(text):
    """Remove blacklisted words from text."""
    words = text.split()
    blacklist = ["id", "name", "nan"]
    return ' '.join([word for word in words if word not in blacklist])

def remove_numbers(text):
    """Remove all numeric characters from text."""
    return re.sub(r'[0-9]', '', text)

def remove_punctuation(text):
    """Remove punctuation from text."""
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

def preprocess_text(text):
    """Apply all preprocessing steps to text."""
    text = make_lower_case(text)
    text = remove_stop_words(text)
    text = remove_punctuation(text)
    text = remove_blacklist_words(text)
    text = remove_numbers(text)
    return text

# ====================  CONCATENATE DESCRIPTIONS ====================

def create_concat_description(df):
    """Concatenate all movie description fields."""
    print("Creating concatenated descriptions...")
    df['concat_description'] = (
        df['keywords'].astype(str) + ' ' +
        df['genres'].astype(str) + ' ' +
        df['original_language'].astype(str) + ' ' +
        df['production_companies'].astype(str) + ' ' +
        df['tagline'].astype(str) + ' ' +
        df['overview'].astype(str)
    )
    return df

# ====================  PREPROCESSING DATASET ====================

def preprocess_descriptions(df):
    """Preprocess all movie descriptions."""
    print("Preprocessing descriptions...")
    start_time = time.time()
    
    df['description'] = df['concat_description'].apply(preprocess_text)
    df = df.drop(['concat_description', 'keywords', 'genres', 'original_language',
                  'production_companies', 'tagline', 'overview'], axis=1)
    
    elapsed = time.time() - start_time
    print(f"Preprocessing completed in {elapsed:.2f} seconds")
    
    return df

# ====================  WORD EMBEDDINGS ====================

def train_word2vec_model(df):
    """Train Word2Vec model on preprocessed descriptions."""
    print("Training Word2Vec model...")
    start_time = time.time()
    
    # Create sentences (list of list of words)
    sentences = [desc.split() for desc in df['description']]
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        sg=1,
        vector_size=300,
        window=10,
        min_count=3,
        seed=14
    )
    
    elapsed = time.time() - start_time
    print(f"Word2Vec model trained in {elapsed:.2f} seconds")
    print(f"Vocabulary size: {len(model.wv)}")
    
    return model, sentences

# ====================  AVERAGING WORD EMBEDDINGS ====================

def avg_desc_vector(description, model):
    """Calculate average word embedding vector for a description."""
    words = description.split()
    sum_vec = np.zeros(300)
    num_words = 0
    
    for word in words:
        if word in model.wv:
            sum_vec += model.wv[word]
            num_words += 1
    
    if num_words > 0:
        avg_vec = sum_vec / num_words
    else:
        avg_vec = np.zeros(300)
    
    return avg_vec

def create_avg_vectors(df, model):
    """Create average word embedding vectors for all descriptions."""
    print("Creating averaged description vectors...")
    start_time = time.time()
    
    df['avg_description_vector'] = df['description'].apply(
        lambda desc: avg_desc_vector(desc, model)
    )
    
    elapsed = time.time() - start_time
    print(f"Average vectors created in {elapsed:.2f} seconds")
    
    return df

# ====================  COSINE SIMILARITIES ====================

def similarity_scores(movie, df, avg_desc_vector_all):
    """Calculate cosine similarities between a movie and all movies."""
    try:
        movie_index = df[df['title'] == movie].index[0]
    except IndexError:
        print(f"Movie '{movie}' not found in database.")
        return None
    
    movie_avg_desc_vector = avg_desc_vector_all[movie_index].reshape(1, -1)
    cosine_similarities = cosine_similarity(movie_avg_desc_vector, avg_desc_vector_all)
    
    return cosine_similarities

# ====================  RECOMMENDATIONS ====================

def recommendations(movie, df, avg_desc_vector_all):
    """Recommend top 5 similar movies based on description similarity."""
    cosine_similarities = similarity_scores(movie, df, avg_desc_vector_all)
    
    if cosine_similarities is None:
        return []
    
    # Get similarity scores with indices
    similarity_scores_list = list(enumerate(cosine_similarities.squeeze().tolist()))
    
    # Sort by similarity score in descending order
    sorted_scores = sorted(similarity_scores_list, key=lambda x: x[1], reverse=True)
    
    # Get top 5 (excluding the movie itself at index 0)
    top5 = sorted_scores[1:6]
    
    # Extract movie titles and scores
    top5_indices = [idx for idx, score in top5]
    top5_scores = [score for idx, score in top5]
    top5_titles = df.iloc[top5_indices]['title'].tolist()
    
    return list(zip(top5_titles, top5_scores))

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function."""
    print("=" * 60)
    print("MOVIE RECOMMENDER SYSTEM USING WORD EMBEDDINGS")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data('movies_metadata.csv', 'keywords.csv')
    
    # Create concatenated descriptions
    df = create_concat_description(df)
    
    # Preprocess descriptions
    df = preprocess_descriptions(df)
    
    # Train Word2Vec model
    model, sentences = train_word2vec_model(df)
    
    # Test word similarities
    print("\nSample word similarities:")
    if 'pilot' in model.wv:
        print(f"Words similar to 'pilot': {model.wv.most_similar('pilot', topn=3)}")
    if 'animation' in model.wv:
        print(f"Words similar to 'animation': {model.wv.most_similar('animation', topn=3)}")
    
    # Create averaged vectors
    df = create_avg_vectors(df, model)
    avg_desc_vector_all = np.array(df['avg_description_vector'].tolist())
    
    # Display dataset info
    print(f"\nDataset shape: {df.shape}")
    print(f"Average vector shape: {avg_desc_vector_all.shape}")
    
    #  Test recommendations
    print("\n" + "=" * 60)
    print("MOVIE RECOMMENDATIONS")
    print("=" * 60)
    
    test_movies = ["Toy Story", "The Godfather", "Avatar", "The Fault in Our Stars"]
    
    for movie_title in test_movies:
        print(f"\nTop 5 recommendations for '{movie_title}':")
        recs = recommendations(movie_title, df, avg_desc_vector_all)
        if recs:
            for i, (rec_title, similarity) in enumerate(recs, 1):
                print(f"  {i}. {rec_title} (Similarity: {similarity:.4f})")
        else:
            print(f"  Could not find recommendations for '{movie_title}'")
    
    print("\n" + "=" * 60)
    print("Recommendation system completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()