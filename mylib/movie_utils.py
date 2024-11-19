import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def clean_title(title):
    """
    Cleans a movie title by removing non-alphanumeric characters.
    """
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# def load_and_clean_data(filepath):
#     """
#     Loads movie data from a CSV file, cleans the movie titles, and adds a new column.

#     Args:
#         filepath (str): Path to the CSV file containing movie data.

#     Returns:
#         DataFrame: A pandas DataFrame with cleaned titles.
#     """
#     movies = pd.read_csv(filepath)
#     movies["clean_title"] = movies["title"].apply(clean_title)
#     return movies


def load_and_clean_data(filepath):
    """Load and clean the movies dataset."""
    try:
        movies = pd.read_csv(filepath)

        # Check if 'title' column exists
        if "title" not in movies.columns:
            raise KeyError("'title' column is missing from the dataset")

        # Clean and preprocess the movies data
        movies["clean_title"] = movies["title"].apply(clean_title)
        return movies
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except KeyError as e:
        raise KeyError(f"Missing column in the dataset: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def initialize_vectorizer(movies):
    """
    Initializes a TF-IDF vectorizer on the cleaned titles in the movies dataset.

    Args:
        movies (DataFrame): The movies DataFrame with a 'clean_title' column.

    Returns:
        TfidfVectorizer, sparse matrix: The vectorizer and the fitted TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])
    return vectorizer, tfidf


def search_movies(title, movies, vectorizer, tfidf):
    """
    Searches for the most similar movies based on the given title.

    Args:
        title (str): The title to search for.
        movies (DataFrame): The movies DataFrame.
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        tfidf (sparse matrix): The TF-IDF matrix of the cleaned titles.

    Returns:
        DataFrame: A DataFrame of the top 5 most similar movies.
    """
    cleaned_title = clean_title(title)
    query_vec = vectorizer.transform([cleaned_title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]  # Reverse to show most similar first
    return results


if __name__ == "__main__":
    # Load and clean the movie data
    movies = load_and_clean_data("./data/movies.csv")

    # Initialize the vectorizer and TF-IDF matrix
    vectorizer, tfidf = initialize_vectorizer(movies)

    # Verify the cleaned titles
    print("Original Titles:")
    print(movies["title"].head())
    print("\nCleaned Titles:")
    print(movies["clean_title"].head())

    # Perform a search
    search_query = "The Matrix"
    print(f"\nSearch results for '{search_query}':")
    results = search_movies(search_query, movies, vectorizer, tfidf)
    print(results)
