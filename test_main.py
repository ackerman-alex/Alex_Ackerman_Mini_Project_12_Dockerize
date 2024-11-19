# import pytest
# from mylib.movie_utils import load_and_clean_data, initialize_vectorizer, search_movies
# from mylib.recommender_utils import find_similar_movies
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer


# @pytest.fixture
# def sample_movies():
#     # Create a small sample DataFrame with at least 5 movies for testing
#     data = {
#         "movieId": [1, 2, 3, 4, 5],
#         "title": [
#             "The Matrix",
#             "Matrix Reloaded",
#             "The Notebook",
#             "Inception",
#             "Interstellar",
#         ],
#         "genres": ["Action", "Action", "Romance", "Sci-Fi", "Sci-Fi"],
#     }
#     movies = pd.DataFrame(data)
#     # Add the clean_title column (simulating what load_and_clean_data does)
#     movies["clean_title"] = movies["title"].apply(
#         lambda x: "".join(c for c in x if c.isalnum() or c.isspace())
#     )
#     return movies


# @pytest.fixture
# def sample_ratings():
#     # Create a small sample ratings DataFrame for testing
#     data = {
#         "userId": [1, 2, 3, 4, 5],
#         "movieId": [1, 1, 2, 3, 4],
#         "rating": [5, 4, 5, 3, 4],
#     }
#     return pd.DataFrame(data)


# def test_load_and_clean_data():
#     # Test integration with real data
#     filepath = "./data/movies.csv"  # Update with a valid path if available
#     movies = load_and_clean_data(filepath)
#     assert "clean_title" in movies.columns
#     assert not movies["clean_title"].isnull().any()


# def test_initialize_vectorizer(sample_movies):
#     vectorizer, tfidf = initialize_vectorizer(sample_movies)
#     assert isinstance(vectorizer, TfidfVectorizer)
#     assert tfidf.shape[0] == len(sample_movies)


# def test_search_movies(sample_movies):
#     vectorizer, tfidf = initialize_vectorizer(sample_movies)
#     results = search_movies("Matrix", sample_movies, vectorizer, tfidf)
#     assert not results.empty  # Ensure there are results
#     assert len(results) <= len(sample_movies)  # Results should not exceed dataset size
#     assert any(
#         "Matrix" in title for title in results["title"].values
#     )  # Validate result titles


# def test_find_similar_movies(sample_movies, sample_ratings):
#     movie_id = 1  # Test with "The Matrix"
#     recommendations = find_similar_movies(movie_id, sample_ratings, sample_movies)
#     assert isinstance(recommendations, pd.DataFrame)
#     assert not recommendations.empty  # Ensure recommendations are returned
#     assert "title" in recommendations.columns  # Validate the structure


# if __name__ == "__main__":
#     # Demonstrate the functionality
#     print("Running sample movie recommendation tests and demonstration...")

#     # Sample movie and rating data
#     sample_movies_data = {
#         "movieId": [1, 2, 3, 4, 5],
#         "title": [
#             "The Matrix",
#             "Matrix Reloaded",
#             "The Notebook",
#             "Inception",
#             "Interstellar",
#         ],
#         "genres": ["Action", "Action", "Romance", "Sci-Fi", "Sci-Fi"],
#     }
#     sample_ratings_data = {
#         "userId": [1, 2, 3, 4, 5],
#         "movieId": [1, 1, 2, 3, 4],
#         "rating": [5, 4, 5, 3, 4],
#     }

#     movies = pd.DataFrame(sample_movies_data)
#     movies["clean_title"] = movies["title"].apply(
#         lambda x: "".join(c for c in x if c.isalnum() or c.isspace())
#     )
#     ratings = pd.DataFrame(sample_ratings_data)

#     # Initialize vectorizer and perform searches
#     vectorizer, tfidf = initialize_vectorizer(movies)

#     print("\nPerforming a search for 'Matrix':")
#     search_results = search_movies("Matrix", movies, vectorizer, tfidf)
#     print(search_results[["title", "genres"]])

#     print("\nFinding collaborative recommendations for 'The Matrix':")
#     recommendations = find_similar_movies(1, ratings, movies)
#     print(recommendations)


import pytest
from mylib.movie_utils import load_and_clean_data, initialize_vectorizer, search_movies
from mylib.recommender_utils import find_similar_movies
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.fixture
def sample_movies():
    """Fixture to create a small sample DataFrame with movie data."""
    data = {
        "movieId": [1, 2, 3, 4, 5],
        "title": [
            "The Matrix",
            "Matrix Reloaded",
            "The Notebook",
            "Inception",
            "Interstellar",
        ],
        "genres": ["Action", "Action", "Romance", "Sci-Fi", "Sci-Fi"],
    }
    movies = pd.DataFrame(data)
    # Simulate cleaning titles
    movies["clean_title"] = movies["title"].apply(
        lambda x: "".join(c for c in x if c.isalnum() or c.isspace())
    )
    return movies


@pytest.fixture
def sample_ratings():
    """Fixture to create a small sample DataFrame with ratings data."""
    data = {
        "userId": [1, 2, 3, 4, 5],
        "movieId": [1, 1, 2, 3, 4],
        "rating": [5, 4, 5, 3, 4],
    }
    return pd.DataFrame(data)


def test_load_and_clean_data(tmp_path):
    """Test the load_and_clean_data function with mock data."""
    # Create a temporary file with sample movie data
    filepath = tmp_path / "movies.csv"
    data = {
        "movieId": [1, 2, 3],
        "title": ["Toy Story", "Jumanji", "Interstellar"],
        "genres": ["Animation", "Adventure", "Sci-Fi"],
    }
    pd.DataFrame(data).to_csv(filepath, index=False)

    # Test the function
    movies = load_and_clean_data(filepath)
    assert "clean_title" in movies.columns
    assert not movies["clean_title"].isnull().any()


def test_initialize_vectorizer(sample_movies):
    """Test the initialize_vectorizer function with sample movies."""
    vectorizer, tfidf = initialize_vectorizer(sample_movies)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert tfidf.shape[0] == len(sample_movies)


def test_search_movies(sample_movies):
    """Test the search_movies function with sample movies."""
    vectorizer, tfidf = initialize_vectorizer(sample_movies)
    results = search_movies("Matrix", sample_movies, vectorizer, tfidf)
    assert not results.empty  # Ensure there are results
    assert len(results) <= len(sample_movies)  # Results should not exceed dataset size
    assert any(
        "Matrix" in title for title in results["title"].values
    )  # Validate result titles


def test_find_similar_movies(sample_movies, sample_ratings):
    """Test the find_similar_movies function with sample data."""
    movie_id = 1  # Test with "The Matrix"
    recommendations = find_similar_movies(movie_id, sample_ratings, sample_movies)
    assert isinstance(recommendations, pd.DataFrame)
    assert not recommendations.empty  # Ensure recommendations are returned
    assert "title" in recommendations.columns  # Validate the structure


if __name__ == "__main__":
    print("Running tests with sample movie recommendation system data...")

    # Sample movie and rating data
    sample_movies_data = {
        "movieId": [1, 2, 3, 4, 5],
        "title": [
            "The Matrix",
            "Matrix Reloaded",
            "The Notebook",
            "Inception",
            "Interstellar",
        ],
        "genres": ["Action", "Action", "Romance", "Sci-Fi", "Sci-Fi"],
    }
    sample_ratings_data = {
        "userId": [1, 2, 3, 4, 5],
        "movieId": [1, 1, 2, 3, 4],
        "rating": [5, 4, 5, 3, 4],
    }

    movies = pd.DataFrame(sample_movies_data)
    movies["clean_title"] = movies["title"].apply(
        lambda x: "".join(c for c in x if c.isalnum() or c.isspace())
    )
    ratings = pd.DataFrame(sample_ratings_data)

    # Demonstrate vectorizer and recommendations
    vectorizer, tfidf = initialize_vectorizer(movies)
    print("\nPerforming a search for 'Matrix':")
    search_results = search_movies("Matrix", movies, vectorizer, tfidf)
    print(search_results[["title", "genres"]])

    print("\nFinding collaborative recommendations for 'The Matrix':")
    recommendations = find_similar_movies(1, ratings, movies)
    print(recommendations)
