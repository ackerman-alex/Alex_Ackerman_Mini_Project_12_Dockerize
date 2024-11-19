from mylib.movie_utils import load_and_clean_data, initialize_vectorizer, search_movies
from mylib.recommender_utils import find_similar_movies
import pandas as pd


def interactive_search(movies, ratings, vectorizer, tfidf):
    """
    Interactive command-line search loop for finding and recommending movies.

    Args:
        movies (DataFrame): DataFrame of movies.
        ratings (DataFrame): DataFrame of user ratings.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer trained on movie titles.
        tfidf (sparse matrix): TF-IDF matrix of movie titles.
    """
    print("Interactive Movie Recommendation Tool")
    print("Type a movie title to search for similar movies. Type 'exit' to quit.")

    while True:
        # Prompt user for a movie title
        title = input("\nEnter a movie title: ").strip()
        if title.lower() == "exit":
            print("Exiting the tool. Goodbye!")
            break

        # Ensure the input has sufficient length for meaningful searches
        if len(title) > 5:
            # Perform content-based search
            search_results = search_movies(title, movies, vectorizer, tfidf)

            if search_results.empty:
                print(f"No results found for '{title}'. Try another title.")
                continue

            print("\nContent-Based Search Results:")
            print(search_results[["title"]].to_string(index=False))

            # Get the first search result's movie ID for collaborative filtering
            movie_id = search_results.iloc[0]["movieId"]
            print(f"Movie ID being tested: {movie_id}")
            print(
                f"\nFinding collaborative recommendations based on '{search_results.iloc[0]['title']}'...\n"
            )

            # Perform collaborative filtering
            try:
                recommendations = find_similar_movies(movie_id, ratings, movies)
                if not recommendations.empty:
                    print("Collaborative Filtering Recommendations:")
                    print(recommendations.to_string(index=False))
                else:
                    print("No collaborative recommendations found.")
            except Exception as e:
                print(f"Error while finding recommendations: {e}")
        else:
            print("Please enter a title with more than 5 characters.")


if __name__ == "__main__":
    # Filepaths to the datasets
    MOVIES_FILEPATH = "./data/movies.csv"
    RATINGS_FILEPATH = "./data/ratings.csv"

    # Load and clean the movie and ratings data
    movies = load_and_clean_data(MOVIES_FILEPATH)
    ratings = pd.read_csv(RATINGS_FILEPATH)

    # Initialize vectorizer and TF-IDF matrix
    vectorizer, tfidf = initialize_vectorizer(movies)

    # Start the interactive search tool
    interactive_search(movies, ratings, vectorizer, tfidf)
