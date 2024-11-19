from mylib.movie_utils import load_and_clean_data, initialize_vectorizer, search_movies
from mylib.recommender_utils import find_similar_movies
import pandas as pd


def main():
    # Filepaths to the datasets
    MOVIES_FILEPATH = "./data/movies.csv"
    RATINGS_FILEPATH = "./data/ratings.csv"

    try:
        # Load and clean the movie and ratings data
        movies = load_and_clean_data(MOVIES_FILEPATH)
        ratings = pd.read_csv(RATINGS_FILEPATH)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return

    # Initialize vectorizer and TF-IDF matrix
    try:
        vectorizer, tfidf = initialize_vectorizer(movies)
    except ValueError as e:
        print(f"Value error initializing vectorizer: {e}")
        return
    except TypeError as e:
        print(f"Type error initializing vectorizer: {e}")
        return

    print("Welcome to the Movie Recommendation System!")
    print("Type a movie title to search for recommendations or 'exit' to quit.")

    while True:
        # Prompt user for input
        title = input("\nEnter a movie title: ").strip()

        # Exit the program
        if title.lower() == "exit":
            print("Goodbye!")
            break

        # Ensure input is long enough for meaningful search
        if len(title) > 2:
            # Perform content-based search
            try:
                search_results = search_movies(title, movies, vectorizer, tfidf)
            except ValueError as e:
                print(f"Value error during content-based search: {e}")
                continue
            except KeyError as e:
                print(f"Key error during content-based search: {e}")
                continue

            if search_results.empty:
                print(f"No results found for '{title}'. Try another title.")
                continue

            print("\nContent-Based Search Results:")
            print(search_results[["title"]].to_string(index=False))

            # Get the first search result's movie ID for collaborative filtering
            movie_id = search_results.iloc[0]["movieId"]
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
            except KeyError as e:
                print(f"Key error during collaborative filtering: {e}")
            except ValueError as e:
                print(f"Value error during collaborative filtering: {e}")
            except IndexError as e:
                print(f"Index error during collaborative filtering: {e}")
        else:
            print("Please enter a title with more than 2 characters.")


if __name__ == "__main__":
    main()
