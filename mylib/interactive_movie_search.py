from movie_utils import load_and_clean_data, initialize_vectorizer, search_movies


def interactive_search(movies, vectorizer, tfidf):
    """
    Interactive command-line search loop.
    """
    print("Movie Search Tool")
    print("Type a movie title to search for similar movies. Type 'exit' to quit.")

    while True:
        title = input("Enter a movie title: ").strip()
        if title.lower() == "exit":
            print("Exiting the search tool. Goodbye!")
            break
        if len(title) > 5:  # Trigger search for sufficiently long inputs
            results = search_movies(title, movies, vectorizer, tfidf)
            print("\nSearch Results:")
            print(results[["title"]])  # Display only the title column for simplicity
        else:
            print("Please enter a title with more than 5 characters.")


if __name__ == "__main__":
    # Filepath to the movies dataset
    filepath = "./data/movies.csv"

    # Load and clean the movie data
    movies = load_and_clean_data(filepath)

    # Initialize the vectorizer and TF-IDF matrix
    vectorizer, tfidf = initialize_vectorizer(movies)

    # Run the interactive search tool
    interactive_search(movies, vectorizer, tfidf)
