import pandas as pd
from mylib.movie_utils import load_and_clean_data, initialize_vectorizer, search_movies


def get_similar_users(movie_id, ratings):
    """
    Finds users who rated the specified movie highly.

    Args:
        movie_id (int): The movie ID.
        ratings (DataFrame): The ratings DataFrame.

    Returns:
        ndarray: Array of user IDs who rated the movie highly.
    """
    return ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)][
        "userId"
    ].unique()


def calculate_similar_user_recommendations(similar_users, ratings):
    """
    Calculates the percentage of similar users who highly rated other movies.

    Args:
        similar_users (ndarray): Array of user IDs.
        ratings (DataFrame): The ratings DataFrame.

    Returns:
        Series: A Series with movie IDs as index and recommendation percentages as values.
    """
    similar_user_recs = ratings[
        (ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)
    ]["movieId"]
    return similar_user_recs.value_counts() / len(similar_users)


def calculate_all_user_recommendations(movie_ids, ratings):
    """
    Calculates the percentage of all users who highly rated the specified movies.

    Args:
        movie_ids (Index): Movie IDs for which to calculate percentages.
        ratings (DataFrame): The ratings DataFrame.

    Returns:
        Series: A Series with movie IDs as index and recommendation percentages as values.
    """
    all_users = ratings[(ratings["movieId"].isin(movie_ids)) & (ratings["rating"] > 4)]
    return all_users["movieId"].value_counts() / len(all_users["userId"].unique())


def compute_recommendation_scores(similar_user_recs, all_user_recs):
    """
    Computes the recommendation scores by dividing the similar user percentages
    by the all user percentages.

    Args:
        similar_user_recs (Series): Recommendation percentages for similar users.
        all_user_recs (Series): Recommendation percentages for all users.

    Returns:
        DataFrame: A DataFrame with movie IDs, similar percentages, all percentages, and scores.
    """
    rec_percentages = pd.concat(
        [similar_user_recs, all_user_recs], axis=1, join="inner"
    )
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    return rec_percentages.sort_values("score", ascending=False)


def find_similar_movies(movie_id, ratings, movies):
    """
    Finds movies similar to the given movie based on collaborative filtering.

    Args:
        movie_id (int): The ID of the movie for which recommendations are sought.
        ratings (DataFrame): The ratings DataFrame.
        movies (DataFrame): The movies DataFrame.

    Returns:
        DataFrame: A DataFrame containing the top 10 recommended movies with their score, title, and genres.
    """
    # Step 1: Find users who liked the movie
    similar_users = get_similar_users(movie_id, ratings)

    # Step 2: Get recommendation percentages for similar users
    similar_user_recs = calculate_similar_user_recommendations(similar_users, ratings)

    # Step 3: Filter movies with similar user recommendation percentage > 10%
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]

    # Step 4: Get recommendation percentages for all users
    all_user_recs = calculate_all_user_recommendations(similar_user_recs.index, ratings)

    # Step 5: Compute recommendation scores
    rec_percentages = compute_recommendation_scores(similar_user_recs, all_user_recs)

    # Step 6: Merge with movie data and return the top 10 results
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[
        ["score", "title", "genres"]
    ]


if __name__ == "__main__":
    # Load the data
    ratings = pd.read_csv("./data/ratings.csv")
    movies = load_and_clean_data("./data/movies.csv")

    # Test collaborative filtering recommendations
    test_movie_id = 1  # Replace with a valid movieId from your dataset
    print(f"\nCollaborative filtering recommendations for movie ID {test_movie_id}:")
    collaborative_results = find_similar_movies(test_movie_id, ratings, movies)
    print(collaborative_results)
