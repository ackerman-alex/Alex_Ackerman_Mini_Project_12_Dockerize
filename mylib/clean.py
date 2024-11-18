"""
Clean Movie Titles Using Regex
"""

import pandas as pd
import re


def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


if __name__ == "__main__":
    movies = pd.read_csv("./data/movies.csv")

    # Verify
    print(movies["title"].head())

    # Apply clean_title
    movies["clean_title"] = movies["title"].apply(clean_title)

    # Verify
    print(movies["clean_title"].head())
