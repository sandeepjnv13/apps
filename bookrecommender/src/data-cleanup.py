# imports
import kagglehub
from matplotlib.pyplot import plot_date
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# source : https://github.com/t-redactyl/llm-semantic-book-recommender
#  https://www.youtube.com/watch?v=Q7mS1VHm3Yw&list=PLWNkg0_Sntv0rL30df5dtShhFGU7KOeYS&index=8&ab_channel=freeCodeCamp.org

# Download latest version
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
books = pd.read_csv(f"{path}/books.csv")


books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - books["published_year"]
book_missing = books[~(books["description"].isna()) &
      ~(books["num_pages"].isna()) &
      ~(books["average_rating"].isna()) &
      ~(books["published_year"].isna())
]
book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]
(
    book_missing_25_words
    .drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
    .to_csv("books_cleaned.csv", index = False)
)

# Now csv is cleaned we can do vector searches on it

