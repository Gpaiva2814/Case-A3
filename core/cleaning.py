import ast
import re

import nltk
import polars as pl
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

data_dir = "data"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()

    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)


def normalize_list_column(x):

    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    return []


if __name__ == "__main__":
    books = pl.read_csv(f"{data_dir}/books_data.csv")

    reviews = pl.read_csv(f"{data_dir}/Books_rating.csv")

    df = reviews.join(books, on="Title", how="left")

    df = df.with_columns(
        pl.col("text")
        .map_elements(clean_text, return_dtype=pl.Utf8)
        .alias("clean_text")
    )
    df = df.with_columns(
        pl.col("summary")
        .map_elements(clean_text, return_dtype=pl.Utf8)
        .alias("clean_summary")
    )
    df = df.with_columns(
        pl.col("authors").map_elements(
            normalize_list_column, return_dtype=pl.List(pl.Utf8)
        )
    )
    df = df.with_columns(
        pl.col("categories").map_elements(
            normalize_list_column, return_dtype=pl.List(pl.Utf8)
        )
    )
    # df.write_csv(f"{data_dir}/cleaned_reviews.csv")
    df.write_parquet(f"{data_dir}/cleaned_reviews.parquet")
    # (df.head()).write_csv(f"{data_dir}/sample_cleaned_reviews.csv")
