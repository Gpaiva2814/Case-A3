import ast
import re

import nltk
import polars as pl
from nltk.corpus import stopwords
from text_unidecode import unidecode

nltk.download("stopwords", quiet=True)
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
        # Se já for uma string indicando desconhecido, padroniza direto
        if x.lower() in ["unknown", "nan", "none", "", "['unknown']", '["unknown"]']:
            return ["unknown"]
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                # Trata caso a lista interna contenha o termo unknown
                if len(parsed) == 1 and str(parsed[0]).lower() == "unknown":
                    return ["unknown"]
                return parsed
        except:
            # Se falhar o parse de algo que não era unknown, vira um alvo para o fill_null
            return None
    return None


def clean_authors(text):
    text = unidecode(str(text)).lower()
    text = re.sub(r"\bby\b\s*|written by\s*", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r".*?\((.*?)\)", r"\1", text)
    text = re.sub(r"[^a-z\s\-]", "", text)
    tokens = text.split()
    return " ".join(tokens)


if __name__ == "__main__":
    books = pl.scan_csv(f"{data_dir}/books_data.csv", ignore_errors=True)
    reviews = pl.scan_csv(f"{data_dir}/Books_rating.csv", ignore_errors=True)

    df = reviews.join(books, on="Title", how="left")

    df = df.with_columns(pl.col("Id").count().over("Id").alias("ratingsCount"))

    df = df.with_columns(
        [
            pl.col("text")
            .map_elements(clean_text, return_dtype=pl.String)
            .alias("clean_text"),
            pl.col("summary")
            .map_elements(clean_text, return_dtype=pl.String)
            .alias("clean_summary"),
            pl.col("categories").map_elements(
                normalize_list_column, return_dtype=pl.List(pl.String)
            ),
            pl.col("authors").map_elements(
                normalize_list_column, return_dtype=pl.List(pl.String)
            ),
        ]
    )

    df = df.with_columns(
        pl.col("authors")
        .list.eval(pl.element().map_elements(clean_authors, return_dtype=pl.String))
        .alias("clean_authors")
    )

    colunas_comuns = [
        "Id",
        "Price",
        "User_id",
        "profileName",
        "score",
        "time",
        "summary",
        "text",
        "description",
        "Title",
        "image",
        "previewLink",
        "publisher",
        "publishedDate",
        "infoLink",
    ]

    df = df.with_columns(pl.col(colunas_comuns).fill_null("Unknown"))
    df = df.with_columns(
        [
            pl.col("categories").fill_null(pl.lit(["unknown"])),
            # pl.col("authors").fill_null(pl.lit(["unknown"])),
            pl.col("clean_authors").fill_null(pl.lit(["unknown"])),
        ]
    )

    df.sink_parquet(f"{data_dir}/cleaned_reviews.parquet")
