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


def normalize_author(name):
    if name is None:
        return None

    name = unidecode(str(name)).lower().strip()

    # remove prefixos
    name = re.sub(r"\bby\b\s*", "", name)
    name = re.sub(r"\bwritten by\b\s*", "", name)

    # remove conteúdo entre colchetes/parênteses
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\(.*?\)", "", name)

    # mantém letras, espaço e hífen
    name = re.sub(r"[^a-z\s\-]", "", name)

    # normaliza hífen
    name = name.replace("-", " ")

    # remove espaços duplicados
    name = " ".join(name.split())

    # junta iniciais:
    # j r r tolkien -> jrr tolkien
    parts = name.split()

    if len(parts) > 1:
        initials = []
        others = []

        for p in parts:
            if len(p) == 1:
                initials.append(p)
            else:
                others.append(p)

        if initials:
            name = "".join(initials) + " " + " ".join(others)

    return name.strip()


def main():
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
            pl.col("categories")
            .map_elements(normalize_list_column, return_dtype=pl.List(pl.String))
            .alias("clean_categories"),
            pl.col("authors")
            .map_elements(normalize_list_column, return_dtype=pl.List(pl.String))
            .alias("clean_authors"),
        ]
    )

    df = df.with_columns(
        pl.col("clean_authors").list.eval(
            pl.element().map_elements(normalize_author, return_dtype=pl.String)
        )
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
            pl.col("clean_categories").fill_null(pl.lit(["unknown"])),
            pl.col("clean_authors").fill_null(pl.lit(["unknown"])),
        ]
    )

    # =========================================================================================
    # TRATATIVA DE METADADOS AUSENTES (CONTROLE DE QUALIDADE)
    # Nota Metodológica: Testes de cruzamento interno via Pandas confirmaram que os registros
    # com autores 'unknown' não possuem duplicatas preenchidas por correspondência de 'Title'.
    # Tratam-se de lacunas estritas na origem.
    # Optou-se pelo descarte analítico dessas linhas, dado que representam produtos de cauda longa
    # (long tail) com baixa relevância amostral e volumetria dispersa, limpando o escopo do modelo.
    # =========================================================================================
    df = df.filter(
        ~pl.col("clean_authors").list.contains("unknown")
        & (pl.col("clean_authors").list.len() > 0)
    )

    df.sink_parquet(f"{data_dir}/cleaned_reviews.parquet")
    print(f"✅ Base de dados limpa e salva em: {data_dir}/cleaned_reviews.parquet")


if __name__ == "__main__":
    main()
