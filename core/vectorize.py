import os

import chromadb
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "book_reviews"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 5000  # 🔥 pode aumentar sem medo agora

os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"


# ==========================
# PREPARE
# ==========================


def prepare_documents(df: pl.DataFrame) -> pl.DataFrame:

    columns = ["Title", "authors", "categories", "summary", "text"]

    for col in columns:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null("").cast(pl.Utf8))

    df = df.with_columns(pl.col("text").str.slice(0, 1000))

    df = df.with_columns(
        (
            pl.lit("Título: ")
            + pl.col("Title")
            + pl.lit("\nAutor: ")
            + pl.col("authors")
            + pl.lit("\nCategoria: ")
            + pl.col("categories")
            + pl.lit("\nResumo: ")
            + pl.col("summary")
            + pl.lit("\nAvaliação: ")
            + pl.col("text")
        ).alias("embedding_text")
    )

    return df


# ==========================
# VECTOR DB
# ==========================


def create_vector_database(df: pl.DataFrame):

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Book reviews embeddings"},
    )

    print("\nCarregando modelo...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nProcessando em batches (sequencial eficiente)...\n")

    for start_idx in tqdm(range(0, df.height, BATCH_SIZE), desc="Batches"):
        batch_df = df.slice(start_idx, BATCH_SIZE)

        texts = batch_df["embedding_text"].to_list()

        embeddings = model.encode(
            texts,
            batch_size=128,  # 🔥 maior = mais rápido (CPU permite)
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        metadatas = [
            {
                "title": title,
                "author": author,
                "category": category,
                "score": float(score),
            }
            for title, author, category, score in zip(
                batch_df["Title"],
                batch_df["authors"],
                batch_df["categories"],
                batch_df["score"],
            )
        ]

        ids = [str(start_idx + i) for i in range(len(texts))]

        # 🔥 salva incrementalmente (sem explodir RAM)
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    print("\nBanco vetorial criado!")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    df = pl.read_csv("data/cleaned_reviews.csv")

    print(f"Documentos carregados: {df.height}")

    df = prepare_documents(df)

    create_vector_database(df)
