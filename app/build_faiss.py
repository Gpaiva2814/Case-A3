import faiss
import numpy as np
import polars as pl

DATA_PATH = "data/streamlit_data.parquet"


def main():
    df = pl.read_parquet(DATA_PATH)

    df = df.filter(pl.col("text_embedding").is_not_null())

    embeddings = np.vstack(df["text_embedding"].to_list()).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "faiss.index")

    np.save("ids.npy", df["Id"].to_numpy())


if __name__ == "__main__":
    main()
