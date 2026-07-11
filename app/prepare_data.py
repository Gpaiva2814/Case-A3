import polars as pl

INPUT = "data/data_sentiment1.parquet"

OUT_AUTHOR = "data/author_metrics.parquet"
OUT_GENRE = "data/genre_metrics.parquet"
OUT_REVIEWS = "data/reviews_interview.parquet"


def prepare():

    print("📥 carregando")

    df = pl.read_parquet(INPUT)

    print("Linhas:", df.height)

    # remove coluna pesada
    if "text_embedding" in df.columns:
        df = df.drop("text_embedding")

    print("🧹 corrigindo tipos")

    df = df.with_columns(
        [
            pl.col("score").cast(pl.Float32, strict=False),
            pl.col("sentiment_score").cast(pl.Float32, strict=False),
            pl.col("aspect_plot").cast(pl.Float32, strict=False),
            pl.col("aspect_style").cast(pl.Float32, strict=False),
            pl.col("aspect_pacing").cast(pl.Float32, strict=False),
        ]
    )

    # =============================
    # SATISFAÇÃO / POLARIZAÇÃO
    # =============================

    df = df.with_columns(
        [
            (pl.col("score") >= 4).cast(pl.Float32).alias("is_positive"),
            ((pl.col("score") <= 2) | (pl.col("score") >= 4))
            .cast(pl.Float32)
            .alias("is_extreme"),
        ]
    )

    # =============================
    # AUTORES
    # =============================

    print("👤 autores")

    authors = (
        df.explode("clean_authors")
        .filter(pl.col("clean_authors").is_not_null())
        .group_by("clean_authors")
        .agg(
            [
                pl.len().alias("reviews"),
                pl.col("score").mean().alias("avg_score"),
                pl.col("is_positive").mean().alias("satisfaction_rate"),
                pl.col("is_extreme").mean().alias("polarization"),
                pl.col("aspect_plot").mean().alias("plot"),
                pl.col("aspect_style").mean().alias("style"),
                pl.col("aspect_pacing").mean().alias("pacing"),
            ]
        )
        .filter(pl.col("reviews") >= 20)
        .sort("reviews", descending=True)
    )

    authors.write_parquet(OUT_AUTHOR, compression="zstd")

    # =============================
    # GENEROS
    # =============================

    print("🏷️ gêneros")

    df = df.with_columns(
        pl.col("categories_clean").map_elements(
            lambda x: (
                x
                if isinstance(x, list)
                else str(x)
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(",")
            ),
            return_dtype=pl.List(pl.String),
        )
    )

    genres = (
        df.explode("categories_clean")
        .filter(pl.col("categories_clean").is_not_null())
        .group_by("categories_clean")
        .agg(
            [
                pl.len().alias("reviews"),
                pl.col("score").mean().alias("avg_score"),
                pl.col("is_positive").mean().alias("satisfaction_rate"),
                pl.col("is_extreme").mean().alias("polarization"),
                pl.col("aspect_plot").mean().alias("plot"),
                pl.col("aspect_style").mean().alias("style"),
                pl.col("aspect_pacing").mean().alias("pacing"),
            ]
        )
        .filter(pl.col("reviews") >= 20)
        .sort("reviews", descending=True)
    )

    genres.write_parquet(OUT_GENRE, compression="zstd")

    # =============================
    # REVIEWS PARA ENTREVISTA
    # =============================

    print("🎤 reviews")

    interviews = df.select(
        [
            "clean_authors",
            "Title",
            "profileName",
            "User_id",
            "score",
            "text",
            "sentiment_score",
            "aspect_plot",
            "aspect_style",
            "aspect_pacing",
        ]
    ).filter(pl.col("clean_authors").is_not_null())

    interviews.write_parquet(OUT_REVIEWS, compression="zstd")

    print("✅ pronto")


if __name__ == "__main__":
    prepare()
