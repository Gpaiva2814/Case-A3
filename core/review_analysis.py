import nltk
import polars as pl
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# baixar léxico (1x)
nltk.download("vader_lexicon")

# =========================
# CONFIG
# =========================

INPUT_PATH = "data/cleaned_reviews.parquet"
OUTPUT_PATH = "data/data_sentiment.parquet"
BATCH_SIZE = 50_000

# =========================
# MODELO
# =========================

sia = SentimentIntensityAnalyzer()


def classify_sentiment(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutro"

    score = sia.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    else:
        return "neutro"


def map_score(score):
    if score is None:
        return None
    if score <= 2:
        return "negativo"
    elif score == 3:
        return "neutro"
    else:
        return "positivo"


# =========================
# PROCESSAMENTO
# =========================


def process_file():

    # lazy scan (não carrega tudo na memória)
    lf = pl.scan_parquet(INPUT_PATH)

    total_rows = lf.select(pl.count()).collect().item()

    results = []

    for start in tqdm(range(0, total_rows, BATCH_SIZE)):
        df = lf.slice(start, BATCH_SIZE).collect()

        # limpeza
        df = df.with_columns(
            pl.col("clean_summary").fill_null("").str.to_lowercase().str.strip_chars()
        )

        # sentimento texto (UDF)
        df = df.with_columns(
            pl.col("clean_summary")
            .map_elements(classify_sentiment)
            .alias("sentiment_text")
        )

        # sentimento via score (se existir)
        if "score" in df.columns:
            df = df.with_columns(
                pl.col("score").map_elements(map_score).alias("sentiment_score")
            )

        results.append(df)

    df_final = pl.concat(results)

    df_final.write_parquet(OUTPUT_PATH)

    print("✅ Finalizado e salvo em parquet")


# =========================
# EXEC
# =========================

if __name__ == "__main__":
    process_file()
