import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# baixar léxico (só 1x)
nltk.download("vader_lexicon")

# =========================
# CONFIG
# =========================

INPUT_PATH = "data/cleaned_reviews.csv"
OUTPUT_PATH = "data/data_sentiment.parquet"
BATCH_SIZE = 50000

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


# =========================
# (OPCIONAL) BASEADO EM SCORE
# =========================


def map_score(score):
    if pd.isna(score):
        return None
    if score <= 2:
        return "negativo"
    elif score == 3:
        return "neutro"
    else:
        return "positivo"


# =========================
# PROCESSAMENTO EM LOTE
# =========================


def process_file():
    chunks = pd.read_csv(INPUT_PATH, chunksize=BATCH_SIZE)

    results = []

    for chunk in tqdm(chunks):
        # limpa básico
        chunk["clean_summary"] = (
            chunk["clean_summary"].fillna("").str.lower().str.strip()
        )

        # sentimento via texto
        chunk["sentiment_text"] = chunk["clean_summary"].apply(classify_sentiment)

        # sentimento via score (se existir)
        if "score" in chunk.columns:
            chunk["sentiment_score"] = chunk["score"].apply(map_score)

        results.append(chunk)

    df_final = pd.concat(results)

    # salva em formato eficiente
    df_final.to_parquet(OUTPUT_PATH, index=False)

    print("✅ Finalizado e salvo em parquet")


# =========================
# EXEC
# =========================

if __name__ == "__main__":
    process_file()
