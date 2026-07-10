import os

import nltk
import polars as pl
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import pipeline

nltk.download("vader_lexicon", quiet=True)

# Força o cache do Hugging Face para o diretório local se necessário
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# =========================
# CONFIG
# =========================

INPUT_PATH = "data/cleaned_reviews_mapped.parquet"
OUTPUT_PATH = "data/data_sentiment.parquet"
BATCH_SIZE = 10_000  # 🔥 Reduzido de 50k para 10k para gerenciar o consumo de memória do LLM Zero-Shot

# =========================
# MODELOS E CONFIG NLP
# =========================

sia = SentimentIntensityAnalyzer()

# Inicializa o classificador Zero-Shot (Modelo leve e otimizado para CPU/GPU)
# Se houver GPU disponível no servidor, mude device=0. Para CPU, use device=-1
classifier = pipeline(
    "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1
)

CANDIDATE_LABELS = ["plot", "writing style", "pacing"]


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
    if score is None or score == "Unknown":
        return None
    try:
        score_num = float(score)
        if score_num <= 2:
            return "negativo"
        elif score_num == 3:
            return "neutro"
        else:
            return "positivo"
    except:
        return None


def extract_zero_shot_aspects(texts_list):
    """
    Executa a classificação Zero-Shot em lote nativo para máxima performance.
    """
    if not texts_list:
        return [], [], []

    # Executa a inferência em batch dentro da biblioteca do Hugging Face
    outputs = classifier(texts_list, candidate_labels=CANDIDATE_LABELS, batch_size=64)

    plots, styles, pacings = [], [], []

    for out in outputs:
        # Cria um dicionário temporário mapeando label -> score
        score_dict = dict(zip(out["labels"], out["scores"]))
        plots.append(score_dict.get("plot", 0.0))
        styles.append(score_dict.get("writing style", 0.0))
        pacings.append(score_dict.get("pacing", 0.0))

    return plots, styles, pacings


# =========================
# PROCESSAMENTO
# =========================


def process_file():

    lf = pl.scan_parquet(INPUT_PATH)
    total_rows = lf.select(pl.len()).collect().item()

    results = []

    print(f"Iniciando pipeline de IA para {total_rows} registros...")
    for start in tqdm(range(0, total_rows, BATCH_SIZE)):
        df = lf.slice(start, BATCH_SIZE).collect()

        # 1. Limpeza básica e Métricas de Eloquência calculadas de forma NATALINA no Polars
        df = df.with_columns(
            [
                pl.col("clean_summary")
                .fill_null("")
                .str.to_lowercase()
                .str.strip_chars(),
                pl.col("text").fill_null("").cast(pl.Utf8),
                pl.col("clean_text").str.split(" ").list.len().alias("review_length"),
                pl.col("clean_text")
                .str.split(" ")
                .list.unique()
                .list.len()
                .alias("unique_words"),
            ]
        )

        # 2. Sentimento do resumo (VADER)
        df = df.with_columns(
            pl.col("clean_summary")
            .map_elements(classify_sentiment, return_dtype=pl.String)
            .alias("sentiment_text")
        )

        if "score" in df.columns:
            df = df.with_columns(
                pl.col("score")
                .map_elements(map_score, return_dtype=pl.String)
                .alias("sentiment_score")
            )

        # 3. EXTRAÇÃO SEMÂNTICA DE ASPECTOS (Zero-Shot Batching Otimizado)
        # Cortamos os reviews longos em 300 caracteres para evitar gargalo e perda de performance
        texts_to_classify = df["text"].str.slice(0, 300).to_list()

        aspect_plots, aspect_styles, aspect_pacings = extract_zero_shot_aspects(
            texts_to_classify
        )

        # Injeta as pontuações probabilísticas geradas pelo modelo de volta no DataFrame
        df = df.with_columns(
            [
                pl.Series("aspect_plot", aspect_plots, dtype=pl.Float64),
                pl.Series("aspect_style", aspect_styles, dtype=pl.Float64),
                pl.Series("aspect_pacing", aspect_pacings, dtype=pl.Float64),
            ]
        )

        results.append(df)

    print("\nConcatenando resultados finais...")
    df_final = pl.concat(results)

    print(f"Salvando dataset enriquecido em: {OUTPUT_PATH}")
    df_final.write_parquet(OUTPUT_PATH)
    print("✅ Finalizado com sucesso!")


# =========================
# EXEC
# =========================

if __name__ == "__main__":
    process_file()
