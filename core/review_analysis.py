import os

import faiss  # 🔥 Adicionado para o Banco Vetorial
import nltk
import numpy as np
import polars as pl
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

nltk.download("vader_lexicon", quiet=True)
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# =========================
# CONFIG
# =========================

INPUT_PATH = "data/cleaned_reviews_mapped.parquet"
OUTPUT_PATH = "data/data_sentiment.parquet"
FAISS_PATH = "vector_db/reviews_index.faiss"

# =========================
# MODELOS E CONFIG NLP
# =========================

sia = SentimentIntensityAnalyzer()
ASPECT_LABELS = [
    "book plot story development character arc ending narrative twist",
    "writing style prose editing vocabulary grammar language quality phrasing",
    "pacing slow fast book speed narrative rhythm dragging reading velocity",
]

model_emb = None
label_embeddings = None


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


def process_file():
    global model_emb, label_embeddings

    print("🧠 Carregando modelo semântico leve para CPU...")
    model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    label_embeddings = model_emb.encode(ASPECT_LABELS, normalize_embeddings=True)

    # 1. Carrega a base completa de forma Lazy
    lf_original = pl.scan_parquet(INPUT_PATH)

    print("⏳ Coletando e preparando todo o subset de cálculo de uma só vez...")
    df_calculo = (
        lf_original.filter(
            (pl.col("User_id") != "Unknown") & (pl.col("ratingsCount") >= 30)
        )
        .select(["Id", "User_id", "text", "clean_summary", "clean_text"])
        .collect()
    )

    total_rows = df_calculo.height
    print(f"📊 Total de registros qualificados para processamento: {total_rows}")

    # 2. Métricas de Eloquência nativas
    df_calculo = df_calculo.with_columns(
        [
            pl.col("text").fill_null("").cast(pl.Utf8),
            pl.col("clean_summary").fill_null("").str.to_lowercase(),
            pl.col("clean_text").str.split(" ").list.len().alias("review_length"),
            pl.col("clean_text")
            .str.split(" ")
            .list.unique()
            .list.len()
            .alias("unique_words"),
        ]
    )

    # 3. INICIALIZA O POOL PARALELO
    print("🚀 Inicializando Pool de multiprocessamento paralelo...")
    pool = model_emb.start_multi_process_pool()

    print("🤖 Executando inferência semântica paralela com TQDM...")
    texts_to_vectorize = df_calculo["text"].to_list()

    # Tamanho do bloco para atualizar a barra de progresso (100k em 100k)
    CHUNK_SIZE = 100_000
    all_embeddings = []

    # O loop percorre a lista em pedaços e o tqdm calcula a velocidade e o tempo restante (ETA)
    for i in tqdm(range(0, total_rows, CHUNK_SIZE), desc="Vetorizando Reviews"):
        chunk_texts = texts_to_vectorize[i : i + CHUNK_SIZE]
        truncated_texts = [str(t)[:140] if t else "" for t in chunk_texts]

        # O pool processa o bloco atual em paralelo usando todas as threads
        chunk_embeddings = model_emb.encode_multi_process(
            truncated_texts,
            pool=pool,
            batch_size=512,
        )
        all_embeddings.append(chunk_embeddings)

    # Desliga o pool imediatamente após o processamento
    model_emb.stop_multi_process_pool(pool)

    print("📦 Consolidando e normalizando os embeddings criados...")
    # Junta todos os blocos em uma única matriz numpy gigante
    text_embeddings = np.vstack(all_embeddings).astype("float32")

    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    text_embeddings_normalized = text_embeddings / norms

    # 4. SALVANDO NO BANCO VETORIAL FAISS
    print("📦 Construindo índice do banco vetorial FAISS...")
    dimension = text_embeddings_normalized.shape[1]  # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(text_embeddings_normalized)

    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_PATH)
    print(f"💾 Banco Vetorial FAISS salvo com sucesso em: {FAISS_PATH}")

    print("⚖️ Calculando matrizes de similaridade de cosseno...")
    similarity_matrix = np.dot(text_embeddings_normalized, label_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, 0, None)

    # Injeta os aspectos calculados E os embeddings puros em formato de lista
    df_calculo = df_calculo.with_columns(
        [
            pl.Series(
                "aspect_plot", similarity_matrix[:, 0].tolist(), dtype=pl.Float64
            ),
            pl.Series(
                "aspect_style", similarity_matrix[:, 1].tolist(), dtype=pl.Float64
            ),
            pl.Series(
                "aspect_pacing", similarity_matrix[:, 2].tolist(), dtype=pl.Float64
            ),
            pl.Series(
                "text_embedding",
                text_embeddings_normalized.tolist(),
                dtype=pl.List(pl.Float32),
            ),
        ]
    )

    print("🎭 Calculando análise de sentimento com VADER...")
    df_calculo = df_calculo.with_columns(
        pl.col("clean_summary")
        .map_elements(classify_sentiment, return_dtype=pl.String)
        .alias("sentiment_text")
    )

    # Seleciona as colunas de feature incluindo o vetor para o merge final
    df_features_todas = df_calculo.select(
        [
            "Id",
            "User_id",
            "review_length",
            "unique_words",
            "aspect_plot",
            "aspect_style",
            "aspect_pacing",
            "sentiment_text",
            "text_embedding",
        ]
    )

    print("\n🔄 Reconstituindo base de dados original via Left Join final...")
    df_original_completo = lf_original.collect()

    if "score" in df_original_completo.columns:
        df_original_completo = df_original_completo.with_columns(
            pl.col("score")
            .map_elements(map_score, return_dtype=pl.String)
            .alias("sentiment_score")
        )

    df_final = df_original_completo.join(
        df_features_todas, on=["Id", "User_id"], how="left"
    )

    df_final = df_final.with_columns(
        [
            pl.col("review_length").fill_null(0),
            pl.col("unique_words").fill_null(0),
            pl.col("aspect_plot").fill_null(0.0),
            pl.col("aspect_style").fill_null(0.0),
            pl.col("aspect_pacing").fill_null(0.0),
            pl.col("sentiment_text").fill_null("neutro"),
            pl.col("text_embedding").fill_null(pl.lit(None, dtype=pl.List(pl.Float32))),
        ]
    )

    print(
        f"Saving dataset unificado final ({df_final.height} linhas) em: {OUTPUT_PATH}"
    )
    df_final.write_parquet(OUTPUT_PATH)
    print("✅ Pipeline executado com sucesso e arquivos locais salvos!")


if __name__ == "__main__":
    process_file()
