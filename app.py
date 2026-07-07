import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# =========================
# CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("📚 Book Review Analytics")


# @st.cache_data
# def load_data():
#     return pd.read_parquet("data/data_sentiment.parquet")
@st.cache_data
def load_data():
    return pd.read_parquet("data/data_sentiment.parquet").head(1000)


df = load_data()

# =========================
# FILTROS
# =========================

st.sidebar.header("Filtros")

authors = st.sidebar.multiselect("Autor", df["author"].dropna().unique())

genres = st.sidebar.multiselect("Gênero", df["genre"].dropna().unique())

filtered_df = df.copy()

if authors:
    filtered_df = filtered_df[filtered_df["author"].isin(authors)]

if genres:
    filtered_df = filtered_df[filtered_df["genre"].isin(genres)]

# =========================
# MÉTRICAS
# =========================

st.subheader("📊 Métricas")

col1, col2, col3, col4, col5 = st.columns(5)

# Score médio
score_mean = filtered_df["score"].mean()

# Percepção geral (% positivo - % negativo)
sent_counts = filtered_df["sentiment_text"].value_counts(normalize=True)
perc_pos = sent_counts.get("positivo", 0)
perc_neg = sent_counts.get("negativo", 0)
perception = perc_pos - perc_neg

# Polarização (% extremos)
polarization = filtered_df["sentiment_text"].isin(["positivo", "negativo"]).mean()

# Consistência do autor (desvio padrão do score)
consistency = filtered_df["score"].std()


# Gap (score vs sentimento)
def map_sentiment_to_num(x):
    return {"negativo": 1, "neutro": 3, "positivo": 5}.get(x, 3)


filtered_df["sent_num"] = filtered_df["sentiment_text"].map(map_sentiment_to_num)

gap = (filtered_df["score"] - filtered_df["sent_num"]).abs().mean()

col1.metric("⭐ Score médio", f"{score_mean:.2f}")
col2.metric("🙂 Percepção geral", f"{perception:.2%}")
col3.metric("🔥 Polarização", f"{polarization:.2%}")
col4.metric("📏 Consistência", f"{consistency:.2f}")
col5.metric("💰 Gap score-texto", f"{gap:.2f}")

# =========================
# WORD CLOUD
# =========================

st.subheader("🧠 Top palavras por sentimento")

sentiment_choice = st.selectbox(
    "Escolha o sentimento", ["positivo", "neutro", "negativo"]
)

text_data = filtered_df[filtered_df["sentiment_text"] == sentiment_choice][
    "clean_summary"
].dropna()

text_blob = " ".join(text_data.astype(str).tolist())

if text_blob:
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("Sem dados suficientes")

# =========================
# RANKING AUTORES
# =========================

st.subheader("🏆 Ranking de autores")

author_rank = (
    filtered_df.groupby("author")
    .agg(score_mean=("score", "mean"), reviews=("score", "count"))
    .query("reviews > 10")
    .sort_values("score_mean", ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    author_rank, x="score_mean", y="author", orientation="h", title="Top autores"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# RANKING LIVROS
# =========================

st.subheader("📖 Ranking de livros")

book_rank = (
    filtered_df.groupby("title")
    .agg(score_mean=("score", "mean"), reviews=("score", "count"))
    .query("reviews > 10")
    .sort_values("score_mean", ascending=False)
    .head(10)
    .reset_index()
)

fig2 = px.bar(book_rank, x="score_mean", y="title", orientation="h", title="Top livros")

st.plotly_chart(fig2, use_container_width=True)
