import ast

import matplotlib.pyplot as plt
import plotly.express as px
import polars as pl
import streamlit as st
from wordcloud import WordCloud

# =========================
# CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("📚 Book Review Analytics")


# =========================
# LOAD DATA
# =========================


@st.cache_data
def load_data():

    df = pl.read_parquet("data/data_sentiment.parquet")

    # converter strings de listas
    df = df.with_columns(
        pl.col("authors").map_elements(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x,
            return_dtype=pl.List(pl.Utf8),
        )
    )

    df = df.with_columns(
        pl.col("categories").map_elements(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x,
            return_dtype=pl.List(pl.Utf8),
        )
    )

    # dataframes normalizados
    df_authors = df.explode("authors")
    df_categories = df.explode("categories")

    return df, df_authors, df_categories


df_raw, df_authors, df_categories = load_data()


# =========================
# FILTROS
# =========================

st.sidebar.header("Filtros")


all_categories = (
    df_categories.select("categories")
    .drop_nulls()
    .unique()
    .sort("categories")["categories"]
    .to_list()
)


default_category = all_categories[0] if all_categories else None


categories = st.sidebar.multiselect(
    "Categoria",
    all_categories,
    default=[default_category] if default_category else [],
)


all_authors = (
    df_authors.select("authors")
    .drop_nulls()
    .unique()
    .sort("authors")["authors"]
    .to_list()
)


authors = st.sidebar.multiselect(
    "Autor",
    all_authors,
)


# =========================
# APLICA FILTROS
# =========================

filtered_raw = df_raw
filtered_authors = df_authors
filtered_categories = df_categories


if categories:
    filtered_categories = filtered_categories.filter(
        pl.col("categories").is_in(categories)
    )

    filtered_raw = filtered_raw.filter(
        pl.col("categories").list.eval(pl.element().is_in(categories)).list.any()
    )

    filtered_authors = filtered_authors.filter(pl.col("categories").is_in(categories))


if authors:
    filtered_authors = filtered_authors.filter(pl.col("authors").is_in(authors))

    filtered_raw = filtered_raw.filter(
        pl.col("authors").list.eval(pl.element().is_in(authors)).list.any()
    )

    filtered_categories = filtered_categories.filter(pl.col("authors").is_in(authors))


# =========================
# MÉTRICAS
# =========================

st.subheader("📊 Métricas")


col1, col2, col3, col4, col5 = st.columns(5)


if filtered_raw.height == 0:
    col1.metric("⭐ Score médio", "-")
    col2.metric("🙂 Percepção geral", "-")
    col3.metric("🔥 Intensidade emocional", "-")
    col4.metric("⚖️ Divergência", "-")
    col5.metric("🧠 Coerência texto-nota", "-")


else:
    score_mean = filtered_raw["score"].mean()

    sentiment_dist = (
        filtered_raw.group_by("sentiment_text")
        .len()
        .with_columns((pl.col("len") / filtered_raw.height).alias("percentage"))
    )

    sent_dict = dict(
        zip(sentiment_dist["sentiment_text"], sentiment_dist["percentage"])
    )

    perc_pos = sent_dict.get("positivo", 0)
    perc_neg = sent_dict.get("negativo", 0)

    perception = perc_pos - perc_neg

    emotional_intensity = perc_pos + perc_neg

    divergence = filtered_raw["score"].std()

    sentiment_map = {
        "negativo": 1,
        "neutro": 3,
        "positivo": 5,
    }

    df_gap = filtered_raw.with_columns(
        pl.col("sentiment_text")
        .replace(
            {
                "negativo": 1,
                "neutro": 3,
                "positivo": 5,
            }
        )
        .cast(pl.Int8)
        .alias("sent_num")
    )

    gap = (df_gap["score"] - df_gap["sent_num"]).abs().mean()

    col1.metric("⭐ Score médio", f"{score_mean:.2f}")

    col2.metric("🙂 Percepção geral", f"{perception:.2%}")

    col3.metric("🔥 Intensidade emocional", f"{emotional_intensity:.2%}")

    col4.metric("⚖️ Divergência", f"{divergence:.2f}")

    col5.metric("🧠 Coerência texto-nota", f"{gap:.2f}")


# =========================
# WORD CLOUD
# =========================

st.subheader("🧠 Top palavras por sentimento")


sentiment_choice = st.selectbox(
    "Escolha o sentimento", ["positivo", "neutro", "negativo"]
)


text_data = (
    filtered_raw.filter(pl.col("sentiment_text") == sentiment_choice)
    .select("clean_summary")
    .drop_nulls()["clean_summary"]
    .to_list()
)


text_blob = " ".join(text_data)


if text_blob:
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
    ).generate(text_blob)

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
    filtered_authors.group_by("authors")
    .agg(
        [
            pl.col("score").mean().alias("score_mean"),
            pl.len().alias("reviews"),
        ]
    )
    .filter(pl.col("reviews") > 10)
    .sort("score_mean", descending=True)
    .head(10)
    .to_pandas()
)


fig = px.bar(
    author_rank,
    x="score_mean",
    y="authors",
    orientation="h",
    title="Top autores",
)


st.plotly_chart(fig, use_container_width=True)


# =========================
# RANKING LIVROS
# =========================

st.subheader("📖 Ranking de livros")


book_rank = (
    filtered_raw.group_by("Title")
    .agg(
        [
            pl.col("score").mean().alias("score_mean"),
            pl.len().alias("reviews"),
        ]
    )
    .filter(pl.col("reviews") > 10)
    .sort("score_mean", descending=True)
    .head(10)
    .to_pandas()
)


fig2 = px.bar(
    book_rank,
    x="score_mean",
    y="Title",
    orientation="h",
    title="Top livros",
)


st.plotly_chart(fig2, use_container_width=True)
