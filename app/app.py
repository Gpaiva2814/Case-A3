import plotly.express as px
import polars as pl
import streamlit as st

st.set_page_config(layout="wide")


@st.cache_data
def load():

    return (
        pl.read_parquet("data/author_metrics.parquet"),
        pl.read_parquet("data/genre_metrics.parquet"),
        pl.read_parquet("data/reviews_interview.parquet"),
    )


authors, genres, reviews = load()


st.title("📚 NLP Review Intelligence")


# =============================
# PERFORMANCE
# =============================


st.header("📊 Performance")


tipo = st.radio("Analisar", ["Autor", "Gênero"])


if tipo == "Autor":
    base = authors

    coluna = "clean_authors"


else:
    base = genres

    coluna = "categories_clean"


top = base.sort("reviews", descending=True).head(50)


selected = st.selectbox(tipo, top.get_column(coluna).to_list())


row = base.filter(pl.col(coluna) == selected).to_dicts()[0]


c1, c2, c3, c4 = st.columns(4)


c1.metric("📚 Reviews", f"{row['reviews']:,}")


c2.metric("⭐ Nota média", f"{row['avg_score']:.2f}")


c3.metric("😊 Satisfação", f"{row['satisfaction_rate']:.1%}")


c4.metric("⚡ Polarização", f"{row['polarization']:.1%}")


st.subheader("🧠 Dimensões NLP")


chart = pl.DataFrame(
    {
        "Dimensão": ["Narrativa", "Escrita", "Ritmo"],
        "Valor": [row["plot"], row["style"], row["pacing"]],
    }
)


st.plotly_chart(
    px.bar(chart, x="Dimensão", y="Valor", range_y=[0, 1]), use_container_width=True
)


# =============================
# ENTREVISTA
# =============================


if tipo == "Autor":
    st.header("🎤 Usuários para entrevista")

    filtered = reviews.filter(pl.col("clean_authors").list.contains(selected))

    # promotor = nota alta + review rica
    promoter = (
        filtered.filter(pl.col("score") >= 4)
        .sort(["aspect_plot", "aspect_style", "aspect_pacing"], descending=True)
        .head(1)
    )

    # detrator = nota baixa + review rica
    detractor = (
        filtered.filter(pl.col("score") <= 2)
        .sort(["aspect_plot", "aspect_style", "aspect_pacing"], descending=True)
        .head(1)
    )

    col1, col2 = st.columns(2)

    for col, data, title in [
        (col1, promoter, "😊 Promotor"),
        (col2, detractor, "😡 Detrator"),
    ]:
        with col:
            if data.height:
                r = data.to_dicts()[0]

                user = r["profileName"] or r["User_id"]

                st.subheader(title)

                st.write(
                    f"""
                    **Usuário:** {user}

                    **Livro:** {r["Title"]}

                    **Nota:** ⭐ {r["score"]}
                    """
                )

                st.info(f'"{r["text"]}"\n\n— {user}')
