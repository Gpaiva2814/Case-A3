import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PATH = "data/cleaned_reviews.parquet"
OUTPUT_PATH = "data/cleaned_reviews_mapped.parquet"
LIMIAR_CONFIANCA = 0.30


def main():
    lf = pl.scan_parquet(INPUT_PATH)

    print("Iniciando a normalização das categorias...")

    # Como 'categories' já é uma lista pura (list[str]), nós damos o explode direto!
    lf_normalizado = (
        lf.explode(
            "categories"
        )  # Separa os itens da lista em linhas (listas vazias viram null aqui)
        .drop_nulls(
            subset=["categories"]
        )  # <-- ISSO DAQUI APAGA OS NULOS GERADOS PELAS LISTAS VAZIAS!
        .with_columns(
            pl.col("categories")
            .str.to_lowercase()  # Tudo em minúsculo
            .str.replace_all(r"[\t\n\r]", " ")  # Remove tabulações
            .str.replace_all(
                r"[^a-zA-Z0-9\s\-_]", ""
            )  # Limpa caracteres especiais restantes
            .str.strip_chars()  # Remove espaços extras
        )
        .filter(
            pl.col("categories") != ""
        )  # Garante que strings que ficaram totalmente vazias ("") também sumam
    )

    print("Definindo categorias âncoras...")
    # Coleta o top 50 de categorias mais frequentes baseado na contagem real
    ancoras = (
        lf_normalizado.select(pl.col("categories"))
        .collect()
        .to_series()
        .value_counts(sort=True)["categories"]
        .to_list()[:50]
    )

    print("Extraindo termos únicos para mapeamento...")
    termos_caoticos = (
        lf_normalizado.select(pl.col("categories"))
        .collect()
        .to_series()
        .unique()
        .to_list()
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(
        f"Vetorizando {len(termos_caoticos)} termos únicos contra {len(ancoras)} âncoras..."
    )
    embeddings_ancoras = model.encode(ancoras)
    embeddings_caoticos = model.encode(termos_caoticos)

    matriz_similaridade = cosine_similarity(embeddings_caoticos, embeddings_ancoras)
    indices_maior_similaridade = np.argmax(matriz_similaridade, axis=1)
    scores_maior_similaridade = np.max(matriz_similaridade, axis=1)

    mapeamento_dict = {}
    for i, termo in enumerate(termos_caoticos):
        score = scores_maior_similaridade[i]

        if score >= LIMIAR_CONFIANCA:
            categoria_correta = ancoras[indices_maior_similaridade[i]]
        else:
            # Se o modelo ficou confuso, mandamos para o unknown por segurança
            categoria_correta = "unknown"

        mapeamento_dict[termo] = categoria_correta

    print("\nAplicando o mapeamento final e salvando...")

    lf_final = lf_normalizado.with_columns(
        pl.col("categories").replace(mapeamento_dict).alias("categories_clean")
    )

    lf_final.sink_parquet(OUTPUT_PATH)

    print(f"Sucesso! O arquivo foi tratado, limpo e salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
