from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from whodunit_stylometry.analysis.classical import (
    ClassicalAnalysisConfig,
    add_tokens,
    corpus_summary,
    leave_one_work_out_classification,
    load_corpus,
    top_tokens_by_author,
)

ANALYSIS_TYPES = {
    "Métodos clásicos": "classical",
    "Métodos de machine learning": "ml",
    "Métodos no supervisados": "unsupervised",
    "Transformers y explicabilidad": "transformers",
}

CLASSICAL_METHODS = {
    "Test de Mendenhall": "mendenhall",
    "Chi-cuadrado de Kilgariff": "kilgariff",
    "Distancia de Burrows": "burrows",
}

METHOD_NAMES = {
    "mendenhall": "Test de Mendenhall",
    "kilgariff": "Chi-cuadrado de Kilgariff",
    "burrows": "Distancia de Burrows",
}


st.set_page_config(
    page_title="Whodunit Stylometry",
    page_icon="WS",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_load_and_tokenize(corpus_path: str, lowercase: bool) -> pd.DataFrame:
    return add_tokens(load_corpus(corpus_path), lowercase=lowercase)


def cached_classical_analysis(
    df: pd.DataFrame,
    selected_method: str,
    vocab_size: int,
    min_freq: int,
    use_function_words: bool,
    lowercase: bool,
    block_size: int,
    n_blocks: int,
    max_word_len: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = ClassicalAnalysisConfig(
        vocab_size=vocab_size,
        min_freq=min_freq,
        use_function_words=use_function_words,
        lowercase=lowercase,
        block_size=block_size,
        n_blocks=n_blocks,
        max_word_len=max_word_len,
        seed=seed,
    )

    return leave_one_work_out_classification(df, method=selected_method, config=config)


def show_placeholder(title: str) -> None:
    st.info(f"{title} todavía no está implementado en la app. La interfaz ya queda preparada para añadirlo.")


def metric_card(label: str, value) -> None:
    st.metric(label=label, value=value)


st.title("Whodunit Stylometry")

with st.sidebar:
    st.header("Corpus")
    default_path = str(Path("corpus/hand_cleaned"))
    corpus_path = st.text_input(
        "Ruta al corpus",
        value=default_path,
        help="Estructura esperada: una carpeta por autor y archivos .txt dentro de cada carpeta.",
    )

    st.header("Tipo de análisis")
    selected_analysis_label = st.radio("Selecciona una familia", list(ANALYSIS_TYPES.keys()))
    selected_analysis = ANALYSIS_TYPES[selected_analysis_label]

    st.header("Opciones")
    lowercase = st.toggle("Normalizar a minúsculas", value=True)

    run_button = st.button("Ejecutar análisis", type="primary", width="stretch")


if selected_analysis != "classical":
    show_placeholder(selected_analysis_label)
    st.stop()

with st.sidebar:
    st.subheader("Métodos clásicos")
    selected_method_label = st.selectbox(
        "Método",
        options=list(CLASSICAL_METHODS.keys()),
    )
    selected_method = CLASSICAL_METHODS[selected_method_label]

    use_function_words = True
    vocab_size = 500
    min_freq = 1
    block_size = 100_000
    n_blocks = 50
    max_word_len = 20
    seed = 0

    if selected_method == "mendenhall":
        st.caption("Parámetros de curvas características")
        block_size = st.number_input("BLOCK_SIZE", min_value=100, max_value=1_000_000, value=100_000, step=5_000)
        n_blocks = st.number_input("N_BLOCKS", min_value=1, max_value=500, value=50, step=1)
        max_word_len = st.slider("Longitud máxima de palabra", min_value=10, max_value=40, value=20, step=1)
        seed = st.number_input("Semilla aleatoria", min_value=0, max_value=10_000, value=0, step=1)
    elif selected_method in {"kilgariff", "burrows"}:
        st.caption("Parámetros léxicos")
        use_function_words = st.toggle("Usar solo palabras funcionales", value=True)
        vocab_size = st.slider("Tamaño del vocabulario", min_value=50, max_value=1500, value=500, step=50)
        min_freq = st.number_input("Frecuencia mínima", min_value=1, max_value=1000, value=1, step=1)

    st.subheader("Visualización")
    top_n = st.slider("Top palabras por autor", min_value=5, max_value=50, value=20, step=5)


if not run_button:
    st.info("Selecciona una ruta de corpus y pulsa Ejecutar análisis.")
    st.stop()

try:
    with st.spinner("Leyendo y tokenizando el corpus..."):
        corpus_df = cached_load_and_tokenize(corpus_path, lowercase=lowercase)
except Exception as exc:
    st.error(str(exc))
    st.stop()

work_summary, author_summary = corpus_summary(corpus_df)

summary_cols = st.columns(4)
with summary_cols[0]:
    metric_card("Autores", author_summary.shape[0])
with summary_cols[1]:
    metric_card("Obras", work_summary.shape[0])
with summary_cols[2]:
    metric_card("Tokens", f"{int(author_summary['total_tokens'].sum()):,}")
with summary_cols[3]:
    metric_card("Media tokens/obra", f"{work_summary['token_count'].mean():,.0f}")

overview_tab, lexical_tab, attribution_tab, details_tab = st.tabs(
    ["Corpus", "Rasgos léxicos", "Atribución clásica", "Contribuciones"]
)

with overview_tab:
    left, right = st.columns([0.55, 0.45])
    with left:
        st.subheader("Resumen por autor")
        st.dataframe(author_summary, width="stretch", hide_index=True)
    with right:
        fig = px.bar(
            author_summary.sort_values("total_tokens"),
            x="total_tokens",
            y="author",
            orientation="h",
            labels={"total_tokens": "Tokens", "author": "Autor"},
            title="Tamaño del corpus por autor",
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Obras detectadas")
    st.dataframe(work_summary, width="stretch", hide_index=True)

with lexical_tab:
    lexical_uses_function_words = selected_method in {"kilgariff", "burrows"} and use_function_words
    top_words = top_tokens_by_author(corpus_df, top_n=top_n, use_function_words=lexical_uses_function_words)
    st.subheader("Palabras más frecuentes por autor")

    selected_author = st.selectbox("Autor", sorted(top_words["author"].unique()))
    author_top_words = top_words[top_words["author"] == selected_author]
    fig = px.bar(
        author_top_words.sort_values("relative_frequency"),
        x="relative_frequency",
        y="token",
        orientation="h",
        labels={"relative_frequency": "Frecuencia relativa", "token": "Token"},
        title=selected_author,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(top_words, width="stretch", hide_index=True)

with attribution_tab:
    with st.spinner(f"Calculando atribución con {selected_method_label}..."):
        results_df, contributions_df = cached_classical_analysis(
            corpus_df,
            selected_method=selected_method,
            vocab_size=vocab_size,
            min_freq=min_freq,
            use_function_words=use_function_words,
            lowercase=lowercase,
            block_size=block_size,
            n_blocks=n_blocks,
            max_word_len=max_word_len,
            seed=seed,
        )

    ok_results = results_df[results_df["status"] == "ok"].copy()
    if ok_results.empty:
        st.warning("No se han podido calcular resultados de atribución con esta configuración.")
        st.dataframe(results_df, width="stretch", hide_index=True)
    else:
        accuracy = round(ok_results["correct"].mean() * 100, 2)
        mean_margin = ok_results["margin_to_second"].mean()
        metrics_by_method = pd.DataFrame(
            [
                {
                    "method": METHOD_NAMES[selected_method],
                    "accuracy": accuracy,
                    "n_works": ok_results.shape[0],
                    "mean_margin": mean_margin,
                }
            ]
        )

        col1, col2 = st.columns([0.35, 0.65])
        with col1:
            st.subheader("Rendimiento")
            st.dataframe(metrics_by_method, width="stretch", hide_index=True)
        with col2:
            fig = px.bar(
                metrics_by_method,
                x="method",
                y="accuracy",
                text="accuracy",
                labels={"method": "Método", "accuracy": "Exactitud (%)"},
                range_y=[0, 100],
                title="Exactitud por método",
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Clasificación por obra")
        visible_cols = [
            "method",
            "author",
            "work",
            "pred_author",
            "correct",
            "min_distance",
            "second_distance",
            "margin_to_second",
            "status",
        ]
        st.dataframe(results_df[visible_cols], width="stretch", hide_index=True)

with details_tab:
    if "contributions_df" not in locals() or contributions_df.empty:
        st.info("Ejecuta la atribución para ver contribuciones por token.")
    else:
        st.subheader("Tokens que más explican la distancia con el autor predicho")
        methods = sorted(contributions_df["method"].dropna().unique())
        selected_detail_method = st.selectbox(
            "Método", methods, format_func=lambda value: METHOD_NAMES.get(value, value)
        )
        works = sorted(
            contributions_df.loc[contributions_df["method"] == selected_detail_method, "work"].dropna().unique()
        )
        selected_work = st.selectbox("Obra", works)
        filtered = contributions_df[
            (contributions_df["method"] == selected_detail_method) & (contributions_df["work"] == selected_work)
        ]
        st.dataframe(filtered, width="stretch", hide_index=True)
