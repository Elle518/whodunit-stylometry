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
    "Delta de Burrows": "burrows",
    "Chi-cuadrado de Kilgariff": "kilgariff",
}


st.set_page_config(
    page_title="Whodunit Stylometry",
    page_icon="WS",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_load_and_tokenize(corpus_path: str, lowercase: bool) -> pd.DataFrame:
    return add_tokens(load_corpus(corpus_path), lowercase=lowercase)


@st.cache_data(show_spinner=False)
def cached_classical_analysis(
    df: pd.DataFrame,
    selected_methods: tuple[str, ...],
    vocab_size: int,
    min_freq: int,
    use_function_words: bool,
    lowercase: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = ClassicalAnalysisConfig(
        vocab_size=vocab_size,
        min_freq=min_freq,
        use_function_words=use_function_words,
        lowercase=lowercase,
    )

    result_frames = []
    contribution_frames = []
    for method in selected_methods:
        results, contributions = leave_one_work_out_classification(df, method=method, config=config)
        result_frames.append(results)
        contribution_frames.append(contributions)

    results_df = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    contributions_df = pd.concat(contribution_frames, ignore_index=True) if contribution_frames else pd.DataFrame()
    return results_df, contributions_df


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

    run_button = st.button("Ejecutar análisis", type="primary", use_container_width=True)


if selected_analysis != "classical":
    show_placeholder(selected_analysis_label)
    st.stop()

with st.sidebar:
    st.subheader("Métodos clásicos")
    selected_method_labels = st.multiselect(
        "Métodos",
        options=list(CLASSICAL_METHODS.keys()),
        default=list(CLASSICAL_METHODS.keys()),
    )
    selected_methods = tuple(CLASSICAL_METHODS[label] for label in selected_method_labels)
    use_function_words = st.toggle("Usar solo palabras funcionales", value=True)
    vocab_size = st.slider("Tamaño del vocabulario", min_value=50, max_value=1500, value=500, step=50)
    min_freq = st.number_input("Frecuencia mínima", min_value=1, max_value=1000, value=1, step=1)
    top_n = st.slider("Top palabras por autor", min_value=5, max_value=50, value=20, step=5)


if not run_button:
    st.info("Selecciona una ruta de corpus y pulsa Ejecutar análisis.")
    st.stop()

if not selected_methods:
    st.warning("Selecciona al menos un método clásico.")
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
        st.dataframe(author_summary, use_container_width=True, hide_index=True)
    with right:
        fig = px.bar(
            author_summary.sort_values("total_tokens"),
            x="total_tokens",
            y="author",
            orientation="h",
            labels={"total_tokens": "Tokens", "author": "Autor"},
            title="Tamaño del corpus por autor",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Obras detectadas")
    st.dataframe(work_summary, use_container_width=True, hide_index=True)

with lexical_tab:
    top_words = top_tokens_by_author(corpus_df, top_n=top_n, use_function_words=use_function_words)
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
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_words, use_container_width=True, hide_index=True)

with attribution_tab:
    with st.spinner("Calculando atribución leave-one-work-out..."):
        results_df, contributions_df = cached_classical_analysis(
            corpus_df,
            selected_methods=selected_methods,
            vocab_size=vocab_size,
            min_freq=min_freq,
            use_function_words=use_function_words,
            lowercase=lowercase,
        )

    ok_results = results_df[results_df["status"] == "ok"].copy()
    if ok_results.empty:
        st.warning("No se han podido calcular resultados de atribución con esta configuración.")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        metrics_by_method = (
            ok_results.groupby("method", as_index=False)
            .agg(
                accuracy=("correct", "mean"),
                n_works=("work", "count"),
                mean_margin=("margin_to_second", "mean"),
            )
            .sort_values("accuracy", ascending=False)
        )
        metrics_by_method["accuracy"] = metrics_by_method["accuracy"].map(lambda value: round(value * 100, 2))

        col1, col2 = st.columns([0.35, 0.65])
        with col1:
            st.subheader("Rendimiento")
            st.dataframe(metrics_by_method, use_container_width=True, hide_index=True)
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
            st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(results_df[visible_cols], use_container_width=True, hide_index=True)

with details_tab:
    if "contributions_df" not in locals() or contributions_df.empty:
        st.info("Ejecuta la atribución para ver contribuciones por token.")
    else:
        st.subheader("Tokens que más explican la distancia con el autor predicho")
        methods = sorted(contributions_df["method"].dropna().unique())
        selected_method = st.selectbox("Método", methods)
        works = sorted(contributions_df.loc[contributions_df["method"] == selected_method, "work"].dropna().unique())
        selected_work = st.selectbox("Obra", works)
        filtered = contributions_df[
            (contributions_df["method"] == selected_method) & (contributions_df["work"] == selected_work)
        ]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
