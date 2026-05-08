from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from whodunit_stylometry.analysis.classical import (
    ClassicalAnalysisConfig,
    add_tokens,
    classify_text_with_mendenhall,
    compare_mendenhall_average_curves,
    compute_mendenhall_author_profiles,
    corpus_summary,
    leave_one_work_out_classification,
    load_corpus,
    mendenhall_average_curves_to_frame,
    top_tokens_by_author,
)

ANALYSIS_TYPES = {
    "Métodos clásicos": "classical",
    "Métodos de machine learning": "ml",
    "Métodos no supervisados": "unsupervised",
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

SIDEBAR_ICON_PATH = Path("logo.png")


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


st.title("Whodunit Stylometry 🕵")

with st.sidebar:
    _, logo_col, _ = st.columns([1, 3, 1])
    with logo_col:
        st.image(SIDEBAR_ICON_PATH, width=120)

    st.header("Corpus")
    default_path = str(Path("/Users/my_user/corpus"))
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

    if "analysis_has_run" not in st.session_state:
        st.session_state.analysis_has_run = False

    if st.button("Ejecutar análisis", type="primary", width="stretch"):
        st.session_state.analysis_has_run = True


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
    seed = 1

    if selected_method == "mendenhall":
        st.caption("Parámetros de curvas características")
        block_size = st.number_input("BLOCK_SIZE", min_value=100, max_value=1_000_000, value=100_000, step=5_000)
        n_blocks = st.number_input("N_BLOCKS", min_value=1, max_value=500, value=50, step=1)
        max_word_len = st.slider("Longitud máxima de palabra", min_value=10, max_value=40, value=20, step=1)
        seed = st.number_input("Semilla aleatoria", min_value=0, max_value=10_000, value=1, step=1)
    elif selected_method in {"kilgariff", "burrows"}:
        st.caption("Parámetros léxicos")
        use_function_words = st.toggle("Usar solo palabras funcionales", value=True)
        vocab_size = st.slider("Tamaño del vocabulario", min_value=50, max_value=1500, value=500, step=50)
        min_freq = st.number_input("Frecuencia mínima", min_value=1, max_value=1000, value=1, step=1)

    top_n = 20
    if selected_method in {"kilgariff", "burrows"}:
        st.subheader("Visualización")
        top_n = st.slider("Top palabras por autor", min_value=5, max_value=50, value=20, step=5)


if not st.session_state.analysis_has_run:
    st.write(
        "Esta aplicación permite explorar un corpus literario mediante técnicas de estilometría "
        "para comparar autores, analizar patrones de escritura y apoyar tareas de atribución de autoría."
    )
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

if selected_method == "mendenhall":
    blocks_tab, average_tab, attribution_tab = st.tabs(
        [
            "Curvas características por bloques",
            "Curva característica media por autor",
            "Test de atribución",
        ]
    )

    try:
        with st.spinner("Calculando curvas de Mendenhall por autor..."):
            block_curves_df, mendenhall_stats_df, average_curves_df, average_curves = (
                compute_mendenhall_author_profiles(corpus_df, config)
            )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with blocks_tab:
        st.subheader("Curvas características por bloques")
        block_curves_df.to_csv("block_curves.csv", index=False)
        block_fig = px.line(
            block_curves_df,
            x="word_length",
            y="relative_frequency",
            color="author",
            line_group="block",
            facet_col="author",
            facet_col_wrap=2,
            labels={
                "word_length": "Longitud de palabra",
                "relative_frequency": "Frecuencia relativa",
                "author": "Autor",
            },
            title=f"Curvas características por bloques de {block_size} tokens",
        )
        block_fig.update_traces(opacity=0.35, line_width=1)
        block_fig.update_layout(showlegend=False, height=900)
        st.plotly_chart(block_fig, use_container_width=True)

        st.subheader("Estabilidad interna por autor")
        st.dataframe(
            mendenhall_stats_df.style.format({"mean_distance": "{:.5f}", "std_distance": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with average_tab:
        st.subheader("Curvas características medias")
        average_fig = px.line(
            average_curves_df,
            x="word_length",
            y="relative_frequency",
            color="author",
            markers=True,
            labels={
                "word_length": "Longitud de palabra",
                "relative_frequency": "Frecuencia relativa media",
                "author": "Autor",
            },
            title="Curvas características medias por autor",
        )
        st.plotly_chart(average_fig, use_container_width=True)

        st.subheader("Distancia Jensen-Shannon entre curvas medias")
        comparison_df = compare_mendenhall_average_curves(average_curves, max_word_len=max_word_len)
        st.dataframe(comparison_df.style.format("{:.5f}"), use_container_width=True)

    with attribution_tab:
        st.subheader("Atribuir una obra externa")

        uploaded_file = st.file_uploader(
            "Selecciona una obra en .txt",
            type=["txt"],
            key="attribution_uploaded_txt",
        )

        if "attribution_text" not in st.session_state:
            st.session_state.attribution_text = None
        if "attribution_filename" not in st.session_state:
            st.session_state.attribution_filename = None

        if uploaded_file is not None:
            st.session_state.attribution_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
            st.session_state.attribution_filename = uploaded_file.name

        if st.session_state.attribution_text is None:
            st.info("Sube un archivo .txt para comparar su curva de Mendenhall con los autores del corpus.")
        else:
            text = st.session_state.attribution_text
            uploaded_filename = st.session_state.attribution_filename or "obra externa"

            st.success(f"Archivo cargado: {uploaded_filename}")
            st.write("Tamaño del texto:", len(text), "caracteres")

            try:
                distances_df, uploaded_curve_df = classify_text_with_mendenhall(text, average_curves, config)
            except ValueError as exc:
                st.error(str(exc))
            else:
                predicted_author = distances_df.iloc[0]["author"]
                predicted_distance = distances_df.iloc[0]["distance"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Autor más similar", predicted_author)
                with col2:
                    st.metric("Distancia", f"{predicted_distance:.5f}")

                comparison_curve_df = pd.concat(
                    [
                        mendenhall_average_curves_to_frame(
                            average_curves,
                            max_word_len=max_word_len,
                        ),
                        uploaded_curve_df,
                    ],
                    ignore_index=True,
                )
                attribution_fig = px.line(
                    comparison_curve_df,
                    x="word_length",
                    y="relative_frequency",
                    color="author",
                    markers=True,
                    labels={
                        "word_length": "Longitud de palabra",
                        "relative_frequency": "Frecuencia relativa media",
                        "author": "Curva",
                    },
                    title=f"Comparación de {uploaded_filename} con las curvas medias",
                )
                st.plotly_chart(attribution_fig, use_container_width=True)
                st.subheader("Ranking de autores")
                st.dataframe(
                    distances_df[["rank", "author", "distance"]].style.format({"distance": "{:.5f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

    st.stop()

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
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_words, use_container_width=True, hide_index=True)

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
        st.dataframe(results_df, use_container_width=True, hide_index=True)
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
        st.dataframe(filtered, use_container_width=True, hide_index=True)
