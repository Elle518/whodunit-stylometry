"""Application for stylometric analysis of literary corpora, built with Streamlit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from whodunit_stylometry.analysis.classical import (
    ClassicalAnalysisConfig,
    burrows_reference_tables,
    classify_text_with_burrows,
    classify_text_with_kilgariff,
    classify_text_with_mendenhall,
    compare_mendenhall_average_curves,
    compute_mendenhall_author_profiles,
    kilgariff_reference_tables,
    mendenhall_average_curves_to_frame,
    top_tokens_by_author,
)
from whodunit_stylometry.analysis.eda import (
    LEXICAL_METRICS,
    PUNCTUATION_METRICS,
    QUALITY_METRICS,
    STRUCTURAL_METRICS,
    EDAConfig,
    run_eda_analysis,
)
from whodunit_stylometry.analysis.embeddings import (
    EMBEDDING_MODELS,
    EmbeddingsConfig,
    estimate_embedding_chunks,
    run_openai_embeddings_experiment,
)
from whodunit_stylometry.analysis.supervised import (
    FEATURE_SETS,
    MODEL_NAMES,
    SupervisedConfig,
    classify_text_with_supervised_model,
    run_mfw_robustness,
    run_supervised_experiment,
)
from whodunit_stylometry.analysis.unsupervised import (
    HIERARCHICAL_METHOD_NAMES,
    UNSUPERVISED_MODEL_NAMES,
    HierarchicalConfig,
    UnsupervisedConfig,
    build_cluster_projection,
    build_dendrogram_figure,
    run_hierarchical_experiment,
    run_unsupervised_experiment,
    run_unsupervised_mfw_robustness,
)
from whodunit_stylometry.utils.data_utils import corpus_summary
from whodunit_stylometry.utils.st_utils import (
    ANALYSIS_TYPES,
    CLASSICAL_METHODS,
    ML_WORKFLOWS,
    SIDEBAR_ICON_PATH,
    UNSUPERVISED_WORKFLOWS,
    cached_load_and_tokenize,
    corpus_cache_fingerprint,
    eda_cache_key,
    embeddings_experiment_cache_key,
    hierarchical_experiment_cache_key,
    metric_card,
    parse_int_list,
    supervised_experiment_cache_key,
    unsupervised_experiment_cache_key,
)

######################
# PAGE CONFIGURATION #
######################
st.set_page_config(
    page_title="Whodunit Stylometry",
    page_icon="WS",
    layout="wide",
)

st.title("Whodunit Stylometry 🕵")

#################################
# GENERAL SIDEBAR CONFIGURATION #
#################################
with st.sidebar:
    _, logo_col, _ = st.columns([1, 3, 1])
    with logo_col:
        st.image(SIDEBAR_ICON_PATH, width=120)

    st.header("Corpus")
    default_path = str(Path("/Users/elle/Desktop/WDI/corpus"))
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


##########################################
# SIDEBAR OPTIONS FOR EACH ANALYSIS TYPE #
##########################################
if selected_analysis == "eda":
    with st.sidebar:
        st.subheader("Análisis exploratorio")
        eda_top_n = st.slider("Top n-grams", min_value=5, max_value=50, value=20, step=5)
        eda_zipf_max_rank = st.slider("Máximo rango Zipf", min_value=500, max_value=10000, value=5000, step=500)


#### REVISAR #####
if selected_analysis == "embeddings":
    with st.sidebar:
        st.subheader("Embeddings de OpenAI")
        embedding_model = st.selectbox("Modelo", list(EMBEDDING_MODELS.keys()), index=0)
        dimension_options = ["Nativa", "512", "1024", "1536"]
        if embedding_model == "text-embedding-3-large":
            dimension_options.append("3072")
        embedding_dimensions_label = st.selectbox("Dimensiones", dimension_options, index=0)
        embedding_dimensions = None if embedding_dimensions_label == "Nativa" else int(embedding_dimensions_label)

        st.caption("Segmentación")
        embedding_chunk_tokens = st.slider("CHUNK_TOKENS", min_value=200, max_value=2000, value=1200, step=100)
        embedding_chunk_overlap = st.slider("CHUNK_OVERLAP", min_value=0, max_value=500, value=150, step=25)
        embedding_min_chunk_tokens = st.slider("MIN_CHUNK_TOKENS", min_value=20, max_value=500, value=120, step=20)

        st.caption("API y visualización")
        embedding_batch_size = st.slider("BATCH_SIZE", min_value=1, max_value=128, value=64, step=1)
        embedding_max_retries = st.number_input("MAX_RETRIES", min_value=1, max_value=10, value=6, step=1)
        embedding_random_state = st.number_input("Semilla", min_value=0, max_value=10_000, value=42, step=1)
        embedding_umap_neighbors = st.slider("UMAP n_neighbors", min_value=2, max_value=50, value=10, step=1)
        embedding_umap_min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=0.99, value=0.08, step=0.01)
        embedding_network_top_k = st.slider("Top-k red", min_value=1, max_value=10, value=3, step=1)
        embedding_nn_k = st.slider("Vecinos por obra", min_value=1, max_value=15, value=6, step=1)
        openai_api_key = st.text_input("OpenAI API key", type="password")

if selected_analysis == "unsupervised":
    with st.sidebar:
        st.subheader("Machine learning no supervisado")
        selected_unsup_workflow_label = st.selectbox("Flujo", list(UNSUPERVISED_WORKFLOWS.keys()))
        selected_unsup_workflow = UNSUPERVISED_WORKFLOWS[selected_unsup_workflow_label]

        unsup_feature_label = "MFW"
        unsup_feature_set = FEATURE_SETS[unsup_feature_label]
        unsup_model_labels = list(UNSUPERVISED_MODEL_NAMES.keys())
        unsup_model_names = tuple(UNSUPERVISED_MODEL_NAMES[label] for label in unsup_model_labels)
        unsup_top_n_mfw = 50
        unsup_seed = 42
        unsup_n_clusters = None
        unsup_top_n_values = [10, 25, 50]
        unsup_seeds = [0, 42, 123]
        hier_method_labels = list(HIERARCHICAL_METHOD_NAMES.keys())
        hier_method_names = tuple(HIERARCHICAL_METHOD_NAMES[label] for label in hier_method_labels)
        hier_dendrogram_method_label = "Ward"
        hier_dendrogram_method = HIERARCHICAL_METHOD_NAMES[hier_dendrogram_method_label]

        if selected_unsup_workflow == "experiment":
            unsup_feature_label = st.selectbox("Conjunto de rasgos", list(FEATURE_SETS.keys()), index=1)
            unsup_feature_set = FEATURE_SETS[unsup_feature_label]
            unsup_model_labels = st.multiselect(
                "Modelos",
                list(UNSUPERVISED_MODEL_NAMES.keys()),
                default=list(UNSUPERVISED_MODEL_NAMES.keys()),
            )
            unsup_model_names = tuple(UNSUPERVISED_MODEL_NAMES[label] for label in unsup_model_labels)
            if unsup_feature_set in {"mfw", "combined"}:
                unsup_top_n_mfw = st.slider("TOP_N_MFW", min_value=5, max_value=200, value=50, step=5)
            unsup_seed = st.number_input("Semilla", min_value=0, max_value=10_000, value=42, step=1)
            use_author_count = st.toggle("Usar un cluster por autor", value=True)
            if not use_author_count:
                unsup_n_clusters = st.number_input("Número de clusters", min_value=2, max_value=50, value=6, step=1)
        elif selected_unsup_workflow == "robustness":
            robustness_preset = st.selectbox("Preset", ["Rápido", "Completo"])
            if robustness_preset == "Rápido":
                unsup_top_n_values = [10, 25, 50]
                unsup_seeds = [0, 42, 123]
            else:
                unsup_top_n_values = [5, 10, 15, 25, 30, 40, 50, 75, 100, 125, 150]
                unsup_seeds = [0, 1, 2, 3, 4, 5, 10, 20, 42, 123, 150]
            unsup_top_n_values = parse_int_list(
                st.text_input("Valores TOP_N_MFW", value=", ".join(map(str, unsup_top_n_values)))
            )
            unsup_seeds = parse_int_list(st.text_input("Semillas", value=", ".join(map(str, unsup_seeds))))
            unsup_model_labels = st.multiselect(
                "Modelos",
                list(UNSUPERVISED_MODEL_NAMES.keys()),
                default=list(UNSUPERVISED_MODEL_NAMES.keys()),
            )
            unsup_model_names = tuple(UNSUPERVISED_MODEL_NAMES[label] for label in unsup_model_labels)
        else:
            unsup_feature_label = st.selectbox("Conjunto de rasgos", list(FEATURE_SETS.keys()), index=1)
            unsup_feature_set = FEATURE_SETS[unsup_feature_label]
            if unsup_feature_set in {"mfw", "combined"}:
                unsup_top_n_mfw = st.slider("TOP_N_MFW", min_value=5, max_value=200, value=50, step=5)
            hier_method_labels = st.multiselect(
                "Métodos de enlace",
                list(HIERARCHICAL_METHOD_NAMES.keys()),
                default=list(HIERARCHICAL_METHOD_NAMES.keys()),
            )
            hier_method_names = tuple(HIERARCHICAL_METHOD_NAMES[label] for label in hier_method_labels)
            hier_dendrogram_method_label = st.selectbox(
                "Método del dendrograma",
                list(HIERARCHICAL_METHOD_NAMES.keys()),
                index=0,
            )
            hier_dendrogram_method = HIERARCHICAL_METHOD_NAMES[hier_dendrogram_method_label]
            use_author_count = st.toggle("Usar un cluster por autor", value=True)
            if not use_author_count:
                unsup_n_clusters = st.number_input("Número de clusters", min_value=2, max_value=50, value=6, step=1)

if selected_analysis == "supervised":
    with st.sidebar:
        st.subheader("Machine learning supervisado")
        selected_ml_workflow_label = st.selectbox("Flujo", list(ML_WORKFLOWS.keys()))
        selected_ml_workflow = ML_WORKFLOWS[selected_ml_workflow_label]

        ml_feature_label = "MFW"
        ml_feature_set = FEATURE_SETS[ml_feature_label]
        ml_model_labels = ["Regresión logística", "Linear SVC"]
        ml_model_names = tuple(MODEL_NAMES[label] for label in ml_model_labels)
        ml_model_label = "Regresión logística"
        ml_model_name = MODEL_NAMES[ml_model_label]
        ml_top_n_mfw = 50
        ml_seed = 42
        ml_n_test_per_author = 2
        ml_keep_correlated = False
        ml_corr_threshold = 0.85
        ml_tolerance = 0.01
        ml_top_n_values = [5, 10, 15, 25, 50, 75, 100]
        ml_seeds = [0, 1, 2, 42, 123]

        if selected_ml_workflow == "experiment":
            ml_feature_label = st.selectbox("Conjunto de rasgos", list(FEATURE_SETS.keys()), index=1)
            ml_feature_set = FEATURE_SETS[ml_feature_label]
            ml_model_labels = st.multiselect(
                "Modelos",
                list(MODEL_NAMES.keys()),
                default=list(MODEL_NAMES.keys()),
            )
            ml_model_names = tuple(MODEL_NAMES[label] for label in ml_model_labels)
            ml_top_n_mfw = st.slider("TOP_N_MFW", min_value=5, max_value=200, value=50, step=5)
            ml_seed = st.number_input("Semilla", min_value=0, max_value=10_000, value=42, step=1)
            ml_n_test_per_author = st.number_input("Obras de test por autor", min_value=1, max_value=5, value=2, step=1)
            if ml_feature_set in {"stylometric", "combined"}:
                ml_keep_correlated = st.toggle("Mantener rasgos correlacionados", value=False)
                ml_corr_threshold = st.slider("Umbral de correlación", 0.50, 0.99, 0.85, 0.01)
        elif selected_ml_workflow == "robustness":
            robustness_preset = st.selectbox("Preset", ["Rápido", "Completo"])
            if robustness_preset == "Rápido":
                ml_top_n_values = [10, 25, 50]
                ml_seeds = [0, 42, 123]
            else:
                ml_top_n_values = [5, 10, 15, 25, 30, 40, 50, 75, 100, 125, 150]
                ml_seeds = [0, 1, 2, 3, 4, 5, 10, 20, 42, 123, 150]
            ml_top_n_values = parse_int_list(
                st.text_input("Valores TOP_N_MFW", value=", ".join(map(str, ml_top_n_values)))
            )
            ml_seeds = parse_int_list(st.text_input("Semillas", value=", ".join(map(str, ml_seeds))))
            ml_model_labels = st.multiselect(
                "Modelos",
                list(MODEL_NAMES.keys()),
                default=["Regresión logística", "Linear SVC"],
            )
            ml_model_names = tuple(MODEL_NAMES[label] for label in ml_model_labels)
            ml_n_test_per_author = st.number_input("Obras de test por autor", min_value=1, max_value=5, value=2, step=1)
            ml_tolerance = st.slider("Tolerancia desde el mejor F1", 0.0, 0.10, 0.01, 0.005)
        else:
            ml_feature_label = st.selectbox("Conjunto de rasgos", list(FEATURE_SETS.keys()), index=1)
            ml_feature_set = FEATURE_SETS[ml_feature_label]
            ml_model_label = st.selectbox("Modelo final", list(MODEL_NAMES.keys()), index=0)
            ml_model_name = MODEL_NAMES[ml_model_label]
            ml_top_n_mfw = st.slider("TOP_N_MFW", min_value=5, max_value=200, value=50, step=5)
            ml_seed = st.number_input("Semilla", min_value=0, max_value=10_000, value=42, step=1)

elif selected_analysis == "classical":
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


########################################
# INICIALIZE APPLICATION AND LOAD DATA #
########################################
if not st.session_state.analysis_has_run:
    st.write(
        "Esta aplicación permite explorar un corpus literario mediante técnicas de estilometría "
        "para comparar autores, analizar patrones de escritura y apoyar tareas de atribución de autoría."
    )
    st.info("Selecciona una ruta de corpus y pulsa Ejecutar análisis.")
    st.stop()

try:
    with st.spinner("Leyendo y procesando el corpus, esta operación puede tardar unos minutos..."):
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
    metric_card("Tokens", f"{int(author_summary['total_tokens'].sum()):,}".replace(",", "."))
with summary_cols[3]:
    metric_card("Media tokens/obra", f"{work_summary['token_count'].mean():,.0f}".replace(",", "."))


##################################################
# PERFORM CORPUS EXPLORATORY DATA ANALYSIS (EDA) #
##################################################
if selected_analysis == "eda":
    eda_config = EDAConfig(top_n=int(eda_top_n), zipf_max_rank=int(eda_zipf_max_rank))
    cache_key = eda_cache_key(corpus_df, lowercase, eda_config)
    eda_cache = st.session_state.setdefault("eda_cache", {})
    if cache_key in eda_cache:
        eda_result = eda_cache[cache_key]
    else:
        with st.spinner("Calculando métricas exploratorias del corpus..."):
            eda_result = run_eda_analysis(corpus_df, eda_config)
        eda_cache[cache_key] = eda_result

    metrics_df = eda_result["metrics_df"]
    summary_tab, quality_tab, structure_tab, lexical_tab, punctuation_tab, authors_tab, vocab_tab, outliers_tab = (
        st.tabs(["Resumen", "Calidad", "Estructura", "Léxico", "Puntuación", "Autores", "Vocabulario", "Outliers"])
    )

    with summary_tab:
        st.subheader("Resumen del corpus")
        works_by_author = metrics_df.groupby("author", as_index=False).size().rename(columns={"size": "obras"})
        works_fig = px.bar(
            works_by_author,
            x="author",
            y="obras",
            labels={"author": "Autor", "obras": "Obras"},
            title="Obras por autor",
        )
        st.plotly_chart(works_fig, width="stretch")

        length_fig = px.histogram(
            metrics_df,
            x="n_tokens_all",
            color="author",
            nbins=40,
            labels={"n_tokens_all": "Tokens por obra", "author": "Autor"},
            title="Distribución de longitud de las obras",
        )
        st.plotly_chart(length_fig, width="stretch")
        st.dataframe(
            metrics_df[["author", "work", "filename", "n_tokens_all", "n_chars", "n_sentences", "n_paragraphs"]],
            width="stretch",
            hide_index=True,
        )

    with quality_tab:
        st.subheader("Métricas de calidad superficial")
        quality_metric = st.selectbox("Métrica", [col for col in QUALITY_METRICS if col in metrics_df.columns])
        col1, col2 = st.columns(2)
        with col1:
            quality_hist = px.histogram(
                metrics_df,
                x=quality_metric,
                color="author",
                nbins=30,
                title=f"Distribución de {quality_metric}",
            )
            st.plotly_chart(quality_hist, width="stretch")
        with col2:
            quality_box = px.box(
                metrics_df,
                x="author",
                y=quality_metric,
                points="all",
                title=f"{quality_metric} por autor",
            )
            st.plotly_chart(quality_box, width="stretch")

        quality_corr = metrics_df[[col for col in QUALITY_METRICS if col in metrics_df.columns]].corr()
        quality_corr_fig = px.imshow(quality_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(quality_corr_fig, width="stretch")

    with structure_tab:
        st.subheader("Longitud y estructura textual")
        structural_metric = st.selectbox(
            "Métrica estructural", [col for col in STRUCTURAL_METRICS if col in metrics_df.columns]
        )
        structure_box = px.box(
            metrics_df,
            x="author",
            y=structural_metric,
            points="all",
            title=f"{structural_metric} por autor",
        )
        st.plotly_chart(structure_box, width="stretch")

        scatter_pairs = [
            ("n_tokens_all", "n_sentences"),
            ("n_tokens_all", "n_paragraphs"),
            ("avg_sentence_len", "avg_paragraph_len"),
            ("n_sentences", "n_paragraphs"),
        ]
        selected_pair = st.selectbox("Relación", [f"{x} vs {y}" for x, y in scatter_pairs])
        x_col, y_col = scatter_pairs[[f"{x} vs {y}" for x, y in scatter_pairs].index(selected_pair)]
        scatter_fig = px.scatter(
            metrics_df,
            x=x_col,
            y=y_col,
            color="author",
            hover_data=["work"],
            title=f"{x_col} vs {y_col}",
        )
        st.plotly_chart(scatter_fig, width="stretch")

    with lexical_tab:
        st.subheader("Diversidad y estructura léxica")
        lexical_metric = st.selectbox("Métrica léxica", [col for col in LEXICAL_METRICS if col in metrics_df.columns])
        lexical_box = px.box(
            metrics_df,
            x="author",
            y=lexical_metric,
            points="all",
            title=f"{lexical_metric} por autor",
        )
        st.plotly_chart(lexical_box, width="stretch")

        lexical_scatter = px.scatter(
            metrics_df,
            x="n_tokens_all",
            y=lexical_metric,
            color="author",
            hover_data=["work"],
            title=f"Efecto de longitud sobre {lexical_metric}",
        )
        st.plotly_chart(lexical_scatter, width="stretch")

    with punctuation_tab:
        st.subheader("Uso de puntuación")
        punctuation_metric = st.selectbox(
            "Métrica de puntuación",
            [col for col in PUNCTUATION_METRICS if col in metrics_df.columns],
        )
        punctuation_box = px.box(
            metrics_df,
            x="author",
            y=punctuation_metric,
            points="all",
            title=f"{punctuation_metric} por autor",
        )
        st.plotly_chart(punctuation_box, width="stretch")

        punctuation_corr = metrics_df[[col for col in PUNCTUATION_METRICS if col in metrics_df.columns]].corr()
        punctuation_corr_fig = px.imshow(
            punctuation_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlación entre métricas de puntuación",
        )
        st.plotly_chart(punctuation_corr_fig, width="stretch")

    with authors_tab:
        st.subheader("Patrones agregados por autor")
        heatmap_fig = px.imshow(
            eda_result["author_heatmap_df"],
            color_continuous_scale="RdBu_r",
            zmin=-2,
            zmax=2,
            aspect="auto",
            title="Métricas medias estandarizadas por autor",
        )
        heatmap_fig.update_layout(height=760)
        st.plotly_chart(heatmap_fig, width="stretch")

        pca_df = eda_result["author_pca_df"]
        pca_fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="author",
            text="author",
            title="PCA de autores con métricas estilométricas estables",
        )
        pca_fig.update_traces(textposition="top center")
        st.plotly_chart(pca_fig, width="stretch")
        st.caption(
            "Varianza explicada: "
            + ", ".join(f"PC{i + 1}={value:.3f}" for i, value in enumerate(eda_result["pca_variance"]))
        )

        st.dataframe(eda_result["author_metric_means"], width="stretch", hide_index=True)

    with vocab_tab:
        st.subheader("Vocabulario por autor")
        vocab_author = st.selectbox("Autor", sorted(metrics_df["author"].unique()))
        vocab_mode = st.radio(
            "Vista",
            ["Palabras", "Palabras sin stopwords", "Bigramas", "Trigramas", "Zipf"],
            horizontal=True,
        )
        if vocab_mode == "Palabras":
            vocab_df = eda_result["top_words"].query("author == @vocab_author")
            x_col = "token"
            y_col = "freq"
        elif vocab_mode == "Palabras sin stopwords":
            vocab_df = eda_result["top_words_no_stop"].query("author == @vocab_author")
            x_col = "token"
            y_col = "freq"
        elif vocab_mode == "Bigramas":
            vocab_df = eda_result["top_bigrams"].query("author == @vocab_author")
            x_col = "ngram"
            y_col = "freq"
        elif vocab_mode == "Trigramas":
            vocab_df = eda_result["top_trigrams"].query("author == @vocab_author")
            x_col = "ngram"
            y_col = "freq"
        else:
            zipf_df = eda_result["zipf_df"].query("author == @vocab_author")
            zipf_fig = px.line(
                zipf_df,
                x="rank",
                y="freq",
                log_x=True,
                log_y=True,
                title=f"Curva de Zipf: {vocab_author}",
                labels={"rank": "Rango", "freq": "Frecuencia"},
            )
            st.plotly_chart(zipf_fig, width="stretch")
            st.dataframe(zipf_df.head(eda_top_n), width="stretch", hide_index=True)
            vocab_df = None

        if vocab_df is not None:
            vocab_fig = px.bar(
                vocab_df.sort_values(y_col),
                x=y_col,
                y=x_col,
                orientation="h",
                title=f"{vocab_mode}: {vocab_author}",
            )
            st.plotly_chart(vocab_fig, width="stretch")
            st.dataframe(vocab_df, width="stretch", hide_index=True)

    with outliers_tab:
        st.subheader("Outliers por grupos de métricas")
        group_labels = {
            "quality": "Calidad",
            "structural": "Estructura",
            "lexical": "Léxico",
            "punctuation": "Puntuación",
        }
        selected_group_label = st.selectbox("Grupo", list(group_labels.values()))
        selected_group = {value: key for key, value in group_labels.items()}[selected_group_label]
        outlier_payload = eda_result["outlier_groups"][selected_group]
        st.dataframe(outlier_payload["summary"], width="stretch", hide_index=True)
        st.subheader("Obras marcadas")
        st.dataframe(outlier_payload["outliers"], width="stretch", hide_index=True)

    st.stop()

if selected_analysis == "embeddings":
    embedding_config = EmbeddingsConfig(
        model=embedding_model,
        dimensions=embedding_dimensions,
        chunk_tokens=int(embedding_chunk_tokens),
        chunk_overlap=int(embedding_chunk_overlap),
        min_chunk_tokens=int(embedding_min_chunk_tokens),
        batch_size=int(embedding_batch_size),
        max_retries=int(embedding_max_retries),
        random_state=int(embedding_random_state),
        umap_neighbors=int(embedding_umap_neighbors),
        umap_min_dist=float(embedding_umap_min_dist),
        network_top_k=int(embedding_network_top_k),
        nearest_neighbors_k=int(embedding_nn_k),
    )

    api_tab, chunks_tab, projections_tab, similarity_tab, neighbors_tab, cohesion_tab, network_tab = st.tabs(
        ["API", "Fragmentos", "Proyección", "Similitud", "Vecinos", "Cohesión", "Red"]
    )

    embedding_cache_key = embeddings_experiment_cache_key(corpus_df, lowercase, embedding_config)
    embedding_cache = st.session_state.setdefault("openai_embeddings_cache", {})
    embedding_result = embedding_cache.get(embedding_cache_key)

    with api_tab:
        st.subheader("Generación de embeddings")
        st.caption(
            "La API key se usa solo para crear el cliente en memoria durante esta ejecución. "
            "No se guarda en archivos ni en la caché de resultados."
        )

        if embedding_config.chunk_overlap >= embedding_config.chunk_tokens:
            st.warning(
                "CHUNK_OVERLAP debería ser menor que CHUNK_TOKENS para evitar fragmentos excesivamente solapados."
            )
            st.stop()

        try:
            estimated_chunks_df = estimate_embedding_chunks(corpus_df, embedding_config)
        except Exception as exc:
            st.error(f"No se han podido estimar los fragmentos: {exc}")
            st.stop()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modelo", embedding_config.model)
        with col2:
            st.metric("Dimensiones", embedding_config.dimensions or "Nativa")
        with col3:
            st.metric("Fragmentos estimados", estimated_chunks_df.shape[0])
        with col4:
            estimated_tokens = int(estimated_chunks_df["token_count"].sum()) if not estimated_chunks_df.empty else 0
            st.metric("Tokens en fragmentos", f"{estimated_tokens:,}")

        if estimated_chunks_df.empty:
            st.warning("No se han generado fragmentos con la configuración actual.")
        else:
            chunk_summary = (
                estimated_chunks_df.groupby("author", as_index=False)
                .agg(obras=("work", "nunique"), fragmentos=("chunk_id", "count"), tokens=("token_count", "sum"))
                .sort_values("fragmentos", ascending=False)
            )
            st.dataframe(chunk_summary, width="stretch", hide_index=True)

        generate_embeddings = st.button("Generar embeddings", type="primary", width="stretch")
        if generate_embeddings:
            if not openai_api_key:
                st.error("Introduce una API key de OpenAI para generar los embeddings.")
            elif estimated_chunks_df.empty:
                st.error("No se han generado fragmentos. Ajusta los parámetros de segmentación.")
            else:
                try:
                    with st.spinner("Generando embeddings con OpenAI y calculando métricas..."):
                        embedding_result = run_openai_embeddings_experiment(
                            corpus_df,
                            embedding_config,
                            api_key=openai_api_key,
                        )
                    embedding_cache[embedding_cache_key] = embedding_result
                    st.success("Embeddings generados correctamente.")
                except Exception as exc:
                    st.error(str(exc))

        if embedding_result is None:
            st.info("Genera embeddings para activar las pestañas de visualización, similitud y cohesión.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Dimensión final", embedding_result["embedding_dimension"])
            with col2:
                st.metric("Embeddings de fragmentos", embedding_result["n_api_inputs"])
            with col3:
                st.metric("Obras agregadas", embedding_result["work_embeddings"].shape[0])
            with col4:
                st.metric("Autores agregados", embedding_result["author_embeddings"].shape[0])

    if embedding_result is None:
        with chunks_tab:
            st.info("Genera embeddings en la pestaña API para ver los fragmentos finales.")
        with projections_tab:
            st.info("Genera embeddings en la pestaña API para ver las proyecciones.")
        with similarity_tab:
            st.info("Genera embeddings en la pestaña API para ver las similitudes.")
        with neighbors_tab:
            st.info("Genera embeddings en la pestaña API para ver los vecinos más cercanos.")
        with cohesion_tab:
            st.info("Genera embeddings en la pestaña API para ver métricas de cohesión.")
        with network_tab:
            st.info("Genera embeddings en la pestaña API para ver la red de similitud.")
        st.stop()

    with chunks_tab:
        st.subheader("Fragmentos generados")
        st.dataframe(embedding_result["chunks_df"], width="stretch", hide_index=True)

    with projections_tab:
        st.subheader("Proyección de obras completas")
        work_projection_df = embedding_result["work_projection_df"].copy()
        projection_mode = st.radio("Proyección", ["UMAP 2D", "UMAP 3D"], horizontal=True)
        if projection_mode == "UMAP 2D":
            projection_fig = px.scatter(
                work_projection_df,
                x="umap_1",
                y="umap_2",
                color="author",
                hover_data=["work", "n_chunks", "token_count"],
                title="UMAP de obras completas en 2D",
                height=720,
            )
        elif projection_mode == "UMAP 3D" and "umap_3" in work_projection_df.columns:
            projection_fig = px.scatter_3d(
                work_projection_df,
                x="umap_1",
                y="umap_2",
                z="umap_3",
                color="author",
                hover_data=["work", "n_chunks", "token_count"],
                title="UMAP de obras completas en 3D",
                height=720,
            )
        else:
            st.info("Esta proyección 3D requiere al menos tres obras y tres dimensiones.")
            projection_fig = None

        if projection_fig is not None:
            st.plotly_chart(projection_fig, width="stretch")
        st.caption(
            "Varianza explicada PCA: "
            + ", ".join(f"PC{i + 1}={value:.3f}" for i, value in enumerate(embedding_result["pca_variance"]))
        )

        st.subheader("UMAP de fragmentos")
        chunk_projection_df = embedding_result["chunks_projection_df"]
        if projection_mode == "UMAP 2D":
            chunk_fig = px.scatter(
                chunk_projection_df,
                x="umap_1",
                y="umap_2",
                color="author",
                hover_data=["work", "chunk_index", "token_count"],
                title="UMAP de fragmentos en 2D",
                height=720,
            )
            chunk_fig.update_traces(marker={"size": 6, "opacity": 0.55})
            st.plotly_chart(chunk_fig, width="stretch")
        elif "umap_3" in chunk_projection_df.columns:
            chunk_fig = px.scatter_3d(
                chunk_projection_df,
                x="umap_1",
                y="umap_2",
                z="umap_3",
                color="author",
                hover_data=["work", "chunk_index", "token_count"],
                title="UMAP de fragmentos en 3D",
                height=720,
            )
            chunk_fig.update_traces(marker={"size": 4, "opacity": 0.45})
            st.plotly_chart(chunk_fig, width="stretch")
        else:
            st.info("La proyección 3D de fragmentos requiere al menos tres fragmentos y tres dimensiones.")

    with similarity_tab:
        st.subheader("Similitud coseno")
        author_sim_fig = px.imshow(
            embedding_result["author_similarity_df"],
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Similitud coseno media entre autores",
        )
        author_sim_fig.update_layout(height=620)
        st.plotly_chart(author_sim_fig, width="stretch")

        work_sim_fig = px.imshow(
            embedding_result["work_similarity_df"],
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Similitud coseno entre obras, ordenada por autor",
        )
        work_sim_fig.update_layout(height=760)
        work_sim_fig.update_xaxes(showticklabels=False)
        work_sim_fig.update_yaxes(showticklabels=False)
        st.plotly_chart(work_sim_fig, width="stretch")

    with neighbors_tab:
        st.subheader("Vecinos más cercanos por obra")
        neighbor_summary_df = embedding_result["neighbor_summary_df"]
        neighbor_fig = px.bar(
            neighbor_summary_df,
            x="neighbor_rank",
            y="pct_same_author",
            labels={"neighbor_rank": "Rango del vecino", "pct_same_author": "% mismo autor"},
            title="Porcentaje de vecinos del mismo autor por rango",
            height=520,
        )
        neighbor_fig.update_yaxes(range=[0, 100])
        st.plotly_chart(neighbor_fig, width="stretch")
        st.dataframe(
            embedding_result["nearest_neighbors_df"].style.format({"cosine_similarity": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )

    with cohesion_tab:
        st.subheader("Cohesión por autor")
        st.metric("Silhouette por autor", f"{embedding_result['silhouette']:.3f}")
        pairs_df = embedding_result["pairs_df"].copy()
        pairs_df["same_author"] = pairs_df["same_author"].map({True: "Mismo autor", False: "Distinto autor"})
        distance_fig = px.box(
            pairs_df,
            x="same_author",
            y="cosine_distance",
            points="all",
            labels={"same_author": "", "cosine_distance": "Distancia coseno"},
            title="Distancias coseno intraautor vs interautor",
            height=560,
        )
        st.plotly_chart(distance_fig, width="stretch")
        st.dataframe(
            embedding_result["author_metrics_df"].style.format(
                {
                    "mean_intra_author_distance": "{:.3f}",
                    "mean_inter_author_distance": "{:.3f}",
                    "separation_margin": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with network_tab:
        st.subheader("Red de similitud entre obras")
        st.plotly_chart(embedding_result["network_fig"], width="stretch")

    st.stop()

if selected_analysis == "unsupervised":
    unsup_display_names = {value: key for key, value in UNSUPERVISED_MODEL_NAMES.items()}

    if selected_unsup_workflow == "experiment":
        if not unsup_model_names:
            st.warning("Selecciona al menos un modelo de clustering.")
            st.stop()

        unsup_config = UnsupervisedConfig(
            feature_set=unsup_feature_set,
            top_n_mfw=int(unsup_top_n_mfw),
            seed=int(unsup_seed),
            selected_models=unsup_model_names,
            n_clusters=int(unsup_n_clusters) if unsup_n_clusters is not None else None,
        )

        data_tab, features_tab, models_tab, clusters_tab, contingency_tab, evaluation_tab, projection_tab = st.tabs(
            ["Datos", "Rasgos", "Modelos", "Clusters", "Contingencia", "Evaluación", "Visualización"]
        )

        experiment_cache_key = unsupervised_experiment_cache_key(corpus_df, lowercase, unsup_config)
        experiment_cache = st.session_state.setdefault("unsupervised_experiment_cache", {})
        if experiment_cache_key in experiment_cache:
            unsup_result = experiment_cache[experiment_cache_key]
        else:
            with st.spinner("Ejecutando experimento de clustering..."):
                unsup_result = run_unsupervised_experiment(corpus_df, unsup_config)
            experiment_cache[experiment_cache_key] = unsup_result

        best_unsup_model_label = unsup_display_names.get(
            unsup_result["best_model_name"],
            unsup_result["best_model_name"],
        )
        best_assignments_df = unsup_result["assignments"][unsup_result["best_model_name"]]

        with data_tab:
            st.subheader("Corpus usado para clustering")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Autores", unsup_result["data"]["author_norm"].nunique())
            with col2:
                st.metric("Obras", unsup_result["data"].shape[0])
            with col3:
                st.metric("Clusters", int(unsup_result["results"].iloc[0]["n_clusters"]))
            with col4:
                st.metric("Mejor modelo", best_unsup_model_label)

            works_by_author = (
                unsup_result["data"].groupby("author_norm", as_index=False).size().rename(columns={"size": "obras"})
            )
            works_fig = px.bar(
                works_by_author,
                x="author_norm",
                y="obras",
                labels={"author_norm": "Autor", "obras": "Obras"},
                title="Obras por autor",
            )
            st.plotly_chart(works_fig, width="stretch")

        with features_tab:
            st.subheader("Rasgos utilizados")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Conjunto", unsup_feature_label)
            with col2:
                st.metric("Número de rasgos", len(unsup_result["feature_cols"]))
            with col3:
                st.metric("TOP_N_MFW", len(unsup_result["mfw_vocab"]) if unsup_result["mfw_vocab"] else "No aplica")

            feature_matrix_df = unsup_result["data"].copy()
            metadata_cols = [col for col in ["author_norm", "work", "file_name"] if col in feature_matrix_df.columns]
            feature_matrix_df = feature_matrix_df[metadata_cols + unsup_result["feature_cols"]]
            st.dataframe(feature_matrix_df, width="stretch", hide_index=True)

        with models_tab:
            st.subheader("Evaluación de algoritmos")
            st.caption(
                "**Tip:** las etiquetas de autor no se usan para crear los clusters; se usan después para calcular "
                "`ARI`, `NMI`, homogeneity, completeness y V-measure."
            )
            results_df = unsup_result["results"].copy()
            results_df["model"] = results_df["model"].map(unsup_display_names)
            st.dataframe(
                results_df.style.format(
                    {
                        "silhouette": "{:.3f}",
                        "ARI": "{:.3f}",
                        "NMI": "{:.3f}",
                        "homogeneity": "{:.3f}",
                        "completeness": "{:.3f}",
                        "v_measure": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            ari_fig = px.bar(
                results_df.sort_values("ARI"),
                x="ARI",
                y="model",
                orientation="h",
                labels={"ARI": "ARI", "model": "Modelo"},
                title="Correspondencia entre clusters y autores",
            )
            st.plotly_chart(ari_fig, width="stretch")

        with clusters_tab:
            st.subheader(f"Asignaciones del mejor modelo: {best_unsup_model_label}")
            assignments_df = best_assignments_df.copy()
            assignments_df["cluster"] = assignments_df["cluster"].astype(str)
            st.dataframe(assignments_df, width="stretch", hide_index=True)

        with contingency_tab:
            st.subheader("Tabla cluster vs autor real")
            st.dataframe(unsup_result["best_cluster_author_table"], width="stretch")

        with evaluation_tab:
            st.subheader("Evaluación tras mapear clusters a autores")
            best_eval = unsup_result["best_eval"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy tras mapping", f"{best_eval['accuracy_after_mapping']:.3f}")
            with col2:
                st.metric("Errores", best_eval["errors_df"].shape[0])

            cm_df = best_eval["confusion_matrix"]
            cm_fig = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Autor predicho", "y": "Autor real", "color": "Obras"},
                title="Matriz de confusión tras mapping cluster-autor",
            )
            cm_fig.update_coloraxes(showscale=False)
            st.plotly_chart(cm_fig, width="stretch")

            st.subheader("Errores")
            st.dataframe(best_eval["errors_df"], width="stretch", hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Errores por autor")
                st.dataframe(
                    unsup_result["errors_by_author"].style.format({"error_rate": "{:.3f}"}),
                    width="stretch",
                    hide_index=True,
                )
            with col2:
                st.subheader("Pares de confusión")
                st.dataframe(unsup_result["confusion_pairs"], width="stretch", hide_index=True)

        with projection_tab:
            st.subheader(f"Proyección t-SNE del mejor modelo: {best_unsup_model_label}")
            projection_df = build_cluster_projection(
                unsup_result["X_scaled"],
                best_assignments_df,
                best_unsup_model_label,
                int(unsup_seed),
            )
            projection_fig = px.scatter(
                projection_df,
                x="x",
                y="y",
                color="cluster",
                symbol="author_norm",
                hover_data=["work", "file_name", "author_norm"],
                labels={"x": "t-SNE 1", "y": "t-SNE 2", "cluster": "Cluster", "author_norm": "Autor"},
                title="Obras proyectadas con t-SNE",
            )
            projection_fig.update_layout(height=700)
            projection_fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(projection_fig, width="stretch")

        st.stop()

    if selected_unsup_workflow == "hierarchical":
        if not hier_method_names:
            st.warning("Selecciona al menos un método de enlace.")
            st.stop()

        hier_display_names = {value: key for key, value in HIERARCHICAL_METHOD_NAMES.items()}
        hier_config = HierarchicalConfig(
            feature_set=unsup_feature_set,
            top_n_mfw=int(unsup_top_n_mfw),
            selected_methods=hier_method_names,
            dendrogram_method=hier_dendrogram_method,
            n_clusters=int(unsup_n_clusters) if unsup_n_clusters is not None else None,
        )

        data_tab, features_tab, methods_tab, dendrogram_tab, clusters_tab, contingency_tab, evaluation_tab = st.tabs(
            ["Datos", "Rasgos", "Métodos", "Dendrograma", "Clusters", "Contingencia", "Evaluación"]
        )

        hier_cache_key = hierarchical_experiment_cache_key(corpus_df, lowercase, hier_config)
        hier_cache = st.session_state.setdefault("hierarchical_experiment_cache", {})
        if hier_cache_key in hier_cache:
            hier_result = hier_cache[hier_cache_key]
        else:
            with st.spinner("Ejecutando clustering jerárquico..."):
                hier_result = run_hierarchical_experiment(corpus_df, hier_config)
            hier_cache[hier_cache_key] = hier_result

        best_hier_method_label = hier_display_names.get(hier_result["best_method"], hier_result["best_method"])
        dendrogram_method_label = hier_display_names.get(
            hier_result["dendrogram_method"],
            hier_result["dendrogram_method"],
        )
        best_assignments_df = hier_result["assignments"][hier_result["best_method"]]
        dendrogram_assignments_df = hier_result["assignments"][hier_result["dendrogram_method"]]

        with data_tab:
            st.subheader("Corpus usado para clustering jerárquico")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Autores", hier_result["data"]["author_norm"].nunique())
            with col2:
                st.metric("Obras", hier_result["data"].shape[0])
            with col3:
                st.metric("Clusters", hier_result["n_clusters"])
            with col4:
                st.metric("Mejor método", best_hier_method_label)

            works_by_author = (
                hier_result["data"].groupby("author_norm", as_index=False).size().rename(columns={"size": "obras"})
            )
            works_fig = px.bar(
                works_by_author,
                x="author_norm",
                y="obras",
                labels={"author_norm": "Autor", "obras": "Obras"},
                title="Obras por autor",
            )
            st.plotly_chart(works_fig, width="stretch")

        with features_tab:
            st.subheader("Rasgos utilizados")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Conjunto", unsup_feature_label)
            with col2:
                st.metric("Número de rasgos", len(hier_result["feature_cols"]))
            with col3:
                st.metric("TOP_N_MFW", len(hier_result["mfw_vocab"]) if hier_result["mfw_vocab"] else "No aplica")

            feature_matrix_df = hier_result["data"].copy()
            metadata_cols = [col for col in ["author_norm", "work", "file_name"] if col in feature_matrix_df.columns]
            feature_matrix_df = feature_matrix_df[metadata_cols + hier_result["feature_cols"]]
            st.dataframe(feature_matrix_df, width="stretch", hide_index=True)

        with methods_tab:
            st.subheader("Comparativa de métodos de enlace")
            st.caption(
                "**Tip:** el clustering jerárquico fusiona obras por distancia entre rasgos. Las etiquetas de autor "
                "solo se usan después para evaluar la correspondencia de los grupos."
            )
            methods_df = hier_result["results"].copy()
            methods_df["method"] = methods_df["method"].map(hier_display_names)
            st.dataframe(
                methods_df.style.format(
                    {
                        "silhouette": "{:.3f}",
                        "ARI": "{:.3f}",
                        "NMI": "{:.3f}",
                        "homogeneity": "{:.3f}",
                        "completeness": "{:.3f}",
                        "v_measure": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            methods_fig = px.bar(
                methods_df.sort_values("ARI"),
                x="ARI",
                y="method",
                orientation="h",
                labels={"ARI": "ARI", "method": "Método"},
                title="Correspondencia entre clusters jerárquicos y autores",
            )
            st.plotly_chart(methods_fig, width="stretch")

        with dendrogram_tab:
            dendrogram_fig = build_dendrogram_figure(
                hier_result["linkages"][hier_result["dendrogram_method"]],
                dendrogram_assignments_df,
                dendrogram_method_label,
            )
            st.plotly_chart(dendrogram_fig, width="stretch")

        with clusters_tab:
            st.subheader(f"Asignaciones del mejor método: {best_hier_method_label}")
            assignments_df = best_assignments_df.copy()
            assignments_df["cluster"] = assignments_df["cluster"].astype(str)
            st.dataframe(assignments_df, width="stretch", hide_index=True)

        with contingency_tab:
            st.subheader("Tabla cluster vs autor real")
            st.dataframe(hier_result["best_cluster_author_table"], width="stretch")

        with evaluation_tab:
            st.subheader("Evaluación tras mapear clusters a autores")
            best_eval = hier_result["best_eval"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy tras mapping", f"{best_eval['accuracy_after_mapping']:.3f}")
            with col2:
                st.metric("Errores", best_eval["errors_df"].shape[0])

            cm_df = best_eval["confusion_matrix"]
            cm_fig = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Autor predicho", "y": "Autor real", "color": "Obras"},
                title="Matriz de confusión tras mapping cluster-autor",
            )
            cm_fig.update_coloraxes(showscale=False)
            st.plotly_chart(cm_fig, width="stretch")

            st.subheader("Errores")
            st.dataframe(best_eval["errors_df"], width="stretch", hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Errores por autor")
                st.dataframe(
                    hier_result["errors_by_author"].style.format({"error_rate": "{:.3f}"}),
                    width="stretch",
                    hide_index=True,
                )
            with col2:
                st.subheader("Pares de confusión")
                st.dataframe(hier_result["confusion_pairs"], width="stretch", hide_index=True)

        st.stop()

    if selected_unsup_workflow == "robustness":
        if not unsup_top_n_values or not unsup_seeds or not unsup_model_names:
            st.warning("Configura al menos un TOP_N_MFW, una semilla y un modelo.")
            st.stop()

        sweep_tab, summary_tab, decision_tab, detail_tab = st.tabs(
            ["Barrido", "Resumen por TOP_N", "Decisión", "Detalle de ejecuciones"]
        )

        robustness_key = (
            corpus_cache_fingerprint(corpus_df),
            lowercase,
            tuple(unsup_top_n_values),
            tuple(unsup_seeds),
            tuple(unsup_model_names),
        )
        robustness_cache = st.session_state.setdefault("unsupervised_robustness_cache", {})
        if robustness_key in robustness_cache:
            unsup_robustness = robustness_cache[robustness_key]
        else:
            with st.spinner("Ejecutando barrido de robustez MFW no supervisado..."):
                unsup_robustness = run_unsupervised_mfw_robustness(
                    corpus_df,
                    top_n_values=unsup_top_n_values,
                    seeds=unsup_seeds,
                    selected_models=unsup_model_names,
                )
            robustness_cache[robustness_key] = unsup_robustness

        with sweep_tab:
            st.subheader("Configuración del barrido")
            config_df = pd.DataFrame(
                [
                    {"parámetro": "TOP_N_MFW", "valor": ", ".join(map(str, unsup_top_n_values))},
                    {"parámetro": "Semillas", "valor": ", ".join(map(str, unsup_seeds))},
                    {"parámetro": "Modelos", "valor": ", ".join(unsup_model_labels)},
                ]
            )
            st.dataframe(config_df, width="stretch", hide_index=True)

        with summary_tab:
            st.subheader("Rendimiento agregado")
            summary_df = unsup_robustness["summary_df"].copy()
            summary_df["model"] = summary_df["model"].map(unsup_display_names)
            st.dataframe(
                summary_df.style.format(
                    {
                        "mean_ARI": "{:.3f}",
                        "std_ARI": "{:.3f}",
                        "min_ARI": "{:.3f}",
                        "mean_NMI": "{:.3f}",
                        "std_NMI": "{:.3f}",
                        "mean_silhouette": "{:.3f}",
                        "std_silhouette": "{:.3f}",
                        "mean_homogeneity": "{:.3f}",
                        "mean_completeness": "{:.3f}",
                        "mean_v_measure": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            robustness_fig = px.line(
                summary_df,
                x="top_n_mfw",
                y="mean_ARI",
                color="model",
                markers=True,
                error_y="std_ARI",
                labels={"top_n_mfw": "TOP_N_MFW", "mean_ARI": "ARI medio", "model": "Modelo"},
                title="Robustez del clustering según número de MFW",
            )
            st.plotly_chart(robustness_fig, width="stretch")

        with decision_tab:
            st.subheader("Mejor combinación")

            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.metric("TOP_N_MFW", unsup_robustness["selected_top_n_mfw"])

            with row1_col2:
                st.metric(
                    "Modelo",
                    unsup_display_names.get(
                        unsup_robustness["selected_model_name"],
                        unsup_robustness["selected_model_name"],
                    ),
                )

            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row2_col1:
                st.metric("ARI medio", f"{unsup_robustness['best_mean_ARI']:.3f}")

            with row2_col2:
                st.metric("NMI medio", f"{unsup_robustness['best_mean_NMI']:.3f}")

            with row2_col3:
                st.metric("Silhouette medio", f"{unsup_robustness['best_mean_silhouette']:.3f}")

            decision_df = unsup_robustness["decision_df"].copy()
            decision_df["model"] = decision_df["model"].map(unsup_display_names)
            st.dataframe(decision_df, width="stretch", hide_index=True)

        with detail_tab:
            st.subheader("Todas las ejecuciones")
            detail_df = unsup_robustness["sweep_df"].copy()
            detail_df["model"] = detail_df["model"].map(unsup_display_names)
            st.dataframe(detail_df, width="stretch", hide_index=True)

        st.stop()

if selected_analysis == "classical":
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

if selected_analysis == "supervised":
    model_display_names = {value: key for key, value in MODEL_NAMES.items()}
    ml_config = SupervisedConfig(
        feature_set=ml_feature_set,
        top_n_mfw=int(ml_top_n_mfw),
        seed=int(ml_seed),
        n_test_per_author=int(ml_n_test_per_author),
        keep_correlated_features=ml_keep_correlated,
        correlation_threshold=float(ml_corr_threshold),
        selected_models=ml_model_names,
    )

    if selected_ml_workflow == "experiment":
        if not ml_model_names:
            st.warning("Selecciona al menos un modelo para ejecutar el experimento.")
            st.stop()

        data_tab, features_tab, models_tab, evaluation_tab, predictions_tab, attribution_tab = st.tabs(
            ["Datos", "Rasgos", "Modelos", "Evaluación", "Predicciones", "Atribuir obra"]
        )

        experiment_cache_key = supervised_experiment_cache_key(corpus_df, lowercase, ml_config)
        experiment_cache = st.session_state.setdefault("supervised_experiment_cache", {})
        if experiment_cache_key in experiment_cache:
            ml_result = experiment_cache[experiment_cache_key]
        else:
            with st.spinner("Entrenando y evaluando modelos supervisados..."):
                ml_result = run_supervised_experiment(corpus_df, ml_config)
            experiment_cache[experiment_cache_key] = ml_result

        with data_tab:
            st.subheader("Partición de entrenamiento y test")
            split_summary = pd.DataFrame(
                [
                    {"split": "Entrenamiento", "obras": ml_result["train_df"].shape[0]},
                    {"split": "Test", "obras": ml_result["test_df"].shape[0]},
                ]
            )
            st.dataframe(split_summary, width="stretch", hide_index=True)

            by_author = (
                pd.concat(
                    [
                        ml_result["train_df"].assign(split="Entrenamiento"),
                        ml_result["test_df"].assign(split="Test"),
                    ]
                )
                .groupby(["split", "author"], as_index=False)
                .size()
                .rename(columns={"size": "obras"})
            )
            split_fig = px.bar(
                by_author,
                x="author",
                y="obras",
                color="split",
                barmode="group",
                labels={"author": "Autor", "obras": "Obras", "split": "Partición"},
                title="Obras por autor en cada partición",
            )
            st.plotly_chart(split_fig, width="stretch")

        with features_tab:
            st.subheader("Rasgos utilizados")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Conjunto", ml_feature_label)
            with col2:
                st.metric("Número de rasgos", len(ml_result["feature_cols"]))
            with col3:
                st.metric("TOP_N_MFW", len(ml_result["mfw_vocab"]) if ml_result["mfw_vocab"] else "No aplica")

            feature_matrix_df = pd.concat(
                [
                    ml_result["train_df"].assign(split="Entrenamiento"),
                    ml_result["test_df"].assign(split="Test"),
                ],
                ignore_index=True,
            )
            metadata_cols = [col for col in ["split", "author", "work", "filename"] if col in feature_matrix_df.columns]
            feature_matrix_df = feature_matrix_df[metadata_cols + ml_result["feature_cols"]]
            st.dataframe(feature_matrix_df, width="stretch", hide_index=True)

            if ml_feature_set in {"stylometric", "combined"} and ml_result["removed_features"]:
                st.subheader("Rasgos eliminados por correlación")
                st.caption(
                    "**Tip:** al eliminar rasgos muy correlacionados se reduce redundancia entre variables y se evita que "
                    "algunos modelos den peso repetido a señales casi equivalentes."
                )
                st.dataframe(
                    pd.DataFrame({"removed_feature": ml_result["removed_features"]}),
                    width="stretch",
                    hide_index=True,
                )

        with models_tab:
            st.subheader("Validación cruzada")
            st.caption(
                "**Tip:** `cv_mean` resume el rendimiento medio en validación cruzada sobre entrenamiento; `cv_std` indica "
                "cuánto varía ese rendimiento entre particiones."
            )
            cv_df = ml_result["cv_results"].copy()
            cv_df["model"] = cv_df["model"].map(model_display_names)
            st.dataframe(
                cv_df.style.format({"cv_mean": "{:.3f}", "cv_std": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )
            cv_fig = px.bar(
                cv_df.sort_values("cv_mean"),
                x="cv_mean",
                y="model",
                error_x="cv_std",
                orientation="h",
                labels={"cv_mean": "F1-macro medio", "model": "Modelo"},
                title="Rendimiento medio en validación cruzada",
            )
            st.plotly_chart(cv_fig, width="stretch")

        with evaluation_tab:
            st.subheader("Evaluación del mejor modelo")
            best_model_label = model_display_names.get(ml_result["best_model_name"], ml_result["best_model_name"])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mejor modelo", best_model_label)
            with col2:
                st.metric("F1-macro test", f"{ml_result['test_f1_macro']:.3f}")
            with col3:
                st.metric("Accuracy test", f"{ml_result['test_accuracy']:.3f}")

            report_df = pd.DataFrame(ml_result["classification_report"]).T.reset_index(names="label")
            st.dataframe(report_df, width="stretch", hide_index=True)

            cm_df = pd.DataFrame(
                ml_result["confusion_matrix"],
                index=ml_result["labels"],
                columns=ml_result["labels"],
            )
            cm_fig = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Autor predicho", "y": "Autor real", "color": "Obras"},
                title="Matriz de confusión en test",
            )
            cm_fig.update_coloraxes(showscale=False)
            st.plotly_chart(cm_fig, width="stretch")

        with predictions_tab:
            st.subheader("Predicciones por obra")
            pred_df = ml_result["pred_df"].copy()
            st.dataframe(pred_df, width="stretch", hide_index=True)

        with attribution_tab:
            st.subheader("Atribuir una nueva obra")
            uploaded_file = st.file_uploader(
                "Selecciona una obra en .txt",
                type=["txt"],
                key="experiment_attribution_uploaded_txt",
            )

            if "experiment_attribution_text" not in st.session_state:
                st.session_state.experiment_attribution_text = None
            if "experiment_attribution_filename" not in st.session_state:
                st.session_state.experiment_attribution_filename = None

            if uploaded_file is not None:
                st.session_state.experiment_attribution_text = uploaded_file.getvalue().decode(
                    "utf-8",
                    errors="replace",
                )
                st.session_state.experiment_attribution_filename = uploaded_file.name

            if st.session_state.experiment_attribution_text is None:
                st.info("Sube un archivo .txt para atribuirlo con el mejor modelo de este experimento.")
            else:
                experiment_model_bundle = {
                    "pipeline": ml_result["best_pipeline"],
                    "feature_cols": ml_result["feature_cols"],
                    "mfw_vocab": ml_result["mfw_vocab"],
                }
                attribution_ranking_df, attribution_explanation_df = classify_text_with_supervised_model(
                    st.session_state.experiment_attribution_text,
                    experiment_model_bundle,
                    ml_config,
                    lowercase=lowercase,
                )
                score_col = "probability" if "probability" in attribution_ranking_df.columns else "score"
                predicted_row = attribution_ranking_df.iloc[0]
                uploaded_filename = st.session_state.experiment_attribution_filename or "obra externa"

                st.success(f"Archivo cargado: {uploaded_filename}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_model_label = model_display_names.get(
                        ml_result["best_model_name"],
                        ml_result["best_model_name"],
                    )
                    st.metric("Modelo", best_model_label)
                with col2:
                    st.metric("Autor predicho", predicted_row["author"])
                with col3:
                    st.metric("Score", f"{predicted_row[score_col]:.3f}")

                ranking_fig = px.bar(
                    attribution_ranking_df.sort_values(score_col),
                    x=score_col,
                    y="author",
                    orientation="h",
                    labels={score_col: "Probabilidad" if score_col == "probability" else "Score", "author": "Autor"},
                    title="Ranking de atribución",
                )
                st.plotly_chart(ranking_fig, width="stretch")
                st.dataframe(
                    attribution_ranking_df.style.format({score_col: "{:.3f}"}),
                    width="stretch",
                    hide_index=True,
                )

                if attribution_explanation_df.empty:
                    st.info("La explicabilidad local está disponible para modelos lineales con coeficientes.")
                else:
                    st.subheader("Rasgos que más empujan la predicción")
                    top_explanation = attribution_explanation_df.head(25)
                    explanation_fig = px.bar(
                        top_explanation.sort_values("contribution"),
                        x="contribution",
                        y="feature",
                        orientation="h",
                        labels={"contribution": "Contribución", "feature": "Rasgo"},
                        title="Contribuciones locales principales",
                    )
                    explanation_fig.update_layout(height=max(400, 25 * len(top_explanation)))
                    st.plotly_chart(explanation_fig, width="stretch")
                    st.dataframe(
                        top_explanation[["feature", "value", "contribution"]].style.format(
                            {"value": "{:.6f}", "contribution": "{:.3f}"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )

        st.stop()

    if selected_ml_workflow == "robustness":
        if not ml_top_n_values or not ml_seeds or not ml_model_names:
            st.warning("Configura al menos un TOP_N_MFW, una semilla y un modelo.")
            st.stop()

        sweep_tab, summary_tab, decision_tab, detail_tab = st.tabs(
            ["Barrido", "Resumen por TOP_N", "Decisión", "Detalle de ejecuciones"]
        )

        with st.spinner("Ejecutando barrido de robustez MFW..."):
            robustness = run_mfw_robustness(
                corpus_df,
                top_n_values=ml_top_n_values,
                seeds=ml_seeds,
                n_test_per_author=int(ml_n_test_per_author),
                tolerance=float(ml_tolerance),
                selected_models=ml_model_names,
            )

        with sweep_tab:
            st.subheader("Configuración del barrido")
            config_df = pd.DataFrame(
                [
                    {"parámetro": "TOP_N_MFW", "valor": ", ".join(map(str, ml_top_n_values))},
                    {"parámetro": "Semillas", "valor": ", ".join(map(str, ml_seeds))},
                    {"parámetro": "Modelos", "valor": ", ".join(ml_model_labels)},
                    {"parámetro": "Tolerancia", "valor": f"{ml_tolerance:.3f}"},
                ]
            )
            st.dataframe(config_df, width="stretch", hide_index=True)

        with summary_tab:
            st.subheader("Rendimiento agregado")
            summary_df = robustness["summary_df"].copy()
            summary_df["model_name"] = summary_df["model_name"].map(model_display_names)
            st.dataframe(
                summary_df.style.format(
                    {
                        "mean_test_f1_macro": "{:.3f}",
                        "std_test_f1_macro": "{:.3f}",
                        "min_test_f1_macro": "{:.3f}",
                        "max_test_f1_macro": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            robustness_fig = px.line(
                summary_df,
                x="top_n_mfw",
                y="mean_test_f1_macro",
                color="model_name",
                markers=True,
                error_y="std_test_f1_macro",
                labels={
                    "top_n_mfw": "TOP_N_MFW",
                    "mean_test_f1_macro": "F1-macro medio",
                    "model_name": "Modelo",
                },
                title="Robustez del rendimiento según número de MFW",
            )
            robustness_fig.add_hline(
                y=robustness["threshold_score"],
                line_dash="dash",
                annotation_text="Umbral",
            )
            robustness_fig.add_vline(
                x=robustness["selected_top_n_mfw"],
                line_dash="dot",
                annotation_text="TOP_N seleccionado",
            )
            st.plotly_chart(robustness_fig, width="stretch")

        with decision_tab:
            st.subheader("Selección del menor TOP_N_MFW suficiente")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mejor media observada", f"{robustness['best_mean_score']:.3f}")
            with col2:
                st.metric("Umbral", f"{robustness['threshold_score']:.3f}")
            with col3:
                st.metric("TOP_N_MFW seleccionado", robustness["selected_top_n_mfw"])
            with col4:
                st.metric(
                    "Modelo seleccionado",
                    model_display_names.get(robustness["selected_model_name"], robustness["selected_model_name"]),
                )

            decision_df = robustness["final_decision_df"].copy()
            decision_df["model_name"] = decision_df["model_name"].map(model_display_names)
            st.dataframe(decision_df, width="stretch", hide_index=True)

        with detail_tab:
            st.subheader("Todas las ejecuciones")
            detail_df = robustness["sweep_df"].copy()
            detail_df["model_name"] = detail_df["model_name"].map(model_display_names)
            st.dataframe(detail_df, width="stretch", hide_index=True)

        st.stop()

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
        st.plotly_chart(block_fig, width="stretch")

        st.subheader("Estabilidad interna por autor")
        st.caption(
            "**Tip:** `mean_distance` resume cuánto varían entre sí los bloques de un mismo autor. "
            "Valores más bajos indican una distribución de longitudes de palabra más estable dentro del corpus. "
            "`std_distance` indica si esa variación es regular o si algunos bloques se alejan mucho más que otros."
        )
        st.dataframe(
            mendenhall_stats_df.style.format({"mean_distance": "{:.5f}", "std_distance": "{:.5f}"}),
            width="stretch",
            hide_index=True,
        )

    with average_tab:
        st.subheader("Curvas características medias")
        st.caption(
            "**Tip:** cada línea resume la frecuencia media de palabras de distintas longitudes en el corpus de un autor. "
            "Curvas más parecidas sugieren perfiles de composición similares, aunque este rasgo por sí solo no siempre "
            "discrimina autores con claridad."
        )
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
        st.plotly_chart(average_fig, width="stretch")

        st.subheader("Distancia Jensen-Shannon entre curvas medias")
        st.caption(
            "**Tip:** los valores más bajos indican autores con curvas medias más parecidas. "
            "La diagonal vale cero porque compara cada autor consigo mismo."
        )
        comparison_df = compare_mendenhall_average_curves(average_curves, max_word_len=max_word_len)
        st.dataframe(comparison_df.style.format("{:.5f}"), width="stretch")

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
                st.plotly_chart(attribution_fig, width="stretch")
                st.subheader("Ranking de autores")
                st.caption(
                    "**Tip:** la obra se atribuye al autor cuya curva media tiene menor distancia con la curva de la obra "
                    "seleccionada. Conviene mirar también la separación respecto al segundo autor ya que si las distancias "
                    "son muy parecidas, la atribución es menos clara."
                )
                st.dataframe(
                    distances_df[["rank", "author", "distance"]].style.format({"distance": "{:.5f}"}),
                    width="stretch",
                    hide_index=True,
                )

    st.stop()

if selected_method == "kilgariff":
    corpus_tab, vocab_tab, attribution_tab, contributions_tab = st.tabs(
        ["Corpus", "Vocabulario global", "Atribución de obra", "Contribuciones"]
    )

    with st.spinner("Preparando perfiles léxicos de Kilgariff..."):
        vocab_df, vocab_frequencies_df = kilgariff_reference_tables(corpus_df, config)

    with corpus_tab:
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

        lexical_uses_function_words = use_function_words
        top_words = top_tokens_by_author(corpus_df, top_n=top_n, use_function_words=lexical_uses_function_words)
        st.subheader("Palabras más frecuentes por autor")
        st.caption(
            "**Tip:** las palabras funcionales suelen ser útiles en estilometría porque dependen menos del tema de la obra "
            "y más de hábitos de escritura relativamente estables."
        )

        selected_author = st.selectbox("Autor", sorted(top_words["author"].unique()), key="kilgariff_top_author")
        author_top_words = top_words[top_words["author"] == selected_author]
        top_fig = px.bar(
            author_top_words.sort_values("relative_frequency"),
            x="relative_frequency",
            y="token",
            orientation="h",
            labels={"relative_frequency": "Frecuencia relativa", "token": "Token"},
            title=selected_author,
        )
        top_fig.update_layout(height=max(400, 25 * len(author_top_words)))
        st.plotly_chart(top_fig, width="stretch")
        st.dataframe(top_words, width="stretch", hide_index=True)

    with vocab_tab:
        st.subheader("Vocabulario común de comparación")
        st.caption(
            "**Tip:** todas las comparaciones usan este mismo vocabulario. Esto hace que las distancias entre autores sean "
            "comparables, porque siempre se calculan sobre los mismos rasgos léxicos."
        )
        if vocab_df.empty:
            st.warning("El vocabulario global ha quedado vacío con la configuración actual.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Términos seleccionados", vocab_df.shape[0])
            with col2:
                st.metric("Frecuencia mínima", min_freq)
            with col3:
                st.metric("Solo palabras funcionales", "Sí" if use_function_words else "No")

            st.dataframe(vocab_df, width="stretch", hide_index=True)

            heatmap_limit = min(50, vocab_df.shape[0])
            heatmap_tokens = vocab_df.head(heatmap_limit)["token"]

            heatmap_df = vocab_frequencies_df[vocab_frequencies_df["token"].isin(heatmap_tokens)]

            heatmap_matrix = heatmap_df.pivot(index="author", columns="token", values="relative_frequency").fillna(0)

            heatmap_fig = px.imshow(
                heatmap_matrix,
                aspect="auto",
                color_continuous_scale="Purples",
                labels={
                    "x": "Token",
                    "y": "Autor",
                    "color": "Frecuencia relativa",
                },
                title=f"Frecuencias relativas de los {heatmap_limit} términos más frecuentes",
            )

            heatmap_fig.update_xaxes(
                tickangle=45,
                tickmode="array",
                tickvals=list(heatmap_matrix.columns),
                ticktext=list(heatmap_matrix.columns),
            )

            heatmap_fig.update_layout(
                height=max(400, 35 * len(heatmap_matrix.index)),
                width=max(800, 18 * len(heatmap_matrix.columns)),
            )

            st.plotly_chart(heatmap_fig, width="stretch")

            st.caption(
                "**Tip:** el color muestra la frecuencia relativa de cada término en cada autor. Diferencias marcadas "
                "pueden indicar palabras especialmente características o poco habituales en un perfil."
            )

    with attribution_tab:
        st.subheader("Atribuir una obra externa")

        uploaded_file = st.file_uploader(
            "Selecciona una obra en .txt",
            type=["txt"],
            key="kilgariff_attribution_uploaded_txt",
        )

        if "kilgariff_attribution_text" not in st.session_state:
            st.session_state.kilgariff_attribution_text = None
        if "kilgariff_attribution_filename" not in st.session_state:
            st.session_state.kilgariff_attribution_filename = None

        if uploaded_file is not None:
            st.session_state.kilgariff_attribution_text = uploaded_file.getvalue().decode(
                "utf-8",
                errors="replace",
            )
            st.session_state.kilgariff_attribution_filename = uploaded_file.name

        if st.session_state.kilgariff_attribution_text is None:
            st.info("Sube un archivo .txt para compararlo con los perfiles léxicos del corpus.")
            kilgariff_distances_df = pd.DataFrame()
            kilgariff_contributions_df = pd.DataFrame()
        else:
            text = st.session_state.kilgariff_attribution_text
            uploaded_filename = st.session_state.kilgariff_attribution_filename or "obra externa"

            st.success(f"Archivo cargado: {uploaded_filename}")

            try:
                kilgariff_distances_df, kilgariff_contributions_df, kilgariff_stats = classify_text_with_kilgariff(
                    text,
                    corpus_df,
                    config,
                )
            except ValueError as exc:
                st.error(str(exc))
                kilgariff_distances_df = pd.DataFrame()
                kilgariff_contributions_df = pd.DataFrame()
            else:
                predicted_author = kilgariff_distances_df.iloc[0]["author"]
                predicted_distance = kilgariff_distances_df.iloc[0]["chi2"]
                second_margin = kilgariff_distances_df.iloc[0]["margin_to_second"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Autor más similar", predicted_author)
                with col2:
                    st.metric("Chi-cuadrado", f"{predicted_distance:.3f}")
                with col3:
                    st.metric("Margen al segundo", f"{second_margin:.3f}")

                ranking_fig = px.bar(
                    kilgariff_distances_df.sort_values("chi2", ascending=False),
                    x="chi2",
                    y="author",
                    orientation="h",
                    labels={"chi2": "Chi-cuadrado", "author": "Autor"},
                    title=f"Ranking de similitud para {uploaded_filename}",
                )
                st.plotly_chart(ranking_fig, width="stretch")

                st.subheader("Ranking de autores")
                st.caption(
                    "**Tip:** en Kilgariff, valores más bajos de chi-cuadrado indican mayor similitud entre la obra "
                    "seleccionada y el corpus del autor. El margen respecto al segundo candidato ayuda a valorar la "
                    "claridad de la atribución."
                )
                ranking_cols = ["rank", "author", "chi2", "margin_to_best"]
                st.dataframe(
                    kilgariff_distances_df[ranking_cols].style.format(
                        {
                            "chi2": "{:.3f}",
                            "margin_to_best": "{:.3f}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with contributions_tab:
        if "kilgariff_contributions_df" not in locals() or kilgariff_contributions_df.empty:
            st.info("Sube una obra en la pestaña de atribución para ver las contribuciones por palabra.")
        else:
            st.subheader("Palabras que más explican la distancia")
            st.caption(
                "**Tip:** las palabras con mayor contribución son las que más aumentan la distancia entre la obra y el "
                "autor candidato. `obs_ref` y `obs_test` son frecuencias observadas. `exp_ref` y `exp_test` son las esperadas "
                "si ambos textos siguieran una distribución similar."
            )
            candidate_authors = kilgariff_distances_df["author"].tolist()
            selected_candidate = st.selectbox(
                "Autor candidato",
                candidate_authors,
                key="kilgariff_contribution_author",
            )
            contribution_limit = st.slider(
                "Número de palabras",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                key="kilgariff_contribution_limit",
            )

            filtered = kilgariff_contributions_df[
                kilgariff_contributions_df["candidate_author"] == selected_candidate
            ].head(contribution_limit)
            contribution_cols = ["rank", "token", "chi2", "obs_ref", "exp_ref", "obs_test", "exp_test"]
            selected_work_title = st.session_state.kilgariff_attribution_filename or "obra externa"

            st.write(f"**Obra seleccionada:** {selected_work_title}")
            st.write(f"**Autor candidato:** {selected_candidate}")

            contribution_fig = px.bar(
                filtered.sort_values("chi2"),
                x="chi2",
                y="token",
                orientation="h",
                labels={"chi2": "Contribución al chi-cuadrado", "token": "Token"},
                title=f"Contribuciones principales frente a {selected_candidate}",
            )

            contribution_fig.update_layout(height=max(400, 25 * len(filtered)))

            st.plotly_chart(contribution_fig, width="stretch")
            st.dataframe(
                filtered[contribution_cols].style.format(
                    {
                        "chi2": "{:.3f}",
                        "exp_ref": "{:.2f}",
                        "exp_test": "{:.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    st.stop()

if selected_method == "burrows":
    corpus_tab, vocab_tab, attribution_tab, contributions_tab = st.tabs(
        ["Corpus", "Vocabulario global", "Atribución de obra", "Contribuciones"]
    )

    try:
        with st.spinner("Preparando perfiles normalizados de Burrows..."):
            vocab_df, zscores_df = burrows_reference_tables(corpus_df, config)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with corpus_tab:
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

        top_words = top_tokens_by_author(corpus_df, top_n=top_n, use_function_words=use_function_words)
        st.subheader("Palabras más frecuentes por autor")
        st.caption(
            "**Tip:** Burrows suele trabajar con palabras frecuentes, especialmente funcionales, porque ayudan a capturar "
            "patrones de estilo menos dependientes del contenido temático."
        )

        selected_author = st.selectbox("Autor", sorted(top_words["author"].unique()), key="burrows_top_author")
        author_top_words = top_words[top_words["author"] == selected_author]
        top_fig = px.bar(
            author_top_words.sort_values("relative_frequency"),
            x="relative_frequency",
            y="token",
            orientation="h",
            labels={"relative_frequency": "Frecuencia relativa", "token": "Token"},
            title=selected_author,
        )
        top_fig.update_layout(height=max(400, 25 * len(author_top_words)))
        st.plotly_chart(top_fig, width="stretch")
        st.dataframe(top_words, width="stretch", hide_index=True)

    with vocab_tab:
        st.subheader("Vocabulario común y puntuaciones z")
        st.caption(
            "**Tip:** Burrows normaliza las frecuencias mediante puntuaciones z. Un z-score positivo indica que un autor "
            "usa ese término por encima de la media del corpus y, uno negativo, por debajo."
        )
        if vocab_df.empty:
            st.warning("El vocabulario global ha quedado vacío con la configuración actual.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Términos seleccionados", vocab_df.shape[0])
            with col2:
                st.metric("Frecuencia mínima", min_freq)
            with col3:
                st.metric("Solo palabras funcionales", "Sí" if use_function_words else "No")

            st.dataframe(
                vocab_df.style.format(
                    {
                        "mean_frequency": "{:.6f}",
                        "std_frequency": "{:.6f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            heatmap_limit = min(50, vocab_df.shape[0])
            heatmap_tokens = vocab_df.head(heatmap_limit)["token"]
            heatmap_df = zscores_df[zscores_df["token"].isin(heatmap_tokens)]
            heatmap_matrix = heatmap_df.pivot(index="author", columns="token", values="z_score")

            heatmap_fig = px.imshow(
                heatmap_matrix,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                labels={
                    "x": "Token",
                    "y": "Autor",
                    "color": "Z-score",
                },
                title=f"Puntuaciones z de los {heatmap_limit} términos más frecuentes",
            )

            heatmap_fig.update_xaxes(
                tickangle=45,
                tickmode="array",
                tickvals=list(heatmap_matrix.columns),
                ticktext=list(heatmap_matrix.columns),
            )

            heatmap_fig.update_layout(
                height=max(400, 35 * len(heatmap_matrix.index)),
                width=max(800, 18 * len(heatmap_matrix.columns)),
            )

            st.plotly_chart(heatmap_fig, width="stretch")

            st.caption(
                "**Tip:** los colores muestran desviaciones respecto al comportamiento medio del corpus. Los tonos "
                "extremos señalan términos especialmente sobreutilizados o infrautilizados por cada autor."
            )

    with attribution_tab:
        st.subheader("Atribuir una obra externa")

        uploaded_file = st.file_uploader(
            "Selecciona una obra en .txt",
            type=["txt"],
            key="burrows_attribution_uploaded_txt",
        )

        if "burrows_attribution_text" not in st.session_state:
            st.session_state.burrows_attribution_text = None
        if "burrows_attribution_filename" not in st.session_state:
            st.session_state.burrows_attribution_filename = None

        if uploaded_file is not None:
            st.session_state.burrows_attribution_text = uploaded_file.getvalue().decode(
                "utf-8",
                errors="replace",
            )
            st.session_state.burrows_attribution_filename = uploaded_file.name

        if st.session_state.burrows_attribution_text is None:
            st.info("Sube un archivo .txt para compararlo con los perfiles normalizados del corpus.")
            burrows_distances_df = pd.DataFrame()
            burrows_contributions_df = pd.DataFrame()
        else:
            text = st.session_state.burrows_attribution_text
            uploaded_filename = st.session_state.burrows_attribution_filename or "obra externa"

            st.success(f"Archivo cargado: {uploaded_filename}")

            try:
                burrows_distances_df, burrows_contributions_df, burrows_stats = classify_text_with_burrows(
                    text,
                    corpus_df,
                    config,
                )
            except ValueError as exc:
                st.error(str(exc))
                burrows_distances_df = pd.DataFrame()
                burrows_contributions_df = pd.DataFrame()
            else:
                predicted_author = burrows_distances_df.iloc[0]["author"]
                predicted_distance = burrows_distances_df.iloc[0]["delta"]
                second_margin = burrows_distances_df.iloc[0]["margin_to_second"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Autor más similar", predicted_author)
                with col2:
                    st.metric("Delta", f"{predicted_distance:.3f}")
                with col3:
                    st.metric("Margen al segundo", f"{second_margin:.3f}")

                ranking_fig = px.bar(
                    burrows_distances_df.sort_values("delta", ascending=False),
                    x="delta",
                    y="author",
                    orientation="h",
                    labels={"delta": "Delta", "author": "Autor"},
                    title=f"Ranking de similitud para {uploaded_filename}",
                )
                st.plotly_chart(ranking_fig, width="stretch")

                st.subheader("Ranking de autores")
                st.caption(
                    "**Tip:** en Delta de Burrows, lo más importante es el orden relativo del ranking de modo que el autor con menor "
                    "Delta es el más similar. Las distancias absolutas son menos interpretables que la separación entre "
                    "candidatos."
                )
                ranking_cols = ["rank", "author", "delta", "margin_to_best"]
                st.dataframe(
                    burrows_distances_df[ranking_cols].style.format(
                        {
                            "delta": "{:.3f}",
                            "margin_to_best": "{:.3f}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with contributions_tab:
        if "burrows_contributions_df" not in locals() or burrows_contributions_df.empty:
            st.info("Sube una obra en la pestaña de atribución para ver las contribuciones por palabra.")
        else:
            st.subheader("Palabras que más explican la distancia")
            st.caption(
                "**Tip:** delta muestra la diferencia absoluta entre los z-scores del autor candidato y la obra "
                "seleccionada para cada palabra. Valores altos indican los términos que más separan la obra de ese perfil "
                "de autor."
            )
            candidate_authors = burrows_distances_df["author"].tolist()
            selected_candidate = st.selectbox(
                "Autor candidato",
                candidate_authors,
                key="burrows_contribution_author",
            )
            contribution_limit = st.slider(
                "Número de palabras",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                key="burrows_contribution_limit",
            )

            filtered = burrows_contributions_df[
                burrows_contributions_df["candidate_author"] == selected_candidate
            ].head(contribution_limit)
            contribution_cols = ["rank", "token", "delta", "z_ref", "z_test"]
            selected_work_title = st.session_state.burrows_attribution_filename or "obra externa"

            st.write(f"**Obra seleccionada:** {selected_work_title}")
            st.write(f"**Autor candidato:** {selected_candidate}")

            contribution_fig = px.bar(
                filtered.sort_values("delta"),
                x="delta",
                y="token",
                orientation="h",
                labels={"delta": "Diferencia absoluta de z-scores", "token": "Token"},
                title=f"Contribuciones principales frente a {selected_candidate}",
            )

            contribution_fig.update_layout(height=max(400, 25 * len(filtered)))

            st.plotly_chart(contribution_fig, width="stretch")
            st.dataframe(
                filtered[contribution_cols].style.format(
                    {
                        "delta": "{:.3f}",
                        "z_ref": "{:.3f}",
                        "z_test": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    st.stop()
