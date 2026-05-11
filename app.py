from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from whodunit_stylometry.analysis.classical import (
    ClassicalAnalysisConfig,
    add_tokens,
    burrows_reference_tables,
    classify_text_with_burrows,
    classify_text_with_kilgariff,
    classify_text_with_mendenhall,
    compare_mendenhall_average_curves,
    compute_mendenhall_author_profiles,
    corpus_summary,
    kilgariff_reference_tables,
    load_corpus,
    mendenhall_average_curves_to_frame,
    top_tokens_by_author,
)
from whodunit_stylometry.analysis.supervised import (
    FEATURE_SETS,
    MODEL_NAMES,
    SupervisedConfig,
    classify_text_with_supervised_model,
    run_mfw_robustness,
    run_supervised_experiment,
)

ANALYSIS_TYPES = {
    "Métodos clásicos": "classical",
    "Métodos supervisados": "supervised",
    "Métodos no supervisados": "unsupervised",
}

CLASSICAL_METHODS = {
    "Test de Mendenhall": "mendenhall",
    "Chi-cuadrado de Kilgariff": "kilgariff",
    "Distancia de Burrows": "burrows",
}

ML_WORKFLOWS = {
    "Experimento supervisado": "experiment",
    "Robustez MFW": "robustness",
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


def show_placeholder(title: str) -> None:
    st.info(f"{title} todavía no está implementado en la app. La interfaz ya queda preparada para añadirlo.")


def metric_card(label: str, value) -> None:
    st.metric(label=label, value=value)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def corpus_cache_fingerprint(df: pd.DataFrame) -> tuple[tuple[int, int], int]:
    fingerprint_cols = [col for col in ["author", "work", "filename", "text"] if col in df.columns]
    fingerprint_df = df[fingerprint_cols].astype(str) if fingerprint_cols else df.astype(str)
    fingerprint_hash = pd.util.hash_pandas_object(fingerprint_df, index=False).sum()
    return df.shape, int(fingerprint_hash)


def supervised_experiment_cache_key(
    df: pd.DataFrame,
    lowercase: bool,
    config: SupervisedConfig,
) -> tuple:
    return (
        corpus_cache_fingerprint(df),
        lowercase,
        config.feature_set,
        config.top_n_mfw,
        config.seed,
        config.n_test_per_author,
        config.keep_correlated_features,
        config.correlation_threshold,
        tuple(config.selected_models),
    )


st.title("Whodunit Stylometry 🕵")

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


if selected_analysis == "unsupervised":
    show_placeholder(selected_analysis_label)
    st.stop()

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
                    {"parámetro": "Tolerancia", "valor": ml_tolerance},
                ]
            )
            st.dataframe(config_df, width="stretch", hide_index=True)

        with summary_tab:
            st.subheader("Rendimiento agregado")
            summary_df = robustness["summary_df"].copy()
            summary_df["mode_best_model_name"] = summary_df["mode_best_model_name"].map(model_display_names)
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
                markers=True,
                error_y="std_test_f1_macro",
                labels={"top_n_mfw": "TOP_N_MFW", "mean_test_f1_macro": "F1-macro medio"},
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mejor media observada", f"{robustness['best_mean_score']:.3f}")
            with col2:
                st.metric("Umbral", f"{robustness['threshold_score']:.3f}")
            with col3:
                st.metric("TOP_N_MFW seleccionado", robustness["selected_top_n_mfw"])

            decision_df = robustness["final_decision_df"].copy()
            decision_df["mode_best_model_name"] = decision_df["mode_best_model_name"].map(model_display_names)
            st.dataframe(decision_df, width="stretch", hide_index=True)

        with detail_tab:
            st.subheader("Todas las ejecuciones")
            detail_df = robustness["sweep_df"].copy()
            detail_df["best_model_name"] = detail_df["best_model_name"].map(model_display_names)
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
