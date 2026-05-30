"""Utils for the stylometry app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from whodunit_stylometry.analysis.eda import EDAConfig
from whodunit_stylometry.analysis.embeddings import EmbeddingsConfig
from whodunit_stylometry.analysis.supervised import SupervisedConfig
from whodunit_stylometry.analysis.unsupervised import (
    HierarchicalConfig,
    UnsupervisedConfig,
)
from whodunit_stylometry.utils.data_utils import load_corpus
from whodunit_stylometry.utils.nlp_utils import (
    add_tokens,
)

ANALYSIS_TYPES = {
    "Análisis exploratorio": "eda",
    "Métodos clásicos": "classical",
    "Métodos supervisados": "supervised",
    "Métodos no supervisados": "unsupervised",
    "Embeddings de OpenAI": "embeddings",
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

UNSUPERVISED_WORKFLOWS = {
    "Experimento de clustering": "experiment",
    "Robustez MFW": "robustness",
    "Clustering jerárquico": "hierarchical",
}

SIDEBAR_ICON_PATH = Path("logo.png")


def metric_card(label: str, value) -> None:
    """Display a Streamlit metric card.

    Args:
        label: Text label shown above the metric value.
        value: Value displayed in the metric card.
    """

    st.metric(label=label, value=value)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string into a list of integers.

    Args:
        value: Comma-separated string containing integer values.

    Returns:
        A list of parsed integers.
    """

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def apply_work_box_hover(fig, metric_name: str):
    """Applies a stable hover format to work-level metric box plots.

    Args:
        fig: Figure-like object with an ``update_traces`` method, such as a
            Plotly figure.
        metric_name: Name of the metric displayed in the hover label.
    """

    fig.update_traces(
        hovertemplate=("Autor: %{x}<br>" "Obra: %{customdata[0]}<br>" f"{metric_name}: %{{y:.6f}}" "<extra></extra>")
    )


def apply_correlation_heatmap_layout(fig, n_features: int):
    """Scales a correlation heatmap layout to keep metric names readable.

    Args:
        fig: Figure-like object with ``update_layout``, ``update_xaxes``, and
            ``update_yaxes`` methods, such as a Plotly figure.
        n_features: Number of features shown in the correlation heatmap.
    """

    size = max(520, min(980, 70 * n_features))
    fig.update_layout(
        height=size,
        margin=dict(l=220, r=40, t=60, b=180),
    )
    fig.update_xaxes(tickangle=45, automargin=True)
    fig.update_yaxes(automargin=True)


def show_metric_table_by_work(metrics_df: pd.DataFrame, metric_cols: list[str]):
    """Displays work-level values for the selected metric columns.

    Args:
        metrics_df: DataFrame containing work-level metric data and optional
            identifier columns such as ``author``, ``work``, and ``filename``.
        metric_cols: Metric column names requested for display. Only columns
            present in ``metrics_df`` are shown.
    """

    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    id_cols = [col for col in ["author", "work", "filename"] if col in metrics_df.columns]
    st.dataframe(
        metrics_df[[*id_cols, *available_cols]],
        width="stretch",
        hide_index=True,
    )


#####################
# CACHING FUNCTIONS #
#####################
@st.cache_data(show_spinner=False)
def cached_load_and_tokenize(corpus_path: str, lowercase: bool) -> pd.DataFrame:
    """Load a corpus from disk and return a tokenized DataFrame.

    The result is cached by Streamlit based on the function arguments.

    Args:
        corpus_path: Path to the corpus file or directory to load.
        lowercase: Whether tokens should be converted to lowercase.

    Returns:
        A DataFrame containing the loaded corpus with tokens added.
    """

    return add_tokens(load_corpus(corpus_path), lowercase=lowercase)


def corpus_cache_fingerprint(df: pd.DataFrame) -> tuple[tuple[int, int], int]:
    """Create a lightweight fingerprint for a corpus DataFrame.

    This feature helps determine if the corpus has changed and, therefore, if
    complex analyses need to be recalculated.

    It's necessary because in Streamlit, the script is rerun many times: when
    changing a tab, moving a slider, uploading a file, etc. Without this function,
    the application could recalculate excessively or, worse, use results from
    an older corpus.

    Args:
        df: Corpus DataFrame to fingerprint.

    Returns:
        A tuple containing:
            - The DataFrame shape as ``(row_count, column_count)``.
            - An integer hash derived from the selected DataFrame values.
    """

    fingerprint_cols = [col for col in ["author", "work", "filename", "text"] if col in df.columns]
    fingerprint_df = df[fingerprint_cols].astype(str) if fingerprint_cols else df.astype(str)
    fingerprint_hash = pd.util.hash_pandas_object(fingerprint_df, index=False).sum()
    return df.shape, int(fingerprint_hash)


def eda_core_cache_key(df: pd.DataFrame, lowercase: bool) -> tuple:
    """Build a cache key for core exploratory data analysis results.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.

    Returns:
        A tuple containing the corpus fingerprint and the lowercase flag.
    """

    return (
        corpus_cache_fingerprint(df),
        lowercase,
    )


def eda_vocab_cache_key(df: pd.DataFrame, lowercase: bool, config: EDAConfig) -> tuple:
    """Build a cache key for EDA vocabulary tables.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.
        config: EDA configuration containing vocabulary settings.

    Returns:
        A tuple containing the corpus fingerprint, the lowercase flag, and selected
        vocabulary configuration values.
    """

    return (
        corpus_cache_fingerprint(df),
        lowercase,
        config.top_n,
        config.zipf_max_rank,
    )


def supervised_experiment_cache_key(
    df: pd.DataFrame,
    lowercase: bool,
    config: SupervisedConfig,
) -> tuple:
    """Build a cache key for supervised experiment results.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.
        config: Supervised experiment configuration containing cache-relevant
            settings.

    Returns:
        A tuple containing the corpus fingerprint, the lowercase flag, selected
        supervised experiment configuration values, and the selected models as
        an immutable tuple.
    """

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


def unsupervised_experiment_cache_key(
    df: pd.DataFrame,
    lowercase: bool,
    config: UnsupervisedConfig,
) -> tuple:
    """Build a cache key for unsupervised experiment results.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.
        config: Unsupervised experiment configuration containing cache-relevant
            settings.

    Returns:
        A tuple containing the corpus fingerprint, the lowercase flag, selected
        unsupervised experiment configuration values, the selected models as an
        immutable tuple, and the number of clusters.
    """

    return (
        corpus_cache_fingerprint(df),
        lowercase,
        config.feature_set,
        config.top_n_mfw,
        config.seed,
        tuple(config.selected_models),
        config.n_clusters,
    )


def hierarchical_experiment_cache_key(
    df: pd.DataFrame,
    lowercase: bool,
    config: HierarchicalConfig,
) -> tuple:
    """Build a cache key for hierarchical experiment results.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.
        config: Hierarchical experiment configuration containing cache-relevant
            settings.

    Returns:
        A tuple containing the corpus fingerprint, the lowercase flag, selected
        hierarchical experiment configuration values, selected methods as an
        immutable tuple, the dendrogram method, and the number of clusters.
    """

    return (
        corpus_cache_fingerprint(df),
        lowercase,
        config.feature_set,
        config.top_n_mfw,
        tuple(config.selected_methods),
        config.dendrogram_method,
        config.n_clusters,
    )


def embeddings_experiment_cache_key(
    df: pd.DataFrame,
    lowercase: bool,
    config: EmbeddingsConfig,
) -> tuple:
    """Build a cache key for embeddings experiment results.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.
        config: Embeddings experiment configuration containing cache-relevant
            settings.

    Returns:
        A tuple containing the corpus fingerprint, the lowercase flag, and
        selected embeddings experiment configuration values.
    """

    return (
        corpus_cache_fingerprint(df),
        lowercase,
        config.model,
        config.dimensions,
        config.chunk_tokens,
        config.chunk_overlap,
        config.min_chunk_tokens,
        config.batch_size,
        config.max_retries,
        config.random_state,
        config.umap_neighbors,
        config.umap_min_dist,
        config.network_top_k,
        config.nearest_neighbors_k,
    )
