"""Utils for the stylometry app."""

from pathlib import Path

import pandas as pd
import streamlit as st

from whodunit_stylometry.analysis.classical import add_tokens, load_corpus
from whodunit_stylometry.analysis.eda import EDAConfig
from whodunit_stylometry.analysis.embeddings import EmbeddingsConfig
from whodunit_stylometry.analysis.supervised import SupervisedConfig
from whodunit_stylometry.analysis.unsupervised import (
    HierarchicalConfig,
    UnsupervisedConfig,
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


def eda_cache_key(df: pd.DataFrame, lowercase: bool, config: EDAConfig) -> tuple:
    """Build a cache key for exploratory data analysis results.

    Args:
        df: DataFrame used to compute the corpus fingerprint.
        lowercase: Whether text normalization lowercases the corpus.
        config: EDA configuration containing cache-relevant analysis settings.

    Returns:
        A tuple containing the corpus fingerprint, the lowercase flag, and selected
        EDA configuration values.
    """
    return (
        corpus_cache_fingerprint(df),
        lowercase,
        config.top_n,
        config.ngram_top_k,
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
