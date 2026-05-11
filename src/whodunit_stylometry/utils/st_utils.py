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


@st.cache_data(show_spinner=False)
def cached_load_and_tokenize(corpus_path: str, lowercase: bool) -> pd.DataFrame:
    return add_tokens(load_corpus(corpus_path), lowercase=lowercase)


def metric_card(label: str, value) -> None:
    st.metric(label=label, value=value)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def corpus_cache_fingerprint(df: pd.DataFrame) -> tuple[tuple[int, int], int]:
    fingerprint_cols = [col for col in ["author", "work", "filename", "text"] if col in df.columns]
    fingerprint_df = df[fingerprint_cols].astype(str) if fingerprint_cols else df.astype(str)
    fingerprint_hash = pd.util.hash_pandas_object(fingerprint_df, index=False).sum()
    return df.shape, int(fingerprint_hash)


def eda_cache_key(df: pd.DataFrame, lowercase: bool, config: EDAConfig) -> tuple:
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
