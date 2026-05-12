"""Exploratory corpus analysis workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from whodunit_stylometry.constants import STOPWORDS
from whodunit_stylometry.utils.nlp_utils import compute_novel_metrics, top_ngrams
from whodunit_stylometry.utils.stats_utils import add_iqr_outlier_flags

QUALITY_METRICS = [
    "n_chars",
    "non_alpha_ratio",
    "digit_ratio",
    "whitespace_ratio",
    "non_alpha_token_ratio",
]

STRUCTURAL_METRICS = [
    "n_tokens_all",
    "n_sentences",
    "n_paragraphs",
    "avg_sentence_len",
    "median_sentence_len",
    "std_sentence_len",
    "sentences_per_1000_tokens",
    "avg_paragraph_len",
    "paragraphs_per_1000_tokens",
    "avg_sentences_per_paragraph",
]

LEXICAL_METRICS = [
    "n_types",
    "ttr",
    "hapax_ratio",
    "mattr_100",
    "stopword_ratio",
    "avg_word_len",
    "median_word_len",
    "long_word_ratio",
]

PUNCTUATION_METRICS = [
    "comma_count_per_1000",
    "period_count_per_1000",
    "semicolon_count_per_1000",
    "colon_count_per_1000",
    "exclam_count_per_1000",
    "question_count_per_1000",
    "ellipsis_count_per_1000",
    "quote_count_per_1000",
    "dialog_dash_count_per_1000",
    "hyphen_count_per_1000",
    "parenthesis_count_per_1000",
]

STABLE_AUTHOR_METRICS = [
    "avg_sentence_len",
    "median_sentence_len",
    "std_sentence_len",
    "sentences_per_1000_tokens",
    "short_sentence_ratio",
    "long_sentence_ratio",
    "avg_paragraph_len",
    "paragraphs_per_1000_tokens",
    "avg_sentences_per_paragraph",
    "mattr_100",
    "stopword_ratio",
    "avg_word_len",
    "long_word_ratio",
    "non_alpha_token_ratio",
    "punctuation_ratio",
    *PUNCTUATION_METRICS,
]


@dataclass(frozen=True)
class EDAConfig:
    """Configuration for exploratory corpus analysis."""

    top_n: int = 20
    zipf_max_rank: int = 5000


def run_eda_analysis(df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    """Compute corpus-level metrics, author summaries, vocabulary tables, and outliers."""

    metrics_df = _build_metrics_df(df)
    outlier_groups = {
        "quality": _outlier_payload(metrics_df, QUALITY_METRICS),
        "structural": _outlier_payload(metrics_df, STRUCTURAL_METRICS),
        "lexical": _outlier_payload(metrics_df, LEXICAL_METRICS),
        "punctuation": _outlier_payload(metrics_df, PUNCTUATION_METRICS),
    }
    desc_by_author = _describe_by_author(metrics_df, STABLE_AUTHOR_METRICS)
    author_metric_means = _author_metric_means(desc_by_author)
    author_heatmap_df = _standardized_author_heatmap(author_metric_means)
    author_pca_df, pca_variance = _author_pca(author_metric_means)
    tokens_by_author = _tokens_by_author(df)

    return {
        "metrics_df": metrics_df,
        "outlier_groups": outlier_groups,
        "desc_by_author": desc_by_author,
        "author_metric_means": author_metric_means,
        "author_heatmap_df": author_heatmap_df,
        "author_pca_df": author_pca_df,
        "pca_variance": pca_variance,
        "top_words": _top_words_by_author(tokens_by_author, config.top_n, remove_stopwords=False),
        "top_words_no_stop": _top_words_by_author(tokens_by_author, config.top_n, remove_stopwords=True),
        "top_bigrams": _top_ngrams_by_author(tokens_by_author, n=2, top_k=config.top_n),
        "top_trigrams": _top_ngrams_by_author(tokens_by_author, n=3, top_k=config.top_n),
        "zipf_df": _zipf_by_author(tokens_by_author, max_rank=config.zipf_max_rank),
    }


def _build_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        metrics = compute_novel_metrics(row.text, STOPWORDS)
        metrics.update(
            {
                "author": row.author,
                "work": row.work,
                "filename": row.filename,
                "token_count": len(row.tokens),
                "char_count": len(row.text),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _outlier_payload(metrics_df: pd.DataFrame, metric_cols: list[str]) -> dict[str, pd.DataFrame]:
    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    checked_df, summary_df = add_iqr_outlier_flags(metrics_df, available_cols)
    outlier_cols = ["author", "work", "filename", "outlier_metrics", *available_cols]
    outliers_df = checked_df.loc[checked_df["has_any_outlier"], outlier_cols].copy()
    outliers_df["outlier_metrics"] = outliers_df["outlier_metrics"].map(", ".join)
    return {
        "summary": summary_df.reset_index(names="metric"),
        "outliers": outliers_df,
        "metrics": available_cols,
    }


def _describe_by_author(metrics_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    desc = metrics_df.groupby("author")[available_cols].describe()
    desc.columns = [
        f"{metric}_{stat}".replace("25%", "p25").replace("50%", "p50").replace("75%", "p75")
        for metric, stat in desc.columns
    ]
    return desc.reset_index()


def _author_metric_means(desc_by_author: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [col for col in desc_by_author.columns if col.endswith("_mean")]
    out = desc_by_author[["author", *mean_cols]].copy()
    out = out.rename(columns={col: col.removesuffix("_mean") for col in mean_cols})
    return out


def _standardized_author_heatmap(author_metric_means: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in author_metric_means.columns if col != "author"]
    scaled = StandardScaler().fit_transform(author_metric_means[metric_cols])
    return pd.DataFrame(scaled, index=author_metric_means["author"], columns=metric_cols)


def _author_pca(author_metric_means: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    metric_cols = [col for col in author_metric_means.columns if col != "author"]
    n_components = min(2, len(author_metric_means), len(metric_cols))
    scaled = StandardScaler().fit_transform(author_metric_means[metric_cols])
    coords = PCA(n_components=n_components).fit_transform(scaled)
    out = author_metric_means[["author"]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1] if n_components > 1 else 0.0
    variance = PCA(n_components=n_components).fit(scaled).explained_variance_ratio_
    return out, [float(value) for value in variance]


def _tokens_by_author(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        author: [token.lower() for tokens in sub["tokens"] for token in tokens if token.isalpha()]
        for author, sub in df.groupby("author")
    }


def _top_words_by_author(
    tokens_by_author: dict[str, list[str]],
    top_n: int,
    remove_stopwords: bool,
) -> pd.DataFrame:
    rows = []
    for author, tokens in tokens_by_author.items():
        selected_tokens = [token for token in tokens if token not in STOPWORDS] if remove_stopwords else tokens
        counter = Counter(selected_tokens)
        total = sum(counter.values())
        for rank, (token, freq) in enumerate(counter.most_common(top_n), start=1):
            rows.append(
                {
                    "author": author,
                    "rank": rank,
                    "token": token,
                    "freq": freq,
                    "relative_freq": freq / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _top_ngrams_by_author(tokens_by_author: dict[str, list[str]], n: int, top_k: int) -> pd.DataFrame:
    rows = []
    for author, tokens in tokens_by_author.items():
        for rank, (ngram, freq) in enumerate(top_ngrams(tokens, n=n, top_k=top_k, stopword_set=STOPWORDS), start=1):
            rows.append({"author": author, "rank": rank, "ngram": ngram, "freq": freq})
    return pd.DataFrame(rows)


def _zipf_by_author(tokens_by_author: dict[str, list[str]], max_rank: int) -> pd.DataFrame:
    rows = []
    for author, tokens in tokens_by_author.items():
        counter = Counter(tokens)
        for rank, (token, freq) in enumerate(counter.most_common(max_rank), start=1):
            rows.append({"author": author, "rank": rank, "token": token, "freq": freq})
    return pd.DataFrame(rows)
