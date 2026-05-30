"""Exploratory corpus analysis workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

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
]

STRUCTURAL_METRICS = [
    "n_tokens_all",
    "n_tokens_alpha",
    "n_tokens_not_alpha",
    "non_alpha_token_ratio",
    "n_sentences",
    "avg_sentence_len",
    "median_sentence_len",
    "std_sentence_len",
    "sentences_per_1000_tokens",
    "short_sentence_ratio",
    "long_sentence_ratio",
    "n_paragraphs",
    "avg_paragraph_len",
    "median_paragraph_len",
    "std_paragraph_len",
    "paragraphs_per_1000_tokens",
    "avg_sentences_per_paragraph",
    "median_sentences_per_paragraph",
]

LEXICAL_METRICS = [
    "n_types",
    "ttr",
    "hapax_count",
    "hapax_ratio",
    "avg_word_len",
    "median_word_len",
    "long_word_ratio",
    "mattr_100",
    "stopword_ratio",
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
    "punctuation_ratio",
]

STABLE_AUTHOR_METRICS = [
    "avg_sentence_len",
    "median_sentence_len",
    "std_sentence_len",
    "sentences_per_1000_tokens",
    "short_sentence_ratio",
    "long_sentence_ratio",
    "avg_paragraph_len",
    "median_paragraph_len",
    "std_paragraph_len",
    "paragraphs_per_1000_tokens",
    "avg_sentences_per_paragraph",
    "median_sentences_per_paragraph",
    "mattr_100",
    "stopword_ratio",
    "avg_word_len",
    "median_word_len",
    "long_word_ratio",
    "non_alpha_token_ratio",
    *PUNCTUATION_METRICS,
]


@dataclass(frozen=True)
class EDAConfig:
    """Configuration for exploratory corpus analysis."""

    top_n: int = 20
    zipf_max_rank: int = 5000


def run_eda_core_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute core exploratory data analysis outputs for a corpus.

    Builds document-level metrics, groups outlier payloads by metric category,
    computes author-level descriptive summaries, derives author metric means,
    standardizes those means for heatmap use, and runs author-level PCA.

    Args:
        df: Input corpus data used to compute document- and author-level
            exploratory metrics. The required columns are determined by
            `_build_metrics_df()` and the downstream helper functions.

    Returns:
        A dictionary containing the following analysis outputs:
            - `metrics_df`: Document-level metrics.
            - `outlier_groups`: Outlier payloads grouped by metric category.
            - `desc_by_author`: Descriptive statistics grouped by author.
            - `author_metric_means`: Mean metric values per author.
            - `author_heatmap_df`: Standardized author metric values for heatmap
              visualization.
            - `author_pca_df`: Author-level PCA coordinates.
            - `pca_variance`: Explained variance information from PCA.
            - `pca_loadings_df`: PCA loading values by metric.
    """

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
    author_pca_df, pca_variance, pca_loadings_df = _author_pca(author_metric_means)

    return {
        "metrics_df": metrics_df,
        "outlier_groups": outlier_groups,
        "desc_by_author": desc_by_author,
        "author_metric_means": author_metric_means,
        "author_heatmap_df": author_heatmap_df,
        "author_pca_df": author_pca_df,
        "pca_variance": pca_variance,
        "pca_loadings_df": pca_loadings_df,
    }


def run_eda_vocab_analysis(df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    """Compute author-level vocabulary analysis tables.

    Tokenizes the input corpus by author, then computes top words with and
    without stopwords, top bigrams, top trigrams, and Zipf rank-frequency data
    according to the provided EDA configuration.

    Args:
        df: Input corpus data. Required columns are determined by
            `_tokens_by_author()` and the downstream vocabulary helper
            functions.
        config: EDA configuration containing vocabulary settings, including
            `top_n` for top token and n-gram counts, and `zipf_max_rank` for
            Zipf output limits.

    Returns:
        A dictionary containing the following vocabulary analysis outputs:
            - `top_words`: Top words by author, including stopwords.
            - `top_words_no_stop`: Top words by author after stopword removal.
            - `top_bigrams`: Top bigrams by author.
            - `top_trigrams`: Top trigrams by author.
            - `zipf_df`: Zipf rank-frequency data by author.
    """

    tokens_by_author = _tokens_by_author(df)
    return {
        "top_words": _top_words_by_author(tokens_by_author, config.top_n, remove_stopwords=False),
        "top_words_no_stop": _top_words_by_author(tokens_by_author, config.top_n, remove_stopwords=True),
        "top_bigrams": _top_ngrams_by_author(tokens_by_author, n=2, top_k=config.top_n),
        "top_trigrams": _top_ngrams_by_author(tokens_by_author, n=3, top_k=config.top_n),
        "zipf_df": _zipf_by_author(tokens_by_author, max_rank=config.zipf_max_rank),
    }


def _build_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build a document-level metrics dataframe from corpus rows.

    Iterates over each row in the input dataframe, computes text metrics using
    `compute_novel_metrics()`, and adds identifying metadata and basic size
    counts for each document.

    Args:
        df: Input corpus dataframe. Each row is expected to provide `text`,
            `tokens`, `author`, `work`, and `filename` fields. If a
            `tokens_all` field exists, it is passed to `compute_novel_metrics()`;
            otherwise, `None` is passed.

    Returns:
        A dataframe with one row per input document. Each row contains the
        metrics returned by `compute_novel_metrics()` plus `author`, `work`,
        `filename`, `token_count`, and `char_count` columns.
    """

    rows = []
    for row in df.itertuples(index=False):
        tokens_all = getattr(row, "tokens_all", None)
        metrics = compute_novel_metrics(row.text, STOPWORDS, tokens_all=tokens_all, tokens_alpha=row.tokens)
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
    """Build summary and row-level outlier data for selected metric columns.

    Filters the requested metric columns to those present in `metrics_df`, adds
    IQR-based outlier flags, extracts rows with at least one outlier, and formats
    the outlier metric names for display.

    Args:
        metrics_df: Metrics dataframe containing document metadata and numeric
            metric columns. Expected metadata columns are `author`, `work`, and
            `filename`. `add_iqr_outlier_flags()` is expected to add
            `has_any_outlier` and `outlier_metrics` columns to its checked
            dataframe output.
        metric_cols: Candidate metric column names to check for outliers.
            Missing columns are ignored.

    Returns:
        A dictionary containing:
            - `summary`: Outlier summary dataframe with the original index reset
              into a `metric` column.
            - `outliers`: Dataframe of rows where `has_any_outlier` is true,
              including metadata, formatted outlier metric names, and available
              metric values.
            - `metrics`: List of metric columns that were present in
              `metrics_df` and included in the outlier check.
    """

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
    """Compute descriptive statistics for selected metrics grouped by author.

    Filters the requested metric columns to those present in `metrics_df`, then
    computes pandas descriptive statistics for each available metric grouped by
    author. Percentile statistic names are normalized from `25%`, `50%`, and
    `75%` to `p25`, `p50`, and `p75`.

    Args:
        metrics_df: Metrics dataframe containing an `author` column and the
            metric columns to summarize.
        metric_cols: Candidate metric column names to summarize. Missing columns
            are ignored.

    Returns:
        A dataframe with one row per author and flattened summary-statistic
        columns. Each generated metric column follows the pattern
        `{metric}_{stat}`, such as `token_count_mean` or `sentence_count_p50`.
    """

    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    desc = metrics_df.groupby("author")[available_cols].describe()
    desc.columns = [
        f"{metric}_{stat}".replace("25%", "p25").replace("50%", "p50").replace("75%", "p75")
        for metric, stat in desc.columns
    ]
    return desc.reset_index()


def _author_metric_means(desc_by_author: pd.DataFrame) -> pd.DataFrame:
    """Extract author-level metric means from descriptive statistics.

    Selects the `author` column and all columns ending in `_mean`, then removes
    the `_mean` suffix from the selected metric column names.

    Args:
        desc_by_author: Author-level descriptive statistics dataframe containing
            an `author` column and zero or more metric mean columns whose names
            end with `_mean`.

    Returns:
        A dataframe containing the `author` column and one column per extracted
        metric mean. Metric mean columns are renamed by removing the `_mean`
        suffix.
    """

    mean_cols = [col for col in desc_by_author.columns if col.endswith("_mean")]
    out = desc_by_author[["author", *mean_cols]].copy()
    out = out.rename(columns={col: col.removesuffix("_mean") for col in mean_cols})
    return out


def _standardized_author_heatmap(author_metric_means: pd.DataFrame) -> pd.DataFrame:
    """Standardize author metric means for heatmap visualization.

    Uses all columns except `author` as metric columns and applies standard
    scaling so each metric has zero mean and unit variance across authors. The
    returned dataframe is indexed by author and preserves the metric column
    names.

    Args:
        author_metric_means: Dataframe containing an `author` column and one or
            more numeric metric columns with author-level mean values.

    Returns:
        A dataframe of standardized metric values. The index contains author
        values from `author_metric_means["author"]`, and the columns are the
        standardized metric columns.
    """

    metric_cols = [col for col in author_metric_means.columns if col != "author"]
    scaled = StandardScaler().fit_transform(author_metric_means[metric_cols])
    return pd.DataFrame(scaled, index=author_metric_means["author"], columns=metric_cols)


def _author_pca(author_metric_means: pd.DataFrame) -> tuple[pd.DataFrame, list[float], pd.DataFrame]:
    """Run PCA on standardized author-level metric means.

    Uses all columns except `author` as metric columns, standardizes them across
    authors, and computes up to two principal components. The returned PCA
    coordinate dataframe always includes `PC1` and `PC2`; when only one
    component can be computed, `PC2` is filled with `0.0`.

    Args:
        author_metric_means: Dataframe containing an `author` column and one or
            more numeric metric columns with author-level mean values.

    Returns:
        A tuple containing:
            - PCA coordinates dataframe with `author`, `PC1`, and `PC2` columns.
            - Explained variance ratio values for the fitted components.
            - PCA loadings dataframe with `component`, `metric`, `loading`, and
              `abs_loading` columns, sorted by component and descending absolute
              loading.
    """

    metric_cols = [col for col in author_metric_means.columns if col != "author"]
    n_components = min(2, len(author_metric_means), len(metric_cols))
    scaled = StandardScaler().fit_transform(author_metric_means[metric_cols])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    out = author_metric_means[["author"]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1] if n_components > 1 else 0.0

    loading_rows = []
    for component_idx in range(n_components):
        component = f"PC{component_idx + 1}"
        for metric, loading in zip(metric_cols, pca.components_[component_idx]):
            loading_rows.append(
                {
                    "component": component,
                    "metric": metric,
                    "loading": float(loading),
                    "abs_loading": abs(float(loading)),
                }
            )
    loadings_df = pd.DataFrame(loading_rows).sort_values(
        ["component", "abs_loading"],
        ascending=[True, False],
        ignore_index=True,
    )

    return out, [float(value) for value in pca.explained_variance_ratio_], loadings_df


def _tokens_by_author(df: pd.DataFrame) -> dict[str, list[str]]:
    """Collect normalized alphabetic tokens grouped by author.

    Groups the input dataframe by `author`, flattens each author's `tokens`
    values, keeps only alphabetic tokens, and lowercases the retained tokens.

    Args:
        df: Input corpus dataframe containing `author` and `tokens` columns.
            Each value in `tokens` is expected to be an iterable of string
            tokens.

    Returns:
        A dictionary mapping each author to a list of lowercase alphabetic
        tokens from that author's rows.
    """

    return {
        author: [token.lower() for tokens in sub["tokens"] for token in tokens if token.isalpha()]
        for author, sub in df.groupby("author")
    }


def _top_words_by_author(
    tokens_by_author: dict[str, list[str]],
    top_n: int,
    remove_stopwords: bool,
) -> pd.DataFrame:
    """Compute top word frequencies by author.

    Counts tokens for each author and returns the most common tokens up to
    `top_n`. Stopwords are excluded before counting when `remove_stopwords` is
    true.

    Args:
        tokens_by_author: Mapping of author names to token lists. Tokens are
            expected to already be normalized as needed by the caller.
        top_n: Maximum number of top tokens to return per author.
        remove_stopwords: Whether to exclude tokens present in `STOPWORDS`
            before counting.

    Returns:
        A dataframe with one row per selected token and the following columns:
        `author`, `rank`, `token`, `freq`, and `relative_freq`. The
        `relative_freq` value is the token frequency divided by the total number
        of selected tokens for that author, or `0.0` if the selected token count
        is zero.
    """

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
    """Compute top n-gram frequencies by author.

    Generates n-grams for each author's tokens using `top_ngrams()`, excluding
    stopwords according to `STOPWORDS`, and returns the most frequent n-grams up
    to `top_k` per author.

    Args:
        tokens_by_author: Mapping of author names to token lists. Tokens are
            expected to already be normalized as needed by the caller.
        n: Size of each n-gram to generate.
        top_k: Maximum number of top n-grams to return per author.

    Returns:
        A dataframe with one row per selected n-gram and the following columns:
        `author`, `rank`, `ngram`, and `freq`.
    """

    rows = []
    for author, tokens in tokens_by_author.items():
        for rank, (ngram, freq) in enumerate(top_ngrams(tokens, n=n, top_k=top_k, stopword_set=STOPWORDS), start=1):
            rows.append({"author": author, "rank": rank, "ngram": ngram, "freq": freq})
    return pd.DataFrame(rows)


def _zipf_by_author(tokens_by_author: dict[str, list[str]], max_rank: int) -> pd.DataFrame:
    """Compute Zipf rank-frequency data by author.

    Counts tokens for each author and returns token frequencies ordered by
    descending frequency up to `max_rank`.

    Args:
        tokens_by_author: Mapping of author names to token lists. Tokens are
            expected to already be normalized as needed by the caller.
        max_rank: Maximum number of ranked tokens to return per author.

    Returns:
        A dataframe with one row per ranked token and the following columns:
        `author`, `rank`, `token`, `freq`, and `relative_freq`. The
        `relative_freq` value is the token frequency divided by the total token
        count for that author, or `0.0` if the author has no tokens.
    """

    rows = []
    for author, tokens in tokens_by_author.items():
        counter = Counter(tokens)
        total = sum(counter.values())
        for rank, (token, freq) in enumerate(counter.most_common(max_rank), start=1):
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
