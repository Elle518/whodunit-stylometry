"""Classical stylometry pipeline for interactive use."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from whodunit_stylometry.constants import STOPWORDS
from whodunit_stylometry.utils.data_utils import discover_corpus
from whodunit_stylometry.utils.nlp_utils import (
    CustomTokenizer,
    fix_gutenberg_linebreaks,
    is_alpha_tokens,
    normalize_text_for_tokenization,
)
from whodunit_stylometry.utils.stats_utils import (
    build_global_vocab,
    burrows_delta,
    compute_feature_stats,
    get_pairwise_burrows_contributions,
    get_pairwise_contributions,
    kilgariff_chi2,
)


@dataclass(frozen=True)
class ClassicalAnalysisConfig:
    """Configuration for the classical stylometry pipeline."""

    vocab_size: int = 500
    min_freq: int = 1
    use_function_words: bool = True
    lowercase: bool = True


def load_corpus(corpus_dir: str | Path) -> pd.DataFrame:
    """Load a corpus organized as ``corpus_dir/author/*.txt``."""

    path = Path(corpus_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"La ruta no es un directorio: {path}")

    df = discover_corpus(path)
    if df.empty:
        raise ValueError("No se han encontrado archivos .txt en subcarpetas de autor.")

    return df


def tokenize_text(text: str, tokenizer: CustomTokenizer, lowercase: bool = True) -> list[str]:
    """Normalize and tokenize a text, keeping only alphabetic tokens."""

    normalized = normalize_text_for_tokenization(fix_gutenberg_linebreaks(text))
    tokens = is_alpha_tokens(tokenizer.tokenize(normalized), keep_alpha=True)
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens


def add_tokens(df: pd.DataFrame, lowercase: bool = True) -> pd.DataFrame:
    """Attach reusable token lists to the corpus table."""

    tokenizer = CustomTokenizer()
    out = df.copy()
    out["tokens"] = [tokenize_text(text, tokenizer=tokenizer, lowercase=lowercase) for text in out["text"]]
    out["token_count"] = out["tokens"].map(len)
    out["type_count"] = out["tokens"].map(lambda tokens: len(set(tokens)))
    out["ttr"] = out.apply(
        lambda row: row["type_count"] / row["token_count"] if row["token_count"] else np.nan,
        axis=1,
    )
    return out


def corpus_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return work-level and author-level descriptive summaries."""

    work_summary = df[
        ["author", "work", "filename", "char_count", "word_count", "token_count", "type_count", "ttr"]
    ].copy()

    author_summary = (
        df.groupby("author", as_index=False)
        .agg(
            n_works=("work", "count"),
            total_tokens=("token_count", "sum"),
            mean_tokens_per_work=("token_count", "mean"),
            total_types=("tokens", lambda rows: len({token for row in rows for token in row})),
            mean_ttr=("ttr", "mean"),
        )
        .sort_values("total_tokens", ascending=False)
    )

    return work_summary, author_summary


def tokens_by_author(df: pd.DataFrame) -> dict[str, list[str]]:
    """Aggregate tokens by author."""

    return {author: [token for tokens in sub["tokens"] for token in tokens] for author, sub in df.groupby("author")}


def tokens_by_work(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return a dictionary with one token sequence per work."""

    return {
        f"{row.author} / {row.work}": row.tokens for row in df[["author", "work", "tokens"]].itertuples(index=False)
    }


def filter_feature_tokens(tokens: list[str], use_function_words: bool) -> list[str]:
    """Optionally keep only function words for classical distance methods."""

    if not use_function_words:
        return tokens
    return [token for token in tokens if token in STOPWORDS]


def prepare_feature_tokens(df: pd.DataFrame, use_function_words: bool) -> pd.DataFrame:
    """Add the token column used by distance-based methods."""

    out = df.copy()
    out["feature_tokens"] = out["tokens"].map(lambda tokens: filter_feature_tokens(tokens, use_function_words))
    out["feature_token_count"] = out["feature_tokens"].map(len)
    return out


def top_tokens_by_author(df: pd.DataFrame, top_n: int = 20, use_function_words: bool = True) -> pd.DataFrame:
    """Compute top tokens for each author."""

    rows = []
    for author, sub in df.groupby("author"):
        counter = Counter(
            token for tokens in sub["tokens"] for token in filter_feature_tokens(tokens, use_function_words)
        )
        total = sum(counter.values())
        for rank, (token, count) in enumerate(counter.most_common(top_n), start=1):
            rows.append(
                {
                    "author": author,
                    "rank": rank,
                    "token": token,
                    "count": count,
                    "relative_frequency": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def leave_one_work_out_classification(
    df: pd.DataFrame,
    method: str,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify each work against author profiles built without that work."""

    if method not in {"kilgariff", "burrows"}:
        raise ValueError("method must be 'kilgariff' or 'burrows'")

    feature_df = prepare_feature_tokens(df, use_function_words=config.use_function_words)
    rows = []
    contribution_rows = []

    for test_idx, test_row in feature_df.iterrows():
        train_df = feature_df.drop(index=test_idx)
        corpora = {
            author: [token for tokens in sub["feature_tokens"] for token in tokens]
            for author, sub in train_df.groupby("author")
        }
        corpora = {author: tokens for author, tokens in corpora.items() if tokens}
        test_tokens = test_row["feature_tokens"]

        if len(corpora) < 2 or not test_tokens:
            rows.append(_skipped_row(test_row, method, "No hay suficientes tokens para comparar."))
            continue

        vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)
        if not vocab:
            rows.append(_skipped_row(test_row, method, "El vocabulario global ha quedado vacío."))
            continue

        try:
            if method == "kilgariff":
                distances, contributions = _kilgariff_distances(corpora, test_tokens, vocab)
                distance_label = "chi2"
            else:
                distances, contributions = _burrows_distances(corpora, test_tokens, vocab)
                distance_label = "delta"
        except ValueError as exc:
            rows.append(_skipped_row(test_row, method, str(exc)))
            continue

        ranking = sorted(distances.items(), key=lambda item: item[1])
        pred_author, min_distance = ranking[0]
        second_distance = ranking[1][1] if len(ranking) > 1 else np.nan

        result_row = {
            "method": method,
            "author": test_row["author"],
            "work": test_row["work"],
            "pred_author": pred_author,
            "correct": pred_author == test_row["author"],
            "min_distance": min_distance,
            "second_distance": second_distance,
            "margin_to_second": second_distance - min_distance if pd.notna(second_distance) else np.nan,
            "status": "ok",
        }
        result_row.update({f"dist_{author}": value for author, value in distances.items()})
        rows.append(result_row)

        best_contributions = contributions[pred_author][:10]
        for rank, contribution in enumerate(best_contributions, start=1):
            contribution_rows.append(
                _format_contribution(method, test_row, pred_author, rank, contribution, distance_label)
            )

    return pd.DataFrame(rows), pd.DataFrame(contribution_rows)


def classify_external_works(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify works from a separate test corpus against the reference corpus."""

    if method not in {"kilgariff", "burrows"}:
        raise ValueError("method must be 'kilgariff' or 'burrows'")

    reference_df = prepare_feature_tokens(reference_df, use_function_words=config.use_function_words)
    test_df = prepare_feature_tokens(test_df, use_function_words=config.use_function_words)
    corpora = {
        author: [token for tokens in sub["feature_tokens"] for token in tokens]
        for author, sub in reference_df.groupby("author")
    }
    corpora = {author: tokens for author, tokens in corpora.items() if tokens}
    test_works = {
        f"{row.author} / {row.work}": row.feature_tokens
        for row in test_df[["author", "work", "feature_tokens"]].itertuples(index=False)
    }

    if method == "kilgariff":
        raw_results = _results_to_tables(
            get_pairwise_contributions(corpora, test_works, config.vocab_size, config.min_freq),
            method=method,
            distance_key="chi2",
        )
    else:
        raw_results = _results_to_tables(
            get_pairwise_burrows_contributions(corpora, test_works, config.vocab_size, config.min_freq),
            method=method,
            distance_key="delta",
        )

    results, contributions = raw_results
    truth = test_df.assign(work_key=test_df["author"] + " / " + test_df["work"])[["work_key", "author", "work"]]
    results = results.merge(truth, left_on="work", right_on="work_key", how="left").drop(columns=["work_key"])
    results["correct"] = results["pred_author"] == results["author"]
    return results, contributions


def _kilgariff_distances(
    corpora: dict[str, list[str]],
    test_tokens: list[str],
    vocab: list[str],
) -> tuple[dict[str, float], dict[str, list[tuple]]]:
    distances = {}
    contributions = {}
    for author, author_tokens in corpora.items():
        chi2, token_contributions = kilgariff_chi2(author_tokens, test_tokens, vocab)
        distances[author] = chi2
        contributions[author] = token_contributions
    return distances, contributions


def _burrows_distances(
    corpora: dict[str, list[str]],
    test_tokens: list[str],
    vocab: list[str],
) -> tuple[dict[str, float], dict[str, list[tuple]]]:
    means, stds, _ = compute_feature_stats(corpora, vocab)
    distances = {}
    contributions = {}
    for author, author_tokens in corpora.items():
        delta, token_contributions = burrows_delta(author_tokens, test_tokens, vocab, means, stds)
        distances[author] = delta
        contributions[author] = token_contributions
    return distances, contributions


def _results_to_tables(
    details: dict[str, dict[str, dict]],
    method: str,
    distance_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    contribution_rows = []
    for work, author_details in details.items():
        ranking = sorted(
            ((author, values[distance_key]) for author, values in author_details.items()),
            key=lambda item: item[1],
        )
        pred_author, min_distance = ranking[0]
        second_distance = ranking[1][1] if len(ranking) > 1 else np.nan
        row = {
            "method": method,
            "work": work,
            "pred_author": pred_author,
            "min_distance": min_distance,
            "second_distance": second_distance,
            "margin_to_second": second_distance - min_distance if pd.notna(second_distance) else np.nan,
            "status": "ok",
        }
        row.update({f"dist_{author}": distance for author, distance in ranking})
        result_rows.append(row)

        for rank, contribution in enumerate(author_details[pred_author]["contributions"][:10], start=1):
            contribution_rows.append(
                _format_contribution(
                    method, {"author": None, "work": work}, pred_author, rank, contribution, distance_key
                )
            )

    return pd.DataFrame(result_rows), pd.DataFrame(contribution_rows)


def _format_contribution(
    method: str,
    test_row,
    candidate_author: str,
    rank: int,
    contribution: tuple,
    distance_label: str,
) -> dict[str, object]:
    if method == "kilgariff":
        token, chi, obs_ref, exp_ref, obs_test, exp_test = contribution
        return {
            "method": method,
            "author": test_row["author"],
            "work": test_row["work"],
            "candidate_author": candidate_author,
            "rank": rank,
            "token": token,
            distance_label: chi,
            "obs_ref": obs_ref,
            "exp_ref": exp_ref,
            "obs_test": obs_test,
            "exp_test": exp_test,
        }

    token, diff, z_ref, z_test = contribution
    return {
        "method": method,
        "author": test_row["author"],
        "work": test_row["work"],
        "candidate_author": candidate_author,
        "rank": rank,
        "token": token,
        distance_label: diff,
        "z_ref": z_ref,
        "z_test": z_test,
    }


def _skipped_row(test_row, method: str, reason: str) -> dict[str, object]:
    return {
        "method": method,
        "author": test_row["author"],
        "work": test_row["work"],
        "pred_author": None,
        "correct": False,
        "min_distance": np.nan,
        "second_distance": np.nan,
        "margin_to_second": np.nan,
        "status": reason,
    }
