"""Classical stylometry pipeline for interactive use."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import jensenshannon

from whodunit_stylometry.constants import STOPWORDS
from whodunit_stylometry.utils.data_utils import discover_corpus
from whodunit_stylometry.utils.nlp_utils import (
    CustomTokenizer,
    fix_gutenberg_linebreaks,
    is_alpha_tokens,
    normalize_text_for_tokenization,
)
from whodunit_stylometry.utils.stats_utils import (
    align_distributions,
    build_global_vocab,
    burrows_delta,
    compute_average_curve,
    compute_feature_stats,
    distance_matrix,
    get_pairwise_burrows_contributions,
    get_pairwise_contributions,
    kilgariff_chi2,
    word_length_distributions_random_blocks,
)


@dataclass(frozen=True)
class ClassicalAnalysisConfig:
    """Configuration for the classical stylometry pipeline."""

    vocab_size: int = 500
    min_freq: int = 1
    use_function_words: bool = True
    lowercase: bool = True
    block_size: int = 100_000
    n_blocks: int = 50
    max_word_len: int = 20
    seed: int = 0


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


def compute_mendenhall_author_profiles(
    df: pd.DataFrame,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[int, float]]]:
    """Compute Mendenhall block curves, stability stats and average curves."""

    rows = []
    stats_rows = []
    average_curves = {}

    for author, tokens in tokens_by_author(df).items():
        distributions = word_length_distributions_random_blocks(
            tokens,
            block_size=config.block_size,
            n_blocks=config.n_blocks,
            seed=config.seed,
        )
        average_curves[author] = compute_average_curve(distributions)

        aligned, lengths = align_distributions(distributions, max_len=config.max_word_len)
        distances = distance_matrix(aligned)
        upper_distances = distances[np.triu_indices_from(distances, k=1)]

        stats_rows.append(
            {
                "author": author,
                "mean_distance": float(np.mean(upper_distances)),
                "std_distance": float(np.std(upper_distances)),
                # "n_blocks": config.n_blocks,
                # "block_size": config.block_size,
            }
        )

        for block_id, aligned_distribution in enumerate(aligned, start=1):
            for length, relative_frequency in zip(lengths, aligned_distribution, strict=True):
                rows.append(
                    {
                        "author": author,
                        "block": block_id,
                        "word_length": length,
                        "relative_frequency": relative_frequency,
                    }
                )

    block_curves_df = pd.DataFrame(rows)
    stats_df = pd.DataFrame(stats_rows).sort_values("mean_distance")
    average_curves_df = mendenhall_average_curves_to_frame(average_curves, max_word_len=config.max_word_len)

    return block_curves_df, stats_df, average_curves_df, average_curves


def mendenhall_average_curves_to_frame(
    average_curves: dict[str, dict[int, float]],
    max_word_len: int = 20,
) -> pd.DataFrame:
    """Convert average Mendenhall curves to long-form tabular data."""

    aligned, lengths = align_distributions(list(average_curves.values()), max_len=max_word_len)
    rows = []
    for author, curve in zip(average_curves.keys(), aligned, strict=True):
        for length, relative_frequency in zip(lengths, curve, strict=True):
            rows.append(
                {
                    "author": author,
                    "word_length": length,
                    "relative_frequency": relative_frequency,
                }
            )
    return pd.DataFrame(rows)


def compare_mendenhall_average_curves(
    average_curves: dict[str, dict[int, float]],
    max_word_len: int = 20,
) -> pd.DataFrame:
    """Compute pairwise Jensen-Shannon distances between average curves."""

    authors = list(average_curves.keys())
    aligned, _ = align_distributions(list(average_curves.values()), max_len=max_word_len)
    distances = distance_matrix(aligned)
    return pd.DataFrame(distances, columns=authors, index=authors)


def classify_text_with_mendenhall(
    text: str,
    average_curves: dict[str, dict[int, float]],
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute one text using Mendenhall average curves."""
    tokenizer = CustomTokenizer()
    test_tokens = tokenize_text(text, tokenizer=tokenizer, lowercase=config.lowercase)
    distributions = word_length_distributions_random_blocks(
        test_tokens,
        block_size=5000,
        n_blocks=20,
        seed=1,
    )
    test_curve = compute_average_curve(distributions)
    aligned_test, lengths = align_distributions([test_curve], max_len=config.max_word_len)
    test_vector = aligned_test[0]

    rows = []
    for author, author_curve in average_curves.items():
        aligned, _ = align_distributions([test_curve, author_curve], max_len=config.max_word_len)
        distance = jensenshannon(aligned[0], aligned[1])
        rows.append({"author": author, "distance": distance})

    distances_df = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    distances_df["rank"] = distances_df.index + 1

    curve_df = pd.DataFrame(
        {
            "author": "Obra seleccionada",
            "word_length": lengths,
            "relative_frequency": test_vector,
        }
    )

    return distances_df, curve_df


def leave_one_work_out_classification(
    df: pd.DataFrame,
    method: str,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify each work against author profiles built without that work."""

    if method not in {"mendenhall", "kilgariff", "burrows"}:
        raise ValueError("method must be 'mendenhall', 'kilgariff' or 'burrows'")

    feature_df = (
        df.copy()
        if method == "mendenhall"
        else prepare_feature_tokens(df, use_function_words=config.use_function_words)
    )
    rows = []
    contribution_rows = []

    for test_idx, test_row in feature_df.iterrows():
        train_df = feature_df.drop(index=test_idx)
        token_col = "tokens" if method == "mendenhall" else "feature_tokens"
        corpora = {
            author: [token for tokens in sub[token_col] for token in tokens]
            for author, sub in train_df.groupby("author")
        }
        corpora = {author: tokens for author, tokens in corpora.items() if tokens}
        test_tokens = test_row[token_col]

        if len(corpora) < 2 or not test_tokens:
            rows.append(_skipped_row(test_row, method, "No hay suficientes tokens para comparar."))
            continue

        try:
            if method == "mendenhall":
                distances, contributions = _mendenhall_distances(
                    corpora,
                    test_tokens,
                    block_size=config.block_size,
                    n_blocks=config.n_blocks,
                    max_word_len=config.max_word_len,
                    seed=config.seed,
                )
                distance_label = "js_distance"
            elif method == "kilgariff":
                vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)
                if not vocab:
                    rows.append(_skipped_row(test_row, method, "El vocabulario global ha quedado vacío."))
                    continue
                distances, contributions = _kilgariff_distances(corpora, test_tokens, vocab)
                distance_label = "chi2"
            else:
                vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)
                if not vocab:
                    rows.append(_skipped_row(test_row, method, "El vocabulario global ha quedado vacío."))
                    continue
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


def _mendenhall_distances(
    corpora: dict[str, list[str]],
    test_tokens: list[str],
    block_size: int,
    n_blocks: int,
    max_word_len: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, list[tuple]]]:
    test_curve = compute_average_curve(
        word_length_distributions_random_blocks(
            tokens=test_tokens,
            block_size=block_size,
            n_blocks=n_blocks,
            normalize=True,
            seed=seed,
        )
    )

    distances = {}
    contributions = {}
    for author_idx, (author, author_tokens) in enumerate(corpora.items()):
        author_curve = compute_average_curve(
            word_length_distributions_random_blocks(
                tokens=author_tokens,
                block_size=block_size,
                n_blocks=n_blocks,
                normalize=True,
                seed=seed + author_idx + 1,
            )
        )
        aligned, lengths = align_distributions([test_curve, author_curve], max_len=max_word_len)
        test_vector, author_vector = aligned
        distances[author] = jensenshannon(test_vector, author_vector)
        contributions[author] = sorted(
            (
                (length, abs(test_freq - author_freq), author_freq, test_freq)
                for length, test_freq, author_freq in zip(lengths, test_vector, author_vector, strict=True)
            ),
            key=lambda row: row[1],
            reverse=True,
        )
    return distances, contributions


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
    if method == "mendenhall":
        length, diff, ref_freq, test_freq = contribution
        return {
            "method": method,
            "author": test_row["author"],
            "work": test_row["work"],
            "candidate_author": candidate_author,
            "rank": rank,
            "word_length": length,
            distance_label: diff,
            "ref_frequency": ref_freq,
            "test_frequency": test_freq,
        }

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
