"""Classical stylometry pipeline for interactive use."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from whodunit_stylometry.constants import STOPWORDS
from whodunit_stylometry.utils.nlp_utils import CustomTokenizer, tokenize_text
from whodunit_stylometry.utils.stats_utils import (
    align_distributions,
    build_global_vocab,
    burrows_delta,
    compute_average_curve,
    compute_feature_stats,
    distance_matrix,
    kilgariff_chi2,
    word_length_distributions_random_blocks,
    z_scores,
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


def tokens_by_author(df: pd.DataFrame) -> dict[str, list[str]]:
    """Aggregate tokens by author."""

    return {author: [token for tokens in sub["tokens"] for token in tokens] for author, sub in df.groupby("author")}


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


def kilgariff_reference_tables(
    df: pd.DataFrame,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Kilgariff vocabulary and author-frequency tables for display."""

    feature_df = prepare_feature_tokens(df, use_function_words=config.use_function_words)
    corpora = {
        author: [token for tokens in sub["feature_tokens"] for token in tokens]
        for author, sub in feature_df.groupby("author")
    }
    corpora = {author: tokens for author, tokens in corpora.items() if tokens}
    vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)

    global_counts = Counter(token for tokens in corpora.values() for token in tokens)
    vocab_rows = [
        {"rank": rank, "token": token, "global_count": global_counts[token]}
        for rank, token in enumerate(vocab, start=1)
    ]

    frequency_rows = []
    for author, tokens in corpora.items():
        counts = Counter(tokens)
        total = sum(counts.values())
        for token in vocab:
            count = counts[token]
            frequency_rows.append(
                {
                    "author": author,
                    "token": token,
                    "count": count,
                    "relative_frequency": count / total if total else np.nan,
                }
            )

    return pd.DataFrame(vocab_rows), pd.DataFrame(frequency_rows)


def burrows_reference_tables(
    df: pd.DataFrame,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Burrows vocabulary and author z-score tables for display."""

    feature_df = prepare_feature_tokens(df, use_function_words=config.use_function_words)
    corpora = {
        author: [token for tokens in sub["feature_tokens"] for token in tokens]
        for author, sub in feature_df.groupby("author")
    }
    corpora = {author: tokens for author, tokens in corpora.items() if tokens}
    vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)

    if not vocab:
        return pd.DataFrame(), pd.DataFrame()

    means, stds, author_freqs = compute_feature_stats(corpora, vocab)
    global_counts = Counter(token for tokens in corpora.values() for token in tokens)

    vocab_rows = [
        {
            "rank": rank,
            "token": token,
            "global_count": global_counts[token],
            "mean_frequency": means[token],
            "std_frequency": stds[token],
        }
        for rank, token in enumerate(vocab, start=1)
    ]

    zscore_rows = []
    for author, freqs in author_freqs.items():
        author_zscores = z_scores(freqs, means, stds, vocab)
        for token in vocab:
            zscore_rows.append(
                {
                    "author": author,
                    "token": token,
                    "relative_frequency": freqs[token],
                    "z_score": author_zscores.get(token, np.nan),
                }
            )

    return pd.DataFrame(vocab_rows), pd.DataFrame(zscore_rows)


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


def classify_text_with_kilgariff(
    text: str,
    reference_df: pd.DataFrame,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Attribute one text using Kilgariff's chi-square distance."""

    tokenizer = CustomTokenizer()
    test_tokens = tokenize_text(text, tokenizer=tokenizer, lowercase=config.lowercase)
    test_feature_tokens = filter_feature_tokens(test_tokens, use_function_words=config.use_function_words)

    reference_df = prepare_feature_tokens(reference_df, use_function_words=config.use_function_words)
    corpora = {
        author: [token for tokens in sub["feature_tokens"] for token in tokens]
        for author, sub in reference_df.groupby("author")
    }
    corpora = {author: tokens for author, tokens in corpora.items() if tokens}

    if len(corpora) < 2:
        raise ValueError("Se necesitan al menos dos autores con tokens suficientes para comparar.")
    if not test_feature_tokens:
        raise ValueError("La obra seleccionada no contiene tokens suficientes con la configuración actual.")

    vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)
    if not vocab:
        raise ValueError("El vocabulario global ha quedado vacío con la configuración actual.")

    distances, contributions = _kilgariff_distances(corpora, test_feature_tokens, vocab)
    ranking = sorted(distances.items(), key=lambda item: item[1])
    second_distance = ranking[1][1] if len(ranking) > 1 else np.nan

    distance_rows = []
    for rank, (author, chi2) in enumerate(ranking, start=1):
        distance_rows.append(
            {
                "rank": rank,
                "author": author,
                "chi2": chi2,
                "margin_to_best": chi2 - ranking[0][1],
                "margin_to_second": (
                    second_distance - ranking[0][1] if rank == 1 and pd.notna(second_distance) else np.nan
                ),
            }
        )

    contribution_rows = []
    test_row = {"author": None, "work": "Obra seleccionada"}
    for author, author_contributions in contributions.items():
        for rank, contribution in enumerate(author_contributions, start=1):
            contribution_rows.append(_format_contribution("kilgariff", test_row, author, rank, contribution, "chi2"))

    stats = {
        "token_count": len(test_tokens),
        "feature_token_count": len(test_feature_tokens),
        "vocab_size": len(vocab),
    }

    return pd.DataFrame(distance_rows), pd.DataFrame(contribution_rows), stats


def classify_text_with_burrows(
    text: str,
    reference_df: pd.DataFrame,
    config: ClassicalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Attribute one text using Burrows's Delta distance."""

    tokenizer = CustomTokenizer()
    test_tokens = tokenize_text(text, tokenizer=tokenizer, lowercase=config.lowercase)
    test_feature_tokens = filter_feature_tokens(test_tokens, use_function_words=config.use_function_words)

    reference_df = prepare_feature_tokens(reference_df, use_function_words=config.use_function_words)
    corpora = {
        author: [token for tokens in sub["feature_tokens"] for token in tokens]
        for author, sub in reference_df.groupby("author")
    }
    corpora = {author: tokens for author, tokens in corpora.items() if tokens}

    if len(corpora) < 2:
        raise ValueError("Se necesitan al menos dos autores con tokens suficientes para comparar.")
    if not test_feature_tokens:
        raise ValueError("La obra seleccionada no contiene tokens suficientes con la configuración actual.")

    vocab = build_global_vocab(corpora, vocab_size=config.vocab_size, min_freq=config.min_freq)
    if not vocab:
        raise ValueError("El vocabulario global ha quedado vacío con la configuración actual.")

    distances, contributions = _burrows_distances(corpora, test_feature_tokens, vocab)
    ranking = sorted(distances.items(), key=lambda item: item[1])
    second_distance = ranking[1][1] if len(ranking) > 1 else np.nan

    distance_rows = []
    for rank, (author, delta) in enumerate(ranking, start=1):
        distance_rows.append(
            {
                "rank": rank,
                "author": author,
                "delta": delta,
                "margin_to_best": delta - ranking[0][1],
                "margin_to_second": (
                    second_distance - ranking[0][1] if rank == 1 and pd.notna(second_distance) else np.nan
                ),
            }
        )

    contribution_rows = []
    test_row = {"author": None, "work": "Obra seleccionada"}
    for author, author_contributions in contributions.items():
        for rank, contribution in enumerate(author_contributions, start=1):
            contribution_rows.append(_format_contribution("burrows", test_row, author, rank, contribution, "delta"))

    stats = {
        "token_count": len(test_tokens),
        "feature_token_count": len(test_feature_tokens),
        "vocab_size": len(vocab),
    }

    return pd.DataFrame(distance_rows), pd.DataFrame(contribution_rows), stats


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
