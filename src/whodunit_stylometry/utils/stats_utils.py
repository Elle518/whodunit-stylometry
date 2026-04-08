"""Statistical utilities for stylometric analysis."""

import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def add_iqr_outlier_flags(df: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag IQR-based outliers for selected metric columns and summarize thresholds.

    For each metric column, computes the first quartile (Q1), third quartile
    (Q3), and interquartile range (IQR), then marks values below
    ``Q1 - 1.5 * IQR`` as low outliers and values above ``Q3 + 1.5 * IQR`` as
    high outliers. The function adds per-metric outlier flags and direction
    labels to a copy of the input DataFrame, along with row-level aggregate
    outlier indicators.

    Args:
        df: Input DataFrame containing the metric columns to evaluate.
        metric_cols: Column names for which IQR-based outlier detection should
            be performed.

    Returns:
        A tuple containing:
            - A copy of the input DataFrame with additional outlier-related
              columns for each metric, plus ``has_any_outlier`` and
              ``outlier_metrics``.
            - A summary DataFrame indexed by metric name with the computed
              quartiles, IQR, bounds, and outlier counts.

    Notes:
        For each metric column ``col``, the returned DataFrame includes:
            - ``{col}_is_outlier``: Boolean flag indicating whether the value is
              an outlier.
            - ``{col}_outlier_side``: ``"low"``, ``"high"``, or ``None``.
        The ``outlier_metrics`` column contains a list of metric names and
        outlier directions for each row.
    """

    df = df.copy()
    summary = {}

    for col in metric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        is_low = df[col] < lower
        is_high = df[col] > upper
        is_outlier = is_low | is_high

        df[f"{col}_is_outlier"] = is_outlier
        df[f"{col}_outlier_side"] = np.where(is_low, "low", np.where(is_high, "high", None))

        summary[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": int(is_outlier.sum()),
            "n_low_outliers": int(is_low.sum()),
            "n_high_outliers": int(is_high.sum()),
        }

    flag_cols = [f"{col}_is_outlier" for col in metric_cols]
    df["has_any_outlier"] = df[flag_cols].any(axis=1)

    df["outlier_metrics"] = df.apply(
        lambda row: [f"{col} ({row[f'{col}_outlier_side']})" for col in metric_cols if row[f"{col}_is_outlier"]],
        axis=1,
    )

    summary_df = pd.DataFrame(summary).T
    return df, summary_df


def word_length_distributions_random_blocks(
    tokens: list[str],
    block_size: int = 5000,
    n_blocks: int = 20,
    normalize: bool = True,
    seed: int | None = None,
) -> list[dict[int, float]]:
    """Compute word-length distributions for randomly sampled contiguous blocks.

    Args:
        tokens (List[str]): Tokenized corpus of a single author.
        block_size (int): Number of tokens per block.
        n_blocks (int): Number of random blocks to sample.
        normalize (bool): If True, return relative frequencies.
        seed (int | None): Random seed for reproducibility.

    Returns:
        List[Dict[int, float]]: One distribution per block.
    """

    if seed is not None:
        random.seed(seed)

    max_start = len(tokens) - block_size
    if max_start <= 0:
        raise ValueError("Corpus too small for the given block_size")

    distributions = []

    for _ in range(n_blocks):
        start = random.randint(0, max_start)
        block = tokens[start : start + block_size]

        lengths = [len(t) for t in block]
        counts = Counter(lengths)

        if normalize:
            total = sum(counts.values())
            counts = {k: v / total for k, v in counts.items()}

        distributions.append(counts)

    return distributions


def align_distributions(
    dists: list[dict[int, float]],
    max_len: int = 20,
) -> tuple[list[list[float]], list[int]]:
    """Align sparse length distributions to a common dense index range.

    Converts each input distribution, represented as a mapping from length to
    probability/value, into a dense list covering all lengths from 1 to
    ``max_len`` inclusive. Missing lengths are filled with ``0.0``.

    Args:
        dists: A list of distributions where each distribution maps an integer
            length to a float value.
        max_len: The maximum length to include in the aligned output. Must be
            greater than or equal to 1.

    Returns:
        A tuple containing:
            - A list of aligned distributions, where each distribution is a
              list of floats for lengths from 1 to ``max_len``.
            - The list of lengths used as the common index range.
    """

    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    lengths = list(range(1, max_len + 1))
    aligned = []

    for d in dists:
        row = [d.get(length, 0.0) for length in lengths]
        aligned.append(row)

    return aligned, lengths


def distance_matrix(distributions: list[list[float]]) -> np.ndarray:
    """Compute the pairwise Jensen-Shannon distance matrix.

    Builds a symmetric distance matrix for the input distributions using the
    Jensen-Shannon distance between each pair of distributions. The diagonal is
    zero because the distance from a distribution to itself is zero.

    Args:
        distributions: A list of dense probability distributions, where each
            distribution is represented as a list of floats.

    Returns:
        A square NumPy array of shape ``(n, n)``, where ``n`` is the number of
        input distributions. Each entry ``[i, j]`` contains the Jensen-Shannon
        distance between distributions ``i`` and ``j``.
    """

    n = len(distributions)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(distributions[i], distributions[j])
            D[i, j] = D[j, i] = d

    return D


def compute_average_curve(distributions: list[dict[int, float]]) -> dict[int, float]:
    """Compute the average characteristic curve across multiple block distributions.

    Aggregates a collection of per-block length distributions into a single
    average distribution by computing the mean frequency for each observed
    length across all input distributions.

    Args:
        distributions: A list of distributions, where each distribution maps an
            integer length to its frequency or normalized value for a block.

    Returns:
        A dictionary mapping each observed length to its average frequency
        across the input distributions.
    """

    total_counts = defaultdict(float)
    n = len(distributions)

    for dist in distributions:
        for length, freq in dist.items():
            total_counts[length] += freq

    return {length: total / n for length, total in total_counts.items()}


def classify_test_works_by_average_curve(
    df_test: pd.DataFrame,
    average_curves: dict[str, dict[int, float]],
    token_col: str = "tokens",
    title_col: str = "title",
    true_author_col: str = "author",
    block_size: int = 5000,
    n_blocks: int = 20,
    max_len: int = 20,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Classify test works by comparing their average curves to author profiles.

    For each test work, this function samples random token blocks, computes the
    word-length distribution for each block, averages those distributions into a
    single characteristic curve, and compares that curve against the precomputed
    average curve of each author using Jensen-Shannon distance. The author with
    the smallest distance is selected as the predicted author.

    Args:
        df_test: DataFrame containing the test works to classify.
        average_curves: Mapping from author name to that author's average
            word-length distribution.
        token_col: Name of the column containing the tokenized text for each
            work.
        title_col: Name of the column containing the work title.
        true_author_col: Name of the column containing the ground-truth author.
        block_size: Number of tokens to include in each sampled block.
        n_blocks: Number of random blocks to sample from each test work.
        max_len: Maximum word length used when aligning distributions before
            distance computation.
        seed: Random seed used for block sampling. Use ``None`` for
            non-deterministic sampling.

    Returns:
        A DataFrame with one row per test work, including:
            - row identifier
            - title
            - true author
            - predicted author
            - minimum Jensen-Shannon distance
            - one distance column per candidate author

        If a work is too short to sample at least one full block, the returned
        row contains ``pred_author=None``, ``min_distance=np.nan``, and a
        status message explaining why it was skipped.
    """

    results = []

    for idx, row in df_test.iterrows():
        tokens = row[token_col]

        # If the work is too short to sample even one block, we cannot compute
        if len(tokens) <= block_size:
            results.append(
                {
                    "row_id": idx,
                    "title": row.get(title_col, idx),
                    "true_author": row.get(true_author_col, None),
                    "pred_author": None,
                    "min_distance": np.nan,
                    "status": f"Texto demasiado pequeño para block_size={block_size}",
                }
            )
            continue

        # 1) Block distributions of the test work
        test_distributions = word_length_distributions_random_blocks(
            tokens=tokens,
            block_size=block_size,
            n_blocks=n_blocks,
            normalize=True,
            seed=seed,
        )

        # 2) Average curve of the test work
        test_curve = compute_average_curve(test_distributions)

        # 3) Compare with each author
        distances = {}
        for author, author_curve in average_curves.items():
            aligned, _ = align_distributions([test_curve, author_curve], max_len=max_len)
            v_test, v_author = aligned
            distances[author] = jensenshannon(v_test, v_author)

        # 4) Nearest author
        pred_author = min(distances, key=distances.get)
        min_distance = distances[pred_author]

        result = {
            "row_id": idx,
            "title": row.get(title_col, idx),
            "true_author": row.get(true_author_col, None),
            "pred_author": pred_author,
            "min_distance": min_distance,
        }

        # We keep all distances for analysis, not just the minimum
        for author, dist in distances.items():
            result[f"dist_{author}"] = dist

        results.append(result)

    return pd.DataFrame(results)


def build_global_vocab(
    corpora: dict[str, list[str]],
    vocab_size: int = 500,
    min_freq: int = 1,
) -> list[str]:
    """Build a fixed global vocabulary from the training corpora.

    The vocabulary is created from the combined token frequencies of all
    reference corpora. Only the most frequent tokens that satisfy the minimum
    frequency threshold are kept.

    Args:
        corpora: Mapping from author name to list of tokens.
        vocab_size: Maximum number of tokens to keep.
        min_freq: Minimum global frequency required for a token to be included.

    Returns:
        A list of tokens defining the global vocabulary.
    """

    total_counts = Counter()

    for tokens in corpora.values():
        total_counts.update(tokens)

    vocab = [token for token, freq in total_counts.most_common() if freq >= min_freq][:vocab_size]
    return vocab


def kilgariff_chi2(
    tokens_a: list[str],
    tokens_b: list[str],
    vocab: list[str],
):
    """Compute Kilgariff's chi-square distance using a fixed global vocabulary.

    Args:
        tokens_a: Token sequence for the reference corpus.
        tokens_b: Token sequence for the test work.
        vocab: Fixed vocabulary used in all comparisons.

    Returns:
        A tuple with:
            - chi2_total: Total chi-square distance.
            - contributions: List of per-token contributions sorted from
              highest to lowest. Each item has the form:
              (token, chi_token, obs_a, exp_a, obs_b, exp_b)
    """

    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)

    total_a = sum(freq_a.values())
    total_b = sum(freq_b.values())
    total = total_a + total_b

    if total == 0:
        raise ValueError("Both token sequences are empty.")

    chi2_total = 0.0
    contributions = []

    for token in vocab:
        obs_a = freq_a.get(token, 0)
        obs_b = freq_b.get(token, 0)
        combined_f = obs_a + obs_b

        # If the token does not appear in either text, it contributes nothing.
        if combined_f == 0:
            continue

        # Expected frequencies based on relative sample sizes
        exp_a = combined_f * (total_a / total)
        exp_b = combined_f * (total_b / total)

        chi_a = ((obs_a - exp_a) ** 2 / exp_a) if exp_a > 0 else 0.0
        chi_b = ((obs_b - exp_b) ** 2 / exp_b) if exp_b > 0 else 0.0

        chi_token = chi_a + chi_b
        chi2_total += chi_token

        contributions.append((token, chi_token, obs_a, exp_a, obs_b, exp_b))

    contributions.sort(key=lambda x: x[1], reverse=True)

    return chi2_total, contributions


def classify_test_works_kilgariff(
    corpora: dict[str, list[str]],
    test_works: dict[str, list[str]],
    vocab_size: int = 500,
    min_freq: int = 1,
):
    """Classify test works by minimum Kilgariff chi-square distance.

    A fixed global vocabulary is built from the training corpora and then used
    in every corpus-vs-test comparison.

    Args:
        corpora: Mapping from author name to training tokens.
        test_works: Mapping from work title/name to test tokens.
        vocab_size: Maximum size of the global vocabulary.
        min_freq: Minimum global frequency threshold for vocabulary inclusion.

    Returns:
        A dictionary of classification results:
            results[work] = {
                "prediccion": predicted_author,
                "ranking": [(author, chi2), ...]
            }
    """

    global_vocab = build_global_vocab(corpora, vocab_size=vocab_size, min_freq=min_freq)

    results = {}

    for work, work_tokens in test_works.items():
        ranking = []

        for author, tokens_corpus in corpora.items():
            chi2, _ = kilgariff_chi2(tokens_corpus, work_tokens, vocab=global_vocab)
            ranking.append((author, chi2))

        ranking.sort(key=lambda x: x[1])  # smaller chi2 = more similar
        results[work] = {
            "prediccion": ranking[0][0],
            "ranking": ranking,
        }

    return results


def get_pairwise_contributions(
    corpora: dict[str, list[str]],
    test_works: dict[str, list[str]],
    vocab_size: int = 500,
    min_freq: int = 1,
):
    """Compute detailed chi-square contributions for each work-author pair.

    This is useful if you want not only the predicted author, but also the
    token-level contribution breakdown for inspection.

    Args:
        corpora: Mapping from author name to training tokens.
        test_works: Mapping from work title/name to test tokens.
        vocab_size: Maximum size of the global vocabulary.
        min_freq: Minimum global frequency threshold.

    Returns:
        A nested dictionary:
            details[work][author] = {
                "chi2": float,
                "contributions": [...]
            }
    """

    global_vocab = build_global_vocab(corpora, vocab_size=vocab_size, min_freq=min_freq)

    details = {}

    for work, work_tokens in test_works.items():
        details[work] = {}

        for author, tokens_corpus in corpora.items():
            chi2, contributions = kilgariff_chi2(tokens_corpus, work_tokens, vocab=global_vocab)
            details[work][author] = {
                "chi2": chi2,
                "contributions": contributions,
            }

    return details


def relative_frequencies(tokens: list[str], vocab: list[str]) -> dict[str, float]:
    """Compute relative frequencies for a fixed vocabulary.

    Args:
        tokens: Token sequence of a text or corpus.
        vocab: Fixed vocabulary.

    Returns:
        A dictionary mapping each token in `vocab` to its relative frequency
        in the input sequence.
    """

    counts = Counter(tokens)
    total = len(tokens)

    if total == 0:
        raise ValueError("Token sequence is empty.")

    return {token: counts.get(token, 0) / total for token in vocab}


def compute_feature_stats(
    corpora: dict[str, list[str]],
    vocab: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """Compute mean and standard deviation for each token across corpora.

    Frequencies are computed corpus by corpus over the fixed global vocabulary.

    Args:
        corpora: Mapping from author name to training tokens.
        vocab: Fixed vocabulary.

    Returns:
        A tuple containing:
            - means: Mean relative frequency of each token across corpora.
            - stds: Standard deviation of each token across corpora.
            - author_freqs: Relative frequencies for each author corpus.
    """

    author_freqs = {author: relative_frequencies(tokens, vocab) for author, tokens in corpora.items()}

    means = {}
    stds = {}

    n_authors = len(author_freqs)
    if n_authors < 2:
        raise ValueError("At least two author corpora are needed to compute Burrows's Delta.")

    for token in vocab:
        values = [author_freqs[author][token] for author in author_freqs]
        mean = sum(values) / len(values)

        # Sample standard deviation
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(variance)

        means[token] = mean
        stds[token] = std

    return means, stds, author_freqs


def z_scores(
    freqs: dict[str, float],
    means: dict[str, float],
    stds: dict[str, float],
    vocab: list[str],
) -> dict[str, float]:
    """Convert relative frequencies into z-scores.

    Tokens whose standard deviation is zero are skipped, since they do not help
    discriminate between authors.

    Args:
        freqs: Relative frequencies for one text/corpus.
        means: Mean relative frequency for each token.
        stds: Standard deviation for each token.
        vocab: Fixed vocabulary.

    Returns:
        A dictionary of z-scores for tokens with non-zero standard deviation.
    """

    z = {}

    for token in vocab:
        std = stds[token]
        if std == 0:
            continue
        z[token] = (freqs[token] - means[token]) / std

    return z


def burrows_delta_from_zscores(
    z_ref: dict[str, float],
    z_test: dict[str, float],
) -> tuple[float, list[tuple[str, float, float, float]]]:
    """Compute Burrows's Delta between two z-score profiles.

    Args:
        z_ref: Z-scores for the reference corpus.
        z_test: Z-scores for the test work.

    Returns:
        A tuple containing:
            - delta: Mean absolute difference between z-scores.
            - contributions: List of token-level contributions sorted from
              highest to lowest. Each tuple has the form:
              (token, abs_diff, z_ref, z_test)
    """

    common_tokens = list(set(z_ref.keys()) & set(z_test.keys()))

    if not common_tokens:
        raise ValueError("No comparable tokens with non-zero standard deviation were found.")

    contributions = []

    for token in common_tokens:
        diff = abs(z_ref[token] - z_test[token])
        contributions.append((token, diff, z_ref[token], z_test[token]))

    contributions.sort(key=lambda x: x[1], reverse=True)
    delta = sum(row[1] for row in contributions) / len(contributions)

    return delta, contributions


def burrows_delta(
    reference_tokens: list[str],
    test_tokens: list[str],
    vocab: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> tuple[float, list[tuple[str, float, float, float]]]:
    """Compute Burrows's Delta between a reference corpus and a test work.

    Args:
        reference_tokens: Tokens from the reference corpus.
        test_tokens: Tokens from the test work.
        vocab: Fixed global vocabulary.
        means: Mean relative frequency for each token.
        stds: Standard deviation for each token.

    Returns:
        A tuple containing:
            - delta: Burrows's Delta distance.
            - contributions: List of token-level contributions sorted from
              highest to lowest. Each tuple contains:
              (token, abs_diff, z_ref, z_test)
    """

    ref_freqs = relative_frequencies(reference_tokens, vocab)
    test_freqs = relative_frequencies(test_tokens, vocab)

    z_ref = z_scores(ref_freqs, means, stds, vocab)
    z_test = z_scores(test_freqs, means, stds, vocab)

    return burrows_delta_from_zscores(z_ref, z_test)


def classify_test_works_burrows(
    corpora: dict[str, list[str]],
    test_works: dict[str, list[str]],
    vocab_size: int = 500,
    min_freq: int = 1,
):
    """Classify test works using Burrows's Delta.

    A fixed global vocabulary is built from the training corpora. Relative
    frequencies are standardized using corpus-level means and standard
    deviations, and each test work is assigned to the author with the lowest
    Delta distance.

    Args:
        corpora: Mapping from author name to training tokens.
        test_works: Mapping from work title/name to test tokens.
        vocab_size: Maximum size of the global vocabulary.
        min_freq: Minimum global frequency threshold for vocabulary inclusion.

    Returns:
        A dictionary of classification results:
            results[work] = {
                "prediccion": predicted_author,
                "ranking": [(author, delta), ...]
            }
    """

    vocab = build_global_vocab(corpora, vocab_size=vocab_size, min_freq=min_freq)
    means, stds, _ = compute_feature_stats(corpora, vocab)

    results = {}

    for work, work_tokens in test_works.items():
        ranking = []

        for author, author_tokens in corpora.items():
            delta, _ = burrows_delta(
                reference_tokens=author_tokens,
                test_tokens=work_tokens,
                vocab=vocab,
                means=means,
                stds=stds,
            )
            ranking.append((author, delta))

        ranking.sort(key=lambda x: x[1])  # smaller Delta = more similar
        results[work] = {
            "prediccion": ranking[0][0],
            "ranking": ranking,
        }

    return results


def get_pairwise_burrows_contributions(
    corpora: dict[str, list[str]],
    test_works: dict[str, list[str]],
    vocab_size: int = 500,
    min_freq: int = 1,
):
    """Compute detailed Burrows's Delta contributions for each work-author pair.

    Args:
        corpora: Mapping from author name to training tokens.
        test_works: Mapping from work title/name to test tokens.
        vocab_size: Maximum size of the global vocabulary.
        min_freq: Minimum global frequency threshold.

    Returns:
        A nested dictionary:
            details[work][author] = {
                "delta": float,
                "contributions": [...]
            }
    """

    vocab = build_global_vocab(corpora, vocab_size=vocab_size, min_freq=min_freq)
    means, stds, _ = compute_feature_stats(corpora, vocab)

    details = {}

    for work, work_tokens in test_works.items():
        details[work] = {}

        for author, author_tokens in corpora.items():
            delta, contributions = burrows_delta(
                reference_tokens=author_tokens,
                test_tokens=work_tokens,
                vocab=vocab,
                means=means,
                stds=stds,
            )
            details[work][author] = {
                "delta": delta,
                "contributions": contributions,
            }

    return details


def drop_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
    method: str = "pearson",
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Identify highly correlated columns and return which to keep and drop.

    Computes the absolute pairwise correlation matrix for the input DataFrame,
    masks its lower triangle and diagonal, and selects columns whose correlation
    with any previous column exceeds the given threshold.

    Args:
        df: Input DataFrame whose columns are evaluated for pairwise correlation.
        threshold: Correlation threshold above which a column is marked for
            removal. Defaults to 0.85.
        method: Correlation method passed to ``DataFrame.corr()``. Defaults to
            ``"pearson"``.

    Returns:
        A tuple containing:
            - kept: Column names not marked for removal.
            - to_drop: Column names whose upper-triangle correlation with at
              least one other column exceeds ``threshold``.
            - upper: Upper-triangular masked absolute correlation matrix.

    Notes:
        The selection of dropped columns depends on column order because only
        the upper triangle of the correlation matrix is considered.
    """
    corr = df.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [col for col in df.columns if col not in to_drop]

    return kept, to_drop, upper
