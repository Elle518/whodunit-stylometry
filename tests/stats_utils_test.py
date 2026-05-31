"""Tests for the statistical utility functions."""

import numpy as np
import pytest

from whodunit_stylometry.utils.stats_utils import (
    align_distributions,
    build_global_vocab,
    classify_test_works_burrows,
    compute_average_curve,
    distance_matrix,
    kilgariff_chi2,
    relative_frequencies,
    word_length_distributions_random_blocks,
)


def test_random_block_distributions_are_normalized_and_reproducible() -> None:
    tokens = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]

    first = word_length_distributions_random_blocks(tokens, block_size=3, n_blocks=2, seed=7)
    second = word_length_distributions_random_blocks(tokens, block_size=3, n_blocks=2, seed=7)

    assert first == second
    assert all(sum(distribution.values()) == pytest.approx(1) for distribution in first)


def test_random_block_distributions_reject_too_small_corpus() -> None:
    with pytest.raises(ValueError, match="Corpus too small"):
        word_length_distributions_random_blocks(["a", "b"], block_size=2)


def test_align_distributions_fills_missing_lengths() -> None:
    aligned, lengths = align_distributions([{1: 0.25, 3: 0.75}, {2: 1.0}], max_len=3)

    assert lengths == [1, 2, 3]
    assert aligned == [[0.25, 0.0, 0.75], [0.0, 1.0, 0.0]]


def test_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    result = distance_matrix([[1.0, 0.0], [0.0, 1.0]])

    assert np.diag(result).tolist() == [0.0, 0.0]
    assert result[0, 1] == pytest.approx(result[1, 0])
    assert result[0, 1] > 0


def test_compute_average_curve_includes_missing_values_as_zero() -> None:
    result = compute_average_curve([{1: 1.0}, {2: 1.0}])

    assert result == {1: 0.5, 2: 0.5}


def test_build_global_vocab_applies_frequency_threshold_and_limit() -> None:
    corpora = {"first": ["a", "a", "b"], "second": ["b", "c"]}

    assert build_global_vocab(corpora, vocab_size=1, min_freq=2) == ["a"]


def test_kilgariff_chi2_is_zero_for_identical_profiles() -> None:
    distance, contributions = kilgariff_chi2(["a", "a", "b"], ["a", "a", "b"], ["a", "b"])

    assert distance == pytest.approx(0)
    assert [token for token, *_ in contributions] == ["a", "b"]


def test_relative_frequencies_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="Token sequence is empty"):
        relative_frequencies([], ["a"])


def test_burrows_delta_prefers_matching_author_profile() -> None:
    corpora = {
        "mostly_a": ["a", "a", "a", "b"],
        "mostly_b": ["a", "b", "b", "b"],
    }

    result = classify_test_works_burrows(
        corpora,
        {"unknown": ["a", "a", "a", "b"]},
        vocab_size=2,
    )

    assert result["unknown"]["prediccion"] == "mostly_a"
    assert result["unknown"]["ranking"][0] == ("mostly_a", pytest.approx(0))
