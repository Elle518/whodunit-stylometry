"""Tests for the NLP utility functions."""

import math

import pytest

from whodunit_stylometry.utils.nlp_utils import (
    CustomTokenizer,
    compute_novel_metrics,
    is_alpha_tokens,
    lexical_metrics,
    mattr,
    normalize_text_for_tokenization,
    normalize_whitespace,
    punctuation_metrics,
    split_paragraphs,
)


def test_custom_tokenizer_preserves_contractions_and_abbreviations() -> None:
    tokenizer = CustomTokenizer()

    assert tokenizer.tokenize("don't e.g. rock'n'roll") == ["don't", "e.g.", "rock'n'roll"]


def test_normalize_text_for_tokenization_standardizes_typography() -> None:
    text = "  “Wait”… -- don’t_go~  "

    assert normalize_text_for_tokenization(text) == '"Wait" ... — don\'tgo'


def test_normalize_whitespace_and_split_paragraphs() -> None:
    text = " first \r\n  line \r\n\r\n\r\n second\tline "

    assert normalize_whitespace(text) == "first \nline \n\nsecond line"
    assert split_paragraphs(normalize_whitespace(text)) == ["first \nline", "second line"]


def test_is_alpha_tokens_can_keep_or_remove_word_like_tokens() -> None:
    tokens = ["hello", "don't", "a.m.", "123", "_", "."]

    assert is_alpha_tokens(tokens) == ["hello", "don't", "a.m."]
    assert is_alpha_tokens(tokens, keep_alpha=False) == ["123", "_", "."]


def test_lexical_metrics_are_case_insensitive() -> None:
    metrics = lexical_metrics(["Case", "case", "unique"])

    assert metrics["n_types"] == 2
    assert metrics["ttr"] == pytest.approx(2 / 3)
    assert metrics["hapax_count"] == 1
    assert metrics["hapax_ratio"] == pytest.approx(1 / 2)


def test_mattr_uses_sliding_windows_and_rejects_short_input() -> None:
    assert mattr(["a", "b", "a", "c"], window=3) == pytest.approx(5 / 6)
    assert math.isnan(mattr(["a", "b"], window=3))


def test_punctuation_metrics_counts_ellipsis_separately_from_periods() -> None:
    metrics = punctuation_metrics('"Well..." Really?', n_tokens=2)

    assert metrics["quote_count"] == 2
    assert metrics["ellipsis_count"] == 1
    assert metrics["period_count"] == 0
    assert metrics["question_count"] == 1
    assert metrics["punctuation_ratio"] == pytest.approx(2)


def test_compute_novel_metrics_accepts_precomputed_tokens() -> None:
    metrics = compute_novel_metrics(
        "Short sentence.\n\nAnd another one!",
        stopword_set={"and"},
        tokens_all=["Short", "sentence", ".", "And", "another", "one", "!"],
        tokens_alpha=["Short", "sentence", "And", "another", "one"],
    )

    assert metrics["n_tokens_all"] == 7
    assert metrics["n_tokens_alpha"] == 5
    assert metrics["n_sentences"] == 2
    assert metrics["n_paragraphs"] == 2
    assert metrics["stopword_ratio"] == pytest.approx(1 / 5)
