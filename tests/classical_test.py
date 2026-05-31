"""Tests for the classical stylometry analysis functions."""

import pandas as pd
import pytest

from whodunit_stylometry.analysis.classical import (
    filter_feature_tokens,
    prepare_feature_tokens,
    tokens_by_author,
    top_tokens_by_author,
)


def test_tokens_by_author_flattens_each_authors_works() -> None:
    works = pd.DataFrame(
        [
            {"author": "first", "tokens": ["one", "two"]},
            {"author": "first", "tokens": ["three"]},
            {"author": "second", "tokens": ["other"]},
        ]
    )

    assert tokens_by_author(works) == {
        "first": ["one", "two", "three"],
        "second": ["other"],
    }


def test_filter_feature_tokens_can_keep_only_function_words() -> None:
    tokens = ["and", "mystery", "the"]

    assert filter_feature_tokens(tokens, use_function_words=True) == ["and", "the"]
    assert filter_feature_tokens(tokens, use_function_words=False) is tokens


def test_prepare_feature_tokens_does_not_modify_input_dataframe() -> None:
    works = pd.DataFrame([{"author": "writer", "tokens": ["and", "mystery"]}])

    result = prepare_feature_tokens(works, use_function_words=True)

    assert list(works.columns) == ["author", "tokens"]
    assert result.loc[0, "feature_tokens"] == ["and"]
    assert result.loc[0, "feature_token_count"] == 1


def test_top_tokens_by_author_reports_rank_and_relative_frequency() -> None:
    works = pd.DataFrame([{"author": "writer", "tokens": ["and", "and", "the"]}])

    result = top_tokens_by_author(works, top_n=1)

    assert result.to_dict("records") == [
        {
            "author": "writer",
            "rank": 1,
            "token": "and",
            "count": 2,
            "relative_frequency": pytest.approx(2 / 3),
        }
    ]
