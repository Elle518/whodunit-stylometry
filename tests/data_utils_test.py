"""Tests for the data utility functions."""

from pathlib import Path

import pandas as pd
import pytest

from whodunit_stylometry.utils.data_utils import (
    discover_corpus,
    load_corpus,
    load_corpus_by_dataframe,
    load_test_by_dataframe,
    merge_if_needed,
)


def test_discover_corpus_collects_non_empty_text_files(tmp_path: Path) -> None:
    author_dir = tmp_path / "author_one"
    author_dir.mkdir()
    (author_dir / "story.txt").write_text("One, two three.", encoding="utf-8")
    (author_dir / "empty.txt").write_text("", encoding="utf-8")
    (author_dir / "notes.md").write_text("ignored", encoding="utf-8")

    result = discover_corpus(tmp_path)

    assert result[["author", "work", "filename"]].to_dict("records") == [
        {"author": "author_one", "work": "story", "filename": "story.txt"}
    ]
    assert result.loc[0, "char_count"] == len("One, two three.")
    assert result.loc[0, "word_count"] == 3


def test_load_corpus_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No existe la ruta"):
        load_corpus(tmp_path / "missing")


def test_dataframe_loaders_group_training_text_and_keep_test_works_separate(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("First", encoding="utf-8")
    second.write_text("Second", encoding="utf-8")
    metadata = pd.DataFrame(
        [
            {"author": "writer", "file_path": str(second)},
            {"author": "writer", "file_path": str(first)},
        ]
    )

    assert load_corpus_by_dataframe(metadata) == {"writer": "First\n\nSecond"}
    assert load_test_by_dataframe(metadata) == {
        "writer_first": "First",
        "writer_second": "Second",
    }


def test_merge_if_needed_adds_only_missing_inventory_columns() -> None:
    metadata = pd.DataFrame([{"file_name": "book.txt", "author": "writer"}])
    inventory = pd.DataFrame([{"file_name": "book.txt", "author": "inventory_writer", "size_bytes": 12}])

    result = merge_if_needed(metadata, inventory)

    assert result.to_dict("records") == [{"file_name": "book.txt", "author": "writer", "size_bytes": 12}]
