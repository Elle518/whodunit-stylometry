"""Data loading and file handling utilities."""

import hashlib
import logging
from pathlib import Path

import pandas as pd


def load_corpus_by_directory(base_path: str | Path, encoding: str = "utf-8") -> dict[str, str]:
    """
    Reads a directory containing subdirectories with `.txt` files and returns a
    dictionary that maps each subdirectory name to the concatenated text of all
    its files.

    Args:
        base_path (str | Path): Path to the base directory (e.g.,
            ``corpus/hand_cleaned``).
        encoding (str, optional): Text encoding used to read the files.
            Defaults to ``"utf-8"``.

    Returns:
        dict[str, str]: Dictionary where keys are subdirectory names and values
        are the concatenated contents of their `.txt` files.
    """

    base_path = Path(base_path)
    corpus: dict[str, str] = {}

    for subdir in sorted(p for p in base_path.iterdir() if p.is_dir()):
        texts = []

        for txt_file in sorted(subdir.glob("*.txt")):
            with txt_file.open(encoding=encoding) as f:
                texts.append(f.read())

        corpus[subdir.name] = "\n\n".join(texts)

    return corpus


def file_md5(path: Path) -> str:
    """Computes the MD5 checksum of a file.

    The file is read in binary mode and processed in chunks to avoid loading
    the entire file into memory.

    Args:
        path: Path to the file to hash.

    Returns:
        The MD5 digest of the file as a hexadecimal string.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_file_inventory(corpus_dir: Path) -> pd.DataFrame:
    """Builds a file inventory DataFrame for a corpus organized by author folders.

    The function scans the immediate subdirectories of ``corpus_dir`` (each assumed
    to represent an author), collects metadata for all ``.txt`` files in each
    author directory, and returns the results as a pandas DataFrame.

    For each text file, the inventory includes author name, book name (derived from
    the filename stem), full path, filename, file size in bytes, and the file MD5
    checksum.

    Args:
        corpus_dir: Path to the root corpus directory. Each immediate subdirectory
            is treated as an author folder.

    Returns:
        A pandas DataFrame with one row per ``.txt`` file and the following columns:
        ``author_norm``, ``title_norm``, ``path``, ``file_name``, ``size_bytes``, and ``md5``.
    """
    rows = []
    for author_dir in sorted([p for p in corpus_dir.iterdir() if p.is_dir()]):
        author = author_dir.name
        for txt_file in sorted(author_dir.glob("*.txt")):
            stat = txt_file.stat()
            rows.append(
                {
                    "author_norm": author,
                    "title_norm": txt_file.stem,
                    "path": str(txt_file),
                    "file_name": txt_file.name,
                    "size_bytes": stat.st_size,
                    "md5": file_md5(txt_file),
                }
            )
    return pd.DataFrame(rows)


def read_book_text(path: Path, encoding: str = "utf-8") -> str:
    """Reads a text file and returns its contents as a string.

    The file is read using the provided text encoding (UTF-8 by default). If a
    decoding error or any other exception occurs, the error is logged and an
    empty string is returned.

    Args:
        path: Path to the text file to read.
        encoding: Text encoding to use when reading the file. Defaults to
            ``"utf-8"``.

    Returns:
        The file contents as a string, or an empty string if an error occurs.
    """
    try:
        text = path.read_text(encoding=encoding)
        return text
    except UnicodeDecodeError as e:
        logging.error(str(e))
    except Exception as e:
        logging.error(str(e))
        return ""


def load_corpus_by_dataframe(
    df: pd.DataFrame,
    encoding: str = "utf-8",
) -> dict[str, str]:
    """
    Reads text files listed in a DataFrame and returns a dictionary mapping each
    author to the concatenated text of all their files.

    The input DataFrame must contain at least these columns:
        - ``author``: author name
        - ``file_path``: path to a `.txt` file

    Args:
        df (pd.DataFrame): DataFrame with columns ``author`` and ``file_path``.
        encoding (str, optional): Text encoding used to read the files.
            Defaults to ``"utf-8"``.

    Returns:
        dict[str, str]: Dictionary where keys are author names and values
        are the concatenated contents of their files.
    """

    corpus: dict[str, str] = {}

    for author, group in df.sort_values(["author", "file_path"]).groupby("author"):
        texts = []

        for file_path in group["file_path"]:
            path = Path(file_path)
            with path.open(encoding=encoding) as f:
                texts.append(f.read())

        corpus[author] = "\n\n".join(texts)

    return corpus
