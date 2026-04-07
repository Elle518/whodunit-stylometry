"""Normalize all texts in `corpus/hand_cleaned` and save them to
`corpus/normalized` preserving the original directory structure.

Usage example:
    > python -m scripts.corpus_normalization
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from whodunit_stylometry.utils.nlp_utils import (
    fix_gutenberg_linebreaks,
    normalize_text_for_tokenization,
)

CORPUS_DIR = Path("corpus")
SRC_DIR = CORPUS_DIR / "hand_cleaned"
DST_DIR = CORPUS_DIR / "normalized"


def main():
    """Normalize all text files in the source corpus directory.

    This function reads every `.txt` file under `SRC_DIR`, applies the text
    normalization pipeline, and writes the normalized output to `DST_DIR`
    while preserving the original directory structure.

    The normalization pipeline includes:
        1. Fixing Gutenberg-specific line breaks.
        2. Normalizing text for tokenization.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    DST_DIR.mkdir(parents=True, exist_ok=True)

    src_files = list(SRC_DIR.rglob("*.txt"))

    for src_file in tqdm(src_files, desc="Normalizing texts", unit="file"):
        rel_path = src_file.relative_to(SRC_DIR)
        dst_file = DST_DIR / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        text = src_file.read_text(encoding="utf-8")
        text = fix_gutenberg_linebreaks(text)
        text = normalize_text_for_tokenization(text)
        dst_file.write_text(text, encoding="utf-8")

    print("Texts normalized successfully!")


if __name__ == "__main__":
    main()
