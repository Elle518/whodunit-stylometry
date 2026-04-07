"""Tokenize normalized texts and save token lists preserving directory structure.

Usage examples:
    > python -m scripts.corpus_tokenization --token-kind alpha --lowercase
    > python -m scripts.corpus_tokenization --token-kind all
    > python -m scripts.corpus_tokenization --token-kind non_alpha --no-lowercase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from whodunit_stylometry.utils.nlp_utils import CustomTokenizer, is_alpha_tokens

CORPUS_DIR = Path("corpus")
SRC_DIR = CORPUS_DIR / "normalized"
DST_ROOT = CORPUS_DIR / "tokenized"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tokenization script.

    Configures and parses the arguments that control which tokens to keep,
    whether tokens should be lowercased, and the input/output directories
    used during tokenization.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Tokenize normalized corpus texts.")
    parser.add_argument(
        "--token-kind",
        choices=("all", "alpha", "non_alpha"),
        default="alpha",
        help="Which tokens to keep.",
    )
    parser.add_argument(
        "--lowercase",
        dest="lowercase",
        action="store_true",
        help="Lowercase tokens after tokenization/filtering.",
    )
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="Preserve original token casing.",
    )
    parser.set_defaults(lowercase=False)

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=SRC_DIR,
        help="Directory with normalized .txt files.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DST_ROOT,
        help="Root directory where tokenized files will be saved.",
    )

    return parser.parse_args()


def filter_tokens(tokens: list[str], token_kind: str) -> list[str]:
    """Filter tokens according to the requested token type.

    Args:
        tokens (list[str]): List of input tokens.
        token_kind (str): Token subset to keep. Must be one of
            {"all", "alpha", "non_alpha"}.

    Returns:
        list[str]: Filtered list of tokens.

    Raises:
        ValueError: If `token_kind` is not supported.
    """
    if token_kind == "all":
        return tokens
    if token_kind == "alpha":
        return is_alpha_tokens(tokens, keep_alpha=True)
    if token_kind == "non_alpha":
        return is_alpha_tokens(tokens, keep_alpha=False)
    raise ValueError(f"Unsupported token kind: {token_kind}")


def maybe_lowercase(tokens: list[str], lowercase: bool) -> list[str]:
    """Convert tokens to lowercase if requested.

    Args:
        tokens (list[str]): List of input tokens.
        lowercase (bool): Whether to lowercase all tokens.

    Returns:
        list[str]: Token list, lowercased if requested.
    """
    if not lowercase:
        return tokens
    return [t.lower() for t in tokens]


def build_output_dir(dst_root: Path, token_kind: str, lowercase: bool) -> Path:
    """Build the output directory path for a tokenization configuration.

    The output path is determined by the token type and casing configuration.

    Args:
        dst_root (Path): Root directory where tokenized files are stored.
        token_kind (str): Token subset to keep.
        lowercase (bool): Whether tokens are lowercased.

    Returns:
        Path: Output directory for the given tokenization configuration.
    """
    case_dir = "lower" if lowercase else "cased"
    return dst_root / token_kind / case_dir


def main():
    """Tokenize normalized corpus texts and save them as JSON files.

    This function reads all `.txt` files under the source directory,
    tokenizes their contents, optionally filters tokens by type, optionally
    lowercases them, and writes the resulting token lists as `.json` files
    while preserving the original directory structure.

    Raises:
        FileNotFoundError: If the source directory does not exist or contains
            no `.txt` files.
    """
    args = parse_args()

    if not args.src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.src_dir}")

    tokenizer = CustomTokenizer()
    src_files = sorted(args.src_dir.rglob("*.txt"))

    if not src_files:
        raise FileNotFoundError(f"No .txt files found in: {args.src_dir}")

    output_base = build_output_dir(args.dst_root, args.token_kind, args.lowercase)
    output_base.mkdir(parents=True, exist_ok=True)

    for src_file in tqdm(src_files, desc="Tokenizing texts", unit="file"):
        rel_path = src_file.relative_to(args.src_dir)
        dst_file = (output_base / rel_path).with_suffix(".json")
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        text = src_file.read_text(encoding="utf-8")
        tokens = tokenizer.tokenize(text)
        tokens = filter_tokens(tokens, args.token_kind)
        tokens = maybe_lowercase(tokens, args.lowercase)

        with open(dst_file, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False)

    print(f"Tokenization completed successfully! Files saved to: {output_base}")


if __name__ == "__main__":
    main()
