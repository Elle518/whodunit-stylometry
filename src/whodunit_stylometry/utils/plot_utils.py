from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_word_length_distributions(author_name: str, distributions: list[dict[int, float]], block_size: int):
    """Plot relative word-length distributions for multiple text blocks of an author.

    This function creates a line plot where each curve represents the relative
    frequency distribution of word lengths for one text block. The x-axis
    corresponds to word length, and the y-axis corresponds to relative
    frequency. All distributions are displayed on the same figure for visual
    comparison.

    Args:
        author_name: Name of the author whose text blocks are being plotted.
        distributions: A list of dictionaries, where each dictionary maps a
            word length (int) to its relative frequency (float) for a single
            text block.
        block_size: Number of tokens in each text block used to compute the
            distributions.
    """

    plt.figure(figsize=(10, 6))

    for i, dist in enumerate(distributions):
        x = sorted(dist.keys())
        y = [dist[k] for k in x]
        plt.plot(x, y, alpha=0.5)

    plt.xlabel("Longitud de palabra")
    plt.ylabel("Frecuencia relativa")
    plt.title(f"Curvas características por bloques de {block_size} tokens - {author_name}")
    plt.grid(True)
    plt.show()


def plot_novels_per_author(df: pd.DataFrame, save_path: Path | None = None):
    """Plot the number of novels per author as a bar chart.

    The function counts how many rows belong to each value in the ``author``
    column, sorts authors by descending count, and displays the result in a
    labeled bar chart. Each bar is annotated with its corresponding count.
    If ``savepath`` is provided, the figure is also saved to disk before
    being shown.

    Args:
        df: Input DataFrame containing at least an ``author`` column, where
            each row represents a novel.
        savepath: Optional path where the generated figure will be saved.
            If ``None``, the figure is only displayed.
    """

    counts = df["author"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 7))

    ax = counts.plot(kind="bar")

    plt.title("Número de novelas por autor")
    plt.xlabel("Autor")
    plt.ylabel("Nº de novelas")
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_zipf_curve_by_author(
    tokens_by_author: dict[str, list],
    save_path: Path | None = None,
    max_rank: int | None = None,
):
    """Plot Zipf curves for each author based on token frequency distributions.

    This function builds a token frequency counter for each author, computes
    relative frequencies sorted in descending order, and plots the resulting
    rank-frequency distributions on a log-log scale. Optionally, the curve for
    each author can be truncated to a maximum rank. If ``save_path`` is
    provided, the figure is saved before being displayed.

    Args:
        tokens_by_author: A dictionary mapping each author's name to a list of
            tokens associated with that author.
        save_path: Optional path where the generated figure will be saved.
            If ``None``, the figure is only displayed.
        max_rank: Optional maximum rank to include in the plot. If provided,
            only the top ``max_rank`` most frequent tokens per author are
            plotted.
    """

    authors_counters = defaultdict(Counter, {author: Counter(tokens) for author, tokens in tokens_by_author.items()})

    plt.figure(figsize=(9, 6))

    for author, counter in sorted(authors_counters.items()):
        freqs = sorted(counter.values(), reverse=True)
        total = sum(freqs)
        rel_freqs = [f / total for f in freqs]

        if max_rank is not None:
            rel_freqs = rel_freqs[:max_rank]

        ranks = np.arange(1, len(rel_freqs) + 1)
        plt.loglog(ranks, rel_freqs, label=author)

    plt.title("Curvas de Zipf por autor")
    plt.xlabel("Rango")
    plt.ylabel("Frecuencia relativa")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
