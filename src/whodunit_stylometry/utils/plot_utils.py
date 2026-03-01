import matplotlib.pyplot as plt


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
