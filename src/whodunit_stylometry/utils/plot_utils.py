import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from whodunit_stylometry.constants import AUTHORS_ABREV_MAP

sns.set_theme(style="whitegrid")


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

    df["author"] = df["author"].map(AUTHORS_ABREV_MAP)

    counts = df["author"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(8, 6))

    ax = counts.plot(kind="bar")

    plt.title("Número de novelas por autor")
    plt.xlabel("Autor")
    plt.ylabel("Nº de novelas")
    plt.xticks(rotation=0)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_publication_year_distribution(df: pd.DataFrame, year_col: str = "year", save_path: Path | None = None):
    """Plot the distribution of works by publication year.

    This function extracts publication years from the specified DataFrame
    column, counts the number of works published in each year, and includes
    years with zero works within the full observed range. The result is
    displayed as a bar chart.

    Args:
        df: Input DataFrame containing publication year data.
        year_col: Name of the column that stores publication years.
            Defaults to ``"year"``.
        save_path: Optional path where the generated figure will be saved.
            If ``None``, the figure is only displayed.
    """

    years = df[year_col].dropna().astype(int)

    full_range = range(years.min(), years.max() + 1)
    counts = years.value_counts().sort_index().reindex(full_range, fill_value=0)

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index, counts.values, width=0.8)

    plt.title("Año de publicación de las obras del corpus")
    plt.xlabel("Año de publicación")
    plt.ylabel("Número de obras")
    plt.xticks(full_range[::5], rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

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


def plot_metrics_histograms(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: Path | None = None,
):
    """Plot histograms for the given metrics using a dynamic subplot layout.

    The function creates as many subplots as needed based on the number of
    metrics provided. Each metric is plotted as a histogram with a KDE curve.
    Any unused axes in the subplot grid are hidden.

    Args:
        df: Input DataFrame containing the metric columns to plot.
        metrics: List of column names in ``df`` to plot as histograms.
        save_path: Optional path where the generated figure will be saved. If ``None``, the figures are only displayed.
    """

    n_metrics = len(metrics)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        sns.histplot(df[metric].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Histograma de {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Frecuencia")

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_metrics_boxplots_by_author(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: Path | None = None,
):
    """Plot one boxplot per metric grouped by author.

    This function iterates over the given metrics and generates a separate
    boxplot for each one, grouping values by the ``author`` column in the
    input DataFrame. Each plot is displayed individually. If ``save_path`` is
    provided, each figure is also saved to disk using the metric name appended
    to the base file name.

    Args:
        df: Input DataFrame containing an ``author`` column and the metric
            columns to plot.
        metrics: List of metric column names in ``df`` to visualize as
            boxplots.
        save_path: Optional base path where each generated figure will be
            saved. The metric name is appended to the file name before the
            extension. If ``None``, the figures are only displayed.
    """

    for metric in metrics:
        plt.figure(figsize=(14, 6))

        sns.boxplot(data=df, x="author", y=metric)

        plt.title(f"Boxplot por autor: {metric}")
        plt.xlabel("Autor")
        plt.ylabel(metric)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path.with_suffix('')}_{metric}_{save_path.suffix}", dpi=150, bbox_inches="tight")

        plt.show()


def plot_bars_by_author_with_std(
    df: pd.DataFrame,
    metrics: list[str],
    agg: str,
    save_path: Path | None = None,
):
    """Plot one bar chart per metric showing an aggregated value by author.

    This function groups the input DataFrame by the ``author`` column and
    computes summary statistics for each metric, including mean, standard
    deviation, median, and count. For every metric in ``metrics``, it creates
    a bar chart where bar heights correspond to the selected aggregation
    (``"mean"`` or ``"median"``) and error bars represent one standard
    deviation.

    If ``save_path`` is provided, each generated figure is saved to disk using
    the aggregation name and metric name in the output file name.

    Args:
        df: Input DataFrame containing an ``author`` column and the metric
            columns to summarize and plot.
        metrics: List of numeric column names in ``df`` to visualize.
        agg: Aggregation to plot for each metric. Supported values are
            ``"mean"`` and ``"median"``.
        save_path: Optional base path where the generated figures will be
            saved. The aggregation name and metric name are appended to the
            file name before the extension. If ``None``, the figures are only
            displayed.
    """

    agg_labels = {
        "mean": "Media",
        "median": "Mediana",
    }

    for metric in metrics:
        author_stats = df.groupby("author")[metric].agg(["mean", "std", "median", "count"]).reset_index()

        plt.figure(figsize=(14, 6))
        plt.bar(
            author_stats["author"],
            author_stats[agg],
            yerr=author_stats["std"],
            capsize=5,
        )
        plt.title(f"{agg_labels[agg]} de {metric} por autor (con desviación estándar)")
        plt.xlabel("Autor")
        plt.ylabel(f"{agg_labels[agg]} de {metric}")
        plt.tight_layout()

        if save_path:
            output_path = save_path.with_name(f"{save_path.stem}_{agg}_{metric}{save_path.suffix}")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.show()


def plot_metrics_correlations_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: Path | None = None,
):
    """Plot a heatmap of correlations between the specified metrics.

    This function computes the pairwise correlation matrix for the selected
    metric columns in the input DataFrame and visualizes it as a heatmap using
    Seaborn. Correlation coefficients are displayed inside each cell.

    Args:
        df: Input DataFrame containing the metric columns to analyze.
        metrics: List of column names in ``df`` for which the correlation
            matrix will be computed.
        save_path: Optional path where the generated figure will be saved.
            If ``None``, the figure is only displayed.
    """

    corr = df[metrics].corr(numeric_only=True)

    n_metrics = len(corr.columns)

    cell_size = 0.8
    min_size = 6
    max_size = 20
    fig_width = min(max(min_size, n_metrics * cell_size), max_size)
    fig_height = min(max(min_size, n_metrics * cell_size), max_size)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Heatmap de correlaciones entre métricas")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_scatter_list_plotly(
    data: pd.DataFrame,
    scatter_pairs: list[tuple[str, str]],
    hue: str = "author",
    work_col: str = "title",
):
    """Create interactive scatter plots for multiple pairs of variables.

    This function iterates over the variable pairs provided in
    ``scatter_pairs`` and generates one Plotly scatter plot for each pair.
    Points can be colored according to the values in the ``hue`` column, if
    that column exists in the input DataFrame. When available, the value in
    ``work_col`` is displayed as the hover name for each point, together with
    the x and y values.

    Args:
        data: Input DataFrame containing the variables to plot.
        scatter_pairs: List of ``(x_var, y_var)`` tuples specifying which
            variable pairs to visualize.
        hue: Name of the column used to color the points. If the column is not
            present in ``data``, points are displayed without color grouping.
            Defaults to ``"author"``.
        work_col: Name of the column whose values are shown as the hover label
            for each point. If the column is not present in ``data``, no hover
            name is displayed. Defaults to ``"title"``.
    """

    for x_var, y_var in scatter_pairs:
        fig = px.scatter(
            data_frame=data,
            x=x_var,
            y=y_var,
            color=hue if hue in data.columns else None,
            hover_name=work_col if work_col in data.columns else None,
            hover_data={
                hue: True if hue in data.columns else False,
                x_var: True,
                y_var: True,
            },
            opacity=0.75,
            title=f"{y_var} vs {x_var}",
        )

        fig.update_layout(
            width=900,
            height=600,
            legend_title_text=hue,
        )

        fig.show()


def plot_standardized_heatmap_by_author(
    df: pd.DataFrame,
    value_cols: str,
    save_path: Path | None = None,
):
    """Plot a heatmap of author-level metrics standardized across authors.

    This function selects the columns in ``df`` whose names end with
    ``value_cols``, uses ``author`` as the row index, and standardizes each
    selected metric independently across authors using z-score scaling. The
    resulting standardized matrix is displayed as a heatmap, which helps
    compare how each author deviates from the cross-author mean for each
    metric.

    Args:
        df (pd.DataFrame): Input DataFrame containing one row per author, an
            ``author`` column, and metric summary columns such as
            ``avg_sentence_len_mean`` or ``mattr_100_mean``.
        value_cols (str): Column-name suffix used to select the metrics to
            visualize. For example, ``"_mean"`` selects all columns ending in
            ``"_mean"``.
        save_path (str | None, optional): Output path where the generated
            figure will be saved. If ``None``, the figure is not saved.

    Notes:
        The heatmap colors represent standardized values per metric:
        positive values indicate that an author is above the mean for that
        metric, and negative values indicate that the author is below the
        mean. This plot is mainly useful for relative comparison across
        authors rather than for inspecting absolute metric values.
    """

    mean_cols = [c for c in df.columns if c.endswith(value_cols)]
    plot_df = df.set_index("author")[mean_cols].copy()
    scaler = StandardScaler()
    plot_df_z = pd.DataFrame(scaler.fit_transform(plot_df), index=plot_df.index, columns=plot_df.columns)

    plt.figure(figsize=(16, 10))
    sns.heatmap(plot_df_z, cmap="vlag", center=0, linewidths=0.5)
    plt.title("Heatmap de métricas estandarizadas por autor")
    plt.xlabel("Métricas")
    plt.ylabel("Autor")
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_authors_pca(
    pca: PCA,
    pca_df: pd.DataFrame,
    save_path: Path | None = None,
):
    """Plot the first two principal components for author-level stylometric data.

    This function creates a scatter plot of authors projected onto the first
    two principal components of a fitted PCA model. Each point is annotated
    with the corresponding author name, and the axis labels include the
    percentage of variance explained by each component.

    Args:
        pca: A fitted ``sklearn.decomposition.PCA`` object.
        pca_df: A DataFrame containing at least the columns ``"PC1"``,
            ``"PC2"``, and ``"author"``. Each row represents one author in
            the PCA space.
        savepath: Optional path where the generated figure will be saved.
            If ``None``, the figure is not saved. Defaults to ``None``.
    """
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=120)

    for _, row in pca_df.iterrows():
        plt.text(row["PC1"] + 0.03, row["PC2"] + 0.03, row["author"], fontsize=10)

    plt.title("PCA of authors based on stylometric features")
    plt.xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    plt.axhline(0, color="red", linestyle="--", linewidth=0.4)
    plt.axvline(0, color="red", linestyle="--", linewidth=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
