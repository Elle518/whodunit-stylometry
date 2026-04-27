import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, StandardScaler

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


def plot_confussion_matrix(
    cm: np.ndarray, labels: list[str], model_name: str, features: str, save_path: Path | None = None
):
    """Plot a confusion matrix as a heatmap with annotations.

    This function takes a confusion matrix and corresponding class labels, and
    visualizes it as a heatmap using Seaborn. Each cell is annotated with the
    integer count from the confusion matrix. The x-axis and y-axis are labeled
    with the provided class names.

    Args:
        cm: A 2D NumPy array representing the confusion matrix counts.
        labels: A list of class names corresponding to the rows and columns of
            the confusion matrix.
        model_name: Name of the model used for prediction.
        save_path: Optional path where the generated figure will be saved.
            If ``None``, the figure is only displayed.
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Autor predicho")
    plt.ylabel("Autor real")
    plt.title(f"Matriz de confusión de {model_name} con {features}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_mfw_performance_robustness(
    summary_by_topn: pd.DataFrame,
    best_mean_score: float,
    threshold_score: float,
    selected_top_n_mfw: int,
):
    """Plot model performance robustness as a function of ``TOP_N_MFW``.

    The plot shows the mean ``test_f1_macro`` for each ``top_n_mfw`` value,
    a shaded band representing plus/minus one standard deviation, and reference
    lines for the best mean score, the acceptance threshold, and the selected
    minimum ``TOP_N_MFW`` value.

    Args:
        summary_by_topn: DataFrame containing at least the columns
            ``"top_n_mfw"``, ``"mean_test_f1_macro"``, and
            ``"std_test_f1_macro"``.
        best_mean_score: Best mean ``test_f1_macro`` value to display as a
            horizontal reference line.
        threshold_score: Threshold score to display as a horizontal reference
            line.
        selected_top_n_mfw: Selected ``TOP_N_MFW`` value to display as a
            vertical reference line.

    Notes:
        Missing values in ``std_test_f1_macro`` are replaced with ``0`` before
        plotting the shaded variability band.
        This function produces a plot as a side effect by calling
        ``matplotlib.pyplot.show()``.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        summary_by_topn["top_n_mfw"],
        summary_by_topn["mean_test_f1_macro"],
        marker="o",
        label="Media test_f1_macro",
    )

    std_values = summary_by_topn["std_test_f1_macro"].fillna(0)

    plt.fill_between(
        summary_by_topn["top_n_mfw"],
        summary_by_topn["mean_test_f1_macro"] - std_values,
        summary_by_topn["mean_test_f1_macro"] + std_values,
        alpha=0.2,
        label="±1 std",
    )

    plt.axhline(
        best_mean_score,
        linestyle="--",
        label=f"Mejor media = {best_mean_score:.4f}",
    )
    plt.axhline(
        threshold_score,
        linestyle=":",
        label=f"Umbral = {threshold_score:.4f}",
    )
    plt.axvline(
        selected_top_n_mfw,
        linestyle="--",
        label=f"TOP_N_MFW mínimo = {selected_top_n_mfw}",
    )

    plt.xlabel("TOP_N_MFW")
    plt.ylabel("test_f1_macro")
    plt.title("Robustez del rendimiento según TOP_N_MFW")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_clusters_2d(
    X_scaled: np.ndarray,
    y_true: Sequence[object],
    clusters: Sequence[int],
    title_prefix: str = "",
    method: Literal["pca", "tsne"] = "pca",
    seed: int = 0,
    figsize: tuple[float, float] = (14, 6),
) -> None:
    """Plot a 2D projection of samples colored by true labels and clusters.

    This function reduces the input feature matrix to two dimensions using
    either PCA or t-SNE, then displays two side-by-side scatter plots:
    one colored by the true labels and the other colored by the predicted
    cluster assignments.

    Args:
        X_scaled: Scaled feature matrix with shape ``(n_samples, n_features)``.
        y_true: Sequence of ground-truth labels, one per sample.
        clusters: Sequence of cluster assignments, one per sample.
        title_prefix: Text prefix added to both subplot titles. Defaults to
            an empty string.
        method: Dimensionality reduction method to use. Must be ``"pca"``
            or ``"tsne"``. Defaults to ``"pca"``.
        seed: Random seed passed to the dimensionality reduction method.
            Defaults to ``0``.
        figsize: Figure size passed to ``matplotlib.pyplot.subplots``.
            Defaults to ``(14, 6)``.
    """
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords = reducer.fit_transform(X_scaled)
        xlab, ylab = "PC1", "PC2"
    elif method == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        coords = reducer.fit_transform(X_scaled)
        xlab, ylab = "t-SNE 1", "t-SNE 2"
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Color by true author
    authors = pd.Series(y_true).astype("category")
    author_codes = authors.cat.codes
    scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=author_codes, s=50)
    axes[0].set_title(f"{title_prefix} - Coloreado por autor real")
    axes[0].set_xlabel(xlab)
    axes[0].set_ylabel(ylab)

    # Author legend
    handles1, _ = scatter1.legend_elements()
    axes[0].legend(
        handles1,
        authors.cat.categories,
        title="Author",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Color by cluster
    scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=clusters, s=50)
    axes[1].set_title(f"{title_prefix} - Coloreado por cluster")
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(ylab)

    handles2, labels2 = scatter2.legend_elements()
    axes[1].legend(
        handles2,
        labels2,
        title="Cluster",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


def plot_dendrogram_colored_labels(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "author_norm",
    file_col: str = "file_name",
    method: str = "ward",
    metric: str = "euclidean",
    figsize: tuple[float, float] = (8, 18),
    truncate_mode: str | None = None,
    p: int = 30,
    leaf_rotation: float = 0,
    leaf_font_size: float = 10,
    save_path: Path | None = None,
) -> NDArray:
    """Plot a dendrogram with leaf labels colored by category and return linkage data.

    The function standardizes the selected feature columns, computes a hierarchical
    clustering linkage matrix, plots a dendrogram using file names as leaf labels,
    and colors each visible leaf label according to the category defined in
    ``label_col``.

    Args:
        data: Input DataFrame containing feature columns and metadata columns.
        feature_cols: Names of numeric columns used to compute clustering.
        label_col: Column whose categorical values are used to color leaf labels.
        file_col: Column used as the displayed label for each dendrogram leaf.
        method: Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
            When set to ``"ward"``, the ``metric`` argument is not passed.
        metric: Distance metric passed to ``linkage`` for non-``"ward"`` methods.
        figsize: Figure size passed to Matplotlib as ``(width, height)``.
        truncate_mode: Dendrogram truncation mode passed to ``dendrogram``.
        p: Dendrogram truncation parameter passed to ``dendrogram``.
        leaf_rotation: Rotation angle for leaf labels.
        leaf_font_size: Font size for leaf labels.
        savepath: Optional path where the generated figure will be saved.
            If ``None``, the figure is only displayed.

    Returns:
        NDArray: The hierarchical clustering linkage matrix returned by ``linkage``.
    """
    X = data[feature_cols].copy()
    y = data[label_col].astype("category")
    labels = [name.replace(".txt", "")[:30] for name in data[file_col].tolist()]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "ward":
        Z = linkage(X_scaled, method=method)
    else:
        Z = linkage(X_scaled, method=method, metric=metric)

    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=labels,
        orientation="left",
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        truncate_mode=truncate_mode,
        p=p,
    )

    authors = y.cat.categories
    cmap = plt.cm.get_cmap("tab10", len(authors))
    color_map = {author: cmap(i) for i, author in enumerate(authors)}

    ax = plt.gca()
    label_to_author = dict(zip(labels, data[label_col]))

    ticklabels = ax.get_ymajorticklabels()
    for lbl in ticklabels:
        text = lbl.get_text()
        author = label_to_author.get(text)
        if author in color_map:
            lbl.set_color(color_map[author])

    handles = [plt.Line2D([0], [0], color=color_map[a], lw=4, label=a) for a in authors]
    plt.legend(handles=handles, title="Autor", loc="upper left")

    plt.title(f"Dendrograma ({method})")
    plt.xlabel("Distancia")
    plt.ylabel("Obras")
    plt.subplots_adjust(right=0.72)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return Z
