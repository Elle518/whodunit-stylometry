"""Unsupervised clustering workflows for interactive authorship exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    confusion_matrix,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, RobustScaler

from whodunit_stylometry.analysis.supervised import (
    FEATURE_SETS,
    _add_mfw_features,
    _build_mfw_vocab,
    _stylometric_feature_cols,
    build_work_features,
)

UNSUPERVISED_MODEL_NAMES = {
    "K-means": "kmeans",
    "Agglomerative": "agglomerative",
    "Gaussian Mixture": "gmm",
}

HIERARCHICAL_METHOD_NAMES = {
    "Ward": "ward",
    "Complete": "complete",
    "Average": "average",
    "Single": "single",
}


@dataclass(frozen=True)
class UnsupervisedConfig:
    """Configuration for unsupervised clustering experiments."""

    feature_set: str = "mfw"
    top_n_mfw: int = 50
    seed: int = 42
    selected_models: tuple[str, ...] = ("kmeans", "agglomerative", "gmm")
    n_clusters: int | None = None


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for hierarchical clustering experiments."""

    feature_set: str = "mfw"
    top_n_mfw: int = 50
    selected_methods: tuple[str, ...] = ("ward", "complete", "average", "single")
    dendrogram_method: str = "ward"
    n_clusters: int | None = None


def prepare_unsupervised_dataset(df: pd.DataFrame, config: UnsupervisedConfig) -> dict[str, Any]:
    """Builds a feature dataset for unsupervised clustering.

    Creates work-level features from the input dataframe, normalizes selected
    metadata column names, and selects the feature columns requested by the
    unsupervised configuration. Depending on ``config.feature_set``, the returned
    feature set may include stylometric features, most-frequent-word features,
    or both.

    Args:
        df: Input dataframe containing the corpus records expected by
            ``build_work_features``.
        config: Unsupervised dataset configuration. The ``feature_set`` attribute
            must be one of ``"stylometric"``, ``"mfw"``, or ``"combined"``. The
            ``top_n_mfw`` attribute is used when most-frequent-word features are
            requested.

    Returns:
        A dictionary with the prepared dataframe and feature metadata:
            - ``data``: The prepared dataframe.
            - ``feature_cols``: Names of the selected feature columns.
            - ``stylometric_cols``: Names of the available stylometric columns.
            - ``mfw_vocab``: Most-frequent-word vocabulary, or an empty list when
              MFW features are not requested.
            - ``mfw_cols``: Names of the generated MFW feature columns, or an
              empty list when MFW features are not requested.
    """

    data = build_work_features(df).rename(columns={"author": "author_norm", "filename": "file_name"})
    stylometric_cols = _stylometric_feature_cols(
        data.rename(columns={"author_norm": "author", "file_name": "filename"})
    )

    mfw_vocab: list[str] = []
    mfw_cols: list[str] = []
    if config.feature_set in {"mfw", "combined"}:
        mfw_vocab = _build_mfw_vocab(data, config.top_n_mfw)
        data = _add_mfw_features(data, mfw_vocab)
        mfw_cols = [f"fw_{word}" for word in mfw_vocab]

    if config.feature_set == "stylometric":
        feature_cols = stylometric_cols
    elif config.feature_set == "mfw":
        feature_cols = mfw_cols
    elif config.feature_set == "combined":
        feature_cols = stylometric_cols + mfw_cols
    else:
        raise ValueError("feature_set must be 'stylometric', 'mfw' or 'combined'")

    return {
        "data": data,
        "feature_cols": feature_cols,
        "stylometric_cols": stylometric_cols,
        "mfw_vocab": mfw_vocab,
        "mfw_cols": mfw_cols,
    }


def run_unsupervised_experiment(df: pd.DataFrame, config: UnsupervisedConfig) -> dict[str, Any]:
    """Runs clustering models and evaluates the best result against author labels.

    Prepares an unsupervised feature dataset, runs the clustering models selected
    in the configuration, chooses the first model in the clustering results table
    as the best model, and computes error-analysis outputs for that model's
    assignments.

    Args:
        df: Input dataframe containing the corpus records expected by
            ``prepare_unsupervised_dataset``.
        config: Unsupervised experiment configuration. Uses ``selected_models``,
            ``n_clusters``, and ``seed`` for clustering, and passes the full
            configuration to ``prepare_unsupervised_dataset``.

    Returns:
        A dictionary combining the prepared dataset, clustering results, and
        evaluation outputs. Includes:
            - ``best_model_name``: Name of the first model in the clustering
              results table.
            - ``best_eval``: Evaluation dataframe for the best model assignments.
            - ``best_cluster_author_table``: Cluster-by-author summary for the
              best model.
            - ``errors_by_author``: Error summary grouped by true author, with
              the index reset.
            - ``confusion_pairs``: Confusion-pair summary for the best model.
    """

    dataset = prepare_unsupervised_dataset(df, config)
    cluster_result = _run_selected_clustering(
        data=dataset["data"],
        feature_cols=dataset["feature_cols"],
        selected_models=config.selected_models,
        n_clusters=config.n_clusters,
        seed=config.seed,
    )
    best_model_name = str(cluster_result["results"].iloc[0]["model"])
    best_assignments = cluster_result["assignments"][best_model_name]
    best_eval = evaluate_cluster_errors(best_assignments, label_col="author_norm", file_col="file_name")

    result = {
        **dataset,
        **cluster_result,
        "best_model_name": best_model_name,
        "best_eval": best_eval,
        "best_cluster_author_table": cluster_author_table(best_assignments, label_col="author_norm"),
        "errors_by_author": errors_by_true_author(best_eval, label_col="author_norm").reset_index(),
        "confusion_pairs": confusion_pairs(best_eval, label_col="author_norm"),
    }
    return result


def run_unsupervised_mfw_robustness(
    df: pd.DataFrame,
    top_n_values: list[int],
    seeds: list[int],
    selected_models: tuple[str, ...],
) -> dict[str, Any]:
    """Runs a robustness sweep for MFW clustering experiments.

    Evaluates most-frequent-word clustering performance across vocabulary sizes,
    random seeds, and selected clustering models. Each combination is run as a
    separate unsupervised experiment using only MFW features. Results are
    aggregated by vocabulary size and model, then ranked by mean ARI, mean NMI,
    mean silhouette score, and vocabulary size.

    Args:
        df: Input dataframe containing the corpus records expected by
            ``run_unsupervised_experiment``.
        top_n_values: MFW vocabulary sizes to evaluate.
        seeds: Random seeds to evaluate for each vocabulary size and model.
        selected_models: Clustering model names to evaluate. Each model is run
            independently as a single selected model in ``UnsupervisedConfig``.

    Returns:
        A dictionary containing sweep results and the selected robustness choice:
            - ``sweep_df``: Per-run metrics for every evaluated combination.
            - ``summary_df``: Aggregated metrics grouped by ``top_n_mfw`` and
              ``model``.
            - ``decision_df``: Copy of ``summary_df`` with a boolean
              ``selected`` column marking the chosen row.
            - ``selected_top_n_mfw``: Vocabulary size from the top-ranked
              summary row.
            - ``selected_model_name``: Model name from the top-ranked summary
              row.
            - ``best_mean_ARI``: Mean ARI for the selected row.
            - ``best_mean_NMI``: Mean NMI for the selected row.
            - ``best_mean_silhouette``: Mean silhouette score for the selected
              row.
    """

    rows = []
    for top_n in top_n_values:
        for seed in seeds:
            for model_name in selected_models:
                config = UnsupervisedConfig(
                    feature_set="mfw",
                    top_n_mfw=top_n,
                    seed=seed,
                    selected_models=(model_name,),
                )
                result = run_unsupervised_experiment(df, config)
                row = result["results"].iloc[0].to_dict()
                row.update(
                    {
                        "top_n_mfw": top_n,
                        "seed": seed,
                        "n_features": len(result["feature_cols"]),
                    }
                )
                rows.append(row)

    sweep_df = pd.DataFrame(rows)
    summary_df = (
        sweep_df.groupby(["top_n_mfw", "model"], as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_ARI=("ARI", "mean"),
            std_ARI=("ARI", "std"),
            min_ARI=("ARI", "min"),
            mean_NMI=("NMI", "mean"),
            std_NMI=("NMI", "std"),
            mean_silhouette=("silhouette", "mean"),
            std_silhouette=("silhouette", "std"),
            mean_homogeneity=("homogeneity", "mean"),
            mean_completeness=("completeness", "mean"),
            mean_v_measure=("v_measure", "mean"),
        )
        .sort_values(["mean_ARI", "mean_NMI", "mean_silhouette", "top_n_mfw"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    best_choice = summary_df.iloc[0]
    decision_df = summary_df.copy()
    decision_df["selected"] = (decision_df["top_n_mfw"] == best_choice["top_n_mfw"]) & (
        decision_df["model"] == best_choice["model"]
    )

    return {
        "sweep_df": sweep_df,
        "summary_df": summary_df,
        "decision_df": decision_df,
        "selected_top_n_mfw": int(best_choice["top_n_mfw"]),
        "selected_model_name": str(best_choice["model"]),
        "best_mean_ARI": float(best_choice["mean_ARI"]),
        "best_mean_NMI": float(best_choice["mean_NMI"]),
        "best_mean_silhouette": float(best_choice["mean_silhouette"]),
    }


def run_hierarchical_experiment(df: pd.DataFrame, config: HierarchicalConfig) -> dict[str, Any]:
    """Evaluates hierarchical clustering methods and prepares dendrogram data.

    Builds the selected unsupervised feature dataset, scales the feature matrix,
    evaluates the configured hierarchical clustering methods against normalized
    author labels, and stores linkage matrices for both selected methods and the
    configured dendrogram method. The best method is chosen as the first row
    after sorting selected methods by ARI, NMI, and silhouette score in
    descending order.

    Args:
        df: Input dataframe containing corpus records expected by
            ``prepare_unsupervised_dataset``.
        config: Hierarchical clustering configuration. Uses ``feature_set`` and
            ``top_n_mfw`` for dataset preparation, ``selected_methods`` for
            evaluation, ``dendrogram_method`` for dendrogram linkage generation,
            and ``n_clusters`` when provided. If ``n_clusters`` is falsey, the
            number of unique normalized authors is used.

    Returns:
        A dictionary combining the prepared dataset, hierarchical clustering
        outputs, and evaluation artifacts. Includes:
            - ``results``: Metrics for selected hierarchical methods.
            - ``assignments``: Mapping from method name to cluster assignment
              dataframe.
            - ``linkages``: Mapping from method name to linkage matrix.
            - ``X_scaled``: Scaled feature matrix used for clustering.
            - ``label_encoder``: Fitted label encoder for author labels.
            - ``best_method``: Top-ranked selected method.
            - ``best_eval``: Error evaluation for the best method.
            - ``best_cluster_author_table``: Cluster-by-author summary for the
              best method.
            - ``errors_by_author``: Error summary grouped by true author, with
              the index reset.
            - ``confusion_pairs``: Confusion-pair summary for the best method.
            - ``dendrogram_method``: Configured dendrogram linkage method.
            - ``n_clusters``: Number of clusters used for flat assignments.
    """

    if not config.selected_methods:
        raise ValueError("Select at least one hierarchical clustering method.")

    dataset = prepare_unsupervised_dataset(
        df,
        UnsupervisedConfig(feature_set=config.feature_set, top_n_mfw=config.top_n_mfw),
    )
    data = dataset["data"]
    feature_cols = dataset["feature_cols"]
    X = data[feature_cols].copy()
    y = data["author_norm"].copy()
    n_clusters = int(config.n_clusters or y.nunique())

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    rows: list[dict[str, Any]] = []
    assignments: dict[str, pd.DataFrame] = {}
    linkages: dict[str, np.ndarray] = {}
    methods_to_run = tuple(dict.fromkeys((*config.selected_methods, config.dendrogram_method)))
    for method in methods_to_run:
        Z = _hierarchical_linkage(X_scaled, method)
        linkages[method] = Z
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        rows.append(
            {
                "method": method,
                "metric": "euclidean",
                "n_clusters": n_clusters,
                "silhouette": _safe_silhouette(X_scaled, clusters),
                "ARI": adjusted_rand_score(y_encoded, clusters),
                "NMI": normalized_mutual_info_score(y_encoded, clusters),
                "homogeneity": homogeneity_score(y_encoded, clusters),
                "completeness": completeness_score(y_encoded, clusters),
                "v_measure": v_measure_score(y_encoded, clusters),
            }
        )
        assignments[method] = pd.DataFrame(
            {
                "file_name": data["file_name"].values,
                "work": data["work"].values,
                "author_norm": y.values,
                "cluster": clusters,
            }
        )

    results_df = (
        pd.DataFrame(rows)
        .query("method in @config.selected_methods")
        .sort_values(["ARI", "NMI", "silhouette"], ascending=False)
        .reset_index(drop=True)
    )
    best_method = str(results_df.iloc[0]["method"])
    best_eval = evaluate_cluster_errors(assignments[best_method], label_col="author_norm", file_col="file_name")

    return {
        **dataset,
        "results": results_df,
        "assignments": assignments,
        "linkages": linkages,
        "X_scaled": X_scaled,
        "label_encoder": label_encoder,
        "best_method": best_method,
        "best_eval": best_eval,
        "best_cluster_author_table": cluster_author_table(assignments[best_method], label_col="author_norm"),
        "errors_by_author": errors_by_true_author(best_eval, label_col="author_norm").reset_index(),
        "confusion_pairs": confusion_pairs(best_eval, label_col="author_norm"),
        "dendrogram_method": config.dendrogram_method,
        "n_clusters": n_clusters,
    }


def build_dendrogram_figure(
    linkage_matrix: np.ndarray,
    assignments_df: pd.DataFrame,
    method_label: str,
) -> go.Figure:
    """Builds a Plotly dendrogram from a SciPy linkage matrix.

    Creates a left-oriented dendrogram using a precomputed SciPy linkage matrix
    and overlays leaf markers colored by author. Leaf labels are built from the
    ``author_norm`` and ``work`` columns in ``assignments_df``. Hover metadata is
    taken from the ``author_norm``, ``work``, ``file_name``, and ``cluster``
    columns.

    Args:
        linkage_matrix: SciPy linkage matrix describing the hierarchical
            clustering tree.
        assignments_df: Dataframe containing one row per clustered item. Must
            align with the observation order used to build ``linkage_matrix``.
            Expected columns are ``author_norm``, ``work``, ``file_name``, and
            ``cluster``.
        method_label: Display label for the hierarchical method, used in the
            figure title.

    Returns:
        A Plotly figure containing dendrogram line traces and author-colored leaf
        marker traces.
    """

    leaf_labels = [f"{row.author_norm} | {row.work}" for row in assignments_df.itertuples(index=False)]
    dendro = dendrogram(linkage_matrix, labels=leaf_labels, orientation="left", no_plot=True)

    fig = go.Figure()
    for xs, ys, color in zip(dendro["icoord"], dendro["dcoord"], dendro["color_list"], strict=False):
        line_color = _plotly_dendrogram_color(color)
        fig.add_trace(
            go.Scatter(
                x=ys,
                y=xs,
                mode="lines",
                line={"color": line_color, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    leaf_x = [5 + 10 * i for i in range(len(dendro["ivl"]))]
    leaf_meta = assignments_df.iloc[dendro["leaves"]].copy()
    leaf_meta["label"] = dendro["ivl"]
    leaf_meta["y"] = leaf_x
    leaf_meta["cluster"] = leaf_meta["cluster"].astype(str)
    author_colors = _author_color_map(leaf_meta["author_norm"])
    for author, author_df in leaf_meta.groupby("author_norm", sort=True):
        fig.add_trace(
            go.Scatter(
                x=[0] * len(author_df),
                y=author_df["y"],
                mode="markers",
                marker={"size": 9, "color": author_colors[str(author)]},
                customdata=author_df[["author_norm", "work", "file_name", "cluster"]],
                hovertemplate=(
                    "Autor: %{customdata[0]}<br>"
                    "Obra: %{customdata[1]}<br>"
                    "Archivo: %{customdata[2]}<br>"
                    "Cluster: %{customdata[3]}<extra></extra>"
                ),
                name=str(author),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"Dendrograma jerárquico ({method_label})",
        height=max(760, 18 * len(dendro["ivl"])),
        xaxis={"title": "Distancia"},
        yaxis={
            "tickmode": "array",
            "tickvals": leaf_x,
            "ticktext": dendro["ivl"],
            "automargin": True,
        },
        margin={"l": 320, "r": 30, "t": 70, "b": 50},
        legend={"title": "Autor"},
    )
    return fig


def build_cluster_projection(
    X_scaled: np.ndarray,
    assignments: pd.DataFrame,
    method_name: str,
    seed: int,
) -> pd.DataFrame:
    """Projects clustered works to two dimensions with t-SNE.

    Computes a two-dimensional t-SNE embedding from a scaled feature matrix and
    appends the resulting coordinates to a copy of the cluster assignments
    dataframe. The cluster labels are converted to strings for display, and the
    clustering method name is stored in a ``model`` column.

    Args:
        X_scaled: Scaled feature matrix with one row per clustered work.
        assignments: Cluster assignment dataframe aligned row-by-row with
            ``X_scaled``. Must contain a ``cluster`` column.
        method_name: Name of the clustering method or model used to create the
            assignments.
        seed: Random seed passed to t-SNE for reproducible projection.

    Returns:
        A copy of ``assignments`` with added ``x``, ``y``, and ``model`` columns.
        The existing ``cluster`` column is converted to string dtype.
    """

    perplexity = max(2, min(30, (len(assignments) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X_scaled)
    projection_df = assignments.copy()
    projection_df["x"] = coords[:, 0]
    projection_df["y"] = coords[:, 1]
    projection_df["cluster"] = projection_df["cluster"].astype(str)
    projection_df["model"] = method_name
    return projection_df


def evaluate_cluster_errors(
    assignments_df: pd.DataFrame,
    label_col: str = "author_norm",
    cluster_col: str = "cluster",
    file_col: str = "file_name",
) -> dict[str, Any]:
    """Maps clusters to majority labels and summarizes induced errors.

    Copies the assignments dataframe, maps each cluster to its majority true
    label, and compares that mapped label with the original label column. The
    function also builds a contingency table, confusion matrix, and dataframe of
    incorrectly mapped rows.

    Args:
        assignments_df: Dataframe containing cluster assignments and true labels.
            Must include ``label_col``, ``cluster_col``, ``file_col``, and
            ``work`` columns.
        label_col: Name of the column containing the true labels.
        cluster_col: Name of the column containing cluster assignments.
        file_col: Name of the column containing file identifiers.

    Returns:
        A dictionary containing:
            - ``data_with_predictions``: Copy of ``assignments_df`` with
              ``pred_author_from_cluster`` and ``correct`` columns added.
            - ``contingency_table``: Crosstab of clusters by true labels.
            - ``cluster_to_author``: Mapping from each cluster to its majority
              true label.
            - ``accuracy_after_mapping``: Mean correctness after majority-label
              mapping.
            - ``confusion_matrix``: Confusion matrix as a dataframe indexed and
              columned by sorted true labels.
            - ``errors_df``: Incorrectly mapped rows sorted by true label,
              predicted label, and file identifier.
    """

    df = assignments_df.copy()
    contingency = pd.crosstab(df[cluster_col], df[label_col])
    cluster_to_author = contingency.idxmax(axis=1).to_dict()
    df["pred_author_from_cluster"] = df[cluster_col].map(cluster_to_author)
    df["correct"] = df["pred_author_from_cluster"] == df[label_col]

    labels = sorted(df[label_col].unique())
    cm = confusion_matrix(df[label_col], df["pred_author_from_cluster"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    errors_df = df.loc[
        ~df["correct"],
        [file_col, "work", label_col, cluster_col, "pred_author_from_cluster"],
    ].copy()

    return {
        "data_with_predictions": df,
        "contingency_table": contingency,
        "cluster_to_author": cluster_to_author,
        "accuracy_after_mapping": df["correct"].mean(),
        "confusion_matrix": cm_df,
        "errors_df": errors_df.sort_values([label_col, "pred_author_from_cluster", file_col]),
    }


def cluster_author_table(assignments_df: pd.DataFrame, label_col: str = "author_norm") -> pd.DataFrame:
    """Builds a cluster-by-label contingency table.

    Counts the number of rows for each combination of cluster assignment and
    true label.

    Args:
        assignments_df: Dataframe containing cluster assignments and true labels.
            Must include a ``cluster`` column and the column named by
            ``label_col``.
        label_col: Name of the column containing the true labels.

    Returns:
        A dataframe whose rows are cluster values, whose columns are label
        values, and whose cells contain counts.
    """

    return pd.crosstab(assignments_df["cluster"], assignments_df[label_col])


def errors_by_true_author(eval_dict: dict[str, Any], label_col: str = "author_norm") -> pd.DataFrame:
    """Summarizes mapped-cluster errors by true label.

    Groups the prediction results by the true-label column and computes the
    total number of works, number of correctly mapped works, number of errors,
    and error rate for each label. Results are sorted by descending error rate.

    Args:
        eval_dict: Evaluation dictionary containing a
            ``data_with_predictions`` dataframe, such as the output of
            ``evaluate_cluster_errors``. The dataframe must contain ``label_col``
            and ``correct`` columns.
        label_col: Name of the column containing the true labels.

    Returns:
        A dataframe indexed by true label with the following columns:
            ``total_works``, ``correct``, ``errors``, and ``error_rate``.
    """

    df = eval_dict["data_with_predictions"]
    return (
        df.groupby(label_col)["correct"]
        .agg(total_works="count", correct="sum")
        .assign(errors=lambda x: x["total_works"] - x["correct"])
        .assign(error_rate=lambda x: x["errors"] / x["total_works"])
        .sort_values("error_rate", ascending=False)
    )


def confusion_pairs(eval_dict: dict[str, Any], label_col: str = "author_norm") -> pd.DataFrame:
    """Counts true-label versus mapped-label pairs among clustering errors.

    Reads the error rows from an evaluation dictionary and counts how often each
    true label is confused with each mapped cluster label. If there are no
    errors, an empty dataframe with the expected output columns is returned.

    Args:
        eval_dict: Evaluation dictionary containing an ``errors_df`` dataframe,
            such as the output of ``evaluate_cluster_errors``. The dataframe must
            contain ``label_col`` and ``pred_author_from_cluster`` columns.
        label_col: Name of the column containing the true labels.

    Returns:
        A dataframe with ``label_col``, ``pred_author_from_cluster``, and
        ``n_errors`` columns, sorted by descending error count. Returns an empty
        dataframe with the same columns when no errors are present.
    """

    errors = eval_dict["errors_df"]
    if errors.empty:
        return pd.DataFrame(columns=[label_col, "pred_author_from_cluster", "n_errors"])
    return (
        errors.groupby([label_col, "pred_author_from_cluster"])
        .size()
        .reset_index(name="n_errors")
        .sort_values("n_errors", ascending=False)
    )


def _run_selected_clustering(
    data: pd.DataFrame,
    feature_cols: list[str],
    selected_models: tuple[str, ...],
    n_clusters: int | None,
    seed: int,
) -> dict[str, Any]:
    """Runs selected clustering models and evaluates them against author labels.

    Scales the selected feature columns, encodes normalized author labels, fits
    each requested clustering model, and computes clustering quality metrics.
    Supported model names are ``"kmeans"``, ``"agglomerative"``, and ``"gmm"``.
    Results are sorted by ARI, NMI, and silhouette score in descending order.

    Args:
        data: Dataframe containing feature columns and metadata columns. Must
            include all columns in ``feature_cols`` plus ``author_norm``,
            ``file_name``, and ``work``.
        feature_cols: Names of the feature columns to use for clustering.
        selected_models: Model names to run. Each value must be one of
            ``"kmeans"``, ``"agglomerative"``, or ``"gmm"``.
        n_clusters: Number of clusters to fit. If ``None`` or otherwise falsey,
            the number of unique normalized authors is used.
        seed: Random seed used by stochastic models.

    Returns:
        A dictionary containing:
            - ``results``: Dataframe with one metric row per selected model,
              sorted by ARI, NMI, and silhouette score.
            - ``assignments``: Mapping from model name to assignment dataframe.
            - ``X_scaled``: Scaled feature matrix used for clustering.
            - ``y_true``: Original normalized author labels as a NumPy array.
            - ``label_encoder``: Fitted label encoder for author labels.
    """

    if not selected_models:
        raise ValueError("Select at least one clustering model.")

    X = data[feature_cols].copy()
    y = data["author_norm"].copy()
    n_clusters = int(n_clusters or y.nunique())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    available_models = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=seed, n_init=20),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "gmm": GaussianMixture(n_components=n_clusters, random_state=seed),
    }

    rows: list[dict[str, Any]] = []
    assignments: dict[str, pd.DataFrame] = {}
    for model_name in selected_models:
        model = available_models[model_name]
        if model_name == "gmm":
            model.fit(X_scaled)
            clusters = model.predict(X_scaled)
        else:
            clusters = model.fit_predict(X_scaled)

        rows.append(
            {
                "model": model_name,
                "n_clusters": n_clusters,
                "silhouette": _safe_silhouette(X_scaled, clusters),
                "ARI": adjusted_rand_score(y_encoded, clusters),
                "NMI": normalized_mutual_info_score(y_encoded, clusters),
                "homogeneity": homogeneity_score(y_encoded, clusters),
                "completeness": completeness_score(y_encoded, clusters),
                "v_measure": v_measure_score(y_encoded, clusters),
            }
        )
        assignments[model_name] = pd.DataFrame(
            {
                "file_name": data["file_name"].values,
                "work": data["work"].values,
                "author_norm": y.values,
                "cluster": clusters,
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["ARI", "NMI", "silhouette"], ascending=False).reset_index(drop=True)
    return {
        "results": results_df,
        "assignments": assignments,
        "X_scaled": X_scaled,
        "y_true": y.values,
        "label_encoder": label_encoder,
    }


def _hierarchical_linkage(X_scaled: np.ndarray, method: str) -> np.ndarray:
    """Computes a hierarchical clustering linkage matrix.

    Uses SciPy's default metric handling for Ward linkage and Euclidean distance
    for all other linkage methods.

    Args:
        X_scaled: Scaled feature matrix with one row per observation.
        method: Hierarchical linkage method passed to ``scipy.cluster.hierarchy.linkage``.

    Returns:
        A SciPy linkage matrix representing the hierarchical clustering tree.
    """

    if method == "ward":
        return linkage(X_scaled, method=method)
    return linkage(X_scaled, method=method, metric="euclidean")


def _plotly_dendrogram_color(color: str) -> str:
    """Converts a SciPy dendrogram color code to a Plotly-compatible color.

    Maps SciPy's default categorical color codes, such as ``"C0"`` and ``"C1"``,
    to hexadecimal color strings. Colors that are not present in the mapping are
    returned unchanged.

    Args:
        color: Color value emitted by SciPy's dendrogram output.

    Returns:
        A hexadecimal color string for known SciPy categorical color codes, or
        the original ``color`` value when no mapping exists.
    """

    scipy_color_map = {
        "C0": "#1f77b4",
        "C1": "#ff7f0e",
        "C2": "#2ca02c",
        "C3": "#d62728",
        "C4": "#9467bd",
        "C5": "#8c564b",
        "C6": "#e377c2",
        "C7": "#7f7f7f",
        "C8": "#bcbd22",
        "C9": "#17becf",
    }
    return scipy_color_map.get(color, color)


def _author_color_map(authors: pd.Series) -> dict[str, str]:
    """Builds a deterministic color map for author labels.

    Assigns colors from a fixed palette to the unique author values in
    alphabetical order. Author values are converted to strings before sorting and
    before being used as dictionary keys. If there are more authors than colors,
    the palette is reused cyclically.

    Args:
        authors: Series containing author labels or values that can be converted
            to strings.

    Returns:
        A dictionary mapping stringified author labels to hexadecimal color
        strings.
    """

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#7f7f7f",
        "#003f5c",
        "#ffa600",
        "#665191",
        "#a05195",
        "#f95d6a",
    ]
    return {str(author): palette[i % len(palette)] for i, author in enumerate(sorted(authors.astype(str).unique()))}


def _safe_silhouette(X_scaled: np.ndarray, clusters: np.ndarray) -> float:
    """Computes a silhouette score when cluster labels are valid.

    Returns ``NaN`` instead of calling ``silhouette_score`` when the labels do
    not define a valid silhouette problem. A valid silhouette score requires at
    least two clusters and fewer clusters than observations.

    Args:
        X_scaled: Scaled feature matrix with one row per observation.
        clusters: Cluster labels aligned row-by-row with ``X_scaled``.

    Returns:
        The silhouette score as a float, or ``NaN`` when there are fewer than two
        clusters or when each observation has its own cluster.
    """

    n_labels = len(set(clusters))
    if n_labels < 2 or n_labels >= len(clusters):
        return float("nan")
    return float(silhouette_score(X_scaled, clusters))
