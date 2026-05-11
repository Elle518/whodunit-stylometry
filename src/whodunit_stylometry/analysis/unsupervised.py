"""Unsupervised clustering workflows for interactive authorship exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class UnsupervisedConfig:
    """Configuration for unsupervised clustering experiments."""

    feature_set: str = "mfw"
    top_n_mfw: int = 50
    seed: int = 42
    selected_models: tuple[str, ...] = ("kmeans", "agglomerative", "gmm")
    n_clusters: int | None = None


def prepare_unsupervised_dataset(df: pd.DataFrame, config: UnsupervisedConfig) -> dict[str, Any]:
    """Build the selected feature matrix for clustering over the full corpus."""

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
    """Run selected clustering models and evaluate them against author labels."""

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
    """Run a robustness sweep over MFW vocabulary sizes, seeds, and clustering models."""

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


def build_cluster_projection(
    X_scaled: np.ndarray,
    assignments: pd.DataFrame,
    method_name: str,
    seed: int,
) -> pd.DataFrame:
    """Project clustered works to two dimensions with t-SNE for display."""

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
    """Map clusters to majority authors and summarize the induced errors."""

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
    """Build a cluster-by-author contingency table."""

    return pd.crosstab(assignments_df["cluster"], assignments_df[label_col])


def errors_by_true_author(eval_dict: dict[str, Any], label_col: str = "author_norm") -> pd.DataFrame:
    """Summarize mapped-cluster errors by true author."""

    df = eval_dict["data_with_predictions"]
    return (
        df.groupby(label_col)["correct"]
        .agg(total_works="count", correct="sum")
        .assign(errors=lambda x: x["total_works"] - x["correct"])
        .assign(error_rate=lambda x: x["errors"] / x["total_works"])
        .sort_values("error_rate", ascending=False)
    )


def confusion_pairs(eval_dict: dict[str, Any], label_col: str = "author_norm") -> pd.DataFrame:
    """Count true-author vs mapped-author pairs among clustering errors."""

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


def _safe_silhouette(X_scaled: np.ndarray, clusters: np.ndarray) -> float:
    n_labels = len(set(clusters))
    if n_labels < 2 or n_labels >= len(clusters):
        return float("nan")
    return float(silhouette_score(X_scaled, clusters))
