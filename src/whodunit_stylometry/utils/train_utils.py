"""Model training and evaluation utilities."""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    completeness_score,
    confusion_matrix,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC, LinearSVC

from whodunit_stylometry.constants import STOPWORDS
from whodunit_stylometry.utils.nlp_utils import build_mfw_features, transform_with_mfw


def run_experiment(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list,
    author_col: str,
    models: dict,
    seed: int = 42,
    scoring: str = "f1_macro",
):
    """Run cross-validated model selection and evaluate the best model on a test set.

    This function trains and compares multiple candidate models using
    cross-validation on the training set. Each model is wrapped in a pipeline
    with median imputation and robust scaling. The model with the highest mean
    cross-validation score is then fitted on the full training data and
    evaluated on the test set.

    Args:
        train_data (pd.DataFrame): Training dataset containing the feature
            columns and target column.
        test_data (pd.DataFrame): Test dataset containing the feature columns
            and target column.
        feature_cols (list): List of column names to use as input features.
        author_col (str): Name of the target column containing the class labels.
        models (dict): Dictionary mapping model names to scikit-learn compatible
            estimator instances.
        seed (int, optional): Random seed used for cross-validation shuffling.
            Defaults to 42.
        scoring (str, optional): Scoring metric passed to
            ``sklearn.model_selection.cross_val_score``. Defaults to
            ``"f1_macro"``.

    Returns:
        dict: A dictionary containing the experiment outputs with the following
        keys:
            - ``cv_results`` (pd.DataFrame): Cross-validation summary for each
              model, including mean and standard deviation of the scores.
            - ``best_model_name`` (str): Name of the best-performing model based
              on mean cross-validation score.
            - ``best_pipeline`` (Pipeline): Fitted preprocessing and model
              pipeline for the selected best model.
            - ``test_f1_macro`` (float): Macro-averaged F1 score on the test
              set.
            - ``classification_report`` (str): Text summary of classification
              metrics on the test set.
            - ``confusion_matrix`` (np.ndarray): Confusion matrix computed on
              the test set.
            - ``labels`` (list): Sorted list of class labels used in the
              confusion matrix.
            - ``pred_df`` (pd.DataFrame): DataFrame containing the file name,
              true label, predicted label, and correctness flag for each test
              sample.
            - ``feature_cols`` (list): The list of feature columns used in the
              experiment.
    """
    X_train = train_data[feature_cols]
    y_train = train_data[author_col].astype(str)

    X_test = test_data[feature_cols]
    y_test = test_data[author_col].astype(str)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    cv_results = []

    for name, model in models.items():
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", clone(model)),
            ]
        )

        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)

        cv_results.append(
            {
                "model": name,
                "cv_mean": scores.mean(),
                "cv_std": scores.std(),
            }
        )

    cv_results_df = pd.DataFrame(cv_results).sort_values("cv_mean", ascending=False).reset_index(drop=True)

    best_model_name = cv_results_df.iloc[0]["model"]
    best_model = clone(models[best_model_name])

    best_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", best_model),
        ]
    )

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    labels = sorted(set(y_train) | set(y_test))

    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    pred_cols = [author_col]
    if "file_name" in test_data.columns:
        pred_cols.insert(0, "file_name")

    pred_df = test_data[pred_cols].copy()
    pred_df["pred_author"] = y_pred
    pred_df["correct"] = pred_df[author_col] == pred_df["pred_author"]

    return {
        "cv_results": cv_results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipe,
        "test_f1_macro": test_f1_macro,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "pred_df": pred_df,
        "feature_cols": feature_cols,
    }


def build_models_with_seed(seed: int) -> dict[str, Any]:
    """Build a collection of classifier instances using a shared random seed.

    The returned mapping contains preconfigured scikit-learn classifier objects.
    Models that support reproducibility receive the provided ``seed`` through
    their ``random_state`` parameter.

    Args:
        seed: Random seed used for estimators that expose a ``random_state``
            parameter.

    Returns:
        A dictionary mapping model names to instantiated classifier objects.

    Notes:
        The returned estimators are newly created on each call.
        Not all estimators in the mapping use ``seed``. For example,
        ``KNeighborsClassifier`` and ``GaussianNB`` are instantiated without a
        random state because they do not use one in this configuration.
    """
    return {
        "logreg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
        "linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=10000,
        ),
        "svc_rbf": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(n_neighbors=3),
        "gaussian_nb": GaussianNB(),
    }


def get_mfw_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that start with the ``"fw_"`` prefix.

    Args:
        df: Input DataFrame whose columns are inspected.

    Returns:
        A list of column names from ``df.columns`` that start with ``"fw_"``.
    """
    return [c for c in df.columns if c.startswith("fw_")]


def summarize_result_row(
    seed: int,
    top_n_mfw: int,
    result_dict: dict,
) -> dict:
    """Build a flat summary row from an experiment result dictionary.

    The summary includes the seed, the selected ``top_n_mfw`` value, the best
    model name, and the overall test macro F1 score.

    Args:
        seed: Random seed associated with the experiment.
        top_n_mfw: Number of top MFW features used in the experiment.
        result_dict: Dictionary containing experiment outputs. It must include
            ``"best_model_name"`` and ``"test_f1_macro"``.

    Returns:
        A dictionary representing a single flattened summary row.
    """
    row = {
        "seed": seed,
        "top_n_mfw": top_n_mfw,
        "best_model_name": result_dict["best_model_name"],
        "test_f1_macro": result_dict["test_f1_macro"],
    }

    return row


def build_train_test_mfw(
    top_n: int,
    train_tokens_by_file: dict[str, list[str]],
    test_tokens_by_file: dict[str, list[str]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build MFW-based train and test feature sets and merge them with metadata.

    The MFW vocabulary is learned from the training tokens only, then reused to
    transform the test tokens so both datasets share the same feature space.
    The resulting feature tables are merged with ``file_name`` and
    ``author_norm`` metadata from the corresponding input DataFrames.

    Args:
        top_n: Number of most frequent function-word features to keep when
            building the training vocabulary.
        train_tokens_by_file: Mapping from training file name to its tokenized
            content.
        test_tokens_by_file: Mapping from test file name to its tokenized
            content.
        train_df: Training metadata DataFrame. It must contain at least the
            ``"file_name"`` and ``"author_norm"`` columns.
        test_df: Test metadata DataFrame. It must contain at least the
            ``"file_name"`` and ``"author_norm"`` columns.

    Returns:
        A tuple containing:
            - The learned MFW vocabulary.
            - The training DataFrame with MFW features and metadata.
            - The test DataFrame with MFW features and metadata.
            - The list of MFW feature column names.

    Raises:
        KeyError: If ``train_df`` or ``test_df`` does not contain required
            columns such as ``"file_name"`` or ``"author_norm"``.
        NameError: If required external names such as ``STOPWORDS``,
            ``build_mfw_features``, ``transform_with_mfw``, or
            ``get_mfw_feature_cols`` are not defined.

    Notes:
        The exact type of ``mfw_vocab`` cannot be inferred safely from this
        function alone, so it is annotated as ``Any``.
    """
    # MFW vocabulary learned from training data only.
    mfw_vocab, train_mfw_df = build_mfw_features(
        tokens_by_file=train_tokens_by_file,
        function_words=STOPWORDS,
        top_n=top_n,
    )

    # Transform test data using the vocabulary learned on the training set.
    test_mfw_df = transform_with_mfw(
        tokens_by_file=test_tokens_by_file,
        mfw=mfw_vocab,
    )

    # Merge with metadata.
    train_fw = train_mfw_df.merge(
        train_df[["file_name", "author_norm"]],
        on="file_name",
        how="left",
    )

    test_fw = test_mfw_df.merge(
        test_df[["file_name", "author_norm"]],
        on="file_name",
        how="left",
    )

    fw_cols = get_mfw_feature_cols(train_fw)
    return mfw_vocab, train_fw, test_fw, fw_cols


def run_clustering_experiment(
    data: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "author_norm",
    file_col: str = "file_name",
    seed: int = 0,
) -> dict[str, Any]:
    """Run several clustering models and evaluate their performance.

    This function scales the selected feature columns, infers the number of
    clusters from the number of unique values in the target label column, and
    evaluates multiple clustering algorithms using both internal and external
    metrics.

    The evaluated models are:
        - KMeans
        - AgglomerativeClustering
        - GaussianMixture

    For each model, the function computes clustering assignments and summary
    metrics. It also returns the scaled feature matrix and the fitted label
    encoder used to transform the true labels.

    Args:
        data: Input dataset containing feature columns, a label column, and a
            file identifier column.
        feature_cols: Names of the columns used as clustering features.
        label_col: Name of the column containing ground-truth labels used for
            external evaluation metrics. Defaults to ``"author_norm"``.
        file_col: Name of the column containing file identifiers used in the
            returned assignment tables. Defaults to ``"file_name"``.
        seed: Random seed used by stochastic models. Defaults to ``0``.

    Returns:
        A dictionary with the following keys:
            - ``"results"``: A DataFrame with one row per model and the computed
              evaluation metrics, sorted by ARI, NMI, and silhouette score in
              descending order.
            - ``"assignments"``: A mapping from model name to a DataFrame with
              file identifiers, true labels, and predicted cluster assignments.
            - ``"X_scaled"``: The standardized feature matrix as returned by
              ``RobustScaler.fit_transform``.
            - ``"y_true"``: The original label values as a NumPy array.
            - ``"label_encoder"``: The fitted ``LabelEncoder`` instance.
            - ``"feature_cols"``: The input feature column names.

    Notes:
        The number of clusters is set to the number of unique values in
        ``label_col``. This assumes the ground-truth label cardinality is an
        appropriate target for all evaluated clustering models.
    """
    X = data[feature_cols].copy()
    y = data[label_col].copy()

    # Encode true labels for external metrics.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_clusters = y.nunique()

    # Scale features.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=seed, n_init=20),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "gmm": GaussianMixture(n_components=n_clusters, random_state=seed),
    }

    rows: list[dict[str, Any]] = []
    cluster_assignments: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        if model_name == "gmm":
            model.fit(X_scaled)
            clusters = model.predict(X_scaled)
        else:
            clusters = model.fit_predict(X_scaled)

        # Internal and external metrics.
        sil = silhouette_score(X_scaled, clusters)
        ari = adjusted_rand_score(y_encoded, clusters)
        nmi = normalized_mutual_info_score(y_encoded, clusters)
        hom = homogeneity_score(y_encoded, clusters)
        comp = completeness_score(y_encoded, clusters)
        v_measure = v_measure_score(y_encoded, clusters)

        rows.append(
            {
                "model": model_name,
                "n_clusters": n_clusters,
                "silhouette": sil,
                "ARI": ari,
                "NMI": nmi,
                "homogeneity": hom,
                "completeness": comp,
                "v_measure": v_measure,
            }
        )

        cluster_assignments[model_name] = pd.DataFrame(
            {file_col: data[file_col].values, label_col: y.values, "cluster": clusters}
        )

    results_df = pd.DataFrame(rows).sort_values(["ARI", "NMI", "silhouette"], ascending=False).reset_index(drop=True)

    return {
        "results": results_df,
        "assignments": cluster_assignments,
        "X_scaled": X_scaled,
        "y_true": y.values,
        "label_encoder": le,
        "feature_cols": feature_cols,
    }


def evaluate_cluster_errors(
    assignments_df: pd.DataFrame,
    label_col: str = "author_norm",
    cluster_col: str = "cluster",
    file_col: str = "file_name",
) -> dict[str, Any]:
    """Evaluate cluster-to-label mapping errors using majority-vote assignment.

    The function builds a contingency table between predicted clusters and true
    labels, assigns each cluster to its most frequent true label, and then uses
    that mapping to derive label predictions from cluster assignments.

    It returns the augmented input data, summary accuracy after mapping,
    a label-space confusion matrix, and a table containing only the
    misclassified rows.

    Args:
        assignments_df: DataFrame containing at least the true label column,
            the predicted cluster column, and a file identifier column.
        label_col: Name of the column containing the true labels.
            Defaults to ``"author_norm"``.
        cluster_col: Name of the column containing the predicted cluster
            assignments. Defaults to ``"cluster"``.
        file_col: Name of the column containing file identifiers.
            Defaults to ``"file_name"``.

    Returns:
        A dictionary with the following keys:
            - ``"data_with_predictions"``: A copy of the input DataFrame with
              two additional columns:
                - ``"pred_author_from_cluster"``: label predicted from the
                  cluster-to-label mapping.
                - ``"correct"``: boolean flag indicating whether the mapped
                  prediction matches the true label.
            - ``"contingency_table"``: Cross-tabulation of clusters by true
              labels.
            - ``"cluster_to_author"``: Mapping from cluster value to the
              majority label assigned to that cluster.
            - ``"accuracy_after_mapping"``: Mean accuracy obtained after
              translating clusters into labels.
            - ``"confusion_matrix"``: Confusion matrix in label space as a
              DataFrame indexed and columned by sorted label values.
            - ``"errors_df"``: Subset of misclassified rows, sorted by true
              label, predicted label, and file identifier.

    Notes:
        When two or more labels are tied within a cluster, the selected majority
        label depends on the behavior of ``pandas.DataFrame.idxmax`` and the
        column ordering in the contingency table.
    """
    df = assignments_df.copy()

    # Cluster x true label table
    contingency = pd.crosstab(df[cluster_col], df[label_col])

    # Assign each cluster to its majority label
    cluster_to_author = contingency.idxmax(axis=1).to_dict()

    # "Translated" prediction in label space
    df["pred_author_from_cluster"] = df[cluster_col].map(cluster_to_author)

    # Correct / incorrect prediction
    df["correct"] = df["pred_author_from_cluster"] == df[label_col]

    # Global summary
    accuracy = df["correct"].mean()

    # Confusion matrix in label space
    labels = sorted(df[label_col].unique())
    cm = confusion_matrix(df[label_col], df["pred_author_from_cluster"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Errors only
    errors_df = df.loc[
        ~df["correct"],
        [file_col, label_col, cluster_col, "pred_author_from_cluster"],
    ].copy()

    return {
        "data_with_predictions": df,
        "contingency_table": contingency,
        "cluster_to_author": cluster_to_author,
        "accuracy_after_mapping": accuracy,
        "confusion_matrix": cm_df,
        "errors_df": errors_df.sort_values([label_col, "pred_author_from_cluster", file_col]),
    }


def cluster_author_table(
    assignments_df: pd.DataFrame,
    label_col: str = "author_norm",
) -> pd.DataFrame:
    """Build a contingency table of clusters versus true labels.

    This function creates a cross-tabulation where rows correspond to values
    in the ``"cluster"`` column and columns correspond to values in
    ``label_col``. Each cell contains the count of rows assigned to the
    corresponding cluster-label pair.

    Args:
        assignments_df: DataFrame containing at least a ``"cluster"`` column
            and the label column specified by ``label_col``.
        label_col: Name of the column containing the true labels.
            Defaults to ``"author_norm"``.

    Returns:
        A pandas DataFrame representing the contingency table of cluster counts
        by label.
    """
    table = pd.crosstab(assignments_df["cluster"], assignments_df[label_col])
    return table


def errors_by_true_author(
    eval_dict: dict[str, Any],
    label_col: str = "author_norm",
) -> pd.DataFrame:
    """Summarize prediction errors grouped by true author label.

    This function reads the ``"data_with_predictions"`` DataFrame from
    ``eval_dict`` and aggregates the ``"correct"`` column by true label.
    It computes the total number of works, the number of correct predictions,
    the number of errors, and the error rate for each label.

    Args:
        eval_dict: Dictionary expected to contain a
            ``"data_with_predictions"`` entry with a pandas DataFrame.
            That DataFrame must include ``label_col`` and ``"correct"``
            columns.
        label_col: Name of the column containing the true labels.
            Defaults to ``"author_norm"``.

    Returns:
        A DataFrame indexed by true label with the following columns:
            - ``"total_works"``: Number of rows for the label.
            - ``"correct"``: Number of rows marked as correct.
            - ``"errors"``: Difference between total works and correct rows.
            - ``"error_rate"``: Proportion of errors over total works.

        The result is sorted by ``"error_rate"`` in descending order.
    """
    df = eval_dict["data_with_predictions"]
    out = (
        df.groupby(label_col)["correct"]
        .agg(total_works="count", correct="sum")
        .assign(errors=lambda x: x["total_works"] - x["correct"])
        .assign(error_rate=lambda x: x["errors"] / x["total_works"])
        .sort_values("error_rate", ascending=False)
    )
    return out


def confusion_pairs(
    eval_dict: dict[str, Any],
    label_col: str = "author_norm",
) -> pd.DataFrame:
    """Summarize the most frequent confusion pairs in misclassified rows.

    This function reads the ``"errors_df"`` DataFrame from ``eval_dict`` and
    counts how often each pair of true label and predicted label occurs among
    the errors. The result is sorted by the number of errors in descending
    order.

    Args:
        eval_dict: Dictionary expected to contain an ``"errors_df"`` entry with
            a pandas DataFrame. That DataFrame must include ``label_col`` and
            ``"pred_author_from_cluster"`` columns.
        label_col: Name of the column containing the true labels.
            Defaults to ``"author_norm"``.

    Returns:
        A DataFrame with one row per confusion pair and the following columns:
            - ``label_col``: The true label.
            - ``"pred_author_from_cluster"``: The mapped predicted label.
            - ``"n_errors"``: Number of times that confusion pair appears.

        The result is sorted by ``"n_errors"`` in descending order.
    """
    errors = eval_dict["errors_df"]
    return (
        errors.groupby([label_col, "pred_author_from_cluster"])
        .size()
        .reset_index(name="n_errors")
        .sort_values("n_errors", ascending=False)
    )


def evaluate_hierarchical_clustering(
    data: pd.DataFrame,
    feature_cols: list[Hashable],
    label_col: Hashable = "author_norm",
    file_col: Hashable = "file_name",
    method: str = "ward",
    metric: str = "euclidean",
    n_clusters: int | None = None,
) -> dict[str, Any]:
    """Evaluate hierarchical clustering against reference labels.

    This function standardizes the selected feature columns, computes a
    hierarchical clustering linkage matrix, assigns cluster labels, and
    evaluates the clustering result using both internal and external metrics.
    External metrics are computed by comparing the predicted clusters against
    the values in ``label_col``.

    Args:
        data: Input DataFrame containing features, labels, and file identifiers.
        feature_cols: Column names used as clustering features.
        label_col: Column name containing the reference labels used for
            evaluation. Defaults to ``"author_norm"``.
        file_col: Column name containing file identifiers to include in the
            assignments output. Defaults to ``"file_name"``.
        method: Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
            Defaults to ``"ward"``.
        metric: Distance metric passed to ``linkage`` when ``method`` is not
            ``"ward"``. Defaults to ``"euclidean"``.
        n_clusters: Number of clusters to extract with ``fcluster``. When
            ``None``, the number of unique values in ``label_col`` is used.

    Returns:
        A dictionary with the following keys:
            - ``"Z"``: The hierarchical linkage matrix.
            - ``"results"``: A one-row DataFrame with clustering evaluation
              metrics and configuration values.
            - ``"assignments"``: A DataFrame containing the file identifier,
              true label, and assigned cluster for each row.
            - ``"X_scaled"``: The standardized feature matrix as returned by
              ``RobustScaler``.
            - ``"y_true"``: The original label values as a NumPy array.
    """
    X = data[feature_cols].copy()
    y = data[label_col].copy()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "ward":
        Z = linkage(X_scaled, method=method)
    else:
        Z = linkage(X_scaled, method=method, metric=metric)

    if n_clusters is None:
        n_clusters = y.nunique()

    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    results = {
        "method": method,
        "metric": metric,
        "n_clusters": n_clusters,
        "silhouette": silhouette_score(X_scaled, clusters),
        "ARI": adjusted_rand_score(y_encoded, clusters),
        "NMI": normalized_mutual_info_score(y_encoded, clusters),
        "homogeneity": homogeneity_score(y_encoded, clusters),
        "completeness": completeness_score(y_encoded, clusters),
        "v_measure": v_measure_score(y_encoded, clusters),
    }

    assignments = pd.DataFrame({file_col: data[file_col].values, label_col: y.values, "cluster": clusters})

    return {
        "Z": Z,
        "results": pd.DataFrame([results]),
        "assignments": assignments,
        "X_scaled": X_scaled,
        "y_true": y.values,
    }


def compare_hierarchical_methods(
    data: pd.DataFrame,
    feature_cols: list[Hashable],
    label_col: Hashable = "author_norm",
    file_col: Hashable = "file_name",
    methods: Sequence[str] = ("ward", "complete", "average", "single"),
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compare multiple hierarchical clustering linkage methods.

    This function evaluates several hierarchical clustering methods using the
    same input data and feature set, then concatenates the resulting metric
    summaries into a single DataFrame sorted by clustering quality metrics.

    For each method, the number of clusters is fixed to the number of unique
    values in ``label_col``.

    Args:
        data: Input DataFrame containing features, labels, and file identifiers.
        feature_cols: Column names used as clustering features.
        label_col: Column name containing the reference labels used for
            evaluation. Defaults to ``"author_norm"``.
        file_col: Column name containing file identifiers passed through to
            ``evaluate_hierarchical_clustering``. Defaults to ``"file_name"``.
        methods: Hierarchical linkage methods to evaluate. Defaults to
            ``("ward", "complete", "average", "single")``.
        metric: Distance metric passed to
            ``evaluate_hierarchical_clustering`` for methods that support it.
            Defaults to ``"euclidean"``.

    Returns:
        A DataFrame containing one row per evaluated method, sorted by ``ARI``,
        ``NMI``, and ``silhouette`` in descending order.
    """
    rows = []

    for method in methods:
        result = evaluate_hierarchical_clustering(
            data=data,
            feature_cols=feature_cols,
            label_col=label_col,
            file_col=file_col,
            method=method,
            metric=metric,
            n_clusters=data[label_col].nunique(),
        )
        rows.append(result["results"])

    return pd.concat(rows, ignore_index=True).sort_values(["ARI", "NMI", "silhouette"], ascending=False)


def coefficients_long(coef_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a class-by-feature coefficient matrix to long format.

    The returned DataFrame contains one row per class-feature pair. It also adds
    the absolute coefficient value and a direction label based on whether the
    coefficient is non-negative or negative.

    Args:
        coef_matrix: DataFrame whose index contains author or class labels and
            whose columns contain feature names. Cell values are coefficients.

    Returns:
        A DataFrame with the columns:
            - `author`: The author or class label from `coef_matrix.index`.
            - `feature`: The feature name from `coef_matrix.columns`.
            - `coefficient`: The coefficient value for the class-feature pair.
            - `abs_coefficient`: The absolute value of `coefficient`.
            - `direction`: `"pushes_toward_author"` when `coefficient >= 0`,
              otherwise `"pushes_away_from_author"`.

    Raises:
        AttributeError: If `coef_matrix` does not provide DataFrame-like methods
            such as `rename_axis`, `reset_index`, or `melt`.
        TypeError: If coefficient values do not support `.abs()` or comparison
            with zero.
    """
    long_df = (
        coef_matrix.rename_axis("author")
        .reset_index()
        .melt(id_vars="author", var_name="feature", value_name="coefficient")
    )
    long_df["abs_coefficient"] = long_df["coefficient"].abs()
    long_df["direction"] = np.where(
        long_df["coefficient"] >= 0,
        "pushes_toward_author",
        "pushes_away_from_author",
    )
    return long_df


def top_coefficients_per_author(coef_matrix: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest positive and negative coefficients for each author.

    For each author row in `coef_matrix`, this function selects the `top_n`
    largest coefficients and the `top_n` smallest coefficients. The result is
    returned in long format with one row per selected author-feature pair.

    Args:
        coef_matrix: DataFrame whose index contains author labels and whose
            columns contain feature names. Cell values are coefficients.
        top_n: Number of largest and smallest coefficients to include for each
            author.

    Returns:
        A DataFrame with the columns:
            - `author`: The author label from `coef_matrix.index`.
            - `feature`: The feature name from `coef_matrix.columns`.
            - `coefficient`: The selected coefficient value.
            - `direction`: `"pushes_toward_author"` for the largest
              coefficients, or `"pushes_away_from_author"` for the smallest
              coefficients.
    """
    rows = []
    for author, row in coef_matrix.iterrows():
        positive = row.sort_values(ascending=False).head(top_n)
        negative = row.sort_values(ascending=True).head(top_n)

        for feature, value in positive.items():
            rows.append(
                {
                    "author": author,
                    "feature": feature,
                    "coefficient": value,
                    "direction": "pushes_toward_author",
                }
            )

        for feature, value in negative.items():
            rows.append(
                {
                    "author": author,
                    "feature": feature,
                    "coefficient": value,
                    "direction": "pushes_away_from_author",
                }
            )

    return pd.DataFrame(rows)


def upper_triangle_pairs(corr: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Return upper-triangle correlation pairs above an absolute threshold.

    This function extracts unique feature pairs from the upper triangle of a
    correlation matrix, excluding the diagonal. It computes the absolute
    correlation for each pair and returns only pairs whose absolute correlation
    is greater than or equal to `threshold`.

    Args:
        corr: Square correlation matrix as a DataFrame. The index and columns
            are expected to contain feature names.
        threshold: Minimum absolute correlation required for a pair to be
            included.

    Returns:
        A DataFrame with the columns:
            - `feature_1`: The row feature name.
            - `feature_2`: The column feature name.
            - `spearman_corr`: The correlation value from `corr`.
            - `abs_corr`: The absolute value of `spearman_corr`.

        Rows are sorted by `abs_corr` in descending order.

    Raises:
        ValueError: If `corr` is not two-dimensional or if its shape is invalid
            for the NumPy upper-triangle mask operation.
        TypeError: If correlation values do not support absolute-value
            calculation or comparison with `threshold`.
    """
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = (
        corr.where(mask)
        .stack()
        .rename("spearman_corr")
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    )
    pairs["abs_corr"] = pairs["spearman_corr"].abs()
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False)


def transformed_features(pipeline: Any, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Apply all pipeline steps except the final classifier.

    This function copies `X`, applies each transformer from `pipeline.steps`
    except the last step, and returns the transformed data as a DataFrame using
    the original index and the provided feature names.

    Args:
        pipeline: Pipeline-like object with a `steps` attribute containing
            `(name, transformer)` pairs. Each transformer is expected to provide
            a `transform` method.
        X: Input feature DataFrame to transform.
        feature_names: Column names to assign to the transformed output. Its
            length must match the number of columns in the transformed data.

    Returns:
        A DataFrame containing the transformed feature values, with `X.index` as
        its index and `feature_names` as its columns.
    """
    steps = list(pipeline.steps[:-1])
    Xt = X.copy()

    for _, transformer in steps:
        Xt = transformer.transform(Xt)

    return pd.DataFrame(Xt, index=X.index, columns=feature_names)


def local_linear_contributions(
    X_model_space: pd.DataFrame,
    coef_matrix: pd.DataFrame,
    instance_index: Hashable,
    author: str,
) -> pd.DataFrame:
    """Compute feature-level linear contributions for one instance and author.

    For the selected instance and author, this function multiplies each
    transformed feature value by the corresponding model coefficient. The result
    shows how much each feature contributes to the linear score for that author,
    before any intercept or probability transformation is applied.

    Args:
        X_model_space: DataFrame containing transformed model features. Its
            columns are expected to match `coef_matrix.columns`.
        coef_matrix: DataFrame whose index contains author labels and whose
            columns contain feature names. Cell values are linear model
            coefficients.
        instance_index: Index label used to select the target row from
            `X_model_space`.
        author: Author label used to select the target coefficient row from
            `coef_matrix`.

    Returns:
        A DataFrame with the columns:
            - `feature`: Feature name from `coef_matrix.columns`.
            - `x_transformed`: Transformed feature value for the selected
              instance.
            - `coefficient`: Coefficient for the selected author and feature.
            - `contribution`: Product of `x_transformed` and `coefficient`.
            - `abs_contribution`: Absolute value of `contribution`.

        Rows are sorted by `abs_contribution` in descending order.
    """
    x = X_model_space.loc[instance_index]
    beta = coef_matrix.loc[author]

    out = pd.DataFrame(
        {
            "feature": coef_matrix.columns,
            "x_transformed": x.values,
            "coefficient": beta.values,
            "contribution": x.values * beta.values,
        }
    )
    out["abs_contribution"] = out["contribution"].abs()

    return out.sort_values("abs_contribution", ascending=False)
