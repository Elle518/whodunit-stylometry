"""Model training and evaluation utilities."""

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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
