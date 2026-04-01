"""Model training and evaluation utilities."""

import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


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
