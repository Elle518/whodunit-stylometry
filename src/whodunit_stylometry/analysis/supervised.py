"""Supervised machine-learning workflows for interactive authorship attribution."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC

from whodunit_stylometry.constants import STOPWORDS
from whodunit_stylometry.utils.nlp_utils import CustomTokenizer, tokenize_text
from whodunit_stylometry.utils.stats_utils import drop_highly_correlated_features

FEATURE_SETS = {
    "Métricas estilométricas": "stylometric",
    "MFW": "mfw",
    "Métricas + MFW": "combined",
}

MODEL_NAMES = {
    "Regresión logística": "logreg",
    "Linear SVC": "linear_svc",
    "SVC RBF": "svc_rbf",
    "Random Forest": "random_forest",
    "KNN": "knn",
    "Gaussian NB": "gaussian_nb",
}


@dataclass(frozen=True)
class SupervisedConfig:
    """Configuration for supervised authorship experiments."""

    feature_set: str = "mfw"
    top_n_mfw: int = 25
    seed: int = 42
    n_test_per_author: int = 2
    keep_correlated_features: bool = False
    correlation_threshold: float = 0.85
    scoring: str = "f1_macro"
    selected_models: tuple[str, ...] = ("logreg", "linear_svc", "svc_rbf", "random_forest", "knn", "gaussian_nb")


def build_models(seed: int) -> dict[str, Any]:
    """Build supported classifiers using a shared seed when possible."""

    return {
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        "linear_svc": LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000),
        "svc_rbf": SVC(kernel="rbf", class_weight="balanced", random_state=seed, probability=True),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=3),
        "gaussian_nb": GaussianNB(),
    }


def build_work_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lightweight stylometric features for each work."""

    rows = []
    for row in df.itertuples(index=False):
        tokens = list(row.tokens)
        token_count = len(tokens)
        counts = Counter(tokens)
        lengths = [len(token) for token in tokens]
        stopword_count = sum(1 for token in tokens if token in STOPWORDS)
        text = row.text
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        sentence_lengths = [_count_alpha_words(sentence) for sentence in sentences]
        paragraphs = [part for part in re.split(r"\n\s*\n+", text) if part.strip()]

        rows.append(
            {
                "author": row.author,
                "work": row.work,
                "filename": row.filename,
                "tokens": tokens,
                "token_count": token_count,
                "ttr": len(counts) / token_count if token_count else np.nan,
                "hapax_ratio": sum(1 for count in counts.values() if count == 1) / len(counts) if counts else np.nan,
                "avg_word_len": float(np.mean(lengths)) if lengths else np.nan,
                "median_word_len": float(np.median(lengths)) if lengths else np.nan,
                "long_word_ratio": sum(1 for length in lengths if length >= 7) / token_count if token_count else np.nan,
                "stopword_ratio": stopword_count / token_count if token_count else np.nan,
                "avg_sentence_len": float(np.mean(sentence_lengths)) if sentence_lengths else np.nan,
                "median_sentence_len": float(np.median(sentence_lengths)) if sentence_lengths else np.nan,
                "short_sentence_ratio": (
                    sum(1 for length in sentence_lengths if length <= 10) / len(sentence_lengths)
                    if sentence_lengths
                    else np.nan
                ),
                "long_sentence_ratio": (
                    sum(1 for length in sentence_lengths if length >= 30) / len(sentence_lengths)
                    if sentence_lengths
                    else np.nan
                ),
                "paragraphs_per_1000_tokens": len(paragraphs) / token_count * 1000 if token_count else np.nan,
                "comma_per_1000": text.count(",") / token_count * 1000 if token_count else np.nan,
                "semicolon_per_1000": text.count(";") / token_count * 1000 if token_count else np.nan,
                "colon_per_1000": text.count(":") / token_count * 1000 if token_count else np.nan,
                "question_per_1000": text.count("?") / token_count * 1000 if token_count else np.nan,
                "exclam_per_1000": text.count("!") / token_count * 1000 if token_count else np.nan,
                "quote_per_1000": (text.count('"') + text.count("'")) / token_count * 1000 if token_count else np.nan,
            }
        )

    return pd.DataFrame(rows)


def split_train_test_by_author(
    features_df: pd.DataFrame,
    n_test_per_author: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic author-balanced train/test split."""

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, sub in features_df.groupby("author", sort=True):
        indices = sub.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = min(n_test_per_author, max(1, len(indices) // 3))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        if len(train_idx) == 0:
            train_idx = test_idx
        train_parts.append(features_df.loc[train_idx])
        test_parts.append(features_df.loc[test_idx])

    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def prepare_supervised_datasets(
    df: pd.DataFrame,
    config: SupervisedConfig,
) -> dict[str, Any]:
    """Build train/test datasets for the selected supervised feature set."""

    features_df = build_work_features(df)
    train_df, test_df = split_train_test_by_author(features_df, config.n_test_per_author, config.seed)
    stylometric_cols = _stylometric_feature_cols(features_df)
    removed_features = []
    correlation_summary = pd.DataFrame()

    if (
        config.feature_set in {"stylometric", "combined"}
        and not config.keep_correlated_features
        and len(stylometric_cols) > 1
    ):
        stylometric_cols, removed_features, correlation_summary = drop_highly_correlated_features(
            train_df[stylometric_cols],
            threshold=config.correlation_threshold,
            method="spearman",
        )

    mfw_vocab: list[str] = []
    mfw_cols: list[str] = []
    if config.feature_set in {"mfw", "combined"}:
        mfw_vocab = _build_mfw_vocab(train_df, config.top_n_mfw)
        train_df = _add_mfw_features(train_df, mfw_vocab)
        test_df = _add_mfw_features(test_df, mfw_vocab)
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
        "features_df": features_df,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "stylometric_cols": stylometric_cols,
        "mfw_vocab": mfw_vocab,
        "mfw_cols": mfw_cols,
        "removed_features": removed_features,
        "correlation_summary": correlation_summary,
    }


def run_supervised_experiment(df: pd.DataFrame, config: SupervisedConfig) -> dict[str, Any]:
    """Compare supervised classifiers and evaluate the best model on the test set."""

    datasets = prepare_supervised_datasets(df, config)
    models = {name: model for name, model in build_models(config.seed).items() if name in config.selected_models}
    result = _run_model_selection(
        train_df=datasets["train_df"],
        test_df=datasets["test_df"],
        feature_cols=datasets["feature_cols"],
        models=models,
        seed=config.seed,
        scoring=config.scoring,
    )
    result.update(datasets)
    return result


def run_mfw_robustness(
    df: pd.DataFrame,
    top_n_values: list[int],
    seeds: list[int],
    n_test_per_author: int,
    tolerance: float,
    selected_models: tuple[str, ...],
) -> dict[str, Any]:
    """Run a sweep over MFW vocabulary sizes and random seeds."""

    rows = []
    for seed in seeds:
        for top_n in top_n_values:
            for model_name in selected_models:
                config = SupervisedConfig(
                    feature_set="mfw",
                    top_n_mfw=top_n,
                    seed=seed,
                    n_test_per_author=n_test_per_author,
                    selected_models=(model_name,),
                )
                result = run_supervised_experiment(df, config)
                rows.append(
                    {
                        "seed": seed,
                        "top_n_mfw": top_n,
                        "model_name": model_name,
                        "test_f1_macro": result["test_f1_macro"],
                        "test_accuracy": result["test_accuracy"],
                        "n_features": len(result["feature_cols"]),
                    }
                )

    sweep_df = pd.DataFrame(rows)
    summary_df = (
        sweep_df.groupby(["top_n_mfw", "model_name"], as_index=False)
        .agg(
            mean_test_f1_macro=("test_f1_macro", "mean"),
            std_test_f1_macro=("test_f1_macro", "std"),
            min_test_f1_macro=("test_f1_macro", "min"),
            max_test_f1_macro=("test_f1_macro", "max"),
            n_runs=("test_f1_macro", "count"),
        )
        .sort_values(["top_n_mfw", "model_name"])
    )
    best_mean = summary_df["mean_test_f1_macro"].max()
    threshold = best_mean - tolerance
    candidates = summary_df[summary_df["mean_test_f1_macro"] >= threshold].sort_values(
        ["top_n_mfw", "mean_test_f1_macro"],
        ascending=[True, False],
    )
    selected_top_n = (
        int(candidates.iloc[0]["top_n_mfw"]) if not candidates.empty else int(summary_df.iloc[-1]["top_n_mfw"])
    )
    selected_model_name = (
        str(candidates.iloc[0]["model_name"]) if not candidates.empty else str(summary_df.iloc[-1]["model_name"])
    )
    final_decision_df = summary_df.copy()
    final_decision_df["keeps_performance"] = final_decision_df["mean_test_f1_macro"] >= threshold
    final_decision_df["selected"] = (final_decision_df["top_n_mfw"] == selected_top_n) & (
        final_decision_df["model_name"] == selected_model_name
    )

    return {
        "sweep_df": sweep_df,
        "summary_df": summary_df,
        "final_decision_df": final_decision_df,
        "best_mean_score": best_mean,
        "threshold_score": threshold,
        "selected_top_n_mfw": selected_top_n,
        "selected_model_name": selected_model_name,
    }


def train_final_model(df: pd.DataFrame, config: SupervisedConfig, model_name: str) -> dict[str, Any]:
    """Fit a final model on the full reference corpus."""

    full_df = build_work_features(df)
    stylometric_cols = _stylometric_feature_cols(full_df)
    mfw_vocab: list[str] = []
    mfw_cols: list[str] = []

    if config.feature_set in {"mfw", "combined"}:
        mfw_vocab = _build_mfw_vocab(full_df, config.top_n_mfw)
        full_df = _add_mfw_features(full_df, mfw_vocab)
        mfw_cols = [f"fw_{word}" for word in mfw_vocab]

    if config.feature_set == "stylometric":
        feature_cols = stylometric_cols
    elif config.feature_set == "mfw":
        feature_cols = mfw_cols
    else:
        feature_cols = stylometric_cols + mfw_cols

    model = build_models(config.seed)[model_name]
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", model),
        ]
    )
    pipeline.fit(full_df[feature_cols], full_df["author"].astype(str))

    return {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "mfw_vocab": mfw_vocab,
        "training_df": full_df,
        "model_name": model_name,
    }


def classify_text_with_supervised_model(
    text: str,
    model_bundle: dict[str, Any],
    config: SupervisedConfig,
    lowercase: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify a single text with a fitted supervised model bundle."""

    tokenizer = CustomTokenizer()
    tokens = tokenize_text(text, tokenizer=tokenizer, lowercase=lowercase)
    sample_df = pd.DataFrame(
        [
            {
                "author": "Obra seleccionada",
                "work": "Obra seleccionada",
                "filename": "uploaded.txt",
                "text": text,
                "tokens": tokens,
            }
        ]
    )
    sample_features = build_work_features(sample_df)
    if config.feature_set in {"mfw", "combined"}:
        sample_features = _add_mfw_features(sample_features, model_bundle["mfw_vocab"])

    feature_cols = model_bundle["feature_cols"]
    pipeline = model_bundle["pipeline"]
    classes = list(pipeline.classes_)

    prediction = pipeline.predict(sample_features[feature_cols])[0]
    transformed = pipeline[:-1].transform(sample_features[feature_cols])
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(transformed)[0]
        score_label = "probability"
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(transformed)
        scores = np.ravel(scores)
        if len(classes) == 2 and len(scores) == 1:
            scores = np.array([-scores[0], scores[0]])
        score_label = "score"
    else:
        scores = np.zeros(len(classes))
        scores[classes.index(prediction)] = 1.0
        score_label = "score"

    ranking_df = pd.DataFrame({"author": classes, score_label: scores})
    ranking_df = ranking_df.sort_values(score_label, ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1
    ranking_df["predicted"] = ranking_df["author"] == prediction

    explanation_df = _linear_feature_contributions(
        pipeline=pipeline,
        sample_features=sample_features,
        feature_cols=feature_cols,
        predicted_author=prediction,
    )

    return ranking_df, explanation_df


def _run_model_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    models: dict[str, Any],
    seed: int,
    scoring: str,
) -> dict[str, Any]:
    X_train = train_df[feature_cols]
    y_train = train_df["author"].astype(str)
    X_test = test_df[feature_cols]
    y_test = test_df["author"].astype(str)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_rows = []
    for name, model in models.items():
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", model),
            ]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})

    cv_results = pd.DataFrame(cv_rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    best_model_name = cv_results.iloc[0]["model"]
    best_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", build_models(seed)[best_model_name]),
        ]
    )
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    labels = sorted(set(y_train) | set(y_test))
    # pred_df = test_df[["filename", "work", "author"]].copy()
    pred_df = test_df[["work", "author"]].copy()
    pred_df["pred_author"] = y_pred
    pred_df["correct"] = pred_df["author"] == pred_df["pred_author"]

    return {
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipe,
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "pred_df": pred_df,
    }


def _count_alpha_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text))


def _stylometric_feature_cols(df: pd.DataFrame) -> list[str]:
    metadata_cols = {"author", "work", "filename", "token_count"}
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in metadata_cols]


def _build_mfw_vocab(df: pd.DataFrame, top_n: int) -> list[str]:
    counts = Counter()
    for tokens in df["tokens"]:
        counts.update(token for token in tokens if token in STOPWORDS)
    return [token for token, _ in counts.most_common(top_n)]


def _add_mfw_features(df: pd.DataFrame, mfw_vocab: list[str]) -> pd.DataFrame:
    feature_rows = []
    for tokens in df["tokens"]:
        counts = Counter(tokens)
        n_tokens = len(tokens)
        feature_rows.append({f"fw_{word}": counts[word] / n_tokens if n_tokens else 0.0 for word in mfw_vocab})
    mfw_df = pd.DataFrame(feature_rows, index=df.index)
    return pd.concat([df.copy(), mfw_df], axis=1)


def _mode_or_first(values: pd.Series) -> str:
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def _linear_feature_contributions(
    pipeline: Pipeline,
    sample_features: pd.DataFrame,
    feature_cols: list[str],
    predicted_author: str,
) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame()

    classes = list(pipeline.classes_)
    if predicted_author not in classes:
        return pd.DataFrame()

    transformed = pipeline[:-1].transform(sample_features[feature_cols])
    class_idx = classes.index(predicted_author)
    coef = clf.coef_[class_idx]
    values = np.ravel(transformed)
    contributions = coef * values
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "value": sample_features[feature_cols].iloc[0].to_numpy(),
                "contribution": contributions,
                "abs_contribution": np.abs(contributions),
            }
        )
        .sort_values("abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
