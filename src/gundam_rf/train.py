from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import load_dataset, make_weak_binary_target, summarize_labels
from .features import build_feature_frame


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across recent scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            sparse=False,
        )


def make_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", _make_one_hot_encoder(), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    classifier = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def _safe_auc(y_true: pd.Series, y_score: np.ndarray, metric: str) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    if metric == "average_precision":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(metric)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_auc(y_test, y_proba, "roc_auc"),
        "average_precision": _safe_auc(y_test, y_proba, "average_precision"),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "per_class": {
            "0": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "1": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            zero_division=0,
            output_dict=True,
        ),
    }


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{idx}" for idx in range(len(classifier.feature_importances_))]

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_training(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    y = make_weak_binary_target(df, unknown_as_negative=True)
    X = build_feature_frame(df)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = make_pipeline(X_train, random_state=random_state)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metrics["data_path"] = str(data_path)
    metrics["label_strategy"] = "weak_unknown_as_negative"
    metrics["n_rows"] = int(len(df))
    metrics["n_features_before_encoding"] = int(X.shape[1])
    metrics["target_counts_after_weak_conversion"] = {
        str(k): int(v) for k, v in y.value_counts().sort_index().items()
    }
    metrics["original_target_summary"] = summarize_labels(df).to_dict(orient="records")

    feature_importance = get_feature_importance(model)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    predictions = df_test[
        [
            col
            for col in [
                "character_id",
                "character_name_jp",
                "character_name_en",
                "universe_code",
                "work_title_jp",
                "death_in_war_target",
                "death_status_label",
                "needs_manual_review",
            ]
            if col in df_test.columns
        ]
    ].copy()
    predictions["y_true_weak"] = y_test.values
    predictions["predicted_probability_death"] = model.predict_proba(X_test)[:, 1]
    predictions["predicted_label_weak"] = model.predict(X_test)
    predictions = predictions.sort_values("predicted_probability_death", ascending=False)
    predictions.to_csv(output_dir / "predictions_holdout.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(model, output_dir / "random_forest_pipeline.joblib")

    return metrics
