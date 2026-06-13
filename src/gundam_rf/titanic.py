from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TITANIC_TARGET_COLUMN = "Survived"
TITANIC_NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
TITANIC_CATEGORICAL_FEATURES = ["Sex", "Embarked"]
TITANIC_FEATURE_COLUMNS = TITANIC_NUMERIC_FEATURES + TITANIC_CATEGORICAL_FEATURES

_COLUMN_ALIASES = {
    "survived": "Survived",
    "pclass": "Pclass",
    "sex": "Sex",
    "age": "Age",
    "sibsp": "SibSp",
    "parch": "Parch",
    "fare": "Fare",
    "embarked": "Embarked",
}

_MISSING_CATEGORY_TOKENS = {
    "",
    "nan",
    "none",
    "<na>",
    "?",
}


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across recent scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_titanic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Titanic column-name variants to Kaggle-style names."""
    rename_map = {}
    for column in df.columns:
        key = str(column).strip().lower()
        if key in _COLUMN_ALIASES:
            rename_map[column] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def _normalize_categorical_series(series: pd.Series) -> pd.Series:
    """Return object-dtype categories with pandas.NA converted to np.nan."""
    cleaned = series.astype("object").where(pd.notna(series), np.nan)
    cleaned = cleaned.map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    return cleaned.map(
        lambda value: np.nan
        if isinstance(value, str) and value.lower() in _MISSING_CATEGORY_TOKENS
        else value
    )


def load_titanic_csv(path: str | Path) -> pd.DataFrame:
    """Load a Titanic training CSV with Kaggle-style or OpenML-style columns."""
    df = pd.read_csv(path)
    return validate_titanic_frame(normalize_titanic_columns(df))


def load_titanic_openml() -> pd.DataFrame:
    """Load Titanic from OpenML when a local CSV is not available.

    This requires network access. For fully reproducible runs, prefer a local CSV
    placed under data/raw/titanic.csv.
    """
    from sklearn.datasets import fetch_openml

    frame = fetch_openml("titanic", version=1, as_frame=True).frame
    return validate_titanic_frame(normalize_titanic_columns(frame))


def validate_titanic_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the DataFrame contains the ordinary Titanic baseline columns."""
    required_columns = [TITANIC_TARGET_COLUMN, *TITANIC_FEATURE_COLUMNS]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Titanic dataset is missing required columns: "
            f"{missing}. Expected columns: {required_columns}"
        )

    result = df[required_columns].copy()
    # OpenML often uses pandas extension dtypes. Convert pandas.NA to np.nan so
    # scikit-learn imputers do not hit "boolean value of NA is ambiguous".
    result = result.where(pd.notna(result), np.nan)
    result[TITANIC_TARGET_COLUMN] = pd.to_numeric(
        result[TITANIC_TARGET_COLUMN], errors="coerce"
    )
    result = result.dropna(subset=[TITANIC_TARGET_COLUMN])
    result[TITANIC_TARGET_COLUMN] = result[TITANIC_TARGET_COLUMN].astype(int)

    supported = set(result[TITANIC_TARGET_COLUMN].unique().tolist())
    if not supported.issubset({0, 1}):
        raise ValueError(f"Survived must be binary 0/1. Found: {sorted(supported)}")

    for column in TITANIC_NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in TITANIC_CATEGORICAL_FEATURES:
        result[column] = _normalize_categorical_series(result[column])

    return result


def build_titanic_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split Titanic data into model features and target."""
    clean = validate_titanic_frame(normalize_titanic_columns(df))
    X = clean[TITANIC_FEATURE_COLUMNS].copy()
    y = clean[TITANIC_TARGET_COLUMN].copy()
    return X, y


def make_titanic_pipeline(random_state: int = 42) -> Pipeline:
    """Create an ordinary Random Forest baseline for Titanic survival prediction."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, TITANIC_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, TITANIC_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_titanic_training(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/titanic_baseline",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and save the Titanic Random Forest baseline."""
    X, y = build_titanic_xy(df)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = make_titanic_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics: dict[str, Any] = {
        "dataset": "titanic",
        "target": TITANIC_TARGET_COLUMN,
        "features": TITANIC_FEATURE_COLUMNS,
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "target_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_roc_auc(y_test, y_proba),
        "average_precision": _safe_average_precision(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions = X_test.copy()
    predictions[TITANIC_TARGET_COLUMN] = y_test.to_numpy()
    predictions["predicted_survived"] = y_pred
    predictions["predicted_survival_probability"] = y_proba
    predictions.to_csv(output_path / "predictions_holdout.csv", index=False)

    importance = _feature_importance_frame(pipeline)
    importance.to_csv(output_path / "feature_importance.csv", index=False)

    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(pipeline, output_path / "random_forest_pipeline.joblib")

    return {
        "metrics": metrics,
        "output_dir": str(output_path),
        "feature_importance": importance,
    }
