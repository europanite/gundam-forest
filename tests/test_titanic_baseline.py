import pandas as pd

from gundam_rf.titanic import build_titanic_xy, run_titanic_training


def _sample_titanic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Survived": [0, 1, 1, 0, 1, 0, 0, 1],
            "Pclass": [3, 1, 3, 2, 1, 3, 2, 1],
            "Sex": [
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "female",
            ],
            "Age": [22, 38, 26, 35, 28, None, 54, 19],
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 0],
            "Parch": [0, 0, 0, 0, 0, 0, 0, 2],
            "Fare": [7.25, 71.28, 7.92, 53.1, 80.0, 8.05, 26.0, 30.0],
            "Embarked": ["S", "C", "S", "S", "C", "S", "S", "Q"],
        }
    )


def test_build_titanic_xy_uses_plain_baseline_columns() -> None:
    X, y = build_titanic_xy(_sample_titanic_frame())
    assert list(X.columns) == [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Sex",
        "Embarked",
    ]
    assert y.tolist() == [0, 1, 1, 0, 1, 0, 0, 1]


def test_run_titanic_training_writes_outputs(tmp_path) -> None:
    result = run_titanic_training(
        _sample_titanic_frame(),
        output_dir=tmp_path,
        test_size=0.25,
        random_state=7,
    )
    assert result["metrics"]["dataset"] == "titanic"
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "predictions_holdout.csv").exists()
    assert (tmp_path / "random_forest_pipeline.joblib").exists()


def test_run_titanic_training_accepts_pandas_na_categories(tmp_path) -> None:
    df = _sample_titanic_frame()
    df.loc[0, "Embarked"] = pd.NA
    df.loc[1, "Sex"] = pd.NA

    result = run_titanic_training(
        df,
        output_dir=tmp_path,
        test_size=0.25,
        random_state=7,
    )

    assert result["metrics"]["dataset"] == "titanic"
    assert (tmp_path / "metrics.json").exists()
