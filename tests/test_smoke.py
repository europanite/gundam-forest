from pathlib import Path

from gundam_rf.data import load_dataset, make_weak_binary_target
from gundam_rf.features import build_feature_frame


def test_dataset_loads() -> None:
    path = Path("data/raw/gundam_character_death_dataset_first_pass.csv")
    df = load_dataset(path)
    assert len(df) > 0
    assert "death_in_war_target" in df.columns


def test_feature_frame_and_target_have_matching_rows() -> None:
    path = Path("data/raw/gundam_character_death_dataset_first_pass.csv")
    df = load_dataset(path)
    X = build_feature_frame(df)
    y = make_weak_binary_target(df)
    assert len(X) == len(y) == len(df)
    assert "death_in_war_target" not in X.columns
