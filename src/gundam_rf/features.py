from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LEAKAGE_COLUMNS = [
    "death_in_war_target",
    "death_status_label",
    "death_label_confidence",
    "death_label_note",
    "death_label_source_url",
    "row_research_status",
    "needs_manual_review",
]

ID_COLUMNS = [
    "character_id",
    "source_line",
    "row_order_in_source",
]

BASE_FEATURE_COLUMNS = [
    "universe_code",
    "work_title_jp",
    "medium_hint",
    "canon_tier_hint",
    "faction_family",
    "affiliation_context_jp",
    "is_military_or_combat_faction_proxy",
    "is_zeon_related_proxy",
    "is_federation_related_proxy",
    "is_named_pilot_or_fighter_proxy",
    "has_alias_or_alt_identity",
    "voice_actor_jp",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-generation settings."""

    include_text_length_features: bool = True
    include_voice_actor_feature: bool = True


def _is_known_text(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return (~text.eq("")) & (~text.str.lower().isin(["unknown", "nan", "none", "-"]))


def build_feature_frame(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Create a model-ready feature frame without direct target leakage."""
    config = config or FeatureConfig()

    available_columns = [col for col in BASE_FEATURE_COLUMNS if col in df.columns]
    X = df[available_columns].copy()

    # Normalize missing-like tokens in categorical fields.
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})

    # Convert bool columns explicitly.
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    if config.include_text_length_features:
        for col in [
            "character_name_jp",
            "raw_name_jp",
            "aliases_or_alt_names_jp",
            "appearance_note_jp",
            "character_name_en",
            "affiliation_context_jp",
        ]:
            if col in df.columns:
                text = df[col].fillna("").astype(str)
                X[f"{col}_char_len"] = text.str.len()
                X[f"{col}_is_known"] = _is_known_text(df[col]).astype(int)

    if config.include_voice_actor_feature and "voice_actor_jp" in df.columns:
        X["has_voice_actor_jp"] = _is_known_text(df["voice_actor_jp"]).astype(int)

    return X
