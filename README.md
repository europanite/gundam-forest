# [Gundam Forest](https://github.com/europanite/gundam-forest "Gundam Forest")

Random Forest Analysis of Kill In Action rate for every personnel in gundam world, like in Titanic.

A JupyterLab-based machine learning project for predicting whether a Gundam character is killed in war or armed conflict.

This repository is designed for exploratory analysis, GitHub publication, and reproducible local execution with Docker Compose.

## Important dataset warning

The current dataset is a **first-pass dataset**.

In the provided CSV, `death_in_war_target` currently contains:

- `1`: confirmed / seed-matched war death
- `Unknown`: not yet row-level verified

There is no fully verified negative class yet.  
For that reason, this project uses a baseline strategy named:

```text
weak_unknown_as_negative
```

This means `Unknown` is temporarily converted to `0` only so that a Random Forest baseline can run.  
This is useful for feature exploration, but it should **not** be treated as a final factual death/survival classifier.

Before using the model seriously, add verified non-death labels and reduce `needs_manual_review=True` rows.

## Quick start with Docker Compose

Build and start JupyterLab:

```bash
docker compose up --build
```

Open JupyterLab:

```text
http://localhost:8888/lab?token=gundam-rf
```

Then open:

```text
notebooks/01_random_forest_death_prediction.ipynb
```

## Run training from the command line

```bash
docker compose run --rm jupyter python scripts/train_model.py
```

The script writes these files:

```text
outputs/metrics.json
outputs/feature_importance.csv
outputs/predictions_holdout.csv
outputs/random_forest_pipeline.joblib
```

## Run tests

```bash
docker compose run --rm jupyter pytest
```

## Dataset columns used by the baseline

The baseline avoids direct label/leakage columns such as:

- `death_in_war_target`
- `death_status_label`
- `death_label_confidence`
- `death_label_note`
- `death_label_source_url`
- `row_research_status`
- `needs_manual_review`

Default model features include:

- universe / work / medium / canon hints
- faction family
- military / Zeon / Federation / pilot proxy flags
- alias flag
- voice actor field
- simple derived text features, such as whether English name, voice actor, or affiliation text exists

## Included baseline outputs

A baseline run has already been executed once. See:

```text
reports/baseline_results.md
outputs/metrics.json
outputs/feature_importance.csv
outputs/predictions_holdout.csv
```

These outputs are generated with the weak `Unknown -> 0` assumption.
