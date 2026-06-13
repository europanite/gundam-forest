# Model Card: Gundam Character War-Death Random Forest Baseline

## Model

Random Forest classifier.

## Target

`death_in_war_target`

Current weak mapping:

| Original value | Training label |
|---|---:|
| `1` | 1 |
| `Unknown` | 0 |

## Intended use

Exploratory feature analysis for a Gundam character war-death prediction dataset.

## Not intended for

Final factual claims about whether a character survived or died.

## Main limitation

The first-pass dataset has confirmed positives and many unknown rows, but no fully verified negative class.  
As a result, the model mainly learns to distinguish seed-confirmed death rows from unverified rows.

## Leakage policy

Direct death-label columns and manual-review status are excluded from model features.

Excluded columns include:

- `death_in_war_target`
- `death_status_label`
- `death_label_confidence`
- `death_label_note`
- `death_label_source_url`
- `row_research_status`
- `needs_manual_review`

## Recommended validation

Before publication or interpretation:

1. Add manually verified negative labels.
2. Audit a sample of high-probability and low-probability predictions.
3. Re-run the notebook.
4. Compare results with simple baselines.
