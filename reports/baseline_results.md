# Baseline Results

These results were generated with the weak label strategy `Unknown -> 0`.

## Metrics

| Metric | Value |
|---|---:|
| balanced_accuracy | 0.7239 |
| roc_auc | 0.8589 |
| average_precision | 0.3472 |

## Confusion matrix

Rows are true labels and columns are predicted labels.

| true \ pred | 0 | 1 |
|---:|---:|---:|
| 0 | 607 | 67 |
| 1 | 24 | 29 |

## Top feature importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `numeric__raw_name_jp_char_len` | 0.112984 |
| 2 | `numeric__character_name_jp_char_len` | 0.103386 |
| 3 | `numeric__affiliation_context_jp_char_len` | 0.057736 |
| 4 | `categorical__canon_tier_hint_Primary animation / screen work` | 0.055407 |
| 5 | `numeric__is_military_or_combat_faction_proxy` | 0.047462 |
| 6 | `categorical__canon_tier_hint_Expanded universe / non-screen or game` | 0.041651 |
| 7 | `categorical__faction_family_Civilian / Other` | 0.029467 |
| 8 | `categorical__universe_code_UC / alternate UC` | 0.025451 |
| 9 | `categorical__work_title_jp_機動戦士ガンダム C.D.A. 若き彗星の肖像` | 0.016694 |
| 10 | `categorical__medium_hint_TV anime` | 0.015400 |
| 11 | `categorical__work_title_jp_機動戦士Ζガンダム` | 0.014839 |
| 12 | `categorical__medium_hint_Manga` | 0.013367 |
| 13 | `categorical__affiliation_context_jp_ティターンズ/地球連邦軍` | 0.013077 |
| 14 | `categorical__faction_family_Other / Unknown faction` | 0.012505 |
| 15 | `categorical__work_title_jp_機動戦士Vガンダム` | 0.012236 |

## Interpretation warning

Because the current data does not contain verified negative labels, these numbers should be read as a pipeline baseline, not as final model quality.