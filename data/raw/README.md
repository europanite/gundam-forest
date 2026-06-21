# Raw data

Place external datasets here. The repository intentionally does not require the
Titanic CSV to be committed.

Expected Titanic baseline file:

```text
data/raw/titanic.csv
```

Required columns:

```text
Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
```

A Kaggle-style Titanic `train.csv` can be renamed to `titanic.csv`. The training
script can also use OpenML as a network-based fallback:

```bash
python scripts/train_titanic_baseline.py --use-openml
```
