import re
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold

def format_age(row, **kwargs):
    if type(row) == float:
        return row

    replacements = {
        'Age': '',
        'to': '-',
        'or older': '+',
        ' ': ''
    }
    pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
    
    formated = pattern.sub(lambda match: replacements[match.group(0)], row)
    return formated


def get_feature_importances(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_importances = np.zeros(X.shape[1])

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        feature_importances += model.feature_importances_

    feature_importances /= skf.get_n_splits()

    feature_names = X.columns.to_list()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df.reset_index(drop=True, inplace=True)
    return importance_df

def append_comparison_table(model_df, comparison_df=None):
    if not isinstance(comparison_df, pd.DataFrame):
        return model_df
    else:
        comparison_df = pd.concat([comparison_df, model_df])
        comparison_df.reset_index(drop=True, inplace=True)

    return comparison_df


def evaluate_model(model, X, y, model_name, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=1),
        "recall": make_scorer(recall_score, average="weighted", zero_division=1),
        "f1": make_scorer(f1_score, average="weighted", zero_division=1),
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=metrics, return_train_score=True)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {}
    results["Model"] = model_name
    
    for metric in metric_names:
        results[f"Train {metric.capitalize()}"] = scores[f'train_{metric}'].mean()
        results[f"Test {metric.capitalize()}"] = scores[f'test_{metric}'].mean()
    return pd.DataFrame([results])