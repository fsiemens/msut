# -*- coding: utf-8 -*-
"""
Modul: optimize
===============
Hyperparameter-Optimierung via GridSearch für die Fahrererkennung. Nutzt
StratifiedGroupKFold und Recording-Level-Aggregation (wie train.py), damit die
Bewertung mit dem finalen Training übereinstimmt.

Unterstützte Modelle: randomforest, logreg, gradientboosting
GradientBoosting-Grid: n_estimators, learning_rate, max_depth, subsample (0.8 = Stochastic GB)

Hauptfunktionen:
    get_param_grids()   - Liefert Parametergrids pro Modell
    run_grid_search()   - Führt GridSearch für ein Modell aus
    run_grid_search_all() - GridSearch für alle konfigurierten Modelle
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .config import MODELS, CV_SPLITS, RANDOM_STATE


def get_param_grids():
    """
    Liefert Parametergrids für GridSearch pro Modell.

    Returns:
        dict: {model_name: {"param_grid": dict, "base_params": dict}}
    """
    return {
        "randomforest": {
            "param_grid": {
                "clf__n_estimators": [150, 300],
                "clf__max_depth": [4, 5],
                "clf__min_samples_split": [10],
                "clf__min_samples_leaf": [4, 8],
                "clf__max_features": ["log2"],
            },
            "base_params": {
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "logreg": {
            "param_grid": {
                "clf__C": [10.0, 100.0],
                "clf__solver": ["saga"],
                "clf__max_iter": [5000],
            },
            "base_params": {
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "gradientboosting": {
            "param_grid": {
                "clf__n_estimators": [150, 250, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 4, 5],
                "clf__subsample": [0.8, 1.0],
            },
            "base_params": {
                "random_state": RANDOM_STATE,
            },
        },
    }


def _recording_level_accuracy(pipe, X, y, groups, classes):
    """
    Berechnet Recording-Level-Genauigkeit: Pro Recording Wahrscheinlichkeiten
    mitteln, argmax = Vorhersage, dann Accuracy über Recordings.

    Args:
        pipe: Fitted Pipeline
        X: Features
        y: Labels (für Gruppen)
        groups: Recording-IDs pro Zeile
        classes: Klassenliste

    Returns:
        float: Accuracy (0..1)
    """
    proba = pipe.predict_proba(X)
    inner = pipe.named_steps["clf"]
    model_classes = list(inner.classes_)
    pdf = pd.DataFrame(proba, columns=model_classes)
    pdf["g"], pdf["y_true"] = groups, y.values
    agg = pdf.groupby("g", sort=False).agg({**{c: "mean" for c in model_classes}, "y_true": "first"})
    tl, pl = [], []
    for gid in agg.index:
        tl.append(agg.loc[gid, "y_true"])
        pl.append(model_classes[np.argmax(agg.loc[gid, model_classes].values)])
    return sum(1 for t, p in zip(tl, pl) if str(t) == str(p)) / len(tl) if tl else 0.0


def run_grid_search(
    X,
    y,
    groups,
    model_name,
    cv_splits=None,
    random_state=None,
    param_grid_override=None,
):
    """
    Führt GridSearch für ein Modell aus. Bewertung erfolgt auf Recording-Ebene.

    Args:
        X: Feature-DataFrame
        y: Labels (Series)
        groups: Recording-IDs (array, parallel zu X/y)
        model_name: "randomforest", "logreg" oder "gradientboosting"
        cv_splits: Anzahl CV-Folds (optional)
        random_state: Random Seed (optional)
        param_grid_override: Überschreibt Parametergrid (optional)

    Returns:
        tuple: (best_params: dict, best_score: float, all_results: list)
    """
    cv_splits = cv_splits or CV_SPLITS
    random_state = random_state or RANDOM_STATE
    grids = get_param_grids()
    if model_name not in grids:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    cfg = grids[model_name]
    param_grid = param_grid_override or cfg["param_grid"]
    base_params = cfg["base_params"]

    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    classes = sorted(y.unique().tolist())

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(product(*values))

    best_score = -1.0
    best_params = None
    all_results = []

    for combo in combinations:
        params = dict(zip(keys, combo))
        clf_params = {k.replace("clf__", ""): v for k, v in params.items()}
        clf_params.update(base_params)

        if model_name == "randomforest":
            clf = RandomForestClassifier(**clf_params)
        elif model_name == "logreg":
            clf = LogisticRegression(**clf_params)
        else:
            clf = GradientBoostingClassifier(**clf_params)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        scores = []
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            acc = _recording_level_accuracy(
                pipe,
                X.iloc[test_idx],
                y.iloc[test_idx],
                groups[test_idx],
                classes,
            )
            scores.append(acc)

        mean_score = np.mean(scores)
        all_results.append({"params": params, "mean_score": mean_score, "scores": scores})

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score, all_results


def run_grid_search_all(
    X,
    y,
    groups,
    cv_splits=None,
    random_state=None,
):
    """
    Führt GridSearch für alle konfigurierten Modelle aus.

    Args:
        X: Feature-DataFrame
        y: Labels (Series)
        groups: Recording-IDs (array)
        cv_splits: Anzahl CV-Folds (optional)
        random_state: Random Seed (optional)

    Returns:
        dict: {model_name: {"best_params": dict, "best_score": float, "all_results": list}}
    """
    results = {}
    for mdl in MODELS:
        best_params, best_score, all_results = run_grid_search(
            X, y, groups, mdl, cv_splits, random_state
        )
        results[mdl] = {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }
    return results
