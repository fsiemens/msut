# -*- coding: utf-8 -*-
"""
Modul: run
==========
Trainings-Pipeline für die Fahrererkennung. Lädt Labels und Recordings, extrahiert
Features, trainiert RandomForest und LogisticRegression mit StratifiedGroupKFold
(Recordings bleiben zusammen, keine Datenleckage). Speichert Modelle und CV-Ergebnisse.

CLI: python run.py [--data-dir DIR] [--labels FILE] [--artifacts DIR] [--config PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .data import load_labels
from .features import extract_features
from .progress import write_progress
from .plots import plot_confusion_matrix, plot_feature_importance, plot_accuracy

def _parse_args():
    """CLI-Argumente parsen und config.apply_overrides vorbereiten."""
    p = argparse.ArgumentParser(description="Trainiert Fahrererkennungs-Modelle.")
    p.add_argument("--data-dir", type=str, help="Ordner mit CSV-Recordings")
    p.add_argument("--labels", type=str, help="Label-Datei")
    p.add_argument("--artifacts", type=str, help="Ausgabe-Ordner für Modelle")
    p.add_argument("--config", type=str, help="Pfad zu config.json")
    p.add_argument("--cv-splits", type=int, help="Anzahl CV-Folds")
    p.add_argument("--random-state", type=int, help="Random Seed")
    return p.parse_args()

def train(
        data_dir : str | Path | None = None, 
        labels : str | Path | pd.DataFrame | None = None,
        artifacts_dir : str | Path | None = None, 
        cv_splits=None, random_state=None, 
        progress_callback : Callable | None = None
    ):
    """
    Trainiert alle konfigurierten Modelle. Verwendet StratifiedGroupKFold, damit
    Fenster desselben Recordings nicht gleichzeitig in Train und Test landen.

    Args:
        data_dir: Ordner mit CSV-Recordings (optional, sonst config)
        labels_file: Label-Datei (optional)
        artifacts_dir: Ausgabe-Ordner für Modelle (optional)
        cv_splits: Anzahl CV-Folds (optional)
        random_state: Random Seed (optional)
    """
    data_dir = data_dir or config.DATA_DIR
    labels = labels if labels is not None else config.LABELS_FILE
    artifacts_dir = Path(artifacts_dir or config.ARTIFACTS_DIR)
    cv_splits = cv_splits or config.CV_SPLITS
    random_state = random_state or config.RANDOM_STATE

    write_progress(artifacts_dir, phase="starting", message="Lade Labels...", callback=progress_callback)
    paths, ids = load_labels(labels, True, data_dir)
    if not paths:
        raise SystemExit("Keine gültigen Labels gefunden.")
    print(f"Lade {len(paths)} Recordings...")

    def _on_extraction_start():
        write_progress(artifacts_dir, phase="extraction", message="Extraktion läuft...", callback=progress_callback)
    result = extract_features(paths, ids, config.FEATURE_SET, on_extraction_start=_on_extraction_start)

    feat_cols = [c for c in result.columns if c not in ("driver_id", "recording")]
    X, y = result[feat_cols], result["driver_id"]
    groups = np.asarray(result["recording"].values)  # Gruppierung nach Recording (keine Datenleckage)
    classes = sorted(y.unique().tolist())
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    artifacts_dir.mkdir(exist_ok=True)
    ergebnis = pd.DataFrame(columns=["Model", "Precision"])
    print("Using models")
    print(config.MODELS)
    total_models = len(config.MODELS)
    accuracies = {}
    for i, mdl in enumerate(config.MODELS):
        completed = config.MODELS[:i]
        in_progress = [mdl]
        write_progress(
            artifacts_dir,
            phase="training",
            total=total_models,
            completed=completed,
            in_progress=in_progress,
            message=f"Trainiere {mdl}...",
            callback=progress_callback
        )

        # TODO Hyperparameter in Config auslagern
        clf = (RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=4, max_features="log2", class_weight="balanced", random_state=random_state)
               if mdl == "randomforest" else    # TODO weitere Modelle hinzufügen
               LogisticRegression(C=100.0, solver="saga", max_iter=5000, class_weight="balanced", random_state=random_state))
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
        tl, pl = [], []
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = pipe.predict_proba(X.iloc[test_idx])
            inner = pipe.named_steps["clf"]
            # Klassen-Reihenfolge kann abweichen – Matrix auf unsere classes mappen
            fp = np.zeros((proba.shape[0], len(classes)))
            for j, cls in enumerate(inner.classes_):
                if str(cls) in classes: fp[:, classes.index(str(cls))] = proba[:, j]
            pdf = pd.DataFrame(fp, columns=classes)
            pdf["g"], pdf["y_true"] = groups[test_idx], result.iloc[test_idx]["driver_id"].values
            # Pro Recording: Wahrscheinlichkeiten mitteln, dann argmax für Vorhersage
            agg = pdf.groupby("g", sort=False)[classes].mean()
            yt = pdf.groupby("g", sort=False)["y_true"].first()
            for gid in agg.index:
                tl.append(yt.loc[gid])
                pl.append(classes[np.argmax(agg.loc[gid][classes].values)])
        acc = sum(1 for t, p in zip(tl, pl) if str(t) == str(p)) / len(tl) if tl else 0
        accuracies[mdl] = acc
        pipe.fit(X, y)  # Finales Modell auf allen Daten
        joblib.dump((pipe, feat_cols, config.FEATURE_SET), artifacts_dir / f"model_{mdl}.joblib")
        ergebnis.loc[len(ergebnis)] = [mdl, acc]
        print(f"{mdl}: {acc:.2%}")
        # Plots: Konfusionsmatrix, Feature Importance
        plot_confusion_matrix(tl, pl, classes, mdl, artifacts_dir)
        plot_feature_importance(pipe, feat_cols, mdl, artifacts_dir)

    write_progress(
        artifacts_dir,
        phase="done",
        total=total_models,
        completed=config.MODELS,
        in_progress=[],
        message="Fertig",
        callback=progress_callback
    )
    plot_accuracy(accuracies, artifacts_dir)
    ergebnis.to_csv(artifacts_dir / "ergebnis.csv")
    print("Gespeichert: artifacts/model_*.joblib, ergebnis.csv, plots/")

if __name__ == "__main__":
    args = _parse_args()
    if args.config:
        config._load_from_file(args.config)  # Alternative config.json laden
    overrides = {}
    if args.data_dir: overrides["data_dir"] = args.data_dir
    if args.labels: overrides["labels_file"] = args.labels
    if args.artifacts: overrides["artifacts_dir"] = args.artifacts
    if args.cv_splits: overrides["cv_splits"] = args.cv_splits
    if args.random_state: overrides["random_state"] = args.random_state
    if overrides:
        config.apply_overrides(**overrides)
    train()