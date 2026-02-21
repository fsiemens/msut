# -*- coding: utf-8 -*-
"""
Modul: predict
==============
Vorhersage-Pipeline für die Fahrererkennung. Lädt trainierte Modelle und Test-Labels,
extrahiert Features, führt Vorhersage pro Modell aus und aggregiert pro Recording
(Vorhersage = argmax über gemittelte Fenster-Wahrscheinlichkeiten). Schreibt
Ergebnisse in test_ergebnis_*.csv.

CLI: python predict.py [--data-dir DIR] [--test-labels FILE] [--artifacts DIR] [--config PATH]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Callable

from . import config
from .data import load_labels
from .features import extract_features
from .progress import write_progress
from .plots import plot_feature_importance, plot_feature_importance_all_models


def _parse_args():
    """CLI-Argumente parsen und config.apply_overrides vorbereiten."""
    p = argparse.ArgumentParser(description="Führt Vorhersage mit trainierten Modellen aus.")
    p.add_argument("--data-dir", type=str, help="Ordner mit Test-CSV-Recordings")
    p.add_argument("--test-labels", type=str, help="Test-Label-Datei (z.B. test_labels.lbl)")
    p.add_argument("--artifacts", type=str, help="Ordner mit Modellen")
    p.add_argument("--config", type=str, help="Pfad zu config.json")
    return p.parse_args()

def predict(data_dir=None, test_labels_file : str | Path | pd.DataFrame | None = None, artifacts_dir=None, progress_callback : Callable | None = None):
    """
    Führt Vorhersage mit allen trainierten Modellen aus. Pro Recording wird die
    Vorhersage aus den gemittelten Fenster-Wahrscheinlichkeiten ermittelt.

    Args:
        data_dir: Ordner mit Test-CSV-Recordings (optional)
        test_labels_file: Test-Label-Datei (optional)
        artifacts_dir: Ordner mit Modellen (optional)
    """
    data_dir = data_dir or config.DATA_DIR
    test_labels_file = test_labels_file if test_labels_file is not None else config.TEST_LABELS_FILE
    artifacts_dir = Path(artifacts_dir or config.ARTIFACTS_DIR)

    write_progress(artifacts_dir, phase="starting", message="Lade Test-Labels...", callback=progress_callback)
    # FEATURE_SET und feat_cols aus erstem Modell laden (müssen übereinstimmen)
    _ld = joblib.load(artifacts_dir / f"model_{config.MODELS[0]}.joblib")
    FEATURE_SET, feat_cols = _ld[2], _ld[1]

    paths, ids = load_labels(test_labels_file, False, data_dir)
    if not paths:
        raise SystemExit("Keine gültigen Test-Labels gefunden.")
    print(f"Lade {len(paths)} Test-Recordings...")

    def _on_extraction_start():
        write_progress(artifacts_dir, phase="extraction", message="Extraktion läuft...", callback=progress_callback)
    result = extract_features(paths, ids, FEATURE_SET, on_extraction_start=_on_extraction_start)

    # Fehlende Features (z.B. wenn TSFresh andere Spalten liefert) mit NaN auffüllen
    for c in feat_cols:
        if c not in result.columns: result[c] = np.nan
    X = result[feat_cols]

    total_models = len(config.MODELS)
    for i, mdl in enumerate(config.MODELS):
        completed = config.MODELS[:i]
        in_progress = [mdl]
        write_progress(
            artifacts_dir,
            phase="prediction",
            total=total_models,
            completed=completed,
            in_progress=in_progress,
            message=f"Vorhersage {mdl}...",
            callback=progress_callback
        )
        pipe = joblib.load(artifacts_dir / f"model_{mdl}.joblib")[0]
        proba = pipe.predict_proba(X)
        inner = pipe.named_steps["clf"]
        # Klassen vom Modell verwenden (nicht aus Test-Labels) – proba ist bereits in dieser Reihenfolge
        model_classes = list(inner.classes_)
        pdf = pd.DataFrame(proba, columns=model_classes)
        pdf["rec"], pdf["y_true"] = result["recording"].values, result["driver_id"].values
        # Pro Recording: Wahrscheinlichkeiten mitteln, argmax = Vorhersage
        agg = pdf.groupby("rec", sort=False).agg({**{c: "mean" for c in model_classes}, "y_true": "first"})
        recs = [{"recording": r, "soll": agg.loc[r, "y_true"], "ist": model_classes[np.argmax(agg.loc[r, model_classes].values)], "korrekt": model_classes[np.argmax(agg.loc[r, model_classes].values)] == agg.loc[r, "y_true"]} for r in agg.index]
        out = artifacts_dir / f"test_ergebnis_{mdl}.csv"
        pd.DataFrame(recs).to_csv(out, index=False)
        plot_feature_importance(pipe, feat_cols, mdl, artifacts_dir)
        df_str = pd.DataFrame(recs).to_string(index=False)
        korrekt = sum(x["korrekt"] for x in recs)
        print(f"\n--- {mdl} ---")
        print(df_str)
        print(f"Korrekt: {korrekt}/{len(recs)}")
        print(f"Gespeichert: {out}, plots/prediction/")

    write_progress(
        artifacts_dir,
        phase="done",
        total=total_models,
        completed=config.MODELS,
        in_progress=[],
        message="Fertig",
        callback=progress_callback
    )
    # Kombinierter Feature-Importance-Plot für alle Modelle
    pipes_all = {mdl: joblib.load(artifacts_dir / f"model_{mdl}.joblib")[0] for mdl in config.MODELS}
    plot_feature_importance_all_models(pipes_all, feat_cols, artifacts_dir)

if __name__ == "__main__":
    args = _parse_args()
    if args.config:
        config._load_from_file(args.config)  # Alternative config.json laden
    overrides = {}
    if args.data_dir: overrides["data_dir"] = args.data_dir
    if args.test_labels: overrides["test_labels_file"] = args.test_labels
    if args.artifacts: overrides["artifacts_dir"] = args.artifacts
    if overrides:
        config.apply_overrides(**overrides)
    predict()
