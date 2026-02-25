# -*- coding: utf-8 -*-
"""
Modul: backend_adapter
=====================
Schnittstelle für Tkinter- und andere GUI-Frontends. Kapselt train() und predict()
mit Fehlerbehandlung. Gibt bei Fehlern (False, msg) bzw. (False, msg, {}) zurück
statt Exceptions zu werfen – damit das Frontend Fehler anzeigen kann ohne zu crashen.

- train(): Trainiert Modelle, apply_overrides für Pfade, write_progress für Status
- predict(): Führt Vorhersage aus, liefert ergebnisse als dict pro Modell
- get_config() / set_config(): Konfiguration für GUI-Voreinstellungen
- validate_csv(): Prüft, ob eine CSV-Datei lesbar ist
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Callable

from . import config
from .progress import write_progress
from .train import train as _train
from .data import load_csv
from .predict import predict as _predict

# Projekt-Root in sys.path, damit Importe auch bei Aufruf von außerhalb (z.B. GUI) funktionieren
_proj = Path(__file__).resolve().parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

def train(data_dir : str | Path, labels : pd.DataFrame, artifacts_dir, progress_callback : Callable | None = None, use_grid_search : bool = False):
    """
    Trainiert Modelle. print()-Ausgaben von run.train() werden abgefangen und
    an log_callback weitergeleitet.

    Args:
        data_dir: Ordner mit CSV-Recordings
        labels: DataFrame mit File/Label-Spalten
        artifacts_dir: Ausgabe-Ordner für Modelle
        progress_callback: Optionaler Callback für Fortschritt
        use_grid_search: Bei True: GridSearch für Hyperparameter vor Training

    Returns:
        (erfolg, ausgabe) – True/False und Log-Text
    """
    try:
        config.apply_overrides(data_dir=str(data_dir), artifacts_dir=str(artifacts_dir))
        # Sofort Fortschritt zurücksetzen, damit Frontend nicht alte "done"-Datei sieht
        out_dir = Path(artifacts_dir)
        write_progress(out_dir, phase="starting", message="Starte Training...", callback=progress_callback)
        _train(labels=labels, progress_callback=progress_callback, use_grid_search=use_grid_search)
        return True, None
    except SystemExit as e:
        # train() wirft SystemExit bei Fehlern (z.B. keine Labels gefunden)
        return False, str(e) if e.code else "Unbekannter Fehler"
    except Exception as e:
        return False, str(e)


def predict(data_dir : str | Path, test_labels_file : pd.DataFrame, artifacts_dir, progress_callback : Callable | None = None):
    """
    Führt Vorhersage aus.

    Args:
        data_dir: Ordner mit Test-CSV-Recordings
        test_labels_file: Pfad zur Test-Label-Datei
        artifacts_dir: Ordner mit trainierten Modellen
        log_callback: Optionaler Callback für Log-Ausgaben

    Returns:
        (erfolg, ausgabe, ergebnisse) – ergebnisse: dict mit Modellnamen als Keys,
        Liste von {recording, soll, ist, korrekt} als Values
    """
    test_labels_file = test_labels_file.copy()
    if not "Label" in test_labels_file:
        test_labels_file["Label"] = ""  # Label-Spalte optional für Vorhersage

    try:
        config.apply_overrides(
            data_dir=str(data_dir),
            artifacts_dir=str(artifacts_dir),
        )
        # Sofort Fortschritt zurücksetzen, damit Frontend nicht alte "done"-Datei sieht
        out_dir = Path(artifacts_dir)
        write_progress(out_dir, phase="starting", message="Starte Vorhersage...", callback=progress_callback)
        _predict(test_labels_file=test_labels_file, progress_callback=progress_callback)
        # predict.py schreibt test_ergebnis_*.csv – einlesen und als dict für GUI zurückgeben
        ergebnisse = {}
        for mdl in config.MODELS:
            csv_path = Path(artifacts_dir) / f"test_ergebnis_{mdl}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                ergebnisse[mdl] = df.to_dict("records")
        return True, None, ergebnisse
    except SystemExit as e:
        return False, str(e) if e.code else "Unbekannter Fehler", {}
    except Exception as e:
        return False, str(e), {}

def get_config() -> dict:
    """
    Liefert aktuelle Konfiguration für Voreinstellungen im GUI.

    Returns:
        dict mit data_dir, labels_file, test_labels_file, artifacts_dir, models, window_sec, step_sec
    """
    return {
        "data_dir":{ "value": config.DATA_DIR, "desc": "Ordner mit CSV-Recordings" },
        "labels_file":{ "value": config.LABELS_FILE, "desc": "Label-Datei" },
        "test_labels_file":{ "value": config.TEST_LABELS_FILE, "desc": "Label-Datei zum Predicten" },
        "artifacts_dir":{ "value": config.ARTIFACTS_DIR, "desc": "Ausgabe-Ordner für Modelle" },
        "models":{ "value": config.MODELS, "desc": "Zu trainierende Modelltypen", "options": ["randomforest", "logreg", "gradientboosting"] },
        "use_grid_search":{ "value": config.USE_GRID_SEARCH, "desc": "GridSearch für Hyperparameter vor Training" },
        "feature_set":{ "value": config.FEATURE_SET, "desc": "Zu verwendende Datensets", "options": ["featuretools", "tsfresh", "both"] },
        "window_sec":{ "value": config.WINDOW_SEC, "desc": "Fensterlänge (in Sekunden)" },
        "step_sec":{ "value": config.STEP_SEC, "desc": "Schrittweite (in Sekunden)" },
        "min_points":{ "value": config.MIN_POINTS, "desc": "Min. Datenpunkte pro Fenster" },
        "max_points":{ "value": config.MAX_POINTS, "desc": "Max. Datenpunkte pro Fenster" },
        "cv_splits":{ "value": config.CV_SPLITS, "desc": "Anzahl Cross-Validation-Folds" },
        "random_state":{ "value": config.RANDOM_STATE, "desc": "Random Seed" },
    }

def set_config(settings : dict):
    """
    Liefert aktuelle Konfiguration für Voreinstellungen im GUI.

    Returns:
        dict mit data_dir, labels_file, test_labels_file, artifacts_dir, models, window_sec, step_sec
    """

    if "data_dir" in settings: config.DATA_DIR = Path(settings["data_dir"])
    if "labels_file" in settings: config.LABELS_FILE = Path(settings["labels_file"])
    if "test_labels_file" in settings: config.TEST_LABELS_FILE = Path(settings["test_labels_file"])
    if "artifacts_dir" in settings: config.ARTIFACTS_DIR = Path(settings["artifacts_dir"])
    if "models" in settings: config.MODELS = list(settings["models"])
    if "feature_set" in settings: config.FEATURE_SET = settings["feature_set"]
    if "window_sec" in settings: config.WINDOW_SEC = int(settings["window_sec"])
    if "step_sec" in settings: config.STEP_SEC = int(settings["step_sec"])
    if "min_points" in settings: config.MIN_POINTS = int(settings["min_points"])
    if "max_points" in settings: config.MAX_POINTS = int(settings["max_points"])
    if "cv_splits" in settings: config.CV_SPLITS = int(settings["cv_splits"])
    if "random_state" in settings: config.RANDOM_STATE = int(settings["random_state"])
    if "use_grid_search" in settings: config.USE_GRID_SEARCH = bool(settings["use_grid_search"])

    print("Applied config")
    print(settings)

def validate_csv(path : str | Path) -> bool:
    """Prüft, ob eine CSV-Datei lesbar ist (z.B. vor Import im GUI)."""
    try:
        load_csv(path)
        return True
    except:
        return False