# -*- coding: utf-8 -*-
"""
Modul: config
=============
Zentrale Konfiguration für das Fahrererkennungs-Projekt. Lädt Einstellungen aus
config.json (falls vorhanden), sonst werden Standardwerte verwendet. Die Funktion
apply_overrides() erlaubt das Überschreiben von Werten (z.B. aus CLI oder GUI).

Verwendung:
    from . import config
"""
import json
import sys
from pathlib import Path

# Projekt-Root für relativen Pfad zu config.json
if getattr(sys, "frozen", False):
    _PROJ = Path(sys.executable).parent
else:
    _PROJ = Path(sys.argv[0]).resolve().parent

# === Standardwerte (werden durch config.json oder apply_overrides überschrieben) ===
DATA_DIR = Path("data")                    # Ordner mit CSV-Recordings
LABELS_FILE = Path("labels.lbl")           # Trainings-Label-Datei
TEST_LABELS_FILE = Path("test_labels.lbl")  # Test-Label-Datei
ARTIFACTS_DIR = Path("artifacts")          # Ausgabe-Ordner für Modelle und Ergebnisse
MODELS = ["randomforest", "logreg", "gradientboosting"]  # Zu trainierende Modelltypen
USE_GRID_SEARCH = False                    # Bei True: GridSearch vor Training
FEATURE_SET = "both"                       # "featuretools" | "tsfresh" | "both"
WINDOW_SEC, STEP_SEC, MIN_POINTS, MAX_POINTS = 25, 12, 300, 500  # Fenster, Schritt, Min/Max-Punkte
CV_SPLITS = 5                              # Anzahl Folds für Cross-Validation
RANDOM_STATE = 42                          # Reproduzierbarkeit
COLUMNS = ["timestamp", "wheel_position", "car0_throttle_position", "car0_brake_position", "car0_velocity_vehicle", "rot_vel"]  # Erforderliche CSV-Spalten


def _load_from_file(path=None):
    """
    Lädt Konfiguration aus JSON-Datei und überschreibt globale Variablen.

    Args:
        path: Pfad zu config.json (optional, sonst _PROJ/config.json)
    """
    global DATA_DIR, LABELS_FILE, TEST_LABELS_FILE, ARTIFACTS_DIR
    global MODELS, FEATURE_SET, WINDOW_SEC, STEP_SEC, MIN_POINTS, MAX_POINTS
    global CV_SPLITS, RANDOM_STATE, USE_GRID_SEARCH
    cfg_path = path or (_PROJ / "config.json")
    if not cfg_path.exists():
        print("Config file not found")
        return
    with open(cfg_path, encoding="utf-8") as f:
        d = json.load(f)
    # Nur vorhandene Keys überschreiben
    if "data_dir" in d: DATA_DIR = Path(d["data_dir"])
    if "labels_file" in d: LABELS_FILE = Path(d["labels_file"])
    if "test_labels_file" in d: TEST_LABELS_FILE = Path(d["test_labels_file"])
    if "artifacts_dir" in d: ARTIFACTS_DIR = Path(d["artifacts_dir"])
    if "models" in d: MODELS = list(d["models"])
    if "feature_set" in d: FEATURE_SET = d["feature_set"]
    if "window_sec" in d: WINDOW_SEC = int(d["window_sec"])
    if "step_sec" in d: STEP_SEC = int(d["step_sec"])
    if "min_points" in d: MIN_POINTS = int(d["min_points"])
    if "max_points" in d: MAX_POINTS = int(d["max_points"])
    if "cv_splits" in d: CV_SPLITS = int(d["cv_splits"])
    if "random_state" in d: RANDOM_STATE = int(d["random_state"])
    if "use_grid_search" in d: USE_GRID_SEARCH = bool(d["use_grid_search"])


def apply_overrides(**kwargs):
    """
    Überschreibt Konfiguration mit übergebenen Werten (z.B. aus CLI oder backend_api).

    Args:
        **kwargs: data_dir, labels_file, test_labels_file, artifacts_dir, models,
                  feature_set, window_sec, step_sec, min_points, max_points,
                  cv_splits, random_state
    """
    global DATA_DIR, LABELS_FILE, TEST_LABELS_FILE, ARTIFACTS_DIR
    global MODELS, FEATURE_SET, WINDOW_SEC, STEP_SEC, MIN_POINTS, MAX_POINTS
    global CV_SPLITS, RANDOM_STATE, USE_GRID_SEARCH
    if "data_dir" in kwargs: DATA_DIR = Path(kwargs["data_dir"])
    if "labels_file" in kwargs: LABELS_FILE = Path(kwargs["labels_file"])
    if "test_labels_file" in kwargs: TEST_LABELS_FILE = Path(kwargs["test_labels_file"])
    if "artifacts_dir" in kwargs: ARTIFACTS_DIR = Path(kwargs["artifacts_dir"])
    if "models" in kwargs: MODELS = list(kwargs["models"])
    if "feature_set" in kwargs: FEATURE_SET = kwargs["feature_set"]
    if "window_sec" in kwargs: WINDOW_SEC = int(kwargs["window_sec"])
    if "step_sec" in kwargs: STEP_SEC = int(kwargs["step_sec"])
    if "min_points" in kwargs: MIN_POINTS = int(kwargs["min_points"])
    if "max_points" in kwargs: MAX_POINTS = int(kwargs["max_points"])
    if "cv_splits" in kwargs: CV_SPLITS = int(kwargs["cv_splits"])
    if "random_state" in kwargs: RANDOM_STATE = int(kwargs["random_state"])
    if "use_grid_search" in kwargs: USE_GRID_SEARCH = bool(kwargs["use_grid_search"])


# Beim Import automatisch config.json laden (falls vorhanden)
_load_from_file()
