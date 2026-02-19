# -*- coding: utf-8 -*-
"""
Modul: backend_api
==================
Schnittstelle für Tkinter- und andere GUI-Frontends. Kapselt run.train() und
predict.predict() mit Fehlerbehandlung und optionalem log_callback. Gibt bei
Fehlern (False, msg) zurück statt Exceptions zu werfen.

API-Funktionen:
    train()            - Trainiert Modelle, gibt (ok, msg) zurück
    predict()          - Führt Vorhersage aus, gibt (ok, out, ergebnisse) zurück
    write_labels_file() - Schreibt Label-Datei aus Dateipfaden und Labels
    get_config()       - Liefert aktuelle Konfiguration als dict
"""

import sys
from pathlib import Path

# Projekt-Root in sys.path, damit Importe auch bei Aufruf von außerhalb funktionieren
_proj = Path(__file__).resolve().parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import config
from config import MODELS, FEATURE_SET
from progress import write_progress


def _log(callback, msg):
    """Leitet Log-Nachricht an callback oder an print weiter."""
    if callback:
        callback(msg)
    else:
        print(msg)


def train(
    data_dir,
    labels_file,
    artifacts_dir,
    log_callback=None,
):
    """
    Trainiert Modelle. print()-Ausgaben von run.train() werden abgefangen und
    an log_callback weitergeleitet.

    Args:
        data_dir: Ordner mit CSV-Recordings
        labels_file: Pfad zur Label-Datei
        artifacts_dir: Ausgabe-Ordner für Modelle
        log_callback: Optionaler Callback für Log-Ausgaben

    Returns:
        (erfolg, ausgabe) – True/False und Log-Text
    """
    try:
        config.apply_overrides(
            data_dir=str(data_dir),
            labels_file=str(labels_file),
            artifacts_dir=str(artifacts_dir),
        )
        # Sofort Fortschritt zurücksetzen, damit Frontend nicht alte "done"-Datei sieht
        out_dir = Path(artifacts_dir)
        write_progress(out_dir, phase="starting", message="Starte Training...")
        from run import train as _train
        import io
        from contextlib import redirect_stdout
        # stdout umleiten, um print-Ausgaben abzufangen und an log_callback zu leiten
        buf = io.StringIO()
        with redirect_stdout(buf):
            _train()
        out = buf.getvalue()
        if log_callback:
            for line in out.splitlines():
                log_callback(line)
        return True, out
    except SystemExit as e:
        # run.train() wirft SystemExit bei Fehlern (z.B. keine Labels)
        return False, str(e) if e.code else "Unbekannter Fehler"
    except Exception as e:
        return False, str(e)


def predict(
    data_dir,
    test_labels_file,
    artifacts_dir,
    log_callback=None,
):
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
    try:
        config.apply_overrides(
            data_dir=str(data_dir),
            test_labels_file=str(test_labels_file),
            artifacts_dir=str(artifacts_dir),
        )
        # Sofort Fortschritt zurücksetzen, damit Frontend nicht alte "done"-Datei sieht
        out_dir = Path(artifacts_dir)
        write_progress(out_dir, phase="starting", message="Starte Vorhersage...")
        from predict import predict as _predict
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _predict()
        out = buf.getvalue()
        if log_callback:
            for line in out.splitlines():
                log_callback(line)
        # predict.py schreibt test_ergebnis_*.csv – diese einlesen und als dict zurückgeben
        import pandas as pd
        ergebnisse = {}
        for mdl in MODELS:
            csv_path = Path(artifacts_dir) / f"test_ergebnis_{mdl}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                ergebnisse[mdl] = df.to_dict("records")
        return True, out, ergebnisse
    except SystemExit as e:
        return False, str(e) if e.code else "Unbekannter Fehler", {}
    except Exception as e:
        return False, str(e), {}


def write_labels_file(file_paths, labels, output_path):
    """
    Schreibt Label-Datei (CSV mit File,Label) aus Liste von Dateipfaden und Labels.
    Die Labels stammen vom Aufrufer (z.B. aus einer bestehenden Label-Datei oder Nutzereingabe).

    Args:
        file_paths: Liste der CSV-Pfade
        labels: Liste der Fahrer-Labels (parallel zu file_paths, gleiche Länge)
        output_path: Zielpfad für die Label-Datei

    Returns:
        Anzahl geschriebener Einträge
    """
    if not file_paths:
        return 0
    if len(labels) != len(file_paths):
        raise ValueError("file_paths und labels müssen die gleiche Länge haben")
    lines = ["File,Label"]
    for fp, lbl in zip(file_paths, labels):
        p = Path(fp)
        # Absoluter Pfad für maximale Robustheit (data.load_labels unterstützt beides)
        lines.append(f"{p.resolve()},{str(lbl).strip()}")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return len(file_paths)


def get_config():
    """
    Liefert aktuelle Konfiguration für Voreinstellungen im GUI.

    Returns:
        dict mit data_dir, labels_file, test_labels_file, artifacts_dir, models, window_sec, step_sec
    """
    return {
        "data_dir": config.DATA_DIR,
        "labels_file": config.LABELS_FILE,
        "test_labels_file": config.TEST_LABELS_FILE,
        "artifacts_dir": config.ARTIFACTS_DIR,
        "models": config.MODELS,
        "window_sec": config.WINDOW_SEC,
        "step_sec": config.STEP_SEC,
    }
