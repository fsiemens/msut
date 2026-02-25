# -*- coding: utf-8 -*-
"""
Modul: data
===========
Datenladen und -aufbereitung für die Fahrererkennung. Liest CSV-Recordings
(Lenkrad, Gas, Bremse, Geschwindigkeit, Gierrate) und Label-Dateien, zerlegt
Zeitreihen in überlappende Sliding Windows und erstellt die Strukturen für
Featuretools (EntitySet) sowie TSFresh (Long-Format).

Wichtig: Fenster werden nach Recording gruppiert, damit bei Cross-Validation
keine Datenleckage entsteht (ganze Fahrten bleiben zusammen).

Hauptfunktionen:
    load_csv()        - Lädt eine Recording-CSV, parst rot_vel (x,y,z), sortiert Timestamps
    find_windows()    - Findet überlappende Fenster mit Mindestanzahl Punkten
    load_labels()     - Lädt Label-Datei (File,Label), unterstützt relative/absolute Pfade
    build_window_data - Baut Fenster- und Beobachtungs-Daten, resampelt auf MAX_POINTS
"""
import numpy as np
import pandas as pd
from pathlib import Path

from . import config


def load_csv(p):
    """
    Lädt eine Recording-CSV und gibt ein Dict mit sortierten Zeitreihen zurück.
    Zeile 2 (Einheiten) wird übersprungen. rot_vel wird geparst (Format "x,y,z").

    Args:
        p: Pfad zur CSV-Datei

    Returns:
        Dict mit Keys: t, steer, gas, brake, speed, yaw_rate
    """
    # Zeile 2 (Einheiten) überspringen; Bindestriche als fehlend markieren
    df = pd.read_csv(p, skiprows=[1], usecols=config.COLUMNS, low_memory=False, na_values=["-"])
    # Timestamps sortieren und ungültige Werte filtern
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    o = np.argsort(t)
    t = t[o]
    # rot_vel enthält oft "x,y,z" – wir extrahieren die Yaw-Rate (Index 1)
    rot = df["rot_vel"].astype(str).str.split(",", n=2, expand=True)
    yaw = pd.to_numeric(rot[1], errors="coerce").to_numpy(float)[o] if rot.shape[1] >= 2 else np.zeros_like(t)
    v = np.isfinite(t)  # Nur gültige Zeitstempel behalten
    return {"t": t[v], "steer": pd.to_numeric(df["wheel_position"], errors="coerce").to_numpy(float)[o][v],
            "gas": pd.to_numeric(df["car0_throttle_position"], errors="coerce").to_numpy(float)[o][v],
            "brake": pd.to_numeric(df["car0_brake_position"], errors="coerce").to_numpy(float)[o][v],
            "speed": pd.to_numeric(df["car0_velocity_vehicle"], errors="coerce").to_numpy(float)[o][v], "yaw_rate": yaw[v]}


def find_windows(t, ws=config.WINDOW_SEC, ss=config.STEP_SEC, mp=config.MIN_POINTS):
    """
    Findet überlappende Zeitfenster in der Timestamp-Reihe t.
    Jedes Fenster hat Länge ws Sekunden, Schrittweite ss, mindestens mp Punkte.

    Args:
        t: Timestamp-Array
        ws: Fensterlänge in Sekunden
        ss: Schrittweite in Sekunden
        mp: Mindestanzahl Punkte pro Fenster

    Returns:
        Liste von (start_idx, end_idx, start_time, end_time)
    """
    # Sliding Window: Start alle ss Sekunden, Fenster ws Sekunden, mindestens mp Punkte
    return [(int(np.searchsorted(t, s)), int(np.searchsorted(t, s + ws)), s, s + ws)
            for s in np.arange(float(t[0]), float(t[-1]) - ws + 1e-9, ss) if np.searchsorted(t, s + ws) - np.searchsorted(t, s) >= mp]


def load_labels(labels : str | Path | pd.DataFrame, training : bool = True, data_dir : str | Path = config.DATA_DIR):
    """
    Lädt Label-Datei (CSV mit File,Label). Unterstützt relative Pfade (relativ zu
    data_dir) und absolute Pfade.

    Args:
        labels_path: Pfad zur Label-Datei
        data_dir: Basis-Ordner für relative Pfade

    Returns:
        (paths, ids) – Liste der Dateipfade und zugehörige Fahrer-IDs
    """
    df = pd.read_csv(labels, sep=None, engine="python") if not isinstance(labels, pd.DataFrame) else labels 
    cols = {c.strip().lower(): c for c in df.columns}
    paths, ids = [], []
    for i, r in df.iterrows():
        f, l = r.get(cols["file"]), r.get(cols["label"])
        if pd.notna(f) and str(f).strip() and (not training or (pd.notna(l) and str(l).strip())):
            fp = Path(str(f).strip())
            # Absolute Pfade unverändert, relative werden mit data_dir zusammengesetzt
            paths.append(fp if fp.is_absolute() else data_dir / fp)
            if training:
                ids.append(str(l).strip().lower())  # Fahrer-ID aus Label-Spalte
            else:
                ids.append(i)  # Bei Vorhersage: Index als Platzhalter (Soll-Label optional)
    return paths, ids


def build_window_data(paths, ids):
    """
    Baut Fenster- und Beobachtungs-Daten für Featuretools/TSFresh. Jedes Fenster
    wird auf MAX_POINTS Punkte resampelt (gleichmäßige Indizes).

    Args:
        paths: Liste der CSV-Pfade
        ids: Liste der Fahrer-IDs (parallel zu paths)

    Returns:
        (window_rows, obs_rows) – DataFrame mit Fenstern, Liste von Dicts für EntitySet
    """
    window_rows, obs_rows, wid = [], [], 0
    for i, p in enumerate(paths):
        d = load_csv(p)
        for i0, i1, ws, we in find_windows(d["t"]):
            # Resampling: Fenster auf MAX_POINTS Punkte begrenzen (gleichmäßige Indizes)
            idx = np.linspace(0, i1 - i0 - 1, min(i1 - i0, config.MAX_POINTS), dtype=int)
            rel_t = d["t"][i0:i1][idx] - d["t"][i0]
            window_rows.append({"window_id": wid, "driver_id": ids[i], "recording": p.name})
            for j in range(len(rel_t)):
                obs_rows.append({"obs_id": f"{wid}_{j}", "window_id": wid, "time": float(rel_t[j]),
                    "steer": float(d["steer"][i0:i1][idx][j]), "gas": float(d["gas"][i0:i1][idx][j]),
                    "brake": float(d["brake"][i0:i1][idx][j]), "speed": float(d["speed"][i0:i1][idx][j]), "yaw_rate": float(d["yaw_rate"][i0:i1][idx][j])})
            wid += 1
    return pd.DataFrame(window_rows), obs_rows
