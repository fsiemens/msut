"""
LBL-Datei (Label-Datei) für die minimale Pipeline.

Format (CSV mit Semikolon oder Komma):
  File,Label
  recording_2026_02_10__13_10_22_fabian.csv,Fabian
  recording_2026_02_10__13_18_02_florian.csv,Florian
  recording_2026_02_10__13_37_56_florian.csv,

Zeilen mit leerem Label werden ausgelassen (nur gelabelte Recordings für Training).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labels_file(
    lbl_path: Path,
    base_dir: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """
    Lädt eine LBL-Datei und gibt (csv_paths, driver_ids) zurück.

    Args:
        lbl_path: Pfad zur LBL-Datei (CSV mit Spalten File, Label).
        base_dir: Basisverzeichnis für relative Dateipfade. Wenn None, wird das
                  Verzeichnis der LBL-Datei verwendet.

    Returns:
        (csv_paths, driver_ids) – nur Zeilen mit nicht-leerem Label.
        driver_ids werden in Kleinbuchstaben normalisiert.

    Raises:
        FileNotFoundError: Wenn lbl_path nicht existiert.
        ValueError: Wenn die LBL-Datei ungültig ist oder keine gültigen Zeilen hat.
    """
    lbl_path = Path(lbl_path).resolve()
    if not lbl_path.exists():
        raise FileNotFoundError(f"LBL-Datei nicht gefunden: {lbl_path}")

    base = base_dir.resolve() if base_dir else lbl_path.parent

    # CSV einlesen (Komma oder Semikolon)
    try:
        df = pd.read_csv(lbl_path, sep=None, engine="python", encoding="utf-8")
    except Exception as e:
        raise ValueError(f"LBL-Datei konnte nicht gelesen werden: {e}") from e

    # Spalten normalisieren (File/Label, file/label)
    cols = {c.strip().lower(): c for c in df.columns}
    if "file" not in cols or "label" not in cols:
        raise ValueError(
            f"LBL-Datei muss Spalten 'File' und 'Label' haben. Gefunden: {list(df.columns)}"
        )
    file_col = cols["file"]
    label_col = cols["label"]

    csv_paths: list[Path] = []
    driver_ids: list[str] = []

    for _, row in df.iterrows():
        f = row.get(file_col)
        lbl = row.get(label_col)
        if pd.isna(f) or str(f).strip() == "":
            continue
        if pd.isna(lbl) or str(lbl).strip() == "":
            continue  # Zeilen ohne Label überspringen
        fpath = Path(str(f).strip())
        if not fpath.is_absolute():
            fpath = base / fpath
        fpath = fpath.resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Recording nicht gefunden: {fpath}")
        csv_paths.append(fpath)
        driver_ids.append(str(lbl).strip().lower())

    if not csv_paths:
        raise ValueError(
            f"LBL-Datei enthält keine Zeilen mit gültigem Label: {lbl_path}"
        )

    return csv_paths, driver_ids


def load_holdout_file(
    lbl_path: Path,
    base_dir: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """
    Lädt eine Holdout-Datei (gleiches Format wie LBL: File, Label).

    Im Gegensatz zu load_labels_file werden ALLE Zeilen zurückgegeben.
    Zeilen mit leerem Label erhalten driver_id="__unlabeled__" (für Evaluation übersprungen).

    Returns:
        (csv_paths, driver_ids)
    """
    lbl_path = Path(lbl_path).resolve()
    if not lbl_path.exists():
        raise FileNotFoundError(f"Holdout-Datei nicht gefunden: {lbl_path}")

    base = base_dir.resolve() if base_dir else lbl_path.parent

    try:
        df = pd.read_csv(lbl_path, sep=None, engine="python", encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Holdout-Datei konnte nicht gelesen werden: {e}") from e

    cols = {c.strip().lower(): c for c in df.columns}
    if "file" not in cols or "label" not in cols:
        raise ValueError(
            f"Holdout-Datei muss Spalten 'File' und 'Label' haben. Gefunden: {list(df.columns)}"
        )
    file_col = cols["file"]
    label_col = cols["label"]

    csv_paths: list[Path] = []
    driver_ids: list[str] = []

    for _, row in df.iterrows():
        f = row.get(file_col)
        lbl = row.get(label_col)
        if pd.isna(f) or str(f).strip() == "":
            continue
        fpath = Path(str(f).strip())
        if not fpath.is_absolute():
            fpath = base / fpath
        fpath = fpath.resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Holdout-Recording nicht gefunden: {fpath}")
        csv_paths.append(fpath)
        if pd.isna(lbl) or str(lbl).strip() == "":
            driver_ids.append("__unlabeled__")  # Platzhalter für Evaluation (wird übersprungen)
        else:
            driver_ids.append(str(lbl).strip().lower())

    if not csv_paths:
        raise ValueError(f"Holdout-Datei enthält keine gültigen Zeilen: {lbl_path}")

    return csv_paths, driver_ids


def save_labels_template(out_path: Path, csv_paths: list[Path], driver_ids: list[str] | None = None) -> None:
    """
    Speichert eine LBL-Datei (z. B. als Vorlage oder nach manueller Auswahl).

    Args:
        out_path: Ausgabepfad für die LBL-Datei.
        csv_paths: Liste der CSV-Pfade (Dateinamen oder relative Pfade).
        driver_ids: Optionale Labels. Wenn None, werden leere Labels geschrieben.
    """
    n = len(csv_paths)
    labels = driver_ids if driver_ids and len(driver_ids) == n else [""] * n
    df = pd.DataFrame({"File": [Path(p).name for p in csv_paths], "Label": labels})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=",")
