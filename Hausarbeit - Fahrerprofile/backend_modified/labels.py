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


def loadLabelFile(
    path: Path,
    baseDir: Path | None = None,
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
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Label-Datei nicht gefunden: {path}")

    base = baseDir.resolve() if baseDir else path.parent

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Label-Datei konnte nicht gelesen werden: {e}") from e

    cols = {c.strip().lower(): c for c in df.columns}
    if "file" not in cols or "label" not in cols:
        raise ValueError(
            f"Label-Datei muss Spalten 'File' und 'Label' haben. Gefunden: {list(df.columns)}"
        )
    fileCol = cols["file"]
    labelCol = cols["label"]

    csvPaths: list[Path] = []
    driverIds: list[str] = []

    for _, row in df.iterrows():
        file = row.get(fileCol)
        label = row.get(labelCol)
        if pd.isna(file) or str(file).strip() == "":
            continue
        if pd.isna(label) or str(label).strip() == "":
            continue  # Zeilen ohne Label überspringen
        filePath = Path(str(file).strip())
        if not filePath.is_absolute():
            filePath = base / filePath
        filePath = filePath.resolve()
        if not filePath.exists():
            raise FileNotFoundError(f"Recording nicht gefunden: {filePath}")
        csvPaths.append(filePath)
        driverIds.append(str(label).strip().lower())

    if not csvPaths:
        raise ValueError(
            f"LBL-Datei enthält keine Zeilen mit gültigem Label: {path}"
        )

    return csvPaths, driverIds


def loadHoldoutFile(
    path: Path,
    baseDir: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """
    Lädt eine Holdout-Datei (gleiches Format wie LBL: File, Label).

    Im Gegensatz zu load_labels_file werden ALLE Zeilen zurückgegeben.
    Zeilen mit leerem Label erhalten driver_id="__unlabeled__" (für Evaluation übersprungen).

    Returns:
        (csv_paths, driver_ids)
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Holdout-Datei nicht gefunden: {path}")

    base = baseDir.resolve() if baseDir else path.parent

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Holdout-Datei konnte nicht gelesen werden: {e}") from e

    cols = {c.strip().lower(): c for c in df.columns}
    if "file" not in cols or "label" not in cols:
        raise ValueError(
            f"Holdout-Datei muss Spalten 'File' und 'Label' haben. Gefunden: {list(df.columns)}"
        )
    fileCol = cols["file"]
    labelCol = cols["label"]

    csv_paths: list[Path] = []
    driver_ids: list[str] = []

    for _, row in df.iterrows():
        file = row.get(fileCol)
        label = row.get(labelCol)
        if pd.isna(file) or str(file).strip() == "":
            continue
        fpath = Path(str(file).strip())
        if not fpath.is_absolute():
            fpath = base / fpath
        fpath = fpath.resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Holdout-Recording nicht gefunden: {fpath}")
        csv_paths.append(fpath)
        if pd.isna(label) or str(label).strip() == "":
            driver_ids.append("__unlabeled__")  # Platzhalter für Evaluation (wird übersprungen)
        else:
            driver_ids.append(str(label).strip().lower())

    if not csv_paths:
        raise ValueError(f"Holdout-Datei enthält keine gültigen Zeilen: {path}")

    return csv_paths, driver_ids


def saveLabelTemplate(out_path: Path, csv_paths: list[Path], driver_ids: list[str] | None = None) -> None:
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
