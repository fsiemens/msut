# projekt_mini – Fahrererkennung

Fahrererkennung anhand von Fahrsimulator-Daten. Das System klassifiziert den Fahrer (florian, matthias, fabian) aus CSV-Recordings mittels maschinellem Lernen (RandomForest, LogisticRegression).

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Installation](#2-installation)
3. [Schnellstart](#3-schnellstart)
4. [Projektstruktur](#4-projektstruktur)
5. [Nutzung](#5-nutzung)
6. [Datenformate](#6-datenformate)
7. [Konfiguration](#7-konfiguration)
8. [Grafiken (plots/)](#8-grafiken-plots)
9. [Weitere Dokumentation](#9-weitere-dokumentation)

---

## 1. Überblick

### Funktionen

| Funktion | Beschreibung |
|----------|--------------|
| **Training** | Trainiert RandomForest und LogisticRegression auf gelabelten Recordings |
| **Vorhersage** | Klassifiziert unbekannte Recordings mit trainierten Modellen |
| **Label-Verwaltung** | Erstellen und Lesen von Label-Dateien (File ↔ Fahrer) |
| **Pipeline-Fortschritt** | `pipeline_progress.json` für Frontend-Statusanzeige (Polling) |
| **Grafiken** | Konfusionsmatrix, Feature Importance, Accuracy, Vorhersage richtig/falsch |

### Technik

- **Features:** Featuretools (Aggregationen) + TSFresh (Zeitreihen-Features)
- **Modelle:** RandomForest, LogisticRegression
- **Validierung:** StratifiedGroupKFold (Recordings bleiben zusammen, keine Datenleckage)
- **Fenster:** 25 Sekunden, Schrittweite 12 Sekunden

---

## 2. Installation

### Voraussetzungen

- Python 3.10 oder höher

### Abhängigkeiten installieren

```bash
cd projekt_mini
pip install -r requirements.txt
```

### Abhängigkeiten (requirements.txt)

- pandas
- numpy
- featuretools
- tsfresh
- scikit-learn
- joblib
- matplotlib

---

## 3. Schnellstart

```bash
cd projekt_mini

# 1. Training (labels.lbl und data/ müssen vorhanden sein)
python run.py

# 2. Vorhersage (test_labels.lbl und trainierte Modelle erforderlich)
python predict.py
```

Alternativ mit `main.py` (falls vorhanden):

```bash
python main.py train
python main.py predict
```

---

## 4. Projektstruktur

```
projekt_mini/
├── config.py              # Zentrale Konfiguration
├── config.json            # Parametrisierte Einstellungen (optional)
├── data.py                # Datenladen, Fensterbildung, Labels
├── features.py            # Feature-Extraktion (Featuretools + TSFresh)
├── run.py                 # Trainings-Pipeline
├── predict.py             # Vorhersage-Pipeline
├── backend_api.py         # GUI-Schnittstelle
├── progress.py            # pipeline_progress.json für Frontend-Status
├── plots.py               # Grafiken (Konfusionsmatrix, Feature Importance, Accuracy, Vorhersage)
├── main.py                # Einstiegspunkt (train | predict)
├── labels.lbl             # Trainings-Labels
├── test_labels.lbl       # Test-Labels
├── data/                  # CSV-Recordings
├── artifacts/             # Modelle und Ergebnisse
│   ├── model_randomforest.joblib
│   ├── model_logreg.joblib
│   ├── ergebnis.txt
│   ├── pipeline_progress.json   # Fortschritt für Frontend-Polling
│   ├── test_ergebnis_*.csv
│   └── plots/             # Unterordner mit Grafiken
│       ├── confusion/     # Konfusionsmatrix pro Modell
│       ├── importance/    # Feature Importance pro Modell
│       ├── accuracy/      # Modell-Genauigkeit (Balkendiagramm)
│       └── prediction/    # Vorhersage richtig/falsch pro Modell
├── requirements.txt
├── README.md              # Diese Dokumentation
└── SCHNITTSTELLEN_BESCHREIBUNG.md   # API für Tkinter-Frontend
```

---

## 5. Nutzung

### CLI (run.py / predict.py)

**Training:**
```bash
python run.py [--data-dir DIR] [--labels FILE] [--artifacts DIR] [--config PATH] [--cv-splits N] [--random-state N]
```

**Vorhersage:**
```bash
python predict.py [--data-dir DIR] [--test-labels FILE] [--artifacts DIR] [--config PATH]
```

### Programm-API

```python
from run import train
from predict import predict

# Training
train(data_dir="data", labels_file="labels.lbl", artifacts_dir="artifacts")

# Vorhersage
predict(data_dir="data", test_labels_file="test_labels.lbl", artifacts_dir="artifacts")
```

### GUI-API (backend_api)

Für Tkinter- und andere GUI-Frontends siehe **SCHNITTSTELLEN_BESCHREIBUNG.md**.

```python
from backend_api import train, predict, write_labels_file, get_config

ok, msg = train(data_dir="data", labels_file="labels.lbl", artifacts_dir="artifacts")
ok, out, ergebnisse = predict(data_dir="data", test_labels_file="test_labels.lbl", artifacts_dir="artifacts")
```

---

## 6. Datenformate

### Label-Datei (labels.lbl, test_labels.lbl)

CSV mit Header, Trennzeichen Komma oder Tab:

```
File,Label
recording_2026_02_10__15_31_22_florian.csv,florian
recording_2026_02_10__15_38_03_matthias.csv,matthias
```

- **File:** Dateiname (relativ zu `data_dir`) oder absoluter Pfad
- **Label:** Fahrer-ID (florian, matthias, fabian oder andere)

### CSV-Recording

Erforderliche Spalten:

| Spalte | Beschreibung |
|--------|--------------|
| `timestamp` | Zeitstempel (Sekunden) |
| `wheel_position` | Lenkradposition |
| `car0_throttle_position` | Gaspedal |
| `car0_brake_position` | Bremspedal |
| `car0_velocity_vehicle` | Fahrzeuggeschwindigkeit |
| `rot_vel` | Rotationsgeschwindigkeit (Format "x,y,z"; Y = Gierrate) |

Die zweite Zeile kann Einheiten enthalten und wird übersprungen.

---

## 7. Konfiguration

### config.json

Optional im Projekt-Root. Beispiel:

```json
{
  "data_dir": "data",
  "labels_file": "labels.lbl",
  "test_labels_file": "test_labels.lbl",
  "artifacts_dir": "artifacts",
  "models": ["randomforest", "logreg"],
  "feature_set": "both",
  "window_sec": 25,
  "step_sec": 12,
  "min_points": 300,
  "max_points": 500,
  "cv_splits": 5,
  "random_state": 42
}
```

### Wichtige Parameter

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `window_sec` | 25 | Fenstergröße in Sekunden |
| `step_sec` | 12 | Schrittweite in Sekunden |
| `feature_set` | "both" | "featuretools" \| "tsfresh" \| "both" |
| `cv_splits` | 5 | Folds für Cross-Validation |

---

## 8. Grafiken (plots/)

Nach Training und Vorhersage werden Grafiken in `artifacts/plots/` erzeugt:

| Ordner | Inhalt |
|--------|--------|
| `confusion/` | Konfusionsmatrix pro Modell (`confusion_randomforest.png`, `confusion_logreg.png`) |
| `importance/` | Feature Importance (Top 30) pro Modell |
| `accuracy/` | Balkendiagramm der Modell-Genauigkeiten |
| `prediction/` | Grafik: Vorhersagen richtig (grün) / falsch (rot) pro Recording |

---

## 9. Weitere Dokumentation

| Datei | Inhalt |
|-------|--------|
| **SCHNITTSTELLEN_BESCHREIBUNG.md** | API für Tkinter-Frontend (train, predict, write_labels_file, get_config, pipeline_progress.json) |
| **config.py**, **data.py**, etc. | Modul-Docstrings und Code-Kommentare |

---

## Fehlerbehandlung

- **CLI:** Bei Fehlern wird `SystemExit` mit Fehlermeldung geworfen
- **backend_api:** Gibt `(False, fehlermeldung)` zurück statt Exceptions

Typische Fehler:
- Keine gültigen Labels gefunden
- Label-Datei nicht gefunden
- Modelle nicht gefunden (vor predict muss train ausgeführt worden sein)
- Ungültige oder fehlende CSV-Spalten
