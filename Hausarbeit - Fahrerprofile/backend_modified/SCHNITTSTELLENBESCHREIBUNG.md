# Schnittstellenbeschreibung – Minimale Pipeline (Fahreridentifikation)

**Zielgruppe:** Frontend-Entwickler, die ein Dashboard/UI für die minimale Pipeline erstellen oder die Pipeline als Subprocess ansteuern.

**Version:** 1.0  
**Stand:** 2026

---

## 1. Übersicht

Die minimale Pipeline (`pipeline_minimal_beispiel/run.py`) ist eine **eigenständige** Implementierung der Fahreridentifikation ohne Abhängigkeit von `pipeline_project`. Sie unterstützt:

- Feature-Extraktion aus Rohdaten (FSChatGPT, FSGemini, auto, featuretools)
- Modell-Training und Evaluation (StratifiedGroupKFold oder LOO)
- Holdout-Vorhersage auf separaten Recordings
- Hyperparameter-Optimierung (GridSearchCV)
- Fortschritts-Polling über JSON
- Plots in thematischen Unterordnern

**Arbeitsverzeichnis:** Bei Aufruf aus dem Projekt-Root: `python pipeline_minimal_beispiel/run.py`. Bei Aufruf aus `pipeline_minimal_beispiel/`: `python run.py`. Pfade sind relativ zum Pipeline-Ordner (`pipeline_minimal_beispiel/`).

---

## 2. Pipeline-Start (CLI)

### 2.1 Befehl

```
python pipeline_minimal_beispiel/run.py [Optionen]
```

**Alternativ** (aus pipeline_minimal_beispiel/):

```
python run.py [Optionen]
```

### 2.2 Alle CLI-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `--data-dir` | Path | `data` | Ordner mit Recording-CSVs. Bei `--labels`: Basisverzeichnis für relative Pfade in der LBL-Datei. Relativ zu pipeline_minimal_beispiel/. |
| `--labels` | Path | – | Pfad zur LBL-Datei (CSV: File, Label). Bei `--extract-from-raw` werden nur die gelisteten Dateien für Training verwendet. Zeilen mit leerem Label werden übersprungen. |
| `--holdout-file` | Path | – | Pfad zur Holdout-Datei (CSV: File, Label). Diese Recordings werden vom Training ausgeschlossen; nach dem Training wird eine Vorhersage erstellt. Label optional. Erfordert `--extract-from-raw`. |
| `--extract-from-raw` | Flag | – | Features aus Rohdaten extrahieren. Ohne dieses Flag werden nur gecachte Features aus `artifacts/features/` verwendet. |
| `--feature-sets` | Liste | FSChatGPT | Feature-Sätze: FSChatGPT, FSGemini, auto, featuretools, merged_all, selected. Mehrere durch Leerzeichen getrennt. |
| `--models` | Liste | logreg, extratrees | Modelle: extratrees, randomforest, logreg, svm_rbf, nearest_centroid. |
| `--with-merged` | Flag | – | Baut merged_all aus den Basis-Sätzen. |
| `--with-selected` | Flag | – | Baut selected (Top-K) aus merged_all. Erfordert `--with-merged`. |
| `--skip-featuretools` | Flag | – | Überspringt die Featuretools-Extraktion. |
| `--loo` | Flag | – | Leave-One-Recording-Out statt K-Fold. |
| `--force` | Flag | – | Cache ignorieren, alles neu extrahieren. |
| `--n-splits` | int | 5 | Folds für StratifiedGroupKFold. |
| `--n-jobs` | int | 1 | Parallele Jobs für Modell-Training (1=sequentiell, -1=alle Kerne). |
| `--tune-models` | str | – | Komma-getrennte Modellnamen für GridSearchCV (z. B. `logreg,svm_rbf`). |
| `--model-params` | str | – | JSON-Dict mit Modell-Parametern (z. B. `{"logreg":{"C":10}}`). |
| `--out-dir` | Path | `artifacts` | Ausgabeverzeichnis (relativ zu pipeline_minimal_beispiel/). |
| `--no-plots` | Flag | – | Keine Grafiken erzeugen. |
| `--window-s` | float | 20.0 | Fensterlänge in Sekunden. |
| `--step-s` | float | 10.0 | Fenster-Versatz in Sekunden. |
| `--min-samples` | int | 300 | Mindestanzahl Samples pro Fenster. |
| `--drop-nan-col-thresh` | float | 0.7 | Spalten mit mehr als 70 % NaN werden entfernt. |
| `--seed` | int | 42 | Zufallsseed. |
| `--top-k-importance` | int | 20 | Top-K Features für Importance-Plots. |
| `--selected-top-k` | int | 60 | Top-K für selected-Feature-Set. |

### 2.3 Beispiel-Befehle

```bash
# Schnellstart mit Cache
python pipeline_minimal_beispiel/run.py --feature-sets FSChatGPT --models logreg

# Rohdaten-Extraktion + Training
python pipeline_minimal_beispiel/run.py --extract-from-raw --feature-sets FSChatGPT --models logreg

# Mit LBL-Datei und Holdout
python pipeline_minimal_beispiel/run.py --extract-from-raw --data-dir . --labels artifacts/training_labels.lbl --holdout-file artifacts/holdout_test.lbl --feature-sets FSChatGPT --models logreg

# Vollständige Evaluation: LOO, alle Modelle, Hyperparameter-Tuning
python pipeline_minimal_beispiel/run.py --extract-from-raw --loo --feature-sets FSChatGPT FSGemini --models extratrees randomforest logreg svm_rbf nearest_centroid --tune-models extratrees,randomforest,logreg,svm_rbf,nearest_centroid --n-jobs 16
```

---

## 3. Eingabeformate

### 3.1 LBL-Datei (Label-Datei)

**Format:** CSV mit Spalten `File`, `Label` (Komma oder Semikolon).

```
File,Label
data/recording_2026_02_10__13_10_22_fabian.csv,Fabian
data/recording_2026_02_10__13_18_02_florian.csv,Florian
data/recording_2026_02_10__13_25_22_matthias.csv,Matthias
```

- **File:** Relativer Pfad zur CSV-Datei (Basis: `--data-dir`).
- **Label:** Fahrer-ID (z. B. Fabian, Florian, Matthias). Zeilen mit leerem Label werden übersprungen.
- Nur Zeilen mit gültigem Label fließen ins Training.

### 3.2 Holdout-Datei

**Format:** Wie LBL-Datei (File, Label).

```
File,Label
artifacts/holdout/recording_2026_02_10__13_37_56_florian.csv,Florian
artifacts/holdout/recording_2026_02_10__13_44_54_matthias.csv,Matthias
artifacts/holdout/recording_2026_02_10__13_51_14_fabian.csv,Fabian
```

- **File:** Relativer Pfad (Basis: `--data-dir`). Kann auf Recordings außerhalb des Trainings-Ordners verweisen (z. B. `artifacts/holdout/`).
- **Label:** Optional. Mit Label: Evaluation möglich (predicted vs. true). Ohne Label: Nur Vorhersage.

---

## 4. Fortschritts-Schnittstelle (Polling)

### 4.1 Datei

**Pfad:** `pipeline_minimal_beispiel/artifacts/pipeline_progress.json`  
(bzw. `{out_dir}/pipeline_progress.json`)

Das Frontend soll alle 1–2 Sekunden lesen (Polling), bis der Subprocess beendet ist.

### 4.2 JSON-Schema

```json
{
  "phase": "training",
  "total": 10,
  "completed": ["FSChatGPT | logreg", "FSChatGPT | extratrees"],
  "in_progress": ["FSGemini | logreg"],
  "message": "Aktuell: FSGemini | logreg",
  "remaining": 7,
  "percent": 20.0
}
```

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| phase | string | `starting`, `extraction`, `preparing`, `training`, `holdout`, `done`. |
| total | int | Gesamtanzahl der (Feature-Set × Modell)-Kombinationen. |
| completed | array | Abgeschlossene Kombinationen (Format: `"dataset | model"`). |
| in_progress | array | Aktuell laufende Kombination(en). |
| message | string | Kurzbeschreibung des aktuellen Schritts. |
| remaining | int | Noch offene Kombinationen. |
| percent | float | Fortschritt in Prozent (0–100). |

### 4.3 Phasen

| phase | Bedeutung |
|-------|-----------|
| starting | Pipeline wird initialisiert. |
| extraction | Feature-Extraktion aus Rohdaten. |
| preparing | Feature-Sets aus Cache laden. |
| training | Modell-Training und Evaluation. |
| holdout | Holdout-Vorhersage (bei --holdout-file). |
| done | Pipeline abgeschlossen. |

---

## 5. Ausgabe-Schnittstellen

### 5.1 Metriken (CSV)

**Pfad:** `pipeline_minimal_beispiel/artifacts/metrics_summary.csv`

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| dataset | string | Feature-Set (z. B. FSChatGPT, FSGemini). |
| model | string | Modellname (z. B. logreg, extratrees). |
| train_windows | int | Anzahl Trainings-Fenster. |
| n_recordings | int | Anzahl Recordings. |
| n_features | int | Anzahl Merkmale. |
| window_accuracy | float | Fenster-Level-Accuracy (0–1). |
| recording_accuracy | float | Recording-Level-Accuracy (0–1), primäre Metrik. |
| recording_f1 | float | F1-Score (Recording-Level). |
| recording_precision | float | Precision (Recording-Level). |
| recording_recall | float | Recall (Recording-Level). |
| model_file | string | Pfad zur gespeicherten Modell-Datei. |

### 5.2 Reports (Markdown)

| Datei | Pfad | Beschreibung |
|-------|------|--------------|
| modellvergleich_uebersicht.md | artifacts/ | Vollständiger Report mit Konfiguration, bester Kombination, Tabelle, eingebetteten Grafiken. |
| run_report.md | artifacts/ | Identisch, für Kompatibilität. |

### 5.3 Plots (PNG) – Unterordner-Struktur

**Basisverzeichnis:** `pipeline_minimal_beispiel/artifacts/plots/`

```
plots/
├── summary/           # Übersichts-Grafiken
│   ├── accuracy_summary.png
│   ├── accuracy_heatmap.png
│   ├── f1_precision_recall_summary.png
│   └── metrics_heatmap_f1_precision_recall.png
├── confusion/         # Confusion Matrices
│   └── confusion_<dataset>_<model>.png
├── metrics/           # F1/Precision/Recall pro Kombination
│   └── metrics_<dataset>_<model>.png
├── importance/        # Feature-Importance
│   └── <modell>/
│       └── <dataset>.png
├── model_specific/    # Modell-spezifische Visualisierungen
│   └── model_specific_<dataset>_<model>.png
├── feature_correlation/
│   └── feature_correlation_<dataset>.png
├── raw/               # Nur bei --extract-from-raw
│   └── raw_sensor_correlation.png
└── loo/               # Nur bei --loo
    ├── loo_recording_heatmap.png
    └── loo_accuracy_per_combo.png
```

| Pfad | Inhalt |
|------|--------|
| summary/accuracy_summary.png | Balkendiagramm Recording-Accuracy pro Modell und Feature-Set. |
| summary/accuracy_heatmap.png | Heatmap Recording-Accuracy (Dataset × Modell). |
| summary/f1_precision_recall_summary.png | F1, Precision, Recall pro Kombination. |
| summary/metrics_heatmap_f1_precision_recall.png | Heatmap F1/Precision/Recall. |
| confusion/confusion_<dataset>_<model>.png | Confusion Matrix. |
| metrics/metrics_<dataset>_<model>.png | F1, Precision, Recall pro Kombination. |
| importance/<modell>/<dataset>.png | Feature-Importance (Top-K). |
| model_specific/model_specific_<dataset>_<model>.png | Baum, Koeffizienten, Entscheidungsgrenze. |
| feature_correlation/feature_correlation_<dataset>.png | Korrelationsmatrix der Features. |
| raw/raw_sensor_correlation.png | Korrelation der Rohdaten-Sensoren. |
| loo/loo_recording_heatmap.png | LOO: Recording × Kombination. |
| loo/loo_accuracy_per_combo.png | LOO: Accuracy pro Kombination. |

### 5.4 LOO-Ergebnisse (bei --loo)

**Verzeichnis:** `artifacts/loo_per_recording/`

**Dateien:** `{dataset}_{model}.csv` mit Spalten `group_id`, `y_true`, `y_pred`, `correct`.

### 5.5 Holdout-Ergebnisse (bei --holdout-file)

**Pfad:** `artifacts/holdout/holdout_predictions.csv`

**Spalten:** `dataset`, `model`, `recording`, `predicted`, `true`, `correct`.

### 5.6 Modelle (joblib)

**Verzeichnis:** `artifacts/models/<modell>/`

**Dateiname:** `{dataset}_{model}.joblib` (z. B. `FSChatGPT_logreg.joblib`)

---

## 6. Ablaufdiagramm (Frontend-Perspektive)

```
1. Benutzer wählt Optionen (Feature-Sets, Modelle, extract-from-raw, labels, holdout, …)
2. Frontend startet Subprocess:
   python pipeline_minimal_beispiel/run.py [Optionen]
   (Arbeitsverzeichnis: Projekt-Root)
3. Polling-Schleife (alle 1–2 s):
   - Lese pipeline_minimal_beispiel/artifacts/pipeline_progress.json
   - Zeige phase, message, completed, in_progress, remaining, percent (Progress-Bar)
   - Wenn Subprocess beendet: Schleife verlassen
4. Bei Erfolg (Exit-Code 0):
   - Lese artifacts/metrics_summary.csv (sortiert nach recording_accuracy)
   - Zeige Plots aus artifacts/plots/ (Unterordner: summary/, confusion/, …)
   - Zeige modellvergleich_uebersicht.md
   - Optional: LOO-CSVs, holdout_predictions.csv
5. Bei Fehler (Exit-Code ≠0):
   - Zeige stderr/stdout des Subprocess
```

---

## 7. Technische Hinweise

- **Keine parallelen Worker:** Im Gegensatz zur Haupt-Pipeline gibt es kein `--n-workers`. Parallele Verarbeitung erfolgt nur über `--n-jobs` (intern in ExtraTrees, RandomForest, LogReg).
- **Pfade:** Alle Pfade in dieser Beschreibung sind relativ zu `pipeline_minimal_beispiel/` oder zum Projekt-Root, je nach Kontext.
- **Encoding:** JSON und CSV sind UTF-8.
- **Exit-Code:** 0 = Erfolg, ≠0 = Fehler. Das Frontend sollte den Subprocess-Exit-Code auswerten.

---

## Anhang: Referenz – pipeline_progress.json (Beispiele)

**Extraktion:**

```json
{
  "phase": "extraction",
  "total": 0,
  "completed": [],
  "in_progress": [],
  "message": "Extrahiere FSChatGPT...",
  "remaining": 0,
  "percent": 0.0
}
```

**Training (laufend):**

```json
{
  "phase": "training",
  "total": 5,
  "completed": ["FSChatGPT | logreg", "FSChatGPT | extratrees"],
  "in_progress": ["FSGemini | logreg"],
  "message": "Aktuell: FSGemini | logreg",
  "remaining": 2,
  "percent": 40.0
}
```

**Fertig:**

```json
{
  "phase": "done",
  "total": 5,
  "completed": ["FSChatGPT | logreg", "FSChatGPT | extratrees", "FSGemini | logreg", "FSGemini | extratrees", "FSGemini | svm_rbf"],
  "in_progress": [],
  "message": "Fertig",
  "remaining": 0,
  "percent": 100.0
}
```
