# Minimales Pipeline-Beispiel

Vereinfachte Referenzimplementierung der Fahreridentifikations-Pipeline. **Komplett eigenständig** – benötigt kein pipeline_project.

## Nutzung

1. **Abhängigkeiten installieren** (im Projektroot):
   ```bash
   pip install -r requirements.txt
   ```

2. **Recording-CSVs** in `data/` ablegen (oder `--data-dir` auf einen anderen Ordner zeigen lassen).

3. **Pipeline starten** (vom Projektroot):
   ```bash
   python pipeline_minimal_beispiel/run.py --extract-from-raw --feature-sets FSChatGPT --models logreg
   ```

   Oder aus dem Ordner `pipeline_minimal_beispiel/`:
   ```bash
   cd pipeline_minimal_beispiel
   python run.py --extract-from-raw --feature-sets FSChatGPT --models logreg
   ```

## Zweck

- Gleiche CLI-Argumente wie die Haupt-Pipeline
- Gleiche Ausgabedateien: `pipeline_progress.json`, `metrics_summary.csv`, `modellvergleich_uebersicht.md`, `plots/*`, `models/*`
- **Vereinfachter Code:** run.py, extraction, toolkit, plots, model_plots, evaluate, report, progress, labels – alles eigenständig in pipeline_minimal_beispiel/.

## CLI-Ausführung (Standalone)

Aus dem Ordner `pipeline_minimal_beispiel/`:

```bash
# Nur FSChatGPT + logreg (lädt aus Cache)
python run.py --feature-sets FSChatGPT --models logreg

# Mit Rohdaten-Extraktion (Daten aus data/)
python run.py --extract-from-raw --feature-sets FSChatGPT --models logreg

# Mit LBL-Datei (ausgewählte Dateien + Labels)
python run.py --extract-from-raw --labels artifacts/recording_labels.lbl --data-dir data --feature-sets FSChatGPT --models logreg

# Mit Holdout (Recordings für Vorhersage ohne Training)
python run.py --extract-from-raw --labels artifacts/recording_labels.lbl --holdout-file artifacts/recording_holdout.lbl --data-dir data --feature-sets FSChatGPT --models logreg

# Mit LOO, merged_all, selected
python run.py --extract-from-raw --with-merged --with-selected --loo --feature-sets FSChatGPT FSGemini --models logreg extratrees
```

**Alternativ** vom Projektroot aus:
```bash
python pipeline_minimal_beispiel/run.py --extract-from-raw --feature-sets FSChatGPT --models logreg
```

## Unterstützte Optionen

| Option | Beschreibung |
|--------|--------------|
| `--feature-sets` | FSChatGPT, FSGemini, auto, featuretools, merged_all, selected |
| `--models` | logreg, extratrees, randomforest, svm_rbf, nearest_centroid |
| `--extract-from-raw` | Features aus Rohdaten extrahieren |
| `--labels` | Pfad zur LBL-Datei (CSV: File, Label). Nur gelistete Dateien; leere Labels werden übersprungen. |
| `--holdout-file` | Pfad zur Holdout-Datei. Recordings vom Training ausschließen; nach Training Vorhersage erstellen. |
| `--data-dir` | Ordner mit recording_*.csv (ohne --labels) oder Basisverzeichnis für Pfade in der LBL-Datei |
| `--with-merged` | merged_all aus Basis-Sätzen bauen |
| `--with-selected` | selected (Top-K) aus merged_all |
| `--skip-featuretools` | Featuretools-Extraktion überspringen |
| `--loo` | Leave-One-Recording-Out |
| `--force` | Cache ignorieren |
| `--n-splits` | Folds für K-Fold |
| `--n-jobs` | Parallele Jobs für Modell-Training (1=sequentiell, -1=alle Kerne; ExtraTrees, RandomForest, LogReg) |
| `--out-dir` | Ausgabeverzeichnis (Standard: artifacts, relativ zum Pipeline-Ordner) |
| `--no-plots` | Keine Grafiken erzeugen |

| `--tune-models` | Komma-getrennte Modellnamen für GridSearchCV (z.B. logreg,svm_rbf) |
| `--model-params` | JSON mit Modell-Parametern pro Modell |

**Nicht unterstützt (nur in Haupt-Pipeline):** `--n-workers`, `--custom-feature-path`

## Modulstruktur

Die Pipeline ist in mehrere Module aufgeteilt:

| Modul | Aufgabe |
|-------|---------|
| `run.py` | Orchestrierung, CLI, Hauptschleife |
| `progress.py` | Fortschritts-JSON für Frontend |
| `labels.py` | LBL-Datei laden (File, Label); Zeilen ohne Label überspringen |
| `extraction.py` | Feature-Laden aus Cache oder Extraktion aus Rohdaten (bei --labels nur gelistete Dateien) |
| `evaluate.py` | CV, Modell-Training, Plots pro Kombination |
| `report.py` | Reports, Summary-Plots, LOO-Dateien |

## Ausgabe

Schreibt in `artifacts/` (oder `--out-dir`):

- `pipeline_progress.json` – Fortschritt für Frontend-Polling
- `metrics_summary.csv` – Metriken aller Kombinationen
- `modellvergleich_uebersicht.md`, `run_report.md` – Reports
- `plots/` – Accuracy-Summary, Heatmaps, Confusion Matrix, Feature-Importance, Modell-spezifisch, Feature-Korrelation, LOO-Plots
- `models/<modell>/<dataset>_<modell>.joblib` – gespeicherte Modelle
- `features/` – Feature-CSVs (bei Extraktion aus Rohdaten)
- `loo_per_recording/` – LOO-Details pro Kombination (bei --loo)
- `holdout/` – Holdout-Vorhersagen (bei --holdout-file)
