# Dokumentation – Minimales Pipeline-Beispiel (Fahreridentifikation)

Diese Dokumentation beschreibt das **minimale studentische Beispiel** der Fahreridentifikations-Pipeline. Sie ist analog zur Hauptdokumentation (HAUSARBEIT_INHALT_WORD.md) aufgebaut und kann direkt in die Hausarbeit übernommen werden.

---

## 1. Zweck und Abgrenzung

Das minimale Pipeline-Beispiel (`pipeline_minimal_beispiel/`) dient als **vereinfachte, nachvollziehbare Referenzimplementierung** der Fahreridentifikation. Es ist **komplett eigenständig** und benötigt **kein pipeline_project**. Alle benötigten Module (Extraktion, Toolkit, Plots, Modell-Plots) liegen im Ordner `pipeline_minimal_beispiel/`.

**Abgrenzung zur Haupt-Pipeline:** Die Haupt-Pipeline unterstützt parallele Worker (`--n-workers`), benutzerdefinierte Feature-Pfade (`--custom-feature-path`) und erweiterte Konfigurationsoptionen. Die minimale Pipeline verzichtet bewusst auf diese Erweiterungen, um den Code überschaubar zu halten. Sie speichert in ein **eigenes Ausgabeverzeichnis** (`artifacts/`), sodass Ergebnisse nicht mit denen der Haupt-Pipeline vermischt werden.

**Zielgruppe:** Studierende, die die Pipeline-Struktur verstehen und den Ablauf von Extraktion über Modell-Training bis zur Auswertung nachvollziehen möchten.

---

## 2. Modulstruktur und Architektur

Die minimale Pipeline ist in mehrere Module aufgeteilt, um die Verantwortlichkeiten klar zu trennen.

### 2.1 Modulübersicht

| Modul | Aufgabe |
|-------|---------|
| **run.py** | Haupt-Script und Orchestrierung: Parst CLI-Argumente (Feature-Sets, Modelle, n_splits, LOO etc.), legt Ausgabeverzeichnisse an, ruft Feature-Laden bzw. Extraktion auf, steuert die Schleife über alle Feature-Set/Modell-Kombinationen und übergibt die Ergebnisse an report.write_reports_and_plots. |
| **progress.py** | Schreibt die Fortschritts-JSON (`pipeline_progress.json`) für das Frontend-Polling. Enthält phase, completed, in_progress und message – das Streamlit-Dashboard liest diese Datei, um den aktuellen Pipeline-Status anzuzeigen. |
| **labels.py** | Lädt LBL-Dateien (CSV mit File, Label): Liest die Zuordnung von Recording-Dateien zu Fahrer-Labels. Zeilen mit leerem Label werden übersprungen. |
| **extraction.py** | Vollständige Feature-Extraktion. Lädt Feature-CSVs aus dem Cache (`artifacts/features/`) oder extrahiert sie bei `--extract-from-raw` aus den Roh-CSVs. Bei `--labels` werden nur die in der LBL-Datei gelisteten Dateien verwendet. Eigenständig. |
| **evaluate.py** | Führt die Evaluation einer einzelnen Feature-Set/Modell-Kombination durch: StratifiedGroupKFold oder LOO, Modell-Training, Speicherung des Modells als joblib, Erzeugung der pro-Kombination-Plots (Confusion Matrix, Feature-Importance, Modell-spezifische Visualisierung). |
| **report.py** | Erstellt die Markdown-Reports (`modellvergleich_uebersicht.md`, `run_report.md`), schreibt `metrics_summary.csv`, erzeugt die Summary-Plots (Accuracy-Balken, Heatmaps, F1/Precision/Recall) und speichert bei LOO die Zuordnungen in `loo_per_recording/`. |

### 2.2 Datenfluss

1. **CLI / Konfiguration:** `run.py` parst Argumente und legt Ausgabeverzeichnisse an.
2. **Feature-Bereitstellung:** `extraction.py` lädt Features aus `artifacts/features/`. Bei `--extract-from-raw` werden Features aus den Roh-CSVs extrahiert. Mit `--labels <datei.lbl>` können gezielt ausgewählte Dateien und deren Labels angegeben werden; Zeilen mit leerem Label werden übersprungen.
3. **Vorbereitung:** Mit `prepare_xy()` werden X, y und Meta-Daten extrahiert; Gruppen (`driver_id::recording`) für StratifiedGroupKFold gebildet.
4. **Training:** Für jede Kombination (Feature-Set × Modell) ruft `run.py` `evaluate.run_single_combo()` auf. Diese Funktion führt die Kreuzvalidierung durch, speichert das Modell und erzeugt die pro-Kombination-Plots.
5. **Ausgabe:** `report.write_reports_and_plots()` schreibt `metrics_summary.csv`, die Markdown-Reports und die Summary-Plots.

---

## 3. Ordnerstruktur und Ausgaben

### 3.1 Ordnerstruktur

| Ordner | Aufgabe |
|--------|---------|
| **pipeline_minimal_beispiel/** | Wurzel des minimalen Beispiels. Enthält die Python-Module (run.py, extraction.py, evaluate.py, report.py, progress.py) sowie das Ausgabeverzeichnis artifacts/. |
| **pipeline_minimal_beispiel/artifacts/** | Eigenständiges Ausgabeverzeichnis für alle Pipeline-Ausgaben. Enthält metrics_summary.csv, modellvergleich_uebersicht.md, pipeline_progress.json sowie die Unterordner features/, models/, plots/, loo_per_recording/ und logs/. |
| **artifacts/features/** | Feature-CSVs (z. B. `features_daten_FSChatGPT.csv`). Werden bei `--extract-from-raw` erzeugt; ohne Extraktion müssen gecachte Features bereits vorhanden sein. |
| **pipeline_minimal_beispiel/artifacts/models/** | Gespeicherte Modelle: pro Modell ein Unterordner (z. B. `logreg/`, `extratrees/`), darin `{dataset}_{modell}.joblib` (z. B. `FSChatGPT_logreg.joblib`). Enthält die komplette scikit-learn-Pipeline (Imputer, Scaler, Classifier). |
| **pipeline_minimal_beispiel/artifacts/plots/** | Grafiken in Unterordnern: `summary/` (Accuracy, F1/Precision/Recall), `confusion/`, `metrics/`, `importance/<modell>/`, `model_specific/`, `feature_correlation/`, `raw/`, `loo/`. |
| **pipeline_minimal_beispiel/artifacts/loo_per_recording/** | Bei Leave-One-Recording-Out: CSV-Dateien mit Zuordnung und Korrektheit pro Recording pro Kombination (z. B. `FSChatGPT_logreg.csv`). |
| **pipeline_minimal_beispiel/artifacts/holdout/** | Bei `--holdout-file`: `holdout_predictions.csv` mit Vorhersage pro Recording und Modell (dataset, model, recording, predicted, true, correct). |
| **pipeline_minimal_beispiel/artifacts/logs/** | Pipeline-Logs (analog zur Haupt-Pipeline). Kann für Fehlerdiagnose und Nachvollziehbarkeit genutzt werden. |

#### LBL-Datei (Label-Datei)

Die LBL-Datei ermöglicht die manuelle Auswahl von Recording-Dateien und deren Zuordnung zu Fahrer-Labels. Format (CSV mit Komma oder Semikolon):

```
File,Label
recording_2026_02_10__13_10_22_fabian.csv,Fabian
recording_2026_02_10__13_18_02_florian.csv,Florian
recording_2026_02_10__13_25_22_matthias.csv,Matthias
recording_2026_02_10__13_37_56_florian.csv,
recording_2026_02_10__13_44_54_matthias.csv,Matthias
recording_2026_02_10__13_51_14_fabian.csv,Fabian
```

Zeilen mit leerem Label werden übersprungen (nur gelabelte Recordings fließen ins Training). Die Dateinamen werden relativ zu `--data-dir` aufgelöst.

#### Holdout-Datei (für Vorhersage ohne Training)

Die Holdout-Datei listet Recordings, die **vom Training ausgeschlossen** werden. Nach dem Training wird für diese Recordings eine Vorhersage erstellt. Format wie LBL (File, Label):

```
File,Label
recording_2026_02_10__13_37_56_florian.csv,Florian
recording_2026_02_10__15_13_03_florian.csv,Florian
```

- **Mit Label:** Ermöglicht Evaluation (Vergleich Vorhersage vs. tatsächlicher Fahrer).
- **Ohne Label:** Nur Vorhersage, keine Evaluation.
- Die Holdout-Recordings werden aus dem Training entfernt (auch wenn sie in der LBL-Datei stehen).
- Erfordert `--extract-from-raw`.

### 3.2 Wichtige Dateien

| Datei | Speicherort | Aufgabe |
|------|-------------|---------|
| **run.py** | pipeline_minimal_beispiel/ | Haupt-Script und Einstiegspunkt. Parst CLI-Argumente, orchestriert Feature-Laden, Modell-Training und Report-Erstellung. |
| **extraction.py** | pipeline_minimal_beispiel/ | Vollständige Feature-Extraktion (FSChatGPT, FSGemini, auto, featuretools, merged_all, selected). Lädt aus Cache oder extrahiert aus Rohdaten. Eigenständig, keine Abhängigkeit von pipeline_project. |
| **evaluate.py** | pipeline_minimal_beispiel/ | Evaluation einer einzelnen Feature-Set/Modell-Kombination: StratifiedGroupKFold oder LOO, Modell-Training, Speicherung, Confusion Matrix, Feature-Importance, Modell-spezifische Plots. |
| **report.py** | pipeline_minimal_beispiel/ | Erstellt modellvergleich_uebersicht.md, run_report.md, metrics_summary.csv sowie die Summary-Plots (Accuracy-Balken, Heatmaps, F1/Precision/Recall). |
| **progress.py** | pipeline_minimal_beispiel/ | Schreibt pipeline_progress.json für das Frontend-Polling (phase, completed, in_progress, message). |
| **labels.py** | pipeline_minimal_beispiel/ | Lädt LBL-Dateien (File, Label), filtert Zeilen ohne Label, löst Pfade relativ zu base_dir auf. |
| **metrics_summary.csv** | artifacts/ | Metriken aller Feature-Set/Modell-Kombinationen: dataset, model, recording_accuracy, window_accuracy, recording_f1, recording_precision, recording_recall sowie Pfad zum gespeicherten Modell. Primäre Ergebnisdatei für die Auswertung. |
| **modellvergleich_uebersicht.md** | artifacts/ | Markdown-Report mit Konfiguration (Feature-Sets, Modelle, Kreuzvalidierung), bester Kombination, Übersichtstabelle aller Metriken und eingebetteten Grafiken zur Modellvergleich. |
| **run_report.md** | artifacts/ | Identisch zu modellvergleich_uebersicht.md. Wird parallel geschrieben für Kompatibilität mit der Haupt-Pipeline. |
| **pipeline_progress.json** | artifacts/ | Fortschrittsdatei für das Streamlit-Dashboard: phase (starting, extraction, training, done), completed (abgeschlossene Kombinationen), in_progress (aktuell laufend), message. |
| **summary/accuracy_summary.png** | artifacts/plots/summary/ | Balkendiagramm: Recording-Accuracy pro Modell und Feature-Set. |
| **summary/accuracy_heatmap.png** | artifacts/plots/summary/ | Heatmap: Recording-Accuracy über Dataset und Modell. |
| **summary/f1_precision_recall_summary.png** | artifacts/plots/summary/ | Balkendiagramm: F1, Precision, Recall pro Kombination. |
| **summary/metrics_heatmap_f1_precision_recall.png** | artifacts/plots/summary/ | Heatmap: F1, Precision, Recall über alle Kombinationen. |
| **confusion/confusion_<dataset>_<model>.png** | artifacts/plots/confusion/ | Confusion Matrix pro Kombination. |
| **metrics/metrics_<dataset>_<model>.png** | artifacts/plots/metrics/ | F1, Precision, Recall pro Kombination. |
| **importance/<modell>/<dataset>.png** | artifacts/plots/importance/ | Feature-Importance (Top-K) pro Modell und Dataset. |
| **model_specific/model_specific_<dataset>_<model>.png** | artifacts/plots/model_specific/ | Modell-spezifische Visualisierung (Baum, Koeffizienten, Entscheidungsgrenze). |
| **feature_correlation/feature_correlation_<dataset>.png** | artifacts/plots/feature_correlation/ | Korrelationsmatrix der Features pro Dataset. |
| **raw/raw_sensor_correlation.png** | artifacts/plots/raw/ | Korrelation der Rohdaten-Sensoren (bei --extract-from-raw). |
| **loo/loo_*.png** | artifacts/plots/loo/ | LOO-Plots (bei --loo). |

---

## 4. Nutzung und Befehle

### 4.1 Python-Befehle (tabellarisch)

| Befehl | Beschreibung |
|--------|--------------|
| `python run.py --feature-sets FSChatGPT --models logreg` | **Schnellstart mit Cache:** Lädt FSChatGPT-Features aus dem Cache (artifacts/features/), trainiert LogReg mit 5-facher StratifiedGroupKFold und schreibt Metriken sowie Plots nach artifacts/. Ohne vorherige Extraktion muss zuerst ein Lauf mit `--extract-from-raw` erfolgen. (Aus pipeline_minimal_beispiel/ ausführen.) |
| `python pipeline_minimal_beispiel/run.py --extract-from-raw --feature-sets FSChatGPT --models logreg` | **Erste Ausführung oder neue Daten:** Extrahiert FSChatGPT-Features aus den Roh-CSVs im Ordner Daten/, speichert sie im Cache, trainiert LogReg und erzeugt die Ausgaben. Verwenden Sie dies, wenn Sie neue Recordings haben oder der Cache leer ist. |
| `python pipeline_minimal_beispiel/run.py --extract-from-raw --labels artifacts/recording_labels.lbl --data-dir Daten --feature-sets FSChatGPT --models logreg` | **Mit LBL-Datei:** Verwendet nur die in der LBL-Datei gelisteten Recordings und deren Labels. `--data-dir` ist das Basisverzeichnis für die Dateipfade in der LBL-Datei. Zeilen mit leerem Label werden übersprungen. |
| `python pipeline_minimal_beispiel/run.py --extract-from-raw --labels artifacts/recording_labels.lbl --holdout-file artifacts/recording_holdout.lbl --data-dir Daten --feature-sets FSChatGPT --models logreg` | **Mit Holdout:** Holdout-Recordings werden vom Training ausgeschlossen; nach dem Training wird eine Vorhersage erstellt. `holdout_predictions.csv` enthält predicted, true, correct. |
| `python pipeline_minimal_beispiel/run.py --extract-from-raw --with-merged --with-selected --loo --feature-sets FSChatGPT FSGemini --models logreg extratrees` | **Vollständige Evaluation:** Wie oben, zusätzlich werden merged_all (Vereinigung aller Basis-Sätze) und selected (Top-K nach Importance) erzeugt. `--loo` aktiviert Leave-One-Recording-Out – jede Fahrt wird einmal komplett als Test zurückgehalten. Ideal für die finale Bewertung der Generalisierung. |
| `python pipeline_minimal_beispiel/run.py --feature-sets FSChatGPT FSGemini --models logreg extratrees --n-jobs 4` | **Parallele Jobs:** n_jobs=4 für ExtraTrees, RandomForest und LogReg. Beschleunigt das Training bei mehreren Modellen. Unter Windows kann n_jobs=-1 zu PermissionErrors führen; n_jobs=1 ist dann zuverlässiger. |

### 4.2 Parameter (tabellarisch)

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `--data-dir` | Path | `data` | Ordner mit Recording-CSVs (relativ zum Pipeline-Ordner): Bei `--extract-from-raw` ohne `--labels` werden alle CSVs in diesem Ordner verwendet. Bei `--labels` ist dies das Basisverzeichnis für relative Pfade in der LBL-Datei. |
| `--labels` | Path | – | Pfad zur LBL-Datei (CSV mit Spalten File, Label). Bei `--extract-from-raw` werden nur die gelisteten Dateien verwendet; Zeilen mit leerem Label werden übersprungen. |
| `--holdout-file` | Path | – | Pfad zur Holdout-Datei (CSV: File, Label). Diese Recordings werden vom Training ausgeschlossen; nach dem Training wird eine Vorhersage erstellt. Label optional (für Evaluation). Erfordert `--extract-from-raw`. |
| `--extract-from-raw` | Flag | – | Aktiviert die Feature-Extraktion aus den Rohdaten. Ohne dieses Flag werden nur gecachte Features verwendet – bei leerem Cache schlägt der Lauf fehl. |
| `--feature-sets` | Liste | FSChatGPT | Welche Feature-Sätze verwendet werden: FSChatGPT, FSGemini, auto, featuretools, merged_all, selected. Mehrere durch Leerzeichen getrennt. |
| `--models` | Liste | logreg, extratrees | Welche Klassifikatoren trainiert werden: extratrees, randomforest, logreg, svm_rbf, nearest_centroid. Mehrere durch Leerzeichen getrennt. |
| `--with-merged` | Flag | – | Baut den Feature-Satz merged_all aus den Basis-Sätzen (FSChatGPT, FSGemini, auto, featuretools). Erfordert, dass diese zuvor extrahiert wurden. |
| `--with-selected` | Flag | – | Baut den Feature-Satz selected: Extrahiert die Top-K wichtigsten Merkmale aus merged_all mittels Extra-Trees-Importance. `--with-merged` muss gesetzt sein. |
| `--skip-featuretools` | Flag | – | Überspringt die Featuretools-Extraktion (Deep Feature Synthesis). Spart Zeit, wenn Sie nur FSChatGPT, FSGemini oder auto nutzen. |
| `--loo` | Flag | – | Verwendet Leave-One-Recording-Out statt K-Fold: Jede Fahrt wird genau einmal als Testmenge zurückgehalten. Strengere Schätzung der Generalisierung, aber rechenintensiver. |
| `--force` | Flag | – | Ignoriert den Feature-Cache und extrahiert alle Features neu. Verwenden Sie dies nach Änderungen an der Extraktionslogik oder bei korruptem Cache. |
| `--n-splits` | int | 5 | Anzahl der Folds für StratifiedGroupKFold. Höhere Werte = mehr Durchläufe, feinere Schätzung, längere Laufzeit. |
| `--n-jobs` | int | 1 | Parallele Jobs für Modell-Training (ExtraTrees, RandomForest, LogReg). 1 = sequentiell, -1 = alle Kerne. Unter Windows kann -1 zu Joblib-PermissionErrors führen. |
| `--out-dir` | Path | `artifacts` | Verzeichnis für Ausgaben (relativ zum Pipeline-Ordner): metrics_summary.csv, models/, plots/, modellvergleich_uebersicht.md. |
| `--no-plots` | Flag | – | Erzeugt keine Grafiken. Spart Zeit bei reinen Metrik-Läufen; die Reports enthalten dann keine eingebetteten Bilder. |
| `--tune-models` | str | – | Komma-getrennte Liste von Modellen, für die GridSearch durchgeführt wird (z. B. `logreg,svm_rbf`). Ohne Angabe werden Standard-Parameter verwendet. |
| `--model-params` | str | – | JSON-Dict mit Modell-spezifischen Parametern (z. B. `{"logreg":{"C":10},"extratrees":{"max_depth":8}}`). Überschreibt die Standard-Konfiguration. |

---

## 5. Modellgeeignetheit (Übersicht)

Die minimale Pipeline nutzt dieselben Modelle wie die Haupt-Pipeline. Die theoretische Einschätzung der Modellgeeignetheit für die Fahreridentifikation bleibt unverändert:

| Modell | Einschätzung | Hinweis |
|--------|--------------|---------|
| Logistische Regression | sehr gut geeignet | Stabil bei kleiner Stichprobe (22 Recordings, 744 Fenster), interpretierbare Koeffizienten, L2-Regularisierung wirkt Overfitting entgegen. Liefert Klassenwahrscheinlichkeiten für Recording-Level-Aggregation. Empfohlene Wahl für die Fahreridentifikation. |
| Random Forest | gut geeignet | Gute Alternative zur Logistischen Regression; Merkmalswichtigkeit zeigt relevante Features. Stabile Performance über verschiedene Feature-Sätze nach Hyperparameter-Optimierung. |
| Extra Trees | eingeschränkt geeignet | Parametereinstellung kritisch – ohne Begrenzung der Baumkomplexität (max_depth, min_samples_leaf) starkes Overfitting-Risiko. Hilfreich für explorative Analysen und Feature-Importance. |
| SVM | bedingt geeignet | Starke Abhängigkeit vom Feature-Set und von C/gamma. Auf FSChatGPT und FSGemini oft schlechter als LogReg; auf merged_all mit vielen Merkmalen nutzbar. Empfindlich gegenüber Skalierung. |
| Nearest Centroid | als Baseline geeignet | Transparent und recheneffizient; typischerweise geringere Genauigkeit als die anderen Modelle. Dient als Vergleichsbasis für die komplexeren Verfahren. |

---

## 6. Beispiel-Resultate

Bei einem typischen Lauf mit FSChatGPT, FSGemini sowie logreg und extratrees (StratifiedGroupKFold, n=3) ergeben sich z. B. folgende Recording-Accuracies:

| Dataset | Modell | Recording-Acc | Window-Acc | F1 | Precision | Recall |
|---------|--------|---------------|------------|-----|-----------|--------|
| FSGemini | logreg | 0,955 | 0,508 | 0,955 | 0,963 | 0,952 |
| FSChatGPT | logreg | 0,909 | 0,601 | 0,910 | 0,917 | 0,911 |
| FSChatGPT | extratrees | 0,864 | 0,632 | 0,856 | 0,909 | 0,857 |
| FSGemini | extratrees | 0,818 | 0,586 | 0,824 | 0,833 | 0,821 |

Die beste Kombination ist **FSGemini + LogReg** mit 95,5 % Recording-Accuracy. Die vollständigen Metriken stehen in `pipeline_minimal_beispiel/artifacts/metrics_summary.csv`.

**Feature-Dokumentation:** Detaillierte Beschreibungen der manuellen Feature-Sätze (Spalten und Berechnungsformeln) finden sich in `docs/FSChatGPT_FEATURES.md` und `docs/FSGemini_FEATURES.md`.

### 6.1 Grafiken zur Modellvergleich

Die folgenden Grafiken stellen alle Modelle und Feature-Sätze gemeinsam gegenüber und ermöglichen einen direkten Vergleich der Kombinationen.

**Recording-Accuracy (Balkendiagramm)**

![Recording-Accuracy pro Modell und Feature-Set](artifacts/plots/summary/accuracy_summary.png)

*Balkendiagramm: Recording-Accuracy pro Modell und Feature-Set*

**Recording-Accuracy (Heatmap)**

![Heatmap Recording-Accuracy: Dataset x Modell](artifacts/plots/summary/accuracy_heatmap.png)

*Heatmap: Recording-Accuracy über Dataset und Modell*

**F1, Precision, Recall (Balkendiagramm)**

![F1, Precision, Recall pro Kombination](artifacts/plots/summary/f1_precision_recall_summary.png)

*F1-Score, Precision und Recall pro Feature-Set/Modell-Kombination*

**F1, Precision, Recall (Heatmap)**

![Heatmap F1, Precision, Recall](artifacts/plots/summary/metrics_heatmap_f1_precision_recall.png)

*Heatmap: F1, Precision und Recall über alle Kombinationen*

---

## 7. Technische Hinweise

- **Struktur:** Die Extraktionslogik liegt in pipeline_minimal_beispiel/extraction.py. Ohne `--extract-from-raw` werden Features aus `artifacts/features/` geladen – bei leerem Cache muss zuerst extrahiert werden.
- **Windows / n_jobs:** Unter Windows kann `n_jobs=-1` zu Joblib-PermissionErrors führen; `n_jobs=1` ist dann zuverlässiger.
- **Frontend:** Die Checkbox „Minimale Pipeline (studentisches Beispiel)“ im Streamlit-Dashboard startet `pipeline_minimal_beispiel/run.py` mit `--out-dir pipeline_minimal_beispiel/artifacts`. Metriken und Plots werden aus diesem Verzeichnis angezeigt.

---

## Anhang: Tabellen für Excel

Die Tabellen aus dieser Dokumentation sind in `pipeline_minimal_beispiel/doc_tabellen_excel/` als CSV-Dateien abgelegt und können direkt in Excel geöffnet werden. Siehe die Dateien in jenem Ordner.
