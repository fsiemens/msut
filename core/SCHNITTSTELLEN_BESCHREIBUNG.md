# Backend-Schnittstellenbeschreibung – Fahrererkennung

Schnittstelle für ein Tkinter-Frontend zur Fahrererkennung (projekt_mini).

---

## 1. Übersicht

Das Backend bietet:
- **Training** von RandomForest- und LogisticRegression-Modellen
- **Vorhersage** mit trainierten Modellen
- **Label-Verwaltung** (Lesen/Schreiben von Trainings- und Test-Labels)
- **Pipeline-Fortschritt** (`pipeline_progress.json`) für Status-Anzeige im Frontend
- **Grafiken** in `artifacts/plots/` (Konfusionsmatrix, Feature Importance, Accuracy, Vorhersage richtig/falsch)

**Projektpfad:** `projekt_mini/` (Working Directory für relative Pfade)

---

## 2. Modul `backend_api`

Das Modul `backend_api` stellt GUI-taugliche Funktionen bereit. Alle Pfade können als `str` oder `pathlib.Path` übergeben werden.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "projekt_mini"))
from backend_api import train, predict, write_labels_file, get_config
```

---

## 3. API-Referenz

### 3.1 `train()`

Trainiert die Modelle (RandomForest, LogisticRegression) auf den angegebenen Daten.

**Signatur:**
```python
def train(
    data_dir: str | Path,
    labels_file: str | Path,
    artifacts_dir: str | Path,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Returns:
        (erfolg: bool, ausgabe: str)
        - erfolg: True bei Erfolg, False bei Fehler
        - ausgabe: Zusammenfassung oder Fehlermeldung
    """
```

**Parameter:**

| Parameter       | Typ                    | Beschreibung                                      |
|----------------|------------------------|---------------------------------------------------|
| `data_dir`     | `str` oder `Path`      | Ordner mit CSV-Recordings                         |
| `labels_file`  | `str` oder `Path`      | Label-Datei (z.B. `labels.lbl`)                   |
| `artifacts_dir`| `str` oder `Path`      | Ausgabeordner für Modelle                         |
| `log_callback` | `(str) -> None` oder `None` | Optional: wird mit Log-Zeilen aufgerufen      |

**Rückgabe:**
- `(True, "randomforest: 88.24%\nlogreg: 94.12%\n...")` bei Erfolg
- `(False, "Fehlermeldung")` bei Fehler

**Hinweis:** Training kann 30–60 Sekunden dauern. In einem GUI-Thread blockierend ausführen oder in einem separaten Thread starten.

---

### 3.2 `predict()`

Führt Vorhersagen mit den trainierten Modellen aus.

**Signatur:**
```python
def predict(
    data_dir: str | Path,
    test_labels_file: str | Path,
    artifacts_dir: str | Path,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str, dict[str, list[dict]]]:
    """
    Returns:
        (erfolg: bool, ausgabe: str, ergebnisse: dict[str, list[dict]])
        - ergebnisse: {"randomforest": [...], "logreg": [...]}
        - Jede Liste: [{"recording": str, "soll": str, "ist": str, "korrekt": bool}, ...]
    """
```

**Parameter:**

| Parameter          | Typ                    | Beschreibung                                      |
|--------------------|------------------------|---------------------------------------------------|
| `data_dir`         | `str` oder `Path`      | Ordner mit Test-CSV-Recordings                    |
| `test_labels_file` | `str` oder `Path`      | Test-Label-Datei (z.B. `test_labels.lbl`)         |
| `artifacts_dir`    | `str` oder `Path`      | Ordner mit Modellen                               |
| `log_callback`     | `(str) -> None` oder `None` | Optional: Log-Zeilen                             |

**Rückgabe:**
- `(True, "Korrekt: 6/7\n...", [{"recording": "...", "soll": "florian", "ist": "florian", "korrekt": True}, ...])`
- `(False, "Fehlermeldung", {})` bei Fehler

---

### 3.3 `write_labels_file()`

Erzeugt eine Label-Datei aus einer Liste von Dateipfaden. Der Fahrer wird aus dem Dateinamen extrahiert (florian, matthias, fabian).

**Signatur:**
```python
def write_labels_file(
    file_paths: list[str | Path],
    output_path: str | Path,
) -> int:
    """
    Returns:
        Anzahl geschriebener Einträge
    """
```

**Parameter:**

| Parameter     | Typ                    | Beschreibung                                      |
|---------------|------------------------|---------------------------------------------------|
| `file_paths`  | `list[str \| Path]`    | Liste der CSV-Dateipfade                          |
| `output_path` | `str` oder `Path`      | Ziel-Label-Datei (z.B. `labels.lbl`)              |

**Beispiel:**
```python
write_labels_file(
    ["data/recording_1_florian.csv", "data/recording_2_matthias.csv"],
    "labels.lbl"
)
```

---

### 3.4 `get_config()`

Liefert die aktuelle Konfiguration (z.B. für Anzeige oder Voreinstellungen).

**Signatur:**
```python
def get_config() -> dict:
    """
    Returns:
        {
            "data_dir": Path,
            "labels_file": Path,
            "test_labels_file": Path,
            "artifacts_dir": Path,
            "models": ["randomforest", "logreg"],
            "window_sec": 25,
            "step_sec": 12,
        }
    """
```

---

## 4. Pipeline-Fortschritt (Status-Anzeige)

Während Training und Vorhersage schreibt das Backend `pipeline_progress.json` in den `artifacts_dir`. Das Frontend kann diese Datei periodisch lesen (Polling), um den aktuellen Pipeline-Status anzuzeigen.

**Pfad:** `{artifacts_dir}/pipeline_progress.json`  
(z.B. `projekt_mini/artifacts/pipeline_progress.json`)

### Format

```json
{
  "phase": "training",
  "total": 2,
  "completed": ["randomforest"],
  "in_progress": ["logreg"],
  "message": "Trainiere logreg...",
  "remaining": 0,
  "percent": 50.0
}
```

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `phase` | string | `"starting"` \| `"extraction"` \| `"training"` \| `"prediction"` \| `"done"` |
| `total` | int | Gesamtanzahl Schritte (z.B. Anzahl Modelle) |
| `completed` | list[str] | Abgeschlossene Schritte (z.B. `["randomforest"]`) |
| `in_progress` | list[str] | Aktuell laufende Schritte (z.B. `["logreg"]`) |
| `message` | string | Anzeige-Text für den Nutzer |
| `remaining` | int | Noch ausstehende Schritte |
| `percent` | float | Fortschritt in Prozent (0–100) |

### Phasen

**Training:**
- `starting` – Lade Labels...
- `extraction` – Extrahiere Features...
- `training` – Trainiere Modell (completed/in_progress pro Modell)
- `done` – Fertig

**Vorhersage:**
- `starting` – Lade Test-Labels...
- `extraction` – Extrahiere Features...
- `prediction` – Vorhersage pro Modell
- `done` – Fertig

### Beispiel: Polling im Frontend

```python
import json
from pathlib import Path

def poll_progress(artifacts_dir: Path, callback):
    """Pollt pipeline_progress.json und ruft callback(progress_dict) auf."""
    path = Path(artifacts_dir) / "pipeline_progress.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            callback(data)
        except Exception:
            pass
```

---

## 5. Label-Datei-Format

CSV mit Header, Trennzeichen beliebig (Komma oder Tab):

```
File,Label
recording_2026_02_10__15_31_22_florian.csv,florian
recording_2026_02_10__15_38_03_matthias.csv,matthias
```

- **File:** Dateiname (relativ zu `data_dir`) oder absoluter Pfad
- **Label:** Fahrer-ID (florian, matthias, fabian oder andere)

---

## 6. CSV-Recording-Format

Jede Recording-Datei muss folgende Spalten enthalten:
- `timestamp`
- `wheel_position`
- `car0_throttle_position`
- `car0_brake_position`
- `car0_velocity_vehicle`
- `rot_vel`

Zweite Zeile kann Einheiten enthalten (wird übersprungen).

---

## 7. Fehlerbehandlung

Alle API-Funktionen geben bei Fehlern `(False, fehlermeldung)` bzw. `(False, fehlermeldung, [])` zurück statt Exceptions zu werfen. Typische Fehler:

- Keine gültigen Labels gefunden
- Label-Datei nicht gefunden
- Modelle nicht gefunden (vor predict muss train ausgeführt worden sein)
- Ungültige CSV-Dateien

---

## 8. Beispiel: Tkinter-Integration

```python
import threading
from backend_api import train, predict, write_labels_file

def on_train_click():
    def run():
        ok, msg = train(  # train() returns (ok, msg)
            data_dir="data",
            labels_file="labels.lbl",
            artifacts_dir="artifacts",
            log_callback=lambda s: root.after(0, lambda: log_widget.insert("end", s + "\n"))
        )
        root.after(0, lambda: on_train_done(ok, msg))
    threading.Thread(target=run, daemon=True).start()

def on_predict_click():
    def run():
        ok, out, ergebnisse = predict(
            data_dir="data",
            test_labels_file="test_labels.lbl",
            artifacts_dir="artifacts",
            log_callback=lambda s: root.after(0, lambda: log_widget.insert("end", s + "\n"))
        )
        root.after(0, lambda: on_predict_done(ok, out, ergebnisse))
    threading.Thread(target=run, daemon=True).start()

def on_train_done(ok, msg):
    if ok:
        status_label.config(text="Training erfolgreich")
    else:
        messagebox.showerror("Fehler", msg)

def on_predict_done(ok, out, ergebnisse):
    if ok:
        status_label.config(text="Vorhersage fertig")
        # ergebnisse: {"randomforest": [...], "logreg": [...]}
    else:
        messagebox.showerror("Fehler", out)
```

---

## 9. Grafiken (artifacts/plots/)

Nach Training und Vorhersage erzeugt das Backend Grafiken in Unterordnern von `{artifacts_dir}/plots/`:

| Ordner | Inhalt |
|--------|--------|
| `confusion/` | Konfusionsmatrix pro Modell |
| `importance/` | Feature Importance (Top 30) pro Modell |
| `accuracy/` | Balkendiagramm der Modell-Genauigkeiten |
| `prediction/` | Vorhersage richtig (grün) / falsch (rot) pro Recording |

Das Frontend kann diese PNG-Dateien laden und anzeigen (z.B. nach Abschluss von train oder predict).

---

## 10. Abhängigkeiten

- Python 3.10+
- pandas, numpy, scikit-learn, joblib, featuretools, tsfresh, matplotlib
