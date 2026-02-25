# -*- coding: utf-8 -*-
"""
Modul: progress
===============
Schreibt pipeline_progress.json für Frontend-Polling. Das Frontend kann die Datei
periodisch lesen, um den Pipeline-Fortschritt anzuzeigen (Phase, abgeschlossene/laufende
Schritte, Prozent). Optional wird ein callback aufgerufen.

Phasen: starting, extraction, training, prediction, done
"""

import json
from pathlib import Path
from typing import Callable

def write_progress(
    out_dir: Path,
    phase: str,
    total: int = 0,
    completed: list | None = None,
    in_progress: list | None = None,
    message: str | None = None,
    callback: Callable[[str, int, list[str], list[str], str | None , int, float], None] | None = None
) -> None:
    """
    Schreibt Fortschritt in pipeline_progress.json für Frontend-Anzeige.

    Args:
        out_dir: Ausgabe-Ordner (z.B. artifacts_dir)
        phase: "starting" | "extraction" | "training" | "prediction" | "done"
        total: Gesamtanzahl Schritte (z.B. Anzahl Modelle)
        completed: Liste abgeschlossener Schritte (z.B. ["randomforest", "logreg"])
        in_progress: Liste laufender Schritte (z.B. ["logreg"])
        message: Anzeige-Text für den Nutzer
    """
    path = out_dir / "pipeline_progress.json"
    completed_list = completed or []
    in_progress_list = in_progress or []
    n_completed = len(completed_list)
    remaining = max(0, total - n_completed - len(in_progress_list))

    if total > 0:
        percent = min(100.0, round(n_completed / total * 100.0, 1))
    elif phase == "done":
        percent = 100.0
    else:
        percent = 0.0

    data = {
        "phase": phase,
        "total": total,
        "completed": completed_list,
        "in_progress": in_progress_list,
        "message": message or "",
        "remaining": remaining,
        "percent": percent,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # Kein Abbruch bei Schreibfehlern (z.B. Ordner nicht beschreibbar)

    if callback is not None:
        callback(phase, total, completed_list, in_progress_list, message, remaining, percent)