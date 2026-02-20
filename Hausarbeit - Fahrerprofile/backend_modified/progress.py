"""
Fortschritts-Meldungen für die minimale Pipeline.

Schreibt pipeline_progress.json für Frontend-Polling (gleiches Format wie pipeline_project).
"""
from __future__ import annotations

from run import FileController
import json
from pathlib import Path


def writeProgress(
    fileController: FileController,
    phase: str,
    total: int = 0,
    completed: list[str] | None = None,
    in_progress: list[str] | None = None,
    message: str | None = None,
) -> None:
    """Schreibt Fortschritt in JSON für Frontend (gleiches Format wie pipeline_project/run.py)."""
    path = fileController.out / "pipeline_progress.json"
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
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def combineFeatureSetAndModelName(featureSet: str, model: str) -> str:
    """Formatiert Feature-Set und Modell als Kombinations-String."""
    return f"{featureSet} | {model}"
