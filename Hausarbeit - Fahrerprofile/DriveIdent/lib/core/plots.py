# -*- coding: utf-8 -*-
"""
Modul: plots
============
Generiert Grafiken für Training und Vorhersage. Speichert in Unterordnern von
artifacts/plots/: confusion/, importance/, accuracy/, prediction/

Hauptfunktionen:
    plot_confusion_matrix()     - Konfusionsmatrix pro Modell
    plot_feature_importance()   - Feature Importance (Top 30) pro Modell
    plot_accuracy()             - Balkendiagramm der Modell-Genauigkeiten
    plot_prediction_results()   - Vorhersage richtig/falsch pro Recording
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def _ensure_dir(path: Path) -> Path:
    """Erstellt Verzeichnis falls nötig und gibt Pfad zurück."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    classes: list[str],
    model_name: str,
    out_dir: Path,
) -> Path | None:
    """
    Erstellt Konfusionsmatrix und speichert sie in out_dir/plots/confusion/.

    Args:
        y_true: Tatsächliche Labels
        y_pred: Vorhergesagte Labels
        classes: Klassenbezeichnungen
        model_name: Modellname für Titel
        out_dir: Ausgabe-Ordner (z.B. artifacts_dir)

    Returns:
        Pfad zur gespeicherten Datei oder None bei Fehler
    """
    try:
        plots_dir = _ensure_dir(out_dir / "plots" / "confusion")
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Konfusionsmatrix – {model_name}")
        plt.tight_layout()
        out_path = plots_dir / f"confusion_{model_name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        return None


def plot_feature_importance(
    pipe,
    feat_cols: list[str],
    model_name: str,
    out_dir: Path,
) -> Path | None:
    """
    Erstellt Feature-Importance-Plot (nur für RandomForest/ExtraTrees oder LogReg).
    Speichert in out_dir/plots/importance/.

    Args:
        pipe: sklearn Pipeline mit clf-Step
        feat_cols: Liste der Feature-Spaltennamen
        model_name: Modellname für Titel
        out_dir: Ausgabe-Ordner (z.B. artifacts_dir)

    Returns:
        Pfad zur gespeicherten Datei oder None bei Fehler
    """
    try:
        clf = pipe.named_steps.get("clf")
        if clf is None:
            return None
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            # LogReg: coef_ ist (n_classes, n_features), Betrag mitteln
            imp = np.abs(clf.coef_).mean(axis=0)
        else:
            return None
        if len(imp) != len(feat_cols):
            return None
        idx = np.argsort(imp)[::-1][:30]  # Top 30
        imp_sorted = imp[idx]
        names_sorted = [feat_cols[i] for i in idx]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(names_sorted)), imp_sorted, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance (Top 30) – {model_name}")
        plt.tight_layout()
        plots_dir = _ensure_dir(out_dir / "plots" / "importance")
        out_path = plots_dir / f"importance_{model_name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"Feature Importance Diagramm konnte nicht erstellt werden: {repr(e)}")
        return None


def plot_accuracy(accuracies: dict[str, float], out_dir: Path) -> Path | None:
    """
    Erstellt Balkendiagramm der Modell-Genauigkeiten.
    Speichert in out_dir/plots/accuracy/.
    Args:
        accuracies: {"randomforest": 0.88, "logreg": 0.94}

    Returns:
        Pfad zur gespeicherten Datei oder None bei Fehler
    """
    try:
        if not accuracies:
            return None
        models = list(accuracies.keys())
        accs = [accuracies[m] * 100 for m in models]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, accs, color=["#2ecc71", "#3498db"][:len(models)], alpha=0.8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Modell-Genauigkeit (CV)")
        ax.set_ylim(0, 105)
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{v:.1f}%", ha="center", fontsize=10)
        plt.tight_layout()
        plots_dir = _ensure_dir(out_dir / "plots" / "accuracy")
        out_path = plots_dir / "accuracy_models.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        return None

