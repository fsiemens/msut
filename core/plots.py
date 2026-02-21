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


def _align_feat_cols(pipe, feat_cols: list[str], imp: np.ndarray) -> list[str]:
    """
    Versucht feat_cols an die tatsächlich im Modell genutzten Features anzupassen.

    Falls beim Featuretools-Merge doppelte Spalten wie driver_id_x/driver_id_y entstehen,
    werden diese in der Pipeline (z.B. SimpleImputer) effektiv entfernt. Dann gilt
    len(imp) != len(feat_cols) und die Importance-Plots würden leer bleiben.
    """
    try:
        if len(imp) == len(feat_cols):
            return feat_cols
        imputer = getattr(pipe, "named_steps", {}).get("imputer") if pipe is not None else None
        stats = getattr(imputer, "statistics_", None)
        if stats is None or len(stats) != len(feat_cols):
            return feat_cols
        keep_mask = np.isfinite(stats)
        aligned = [c for c, keep in zip(feat_cols, keep_mask) if bool(keep)]
        return aligned if len(aligned) == len(imp) else feat_cols
    except Exception:
        return feat_cols


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
    Erstellt Feature-Importance-Plot (RandomForest, GradientBoosting, ExtraTrees oder LogReg).
    GradientBoosting nutzt feature_importances_; LogReg nutzt Betrag von coef_.
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
        feat_cols_aligned = _align_feat_cols(pipe, feat_cols, imp)
        if len(imp) != len(feat_cols_aligned):
            return None
        idx = np.argsort(imp)[::-1][:30]  # Top 30
        imp_sorted = imp[idx]
        names_sorted = [feat_cols_aligned[i] for i in idx]
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
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"][:len(models)]
        bars = ax.bar(models, accs, color=colors, alpha=0.8)
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


def plot_feature_importance_all_models(
    pipes: dict,
    feat_cols: list[str],
    out_dir: Path,
    top_n: int = 20,
) -> Path | None:
    """
    Erstellt einen kombinierten Feature-Importance-Plot für alle Modelle
    (Subplots nebeneinander). Speichert in out_dir/plots/importance/.

    Args:
        pipes: {model_name: fitted Pipeline}
        feat_cols: Liste der Feature-Spaltennamen
        out_dir: Ausgabe-Ordner (z.B. artifacts_dir)
        top_n: Anzahl der Top-Features pro Modell (Standard: 20)

    Returns:
        Pfad zur gespeicherten Datei oder None bei Fehler
    """
    try:
        models = [m for m in pipes if pipes[m] is not None]
        if not models:
            return None
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8), sharey=False)
        if n_models == 1:
            axes = [axes]
        for ax, mdl in zip(axes, models):
            pipe = pipes[mdl]
            clf = pipe.named_steps.get("clf") if pipe else None
            if clf is None:
                ax.set_title(f"{mdl} – keine Importance")
                continue
            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                imp = np.abs(clf.coef_).mean(axis=0)
            else:
                ax.set_title(f"{mdl} – keine Importance")
                continue
            feat_cols_aligned = _align_feat_cols(pipe, feat_cols, imp)
            if len(imp) != len(feat_cols_aligned):
                ax.set_title(f"{mdl} – Dimension mismatch")
                continue
            idx = np.argsort(imp)[::-1][:top_n]
            imp_sorted = imp[idx]
            names_sorted = [feat_cols_aligned[i] for i in idx]
            ax.barh(range(len(names_sorted)), imp_sorted, color="steelblue", alpha=0.8)
            ax.set_yticks(range(len(names_sorted)))
            ax.set_yticklabels(names_sorted, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance (Top {top_n}) – {mdl}")
        plt.tight_layout()
        plots_dir = _ensure_dir(out_dir / "plots" / "importance")
        out_path = plots_dir / "importance_all_models.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"Kombinierter Importance-Plot konnte nicht erstellt werden: {repr(e)}")
        return None
