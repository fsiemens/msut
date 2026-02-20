"""
Report- und Plot-Erstellung für die minimale Pipeline.

Schreibt metrics_summary.csv, modellvergleich_uebersicht.md, run_report.md,
Summary-Plots und LOO-Dateien.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd


def write_reports_and_plots(
    out_dir: Path,
    plots_dir: Path,
    summary: pd.DataFrame,
    loo_per_rec: dict[str, pd.DataFrame],
    feature_sets: list[str],
    models: list[str],
    tune_models: set[str],
    n_recordings: int,
    n_splits: int,
    use_loo: bool,
    extract_from_raw: bool,
    no_plots: bool,
) -> None:
    """Schreibt alle Reports, Summary-Plots und LOO-Dateien."""
    from plots import (
        plot_accuracy_summary,
        plot_f1_precision_recall_summary,
        plot_heatmap_accuracy_matrix,
        plot_heatmap_metrics_f1_precision_recall,
        plot_loo_accuracy_per_combo,
        plot_loo_recording_heatmap,
    )

    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    summary_dir = plots_dir / "summary"
    loo_dir_plots = plots_dir / "loo"
    if not no_plots and not summary.empty:
        summary_dir.mkdir(parents=True, exist_ok=True)
        plot_accuracy_summary(summary, summary_dir / "accuracy_summary.png", x_col="model", y_col="recording_accuracy", hue_col="dataset", title="Recording-Accuracy pro Modell und Feature-Set")
        plot_heatmap_accuracy_matrix(summary, summary_dir / "accuracy_heatmap.png", row_col="dataset", col_col="model", value_col="recording_accuracy", title="Recording-Accuracy: Dataset x Modell")
        plot_f1_precision_recall_summary(summary, summary_dir / "f1_precision_recall_summary.png")
        plot_heatmap_metrics_f1_precision_recall(summary, summary_dir / "metrics_heatmap_f1_precision_recall.png")

    if loo_per_rec:
        loo_dir = out_dir / "loo_per_recording"
        loo_dir.mkdir(parents=True, exist_ok=True)
        for key, per_df in loo_per_rec.items():
            per_df.to_csv(loo_dir / f"{key.replace(' ', '_')}.csv", index=False)
        if not no_plots:
            loo_dir_plots.mkdir(parents=True, exist_ok=True)
            plot_loo_recording_heatmap(loo_per_rec, loo_dir_plots / "loo_recording_heatmap.png")
            plot_loo_accuracy_per_combo(loo_per_rec, loo_dir_plots / "loo_accuracy_per_combo.png")

    cv_mode = "Leave-One-Recording-Out" if use_loo else f"StratifiedGroupKFold (n={n_splits})"
    tune_str = ", ".join(sorted(tune_models)) if tune_models else "keine"
    best = summary.iloc[0] if not summary.empty else {}
    rep = [
        "# Pipeline – Modellvergleich (Minimal)\n",
        f"\n*Zuletzt aktualisiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        f"\n## Konfiguration\n",
        f"- **Feature-Sets:** {', '.join(feature_sets)}\n",
        f"- **Modelle:** {', '.join(models)}\n",
        f"- **Hyperparameter-Optimierung:** {tune_str}\n",
        f"- **Kreuzvalidierung:** {cv_mode}\n",
        f"- **Recordings:** {n_recordings}\n",
        f"\n## Beste Kombination\n",
        f"- **Dataset:** {best.get('dataset', '-')} | **Modell:** {best.get('model', '-')}\n",
        f"- **Recording-Accuracy:** {float(best.get('recording_accuracy', 0)):.3f}\n",
        f"- **Window-Accuracy:** {float(best.get('window_accuracy', 0)):.3f}\n",
        f"- **F1 (Recording):** {float(best.get('recording_f1', 0)):.3f}\n",
        f"\n## Übersicht aller Kombinationen (sortiert nach Recording-Accuracy)\n",
        "\n| Dataset | Modell | Recording-Acc | Window-Acc | F1 | Precision | Recall |\n",
        "|---------|--------|---------------|------------|-----|-----------|--------|\n",
    ]
    for _, row in summary.iterrows():
        rep.append(f"| {row['dataset']} | {row['model']} | {float(row['recording_accuracy']):.3f} | {float(row['window_accuracy']):.3f} | {float(row.get('recording_f1', 0)):.3f} | {float(row.get('recording_precision', 0)):.3f} | {float(row.get('recording_recall', 0)):.3f} |\n")

    # Auswertung: Grafiken zur Modellvergleich einbetten
    if not no_plots and not summary.empty:
        rep.append("\n## Auswertung – Grafiken zur Modellvergleich\n")
        rep.append("\nDie folgenden Grafiken unterstützen den direkten Vergleich der Modelle über verschiedene Feature-Sätze.\n")

        rep.append("\n### Recording-Accuracy (Balkendiagramm)\n")
        rep.append("\n![Recording-Accuracy pro Modell und Feature-Set](plots/summary/accuracy_summary.png)\n")
        rep.append("\n*Balkendiagramm: Recording-Accuracy pro Modell und Feature-Set*\n")

        rep.append("\n### Recording-Accuracy (Heatmap)\n")
        rep.append("\n![Heatmap Recording-Accuracy: Dataset x Modell](plots/summary/accuracy_heatmap.png)\n")
        rep.append("\n*Heatmap: Recording-Accuracy über Dataset und Modell*\n")

        rep.append("\n### F1, Precision, Recall (Balkendiagramm)\n")
        rep.append("\n![F1, Precision, Recall pro Kombination](plots/summary/f1_precision_recall_summary.png)\n")
        rep.append("\n*F1-Score, Precision und Recall pro Feature-Set/Modell-Kombination*\n")

        rep.append("\n### F1, Precision, Recall (Heatmap)\n")
        rep.append("\n![Heatmap F1, Precision, Recall](plots/summary/metrics_heatmap_f1_precision_recall.png)\n")
        rep.append("\n*Heatmap: F1, Precision und Recall über alle Kombinationen*\n")

        rep.append("\n### Confusion Matrices pro Kombination\n")
        for _, row in summary.iterrows():
            ds, mdl = row["dataset"], row["model"]
            rep.append(f"\n#### {ds} | {mdl}\n")
            rep.append(f"\n![Confusion Matrix {ds} {mdl}](plots/confusion/confusion_{ds}_{mdl}.png)\n")

        if use_loo and loo_per_rec:
            rep.append("\n### LOO: Leave-One-Recording-Out\n")
            rep.append("\n![LOO: Korrekt pro Recording](plots/loo/loo_recording_heatmap.png)\n")
            rep.append("\n![LOO Recording-Accuracy pro Kombination](plots/loo/loo_accuracy_per_combo.png)\n")

    rep.append("\n## Ordnerstruktur der Plots\n")
    rep.append("```\n")
    rep.append("plots/\n")
    rep.append("├── summary/           # Übersichts-Plots (Accuracy, F1, Precision, Recall)\n")
    rep.append("├── confusion/         # Confusion Matrix pro Kombination\n")
    rep.append("├── metrics/           # F1/Precision/Recall pro Kombination\n")
    rep.append("├── importance/        # Feature-Importance (Top-K) pro Modell/Dataset\n")
    rep.append("├── model_specific/    # Modell-spezifische Visualisierungen\n")
    rep.append("├── feature_correlation/  # Feature-Korrelationsmatrix pro Dataset\n")
    rep.append("├── raw/               # Rohdaten-Plots (bei --extract-from-raw)\n")
    rep.append("└── loo/               # LOO-Plots (bei --loo)\n")
    rep.append("```\n")
    rep.append("\n## Weitere Grafiken (Übersicht)\n")
    rep.append("- `summary/accuracy_summary.png` – Balkendiagramm Recording-Accuracy\n")
    rep.append("- `summary/accuracy_heatmap.png` – Heatmap Recording-Accuracy\n")
    rep.append("- `summary/f1_precision_recall_summary.png` – F1, Precision, Recall\n")
    rep.append("- `summary/metrics_heatmap_f1_precision_recall.png` – Heatmap F1/Precision/Recall\n")
    rep.append("- `feature_correlation/feature_correlation_<dataset>.png` – Korrelationsmatrix der Features\n")
    rep.append("- `confusion/confusion_<dataset>_<model>.png` – Confusion Matrix\n")
    rep.append("- `metrics/metrics_<dataset>_<model>.png` – F1, Precision, Recall pro Kombination\n")
    rep.append("- `importance/<model>/<dataset>.png` – Feature-Importance (Top-K)\n")
    rep.append("- `model_specific/model_specific_<dataset>_<model>.png` – Modell-spezifische Visualisierung\n")
    if extract_from_raw:
        rep.append("- `raw/raw_sensor_correlation.png` – Korrelation der Rohdaten-Sensoren\n")
    if use_loo and loo_per_rec:
        rep.append("- `loo/loo_recording_heatmap.png` – LOO: Korrekt pro Recording\n")
        rep.append("- `loo/loo_accuracy_per_combo.png` – LOO Recording-Accuracy pro Kombination\n")
    (out_dir / "modellvergleich_uebersicht.md").write_text("".join(rep), encoding="utf-8")
    (out_dir / "run_report.md").write_text("".join(rep), encoding="utf-8")
