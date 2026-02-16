"""
Minimales studentisches Beispiel der Pipeline – komplett eigenständig.

Keine Abhängigkeit von pipeline_project. Enthält eigene Extraktion, Toolkit, Plots.
Unterstützt dieselben CLI-Argumente und schreibt dieselben Ausgabedateien:
- pipeline_progress.json (Fortschritt)
- metrics_summary.csv, modellvergleich_uebersicht.md, run_report.md
- plots/*, models/*, features/*

Ausführung (Standalone, aus pipeline_minimal_beispiel/):
  python run.py --feature-sets FSChatGPT --models logreg
  python run.py --extract-from-raw --feature-sets FSChatGPT --models logreg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Nur Pipeline-Ordner für lokale Imports (labels, extraction, evaluate, toolkit)
_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))


def _parse_args():
    """CLI-Argumente parsen."""
    ap = argparse.ArgumentParser(description="Minimale Pipeline – gleiche Schnittstelle wie pipeline_project/run.py")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="Ordner mit Recording-CSVs (bei --extract-from-raw ohne --labels) oder Basisverzeichnis für relative Pfade in der LBL-Datei. Relativ zu pipeline_minimal_beispiel/.")
    ap.add_argument("--labels", type=Path, default=None, metavar="LBL", help="Pfad zur LBL-Datei (CSV: File,Label). Dateien mit leerem Label werden übersprungen. Bei --extract-from-raw werden nur diese Dateien verwendet.")
    ap.add_argument("--holdout-file", type=Path, default=None, metavar="LBL", help="Pfad zur Holdout-Datei (CSV: File,Label). Diese Recordings werden vom Training ausgeschlossen; nach dem Training wird eine Vorhersage erstellt. Label optional (für Evaluation). Erfordert --extract-from-raw.")
    ap.add_argument("--extract-from-raw", action="store_true")
    ap.add_argument("--feature-sets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--with-merged", action="store_true")
    ap.add_argument("--with-selected", action="store_true")
    ap.add_argument("--skip-featuretools", action="store_true")
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallele Jobs für Modell-Training (1=sequentiell, -1=alle Kerne)")
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--step-s", type=float, default=10.0)
    ap.add_argument("--min-samples", type=int, default=300)
    ap.add_argument("--drop-nan-col-thresh", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"), help="Ausgabeverzeichnis (relativ zu pipeline_minimal_beispiel/)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--top-k-importance", type=int, default=20)
    ap.add_argument("--selected-top-k", type=int, default=60)
    ap.add_argument("--tune-models", type=str, default="", metavar="NAMES", help="Komma-getrennte Modellnamen für Hyperparameter-Optimierung (z.B. logreg,svm_rbf)")
    ap.add_argument("--model-params", type=str, default=None, metavar="JSON", help="JSON-Dict mit Modell-Parametern pro Modell")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    root = _PIPELINE_DIR
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir = out_dir.resolve()
    features_dir = out_dir / "features"
    models_dir = out_dir / "models"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    # Plots-Unterordner: summary, confusion, metrics, importance, model_specific, feature_correlation, raw, loo
    plot_subdirs = ("summary", "confusion", "metrics", "importance", "model_specific", "feature_correlation", "raw", "loo")
    for d in (out_dir, features_dir, models_dir, plots_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    for sub in plot_subdirs:
        (plots_dir / sub).mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from toolkit import get_all_model_names, prepare_xy
    from extraction import load_or_extract_features
    from evaluate import run_single_combo
    from progress import fmt_combo, write_progress
    from report import write_reports_and_plots

    feature_sets = args.feature_sets or ["FSChatGPT"]
    models = args.models or get_all_model_names()[:2]
    feature_sets = [f.lower().strip() for f in feature_sets]
    models = [m.lower().strip() for m in models]
    tune_models = {x.strip().lower() for x in (args.tune_models or "").split(",") if x.strip()}
    model_params_all: dict[str, dict] = {}
    if args.model_params:
        try:
            model_params_all = json.loads(args.model_params)
        except json.JSONDecodeError:
            pass

    if args.holdout_file and not args.extract_from_raw:
        print("HINWEIS: --holdout-file erfordert --extract-from-raw. Extraktion wird aktiviert.")
        args.extract_from_raw = True

    write_progress(out_dir, "starting", total=0, completed=[], in_progress=[], message="Pipeline wird initialisiert…")

    try:
        features_by_name, holdout_features_by_name = load_or_extract_features(
            root=root,
            features_dir=features_dir,
            plots_dir=plots_dir,
            feature_sets=feature_sets,
            extract_from_raw=args.extract_from_raw,
            data_dir=args.data_dir,
            labels_file=args.labels,
            holdout_file=args.holdout_file,
            with_merged=args.with_merged,
            with_selected=args.with_selected,
            skip_featuretools=args.skip_featuretools,
            force=args.force,
            no_plots=args.no_plots,
            window_s=args.window_s,
            step_s=args.step_s,
            min_samples=args.min_samples,
            drop_nan_col_thresh=args.drop_nan_col_thresh,
            n_splits=args.n_splits,
            seed=args.seed,
            selected_top_k=args.selected_top_k,
        )
    except FileNotFoundError as e:
        print(f"FEHLER: {e}")
        return 1

    if not features_by_name:
        print("FEHLER: Keine Feature-Sets. --extract-from-raw oder Cache nutzen.")
        return 1

    if args.extract_from_raw:
        write_progress(out_dir, "extraction", message="Extraktion abgeschlossen")
    else:
        write_progress(out_dir, "preparing", message="Lade Feature-Sets aus Cache…")

    fs_data: dict[str, tuple] = {}
    for fs_name, df in features_by_name.items():
        try:
            X, y, meta = prepare_xy(df, drop_nan_col_thresh=args.drop_nan_col_thresh)
        except Exception as e:
            print(f"[skip] {fs_name}: {e}")
            continue
        meta = meta.copy()
        meta["group_id"] = meta["driver_id"].astype(str) + "::" + meta["recording"].astype(str)
        groups = meta["group_id"].to_numpy()
        labels = sorted(y.unique().tolist())
        n_groups = len(np.unique(groups))
        fs_data[fs_name] = (X, y, meta, labels, groups, n_groups)
        if not args.no_plots:
            from plots import plot_feature_correlation_heatmap
            plot_feature_correlation_heatmap(X, plots_dir / "feature_correlation" / f"feature_correlation_{fs_name}.png", title=f"Feature-Korrelation: {fs_name}", max_features=50)

    tasks = [(fs, m) for fs in fs_data for m in models]
    n_recordings = max((r for _, (_, _, _, _, _, r) in fs_data.items()), default=0)
    write_progress(out_dir, "training", total=len(tasks), completed=[], in_progress=[], message="Starte Modell-Training…")

    summary_rows: list[dict] = []
    loo_per_rec: dict[str, pd.DataFrame] = {}
    n_jobs_models = {"extratrees", "randomforest", "logreg"}

    for fs_name, model_name in tasks:
        write_progress(out_dir, "training", total=len(tasks), completed=[fmt_combo(r["dataset"], r["model"]) for r in summary_rows], in_progress=[fmt_combo(fs_name, model_name)], message=f"Aktuell: {fs_name} | {model_name}")
        X, y, meta, labels, groups, n_groups = fs_data[fs_name]
        use_tune = model_name in tune_models
        mp = dict(model_params_all.get(model_name, {}))
        if model_name in n_jobs_models:
            mp["n_jobs"] = args.n_jobs
        n_splits_eff = max(2, min(args.n_splits, n_groups))
        if args.loo:
            n_splits_eff = n_groups

        row, loo_df = run_single_combo(
            fs_name=fs_name,
            model_name=model_name,
            X=X, y=y, meta=meta, labels=labels, groups=groups, n_groups=n_groups,
            models_dir=models_dir,
            plots_dir=plots_dir,
            n_splits_eff=n_splits_eff,
            use_tune=use_tune,
            model_params=mp if mp else None,
            seed=args.seed,
            no_plots=args.no_plots,
            top_k_importance=args.top_k_importance,
        )
        summary_rows.append(row)
        if loo_df is not None and args.loo:
            loo_per_rec[f"{fs_name}_{model_name}"] = loo_df

    summary = pd.DataFrame(summary_rows).sort_values(["recording_accuracy", "dataset"], ascending=[False, True])

    # Holdout-Vorhersage (nach Training)
    if args.holdout_file and holdout_features_by_name:
        from toolkit import align_X_for_model, load_model_bundle

        write_progress(out_dir, "holdout", message="Holdout-Vorhersage…")
        holdout_rows: list[dict] = []
        for row in summary_rows:
            fs_name, model_name = row["dataset"], row["model"]
            if fs_name not in holdout_features_by_name:
                continue
            model_path = models_dir / model_name / f"{fs_name}_{model_name}.joblib"
            if not model_path.exists():
                continue
            holdout_df = holdout_features_by_name[fs_name]
            pipe, feature_cols, labels = load_model_bundle(model_path)
            try:
                X_ho, meta_ho = align_X_for_model(holdout_df, feature_cols)
            except ValueError:
                continue
            proba = pipe.predict_proba(X_ho)
            meta_ho = meta_ho.copy()
            meta_ho["group_id"] = meta_ho["driver_id"].astype(str) + "::" + meta_ho["recording"].astype(str)
            clf_classes = list(pipe.classes_)
            dfp = pd.DataFrame(proba, columns=clf_classes)
            dfp["group_id"] = meta_ho["group_id"].to_numpy()
            dfp["y_true"] = meta_ho["driver_id"].to_numpy()
            agg = dfp.groupby("group_id", sort=False)[clf_classes].mean()
            for gid, r in agg.iterrows():
                y_pred = clf_classes[np.argmax(r[clf_classes].values)]
                y_true = dfp[dfp["group_id"] == gid]["y_true"].iloc[0]
                rec = str(gid).split("::")[-1] if "::" in str(gid) else str(gid)
                holdout_rows.append({
                    "dataset": fs_name, "model": model_name, "recording": rec,
                    "predicted": y_pred, "true": y_true if y_true != "__unlabeled__" else "",
                    "correct": y_true != "__unlabeled__" and y_true == y_pred,
                })
        if holdout_rows:
            holdout_df_out = pd.DataFrame(holdout_rows)
            holdout_dir = out_dir / "holdout"
            holdout_dir.mkdir(parents=True, exist_ok=True)
            holdout_df_out.to_csv(holdout_dir / "holdout_predictions.csv", index=False)
            n_eval = sum(1 for r in holdout_rows if r["true"])
            n_correct = sum(1 for r in holdout_rows if r.get("correct"))
            print(f"Holdout: {holdout_dir / 'holdout_predictions.csv'} ({n_correct}/{n_eval} korrekt bei {n_eval} gelabelten)")

    write_progress(out_dir, "done", total=len(tasks), completed=[fmt_combo(r["dataset"], r["model"]) for r in summary_rows], in_progress=[], message="Fertig")
    write_reports_and_plots(
        out_dir=out_dir,
        plots_dir=plots_dir,
        summary=summary,
        loo_per_rec=loo_per_rec,
        feature_sets=feature_sets,
        models=models,
        tune_models=tune_models,
        n_recordings=n_recordings,
        n_splits=args.n_splits,
        use_loo=args.loo,
        extract_from_raw=args.extract_from_raw,
        no_plots=args.no_plots,
    )

    write_progress(out_dir, "done", total=len(tasks), completed=[fmt_combo(r["dataset"], r["model"]) for r in summary_rows], in_progress=[], message="Fertig")
    print(f"Ergebnisse: {out_dir / 'metrics_summary.csv'}")
    print(f"Modellvergleich: {out_dir / 'modellvergleich_uebersicht.md'}")
    print(f"Modelle: {models_dir}")
    print(f"Plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
