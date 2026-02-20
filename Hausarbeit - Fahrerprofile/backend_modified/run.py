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
import pandas as pd
from toolkit import get_all_model_names, prepare_xy
from extraction import loadOrExtractFeatures
from evaluate import run_single_combo
from progress import combineFeatureSetAndModelName, writeProgress
from report import write_reports_and_plots

class PlotDirectory: 
    def __init__(self, root : Path):
        self.root = root
        self.summary = root / "summary"
        self.confusion = root / "confusion"
        self.metrics = root / "metrics"
        self.importance = root / "importance"
        self.modelSpecific = root / "modelSpecific"
        self.featureCorrelation = root / "featureCorrelation"
        self.raw = root / "raw"
        self.loo = root / "loo"

    def getAll(self) -> list[Path]:
        return [self.root, self.summary, self.confusion, self.metrics, self.importance, self.modelSpecific, self.featureCorrelation, self.raw, self.loo]

class FileController:

    def __init__(self, root : Path, out : Path):
        self.root = root
        self.out = out
        self.features = self.out / "features"
        self.models = self.out / "models"
        self.logs = self.out / "logs"
        self.plots = PlotDirectory(self.out / "plots")

    def getAll(self) -> list[Path]:
        directories = [self.root, self.out, self.features, self.models, self.logs]
        directories.extend(self.plots.getAll())
        return directories
    
    def createAll(self):
        for dir in self.getAll():
            dir.mkdir(parents=True, exist_ok=True)

def _parseArgs():
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

def initializeFilesystem(args) -> FileController:
    rootDir = Path(__file__).resolve().parent
    if str(rootDir) not in sys.path:
        sys.path.insert(0, str(rootDir))
    
    outputDir = args.out_dir if args.out_dir.is_absolute() else rootDir / args.out_dir
    outputDir = outputDir.resolve()
    fileController = FileController(rootDir, outputDir)
    fileController.createAll()
    return fileController

def readArgs(args) -> tuple[list[str], list[str], set[str], dict[str, dict]]:
    featureSets = args.feature_sets or ["FSChatGPT"]
    featureSets = [f.lower().strip() for f in featureSets]

    models = args.models or get_all_model_names()[:2]
    models = [model.lower().strip() for model in models]

    tuneModels = {model.strip().lower() for model in (args.tune_models or "").split(",") if model.strip()}
    modelParamsAll: dict[str, dict] = {}
    if args.model_params:
        try:
            modelParamsAll = json.loads(args.model_params)
        except json.JSONDecodeError:
            pass

    if args.holdout_file and not args.extract_from_raw:
        print("HINWEIS: --holdout-file erfordert --extract-from-raw. Extraktion wird aktiviert.")
        args.extract_from_raw = True

    return (featureSets, models, tuneModels, modelParamsAll)

def main() -> int:
    args = _parseArgs()

    fileController = initializeFilesystem(args)
    featureSets, models, tuneModels, modelParamsAll = readArgs(args)

    writeProgress(fileController, "starting", total=0, completed=[], in_progress=[], message="Pipeline wird initialisiert…")

    try:
        featuresByName, holdoutFeaturesByName = loadOrExtractFeatures(
            root= fileController.root,
            features_dir= fileController.features,
            plots_dir= fileController.plots.root,
            feature_sets= featureSets,
            extract_from_raw= args.extract_from_raw,
            data_dir= args.data_dir,
            labels_file= args.labels,
            holdout_file= args.holdout_file,
            with_merged= args.with_merged,
            with_selected= args.with_selected,
            skip_featuretools= args.skip_featuretools,
            force= args.force,
            no_plots= args.no_plots,
            window_s= args.window_s,
            step_s= args.step_s,
            min_samples= args.min_samples,
            drop_nan_col_thresh= args.drop_nan_col_thresh,
            n_splits= args.n_splits,
            seed= args.seed,
            selected_top_k= args.selected_top_k,
        )
    except FileNotFoundError as e:
        print(f"FEHLER: {e}")
        return 1

    if not featuresByName:
        print("FEHLER: Keine Feature-Sets. --extract-from-raw oder Cache nutzen.")
        return 1

    if args.extract_from_raw:
        writeProgress(fileController, "extraction", message="Extraktion abgeschlossen")
    else:
        writeProgress(fileController, "preparing", message="Lade Feature-Sets aus Cache…")

    fs_data: dict[str, tuple] = {}
    for fs_name, df in featuresByName.items():
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
            plot_feature_correlation_heatmap(X, fileController.plots.root / "feature_correlation" / f"feature_correlation_{fs_name}.png", title=f"Feature-Korrelation: {fs_name}", max_features=50)

    tasks = [(fs, m) for fs in fs_data for m in models]
    n_recordings = max((r for _, (_, _, _, _, _, r) in fs_data.items()), default=0)
    writeProgress(fileController, "training", total=len(tasks), completed=[], in_progress=[], message="Starte Modell-Training…")

    summary_rows: list[dict] = []
    loo_per_rec: dict[str, pd.DataFrame] = {}
    n_jobs_models = {"extratrees", "randomforest", "logreg"}

    for featureSetName, modelName in tasks:
        writeProgress(fileController, "training", total=len(tasks), completed=[combineFeatureSetAndModelName(r["dataset"], r["model"]) for r in summary_rows], in_progress=[combineFeatureSetAndModelName(featureSetName, modelName)], message=f"Aktuell: {featureSetName} | {modelName}")
        X, y, meta, labels, groups, n_groups = fs_data[featureSetName]
        use_tune = modelName in tuneModels
        mp = dict(modelParamsAll.get(modelName, {}))
        if modelName in n_jobs_models:
            mp["n_jobs"] = args.n_jobs
        n_splits_eff = max(2, min(args.n_splits, n_groups))
        if args.loo:
            n_splits_eff = n_groups

        row, loo_df = run_single_combo(
            featureSetName=featureSetName,
            model_name=modelName,
            X=X, y=y, meta=meta, labels=labels, groups=groups, n_groups=n_groups,
            models_dir=fileController.models,
            plots_dir=fileController.plots.root,
            n_splits_eff=n_splits_eff,
            use_tune=use_tune,
            model_params=mp if mp else None,
            seed=args.seed,
            no_plots=args.no_plots,
            top_k_importance=args.top_k_importance,
        )
        summary_rows.append(row)
        if loo_df is not None and args.loo:
            loo_per_rec[f"{featureSetName}_{modelName}"] = loo_df

    summary = pd.DataFrame(summary_rows).sort_values(["recording_accuracy", "dataset"], ascending=[False, True])

    # Holdout-Vorhersage (nach Training)
    if args.holdout_file and holdoutFeaturesByName:
        from toolkit import align_X_for_model, load_model_bundle

        writeProgress(fileController, "holdout", message="Holdout-Vorhersage…")
        holdout_rows: list[dict] = []
        for row in summary_rows:
            fs_name, model_name = row["dataset"], row["model"]
            if fs_name not in holdoutFeaturesByName:
                continue
            model_path = fileController.models / model_name / f"{fs_name}_{model_name}.joblib"
            if not model_path.exists():
                continue
            holdout_df = holdoutFeaturesByName[fs_name]
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
            holdout_dir = fileController.out / "holdout"
            holdout_dir.mkdir(parents=True, exist_ok=True)
            holdout_df_out.to_csv(holdout_dir / "holdout_predictions.csv", index=False)
            n_eval = sum(1 for r in holdout_rows if r["true"])
            n_correct = sum(1 for r in holdout_rows if r.get("correct"))
            print(f"Holdout: {holdout_dir / 'holdout_predictions.csv'} ({n_correct}/{n_eval} korrekt bei {n_eval} gelabelten)")

    writeProgress(fileController, "done", total=len(tasks), completed=[combineFeatureSetAndModelName(r["dataset"], r["model"]) for r in summary_rows], in_progress=[], message="Fertig")
    write_reports_and_plots(
        out_dir=fileController.out,
        plots_dir=fileController.plots.root,
        summary=summary,
        loo_per_rec=loo_per_rec,
        feature_sets=featureSets,
        models=models,
        tune_models=tuneModels,
        n_recordings=n_recordings,
        n_splits=args.n_splits,
        use_loo=args.loo,
        extract_from_raw=args.extract_from_raw,
        no_plots=args.no_plots,
    )

    writeProgress(fileController, "done", total=len(tasks), completed=[combineFeatureSetAndModelName(r["dataset"], r["model"]) for r in summary_rows], in_progress=[], message="Fertig")
    print(f"Ergebnisse: {fileController.out / 'metrics_summary.csv'}")
    print(f"Modellvergleich: {fileController.out / 'modellvergleich_uebersicht.md'}")
    print(f"Modelle: {fileController.models}")
    print(f"Plots: {fileController.plots.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
