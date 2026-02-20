"""
Evaluation einer einzelnen Feature-Set/Modell-Kombination für die minimale Pipeline.

Führt CV durch, speichert Modell, erzeugt Plots (Confusion, Metrics, Importance, Modell-spezifisch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from toolkit import MODEL_SPECIFIC_PLOTS, build_pipeline, get_feature_importance, save_model_bundle
from plots import plot_confusion_matrix, plot_feature_importance, plot_metrics_per_combo
from model_plots import plot_model_specific


def run_single_combo(
    featureSetName: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    labels: list[str],
    groups: np.ndarray,
    n_groups: int,
    models_dir: Path,
    plots_dir: Path,
    n_splits_eff: int,
    use_tune: bool,
    model_params: dict,
    seed: int,
    no_plots: bool,
    top_k_importance: int,
) -> tuple[dict, pd.DataFrame | None]:
    """
    Evaluiert eine (featureSetName, model_name)-Kombination per CV, speichert Modell und Plots.

    Returns:
        (summary_row, loo_per_rec_df or None)
    """

    pipe = build_pipeline(model_name, seed=seed, tune=use_tune, model_params=model_params or None)
    cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

    win_true, win_pred = [], []
    rec_true, rec_pred = [], []
    per_rec_rows: list[dict] = []

    for tr_idx, te_idx in cv.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        meta_te = meta.iloc[te_idx]
        groups_te = groups[te_idx]
        groups_tr = groups[tr_idx]

        if use_tune:
            pipe.fit(X_tr, y_tr, clf__groups=groups_tr)
        else:
            pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)

        win_true.extend(y_te.tolist())
        win_pred.extend(pred.tolist())

        clf = pipe.named_steps["clf"]
        inner = getattr(clf, "best_estimator_", clf)
        full_proba = np.zeros((proba.shape[0], len(labels)))
        col_map = {c: i for i, c in enumerate(labels)}
        for j, cls in enumerate(inner.classes_):
            s = str(cls)
            if s in col_map:
                full_proba[:, col_map[s]] = proba[:, j]
        dfp = pd.DataFrame(full_proba, columns=labels)
        dfp["group_id"] = groups_te
        dfp["y_true"] = meta_te["driver_id"].to_numpy()
        agg = dfp.groupby("group_id", sort=False)[labels].mean()
        for gid, row in agg.iterrows():
            yt = dfp[dfp["group_id"] == gid]["y_true"].iloc[0]
            yp = labels[np.argmax(row[labels].values)]
            rec_true.append(yt)
            rec_pred.append(yp)
            n_win = len(dfp[dfp["group_id"] == gid])
            per_rec_rows.append({"group_id": str(gid), "y_true": yt, "y_pred": yp, "correct": yt == yp, "n_windows": n_win})

    window_acc = accuracy_score(win_true, win_pred)
    recording_acc = accuracy_score(rec_true, rec_pred)
    cm = confusion_matrix(rec_true, rec_pred, labels=labels)
    rec_f1 = float(f1_score(rec_true, rec_pred, average="macro", zero_division=0))
    rec_prec = float(precision_score(rec_true, rec_pred, average="macro", zero_division=0))
    rec_rec = float(recall_score(rec_true, rec_pred, average="macro", zero_division=0))
    win_f1 = float(f1_score(win_true, win_pred, average="macro", zero_division=0))
    win_prec = float(precision_score(win_true, win_pred, average="macro", zero_division=0))
    win_rec = float(recall_score(win_true, win_pred, average="macro", zero_division=0))

    if use_tune:
        pipe.fit(X, y, clf__groups=groups)
    else:
        pipe.fit(X, y)
    model_subdir = models_dir / model_name
    model_subdir.mkdir(parents=True, exist_ok=True)
    model_path = model_subdir / f"{featureSetName}_{model_name}.joblib"
    save_model_bundle(model_path, pipe, X.columns.tolist(), labels)

    loo_df = pd.DataFrame(per_rec_rows) if per_rec_rows else None

    if not no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(cm, labels, plots_dir / "confusion" / f"confusion_{featureSetName}_{model_name}.png", title=f"{featureSetName} | {model_name}", acc=recording_acc)
        plot_metrics_per_combo(featureSetName, model_name, rec_f1, rec_prec, rec_rec, plots_dir / "metrics" / f"metrics_{featureSetName}_{model_name}.png")
        imp_df = get_feature_importance(pipe, X.columns.tolist())
        if imp_df is not None and not imp_df.empty:
            imp_subdir = plots_dir / "importance" / model_name
            imp_subdir.mkdir(parents=True, exist_ok=True)
            plot_feature_importance(imp_df, imp_subdir / f"{featureSetName}.png", top_k=top_k_importance, title=f"Top Features: {featureSetName} | {model_name}")
        try:
            plot_model_specific(model_name, pipe, X.columns.tolist(), labels, plots_dir / "model_specific" / f"model_specific_{featureSetName}_{model_name}.png", plot_types=MODEL_SPECIFIC_PLOTS, fs_name=featureSetName, X=X, y=y)
        except Exception as e:
            print(f"  [skip] Modell-Plot {model_name}: {e}")

    summary_row = {
        "dataset": featureSetName,
        "model": model_name,
        "train_windows": int(len(X)),
        "n_recordings": n_groups,
        "n_features": int(X.shape[1]),
        "window_accuracy": window_acc,
        "recording_accuracy": recording_acc,
        "recording_f1": rec_f1,
        "recording_precision": rec_prec,
        "recording_recall": rec_rec,
        "window_f1": win_f1,
        "window_precision": win_prec,
        "window_recall": win_rec,
        "model_file": str(model_path),
    }
    return summary_row, loo_df
