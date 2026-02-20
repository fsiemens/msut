"""Grafiken für pipeline_minimal_beispiel – eigenständig."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
    acc: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    t = title
    if acc is not None:
        t += f" | acc={acc:.3f}"
    ax.set_title(t)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_accuracy_summary(
    df: pd.DataFrame,
    out_path: Path,
    x_col: str = "model",
    y_col: str = "accuracy",
    hue_col: str | None = "dataset",
    title: str = "Accuracy pro Modell und Feature-Set",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if hue_col and hue_col in df.columns:
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    else:
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
    title: str = "Top Feature Importance",
) -> None:
    if importance_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = importance_df.head(top_k)
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.25)))
    ax.barh(range(len(top)), top["importance"].values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_correlation_heatmap(
    X: pd.DataFrame,
    out_path: Path,
    title: str = "Feature-Korrelationsmatrix",
    max_features: int = 50,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X_num = X.select_dtypes(include=["number"])
    if X_num.empty:
        return
    cols = X_num.columns.tolist()
    if len(cols) > max_features:
        var = X_num.var().sort_values(ascending=False)
        cols = var.head(max_features).index.tolist()
        X_num = X_num[cols]
    corr = X_num.corr()
    figsize = max(8, corr.shape[0] * 0.25), max(6, corr.shape[1] * 0.25)
    fig, ax = plt.subplots(figsize=figsize)
    show_labels = len(cols) <= 40
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        xticklabels=cols if show_labels else False,
        yticklabels=cols if show_labels else False,
    )
    ax.set_title(f"{title} (n={len(cols)} Features)")
    if show_labels:
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics_per_combo(
    fs_name: str,
    model_name: str,
    f1: float,
    precision: float,
    recall: float,
    out_path: Path,
    title: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics = ["F1-Score", "Precision", "Recall"]
    raw_values = [f1, precision, recall]
    values = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in raw_values]
    display_values = [max(v, 0.02) if v == 0 else v for v in values]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(metrics, display_values, color=colors, alpha=0.85)
    for bar, v_raw, v_display in zip(bars, raw_values, values):
        label = "n/a" if (v_raw is None or (isinstance(v_raw, float) and np.isnan(v_raw))) else f"{v_display:.3f}"
        y_pos = bar.get_height() + 0.02
        if v_display <= 0:
            y_pos = 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label, ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Wert (Recording-Ebene, macro)")
    ax.set_title(title or f"{fs_name} | {model_name}")
    if all(v <= 0 for v in values):
        ax.text(0.5, 0.5, "Keine Metriken verfügbar\n(evtl. kein predict_proba)", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_f1_precision_recall_summary(
    df: pd.DataFrame,
    out_path: Path,
    row_col: str = "dataset",
    col_col: str = "model",
    metric_cols: tuple[str, str, str] = ("recording_f1", "recording_precision", "recording_recall"),
    title: str = "F1, Precision, Recall pro Modell und Feature-Set",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["F1-Score", "Precision", "Recall"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric_col, label in zip(axes, metric_cols, labels):
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, f"{metric_col} fehlt", ha="center", va="center")
            continue
        sns.barplot(data=df, x=col_col, y=metric_col, hue=row_col, ax=ax)
        ax.set_title(label)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap_metrics_f1_precision_recall(
    df: pd.DataFrame,
    out_path: Path,
    row_col: str = "dataset",
    col_col: str = "model",
    metric_cols: tuple[str, str, str] = ("recording_f1", "recording_precision", "recording_recall"),
    title: str = "F1, Precision, Recall: Dataset x Modell",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["F1-Score", "Precision", "Recall"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric_col, label in zip(axes, metric_cols, labels):
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, f"{metric_col} fehlt", ha="center", va="center")
            continue
        pivot = df.pivot(index=row_col, columns=col_col, values=metric_col)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
        ax.set_title(label)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap_accuracy_matrix(
    df: pd.DataFrame,
    out_path: Path,
    row_col: str = "dataset",
    col_col: str = "model",
    value_col: str = "accuracy",
    title: str = "Accuracy: Dataset x Modell",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index=row_col, columns=col_col, values=value_col)
    fig, ax = plt.subplots(figsize=(6, max(4, pivot.shape[0] * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loo_recording_heatmap(
    loo_per_rec: dict[str, pd.DataFrame],
    out_path: Path,
    title: str = "LOO: Korrekt pro Recording x Kombination",
) -> None:
    if not loo_per_rec:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_groups = set()
    for per_df in loo_per_rec.values():
        all_groups.update(per_df["group_id"].astype(str).tolist())
    all_groups = sorted(all_groups)
    combos = list(loo_per_rec.keys())
    mat = np.zeros((len(all_groups), len(combos)))
    for c, (key, per_df) in enumerate(loo_per_rec.items()):
        for _, row in per_df.iterrows():
            g = str(row["group_id"])
            if g in all_groups:
                r = all_groups.index(g)
                mat[r, c] = 1.0 if row["correct"] else 0.0
    df_plot = pd.DataFrame(mat, index=all_groups, columns=[c.replace("_", " | ") for c in combos])
    fig, ax = plt.subplots(figsize=(max(8, len(combos) * 1.2), max(5, len(all_groups) * 0.3)))
    sns.heatmap(df_plot, cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Korrekt"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Kombination (Feature-Set | Modell)")
    ax.set_ylabel("Recording (group_id)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loo_accuracy_per_combo(
    loo_per_rec: dict[str, pd.DataFrame],
    out_path: Path,
    title: str = "LOO Recording-Accuracy pro Kombination",
) -> None:
    if not loo_per_rec:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combos, accs = [], []
    for key, per_df in loo_per_rec.items():
        if len(per_df) > 0:
            acc = per_df["correct"].mean()
            combos.append(key.replace("_", " | "))
            accs.append(acc)
    fig, ax = plt.subplots(figsize=(max(8, len(combos) * 0.6), 5))
    ax.bar(combos, accs, color="steelblue", alpha=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recording-Accuracy")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
