"""Modell-spezifische Grafiken für pipeline_minimal_beispiel – eigenständig."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _get_inner_estimator(pipe) -> Any:
    clf = pipe.named_steps.get("clf")
    if clf is None:
        return None
    return getattr(clf, "best_estimator_", clf)


def plot_tree_model(pipe, feature_names: list[str], labels: list[str], out_path: Path, title: str = "Entscheidungsbaum", max_depth: int = 5) -> None:
    try:
        from sklearn.tree import plot_tree
    except ImportError:
        return
    inner = _get_inner_estimator(pipe)
    if inner is None:
        return
    tree_obj = inner.estimators_[0] if hasattr(inner, "estimators_") and len(inner.estimators_) > 0 else (inner if hasattr(inner, "tree_") else None)
    if tree_obj is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    fn = feature_names if len(feature_names) <= 50 else None
    plot_tree(tree_obj, max_depth=max_depth, feature_names=fn, class_names=[str(l) for l in labels], filled=True, rounded=True, ax=ax, fontsize=8)
    ax.set_title(f"{title} (max_depth={max_depth}, erster Baum)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_coefficient_model(pipe, feature_names: list[str], labels: list[str], out_path: Path, title: str = "Modell-Koeffizienten", top_k: int = 25) -> None:
    inner = _get_inner_estimator(pipe)
    if inner is None or not hasattr(inner, "coef_"):
        return
    coef = inner.coef_
    imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if len(imp) != len(feature_names):
        return
    n_f = min(len(imp), top_k)
    idx = np.argsort(imp)[-n_f:][::-1]
    imp_s, fn_s = imp[idx], [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, n_f * 0.3)))
    ax.barh(range(n_f), imp_s, color="steelblue", alpha=0.8)
    ax.set_yticks(range(n_f))
    ax.set_yticklabels(fn_s, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("|Koeffizient|")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_centroid_model(pipe, feature_names: list[str], labels: list[str], out_path: Path, title: str = "Centroid pro Klasse", top_k_features: int = 25) -> None:
    inner = _get_inner_estimator(pipe)
    if inner is None or not hasattr(inner, "centroids_"):
        return
    centroids = inner.centroids_
    n_classes, n_feat = centroids.shape
    k = min(top_k_features, n_feat)
    var_per_f = np.var(centroids, axis=0)
    idx = np.argsort(var_per_f)[-k:][::-1]
    C, fn = centroids[:, idx], [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, k * 0.35), max(4, n_classes * 0.4)))
    sns.heatmap(C, xticklabels=fn, yticklabels=[str(l) for l in labels], cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_decision_boundary_2d_generic(pipe, X: pd.DataFrame, y: pd.Series, feature_names: list[str], labels: list[str], out_path: Path, title: str = "Entscheidungsgrenze") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c for c in feature_names if c in X.columns]
    if len(numeric_cols) < 2:
        return
    try:
        from sklearn.feature_selection import f_classif
        X_num = X[numeric_cols].fillna(X[numeric_cols].median())
        _, scores = f_classif(X_num, y)
        idx = np.argsort(scores)[-2:][::-1]
        f1, f2 = numeric_cols[idx[0]], numeric_cols[idx[1]]
    except Exception:
        var = X[numeric_cols].var()
        top2 = var.nlargest(2).index.tolist()
        if len(top2) < 2:
            return
        f1, f2 = top2[0], top2[1]
    x1, x2 = X[f1].to_numpy(), X[f2].to_numpy()
    pad1 = max((x1.max() - x1.min()) * 0.05, 1e-6)
    pad2 = max((x2.max() - x2.min()) * 0.05, 1e-6)
    xx, yy = np.meshgrid(np.linspace(x1.min() - pad1, x1.max() + pad1, 80), np.linspace(x2.min() - pad2, x2.max() + pad2, 80))
    cols = {c: (xx.ravel() if c == f1 else yy.ravel() if c == f2 else np.full(xx.size, X[c].median())) for c in X.columns}
    grid_df = pd.DataFrame(cols)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(grid_df)
        Z = proba[:, 0] if proba.shape[1] > 0 else np.zeros(grid_df.shape[0])
    else:
        pred = pipe.predict(grid_df)
        col_map = {str(l): i for i, l in enumerate(labels)}
        Z = np.array([col_map.get(str(z), 0) / max(len(labels) - 1, 1) for z in pred])
    Z = np.asarray(Z, dtype=float).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, levels=20, cmap="RdYlBu_r", alpha=0.8)
    ax.contour(xx, yy, Z, levels=5, colors="black", linewidths=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, cls in enumerate(labels):
        mask = (y == cls).to_numpy()
        ax.scatter(x1[mask], x2[mask], c=[colors[i]], s=30, alpha=0.8, edgecolors="white", label=cls)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_logreg_combined(pipe, X: pd.DataFrame, y: pd.Series, feature_names: list[str], labels: list[str], out_path: Path, fs_name: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inner = _get_inner_estimator(pipe)
    if inner is None:
        return
    proba = pipe.predict_proba(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_base = f"{fs_name} | logreg" if fs_name else "logreg"
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfekt kalibriert")
    for k, cls in enumerate(inner.classes_):
        pred_p = proba[:, k]
        y_binary = (y == cls).astype(int)
        bins = np.percentile(pred_p, np.linspace(0, 100, 11))
        bc, ba = [], []
        for j in range(len(bins) - 1):
            mask = (pred_p >= bins[j]) & (pred_p < bins[j + 1])
            if mask.sum() > 0:
                bc.append(pred_p[mask].mean())
                ba.append(y_binary[mask].mean())
        if len(bc) > 1:
            ax.plot(bc, ba, "o-", label=str(cls), linewidth=2, markersize=6)
    ax.set_xlabel("Vorhergesagte P(Klasse)")
    ax.set_ylabel("Tatsächlicher Anteil")
    ax.set_title(f"Kalibrierung – {title_base}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal")
    coef = inner.coef_
    imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if len(imp) == len(feature_names):
        idx = np.argsort(imp)[-2:][::-1]
        top2 = [feature_names[i] for i in idx if feature_names[i] in X.columns]
        if len(top2) >= 2:
            f1, f2 = top2[0], top2[1]
            x1, x2 = X[f1].to_numpy(), X[f2].to_numpy()
            pad1, pad2 = (x1.max() - x1.min()) * 0.05, (x2.max() - x2.min()) * 0.05
            xx, yy = np.meshgrid(np.linspace(x1.min() - pad1, x1.max() + pad1, 60), np.linspace(x2.min() - pad2, x2.max() + pad2, 60))
            cols = {c: (xx.ravel() if c == f1 else yy.ravel() if c == f2 else np.full(xx.size, X[c].median())) for c in X.columns}
            grid_df = pd.DataFrame(cols)
            Z = pipe.predict_proba(grid_df)[:, 0]
            Z = Z.reshape(xx.shape)
            ax2 = axes[1]
            ax2.contourf(xx, yy, Z, levels=20, cmap="RdYlBu_r", alpha=0.8)
            ax2.contour(xx, yy, Z, levels=5, colors="black", linewidths=0.5)
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            for i, cls in enumerate(labels):
                mask = (y == cls).to_numpy()
                ax2.scatter(x1[mask], x2[mask], c=[colors[i]], s=25, alpha=0.8, edgecolors="white", label=cls)
            ax2.set_xlabel(f1)
            ax2.set_ylabel(f2)
            ax2.set_title("Entscheidungsgrenze (Top-2 Features)")
            ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_specific(
    model_name: str,
    pipe,
    feature_names: list[str],
    labels: list[str],
    out_path: Path,
    plot_types: dict[str, str],
    fs_name: str = "",
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> None:
    plot_type = plot_types.get(model_name.lower())
    title = f"{fs_name} | {model_name}" if fs_name else model_name
    if plot_type == "tree":
        plot_tree_model(pipe, feature_names, labels, out_path, title=title)
    elif plot_type == "coefficients":
        inner = _get_inner_estimator(pipe)
        if inner is not None and hasattr(inner, "coef_"):
            coef = inner.coef_
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            if len(imp) == len(feature_names):
                plot_coefficient_model(pipe, feature_names, labels, out_path, title=title)
            elif X is not None and y is not None:
                plot_decision_boundary_2d_generic(pipe, X, y, feature_names, labels, out_path, title=title)
    elif plot_type == "logreg_combined":
        if X is not None and y is not None:
            plot_logreg_combined(pipe, X, y, feature_names, labels, out_path, fs_name=fs_name)
    elif plot_type == "centroid":
        plot_centroid_model(pipe, feature_names, labels, out_path, title=title)
    elif plot_type == "decision_boundary_2d":
        if X is not None and y is not None:
            plot_decision_boundary_2d_generic(pipe, X, y, feature_names, labels, out_path, title=title)
