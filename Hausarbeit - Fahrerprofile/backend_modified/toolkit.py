"""
Minimales Pipeline-Toolkit für pipeline_minimal_beispiel – eigenständig, ohne pipeline_project.

Enthält: prepare_xy, save/load_model_bundle, align_X_for_model, build_pipeline,
get_feature_importance, get_all_model_names.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NON_FEATURE_COLS = ["driver_id", "recording", "window_start_s", "window_end_s"]
META_COLS_EXCLUDE = frozenset({"window_samples", "window_dur_s", "fs_hz_est"})

ALL_MODEL_NAMES = ["extratrees", "randomforest", "logreg", "svm_rbf", "nearest_centroid"]
MODEL_SPECIFIC_PLOTS = {
    "extratrees": "tree",
    "randomforest": "tree",
    "logreg": "logreg_combined",
    "svm_rbf": "decision_boundary_2d",
    "nearest_centroid": "centroid",
}


def _apply_model_params(base_kw: dict, model_params: dict) -> dict:
    out = dict(base_kw)
    for k, v in model_params.items():
        if v is not None and str(v).lower() == "none":
            v = None
        elif isinstance(v, str) and k == "gamma" and v not in ("scale", "auto"):
            try:
                v = float(v)
            except ValueError:
                pass
        elif isinstance(v, str) and k == "max_features" and v == "None":
            v = None
        out[k] = v
    return out


def build_pipeline(
    model_name: str,
    seed: int = 42,
    tune: bool = False,
    model_params: dict | None = None,
    n_splits_tune: int = 3,
    **kwargs: object,
) -> Pipeline:
    model_name = model_name.lower().strip()
    params = model_params or {}

    def _wrap_tune(base_clf, tune_grid: dict | None):
        if not tune or not tune_grid:
            return base_clf
        cv = StratifiedGroupKFold(n_splits=min(n_splits_tune, 5), shuffle=True, random_state=seed)
        return GridSearchCV(base_clf, param_grid=tune_grid, cv=cv, scoring="accuracy", n_jobs=1, refit=True)

    if model_name == "extratrees":
        base = {"n_estimators": 500, "random_state": seed, "n_jobs": 1, "class_weight": "balanced"}
        clf = ExtraTreesClassifier(**_apply_model_params(base, params))
        clf = _wrap_tune(clf, {"max_depth": [5, 8, 12], "min_samples_leaf": [2, 5]})
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_name == "randomforest":
        base = {"n_estimators": 500, "random_state": seed, "n_jobs": 1, "class_weight": "balanced"}
        clf = RandomForestClassifier(**_apply_model_params(base, params))
        clf = _wrap_tune(clf, {"max_depth": [5, 8, 12], "min_samples_leaf": [2, 5]})
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_name == "logreg":
        base = {"max_iter": 5000, "class_weight": "balanced", "n_jobs": 1, "solver": "lbfgs"}
        clf = LogisticRegression(**_apply_model_params(base, params))
        clf = _wrap_tune(clf, {"C": [0.1, 1.0, 10.0]})
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "svm_rbf":
        base = {"kernel": "rbf", "C": 10.0, "gamma": "scale", "class_weight": "balanced", "probability": True}
        clf = SVC(**_apply_model_params(base, params))
        clf = _wrap_tune(clf, {"C": [1.0, 10.0], "gamma": ["scale", "auto"]})
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "nearest_centroid":
        base = {"metric": "euclidean", "shrink_threshold": None}
        clf = NearestCentroid(**_apply_model_params(base, params))
        clf = _wrap_tune(clf, {"shrink_threshold": [None, 0.1, 0.5, 1.0]})
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])

    raise ValueError(f"model_name must be one of {ALL_MODEL_NAMES}")


def get_all_model_names() -> list[str]:
    return list(ALL_MODEL_NAMES)


def get_feature_importance(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    clf = pipe.named_steps.get("clf")
    if clf is None:
        return None
    inner = getattr(clf, "best_estimator_", clf)
    if hasattr(inner, "feature_importances_"):
        imp = np.array(inner.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    if hasattr(inner, "coef_"):
        coef = inner.coef_
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        if len(imp) == len(feature_names):
            return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    return None


def prepare_xy(
    df: pd.DataFrame,
    *,
    drop_nan_col_thresh: float = 0.7,
    non_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if non_feature_cols is None:
        non_feature_cols = NON_FEATURE_COLS
    meta = df[["driver_id", "recording", "window_start_s", "window_end_s"]].copy()
    y = df["driver_id"].astype(str)
    drop_cols = [c for c in non_feature_cols if c in df.columns]
    for c in df.columns:
        if c in META_COLS_EXCLUDE or any(c.endswith(f"__{m}") for m in META_COLS_EXCLUDE):
            drop_cols.append(c)
    X = df.drop(columns=list(dict.fromkeys(drop_cols))).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    nan_frac = X.isna().mean()
    keep = nan_frac[nan_frac <= drop_nan_col_thresh].index.tolist()
    X = X[keep]
    return X, y, meta


def save_model_bundle(path: Path, pipe: Pipeline, feature_columns: list[str], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_columns": feature_columns, "labels": labels}, path)


def load_model_bundle(path: Path) -> tuple[Pipeline, list[str], list[str]]:
    bundle = joblib.load(path)
    pipe: Pipeline = bundle["pipeline"]
    cols = list(bundle["feature_columns"])
    labels = list(bundle.get("labels", []))
    return pipe, cols, labels


def align_X_for_model(df_features: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for c in NON_FEATURE_COLS:
        if c not in df_features.columns:
            raise ValueError(f"Missing meta column: {c}")
    meta = df_features[NON_FEATURE_COLS].copy()
    X = df_features.drop(columns=[c for c in NON_FEATURE_COLS if c in df_features.columns]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_columns]
    return X, meta
