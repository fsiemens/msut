# -*- coding: utf-8 -*-
"""
Modul: features
===============
Feature-Extraktion für die Fahrererkennung. Nutzt Featuretools (Aggregationen über
Beobachtungen pro Fenster) und/oder TSFresh (Zeitreihen-Features). Beide Ansätze
erzeugen einen Feature-Vektor pro Fenster (window_id) mit driver_id und recording.

feature_set: "featuretools" | "tsfresh" | "both"
"""
import numpy as np
import pandas as pd
import featuretools as ft
from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute

from . import config
from .data import load_csv, find_windows, build_window_data


def extract_features(paths, ids, feature_set=config.FEATURE_SET, on_extraction_start=None) -> pd.DataFrame:
    """
    Extrahiert Features aus allen Recordings.

    Args:
        paths: Liste der CSV-Pfade zu den Recordings
        ids: Liste der Fahrer-IDs (parallel zu paths)
        feature_set: "featuretools" | "tsfresh" | "both"
        on_extraction_start: Optionaler Callback, wird sofort beim Start aufgerufen (für pipeline_progress)

    Returns:
        DataFrame mit einer Zeile pro Fenster (Spalten: driver_id, recording, + Feature-Spalten)
    """
    if on_extraction_start:
        on_extraction_start()
    win_df, obs_rows = build_window_data(paths, ids)
    result_ft, result_ts = None, None

    if feature_set in ("featuretools", "both"):
        # EntitySet: Fenster (f) mit Beobachtungen (b) verknüpft über window_id
        obs_df = pd.DataFrame(obs_rows)
        es = ft.EntitySet(id="f").add_dataframe(dataframe_name="f", dataframe=win_df, index="window_id").add_dataframe(dataframe_name="b", dataframe=obs_df, index="obs_id", time_index="time").add_relationship("f", "window_id", "b", "window_id")
        # Aggregationen: mean, std, min, max, sum, skew, kurtosis pro Fenster
        fm, _ = ft.dfs(entityset=es, target_dataframe_name="f", agg_primitives=["mean", "std", "min", "max", "sum", "skew", "kurtosis"], trans_primitives=[], max_depth=1, verbose=False)
        result_ft : pd.DataFrame = win_df.merge(fm.reset_index(), on="window_id", how="left").drop(columns=["window_id"])
        # Spaltennamen bereinigen (sonderzeichen entfernen)
        result_ft.columns = ["".join(c if (c.isalnum() or c in "_-") else "_" for c in str(x)) for x in result_ft.columns]
        for c in result_ft.columns:
            if c not in ("driver_id", "recording"): result_ft[c] = pd.to_numeric(result_ft[c], errors="coerce")
        result_ft["driver_id"], result_ft["recording"] = win_df["driver_id"].values, win_df["recording"].values

    if feature_set in ("tsfresh", "both"):
        # TSFresh erwartet Long-Format: id, time, kind, value
        ts_rows, wid = [], 0
        for i, p in enumerate(paths):
            d = load_csv(p)
            for i0, i1, ws, we in find_windows(d["t"]):
                rel_t = d["t"][i0:i1] - d["t"][i0]
                for sig, arr in [("steer", d["steer"]), ("gas", d["gas"]), ("brake", d["brake"]), ("speed", d["speed"]), ("yaw_rate", d["yaw_rate"])]:
                    ts_rows.append(pd.DataFrame({"id": wid, "time": rel_t, "kind": sig, "value": arr[i0:i1]}))
                wid += 1
        if ts_rows:
            ts_feat = tsfresh_impute(tsfresh_extract(pd.concat(ts_rows, ignore_index=True), column_id="id", column_sort="time", column_kind="kind", column_value="value", default_fc_parameters=MinimalFCParameters(), n_jobs=0, disable_progressbar=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
            result_ts : pd.DataFrame = win_df[["driver_id", "recording"]].copy()
            for c in ts_feat.columns: result_ts[c] = ts_feat[c].values

    # Ergebnis aus beiden Quellen zusammensetzen (bei "both")
    result : pd.DataFrame = result_ft.copy() if feature_set == "both" else (result_ts if feature_set == "tsfresh" else result_ft)
    if feature_set == "both":
        for c in result_ts.columns:
            if c not in ("driver_id", "recording"): result[c] = result_ts[c].values
    return result
