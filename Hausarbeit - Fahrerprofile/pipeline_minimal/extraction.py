"""
Feature-Extraktion aus Rohdaten – eigenständige Implementierung für pipeline_minimal_beispiel.

Enthält alle Extraktionslogik:
- FSChatGPT, FSGemini, auto (tsfresh), featuretools
- merge_feature_sources, select_top_features
- load_signals, load_or_extract_features
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

EPS = 1e-12
KEY_COLS = ["driver_id", "recording", "window_start_s", "window_end_s"]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx < EPS or sy < EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spectral_centroid_flatness(x: np.ndarray, fs_hz: float) -> tuple[float, float]:
    m = np.isfinite(x)
    x = x[m]
    if len(x) < 32 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return (np.nan, np.nan)
    x = x - np.mean(x)
    X = rfft(x)
    p = (np.abs(X) ** 2) + EPS
    f = rfftfreq(len(x), d=1.0 / fs_hz)
    valid = f > 0
    f, p = f[valid], p[valid]
    if len(f) == 0:
        return (np.nan, np.nan)
    centroid = float(np.sum(f * p) / (np.sum(p) + EPS))
    flatness = float(np.exp(np.mean(np.log(p))) / (np.mean(p) + EPS))
    return centroid, flatness


def _wavelet_energy_ratios(x: np.ndarray) -> dict[str, float]:
    try:
        import pywt  # type: ignore
    except Exception:
        return {"wav_d1_ratio": np.nan, "wav_d2_ratio": np.nan, "wav_d3_ratio": np.nan}
    m = np.isfinite(x)
    x = x[m]
    if len(x) < 64:
        return {"wav_d1_ratio": np.nan, "wav_d2_ratio": np.nan, "wav_d3_ratio": np.nan}
    x = x - np.mean(x)
    wavelet = "db4"
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    level = int(min(3, max_level)) if max_level >= 1 else 0
    if level < 1:
        return {"wav_d1_ratio": np.nan, "wav_d2_ratio": np.nan, "wav_d3_ratio": np.nan}
    coeffs = pywt.wavedec(x, wavelet, level=level)
    details = coeffs[1:]
    energies = [float(np.sum(np.square(d))) for d in details]
    total = float(np.sum(energies)) + EPS
    out = {"wav_d1_ratio": np.nan, "wav_d2_ratio": np.nan, "wav_d3_ratio": np.nan}
    if len(details) >= 1:
        out["wav_d1_ratio"] = float(energies[-1] / total)
    if len(details) >= 2:
        out["wav_d2_ratio"] = float(energies[-2] / total)
    if len(details) >= 3:
        out["wav_d3_ratio"] = float(energies[-3] / total)
    return out


def _state_transitions(states: np.ndarray, n_states: int) -> tuple[np.ndarray, int]:
    if len(states) < 2:
        return (np.zeros((n_states, n_states), dtype=int), 0)
    tm = np.zeros((n_states, n_states), dtype=int)
    n, prev = 0, int(states[0])
    for s in states[1:]:
        s = int(s)
        tm[prev, s] += 1
        if s != prev:
            n += 1
        prev = s
    return tm, n


def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def _as_float_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _parse_vec3(series: pd.Series) -> np.ndarray:
    if series is None:
        return np.empty((0, 3), dtype=float)
    parts = series.astype(str).str.split(",", n=2, expand=True)
    if parts.shape[1] != 3:
        return np.full((len(series), 3), np.nan, dtype=float)
    out = np.empty((len(series), 3), dtype=float)
    out[:, 0] = pd.to_numeric(parts[0], errors="coerce").to_numpy(dtype=float)
    out[:, 1] = pd.to_numeric(parts[1], errors="coerce").to_numpy(dtype=float)
    out[:, 2] = pd.to_numeric(parts[2], errors="coerce").to_numpy(dtype=float)
    return out


def _nan_stats(x: np.ndarray) -> dict[str, float]:
    m = _finite(x)
    if m.sum() == 0:
        return {"mean": np.nan, "std": np.nan, "q95": np.nan}
    xf = x[m]
    return {"mean": float(np.mean(xf)), "std": float(np.std(xf, ddof=0)), "q95": float(np.quantile(xf, 0.95))}


def _zero_crossings(x: np.ndarray) -> int:
    m = _finite(x)
    x = x[m]
    if len(x) < 2:
        return 0
    s = np.sign(x)
    s[s == 0] = np.nan
    valid = np.isfinite(s[:-1]) & np.isfinite(s[1:])
    return int(np.sum(s[:-1][valid] * s[1:][valid] < 0))


def _run_lengths_between_sign_changes(x: np.ndarray) -> np.ndarray:
    m = _finite(x)
    x = x[m]
    if len(x) < 2:
        return np.array([], dtype=int)
    s = np.sign(x)
    s[s == 0] = 0
    changes = np.where(np.diff(s) != 0)[0] + 1
    if len(changes) == 0:
        return np.array([len(x)], dtype=int)
    idx = np.r_[0, changes, len(x)]
    return np.diff(idx)


def _gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.full_like(x, np.nan, dtype=float)
    return np.gradient(x, t)


def _third_derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = _gradient(x, t)
    a = _gradient(v, t)
    return _gradient(a, t)


def _robust_peak_count(x: np.ndarray) -> int:
    m = _finite(x)
    x = np.abs(x[m])
    if len(x) < 5:
        return 0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    thr = med + 6.0 * mad
    peaks, _ = find_peaks(x, height=thr, distance=max(1, len(x) // 200))
    return int(len(peaks))


def _downsample_for_entropy(x: np.ndarray, max_points: int = 250) -> np.ndarray:
    m = _finite(x)
    x = x[m]
    if len(x) <= 2:
        return x
    step = max(1, len(x) // max_points)
    return x[::step]


def _approximate_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = _downsample_for_entropy(x)
    n = len(x)
    if n <= m + 1:
        return np.nan
    if r is None:
        r = 0.2 * (np.std(x) + EPS)

    def _phi(mm: int) -> float:
        X = np.array([x[i : i + mm] for i in range(n - mm + 1)])
        d = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        C = np.mean(d <= r, axis=0)
        return float(np.mean(np.log(C + EPS)))

    return _phi(m) - _phi(m + 1)


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = _downsample_for_entropy(x)
    n = len(x)
    if n <= m + 1:
        return np.nan
    if r is None:
        r = 0.2 * (np.std(x) + EPS)

    def _count(mm: int) -> int:
        X = np.array([x[i : i + mm] for i in range(n - mm + 1)])
        d = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        np.fill_diagonal(d, np.inf)
        return int(np.sum(d <= r))

    B, A = _count(m), _count(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log((A + EPS) / (B + EPS)))


def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = _downsample_for_entropy(x)
    n = len(x)
    if n < (order - 1) * delay + 2:
        return np.nan
    patterns: dict[tuple[int, ...], int] = {}
    count = 0
    for i in range(n - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        if not np.all(np.isfinite(window)):
            continue
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
        count += 1
    if count == 0:
        return np.nan
    p = np.array(list(patterns.values()), dtype=float) / float(count)
    H = -np.sum(p * np.log(p + EPS))
    Hmax = np.log(math.factorial(order))
    return float(H / (Hmax + EPS))


def _fft_band_energy_ratio(x: np.ndarray, fs_hz: float, f_lo: float, f_hi: float) -> tuple[float, float]:
    m = _finite(x)
    x = x[m]
    if len(x) < 16 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return (np.nan, np.nan)
    x = x - np.mean(x)
    X = rfft(x)
    pxx = (np.abs(X) ** 2) / len(x)
    freqs = rfftfreq(len(x), d=1.0 / fs_hz)
    valid = freqs > 0
    freqs, pxx = freqs[valid], pxx[valid]
    if len(freqs) == 0:
        return (np.nan, np.nan)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    band_energy = float(np.sum(pxx[band]))
    total_energy = float(np.sum(pxx)) + EPS
    return (band_energy, float(band_energy / total_energy))


def _cross_corr_max(x: np.ndarray, y: np.ndarray, max_lag_samples: int) -> tuple[float, int]:
    m = _finite(x) & _finite(y)
    x, y = x[m], y[m]
    if len(x) < 16 or len(y) != len(x):
        return (np.nan, 0)
    x, y = (x - np.mean(x)) / (np.std(x) + EPS), (y - np.mean(y)) / (np.std(y) + EPS)
    best, best_lag = None, 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        xs = x[:lag] if lag < 0 else x[lag:]
        ys = y[-lag:] if lag < 0 else y[:-lag]
        if lag == 0:
            xs, ys = x, y
        if len(xs) < 8:
            continue
        c = float(np.mean(xs * ys))
        if best is None or abs(c) > abs(best):
            best, best_lag = c, lag
    return (float(best) if best is not None else np.nan, int(best_lag))


def _conditional_entropy_discrete(target: np.ndarray, cond: np.ndarray, bins: int = 10) -> float:
    m = _finite(target) & _finite(cond)
    t, c = target[m], cond[m]
    if len(t) < 50:
        return np.nan
    try:
        t_edges = np.quantile(t, np.linspace(0, 1, bins + 1))
        c_edges = np.quantile(c, np.linspace(0, 1, bins + 1))
    except Exception:
        return np.nan
    t_edges, c_edges = np.unique(t_edges), np.unique(c_edges)
    if len(t_edges) < 3 or len(c_edges) < 3:
        return np.nan
    t_bin = np.digitize(t, t_edges[1:-1], right=True)
    c_bin = np.digitize(c, c_edges[1:-1], right=True)
    joint = np.zeros((t_bin.max() + 1, c_bin.max() + 1), dtype=float)
    for tb, cb in zip(t_bin, c_bin, strict=False):
        joint[int(tb), int(cb)] += 1.0
    joint /= float(np.sum(joint) + EPS)
    p_c = np.sum(joint, axis=0) + EPS
    p_t_given_c = joint / p_c[None, :]
    return float(-np.sum(joint * np.log(p_t_given_c + EPS)))


def _gas_to_brake_reaction_times(t: np.ndarray, gas: np.ndarray, brake: np.ndarray) -> np.ndarray:
    m = _finite(t) & _finite(gas) & _finite(brake)
    t, gas, brake = t[m], gas[m], brake[m]
    if len(t) < 10:
        return np.array([], dtype=float)
    gas_thr, brake_thr = 0.05, 0.05
    gas_down = np.where((gas[:-1] >= gas_thr) & (gas[1:] < gas_thr))[0] + 1
    brake_up = np.where((brake[:-1] < brake_thr) & (brake[1:] >= brake_thr))[0] + 1
    if len(gas_down) == 0 or len(brake_up) == 0:
        return np.array([], dtype=float)
    bt = t[brake_up]
    out: list[float] = []
    for idx in gas_down:
        t0 = t[idx]
        j = np.searchsorted(bt, t0, side="left")
        if j < len(bt) and bt[j] - t0 <= 2.0:
            out.append(float(bt[j] - t0))
    return np.array(out, dtype=float)


@dataclass(frozen=True)
class Signals:
    t: np.ndarray
    steer: np.ndarray
    gas: np.ndarray
    brake: np.ndarray
    lin_acc: np.ndarray | None = None
    rot_vel: np.ndarray | None = None
    speed: np.ndarray | None = None
    lane_dev: np.ndarray | None = None


def _choose_first_available(df: pd.DataFrame, cols: list[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None


def load_signals(csv_path: Path) -> Signals:
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns.tolist())
    vec_cols = [c for c in ["lin_acc", "rot_vel", "car0_vehicle_pos", "rrp_pos"] if c in cols]
    scalar_cols = [
        c for c in [
            "timestamp", "wheel_position", "car0_throttle_position", "car0_brake_position",
            "throttle", "brakes", "car0_velocity", "car0_velocity_vehicle",
        ] if c in cols
    ]
    usecols = scalar_cols + vec_cols
    dtype: dict[str, Any] = {c: "string" for c in vec_cols}
    for c in scalar_cols:
        dtype[c] = "float64"
    df = pd.read_csv(csv_path, skiprows=[1], usecols=usecols, dtype=dtype, na_values=["-"], engine="python")
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {csv_path}")
    t = _as_float_array(df["timestamp"])
    order = np.argsort(t)
    t = t[order]
    steer_col = _choose_first_available(df, ["wheel_position"])
    if steer_col is None:
        raise ValueError(f"Missing wheel_position in {csv_path}")
    steer = _as_float_array(df[steer_col])[order]
    gas_col = _choose_first_available(df, ["car0_throttle_position", "throttle"])
    brake_col = _choose_first_available(df, ["car0_brake_position", "brakes"])
    gas = _as_float_array(df[gas_col])[order] if gas_col else np.full_like(t, np.nan)
    brake = _as_float_array(df[brake_col])[order] if brake_col else np.full_like(t, np.nan)
    speed_col = _choose_first_available(df, ["car0_velocity_vehicle", "car0_velocity"])
    speed = _as_float_array(df[speed_col])[order] if speed_col else None
    lin_acc = _parse_vec3(df["lin_acc"])[order] if "lin_acc" in df.columns else None
    rot_vel = _parse_vec3(df["rot_vel"])[order] if "rot_vel" in df.columns else None
    lane_dev = None
    if "rrp_pos" in df.columns and "car0_vehicle_pos" in df.columns:
        rrp = _parse_vec3(df["rrp_pos"])[order]
        pos = _parse_vec3(df["car0_vehicle_pos"])[order]
        lane_dev = (pos[:, 0] - rrp[:, 0]).astype(float)
    m = np.isfinite(t)
    t, steer, gas, brake = t[m], steer[m], gas[m], brake[m]
    if speed is not None:
        speed = speed[m]
    if lin_acc is not None:
        lin_acc = lin_acc[m]
    if rot_vel is not None:
        rot_vel = rot_vel[m]
    if lane_dev is not None:
        lane_dev = lane_dev[m]
    if len(t) >= 2:
        dup = np.r_[False, np.diff(t) <= 0]
        if np.any(dup):
            keep = ~dup
            t, steer, gas, brake = t[keep], steer[keep], gas[keep], brake[keep]
            if speed is not None:
                speed = speed[keep]
            if lin_acc is not None:
                lin_acc = lin_acc[keep]
            if rot_vel is not None:
                rot_vel = rot_vel[keep]
            if lane_dev is not None:
                lane_dev = lane_dev[keep]
    return Signals(t=t, steer=steer, gas=gas, brake=brake, lin_acc=lin_acc, rot_vel=rot_vel, speed=speed, lane_dev=lane_dev)


def _compute_yaw_rate(sig: Signals) -> np.ndarray | None:
    if sig.rot_vel is None:
        return None
    rv = sig.rot_vel
    if rv.ndim != 2 or rv.shape[1] < 2:
        return None
    return rv[:, 1].astype(float)


def _iter_windows(t: np.ndarray, window_s: float, step_s: float, min_samples: int) -> list[tuple[int, int, float, float]]:
    if len(t) < min_samples:
        return []
    t0, t1 = float(t[0]), float(t[-1])
    starts = np.arange(t0, t1 - window_s + EPS, step_s, dtype=float)
    out: list[tuple[int, int, float, float]] = []
    for s in starts:
        i0 = int(np.searchsorted(t, s, side="left"))
        i1 = int(np.searchsorted(t, s + window_s, side="left"))
        if i1 - i0 < min_samples:
            continue
        out.append((i0, i1, float(s), float(s + window_s)))
    return out


def extract_window_features(sig: Signals, i0: int, i1: int) -> dict[str, float]:
    t = sig.t[i0:i1]
    steer, gas, brake = sig.steer[i0:i1], sig.gas[i0:i1], sig.brake[i0:i1]
    if len(t) < 10:
        return {}
    dt = np.diff(t)
    fs = float(1.0 / (np.median(dt[dt > 0]) + EPS)) if np.any(dt > 0) else np.nan
    out: dict[str, float] = {}
    steer_jerk = _third_derivative(steer, t)
    st = _nan_stats(steer_jerk)
    out["steer_jerk_mean"], out["steer_jerk_std"] = st["mean"], st["std"]
    out["steer_jerk_abs_q95"] = float(_nan_stats(np.abs(steer_jerk))["q95"])
    out["steer_jerk_extreme_peaks"] = float(_robust_peak_count(steer_jerk))
    out["steer_sampen"] = _sample_entropy(steer)
    out["steer_appen"] = _approximate_entropy(steer)
    out["steer_permen"] = _permutation_entropy(steer, order=3, delay=1)
    band_e, band_ratio = _fft_band_energy_ratio(steer, fs_hz=fs, f_lo=0.3, f_hi=1.0)
    out["steer_hf_energy_0p3_1hz"], out["steer_hf_energy_ratio_0p3_1hz"] = band_e, band_ratio
    dsteer = np.diff(steer)
    out["steer_delta_zero_crossings"] = float(_zero_crossings(dsteer))
    runlens = _run_lengths_between_sign_changes(dsteer)
    out["steer_mean_correction_len_samples"] = float(np.mean(runlens)) if len(runlens) else np.nan
    out["steer_std_correction_len_samples"] = float(np.std(runlens)) if len(runlens) else np.nan
    out["steer_mean_time_between_dir_changes_s"] = float(np.mean(runlens) / (fs + EPS)) if len(runlens) and np.isfinite(fs) else np.nan
    out["steer_path_len"] = float(np.nansum(np.abs(np.diff(steer))))
    gas_jerk = _third_derivative(gas, t)
    brk_jerk = _third_derivative(brake, t)
    out["gas_jerk_std"] = float(_nan_stats(gas_jerk)["std"])
    out["brake_jerk_std"] = float(_nan_stats(brk_jerk)["std"])
    out["gas_jerk_extreme_peaks"] = float(_robust_peak_count(gas_jerk))
    out["brake_jerk_extreme_peaks"] = float(_robust_peak_count(brk_jerk))
    out["gas_sampen"] = _sample_entropy(gas)
    out["gas_appen"] = _approximate_entropy(gas)
    out["gas_permen"] = _permutation_entropy(gas, order=3, delay=1)
    out["brake_sampen"] = _sample_entropy(brake)
    out["brake_appen"] = _approximate_entropy(brake)
    out["brake_permen"] = _permutation_entropy(brake, order=3, delay=1)
    rt = _gas_to_brake_reaction_times(t, gas, brake)
    out["gas_to_brake_rt_mean_s"] = float(np.mean(rt)) if len(rt) else np.nan
    out["gas_to_brake_rt_std_s"] = float(np.std(rt)) if len(rt) else np.nan
    out["gas_to_brake_rt_count"] = float(len(rt))
    if len(gas) >= 2:
        dg = np.diff(gas)
        out["gas_delta_rms"] = float(np.sqrt(np.mean(dg * dg)))
        out["gas_delta_strong_ratio"] = float(np.mean(np.abs(dg) > np.quantile(np.abs(dg), 0.9) + EPS))
    else:
        out["gas_delta_rms"] = out["gas_delta_strong_ratio"] = np.nan
    if len(brake) >= 2:
        db = np.diff(brake)
        out["brake_delta_rms"] = float(np.sqrt(np.mean(db * db)))
        out["brake_delta_strong_ratio"] = float(np.mean(np.abs(db) > np.quantile(np.abs(db), 0.9) + EPS))
    else:
        out["brake_delta_rms"] = out["brake_delta_strong_ratio"] = np.nan
    if sig.lin_acc is not None:
        acc = sig.lin_acc[i0:i1]
        acc_h = np.sqrt(acc[:, 0] ** 2 + acc[:, 2] ** 2)
        out["lin_acc_h_rms"] = float(np.sqrt(np.nanmean(acc_h * acc_h)))
        out["lin_acc_h_std"] = float(np.nanstd(acc_h))
        out["lin_acc_h_permen"] = _permutation_entropy(acc_h, order=3, delay=1)
    else:
        out["lin_acc_h_rms"] = out["lin_acc_h_std"] = out["lin_acc_h_permen"] = np.nan
    if sig.rot_vel is not None:
        yaw_rate = sig.rot_vel[i0:i1][:, 1]
        out["yaw_rate_std"] = float(np.nanstd(yaw_rate))
        out["yaw_rate_sampen"] = _sample_entropy(yaw_rate)
    else:
        out["yaw_rate_std"] = out["yaw_rate_sampen"] = np.nan
    if sig.lane_dev is not None:
        ld = sig.lane_dev[i0:i1]
        out["lane_dev_var"] = float(np.nanvar(ld))
        out["lane_dev_sampen"] = _sample_entropy(ld)
        out["lane_dev_permen"] = _permutation_entropy(ld, order=3, delay=1)
        lane_rms = float(np.sqrt(np.nanmean(ld * ld)))
        out["correction_efficiency_steer_per_lane_rms"] = float(out["steer_path_len"] / (lane_rms + EPS))
        ld2 = ld.copy()
        ld2[np.abs(ld2) < np.nanquantile(np.abs(ld2), 0.2)] = 0.0
        out["lane_overshoot_sign_changes"] = float(_zero_crossings(np.diff(ld2)))
        max_lag = int((2.0 * fs)) if np.isfinite(fs) else 0
        c, lag = _cross_corr_max(steer, ld, max_lag_samples=max(1, max_lag))
        out["steer_lane_xcorr_max"] = c
        out["steer_lane_xcorr_lag_s"] = float(lag / (fs + EPS)) if np.isfinite(fs) else np.nan
        steer_l = steer[lag:] if lag > 0 else steer[:lag] if lag < 0 else steer
        ld_l = ld[:-lag] if lag > 0 else ld[-lag:] if lag < 0 else ld
        out["lane_given_steer_cond_entropy"] = _conditional_entropy_discrete(ld_l, steer_l, bins=10)
    else:
        for k in ["lane_dev_var", "lane_dev_sampen", "lane_dev_permen", "correction_efficiency_steer_per_lane_rms",
                  "lane_overshoot_sign_changes", "steer_lane_xcorr_max", "steer_lane_xcorr_lag_s", "lane_given_steer_cond_entropy"]:
            out[k] = np.nan
    max_lag = int((2.0 * fs)) if np.isfinite(fs) else 0
    c_gs, lag_gs = _cross_corr_max(gas, steer, max_lag_samples=max(1, max_lag))
    out["gas_steer_xcorr_max"] = c_gs
    out["gas_steer_xcorr_lag_s"] = float(lag_gs / (fs + EPS)) if np.isfinite(fs) else np.nan
    out["fs_hz_est"] = fs
    out["window_samples"] = float(len(t))
    return out


def extract_window_features_custom2(sig: Signals, i0: int, i1: int) -> dict[str, float]:
    t = sig.t[i0:i1]
    steer, gas, brake = sig.steer[i0:i1], sig.gas[i0:i1], sig.brake[i0:i1]
    if len(t) < 10:
        return {}
    dt = np.diff(t)
    fs = float(1.0 / (np.median(dt[dt > 0]) + EPS)) if np.any(dt > 0) else np.nan
    dur = float(t[-1] - t[0]) if len(t) >= 2 else np.nan
    out: dict[str, float] = {}
    out["fs_hz_est"], out["window_samples"], out["window_dur_s"] = fs, float(len(t)), dur
    out["steer_mean"] = float(np.nanmean(steer))
    out["steer_std"] = float(np.nanstd(steer))
    out["steer_abs_mean"] = float(np.nanmean(np.abs(steer)))
    out["steer_abs_q95"] = float(np.nanquantile(np.abs(steer), 0.95))
    out["steer_range"] = float(np.nanmax(steer) - np.nanmin(steer))
    out["gas_mean"], out["gas_std"] = float(np.nanmean(gas)), float(np.nanstd(gas))
    out["brake_mean"], out["brake_std"] = float(np.nanmean(brake)), float(np.nanstd(brake))
    steer_rate = _gradient(steer, t)
    steer_acc = _gradient(steer_rate, t)
    out["steer_rate_abs_mean"] = float(np.nanmean(np.abs(steer_rate)))
    out["steer_rate_abs_q95"] = float(np.nanquantile(np.abs(steer_rate), 0.95))
    out["steer_rate_std"] = float(np.nanstd(steer_rate))
    out["steer_acc_std"] = float(np.nanstd(steer_acc))
    sr = steer_rate.copy()
    deadband = float(np.nanquantile(np.abs(sr), 0.2)) if np.isfinite(np.nanquantile(np.abs(sr), 0.2)) else 0.0
    sr[np.abs(sr) < max(1e-6, deadband)] = 0.0
    signs = np.sign(sr)
    valid = (signs[:-1] != 0) & (signs[1:] != 0)
    reversals = int(np.sum((signs[:-1][valid] * signs[1:][valid]) < 0))
    out["steer_reversals"] = float(reversals)
    out["steer_reversal_rate_per_s"] = float(reversals / (dur + EPS)) if np.isfinite(dur) and dur > 0 else np.nan
    active_thr = float(np.nanquantile(np.abs(steer_rate), 0.7)) if len(steer_rate) > 5 else np.nan
    out["steer_active_ratio"] = float(np.nanmean(np.abs(steer_rate) > (active_thr + EPS))) if np.isfinite(active_thr) else np.nan
    out["steer_rate_spec_centroid_hz"], out["steer_rate_spec_flatness"] = _spectral_centroid_flatness(steer_rate, fs_hz=fs)
    wav = _wavelet_energy_ratios(steer_rate)
    out["steer_rate_wav_d1_ratio"], out["steer_rate_wav_d2_ratio"], out["steer_rate_wav_d3_ratio"] = wav["wav_d1_ratio"], wav["wav_d2_ratio"], wav["wav_d3_ratio"]
    gas_thr, brake_thr = 0.05, 0.05
    gas_on, brake_on = gas > gas_thr, brake > brake_thr
    overlap, coast = gas_on & brake_on, (~gas_on) & (~brake_on)
    out["gas_on_ratio"] = float(np.mean(gas_on))
    out["brake_on_ratio"] = float(np.mean(brake_on))
    out["pedal_overlap_ratio"] = float(np.mean(overlap))
    out["coast_ratio"] = float(np.mean(coast))
    states = np.zeros(len(t), dtype=int)
    states[gas_on], states[brake_on], states[overlap] = 1, 2, 3
    tm, ntr = _state_transitions(states, n_states=4)
    out["pedal_state_transitions"] = float(ntr)
    out["pedal_state_transitions_per_s"] = float(ntr / (dur + EPS)) if np.isfinite(dur) and dur > 0 else np.nan
    out["trans_gas_to_brake"] = float(tm[1, 2] + tm[1, 3])
    out["trans_brake_to_gas"] = float(tm[2, 1] + tm[2, 3])
    brake_rate = _gradient(brake, t)
    out["brake_rate_max"] = float(np.nanmax(brake_rate))
    out["brake_rate_q95"] = float(np.nanquantile(brake_rate, 0.95))
    onsets = int(np.sum((~brake_on[:-1]) & (brake_on[1:])))
    out["brake_onsets"] = float(onsets)
    out["brake_onsets_per_s"] = float(onsets / (dur + EPS)) if np.isfinite(dur) and dur > 0 else np.nan
    rt = _gas_to_brake_reaction_times(t, gas, brake)
    out["gas_to_brake_rt_mean_s"] = float(np.mean(rt)) if len(rt) else np.nan
    out["gas_to_brake_rt_std_s"] = float(np.std(rt)) if len(rt) else np.nan
    out["gas_to_brake_rt_count"] = float(len(rt))
    yaw = _compute_yaw_rate(sig)
    speed = sig.speed
    if speed is not None:
        sp = speed[i0:i1]
        out["speed_mean"] = float(np.nanmean(sp))
        out["speed_std"] = float(np.nanstd(sp))
        out["speed_q95"] = float(np.nanquantile(sp, 0.95))
        out["steer_rate_abs_mean_per_speed"] = float(np.nanmean(np.abs(steer_rate) / (sp + 0.5)))
        out["steer_abs_mean_per_speed"] = float(np.nanmean(np.abs(steer) / (sp + 0.5)))
    else:
        out["speed_mean"] = out["speed_std"] = out["speed_q95"] = out["steer_rate_abs_mean_per_speed"] = out["steer_abs_mean_per_speed"] = np.nan
    if yaw is not None:
        yw = yaw[i0:i1]
        out["yaw_rate_std"] = float(np.nanstd(yw))
        out["yaw_rate_abs_mean"] = float(np.nanmean(np.abs(yw)))
        out["corr_steer_yaw"] = _corr(steer, yw)
        out["corr_steer_rate_yaw"] = _corr(steer_rate, yw)
        if speed is not None:
            sp = speed[i0:i1]
            curv = np.abs(yw) / (sp + 0.5)
            out["curv_mean"] = float(np.nanmean(curv))
            out["curv_q95"] = float(np.nanquantile(curv, 0.95))
        else:
            out["curv_mean"] = out["curv_q95"] = np.nan
    else:
        out["yaw_rate_std"] = out["yaw_rate_abs_mean"] = out["corr_steer_yaw"] = out["corr_steer_rate_yaw"] = out["curv_mean"] = out["curv_q95"] = np.nan
    if sig.lin_acc is not None:
        acc = sig.lin_acc[i0:i1]
        acc_mag = np.sqrt(np.nansum(acc * acc, axis=1))
        out["acc_mag_rms"] = float(np.sqrt(np.nanmean(acc_mag * acc_mag)))
        out["acc_mag_q95"] = float(np.nanquantile(acc_mag, 0.95))
        out["corr_gas_accmag"] = _corr(gas, acc_mag)
        out["corr_brake_accmag"] = _corr(brake, acc_mag)
    else:
        out["acc_mag_rms"] = out["acc_mag_q95"] = out["corr_gas_accmag"] = out["corr_brake_accmag"] = np.nan
    return out


def extract_features_for_recording(csv_path: Path, driver_id: str, window_s: float, step_s: float, min_samples: int) -> pd.DataFrame:
    sig = load_signals(csv_path)
    rows: list[dict[str, Any]] = []
    windows = _iter_windows(sig.t, window_s=window_s, step_s=step_s, min_samples=min_samples)
    for i0, i1, s, e in windows:
        feats = extract_window_features(sig, i0, i1)
        if not feats:
            continue
        feats["driver_id"] = driver_id
        feats["recording"] = csv_path.name
        feats["window_start_s"] = float(s)
        feats["window_end_s"] = float(e)
        rows.append(feats)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def extract_features_for_recording_custom2(csv_path: Path, driver_id: str, window_s: float, step_s: float, min_samples: int) -> pd.DataFrame:
    sig = load_signals(csv_path)
    rows: list[dict[str, Any]] = []
    windows = _iter_windows(sig.t, window_s=window_s, step_s=step_s, min_samples=min_samples)
    for i0, i1, s, e in windows:
        feats = extract_window_features_custom2(sig, i0, i1)
        if not feats:
            continue
        feats["driver_id"] = driver_id
        feats["recording"] = csv_path.name
        feats["window_start_s"] = float(s)
        feats["window_end_s"] = float(e)
        rows.append(feats)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _tsfresh_long_dataframe(sig: Signals, windows: list[tuple[int, int, float, float]], window_id_offset: int = 0) -> pd.DataFrame:
    yaw_rate = _compute_yaw_rate(sig)
    kinds: list[tuple[str, np.ndarray]] = [("steer", sig.steer), ("gas", sig.gas), ("brake", sig.brake)]
    if sig.speed is not None:
        kinds.append(("speed", sig.speed))
    if yaw_rate is not None:
        kinds.append(("yaw_rate", yaw_rate))
    if sig.lane_dev is not None:
        kinds.append(("lane_dev", sig.lane_dev))
    rows: list[pd.DataFrame] = []
    for w_i, (i0, i1, _ws, _we) in enumerate(windows):
        wid = window_id_offset + w_i
        tt = sig.t[i0:i1]
        if len(tt) < 2:
            continue
        rel_t = tt - tt[0]
        for kind, arr in kinds:
            vals = pd.to_numeric(pd.Series(arr[i0:i1]), errors="coerce").to_numpy(dtype=float)
            rows.append(pd.DataFrame({"id": wid, "time": rel_t, "kind": kind, "value": vals}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["id", "time", "kind", "value"])


def extract_features_auto_tsfresh(
    csv_path: Path,
    driver_id: str,
    window_s: float,
    step_s: float,
    min_samples: int,
    window_id_offset: int = 0,
    fc_params: str = "efficient",
    n_jobs: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig = load_signals(csv_path)
    windows = _iter_windows(sig.t, window_s=window_s, step_s=step_s, min_samples=min_samples)
    if not windows:
        return (pd.DataFrame(), pd.DataFrame())
    long_df = _tsfresh_long_dataframe(sig, windows, window_id_offset=window_id_offset)
    if long_df.empty:
        return (pd.DataFrame(), pd.DataFrame())
    try:
        from tsfresh import extract_features as ts_extract_features
        from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
        from tsfresh.utilities.dataframe_functions import impute as ts_impute
    except Exception as e:
        raise RuntimeError("tsfresh is required for --feature-set auto. Install via: pip install tsfresh") from e
    params = EfficientFCParameters() if fc_params.lower().strip() == "efficient" else MinimalFCParameters()
    feats = ts_extract_features(long_df, column_id="id", column_sort="time", column_kind="kind", column_value="value",
                                default_fc_parameters=params, n_jobs=int(n_jobs), disable_progressbar=True)
    ts_impute(feats)
    meta_rows = [{"id": window_id_offset + w_i, "driver_id": driver_id, "recording": csv_path.name,
                  "window_start_s": ws, "window_end_s": we} for w_i, (_i0, _i1, ws, we) in enumerate(windows)]
    meta = pd.DataFrame(meta_rows).set_index("id")
    return feats, meta


def _safe_col(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in s)


def _sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n, dtype=int)


def extract_featuretools_features_for_csvs(
    csv_paths: list[Path],
    window_s: float,
    step_s: float,
    min_samples: int,
    max_obs_per_window: int = 500,
    max_depth: int = 1,
    driver_ids: list[str] | None = None,
) -> pd.DataFrame:
    try:
        import featuretools as ft  # type: ignore
    except Exception as e:
        raise RuntimeError("featuretools is not installed. Install via: pip install featuretools") from e
    if driver_ids is not None and len(driver_ids) != len(csv_paths):
        raise ValueError("driver_ids must have the same length as csv_paths (or be None).")
    windows_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    next_window_id = 0
    for i, csv_path in enumerate(csv_paths):
        sig = load_signals(csv_path)
        yaw = _compute_yaw_rate(sig)
        did = driver_ids[i] if driver_ids is not None else "unknown"
        windows = _iter_windows(sig.t, window_s=window_s, step_s=step_s, min_samples=min_samples)
        for i0, i1, ws, we in windows:
            wid = next_window_id
            next_window_id += 1
            windows_rows.append({"window_id": wid, "driver_id": did, "recording": csv_path.name, "window_start_s": ws, "window_end_s": we})
            n = i1 - i0
            idx = _sample_indices(n, max_obs_per_window)
            tt = sig.t[i0:i1][idx]
            rel_t = tt - tt[0]
            steer = sig.steer[i0:i1][idx]
            gas = sig.gas[i0:i1][idx]
            brake = sig.brake[i0:i1][idx]
            speed = sig.speed[i0:i1][idx] if sig.speed is not None else None
            yaw_w = yaw[i0:i1][idx] if yaw is not None else None
            for j in range(len(rel_t)):
                row = {"obs_id": f"{wid}_{j}", "window_id": wid, "time": float(rel_t[j]), "steer": float(steer[j]), "gas": float(gas[j]), "brake": float(brake[j])}
                if speed is not None:
                    row["speed"] = float(speed[j])
                if yaw_w is not None:
                    row["yaw_rate"] = float(yaw_w[j])
                obs_rows.append(row)
    if not windows_rows:
        return pd.DataFrame()
    windows_df = pd.DataFrame(windows_rows)
    obs_df = pd.DataFrame(obs_rows)
    es = ft.EntitySet(id="driving")
    es = es.add_dataframe(dataframe_name="windows", dataframe=windows_df, index="window_id")
    es = es.add_dataframe(dataframe_name="observations", dataframe=obs_df, index="obs_id", time_index="time")
    es = es.add_relationship("windows", "window_id", "observations", "window_id")
    agg_primitives = ["mean", "std", "min", "max", "sum", "skew", "kurtosis"]
    fm, _ = ft.dfs(entityset=es, target_dataframe_name="windows", agg_primitives=agg_primitives, trans_primitives=[], max_depth=max_depth, verbose=False)
    fm = fm.reset_index()
    meta_cols = ["driver_id", "recording", "window_start_s", "window_end_s"]
    if all(c in fm.columns for c in ["window_id"] + meta_cols):
        out = fm
    else:
        meta = windows_df[["window_id"] + meta_cols].copy()
        out = meta.merge(fm, on="window_id", how="left")
    out = out.drop(columns=["window_id"])
    out.columns = [_safe_col(str(c)) for c in out.columns]
    for c in out.columns:
        if c in {"driver_id", "recording"}:
            continue
        try:
            out[c] = pd.to_numeric(out[c])
        except Exception:
            pass
    return out


def infer_driver_id_from_filename(p: Path) -> str:
    stem = p.stem.lower()
    parts = [x for x in re.split(r"[_\-]+", stem) if x]
    scenario_tags = {"night", "day", "test", "city", "highway"}
    for tok in reversed(parts):
        if tok in scenario_tags or tok.isdigit():
            continue
        if re.fullmatch(r"[a-zäöüß]+", tok):
            return tok
    raise ValueError(f"Could not infer driver_id from filename: {p.name}")


def extract_handcrafted(csv_paths: list[Path], driver_ids: list[str], window_s: float, step_s: float, min_samples: int) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []
    for p, did in zip(csv_paths, driver_ids, strict=True):
        df = extract_features_for_recording(p, driver_id=did, window_s=window_s, step_s=step_s, min_samples=min_samples)
        if not df.empty:
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def extract_custom2(csv_paths: list[Path], driver_ids: list[str], window_s: float, step_s: float, min_samples: int) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []
    for p, did in zip(csv_paths, driver_ids, strict=True):
        df = extract_features_for_recording_custom2(p, driver_id=did, window_s=window_s, step_s=step_s, min_samples=min_samples)
        if not df.empty:
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def extract_auto(
    csv_paths: list[Path],
    driver_ids: list[str],
    window_s: float,
    step_s: float,
    min_samples: int,
    auto_select: bool = False,
    auto_params: str = "efficient",
    tsfresh_n_jobs: int = 0,
) -> pd.DataFrame:
    all_feats: list[pd.DataFrame] = []
    all_meta: list[pd.DataFrame] = []
    wid_offset = 0
    for p, did in zip(csv_paths, driver_ids, strict=True):
        feats, meta = extract_features_auto_tsfresh(p, driver_id=did, window_s=window_s, step_s=step_s, min_samples=min_samples,
                                                    window_id_offset=wid_offset, fc_params=auto_params, n_jobs=tsfresh_n_jobs)
        if feats.empty or meta.empty:
            continue
        all_feats.append(feats)
        all_meta.append(meta)
        wid_offset += int(len(meta))
    if not all_feats:
        return pd.DataFrame()
    X = pd.concat(all_feats, axis=0).sort_index()
    meta = pd.concat(all_meta, axis=0).sort_index()
    y = meta["driver_id"].astype(str)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if auto_select:
        try:
            from tsfresh import select_features as ts_select_features
            X = ts_select_features(X, pd.factorize(y.astype(str))[0])
        except Exception:
            pass
    return pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)


def extract_featuretools(
    csv_paths: list[Path],
    driver_ids: list[str],
    window_s: float,
    step_s: float,
    min_samples: int,
    max_obs_per_window: int = 500,
    max_depth: int = 1,
) -> pd.DataFrame:
    return extract_featuretools_features_for_csvs(
        csv_paths=csv_paths, window_s=window_s, step_s=step_s, min_samples=min_samples,
        max_obs_per_window=max_obs_per_window, max_depth=max_depth, driver_ids=driver_ids,
    )


def merge_feature_sources(feature_dfs: dict[str, pd.DataFrame], sources: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for src in sources:
        if src not in feature_dfs:
            raise ValueError(f"Missing source for merge: {src}")
        df = feature_dfs[src].copy()
        missing = [c for c in KEY_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Source {src} missing key cols: {missing}")
        meta = df[KEY_COLS].copy()
        feat_cols = [c for c in df.columns if c not in KEY_COLS]
        feats = df[feat_cols].copy().add_prefix(f"{src}__")
        one = pd.concat([meta, feats], axis=1)
        merged = one if merged is None else merged.merge(one, on=KEY_COLS, how="inner")
    if merged is None:
        raise ValueError("No sources provided for merge")
    return merged


def select_top_features(
    df: pd.DataFrame,
    top_k: int,
    drop_nan_col_thresh: float,
    n_splits: int,
    seed: int,
    n_repeats: int = 1,
    n_estimators: int = 900,
) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedGroupKFold
    meta = df[KEY_COLS].copy()
    meta["group_id"] = meta["driver_id"].astype(str) + "::" + meta["recording"].astype(str)
    y = meta["driver_id"].astype(str)
    X = df.drop(columns=[c for c in KEY_COLS if c in df.columns]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    nan_frac = X.isna().mean()
    keep = nan_frac[nan_frac <= drop_nan_col_thresh].index.tolist()
    X = X[keep]
    groups = meta["group_id"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits_eff = max(2, min(n_splits, n_groups))
    cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    importances = np.zeros(X.shape[1], dtype=float)
    denom = 0
    for rep in range(max(1, n_repeats)):
        rs = seed + 1000 * rep
        for tr, _ in cv.split(X, y, groups=groups):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            imp = SimpleImputer(strategy="median")
            clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=rs, n_jobs=1, class_weight="balanced", max_features="sqrt")
            clf.fit(imp.fit_transform(X_tr), y_tr)
            importances += np.array(clf.feature_importances_, dtype=float)
            denom += 1
    if denom <= 0:
        return pd.DataFrame()
    importances /= float(denom)
    rank = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    sel_cols = rank.index[: min(top_k, len(rank))].tolist()
    return pd.concat([meta[KEY_COLS].copy(), X[sel_cols].copy()], axis=1)


def plot_raw_sensor_correlation_heatmap(
    csv_paths: list,
    out_path: Path,
    title: str = "Korrelation der Rohdaten (Sensoren)",
    max_features: int = 50,
    max_rows_total: int = 15000,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _load_raw():
        dfs = []
        rows_so_far = 0
        for p in csv_paths[:10]:
            if rows_so_far >= max_rows_total:
                break
            try:
                df = pd.read_csv(p, nrows=min(2000, max_rows_total - rows_so_far))
                numeric = df.select_dtypes(include=["number"])
                if not numeric.empty:
                    dfs.append(numeric)
                    rows_so_far += len(numeric)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df = _load_raw()
    if df.empty or len(df.columns) < 2:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Keine numerischen Sensordaten gefunden.", ha="center", va="center")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return
    cols = df.columns.tolist()
    if len(cols) > max_features:
        var = df.var().sort_values(ascending=False)
        cols = var.head(max_features).index.tolist()
        df = df[cols]
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(max(10, corr.shape[0] * 0.3), max(8, corr.shape[1] * 0.3)))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(f"{title} (n={len(cols)} Kanäle)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_or_extract_features(
    root: Path,
    features_dir: Path,
    plots_dir: Path,
    feature_sets: list[str],
    extract_from_raw: bool,
    data_dir: Path,
    labels_file: Path | None = None,
    holdout_file: Path | None = None,
    with_merged: bool = False,
    with_selected: bool = False,
    skip_featuretools: bool = False,
    force: bool = False,
    no_plots: bool = False,
    window_s: float = 20.0,
    step_s: float = 10.0,
    min_samples: int = 300,
    drop_nan_col_thresh: float = 0.7,
    n_splits: int = 5,
    seed: int = 42,
    selected_top_k: int = 60,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_sets = {"FSChatGPT", "FSGemini", "auto", "featuretools"}
    _fs_lower = {str(f).lower() for f in feature_sets}
    _want = lambda n: n.lower() in _fs_lower
    features_by_name: dict[str, pd.DataFrame] = {}
    holdout_features_by_name: dict[str, pd.DataFrame] = {}
    holdout_paths_set: set[Path] = set()
    holdout_paths: list[Path] = []
    holdout_driver_ids: list[str] = []

    data_dir_resolved = (data_dir if data_dir.is_absolute() else root / data_dir).resolve() if extract_from_raw or holdout_file else None
    if holdout_file is not None and data_dir_resolved:
        from labels import load_holdout_file
        h_path = holdout_file if holdout_file.is_absolute() else root / holdout_file
        holdout_paths, holdout_driver_ids = load_holdout_file(h_path, base_dir=data_dir_resolved)
        holdout_paths_set = {p.resolve() for p in holdout_paths}
        n_labeled = sum(1 for d in holdout_driver_ids if d != "__unlabeled__")
        print(f"[holdout] {len(holdout_paths)} Recordings ({n_labeled} mit Label)")

    if extract_from_raw:
        data_dir = data_dir if data_dir.is_absolute() else root / data_dir
        data_dir = data_dir.resolve()

        if labels_file is not None:
            from labels import load_labels_file
            lbl_path = labels_file if labels_file.is_absolute() else root / labels_file
            all_paths, all_ids = load_labels_file(lbl_path, base_dir=data_dir)
            if holdout_paths_set:
                csv_paths = [p for p in all_paths if p.resolve() not in holdout_paths_set]
                driver_ids = [d for p, d in zip(all_paths, all_ids) if p.resolve() not in holdout_paths_set]
                print(f"[extract] {len(csv_paths)} Recordings aus LBL-Datei (Holdout ausgeschlossen), drivers: {sorted(set(driver_ids))}")
            else:
                csv_paths, driver_ids = all_paths, all_ids
                print(f"[extract] {len(csv_paths)} Recordings aus LBL-Datei, drivers: {sorted(set(driver_ids))}")
        else:
            csv_paths = sorted(data_dir.glob("*.csv"))
            csv_paths = [p for p in csv_paths if p.resolve() not in holdout_paths_set]
            if not csv_paths:
                raise FileNotFoundError(f"Keine CSV in {data_dir} (nach Holdout-Ausschluss)")
            driver_ids = [infer_driver_id_from_filename(p) for p in csv_paths]
            print(f"[extract] {len(csv_paths)} Recordings (Holdout ausgeschlossen), drivers: {sorted(set(driver_ids))}")

        if not no_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)
            (plots_dir / "raw").mkdir(parents=True, exist_ok=True)
            plot_raw_sensor_correlation_heatmap(csv_paths, plots_dir / "raw" / "raw_sensor_correlation.png")

        for name, fn in [("FSChatGPT", extract_handcrafted), ("FSGemini", extract_custom2)]:
            if _want(name):
                p = features_dir / f"features_daten_{name}.csv"
                if p.exists() and not force:
                    features_by_name[name] = pd.read_csv(p)
                else:
                    df = fn(csv_paths, driver_ids, window_s, step_s, min_samples)
                    if not df.empty:
                        features_dir.mkdir(parents=True, exist_ok=True)
                        df.to_csv(p, index=False)
                        features_by_name[name] = df

        if _want("auto"):
            p = features_dir / "features_daten_auto.csv"
            if p.exists() and not force:
                features_by_name["auto"] = pd.read_csv(p)
            else:
                df = extract_auto(csv_paths, driver_ids, window_s, step_s, min_samples)
                if not df.empty:
                    features_dir.mkdir(parents=True, exist_ok=True)
                    df.to_csv(p, index=False)
                    features_by_name["auto"] = df

        if _want("featuretools") and not skip_featuretools:
            p = features_dir / "features_daten_featuretools.csv"
            if p.exists() and not force:
                features_by_name["featuretools"] = pd.read_csv(p)
            else:
                try:
                    df = extract_featuretools(csv_paths, driver_ids, window_s, step_s, min_samples)
                    if not df.empty:
                        features_dir.mkdir(parents=True, exist_ok=True)
                        df.to_csv(p, index=False)
                        features_by_name["featuretools"] = df
                except Exception as e:
                    print(f"  [skip] featuretools: {e}")

        want_merged = with_merged or _want("merged_all") or _want("selected")
        if want_merged and len(features_by_name) >= 2:
            src = [s for s in base_sets if s in features_by_name]
            merged = merge_feature_sources(features_by_name, src)
            features_dir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(features_dir / "features_daten_merged_all.csv", index=False)
            features_by_name["merged_all"] = merged

        if (with_selected or _want("selected")) and "merged_all" in features_by_name:
            sel = select_top_features(features_by_name["merged_all"], selected_top_k, drop_nan_col_thresh, n_splits, seed)
            if not sel.empty:
                sel.to_csv(features_dir / "features_daten_selected.csv", index=False)
                features_by_name["selected"] = sel

        if holdout_paths_set and holdout_paths:
            for name, fn in [("FSChatGPT", extract_handcrafted), ("FSGemini", extract_custom2)]:
                if _want(name):
                    df_h = fn(holdout_paths, holdout_driver_ids, window_s, step_s, min_samples)
                    if not df_h.empty:
                        holdout_features_by_name[name] = df_h
            if _want("auto"):
                df_h = extract_auto(holdout_paths, holdout_driver_ids, window_s, step_s, min_samples)
                if not df_h.empty:
                    holdout_features_by_name["auto"] = df_h
            if _want("featuretools") and not skip_featuretools:
                try:
                    df_h = extract_featuretools(holdout_paths, holdout_driver_ids, window_s, step_s, min_samples)
                    if not df_h.empty:
                        holdout_features_by_name["featuretools"] = df_h
                except Exception as e:
                    print(f"  [skip] featuretools (holdout): {e}")
            if "merged_all" in features_by_name and len(holdout_features_by_name) >= 2:
                src = [s for s in base_sets if s in holdout_features_by_name]
                if len(src) >= 2:
                    holdout_features_by_name["merged_all"] = merge_feature_sources(holdout_features_by_name, src)
            if "selected" in features_by_name and "merged_all" in holdout_features_by_name:
                merged_h = holdout_features_by_name["merged_all"]
                train_sel = features_by_name["selected"]
                keep = [c for c in train_sel.columns if c in merged_h.columns]
                if keep:
                    holdout_features_by_name["selected"] = merged_h[keep].copy()
    else:
        _legacy_names = {"fschatgpt": "handcrafted", "fsgemini": "custom2", "FSChatGPT": "handcrafted", "FSGemini": "custom2"}
        for name in feature_sets:
            p = features_dir / f"features_daten_{name}.csv"
            legacy = _legacy_names.get(name) or _legacy_names.get(name.lower() if name else "")
            if not p.exists() and legacy:
                p = features_dir / f"features_daten_{legacy}.csv"
            if p.exists():
                features_by_name[name] = pd.read_csv(p)

    return features_by_name, holdout_features_by_name
