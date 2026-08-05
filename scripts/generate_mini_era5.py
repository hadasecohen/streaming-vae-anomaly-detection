#!/usr/bin/env python3
"""
Generate small, fully-synthetic ERA5-shaped datasets for this repo's demo/
verification path (no real ERA5 download needed).

Columns match the real ERA5 single-levels-timeseries export used during
thesis development: valid_time + 13 features (u100, v100, u10, v10, fg10,
d2m, t2m, msl, skt, sp, ssrd, strd, tp). Values are synthetic (diurnal +
seasonal sinusoids plus noise, roughly scaled to realistic ERA5 ranges) —
not sampled from real reanalysis data.

SIZE: 4 years warmup (training) + 4 years test, hourly — matches the
"at least 3-4 full years, test the same length" sizing decision.

Usage (from the project root):
    python scripts/generate_mini_era5.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "era5"

YEAR_HOURS = 8766             # 365.25 days * 24h
WARMUP_END = 4 * YEAR_HOURS    # 35064 — 4 years warmup/training
N_TOTAL = 2 * WARMUP_END       # 70128 — 4 years warmup + 4 years test
SEED = 42

# (mean, daily_amp, annual_amp, noise_std) per feature — rough ERA5 orders of magnitude
FEATURE_PARAMS = {
    "u100": (3.0, 2.0, 1.5, 1.2),
    "v100": (-2.0, 2.0, 1.5, 1.2),
    "u10":  (2.0, 1.5, 1.0, 1.0),
    "v10":  (-1.8, 1.5, 1.0, 1.0),
    "fg10": (6.5, 1.0, 2.0, 0.8),
    "d2m":  (282.0, 3.0, 8.0, 1.0),
    "t2m":  (295.0, 5.0, 10.0, 1.2),
    "msl":  (101100.0, 150.0, 400.0, 60.0),
    "skt":  (300.0, 6.0, 10.0, 1.3),
    "sp":   (100300.0, 150.0, 400.0, 60.0),
    "ssrd": (6.5e5, 6.5e5, 3.0e5, 5.0e4),   # clipped at 0 (night = 0 radiation)
    "strd": (1.35e6, 3.0e4, 8.0e4, 2.0e4),
    "tp":   (5.0e-5, 0.0, 3.0e-5, 4.0e-5),  # clipped at 0
}
FEATURES = list(FEATURE_PARAMS.keys())


def make_base_series(n_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_total)
    hour_of_day = t % 24
    day_of_year = (t // 24) % 365

    cols = {}
    for name, (mean, daily_amp, annual_amp, noise_std) in FEATURE_PARAMS.items():
        daily = daily_amp * np.sin(2 * np.pi * hour_of_day / 24)
        annual = annual_amp * np.sin(2 * np.pi * day_of_year / 365)
        noise = rng.normal(0, noise_std, size=n_total)
        series = mean + daily + annual + noise
        if name in ("ssrd", "strd", "tp"):
            series = np.clip(series, 0, None)
        cols[name] = series.astype("float32")

    valid_time = pd.date_range("2000-01-01", periods=n_total, freq="h")
    df = pd.DataFrame(cols, index=valid_time)
    df.index.name = "valid_time"
    return df


def inject_point(df: pd.DataFrame, warmup_end: int, rng, n_spikes: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    y = np.zeros(len(df), dtype="int8")
    candidates = np.arange(warmup_end + 20, len(df) - 5)
    idxs = rng.choice(candidates, size=n_spikes, replace=False)
    for i in idxs:
        direction = rng.choice([-1, 1], size=len(FEATURES))
        for j, name in enumerate(FEATURES):
            sigma = FEATURE_PARAMS[name][3]
            col = df.columns.get_loc(name)
            df.iloc[i, col] = np.float32(df.iloc[i, col] + direction[j] * 4.0 * sigma)
        y[i] = 1
    return df, y


def inject_contextual(df: pd.DataFrame, warmup_end: int, rng, n_blocks: int = 12, block_len: int = 24) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    y = np.zeros(len(df), dtype="int8")
    starts = rng.choice(np.arange(warmup_end + 20, len(df) - block_len - 5), size=n_blocks, replace=False)
    for s in starts:
        e = s + block_len
        # Wrong-phase daily cycle: shift the diurnal pattern by 12h (day/night swapped)
        for name in FEATURES:
            _, daily_amp, _, _ = FEATURE_PARAMS[name]
            shift = daily_amp * np.sin(2 * np.pi * (np.arange(s, e) % 24) / 24 + np.pi)
            col = df.columns.get_loc(name)
            df.iloc[s:e, col] = (df.iloc[s:e, col].to_numpy() + shift).astype("float32")
        y[s:e] = 1
    return df, y


def inject_group(df: pd.DataFrame, warmup_end: int, rng, n_blocks: int = 8, block_len: int = 48) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    y = np.zeros(len(df), dtype="int8")
    starts = rng.choice(np.arange(warmup_end + 20, len(df) - block_len - 5), size=n_blocks, replace=False)
    for s in starts:
        e = s + block_len
        direction = rng.choice([-1, 1])
        for name in FEATURES:
            sigma = FEATURE_PARAMS[name][3]
            col = df.columns.get_loc(name)
            df.iloc[s:e, col] = (df.iloc[s:e, col].to_numpy() + direction * 1.5 * sigma).astype("float32")
        y[s:e] = 1
    return df, y


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    base = make_base_series(N_TOTAL, SEED)

    # clean: mirrors the raw ERA5 download format (no label column at all)
    clean = base.copy()
    clean["latitude"] = -43.5
    clean["longitude"] = 172.5
    clean.to_csv(OUT_DIR / "reanalysis-era5-single-levels-timeseries.csv")

    for name, inject_fn in [
        ("era5_point_rare.csv", inject_point),
        ("era5_contextual_anomalies.csv", inject_contextual),
        ("era5_group_anomalies.csv", inject_group),
    ]:
        df_out, y = inject_fn(base, WARMUP_END, rng)
        df_out["is_anomaly"] = y
        df_out["anomaly_type"] = ""
        df_out.to_csv(OUT_DIR / name)
        print(f"{name}: {len(df_out)} rows, {int(y.sum())} anomalous rows "
              f"({100 * y.sum() / len(y):.2f}%), test anomalies: {int(y[WARMUP_END:].sum())}")

    print(f"\nWrote 4 mini ERA5-shaped CSVs to {OUT_DIR}  (N_TOTAL={N_TOTAL}, WARMUP_END={WARMUP_END})")


if __name__ == "__main__":
    main()
