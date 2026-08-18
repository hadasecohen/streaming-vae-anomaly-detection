# data_providers.py
from pathlib import Path

from src.Data_tools.era5.era5_adapter import ERA5Adapter
from src.logging_utils import Logger, Verbosity

"""
cfg example (from YAML):
data:
    name: era5
    data_path: ...
    warmup_end: 302227
    args: {scaler_kind: zscore}
"""


def data_fn_builder(cfg, logger: Logger):
    name = cfg["name"].lower()

    if name == "era5":
        return load_era5_dataset(cfg, logger)
    else:
        raise ValueError(f"Unknown dataset name: {cfg['name']}. This curated build only supports 'era5'.")


def _read_era5_raw(cfg):
    """
    Read the raw ERA5 dataframe in either of two equivalent modes, always
    returning rows in the same warmup-then-test order so downstream code
    (column drop/fill, adapter construction) is identical regardless of mode.

    Mode 1 — single file, row-index split:
      data_path    : path to one CSV spanning both warmup and test
      warmup_start : int (default 0) — skip rows before this index
      warmup_end   : int (required) — first row of the test stream

    Mode 2 — two files, pre-split:
      warmup_path  : path to the warmup-only CSV
      test_path    : path to the test-only CSV
      warmup_start : int (default 0) — skip rows before this index within
                     the warmup file (rare; usually left at 0)

    Exactly one mode's keys may be set. The two files in mode 2 must be in
    chronological order with no overlap (warmup's last timestamp before
    test's first) — this is checked explicitly, since a silent gap or
    overlap here would corrupt the warmup/test boundary.

    Returns (df_raw, warmup_start, warmup_length, data_path_display).
    """
    import pandas as pd

    data_path = cfg.get("data_path")
    warmup_path = cfg.get("warmup_path")
    test_path = cfg.get("test_path")
    two_file_mode = warmup_path is not None or test_path is not None

    if data_path is not None and two_file_mode:
        raise ValueError(
            "ERA5 config: set either 'data_path' (single file + warmup_start/warmup_end) "
            "or 'warmup_path'+'test_path' (two files) — not both."
        )

    if two_file_mode:
        if warmup_path is None or test_path is None:
            raise ValueError(
                "ERA5 config: two-file mode requires both 'warmup_path' and 'test_path'."
            )
        warmup_path = Path(warmup_path)
        test_path = Path(test_path)
        warmup_raw = pd.read_csv(warmup_path, parse_dates=["valid_time"], low_memory=False)
        test_raw = pd.read_csv(test_path, parse_dates=["valid_time"], low_memory=False)

        # A clean (anomaly-free) warmup file paired with an anomaly-injected
        # test file won't have the same columns — e.g. a shared
        # era5_clean_warmup.csv has no is_anomaly column at all, while an
        # injected test file does. Reconcile before concat: whichever side
        # lacks is_anomaly gets it filled with 0, so the combined column is
        # always a clean int, never NaN (pd.concat would otherwise silently
        # upcast it to float64 with NaN for the missing side's rows).
        if "is_anomaly" not in warmup_raw.columns:
            warmup_raw["is_anomaly"] = 0
        if "is_anomaly" not in test_raw.columns:
            test_raw["is_anomaly"] = 0

        if warmup_raw["valid_time"].max() >= test_raw["valid_time"].min():
            raise ValueError(
                "ERA5 two-file config: warmup_path's last timestamp "
                f"({warmup_raw['valid_time'].max()}) is not before test_path's first "
                f"({test_raw['valid_time'].min()}) — files must be in chronological "
                "order with no overlap."
            )

        df_raw = pd.concat([warmup_raw, test_raw], ignore_index=True)
        warmup_start = int(cfg.get("warmup_start", 0))
        warmup_length = len(warmup_raw)
        data_path_display = f"{warmup_path} + {test_path} (warmup+test)"
    else:
        if data_path is None:
            raise ValueError(
                "ERA5 config requires either 'data_path' (single file) or "
                "'warmup_path'+'test_path' (two files)."
            )
        data_path = Path(data_path)
        df_raw = pd.read_csv(data_path, parse_dates=["valid_time"], low_memory=False)
        warmup_start = int(cfg.get("warmup_start", 0))
        warmup_length = int(cfg.get("warmup_end", 0))  # warmup_end is the test-stream boundary
        if warmup_length == 0:
            raise ValueError(
                "ERA5 config requires 'warmup_end' (the last warmup row / first test row). "
                "Set it explicitly, e.g.  warmup_end: 302227"
            )
        data_path_display = str(data_path)

    return df_raw, warmup_start, warmup_length, data_path_display


def load_era5_dataset(cfg, logger: Logger):
    """
    Loader for ERA5 hourly climate reanalysis data.

    The CSV(s) may be:
      (a) the raw ERA5 download (all normal, no is_anomaly column), or
      (b) an anomaly-injected file (has is_anomaly and anomaly_type columns).

    Columns dropped automatically:
      sst          — all-NaN for the Canterbury grid point
      latitude     — constant (single grid point)
      longitude    — constant (single grid point)
      anomaly_type — metadata column, not a feature

    Warmup window: rows [warmup_start, warmup_end). The source data can be
    given as one file with a row-index split, or as two pre-split files —
    see `_read_era5_raw` for the two equivalent config shapes.

    Expected cfg keys
    -----------------
    data_path   : (mode 1) path to the (possibly injected) ERA5 CSV
    warmup_start: int (default 0)
    warmup_end  : (mode 1, required) int — first row of the test stream
    warmup_path : (mode 2) path to the warmup-only ERA5 CSV
    test_path   : (mode 2) path to the test-only ERA5 CSV
    args        : optional dict
        z_clip                : float or null — symmetric z-score clipping
        scaler_kind            : "zscore" (default) | "minmax"
        val_ratio               : float (default 0.2)
        cyclic_time_features    : bool (default False)
    """
    import pandas as pd

    DROP_COLS = {"sst", "latitude", "longitude", "anomaly_type"}

    args = cfg.get("args", {}) or {}
    z_clip               = args.get("z_clip", None)
    scaler_kind          = args.get("scaler_kind", "zscore")
    scaler_online        = args.get("scaler_online", False)
    minmax_clip          = args.get("minmax_clip", True)
    val_ratio            = args.get("val_ratio", 0.2)
    cyclic_time_features = args.get("cyclic_time_features", False)

    # --- load CSV(s) ---
    df_raw, warmup_start, warmup_length, data_path_display = _read_era5_raw(cfg)
    df_raw = df_raw.set_index("valid_time")

    # Drop constant / all-NaN / metadata columns
    drop = [c for c in DROP_COLS if c in df_raw.columns]
    df_raw = df_raw.drop(columns=drop)

    # Backfill first ~24 NaN rows of accumulated fields (fg10, ssrd, strd, tp)
    df_raw = df_raw.bfill().ffill()

    # Separate labels (present only in injected files)
    if "is_anomaly" in df_raw.columns:
        y_ser = df_raw.pop("is_anomaly").astype("int8").rename("is_anomaly")
    else:
        y_ser = pd.Series(0, index=df_raw.index, dtype="int8", name="is_anomaly")
    y_df = y_ser.to_frame()

    X_df = df_raw.astype("float32")
    timestamps = df_raw.index.to_numpy()
    T = len(X_df)

    n_test       = T - warmup_length
    n_test_anom  = int(y_ser.iloc[warmup_length:].sum())
    logger.info_low(lambda: (
        "# ERA5 Dataset:\n"
        "#   data_path:              {}\n"
        "#   total rows:             {}\n"
        "#   features:               {}  {}\n"
        "#   warmup_start row:       {}\n"
        "#   warmup_end   row:       {}\n"
        "#   test rows:              {}\n"
        "#   test anomalies:         {}  ({:.2f}% of test)\n"
    ).format(
        data_path_display, T,
        X_df.shape[1], list(X_df.columns),
        warmup_start, warmup_length,
        n_test,
        n_test_anom, 100 * n_test_anom / max(n_test, 1),
    ))

    def get_data_adapter():
        adapter = ERA5Adapter(
            X_raw=X_df,
            y_raw=y_df,
            warmup_length=warmup_length,
            warmup_start=warmup_start,
            timestamps=timestamps,
            logger=logger,
            z_clip=z_clip,
            scaler_kind=scaler_kind,
            scaler_online=scaler_online,
            minmax_clip=minmax_clip,
            val_ratio=val_ratio,
            cyclic_time_features=cyclic_time_features,
        )
        adapter.fit_scaler()
        adapter.transform()
        meta = {
            "input_dim": adapter.X.shape[1],
            "eval_idx": adapter.warmup_length,
            "n_cyclic_features": 4 if cyclic_time_features else 0,
        }
        return adapter, meta

    return get_data_adapter
