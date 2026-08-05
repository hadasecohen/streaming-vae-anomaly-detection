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

    match name:
        case "era5":
            return load_era5_dataset(cfg, logger)
        case _:
            raise ValueError(f"Unknown dataset name: {cfg['name']}. This curated build only supports 'era5'.")


def load_era5_dataset(cfg, logger: Logger):
    """
    Loader for ERA5 hourly climate reanalysis data.

    The CSV may be:
      (a) the raw ERA5 download (all normal, no is_anomaly column), or
      (b) an anomaly-injected file (has is_anomaly and anomaly_type columns).

    Columns dropped automatically:
      sst          — all-NaN for the Canterbury grid point
      latitude     — constant (single grid point)
      longitude    — constant (single grid point)
      anomaly_type — metadata column, not a feature

    Warmup window: rows [warmup_start, warmup_end).
      warmup_start (optional, default 0) — skip rows before this index.
      warmup_end   (required) — first row of the test stream.

    Expected cfg keys
    -----------------
    data_path   : path to the (possibly injected) ERA5 CSV
    warmup_start: int (default 0)
    warmup_end  : int — first row of the test stream
    args        : optional dict
        z_clip                : float or null — symmetric z-score clipping
        scaler_kind            : "zscore" (default) | "minmax"
        val_ratio               : float (default 0.2)
        cyclic_time_features    : bool (default False)
    """
    import pandas as pd

    DROP_COLS = {"sst", "latitude", "longitude", "anomaly_type"}

    data_path = Path(cfg["data_path"])
    args = cfg.get("args", {}) or {}
    z_clip               = args.get("z_clip", None)
    scaler_kind          = args.get("scaler_kind", "zscore")
    scaler_online        = args.get("scaler_online", False)
    minmax_clip          = args.get("minmax_clip", True)
    val_ratio            = args.get("val_ratio", 0.2)
    cyclic_time_features = args.get("cyclic_time_features", False)

    # --- load CSV ---
    df_raw = pd.read_csv(data_path, parse_dates=["valid_time"], low_memory=False)
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

    warmup_start = int(cfg.get("warmup_start", 0))
    warmup_length = int(cfg.get("warmup_end", 0))   # warmup_end is the test-stream boundary
    if warmup_length == 0:
        raise ValueError(
            "ERA5 config requires 'warmup_end' (the last warmup row / first test row). "
            "Set it explicitly, e.g.  warmup_end: 302227"
        )

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
        data_path, T,
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
