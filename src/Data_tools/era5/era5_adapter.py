# era5_adapter.py
#
# Adapter for ERA5 hourly climate reanalysis data.
#
# Warmup window: rows [warmup_start, warmup_length).
#   warmup_start  — first row used for offline training (default 0).
#                   Rows before this index are ignored entirely.
#   warmup_length — first row of the test stream.
#
# The scaler is fitted on X[warmup_start:warmup_length] (normal data only).
# The 80/20 train/val split is taken chronologically within that slice.
# The test set is X[warmup_length:].

import numpy as np
import pandas as pd
from src.logging_utils import Logger
from src.Data_tools.dataset_adapter import NormalizingAdapter, PointDataset, WindowDataset


class ERA5Adapter(NormalizingAdapter):

    def __init__(
        self,
        X_raw: pd.DataFrame,
        y_raw: pd.DataFrame,
        warmup_length: int,
        logger: Logger,
        warmup_start: int = 0,
        timestamps: np.ndarray | None = None,
        z_clip: float | None = None,
        scaler_kind: str = "zscore",
        scaler_online: bool = False,
        minmax_clip: bool = True,
        val_ratio: float = 0.2,
        cyclic_time_features: bool = False,
    ):
        super().__init__(
            X_raw=X_raw,
            y_raw=y_raw,
            warmup_length=warmup_length,
            logger=logger,
            z_clip=z_clip,
            scaler_kind=scaler_kind,
            scaler_online=scaler_online,
            minmax_clip=minmax_clip,
        )
        if timestamps is not None:
            self.t = timestamps

        if cyclic_time_features and timestamps is not None:
            self._append_cyclic_time_features(timestamps, logger)
        T = self.X.shape[0]
        if not (0 <= warmup_start < warmup_length <= T):
            raise ValueError(
                f"warmup_start={warmup_start}, warmup_length={warmup_length}, T={T}: "
                "must satisfy 0 <= warmup_start < warmup_length <= T"
            )
        self.warmup_start = warmup_start
        self.val_ratio = val_ratio

    # ------------------------------------------------------------------
    # Optional cyclic time encoding
    # ------------------------------------------------------------------
    def _append_cyclic_time_features(self, timestamps: np.ndarray, logger: Logger) -> None:
        """Append sin/cos encodings of day-of-year and hour-of-day to self.X.

        Produces four additional features per row:
          sin_doy, cos_doy  — annual cycle  (period = 365.25 days)
          sin_hod, cos_hod  — diurnal cycle (period = 24 hours)

        These are appended BEFORE fit_scaler / transform, so the scaler treats
        them like any other feature.  For ERA5 the z-score of a full sinusoid
        is well-behaved (mean≈0, std≈0.707).
        """
        ts = pd.DatetimeIndex(timestamps)
        doy = ts.day_of_year.to_numpy(dtype="float32")
        hod = ts.hour.to_numpy(dtype="float32")

        sin_doy = np.sin(2.0 * np.pi * doy / 365.25).astype("float32")
        cos_doy = np.cos(2.0 * np.pi * doy / 365.25).astype("float32")
        sin_hod = np.sin(2.0 * np.pi * hod / 24.0).astype("float32")
        cos_hod = np.cos(2.0 * np.pi * hod / 24.0).astype("float32")

        cyclic = np.stack([sin_doy, cos_doy, sin_hod, cos_hod], axis=1)  # (N, 4)
        self.X = np.concatenate([self.X, cyclic], axis=1)
        self.col_names = self.col_names + ["sin_doy", "cos_doy", "sin_hod", "cos_hod"]
        logger.info_low(lambda: "ERA5: appended cyclic time features [sin_doy, cos_doy, sin_hod, cos_hod] → input_dim now {}".format(self.X.shape[1]))

    # ------------------------------------------------------------------
    # Override fit_scaler to calibrate only on [warmup_start, warmup_length)
    # ------------------------------------------------------------------
    def fit_scaler(self) -> None:
        calib_X = self.X[self.warmup_start:self.warmup_length]
        normal_mask = self.y[self.warmup_start:self.warmup_length] == 0
        calib_X = calib_X[normal_mask] if normal_mask.any() else calib_X

        if self.scaler_kind == "minmax":
            self.col_min_ = np.nanmin(calib_X, axis=0)
            self.col_max_ = np.nanmax(calib_X, axis=0)
            span = self.col_max_ - self.col_min_
            constant_cols = np.sum(span < 1e-8)
            if constant_cols:
                print(f"[minmax] {constant_cols} constant features (span < 1e-8) — clipped to 1e-8")
            self.col_max_ = np.where(span < 1e-8, self.col_min_ + 1e-8, self.col_max_)
        else:
            self.mu    = np.nanmean(calib_X, axis=0)
            self.sigma = np.nanstd(calib_X, axis=0, ddof=1)
            self.sigma = np.maximum(self.sigma, 1e-3)
            print(f"Small σ features (<0.05): {np.sum(self.sigma < 0.05)}/{len(self.sigma)}")
            print(f"Minimum σ: {np.min(self.sigma)}")

        if self.scaler_online:
            self._init_welford_from_batch(calib_X)

    # ------------------------------------------------------------------
    # Point mode
    # ------------------------------------------------------------------
    def preprocess_datasets_in_points(self):
        X = self.X
        _, M = X.shape

        n_warmup = self.warmup_length - self.warmup_start
        train_end = self.warmup_start + int(round(n_warmup * (1 - self.val_ratio)))

        self.train_ds = PointDataset(
            X=X[self.warmup_start:train_end][:, np.newaxis, :],
            y=self.y[self.warmup_start:train_end],
            t=self.t[self.warmup_start:train_end],
        )
        self.val_ds = PointDataset(
            X=X[train_end:self.warmup_length][:, np.newaxis, :],
            y=self.y[train_end:self.warmup_length],
            t=self.t[train_end:self.warmup_length],
        )
        self.test_ds = PointDataset(
            X=X[self.warmup_length:][:, np.newaxis, :],
            y=self.y[self.warmup_length:],
            t=self.t[self.warmup_length:],
        )

    # ------------------------------------------------------------------
    # Window mode
    # ------------------------------------------------------------------
    def preprocess_datasets_in_windows(
        self,
        window_size: int,
        stride: int,
        train_on_normal_only: bool = True,
    ):
        X = self.X
        _, M = X.shape

        n_warmup  = self.warmup_length - self.warmup_start
        train_end = self.warmup_start + int(round(n_warmup * (1 - self.val_ratio)))

        # Warmup windows — built from X[warmup_start:warmup_length]
        X_wup = X[self.warmup_start:self.warmup_length]
        t_wup = self.t[self.warmup_start:self.warmup_length]
        n_wup = len(X_wup)

        # train_end relative to warmup slice
        train_end_rel = train_end - self.warmup_start

        warmup_starts = np.arange(0, n_wup - window_size + 1, stride)
        train_starts  = warmup_starts[warmup_starts + window_size - 1 <  train_end_rel]
        val_starts    = warmup_starts[warmup_starts + window_size - 1 >= train_end_rel]

        def build_warmup_windows(starts_arr):
            Xw, yw, tw, ss = [], [], [], []
            for s in starts_arr:
                e = s + window_size
                Xw.append(X_wup[s:e])
                yw.append(0)
                tw.append(t_wup[s:e])
                ss.append(s)
            X_out = np.stack(Xw) if Xw else np.empty((0, window_size, M), dtype="float32")
            y_out = np.array(yw, dtype=int)
            t_out = np.array(tw, dtype=object) if tw else np.empty((0, window_size), dtype=object)
            s_out = np.array(ss, dtype=int)
            return X_out, y_out, t_out, s_out

        # Test windows — built from X[warmup_length:]
        X_test = X[self.warmup_length:]
        y_test = self.y[self.warmup_length:]
        t_test = self.t[self.warmup_length:]
        n_test = len(X_test)

        test_starts = np.arange(0, n_test - window_size + 1, stride)

        def build_test_windows(starts_arr):
            Xw, yw, tw, ss = [], [], [], []
            for s in starts_arr:
                e = s + window_size
                Xw.append(X_test[s:e])
                yw.append(int(y_test[s:e].max()))
                tw.append(t_test[s:e])
                ss.append(s)
            X_out = np.stack(Xw) if Xw else np.empty((0, window_size, M), dtype="float32")
            y_out = np.array(yw, dtype=int)
            t_out = np.array(tw, dtype=object) if tw else np.empty((0, window_size), dtype=object)
            s_out = np.array(ss, dtype=int)
            return X_out, y_out, t_out, s_out

        X_tr, y_tr, t_tr, s_tr = build_warmup_windows(train_starts)
        X_va, y_va, t_va, s_va = build_warmup_windows(val_starts)
        X_te, y_te, t_te, s_te = build_test_windows(test_starts)

        self.train_ds = WindowDataset(X=X_tr, y=y_tr, t=t_tr, starts=s_tr)
        self.val_ds   = WindowDataset(X=X_va, y=y_va, t=t_va, starts=s_va)
        self.test_ds  = WindowDataset(X=X_te, y=y_te, t=t_te, starts=s_te)
        self.window_size = window_size
        self.stride = stride
