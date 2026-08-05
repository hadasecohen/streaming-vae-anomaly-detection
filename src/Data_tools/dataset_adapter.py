# streaming/datasets.py

from dataclasses import dataclass
import numpy as np
import random
from typing import Iterator, Tuple, Dict, Optional

import pandas as pd

from src.logging_utils import Logger, Verbosity

@dataclass
class StreamSample:
    x: np.ndarray        # shape: (1, T, M) scaled sample
    y: Optional[int]     # 0/1 if available (labels may be delayed/masked)
    t: Optional[np.ndarray] # 1-D array of timestamps for the window (len L),
                         # or scalar for point
    is_eval: bool        # True if belongs to evaluation segment
    x_raw: Optional[np.ndarray] = None  # raw unscaled data (T, M); non-None only when scaler_online=True

@dataclass
class WindowDataset:
    X: np.ndarray      # (N, window_size, M)
    y: np.ndarray      # (N,)
    t: np.ndarray      # (N, window_size) or object array
    starts: np.ndarray # (N,) start indices in original series


import numpy as np
from dataclasses import dataclass

@dataclass
class PointDataset:
    X: np.ndarray   # (N, M)
    y: np.ndarray   # (N,)
    t: np.ndarray   # (N,)

@dataclass
class WarmupSegment:
    """One contiguous chronological block within the warmup period."""
    role: str        # 'train' or 'val'
    X: np.ndarray   # (N, W, M)  window features
    y: np.ndarray   # (N,)       binary labels (0=normal, 1=anomaly)

# Base adapter: normalize & yield a labeled stream
class DatasetAdapter:
    
    def fit_scaler(self, X_calib: np.ndarray) -> None:
        raise NotImplementedError
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def stream(self, shuffle: bool=False) -> Iterator[StreamSample]:
        raise NotImplementedError
    
    def meta(self) -> Dict:  # e.g., {"input_dim": M}
        raise NotImplementedError
    
    def preprocess_datasets_in_points(self):
        raise NotImplementedError
    
    def preprocess_datasets_in_windows(
        self,
        window_size: int,
        stride: int,
        train_on_normal_only: bool = True
    ):
        raise NotImplementedError

    def stream_all(self) -> Iterator[StreamSample]:
        """Yield the full time series in order for fully-online mode.

        Warmup samples (train_ds + val_ds, or warmup_segments if interleaved)
        are emitted with is_eval=False; test_ds samples with is_eval=True.
        Requires preprocess_datasets_in_points/windows to have been called first.
        """
        warmup_segs = getattr(self, "warmup_segments", None)
        if warmup_segs is not None:
            # Interleaved adapters (e.g. FraudAdapter): segments are chronological
            for seg in warmup_segs:
                for i in range(len(seg.X)):
                    yield StreamSample(x=seg.X[i:i+1], y=int(seg.y[i]), t=None, is_eval=False)
        else:
            # Standard adapters: train_ds then val_ds are already in time order
            for ds in [self.train_ds, self.val_ds]:
                for i in range(len(ds.X)):
                    yield StreamSample(x=ds.X[i:i+1], y=int(ds.y[i]), t=ds.t[i], is_eval=False)
        yield from self.stream(shuffle=False)
    
    def apply_nor_don_enrichment(self, window_size: int) -> None:
        """Apply NOR/DON enrichment in-place (call before fit_scaler/transform).

        Transforms self.X from (T, M) to (C', 2*M) and updates self.y, self.t,
        self.warmup_length, and self.warmup_start (if present) to match.
        Also updates self.col_names if the attribute exists.

        Disabled when window_size is None or 0.
        """
        if not window_size:
            return

        from src.Data_tools.timeseries_enrichment import enrich_nor_don

        warmup_start = getattr(self, "warmup_start", 0)
        X_out, y_out, t_out, warmup_length_new, warmup_start_new = enrich_nor_don(
            X=self.X,
            y=self.y,
            t=self.t,
            warmup_length=self.warmup_length,
            window_size=window_size,
            warmup_start=warmup_start,
        )

        if hasattr(self, "col_names") and self.col_names is not None:
            self.col_names = (
                [f"NOR_{c}" for c in self.col_names]
                + [f"DON_{c}" for c in self.col_names]
            )

        self.X = X_out
        self.y = y_out
        self.t = t_out
        self.warmup_length = warmup_length_new
        if hasattr(self, "warmup_start"):
            self.warmup_start = warmup_start_new

    def validate_features(self, df: pd.DataFrame, logger: Logger):
        # 1. Check for any NaN
        if df.isna().any().any():
            logger.error(lambda: f"X - Feature dataframe contains NaN values:\n{df.isna().sum().to_string()}")
        else:
            logger.info_low(lambda: "V - No NaN values found in features")

        # 2. Check for Inf / -Inf
        mask_inf = np.isinf(df.values)
        if mask_inf.any():
            rows, cols = np.where(mask_inf)
            logger.error(lambda: f"X - Features contain +/-Inf values.\n Example locations: {list(zip(rows[:10], cols[:10]))}")
        else:
            logger.info_low(lambda: "V - No Inf or -Inf found in features")

        # 3. Check for non-finite values overall (Missing values, bad values, etc)
        if not np.all(np.isfinite(df.values)):
            logger.error(lambda: "X - Some feature values are not finite")
            if logger.level in (Verbosity.ERRORS, Verbosity.DEBUG) :
                bad = df.values[~np.isfinite(df.values)]
                logger.error(lambda: f"Non-finite values found: {bad[:10]}")
        else:
            logger.info_low(lambda: "V - All features are finite")

    # separate training and val sets
    def separate_train_and_val_set(self, n_win):
        n_train = int(np.floor((n_win * 0.8)))
        n_val = n_win - n_train
        idx_train = random.sample(range(n_win), n_train)
        idx_val = list(set(idx_train) ^ set(range(n_win)))
        return idx_train, idx_val, n_train, n_val


class NormalizingAdapter(DatasetAdapter):
    """Intermediate base for adapters that use per-feature Z-score normalization.

    Provides __init__, fit_scaler, transform, meta, and stream.
    Subclasses implement preprocess_datasets_in_points / preprocess_datasets_in_windows.
    """

    def __init__(
        self,
        X_raw: pd.DataFrame,
        y_raw: pd.DataFrame,
        warmup_length: int,
        logger: Logger,
        z_clip: float | None = None,
        scaler_kind: str = "zscore",
        scaler_online: bool = False,
        minmax_clip: bool = True,
    ):
        if scaler_kind not in ("zscore", "minmax"):
            raise ValueError(f"scaler_kind must be 'zscore' or 'minmax', got '{scaler_kind}'")
        X_df = X_raw.loc[:, ~X_raw.columns.str.contains("^Unnamed", case=False)]
        self.col_names = list(X_df.columns)
        self.validate_features(X_df, logger)

        if "is_anomaly" in y_raw.columns:
            y_ser = y_raw["is_anomaly"].astype(bool)
        else:
            y_ser = y_raw.squeeze().astype(bool)
        self.validate_features(y_ser, logger)

        self.t: np.ndarray = np.arange(len(X_df), dtype=np.int64)
        self.X = X_df.to_numpy(dtype="float32")
        self.y: np.ndarray = y_ser.to_numpy(dtype="int8")
        self.warmup_length = warmup_length
        self.z_clip = z_clip
        self.scaler_kind = scaler_kind
        self.scaler_online = scaler_online
        self.minmax_clip = minmax_clip
        self.mu = None
        self.sigma = None
        self.col_min_ = None   # minmax scalars
        self.col_max_ = None
        # Welford running state for online zscore updates
        self.n_seen: int = 0
        self.M2: np.ndarray | None = None

        assert (
            self.X.ndim == 2
            and self.y.ndim == 1
            and self.t.ndim == 1
            and len(self.X) == len(self.y) == len(self.t)
        )

    def analyze_distributions(
        self,
        logger: Logger,
        skew_threshold: float = 2.0,
        high_skew_threshold: float = 5.0,
    ) -> dict:
        """Analyze per-feature distributions on the calibration window.

        Computes skewness, checks feasibility of log-transform, and prints
        recommendations. Call this *before* fit_scaler to inform preprocessing.

        Args:
            skew_threshold:      |skewness| above this → recommend log-transform.
            high_skew_threshold: |skewness| above this → strongly recommend it.

        Returns:
            dict with keys:
              'skewness'        : array of skewness values per feature
              'log_recommended' : list of (col_name, skewness) for skewed features
              'log_feasible'    : list of col names where log-transform is safe
                                  (all values > 0 after shift)
        """
        if self.warmup_length > 0:
            calib_X = self.X[:self.warmup_length]
            normal_mask = self.y[:self.warmup_length] == 0
            calib_X = calib_X[normal_mask] if normal_mask.any() else calib_X
        else:
            calib_X = self.X

        n, m = calib_X.shape
        skewness = np.zeros(m, dtype=np.float64)
        for j in range(m):
            col = calib_X[:, j]
            col = col[np.isfinite(col)]
            if len(col) < 3:
                continue
            mean = col.mean()
            std = col.std()
            if std < 1e-8:
                continue
            skewness[j] = np.mean(((col - mean) / std) ** 3)

        col_min = np.nanmin(calib_X, axis=0)

        log_recommended = []
        log_feasible = []
        rows = []

        for j, name in enumerate(self.col_names):
            sk = skewness[j]
            if abs(sk) < skew_threshold:
                level = "ok"
                action = ""
            elif abs(sk) < high_skew_threshold:
                level = "skewed"
                action = "consider log-transform"
                log_recommended.append((name, float(sk)))
            else:
                level = "HIGHLY skewed"
                action = "strongly recommend log-transform"
                log_recommended.append((name, float(sk)))

            feasible = col_min[j] > 0
            if action and feasible:
                log_feasible.append(name)
                action += " (feasible: min > 0)"
            elif action:
                shift_needed = -col_min[j] + 1e-3
                action += f" (needs shift +{shift_needed:.4f} before log)"

            rows.append((name, sk, level, action))

        header = f"{'Feature':<30}  {'Skewness':>10}  {'Status':<16}  Action"
        logger.info_low(lambda: "\n=== Distribution Analysis ===\n" + header)
        for name, sk, level, action in rows:
            logger.info_low(lambda n=name, s=sk, lv=level, a=action:
                        f"  {n:<30}  {s:>10.3f}  {lv:<16}  {a}")

        n_skewed = len(log_recommended)
        logger.info_low(lambda: (
            f"\nSummary: {n_skewed}/{m} features are skewed (|skew| > {skew_threshold}).\n"
            f"Log-transform directly feasible for: {log_feasible or 'none'}\n"
            "=== End Distribution Analysis ===\n"
        ))

        return {
            "skewness": skewness,
            "log_recommended": log_recommended,
            "log_feasible": log_feasible,
        }

    def apply_log_transform(self, features: list[str], shift_epsilon: float = 1e-3) -> None:
        """Apply log1p-style transform to selected features in self.X (in-place).

        For each feature in `features`:
          - If min(col) <= 0, shifts the column so min = shift_epsilon before logging.
          - Applies log(x) transform.

        Call this *before* fit_scaler.

        Args:
            features:       list of column names to transform.
            shift_epsilon:  small positive value to ensure positivity after shifting.
        """
        for name in features:
            if name not in self.col_names:
                raise ValueError(f"Feature '{name}' not found in col_names.")
            j = self.col_names.index(name)
            col = self.X[:, j]
            col_min = np.nanmin(col)
            if col_min <= 0:
                shift = -col_min + shift_epsilon
                col = col + shift
            self.X[:, j] = np.log(col)

    def fit_scaler(self) -> None:
        if self.warmup_length > 0:
            calib_X = self.X[:self.warmup_length]
            normal_mask = self.y[:self.warmup_length] == 0
            calib_X = calib_X[normal_mask] if normal_mask.any() else calib_X
        else:
            calib_X = self.X

        if self.scaler_kind == "minmax":
            self.col_min_ = np.nanmin(calib_X, axis=0)
            self.col_max_ = np.nanmax(calib_X, axis=0)
            span = self.col_max_ - self.col_min_
            constant_cols = np.sum(span < 1e-8)
            if constant_cols:
                print(f"[minmax] {constant_cols} constant features (span < 1e-8) — clipped to 1e-8")
            self.col_max_ = np.where(span < 1e-8, self.col_min_ + 1e-8, self.col_max_)
        else:  # zscore
            self.mu = np.nanmean(calib_X, axis=0)
            self.sigma = np.nanstd(calib_X, axis=0, ddof=1)  # sample std (Bessel's correction)
            self.sigma = np.maximum(self.sigma, 1e-3)
            print(f"Small σ features (<0.05): {np.sum(self.sigma < 0.05)}/{len(self.sigma)}")
            print(f"Minimum σ: {np.min(self.sigma)}")

        if self.scaler_online:
            self._init_welford_from_batch(calib_X)

    def _init_welford_from_batch(self, calib_X: np.ndarray) -> None:
        """Seed Welford running state from the calibration batch so that
        update_scaler() continues the online update without a discontinuity."""
        n = len(calib_X)
        self.n_seen = n
        if self.scaler_kind == "zscore":
            # M2 = (n-1) * sample_var recovers exactly from the batch fit
            self.M2 = (n - 1) * np.nanvar(calib_X, axis=0, ddof=1)
        # minmax: col_min_ / col_max_ already set by fit_scaler

    def update_scaler(self, x_batch: np.ndarray) -> None:
        """Online update of scaler statistics using new (normal) samples.

        No-op when scaler_online=False (static mode).  Call with normal
        samples only — anomalous samples would contaminate the statistics.

        Args:
            x_batch: raw un-transformed samples, shape (N, M) or (1, M).
        """
        if not self.scaler_online:
            return

        x = np.atleast_2d(x_batch).reshape(-1, len(self.col_names))

        if self.scaler_kind == "zscore":
            # Vectorised Welford's online algorithm (Knuth Vol.2 §4.2.2)
            for row in x:
                self.n_seen += 1
                delta = row - self.mu
                self.mu += delta / self.n_seen
                self.M2 += delta * (row - self.mu)   # uses updated mu
            if self.n_seen > 1:
                self.sigma = np.sqrt(self.M2 / (self.n_seen - 1))
                self.sigma = np.maximum(self.sigma, 1e-3)
        else:  # minmax
            self.col_min_ = np.minimum(self.col_min_, np.nanmin(x, axis=0))
            new_max = np.nanmax(x, axis=0)
            self.col_max_ = np.maximum(self.col_max_, new_max)
            span = self.col_max_ - self.col_min_
            self.col_max_ = np.where(span < 1e-8, self.col_min_ + 1e-8, self.col_max_)

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        """Transform a single raw sample using current scaler statistics.

        Suitable for online use after update_scaler() calls.  Preserves the
        input shape exactly (works for (M,), (1,M), or (1,1,M) inputs).
        """
        shape = x.shape
        x_flat = x.reshape(-1, len(self.col_names)).astype(np.float32)

        if self.scaler_kind == "minmax":
            z = (x_flat - self.col_min_) / (self.col_max_ - self.col_min_)
            if self.minmax_clip:
                z = np.clip(z, 0.0, 1.0)
        else:  # zscore
            x_finite = np.where(np.isfinite(x_flat), x_flat, self.mu)
            z = (x_finite - self.mu) / self.sigma
            if self.z_clip is not None:
                z = np.clip(z, -self.z_clip, self.z_clip)

        return z.reshape(shape)

    def transform(self) -> np.ndarray:
        if self.scaler_kind == "minmax":
            fill = self.col_min_  # fill NaN with per-feature min (maps to 0)
            self.X = np.where(np.isfinite(self.X), self.X, fill)
            Z = (self.X - self.col_min_) / (self.col_max_ - self.col_min_)
            # Values outside the warmup range end up outside [0,1]; clip to [0,1]
            # so BCE targets stay valid.  Anomalies that exceed training range will
            # saturate at 1 — the recon loss on clipped targets is still meaningful.
            if self.z_clip is not None:
                print("[minmax] z_clip is ignored when scaler_kind='minmax' (already in [0,1])")
            out_of_range = np.mean((Z < 0) | (Z > 1))
            if out_of_range > 0:
                print(f"[minmax] {out_of_range*100:.1f}% of values outside [0,1] before clipping "
                      f"(test anomalies exceeding warmup range — expected)")
            if self.minmax_clip:
                Z = np.clip(Z, 0.0, 1.0)
        else:  # zscore
            self.X = np.where(np.isfinite(self.X), self.X, self.mu)
            Z = (self.X - self.mu) / self.sigma
            print("Max |Z|:", np.max(np.abs(Z)))
            print("Mean |Z|:", np.mean(np.abs(Z)))
            if self.z_clip is not None:
                Z = np.clip(Z, -self.z_clip, self.z_clip)

        if self.warmup_length > 0:
            Z_warm = Z[:self.warmup_length]
            Z_test = Z[self.warmup_length:]
            warm_min = np.nanmin(Z_warm, axis=0)
            warm_max = np.nanmax(Z_warm, axis=0)
            test_min = np.nanmin(Z_test, axis=0) if len(Z_test) > 0 else warm_min
            test_max = np.nanmax(Z_test, axis=0) if len(Z_test) > 0 else warm_max
            print(f"  {'column':30s}  {'warmup_min':>10s}  {'warmup_max':>10s}  {'test_min':>10s}  {'test_max':>10s}")
            for name, wn, wx, tn, tx in zip(self.col_names, warm_min, warm_max, test_min, test_max):
                print(f"  {name:30s}  {wn:10.3f}  {wx:10.3f}  {tn:10.3f}  {tx:10.3f}")
        else:
            col_min = np.nanmin(Z, axis=0)
            col_max = np.nanmax(Z, axis=0)
            for name, mn, mx in zip(self.col_names, col_min, col_max):
                print(f"  {name:30s}  min={mn:8.3f}  max={mx:8.3f}")

        if self.scaler_online:
            self._X_raw = self.X.copy()  # NaN-filled raw data; needed for per-sample re-scaling during streaming
        self.X = Z
        return Z

    def meta(self) -> dict:
        return {"input_dim": self.X.shape[1]}

    def stream(self, shuffle: bool):
        X = self.test_ds.X
        y = self.test_ds.y
        t = self.test_ds.t

        if not self.scaler_online or not hasattr(self, '_X_raw'):
            for i in range(len(X)):
                yield StreamSample(x=X[i:i+1], y=int(y[i]), t=t[i], is_eval=True)
            return

        # Online scaling: re-apply transform_one() with current (possibly updated) stats
        # so samples arriving after scaler updates are normalized with up-to-date statistics.
        X_raw_test = self._X_raw[self.warmup_length:]
        M = X.shape[-1]

        if hasattr(self.test_ds, 'starts'):
            # Window mode: reconstruct each raw window via its start index
            W = X.shape[1]
            starts = self.test_ds.starts
            for i in range(len(X)):
                raw = X_raw_test[starts[i]:starts[i] + W]      # (T, M)
                scaled = self.transform_one(raw)                 # (T, M)
                yield StreamSample(
                    x=scaled[np.newaxis], y=int(y[i]), t=t[i], is_eval=True,
                    x_raw=raw.reshape(-1, M),
                )
        else:
            # Point mode: each sample is one row
            for i in range(len(X)):
                raw = X_raw_test[i]                              # (M,)
                scaled = self.transform_one(raw)                 # (M,)
                yield StreamSample(
                    x=scaled[np.newaxis, np.newaxis, :],         # (1, 1, M)
                    y=int(y[i]), t=t[i], is_eval=True,
                    x_raw=raw[np.newaxis],                       # (1, M)
                )



