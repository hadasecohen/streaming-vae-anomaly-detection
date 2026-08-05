import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Callable

from src.Streaming.streaming_tools import OnlineSigmaPrior

from src.utils import MIN_TH_BOOTSTRAP

@dataclass
class ThresholdState:
    q: float        # target quantile for the anomaly score
    buffer_size: int  # how many recent scores we retain
    ema_alpha: float   # smoothing of the quantile estimate
    value: float       # current threshold used for decisions


@dataclass
class Metric:
    weight: float
    compute_score_fn: Callable[[float, float, np.ndarray,  np.ndarray], float]
    threshold: ThresholdState
    ref_buffer: deque  # defines normality - stores recent scores for this metric
    # learn_buffer: deque  # measures adaptation post-update score history
    # Estimate the q-quantile as threshold candidate.
    # Maintain a rolling quantile of recent reconstruction errors and smooth it with EMA.
    # Quantile: q = np.quantile(errors, self.th.q)
    # EMA:      T_t = (1 - alpha) * T_{t-1} + alpha * q,   with alpha = self.th.ema_alpha
    sigma_prior : OnlineSigmaPrior

    # --- Band state ---
    # Used to compute slope (drift vs spike)
    z_hist: deque = field(default_factory=lambda: deque(maxlen=200))

    # prevents “single-step flip-flops”
    anom_streak: int = 0
    recovering_streak: int = 0

    #   Bootstrap with the raw error until there are a minimum number of samples (MIN_Q_BOOTSTRAP),
    #   to avoid noisy quantiles on tiny samples.
    #   cap history with a fixed-size deque (self.th.buffer_size) -> O(1) updates.
    def update_threshold(self):

        buf = np.asarray(self.ref_buffer, dtype=np.float64)
        if buf.size == 0:
            return
        if buf.size >= MIN_TH_BOOTSTRAP:
            new_th_val = float(np.quantile(buf, self.threshold.q))
        else:
            new_th_val = float(buf[-1])

        if np.isfinite(new_th_val):
            if np.isfinite(self.threshold.value):
                self.threshold.value = (1.0 - self.threshold.ema_alpha) * self.threshold.value + self.threshold.ema_alpha * new_th_val
            else:
                self.threshold.value = new_th_val

    def soft_reset_metric(self, tail_keep=200):
        old = list(self.ref_buffer)
        new = old[-tail_keep:] if len(old) > tail_keep else old
        self.ref_buffer = deque(new, maxlen=self.ref_buffer.maxlen)

        # re-bootstrap threshold immediately from what remains
        if len(new) >= MIN_TH_BOOTSTRAP:
            self.threshold.value = float(np.quantile(np.asarray(new), self.threshold.q))
        else:
            self.threshold.value = float("nan")  # will bootstrap from incoming points

@dataclass
class PredStats:
    metric_name: str
    pred: int
    loss: float
    raw_margin: float
    threshold: float
    signed_conf: float
    conf_0_1 : float
    z: float = 0.0
    sigma: float = 1.0
