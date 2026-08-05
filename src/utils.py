from __future__ import annotations
import numpy as np
import math
import os
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


MIN_TH_BOOTSTRAP = 20

TANH_S_BY_METRIC = {
    "elbo": 1.0,
    "recon": 1.0,
    "kl": 2.0,          # heavier tail
    "mu_norm": 2.0,     # can spike
    "mah_diag": 3.0,    # very heavy tail
    "logvar_target_mse": 2.0,
}

EPS_BY_METRIC = {
    "mu_norm": 1e-3,
    "mah_diag": 1e-3,
    "logvar_sumexp": 1e-3,
    "logvar_target_mse": 1e-3,
    "kl": 1e-3,
}

Activations: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "identity": nn.Identity,
}

def set_global_seed(seed: int | None):

    if seed is None:
        seed = int(time.time() * 1e6) % (2**32 - 1)

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Disable non-deterministic SDPA backends (Flash Attention, mem-efficient).
    # PyTorch 2.x TransformerEncoderLayer uses SDPA internally; Flash/mem-efficient
    # can produce different results across runs even with the same seed.
    # Only the math (reference) backend is guaranteed deterministic.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

    # CPU threading determinism (best set in shell too)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    return seed


#  Choose frequencies k for sin(2π k t), t in [0,1], as powers of two.
# - T: window length
# - growth_rule:
#     "sqrt": k_max ≈ sqrt(T)
#     "sqrt_half": k_max ≈ sqrt(T / 2)  (more conservative)
def make_positional_freqs(T: int,
                          min_cycles: int = 1,
                          growth_rule: str = "sqrt") -> List[int]:
    if growth_rule == "sqrt":
        k_max = max(4, int(math.sqrt(T)))           # at least allow up to 4
    elif growth_rule == "sqrt_half":
        k_max = max(4, int(math.sqrt(T / 2.0)))
    else:
        raise ValueError(f"Unknown growth_rule: {growth_rule}")

    freqs = []
    k = 1
    while k <= k_max:
        if k >= min_cycles:
            freqs.append(k)
        k *= 2
    return freqs


def make_hourly_domain_freqs(T: int) -> List[float]:
    candidate_periods = [24, 168, 336]
    freqs = []

    for P in candidate_periods:
        if P <= T:
            k = T / P   # cycles per normalized window
            freqs.append(k)

    return sorted(set(freqs))

from typing import Tuple

def to_plain(o):
    # make numpy / torch scalars JSON-friendly
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o
    
# --------- utilities ---------
def _as_iso(ts) -> str:
    # Works for np.datetime64, pandas timestamps, and python datetime
    try:
        return str(ts)  # np.datetime64 prints ISO-8601
    except Exception:
        return f"{ts}"


import numpy as np

# Results Table

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _fmt_num(x: Any, nd: int = 3) -> str:
    """Format numbers compactly; keep ints as ints; floats rounded; handle NaN/inf."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{nd}f}"
    return str(x)

def _safe_get(d: Dict[str, Any], path: Iterable[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _make_table(rows: List[Tuple[str, List[str]]], headers: List[str]) -> str:
    """
    rows: [(row_name, [col1, col2, ...]), ...]
    headers: ["metric", "elbo", "recon", ...]
    """
    # column widths
    col_widths = [len(h) for h in headers]
    for rname, cols in rows:
        col_widths[0] = max(col_widths[0], len(rname))
        for j, val in enumerate(cols, start=1):
            col_widths[j] = max(col_widths[j], len(val))

    def fmt_row(vals: List[str]) -> str:
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in col_widths)

    out = []
    out.append(fmt_row(headers))
    out.append(sep)
    for rname, cols in rows:
        out.append(fmt_row([rname] + cols))
    return "\n".join(out)

def format_metrics_stats(
    metrics_stats: Dict[str, Any],
    *,
    nd: int = 3,
    top_metrics: Optional[List[str]] = None,
    score_keys: Optional[List[str]] = None,
    include_counts: bool = True,
) -> str:
    """
    Turn your nested dict into readable tables.
    Call like: logger.info_low(lambda: format_metrics_stats(metrics_stats))
    """
    # what to prioritize / order
    if score_keys is None:
        score_keys = ["conf_pr", "conf_roc", "f1", "precision", "recall", "accuracy"]
    if top_metrics is None:
        # order of metric blocks in the table (only those present will be used)
        top_metrics = ["final", "elbo", "recon", "kl", "mah_diag", "mu_norm", "logvar_sumexp", "logvar_target_mse"]

    lines: List[str] = []

    metrics = metrics_stats.get("metrics", {})
    if isinstance(metrics, dict) and "metrics" in metrics:
        # in case user passes the outer dict that has {'metrics': {...}}
        metrics = metrics["metrics"]

    # --- Metrics table ---
    metric_blocks = metrics if isinstance(metrics, dict) else {}
    chosen_blocks = [k for k in top_metrics if k in metric_blocks]
    # include any remaining blocks not in top_metrics at the end
    for k in metric_blocks.keys():
        if k not in chosen_blocks:
            chosen_blocks.append(k)

    if chosen_blocks:
        headers = ["metric"] + chosen_blocks
        rows: List[Tuple[str, List[str]]] = []

        for sk in score_keys:
            row_vals = []
            for bk in chosen_blocks:
                val = metric_blocks.get(bk, {}).get(sk, None)
                row_vals.append(_fmt_num(val, nd))
            rows.append((sk, row_vals))

        # add mean/std if present (nice for debugging score distributions)
        for extra in ["score_mean", "score_std"]:
            if any(extra in metric_blocks.get(b, {}) for b in chosen_blocks):
                rows.append((extra, [_fmt_num(metric_blocks.get(b, {}).get(extra), nd) for b in chosen_blocks]))

        lines.append("METRICS (higher is better for conf_pr/conf_roc/f1/precision/recall/accuracy)")
        lines.append(_make_table(rows, headers))

        if include_counts:
            # show counts once (take from 'final' if exists else first block)
            ref = metric_blocks.get("final") or metric_blocks.get(chosen_blocks[0], {})
            n = ref.get("n")
            n_pos = ref.get("n_pos")
            n_neg = ref.get("n_neg")
            n_fin = ref.get("n_finite")
            if any(v is not None for v in [n, n_fin, n_pos, n_neg]):
                lines.append(
                    f"Counts: n={_fmt_num(n)}  n_finite={_fmt_num(n_fin)}  pos={_fmt_num(n_pos)}  neg={_fmt_num(n_neg)}"
                )

    # --- Threshold tuning / search ---
    tuning        = metrics_stats.get("final_threshold_tuning", {})
    tuning_f2     = metrics_stats.get("final_threshold_tuning_f2", {})
    tuning_youden = metrics_stats.get("final_threshold_tuning_youden", {})
    search        = metrics_stats.get("final_threshold_search", {})

    def _fmt_thr_block(title: str, d: Dict[str, Any]) -> str:
        if not isinstance(d, dict) or not d:
            return ""
        parts = [title]
        thr = d.get("best_thr_f1", d.get("thr"))
        if thr is not None:
            parts.append(f"  thr={_fmt_num(thr, nd)}")

        best_stats = d.get("best_stats")
        if isinstance(best_stats, dict):
            core = ["f1", "f2", "precision", "recall", "accuracy", "tp", "fp", "tn", "fn"]
            s = "  " + "  ".join(f"{k}={_fmt_num(best_stats.get(k), nd)}" for k in core if k in best_stats)
            parts.append(s)
        else:
            core = ["f1", "f2", "precision", "recall", "accuracy"]
            s = "  " + "  ".join(f"{k}={_fmt_num(d.get(k), nd)}" for k in core if k in d)
            if s.strip():
                parts.append(s)
        return "\n".join(parts)

    block = _fmt_thr_block("THRESHOLD TUNING (optimize F1)", tuning)
    if block:
        lines.append(block)

    block = _fmt_thr_block("THRESHOLD TUNING (optimize F2)", tuning_f2)
    if block:
        lines.append(block)

    block = _fmt_thr_block("THRESHOLD TUNING (Youden J)", tuning_youden)
    if block:
        lines.append(block)

    block = _fmt_thr_block("THRESHOLD SEARCH", search)
    if block:
        lines.append(block)

    # fallback if nothing formatted
    if not lines:
        return str(metrics_stats)

    return "\n\n".join(lines)

# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_dim: int, max_len: int = 10000):
#         super().__init__()

#         # Generates a sinusoidal positional encoding matrix
#         pos_encoding = torch.zeros(max_len, embed_dim)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, embed_dim, 2,  dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
#         pos_encoding[:, 0::2] = torch.sin(position * div_term)
#         pos_encoding[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pos_encoding", pos_encoding)

#     def forward(self, x):
#         T = x.size(1) # from x = (B, T, D) to (1,T,D)
#         # Ensure the positional encoding matrix aligns with the input's sequence length
#         return x + self.pos_encoding[:T, :].unsqueeze(0) 

