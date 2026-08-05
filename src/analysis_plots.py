from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import ast
import numpy as np
import math
import os
from typing import Any, Dict
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

# Trial_Paths (src.logging_utils) is used only as a type hint below — not
# imported at runtime to avoid a circular import (logging_utils imports
# from this module via `from src.analysis_plots import *`).
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.logging_utils import Trial_Paths


# Desired ordering (edit if needed)
DESIRED_METRIC_ORDER = ["final", "recon", "elbo", "elbo_w_var", "kl", "latent_z",
            "mu_norm","mu2_sum", "mah_diag", "mah_full_cov", "gmm_latent",
            "variance_mismatch", "logvar_mean", "logvar_sumexp", "logvar_target_mse"]


def plt_confusions(kind: str, payload: Dict[str, Any], paths: Trial_Paths) :
    if kind != "confusions":
        return
    
    weights = payload.get("metric_weights", {})

    metric_names = [
        m for m in DESIRED_METRIC_ORDER
        if m in payload.get("metrics", {})
        and isinstance(payload["metrics"].get(m), dict)
        and "scores_0_1" in payload["metrics"][m]
    ]

    if not metric_names:
        raise ValueError("No metrics with 'scores_0_1' found in payload.")
    
    # Global max for consistent y-scale
    global_max = 0

    for metric_name in metric_names:
        metric = payload["metrics"][metric_name]
        global_max = max(global_max, metric["tp"],metric["fn"], metric["tn"], metric["fp"])

    # Avoid zero-height axis if everything is zero
    y_max = max(1, int(global_max))
    y_pad = max(1, int(0.08 * y_max))   # a little headroom for text
    y_lim_top = y_max + y_pad

    # Layout
    n = len(payload["metrics"])
    ncols=2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    labels = ["TP", "FN", "TN", "FP", ]
    colors = ["green", "red", "green", "red"]
    
    idx=0
    #for idx, name in enumerate(metric_names):
    for metric_name in metric_names:
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        metric = payload["metrics"][metric_name]
        counts = [metric["tp"], metric["fn"], metric["tn"], metric["fp"]]

        bars = ax.bar(labels, counts, color=colors)
        ax.axvline(
            x=1.5,
            linestyle="--",
            linewidth=1,
            alpha=0.6
        )
        w = weights.get(metric_name, None)
        ylabel = f"{metric_name} (w={w:.2f})" if w is not None else metric_name
        ax.set_title(ylabel)
        ax.set_ylim(0, y_lim_top)

        # Add value labels above bars
        for b, val in zip(bars, counts):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val + max(1, 0.01 * y_lim_top),
                str(val),
                ha="center",
                va="bottom",
                fontsize=10
            )

        ax.set_ylabel("count")
        idx+=1

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Prediction counts per metric (shared y-scale)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(paths.conf_hist_png_path, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_stacked(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "confusions":
        return

    weights = payload.get("metric_weights", {})
    
    metric_names = [
        m for m in DESIRED_METRIC_ORDER
        if m in payload.get("metrics", {})
        and isinstance(payload["metrics"].get(m), dict)
        and "scores_0_1" in payload["metrics"][m]
    ]

    if not metric_names:
        raise ValueError("No metrics with 'scores_0_1' found in payload.")

    # Global max for consistent y-scale
    global_max = 0

    for metric_name in metric_names:
        metric = payload["metrics"][metric_name]
        global_max = max(global_max, metric["tp"],metric["fn"], metric["tn"], metric["fp"])

    y_max = max(1, int(global_max))
    y_pad = max(1, int(0.08 * y_max))
    y_lim_top = y_max + y_pad

    n = len(payload["metrics"])
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.5*nrows))
    axes = np.atleast_2d(axes)

    x = np.array([0, 1])
    xlabels = ["Anomalies (y=1)", "Normals (y=0)"]

    idx=0
    #for idx, name in enumerate(metric_names):
    for metric_name in metric_names:
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        metric = payload["metrics"][metric_name]
        tp, fn, tn, fp = metric["tp"], metric["fn"], metric["tn"], metric["fp"]

        correct = np.array([tp, tn])
        incorrect = np.array([fn, fp])

        bars_correct = ax.bar(x, correct)  # default color
        bars_incorrect = ax.bar(x, incorrect, bottom=correct)  # stacked

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, ha="right")
        ax.set_ylim(0, y_lim_top)
        w = weights.get(metric_name, None)
        ylabel = f"{metric_name} (w={w:.2f})" if w is not None else metric_name
        ax.set_title(ylabel)
        ax.set_ylabel("count")

        # Manually color: green for correct, red for incorrect
        for b in bars_correct:
            b.set_color("green")
        for b in bars_incorrect:
            b.set_color("red")

        # annotate counts (correct in bottom, incorrect in top)
        for i in range(2):
            if correct[i] > 0:
                ax.text(x[i], correct[i]/2, str(int(correct[i])), ha="center", va="center", fontsize=10)
            if incorrect[i] > 0:
                ax.text(x[i], correct[i] + incorrect[i]/2, str(int(incorrect[i])), ha="center", va="center", fontsize=10)

        # total labels above
        totals = correct + incorrect
        for i in range(2):
            ax.text(x[i], totals[i] + max(1, 0.01*y_lim_top), f"total={int(totals[i])}", ha="center", va="bottom", fontsize=9)
        idx+=1

    # hide empty axes
    for idx in range(n, nrows*ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Per-metric correctness by true class (stacked)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(paths.conf_stack_hist_png_path, bbox_inches="tight")
    plt.close(fig)

def plot_all_metrics_scores_over_time(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "scores_over_time":
        return

    weights = payload.get("metric_weights", {})

    labels = np.asarray(payload["labels"])
    t = np.arange(len(labels))

    metric_names = [
        m for m in DESIRED_METRIC_ORDER
        if m in payload.get("metrics", {})
        and isinstance(payload["metrics"].get(m), dict)
        and "scores_0_1" in payload["metrics"][m]
    ]

    if not metric_names:
        raise ValueError("No metrics with 'scores_0_1' found in payload.")

    n = len(metric_names)

    fig_h = max(3.0, 2.0 * n)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(14, fig_h),
        sharex=True
    )
    if n == 1:
        axes = [axes]

    anom_thresh = payload.get("anom_if_score_above", 0.5)

    for ax, metric_name in zip(axes, metric_names):
        scores = np.asarray(payload["metrics"][metric_name]["scores_0_1"], dtype=float)

        ax.axhspan(anom_thresh, 1.02, alpha=0.08, color="red", zorder=0)
        ax.axhspan(-0.02, anom_thresh, alpha=0.06, color="green", zorder=0)

        # score line
        ax.plot(t, scores, linewidth=1)

        # threshold
        ax.axhline(
            y=anom_thresh,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

        # anomaly dots (RED)
        anom = labels == 1
        ax.scatter(
            t[anom],
            scores[anom],
            s=8,
            c="red",
            rasterized=True
        )

        w = weights.get(metric_name, None)
        ylabel = f"{metric_name} (w={w:.2f})" if w is not None else metric_name
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel, rotation=0, labelpad=35, va="center")

        # subtle row separator
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(0.4)

    axes[-1].set_xlabel("t (sample index)")
    fig.suptitle("All metrics scores (0–1) over time\n(red dots = anomalies)", y=0.995)

    # single legend
    handles = [
        plt.Line2D([0], [0], linestyle="--", linewidth=1.2, label=f"threshold ({anom_thresh})"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color="red", label="anomaly (GT)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(paths.plot_graphs_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_all_metrics_scores_scatter_colored(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "scores_over_time":
        return

    weights = payload.get("metric_weights", {})
    
    labels = np.asarray(payload["labels"])
    t = np.arange(len(labels))

    metric_names = [
        m for m in DESIRED_METRIC_ORDER
        if m in payload.get("metrics", {})
        and isinstance(payload["metrics"].get(m), dict)
        and "scores_0_1" in payload["metrics"][m]
    ]

    if not metric_names:
        raise ValueError("No metrics with 'scores_0_1' found in payload.")

    n = len(metric_names)
    
    # Figure sizing: scale height with number of metrics
    fig_h = max(3.0, 2.2 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    anom_thresh = payload.get("anom_if_score_above", 0.5)

    for ax, metric_name in zip(axes, metric_names):
        scores = np.asarray(payload["metrics"][metric_name]["scores_0_1"], dtype=float)
        if len(scores) != len(labels):
            raise ValueError(f"Length mismatch for {metric_name}: scores={len(scores)} labels={len(labels)}")

        # background zones
        ax.axhspan(anom_thresh, 1.02, alpha=0.08, color="red", zorder=0)
        ax.axhspan(-0.02, anom_thresh, alpha=0.06, color="green", zorder=0)

        # threshold line
        ax.axhline(y=anom_thresh, linestyle="--", linewidth=1.2)

        # scatter (rasterized keeps output smaller even at high DPI)
        ax.scatter(t[labels == 0], scores[labels == 0], s=4, c="green", rasterized=True)
        ax.scatter(t[labels == 1], scores[labels == 1], s=4, c="red", rasterized=True)


        
        w = weights.get(metric_name, None)
        ylabel = f"{metric_name} (w={w:.2f})" if w is not None else metric_name
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel, rotation=0, labelpad=35, va="center")

        # thin separator line between rows (optional)
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(0.6)

    # Optional month annotation strip — one label per contiguous anomaly group
    timestamps = payload.get("timestamps")
    anom_groups = None  # list of (center_x, month)
    if timestamps is not None and len(timestamps) == len(labels):
        try:
            ts_pd = [pd.Timestamp(ts) for ts in timestamps]
            anom_idx = np.where(labels == 1)[0]
            if len(anom_idx):
                # Split into contiguous runs (gap > 1 step = new group)
                breaks = np.where(np.diff(anom_idx) > 1)[0] + 1
                groups = np.split(anom_idx, breaks)
                anom_groups = []
                for grp in groups:
                    center = int(grp[len(grp) // 2])   # middle step of the run
                    month  = ts_pd[center].month
                    anom_groups.append((grp[0], grp[-1], center, month))
        except Exception:
            anom_groups = None

    if anom_groups:
        MONTH_COLORS = [
            "#08306b","#6baed6","#41ab5d","#41ab5d","#fee391","#fd8d3c",
            "#d73027","#d73027","#fd8d3c","#fee391","#6baed6","#08306b",
        ]
        MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]

        strip_ax = fig.add_axes([
            axes[-1].get_position().x0,
            0.005,
            axes[-1].get_position().width,
            0.035,
        ])
        strip_ax.set_xlim(axes[-1].get_xlim())
        strip_ax.set_ylim(0, 1)
        strip_ax.axis("off")

        for start, end, center, month in anom_groups:
            color = MONTH_COLORS[month - 1]
            abbr  = MONTH_ABBR[month - 1]
            # shade the span of this anomaly group
            strip_ax.axvspan(start, end, color=color, alpha=0.35, linewidth=0)
            strip_ax.text(
                center, 0.5, abbr,
                ha="center", va="center",
                fontsize=7, color=color, fontweight="bold",
            )

    axes[-1].set_xlabel("t (sample index)")
    fig.suptitle("Metric scores (0–1) over time (colored by label)", y=0.995)

    # One legend for the whole figure (avoid repeating)
    handles = [
        plt.Line2D([0], [0], linestyle="--", linewidth=1.2, label=f"threshold ({anom_thresh})"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color="green", label="normal (0)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color="red", label="anomaly (1)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0.045 if anom_groups else 0, 1, 0.98])
    fig.savefig(paths.colored_graph_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_all_metrics_heatmap(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "scores_over_time":
        return

    metric_names = [
        m for m in DESIRED_METRIC_ORDER
        if m in payload.get("metrics", {})
        and isinstance(payload["metrics"].get(m), dict)
        and "scores_0_1" in payload["metrics"][m]
    ]

    if not metric_names:
        raise ValueError("No metrics with 'scores_0_1' found in payload.")

    labels = np.asarray(payload["labels"]).astype(int)
    T = len(labels)

    M = np.vstack([
        np.asarray(payload["metrics"][name]["scores_0_1"], dtype=float)
        for name in metric_names
    ])

    assert M.shape[1] == T

    n_metrics = len(metric_names)

    # ---- Colormap centered at anom_if_score_above ---
    anom_thresh = payload.get("anom_if_score_above", 0.5)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=anom_thresh, vmax=1.0)

    fig, ax = plt.subplots(
        figsize=(14, max(3, 0.6 * (n_metrics + 1)))
    )

    # ---- Draw heatmap for metrics only ----
    im = ax.imshow(
        M,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
        origin="upper"
    )

    # ---- Thin horizontal separators between metric rows ----
    for y in range(len(metric_names)):
        plt.axhline(y + 0.5, color="black", linewidth=0.4, alpha=0.6)

    # ---- Axes labels ----
    ax.set_yticks(np.arange(n_metrics + 1))
    ax.set_yticklabels(metric_names + ["LABEL (GT)"])
    ax.set_xlabel("t (sample index)")
    ax.set_ylabel("metric")
    ax.set_title("All metrics scores (0–1) over time")

    # ---- Draw label row background (white) ----
    label_row_y = n_metrics
    ax.add_patch(
        Rectangle(
            (-0.5, label_row_y - 0.5),
            T,
            1.0,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8
        )
    )

    for t in np.where(labels == 1)[0]:
        ax.add_patch(
            Rectangle(
                (t - 0.5, label_row_y - 0.5),
                1.0,
                1.0,
                facecolor="black",
                edgecolor="none"
            )
        )

    # ---- Separator line above label row ----
    ax.axhline(n_metrics - 0.5, color="black", linewidth=1)

    # ---- Colorbar legend for scores ----
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("score (cold = normal, warm = anomaly)")
    cbar.set_ticks([0.0, anom_thresh, 1.0])
    cbar.set_ticklabels(["0 (normal)", f"{anom_thresh} (threshold)", "1 (anomaly)"])

    plt.tight_layout()
    plt.savefig(paths.metric_confidence_heatmap, bbox_inches="tight", dpi=150)
    plt.close()

def plot_decoder_sensitivity(kind: str, payload: dict, paths: Trial_Paths):
    """Plot decoder sensitivity over time.

    Sensitivity = mean std of reconstructions when the latent is perturbed by N(0,1) noise.
    High → decoder responds to z (healthy). Low → decoder ignores z (posterior collapse).
    """
    if kind != "scores_over_time":
        return

    sensitivity = payload.get("decoder_sensitivity")
    if not sensitivity:
        return

    labels = np.asarray(payload.get("labels", []))
    s = np.asarray(sensitivity, dtype=float)
    t = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(14, 3.5))

    if len(labels) == len(s):
        ax.fill_between(t, 0, s.max() * 1.05, where=(labels == 1),
                        color="red", alpha=0.10, label="anomaly (GT)")

    ax.plot(t, s, linewidth=1.2, color="steelblue", label="sensitivity")
    ax.set_xlabel("step (eval index)")
    ax.set_ylabel("mean std of recon")
    ax.set_title("Decoder sensitivity to latent perturbations\n"
                 "(low = decoder ignores z → posterior collapse)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(paths.decoder_sensitivity_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def read_log_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()

    if first.lstrip().startswith("#"):
        header = first.lstrip()[1:].strip()
        cols = [c.strip() for c in header.split(",")]
        df = pd.read_csv(path, header=None, skiprows=1)
        df.columns = cols
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    return df

def parse_metrics_cell(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    s = str(s).strip()
    if not s:
        return {}
    # ast.literal_eval can't handle bare nan/inf produced by float("nan")/float("inf")
    import re
    s = re.sub(r'\bnan\b', 'None', s)
    s = re.sub(r'\binf\b', 'None', s)
    return ast.literal_eval(s)

def expand_metrics(df, metrics_col="metrics"):
    metrics_list = df[metrics_col].apply(parse_metrics_cell).tolist()

    metric_names = set()
    for md in metrics_list:
        metric_names.update(md.keys())
    metric_names = sorted(metric_names)

    out = df.copy()
    for name in metric_names:
        out[f"score__{name}"] = np.nan
        out[f"threshold__{name}"] = np.nan

    for i, md in enumerate(metrics_list):
        for name, entry in md.items():
            out.at[out.index[i], f"score__{name}"] = entry.get("score", np.nan)
            out.at[out.index[i], f"threshold__{name}"] = entry.get("threshold", np.nan)

    return out

def _as_float(x, name):
    # common “trailing comma” bug: 2.2, becomes (2.2,)
    if isinstance(x, (tuple, list, np.ndarray)):
        if len(x) == 1:
            x = x[0]
        else:
            raise TypeError(f"{name} must be scalar, got {x} ({type(x)})")
    return float(x)

def plot_metric_lines_scores_thresholds(kind: str, payload: dict, paths: "Trial_Paths"):
    """
    One HIGH-RES figure with ALL metrics stacked vertically.

    For each metric subplot:
      - Background shading: GT normal (green) / GT anomaly (red)
      - score + threshold lines
      - trained markers (black for trained=1, lightgray for trained=0)
      - pred stripe: white baseline + black where pred==1

    Saves ONE PNG: paths.plot_scores_vs_thresholds
    """
    if kind != "scores_over_time":
        return

    # ------------------------
    # Plot controls
    # ------------------------
    step_start: int | None = None
    step_end: int | None = None

    clip_q_low: float = 1.0
    clip_q_high: float = 99.0
    pad_frac: float = 0.05
    target_xticks: int = 14

    figsize_w: float = 16.0
    row_h: float = 2.3
    dpi: int = 450  # HIGH RES

    # ------------------------
    # Load + expand
    # ------------------------
    df = read_log_csv(paths.predictions_csv)
    df = expand_metrics(df)

    if "step" not in df.columns:
        raise ValueError("CSV missing 'step' column.")

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"].notna()].copy()

    if step_start is not None:
        df = df[df["step"] >= step_start]
    if step_end is not None:
        df = df[df["step"] <= step_end]

    df = df.sort_values("step").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows left after filtering.")

    # required columns
    if "label" not in df.columns:
        raise ValueError("CSV missing 'label' column.")
    if "final_pred" not in df.columns:
        df["final_pred"] = 0
    if "trained" not in df.columns:
        df["trained"] = 0

    # ------------------------
    # Choose metrics robustly
    # payload["metrics"] might be:
    #   - None
    #   - list like ["elbo", "recon"]
    #   - dict like {"elbo": {...}, "recon": {...}}
    # ------------------------
    metrics_obj = payload.get("metrics", None)
    if metrics_obj is None:
        metrics = sorted({c.split("__", 1)[1] for c in df.columns if c.startswith("score__")})
    elif isinstance(metrics_obj, dict):
        metrics = list(metrics_obj.keys())
    else:
        # list/tuple/set/np array
        metrics = list(metrics_obj)

    # keep only metrics that actually exist in df
    metrics = [m for m in metrics if f"score__{m}" in df.columns and f"threshold__{m}" in df.columns]

    if len(metrics) == 0:
        raise ValueError("No valid metrics to plot (missing score__/threshold__ columns).")

    # ------------------------
    # Global arrays (aligned)
    # ------------------------
    x_all = df["step"].astype(int).to_numpy()

    labels_all = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    preds_all = pd.to_numeric(df["final_pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    trained_all = pd.to_numeric(df["trained"], errors="coerce").fillna(0).astype(int).to_numpy()

    is_anom_all = labels_all == 1
    is_norm_all = labels_all == 0

    # ------------------------
    # Figure
    # ------------------------
    n = len(metrics)
    fig_h = max(3.0, row_h * n)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize_w, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    # x ticks
    step_min = int(np.nanmin(x_all))
    step_max = int(np.nanmax(x_all))
    step_range = max(step_max - step_min, 1)
    tick_step = max(1, step_range // target_xticks)
    xticks = np.arange(step_min, step_max + 1, tick_step)

    # ------------------------
    # Plot each metric
    # ------------------------
    for ax, m in zip(axes, metrics):
        s = pd.to_numeric(df[f"score__{m}"], errors="coerce").to_numpy()
        th = pd.to_numeric(df[f"threshold__{m}"], errors="coerce").to_numpy()

        # robust y limits from BOTH score and threshold
        y = np.concatenate([s[np.isfinite(s)], th[np.isfinite(th)]], axis=0)
        if y.size == 0:
            ax.set_visible(False)
            continue

        lo = np.percentile(y, clip_q_low)
        hi = np.percentile(y, clip_q_high)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = np.nanmin(y)
            hi = np.nanmax(y)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = -1.0, 1.0

        pad = (hi - lo) * pad_frac
        y_min = lo - pad
        y_max = hi + pad

        # background shading
        ax.fill_between(x_all, y_min, y_max, where=is_anom_all, color="red", alpha=0.08)
        ax.fill_between(x_all, y_min, y_max, where=is_norm_all, color="green", alpha=0.05)

        # main curves
        ax.plot(x_all, s, linewidth=1.3, label="score")
        ax.plot(x_all, th, linewidth=1.3, label="threshold")

        # trained markers
        tr1 = trained_all == 1
        tr0 = trained_all == 0
        ax.scatter(x_all[tr1], s[tr1], color="black", s=10, marker="o", zorder=6)
        ax.scatter(x_all[tr0], s[tr0], color="lightgray", s=7, marker="o", zorder=5)

        # pred stripe (below data)
        pred_y = (y_min - 0.25 * pad)
        ax.plot(x_all, np.full_like(x_all, pred_y, dtype=float),
                color="white", linewidth=4, solid_capstyle="butt", zorder=3)
        y_black = np.where(preds_all == 1, pred_y, np.nan)
        ax.plot(x_all, y_black, color="black", linewidth=4,
                solid_capstyle="butt", zorder=4)

        ax.set_title(m)
        ax.set_ylim(y_min - 0.5 * pad, y_max)

        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(0.6)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xlabel("step")

    # ------------------------
    # Single legend for whole figure
    # ------------------------
    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.3, label="score"),
        plt.Line2D([0], [0], color="black", linewidth=1.3, linestyle="--", label="threshold"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, color="green", label="GT normal (0)"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, color="red", label="GT anomaly (1)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color="black", label="trained=1"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color="lightgray", label="trained=0"),
        plt.Line2D([0], [0], color="black", linewidth=4, label="pred=1 stripe"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.99))

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(paths.plot_scores_vs_thresholds, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_latent_pca_2d(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "2d_3d_display":
        return
    # month plot will write the 2D PNG; skip here to avoid overwrite
    if payload.get("timestamps"):
        return

    labels = np.asarray(payload["labels"])
    Z = np.asarray(payload["latent_mu"], dtype=float)

    if Z.ndim == 3:
        N, T, Zdim = Z.shape
        Z = Z.reshape(N * T, Zdim)
        labels = np.repeat(labels, T)
    elif Z.ndim != 2:
        raise ValueError("latent_mu must be (N,Z) or (N,T,Z)")

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(Z2[labels == 0, 0], Z2[labels == 0, 1],
               s=8, c="green", alpha=0.6, label="normal")
    ax.scatter(Z2[labels == 1, 0], Z2[labels == 1, 1],
               s=8, c="red", alpha=0.6, label="anomaly")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent Space (PCA 2D)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(paths.plot_latent_pca_2d, dpi=300)
    plt.close(fig)

def plot_latent_pca_3d(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "2d_3d_display":
        return
    # month plot will write both PNG and HTML; skip here to avoid overwrite
    if payload.get("timestamps"):
        return

    labels = np.asarray(payload["labels"])
    Z = np.asarray(payload["latent_mu"], dtype=float)

    if Z.ndim == 3:
        N, T, Zdim = Z.shape
        Z = Z.reshape(N * T, Zdim)
        labels = np.repeat(labels, T)
    elif Z.ndim != 2:
        raise ValueError("latent_mu must be (N,Z) or (N,T,Z)")

    pca = PCA(n_components=3)
    Z3 = pca.fit_transform(Z)

    # --- static PNG (always saved) ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Z3[labels == 0, 0], Z3[labels == 0, 1], Z3[labels == 0, 2],
               s=6, c="green", alpha=0.5, label="normal")
    ax.scatter(Z3[labels == 1, 0], Z3[labels == 1, 1], Z3[labels == 1, 2],
               s=6, c="red", alpha=0.6, label="anomaly")

    ax.set_title("Latent Space (PCA 3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    fig.tight_layout()
    fig.savefig(paths.plot_latent_pca_3d, dpi=300)
    plt.close(fig)

    # --- interactive HTML (standalone only) ---
    if paths.standalone:
        import plotly.graph_objects as go

        normal_mask = labels == 0
        anom_mask   = labels == 1

        fig_html = go.Figure()
        fig_html.add_trace(go.Scatter3d(
            x=Z3[normal_mask, 0], y=Z3[normal_mask, 1], z=Z3[normal_mask, 2],
            mode="markers",
            marker=dict(size=3, color="green", opacity=0.5),
            name="normal",
        ))
        fig_html.add_trace(go.Scatter3d(
            x=Z3[anom_mask, 0], y=Z3[anom_mask, 1], z=Z3[anom_mask, 2],
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.6),
            name="anomaly",
        ))
        fig_html.update_layout(
            title="Latent Space (PCA 3D) — interactive",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            legend=dict(itemsizing="constant"),
        )
        fig_html.write_html(paths.plot_latent_pca_3d_html)


def plot_latent_pca_3d_confusion(kind: str, payload: dict, paths: Trial_Paths):
    """3D PCA latent scatter coloured by TP/FP/TN/FN confusion class.

    Static PNG: 4 colours (TN=blue, TP=orange, FP=red, FN=magenta).
    Interactive HTML: same 4 groups, plus a second trace-set that colours
    each sample by its continuous final conf01 score, togglable in the legend.
    """
    if kind != "2d_3d_display":
        return

    labels = np.asarray(payload["labels"])
    Z = np.asarray(payload["latent_mu"], dtype=float)
    final_scores = np.asarray(
        payload["metrics"]["final"]["scores_0_1"], dtype=float
    )
    ensemble_threshold = float(payload.get("ensemble_threshold", 0.5))

    if Z.ndim == 3:
        N, T, Zdim = Z.shape
        Z = Z.reshape(N * T, Zdim)
        labels = np.repeat(labels, T)
        final_scores = np.repeat(final_scores, T)
    elif Z.ndim != 2:
        raise ValueError("latent_mu must be (N,Z) or (N,T,Z)")

    if len(final_scores) != len(labels):
        # scores may be shorter if streaming stopped early; trim to match
        n = min(len(final_scores), len(labels))
        labels = labels[:n]
        Z = Z[:n]
        final_scores = final_scores[:n]

    # derive per-sample binary prediction from the continuous score
    preds = (final_scores >= ensemble_threshold).astype(int)
    # handle NaN scores: treat as predicted-normal (conservative)
    preds[~np.isfinite(final_scores)] = 0

    tn_mask = (preds == 0) & (labels == 0)
    tp_mask = (preds == 1) & (labels == 1)
    fp_mask = (preds == 1) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)

    pca = PCA(n_components=3)
    Z3 = pca.fit_transform(Z)

    # --- static PNG ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    groups = [
        (tn_mask, "TN – normal, correct",  "steelblue",  4, 0.35),
        (tp_mask, "TP – anomaly, detected", "darkorange", 18, 0.9),
        (fn_mask, "FN – anomaly, missed",   "magenta",    18, 0.9),
        (fp_mask, "FP – normal, false alarm","crimson",   10, 0.7),
    ]
    for mask, label_str, color, size, alpha in groups:
        if mask.any():
            ax.scatter(
                Z3[mask, 0], Z3[mask, 1], Z3[mask, 2],
                s=size, c=color, alpha=alpha, label=f"{label_str} (n={mask.sum()})"
            )

    ax.set_title(f"Latent Space — confusion (thr={ensemble_threshold:.2f})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(paths.plot_latent_pca_3d_confusion, dpi=300)
    plt.close(fig)


def plot_latent_pca_3d_month(kind: str, payload: dict, paths: Trial_Paths):
    """3D PCA latent scatter coloured by calendar month (Jan=1 … Dec=12).

    Seasonal colour scheme (winter=blue, summer=red, transitions in between).
    A second layer overlays FP and FN samples as large X/diamond markers.

    Outputs:
      - latent_pca_3d_month.html — interactive 3D HTML coloured by month
    """
    if kind != "2d_3d_display":
        return

    timestamps = payload.get("timestamps")
    if not timestamps:
        return

    # Seasonal palette — Index 0 = January
    MONTH_COLORS = [
        "#08306b",  # Jan  — dark blue
        "#6baed6",  # Feb  — light blue
        "#41ab5d",  # Mar  — green
        "#41ab5d",  # Apr  — green
        "#fee391",  # May  — yellow
        "#fd8d3c",  # Jun  — orange
        "#d73027",  # Jul  — red
        "#d73027",  # Aug  — red
        "#fd8d3c",  # Sep  — orange
        "#fee391",  # Oct  — yellow
        "#6baed6",  # Nov  — light blue
        "#08306b",  # Dec  — dark blue
    ]
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    Z = np.asarray(payload["latent_mu"], dtype=float)
    if Z.ndim == 3:
        Z = Z[:, 0, :]          # take first timestep of each window → (N, Z)
    elif Z.ndim != 2:
        raise ValueError("latent_mu must be (N,Z) or (N,T,Z)")

    months = []
    years  = []
    for t in timestamps:
        try:
            ts = pd.Timestamp(t)
            months.append(ts.month)
            years.append(ts.year)
        except Exception:
            months.append(0)
            years.append(0)
    months = np.array(months, dtype=int)
    years  = np.array(years,  dtype=int)

    n = min(len(Z), len(months))
    Z = Z[:n]
    months = months[:n]
    years  = years[:n]

    valid = months > 0
    if not valid.any():
        return

    Z_v = Z[valid]
    m_v = months[valid]
    y_v = years[valid]

    # ── Year filtering ──────────────────────────────────────────────────────
    plot_cfg    = payload.get("plot_settings", {})
    focus_years = plot_cfg.get("focus_years")
    year_window = int(plot_cfg.get("year_window", 0))
    ym = np.ones(len(Z_v), dtype=bool)  # keep all by default
    if focus_years:
        allowed = set()
        for yr in focus_years:
            for delta in range(-year_window, year_window + 1):
                allowed.add(yr + delta)
        ym = np.isin(y_v, list(allowed))
        if not ym.any():
            return
    Z_v = Z_v[ym]
    m_v = m_v[ym]

    pca3 = PCA(n_components=3)
    Z3 = pca3.fit_transform(Z_v)
    pca2 = PCA(n_components=2)
    Z2 = pca2.fit_transform(Z_v)

    # ── confusion masks ────────────────────────────────────────────────────
    fp_mask = fn_mask = tp_mask = tn_mask = np.zeros(len(Z_v), dtype=bool)
    labels_raw   = payload.get("labels")
    final_scores = payload.get("metrics", {}).get("final", {}).get("scores_0_1")
    ensemble_thr = float(payload.get("ensemble_threshold", 0.5))
    if labels_raw is not None and final_scores is not None:
        labels_arr = np.asarray(labels_raw)[:n][valid][ym]
        scores_arr = np.asarray(final_scores, dtype=float)
        scores_arr = scores_arr[:n][valid][ym] if len(scores_arr) >= n else np.full(len(Z_v), np.nan)
        preds_arr  = (scores_arr >= ensemble_thr).astype(int)
        preds_arr[~np.isfinite(scores_arr)] = 0
        tn_mask = (preds_arr == 0) & (labels_arr == 0)
        tp_mask = (preds_arr == 1) & (labels_arr == 1)
        fp_mask = (preds_arr == 1) & (labels_arr == 0)
        fn_mask = (preds_arr == 0) & (labels_arr == 1)

    # ── Subsample normals to cap file size (keep all anomalies) ───────────
    max_normals = int(plot_cfg.get("max_normals_per_month", 5_000))
    if max_normals > 0:
        rng = np.random.default_rng(42)
        anom_mask = tp_mask | fp_mask | fn_mask
        sub_idx_parts = []
        for mo in range(1, 13):
            mo_idx = np.where(m_v == mo)[0]
            if not len(mo_idx):
                continue
            anom_in_mo   = mo_idx[anom_mask[mo_idx]]
            normal_in_mo = mo_idx[~anom_mask[mo_idx]]
            n_sample = min(max_normals, len(normal_in_mo))
            sampled_normals = rng.choice(normal_in_mo, size=n_sample, replace=False)
            sub_idx_parts.append(anom_in_mo)
            sub_idx_parts.append(sampled_normals)
        sub_idx = np.sort(np.concatenate(sub_idx_parts))
        Z3      = Z3[sub_idx]
        m_v     = m_v[sub_idx]
        tn_mask = tn_mask[sub_idx]
        tp_mask = tp_mask[sub_idx]
        fp_mask = fp_mask[sub_idx]
        fn_mask = fn_mask[sub_idx]

    # ── Interactive HTML ──────────────────────────────────────────────────
    if not paths.standalone:
        return

    import plotly.graph_objects as go

    fig_html = go.Figure()

    for mo in range(1, 13):
        mask = m_v == mo
        if not mask.any():
            continue
        fig_html.add_trace(go.Scatter3d(
            x=Z3[mask, 0], y=Z3[mask, 1], z=Z3[mask, 2],
            mode="markers",
            marker=dict(size=3, color=MONTH_COLORS[mo - 1], opacity=0.6),
            name=MONTH_NAMES[mo - 1],
            text=[f"month={MONTH_NAMES[mo - 1]}" for _ in range(mask.sum())],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            legendgroup="months",
        ))

    overlay_groups = [
        (tn_mask, "TN – normal, correct",  "steelblue",  3, 0.3),
        (tp_mask, "TP – anomaly, detected", "darkorange", 6, 0.9),
        (fn_mask, "FN – missed",            "magenta",    6, 0.9),
        (fp_mask, "FP – false alarm",       "crimson",    5, 0.7),
    ]
    for mask, label_str, color, sz, op in overlay_groups:
        if not mask.any():
            continue
        mo_labels = [MONTH_NAMES[m - 1] for m in m_v[mask]]
        fig_html.add_trace(go.Scatter3d(
            x=Z3[mask, 0], y=Z3[mask, 1], z=Z3[mask, 2],
            mode="markers",
            marker=dict(size=sz, color=color, opacity=op),
            name=f"{label_str} (n={mask.sum()})",
            text=[f"{label_str} | month={m}" for m in mo_labels],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            legendgroup="confusion",
        ))

    fig_html.update_layout(
        title="Latent Space (PCA 3D) — coloured by month",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
    )
    fig_html.write_html(paths.plot_latent_pca_3d_month_html, include_plotlyjs='cdn')


# ── Manifold projection helpers (shared by UMAP / Isomap / t-SNE sinks) ──────

_MANIFOLD_MONTH_COLORS = [
    "#08306b","#6baed6","#41ab5d","#41ab5d","#fee391","#fd8d3c",
    "#d73027","#d73027","#fd8d3c","#fee391","#6baed6","#08306b",
]
_MANIFOLD_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]


def _manifold_prepare(payload):
    """Extract and validate all arrays needed for manifold plots from payload.

    Returns (Z, months, hod, doy, tn, tp, fp, fn, sub_idx, plot_cfg)
    or None if the payload is missing required data.
    """
    import plotly.graph_objects as go

    timestamps = payload.get("timestamps")
    if not timestamps:
        return None
    Z = np.asarray(payload.get("latent_mu", []), dtype=float)
    if Z.ndim == 3:
        Z = Z[:, 0, :]
    if Z.ndim != 2 or len(Z) == 0:
        return None

    ts_pd = pd.DatetimeIndex([pd.Timestamp(t) for t in timestamps])
    n = min(len(Z), len(ts_pd))
    Z, ts_pd = Z[:n], ts_pd[:n]

    months = ts_pd.month.to_numpy(dtype=int)
    hod    = ts_pd.hour.to_numpy(dtype=int)
    doy    = ts_pd.day_of_year.to_numpy(dtype=int)

    # Confusion masks
    tn = tp = fp = fn = np.zeros(n, dtype=bool)
    labels_raw   = payload.get("labels")
    final_scores = payload.get("metrics", {}).get("final", {}).get("scores_0_1")
    thr = float(payload.get("ensemble_threshold", 0.5))
    if labels_raw is not None and final_scores is not None:
        la = np.asarray(labels_raw, dtype=int)[:n]
        sc = np.asarray(final_scores, dtype=float)
        sc = sc[:n] if len(sc) >= n else np.full(n, np.nan)
        pr = (sc >= thr).astype(int)
        pr[~np.isfinite(sc)] = 0
        tn = (pr == 0) & (la == 0)
        tp = (pr == 1) & (la == 1)
        fp = (pr == 1) & (la == 0)
        fn = (pr == 0) & (la == 1)

    # Subsample normals
    plot_cfg = payload.get("plot_settings", {})
    max_nm   = int(plot_cfg.get("max_normals_per_month", 5_000))
    anom     = tp | fp | fn
    rng      = np.random.default_rng(42)
    parts    = []
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        if not len(idx): continue
        parts.append(idx[anom[idx]])
        norm = idx[~anom[idx]]
        parts.append(rng.choice(norm, size=min(max_nm, len(norm)), replace=False))
    sub_idx = np.sort(np.concatenate(parts)) if parts else np.arange(n)

    return Z, months, hod, doy, tn, tp, fp, fn, sub_idx, plot_cfg


def _manifold_write_month(Z3s, mos, tn, tp, fp, fn, title, ax, out_path):
    import plotly.graph_objects as go
    fig = go.Figure()
    for mo in range(1, 13):
        mask = mos == mo
        if not mask.any(): continue
        fig.add_trace(go.Scatter3d(
            x=Z3s[mask,0], y=Z3s[mask,1], z=Z3s[mask,2], mode="markers",
            marker=dict(size=3, color=_MANIFOLD_MONTH_COLORS[mo-1], opacity=0.65),
            name=_MANIFOLD_MONTH_NAMES[mo-1],
            hovertemplate=f"month={_MANIFOLD_MONTH_NAMES[mo-1]}<extra></extra>",
            legendgroup="months",
        ))
    for mask, label, color, sz, op in [
        (tn,"TN – normal","steelblue",3,0.20), (tp,"TP – detected","darkorange",7,0.90),
        (fn,"FN – missed","magenta",7,0.90),   (fp,"FP – false alarm","crimson",5,0.70),
    ]:
        if not mask.any(): continue
        fig.add_trace(go.Scatter3d(
            x=Z3s[mask,0], y=Z3s[mask,1], z=Z3s[mask,2], mode="markers",
            marker=dict(size=sz, color=color, opacity=op),
            name=f"{label} (n={mask.sum()})",
            legendgroup="confusion",
        ))
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=ax[0], yaxis_title=ax[1], zaxis_title=ax[2]),
                      legend=dict(itemsizing="constant", groupclick="toggleitem"))
    fig.write_html(out_path, include_plotlyjs="cdn")


def _manifold_write_cyclic(Z3s, color_vals, cmin, cmax, colorbar_title,
                            hover_extra, title, ax, out_path):
    import plotly.graph_objects as go
    order = np.argsort(color_vals)
    fig = go.Figure(go.Scatter3d(
        x=Z3s[order,0], y=Z3s[order,1], z=Z3s[order,2], mode="markers",
        marker=dict(size=3, color=color_vals[order], cmin=cmin, cmax=cmax,
                    colorscale="HSV", opacity=0.70,
                    colorbar=dict(title=colorbar_title), showscale=True),
        text=[hover_extra[i] for i in order],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=ax[0], yaxis_title=ax[1], zaxis_title=ax[2]))
    fig.write_html(out_path, include_plotlyjs="cdn")


# ── UMAP sink ─────────────────────────────────────────────────────────────────

def plot_latent_umap(kind: str, payload: dict, paths: Trial_Paths):
    """3D UMAP projection: month, hour-of-day, day-of-year HTML outputs.

    Enabled by setting  plot_settings.latent_umap: true  in the run YAML.
    Hyperparameters:  plot_settings.umap_n_neighbors (default 150)
                      plot_settings.umap_min_dist    (default 0.1)
    """
    if kind != "2d_3d_display" or not paths.standalone:
        return
    result = _manifold_prepare(payload)
    if result is None:
        return
    Z, months, hod, doy, tn, tp, fp, fn, sub_idx, plot_cfg = result
    if not plot_cfg.get("latent_umap", False):
        return

    try:
        import umap as umap_lib
    except ImportError:
        print("[plot_latent_umap] umap-learn not installed — skipping. pip install umap-learn")
        return

    n_neighbors = int(plot_cfg.get("umap_n_neighbors", 150))
    min_dist    = float(plot_cfg.get("umap_min_dist", 0.1))
    reducer = umap_lib.UMAP(n_components=3, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=42, n_jobs=-1)
    Z3 = reducer.fit_transform(Z)

    Z3s  = Z3[sub_idx]
    mos  = months[sub_idx]
    hods = hod[sub_idx]
    doys = doy[sub_idx]
    ax   = ("UMAP-1", "UMAP-2", "UMAP-3")
    sub  = f"n_neighbors={n_neighbors}  min_dist={min_dist}"

    _manifold_write_month(
        Z3s, mos, tn[sub_idx], tp[sub_idx], fp[sub_idx], fn[sub_idx],
        title=f"Latent – UMAP 3D · month<br><sup>{sub}</sup>",
        ax=ax, out_path=paths.latent_umap_month_html)
    _manifold_write_cyclic(
        Z3s, hods, 0, 23, "Hour of day",
        [f"hour={h}  month={_MANIFOLD_MONTH_NAMES[m-1]}" for h,m in zip(hods,mos)],
        f"Latent – UMAP 3D · hour of day<br><sup>{sub}</sup>", ax, paths.latent_umap_hod_html)
    _manifold_write_cyclic(
        Z3s, doys, 1, 365, "Day of year",
        [f"doy={d}  month={_MANIFOLD_MONTH_NAMES[m-1]}" for d,m in zip(doys,mos)],
        f"Latent – UMAP 3D · day of year<br><sup>{sub}</sup>", ax, paths.latent_umap_doy_html)


# ── Isomap sink ───────────────────────────────────────────────────────────────

def plot_latent_isomap(kind: str, payload: dict, paths: Trial_Paths):
    """3D Isomap projection: month, hour-of-day, day-of-year HTML outputs.

    Enabled by setting  plot_settings.latent_isomap: true  in the run YAML.
    Hyperparameter:     plot_settings.isomap_n_neighbors (default 30)
    """
    if kind != "2d_3d_display" or not paths.standalone:
        return
    result = _manifold_prepare(payload)
    if result is None:
        return
    Z, months, hod, doy, tn, tp, fp, fn, sub_idx, plot_cfg = result
    if not plot_cfg.get("latent_isomap", False):
        return

    from sklearn.manifold import Isomap
    n_neighbors = int(plot_cfg.get("isomap_n_neighbors", 30))
    iso = Isomap(n_neighbors=n_neighbors, n_components=3, n_jobs=-1)
    Z3  = iso.fit_transform(Z)

    Z3s  = Z3[sub_idx]
    mos  = months[sub_idx]
    hods = hod[sub_idx]
    doys = doy[sub_idx]
    ax   = ("Isomap-1", "Isomap-2", "Isomap-3")
    sub  = f"n_neighbors={n_neighbors}"

    _manifold_write_month(
        Z3s, mos, tn[sub_idx], tp[sub_idx], fp[sub_idx], fn[sub_idx],
        title=f"Latent – Isomap 3D · month<br><sup>{sub}</sup>",
        ax=ax, out_path=paths.latent_isomap_month_html)
    _manifold_write_cyclic(
        Z3s, hods, 0, 23, "Hour of day",
        [f"hour={h}  month={_MANIFOLD_MONTH_NAMES[m-1]}" for h,m in zip(hods,mos)],
        f"Latent – Isomap 3D · hour of day<br><sup>{sub}</sup>", ax, paths.latent_isomap_hod_html)
    _manifold_write_cyclic(
        Z3s, doys, 1, 365, "Day of year",
        [f"doy={d}  month={_MANIFOLD_MONTH_NAMES[m-1]}" for d,m in zip(doys,mos)],
        f"Latent – Isomap 3D · day of year<br><sup>{sub}</sup>", ax, paths.latent_isomap_doy_html)


# ── t-SNE sink ────────────────────────────────────────────────────────────────

def plot_latent_tsne(kind: str, payload: dict, paths: Trial_Paths):
    """3D t-SNE projection: month, hour-of-day, day-of-year HTML outputs.

    Enabled by setting  plot_settings.latent_tsne: true  in the run YAML.
    Hyperparameters:  plot_settings.tsne_perplexity (default 30)
                      plot_settings.tsne_n_iter     (default 1000)

    Note: t-SNE does NOT preserve global distances. It is good for seeing
    whether anomaly points cluster separately, but not for reading temporal
    direction. Prefer Isomap or UMAP for that.
    """
    if kind != "2d_3d_display" or not paths.standalone:
        return
    result = _manifold_prepare(payload)
    if result is None:
        return
    Z, months, hod, doy, tn, tp, fp, fn, sub_idx, plot_cfg = result
    if not plot_cfg.get("latent_tsne", False):
        return

    from sklearn.manifold import TSNE
    import sklearn
    perplexity  = float(plot_cfg.get("tsne_perplexity", 30))
    n_iter      = int(plot_cfg.get("tsne_n_iter", 1000))
    iter_kwarg  = ("max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5)
                   else "n_iter")
    tsne = TSNE(n_components=3, perplexity=perplexity, **{iter_kwarg: n_iter},
                init="pca", random_state=42, n_jobs=-1)
    Z3   = tsne.fit_transform(Z)

    Z3s  = Z3[sub_idx]
    mos  = months[sub_idx]
    hods = hod[sub_idx]
    doys = doy[sub_idx]
    ax   = ("tSNE-1", "tSNE-2", "tSNE-3")
    sub  = f"perplexity={perplexity}  n_iter={n_iter}  KL={tsne.kl_divergence_:.3f}"

    _manifold_write_month(
        Z3s, mos, tn[sub_idx], tp[sub_idx], fp[sub_idx], fn[sub_idx],
        title=f"Latent – t-SNE 3D · month<br><sup>{sub}</sup>",
        ax=ax, out_path=paths.latent_tsne_month_html)
    _manifold_write_cyclic(
        Z3s, hods, 0, 23, "Hour of day",
        [f"hour={h}  month={_MANIFOLD_MONTH_NAMES[m-1]}" for h,m in zip(hods,mos)],
        f"Latent – t-SNE 3D · hour of day<br><sup>{sub}</sup>", ax, paths.latent_tsne_hod_html)
    _manifold_write_cyclic(
        Z3s, doys, 1, 365, "Day of year",
        [f"doy={d}  month={_MANIFOLD_MONTH_NAMES[m-1]}" for d,m in zip(doys,mos)],
        f"Latent – t-SNE 3D · day of year<br><sup>{sub}</sup>", ax, paths.latent_tsne_doy_html)


def plot_latent_norm_over_time(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "2d_3d_display":
        return

    labels = np.asarray(payload["labels"])
    Z = np.asarray(payload["latent_mu"], dtype=float)

    if Z.ndim == 3:
        Z = Z.mean(axis=1)

    norm = np.linalg.norm(Z, axis=1)
    t = np.arange(len(norm))

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t, norm, linewidth=1)
    ax.scatter(t[labels == 1], norm[labels == 1],
               c="red", s=8, label="anomaly")

    ax.set_title("Latent Norm Over Time")
    ax.set_xlabel("t")
    ax.set_ylabel("||z||")
    ax.legend()

    fig.tight_layout()
    fig.savefig(paths.plot_latent_norm_over_time, dpi=300)
    plt.close(fig)

def plot_latent_mahalanobis(kind: str, payload: dict, paths: Trial_Paths):
    if kind != "2d_3d_display":
        return

    labels = np.asarray(payload["labels"])
    Z = np.asarray(payload["latent_mu"], dtype=float)

    if Z.ndim == 3:
        Z = Z.mean(axis=1)

    mu = np.asarray(payload["latent_mean"])
    cov_inv = np.asarray(payload["latent_cov_inv"])

    diffs = Z - mu
    m_dist = np.sqrt(np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs))

    t = np.arange(len(m_dist))

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t, m_dist, linewidth=1)
    ax.scatter(t[labels == 1], m_dist[labels == 1],
               c="red", s=8, label="anomaly")

    ax.set_title("Latent Mahalanobis Distance Over Time")
    ax.set_xlabel("t")
    ax.set_ylabel("Mahalanobis Distance")
    ax.legend()

    fig.tight_layout()
    fig.savefig(paths.plot_latent_mahalanobis, dpi=300)
    plt.close(fig)
