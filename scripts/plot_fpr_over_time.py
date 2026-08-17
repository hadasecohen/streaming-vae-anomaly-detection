"""
Plot rolling metric over time from _cc_summary.npz files.

Walks a session directory, groups runs by arch and variant, averages over
seeds, and plots a rolling metric (FPR, F1, Recall, or Precision) over
the stream.  Works on both clean baseline (FPR only) and anomaly experiments.

Usage:
    # Clean baseline — FPR only (no positives)
    python scripts/plot_fpr_over_time.py runs/trials_suite/v7/q0_baseline/clean_baseline

    # Anomaly experiment — F1 over time
    python scripts/plot_fpr_over_time.py runs/trials_suite/v7/q1_arch/arch_comparison_best --metric f1

    # Multiple metrics
    python scripts/plot_fpr_over_time.py <session_dir> --metric fpr recall

    # Filter to specific anomaly type or arch
    python scripts/plot_fpr_over_time.py <session_dir> --anomaly Group --arch MLP LSTM

Options:
    --metric    One or more of: fpr, f1, recall, precision   [default: fpr]
    --window    Rolling window size in steps                  [default: 100000]
    --anomaly   Filter by anomaly type (e.g. Group Contextual Clean)
    --arch      Filter by architecture name
    --out       Output PNG path  [default: <session_dir>/rolling_metrics.png]
"""

import argparse
import glob
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d


# confusion_enc: {0:TP, 1:FP, 2:TN, 3:FN}
_ENC = {"TP": 0, "FP": 1, "TN": 2, "FN": 3}

ARCH_ORDER  = ["MLP", "MLP_Cyclic", "LSTM", "TF_VAE"]
ARCH_LABELS = {
    "MLP":        "MLP-VAE",
    "MLP_Cyclic": "MLP-VAE-Cyclic",
    "LSTM":       "LSTM-VAE",
    "TF_VAE":     "Transformer-VAE",
}
COLORS = {
    "MLP":        "#2196F3",
    "MLP_Cyclic": "#4CAF50",
    "LSTM":       "#FF9800",
    "TF_VAE":     "#E91E63",
}
METRIC_LABELS = {
    "fpr":       "FPR",
    "f1":        "F1",
    "recall":    "Recall",
    "precision": "Precision",
}


def rolling_metric(confusion_enc: np.ndarray, metric: str, window: int,
                   fill_nan: bool = False) -> np.ndarray:
    ce = confusion_enc.astype(int)
    tp = (ce == _ENC["TP"]).astype(float)
    fp = (ce == _ENC["FP"]).astype(float)
    tn = (ce == _ENC["TN"]).astype(float)
    fn = (ce == _ENC["FN"]).astype(float)

    def roll(arr):
        return pd.Series(arr).rolling(window=window, min_periods=1).sum().to_numpy()

    TP = roll(tp); FP = roll(fp); TN = roll(tn); FN = roll(fn)

    with np.errstate(invalid="ignore", divide="ignore"):
        gt_pos = TP + FN
        gt_neg = TN + FP
        if metric == "fpr":
            values = np.where(gt_neg > 0, FP / gt_neg, np.nan)
        elif metric == "recall":
            values = np.where(gt_pos > 0, TP / gt_pos, np.nan)
        elif metric == "precision":
            denom = TP + FP
            values = np.where(denom > 0, TP / denom, np.nan)
        elif metric == "f1":
            denom = 2 * TP + FP + FN
            values = np.where((gt_pos > 0) & (denom > 0), 2 * TP / denom, np.nan)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if fill_nan:
        values = np.nan_to_num(values, nan=0.0)
    return values


def discover_runs(session_dir: Path, anomaly_filter=None, arch_filter=None):
    """
    Walk session_dir for _cc_summary.npz files.
    Returns dict: (anomaly, arch, variant) -> list of npz paths.
    """
    runs = defaultdict(list)
    for npz in sorted(session_dir.rglob("_cc_summary.npz")):
        cfg_path = npz.parent / "_suite_config.yaml"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        meta = cfg.get("_compare_meta") or {}
        anomaly = meta.get("anomaly", npz.parent.parent.parent.name)
        arch    = meta.get("arch",    npz.parent.parent.name)
        variant = str(meta.get("variant", "default"))

        if anomaly_filter and anomaly not in anomaly_filter:
            continue
        if arch_filter and arch not in arch_filter:
            continue

        runs[(anomaly, arch, variant)].append(str(npz))
    return runs


_SMOOTH_REF_N = 18883  # window-mode ERA5 steps — sigma is calibrated to this length


def _smooth_safe(s: np.ndarray, sigma: float, n: int) -> np.ndarray:
    """Gaussian smooth with NaN-safe interpolation and auto-scaled sigma."""
    if sigma <= 0:
        return s
    actual_sigma = sigma * n / _SMOOTH_REF_N  # scale proportionally to series length
    actual_sigma = max(actual_sigma, 1.0)
    out = s.copy()
    finite = np.isfinite(s)
    if finite.sum() < 2:
        return out
    idx = np.arange(n)
    filled = np.interp(idx, idx[finite], s[finite])
    smoothed = gaussian_filter1d(filled, sigma=actual_sigma)
    out[:] = smoothed
    out[~finite] = np.nan
    return out


def get_anomaly_cuts(ce_ref: np.ndarray, n_buckets: int = 5):
    """
    Split GT-positive steps into n_buckets equal groups by anomaly count.
    Each bucket covers a time range containing the same number of GT-anomaly steps,
    so every column in the summary table has equal anomaly signal regardless of
    how anomalies are distributed across the stream.

    Returns (cuts_steps, cuts_pct):
      cuts_steps  list of n_buckets+1 step-index boundaries into ce_ref
      cuts_pct    list of n_buckets+1 percentages (for column headers)
    Returns (None, None) when there are fewer anomaly steps than n_buckets.
    """
    n = len(ce_ref)
    gt_pos_idx = np.where(
        (ce_ref == _ENC["TP"]) | (ce_ref == _ENC["FN"])
    )[0]
    total = len(gt_pos_idx)
    if total < n_buckets:
        return None, None
    cuts = [0]
    for i in range(1, n_buckets):
        boundary = int(i * total / n_buckets)
        cuts.append(int(gt_pos_idx[boundary]))
    cuts.append(n)
    pcts = [int(round(100.0 * s / n)) for s in cuts]
    return cuts, pcts


def _metric_from_counts(tp, fp, tn, fn, metric: str) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "f1":
            denom = 2 * tp + fp + fn
            return float(2 * tp / denom) if denom > 0 else np.nan
        elif metric == "recall":
            denom = tp + fn
            return float(tp / denom) if denom > 0 else np.nan
        elif metric == "precision":
            denom = tp + fp
            return float(tp / denom) if denom > 0 else np.nan
        elif metric == "fpr":
            denom = fp + tn
            return float(fp / denom) if denom > 0 else np.nan
    return np.nan


def bucket_metric(ces: list, cuts: list, metric: str) -> list:
    """
    For each bucket defined by consecutive cut pairs, sum TP/FP/FN across
    all steps in that range and compute the metric directly from those totals.
    Averaged over seeds.  Returns n_buckets+1 values (buckets + overall).
    """
    per_seed = []
    for ce in ces:
        row = []
        for i in range(len(cuts) - 1):
            seg = ce[cuts[i]:cuts[i + 1]]
            tp = int(np.sum(seg == _ENC["TP"]))
            fp = int(np.sum(seg == _ENC["FP"]))
            tn = int(np.sum(seg == _ENC["TN"]))
            fn = int(np.sum(seg == _ENC["FN"]))
            row.append(_metric_from_counts(tp, fp, tn, fn, metric))
        # overall = full run
        tp = int(np.sum(ce == _ENC["TP"]))
        fp = int(np.sum(ce == _ENC["FP"]))
        tn = int(np.sum(ce == _ENC["TN"]))
        fn = int(np.sum(ce == _ENC["FN"]))
        row.append(_metric_from_counts(tp, fp, tn, fn, metric))
        per_seed.append(row)
    arr = np.array(per_seed, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0).tolist()


def collect_arch_series(runs_for_anomaly, metric, window, arch_order, smooth=0,
                        fill_nan=False):
    """Return dict (arch, variant) -> (x, mean, std, raw_matrix, variant, ce_ref, ces)."""
    result = {}
    for arch in arch_order:
        for (_, a, variant), paths in runs_for_anomaly.items():
            if a != arch:
                continue
            series = []
            ce_list = []
            for p in paths:
                d = np.load(p, allow_pickle=True)
                ce = d["confusion_enc"].astype(int)
                s = rolling_metric(ce, metric, window, fill_nan=fill_nan)
                series.append(s)
                ce_list.append(ce)
            if not series:
                continue
            min_n = min(len(s) for s in series)
            series = [s[:min_n] for s in series]
            ces = [ce[:min_n] for ce in ce_list]
            ce_ref = ces[0]
            n = min_n
            if smooth > 0:
                series = [_smooth_safe(s, smooth, n) for s in series]
            x   = np.linspace(0, 100, n)
            mat = np.stack(series)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_v = np.nanmean(mat, axis=0)
                std_v  = np.nanstd(mat, axis=0) if mat.shape[0] > 1 else np.zeros(n)
            result[(arch, variant)] = (x, mean_v, std_v, mat, variant, ce_ref, ces)
    return result


def _variant_linestyle(variant: str) -> str:
    """Dashed for point-mode variants, solid for window-mode."""
    return "--" if "point" in variant.lower() else "-"


def plot_metric(ax, runs_for_anomaly, metric, window, arch_order, smooth=0, shade=True,
               lw=0.6, fill_nan=False):
    """Plot one metric panel. Returns collected series dict for table use."""
    series_by_arch = collect_arch_series(runs_for_anomaly, metric, window, arch_order,
                                         smooth=smooth, fill_nan=fill_nan)
    arches_present = {a for (a, _) in series_by_arch}
    multi_variant = len({v for (_, v) in series_by_arch}) > 1
    # Single architecture, multiple variants (e.g. one arch swept over EMA/
    # threshold/stride values): COLORS is keyed by arch, so every line would
    # otherwise share one color. Assign each variant its own color from a
    # qualitative colormap instead; keep the normal per-arch coloring
    # whenever more than one architecture is present.
    variant_color_by_key = {}
    if len(arches_present) == 1 and multi_variant:
        variants_sorted = sorted({v for (_, v) in series_by_arch})
        cmap = matplotlib.colormaps.get_cmap("tab10" if len(variants_sorted) <= 10 else "tab20")
        variant_color_by_key = {
            (next(iter(arches_present)), v): mcolors.to_hex(cmap(i / max(len(variants_sorted) - 1, 1)))
            for i, v in enumerate(variants_sorted)
        }
    for (arch, variant), (x, mean_v, std_v, mat, _, _, _ces) in series_by_arch.items():
        color = variant_color_by_key.get((arch, variant)) or COLORS.get(arch, "#888888")
        label = ARCH_LABELS.get(arch, arch)
        if multi_variant:
            label = f"{label} ({variant})"
        ls = _variant_linestyle(variant) if (multi_variant and not variant_color_by_key) else "-"
        ax.plot(x, mean_v, label=label, color=color, lw=lw, linestyle=ls)
        if shade:
            ax.fill_between(x,
                            np.clip(mean_v - std_v, 0, None),
                            mean_v + std_v,
                            alpha=0.18, color=color)
    return series_by_arch


_INTERVALS = [0, 10, 25, 50, 75, 100]  # fallback fixed breakpoints (%)
_N_BUCKETS  = 5                         # buckets for dynamic anomaly-equal splits


def _resolve_cuts(series_by_arch, n_buckets=_N_BUCKETS):
    """
    Return (cuts_steps_per_arch, col_labels, dynamic, cut_xpcts) where
    cut_xpcts are interior bucket boundary positions in 0-100% for axvlines.
    """
    ref_cuts, ref_pcts = None, None
    for key, entry in series_by_arch.items():
        ce_ref = entry[5]
        if ce_ref is not None:
            ref_cuts, ref_pcts = get_anomaly_cuts(ce_ref, n_buckets)
        if ref_cuts is not None:
            break

    if ref_cuts is not None:
        col_labels = [f"{ref_pcts[i]/100:.2f}–{ref_pcts[i+1]/100:.2f}"
                      for i in range(len(ref_cuts) - 1)]
        col_labels.append("Overall")
        cuts_per_arch = {}
        for key, entry in series_by_arch.items():
            n = len(entry[0])
            cuts_per_arch[key] = [min(c, n) for c in ref_cuts]
        return cuts_per_arch, col_labels, True, list(ref_pcts[1:-1])

    intervals = _INTERVALS
    col_labels = [f"{intervals[i]/100:.2f}–{intervals[i+1]/100:.2f}"
                  for i in range(len(intervals) - 1)]
    col_labels.append("Overall")
    cuts_per_arch = {}
    for key, entry in series_by_arch.items():
        n = len(entry[0])
        cuts_per_arch[key] = [max(0, min(n, n * p // 100)) for p in intervals]
    return cuts_per_arch, col_labels, False, list(intervals[1:-1])


def _make_metric_cmap(metric: str):
    """RdYlGn colormap, inverted for lower-is-better metrics (FPR)."""
    base = matplotlib.colormaps.get_cmap("RdYlGn")
    lims = (0.90, 0.12) if metric == "fpr" else (0.12, 0.90)
    return mcolors.LinearSegmentedColormap.from_list(
        "RdYlGn_mid", base(np.linspace(lims[0], lims[1], 256))
    )


def add_summary_table(ax, series_by_arch, metric,
                      cuts_per_arch=None, col_labels=None, fig_width: float = 36.0):
    """Add a summary table below ax using dynamic anomaly-equal column buckets."""
    if cuts_per_arch is None:
        cuts_per_arch, col_labels, *_ = _resolve_cuts(series_by_arch)

    def fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "—"

    all_vals_by_row = []
    multi_variant = len({v for (_, v) in series_by_arch}) > 1
    row_labels = []
    for (arch, variant), (x, mean_v, std_v, mat, _, ce_ref, ces) in series_by_arch.items():
        cuts = cuts_per_arch[(arch, variant)]
        vals = bucket_metric(ces, cuts, metric)
        name = ARCH_LABELS.get(arch, arch)
        if multi_variant:
            name = f"{name} ({variant})"
        row_labels.append(name)
        all_vals_by_row.append(vals)

    if not row_labels:
        return

    cmap = _make_metric_cmap(metric)
    finite_vals = [v for row in all_vals_by_row for v in row if not np.isnan(v)]
    vmin = min(finite_vals) if finite_vals else 0.0
    vmax = max(finite_vals) if finite_vals else 1.0
    if vmax == vmin:
        vmax = vmin + 0.01

    def cell_rgba(v):
        if np.isnan(v):
            return "#eeeeee"
        t = np.clip((v - vmin) / (vmax - vmin), 0, 1)
        r, g, b, _ = cmap(t)
        return (r, g, b)

    cell_vals, cell_colors = [], []
    for vals in all_vals_by_row:
        cell_vals.append([fmt(v) for v in vals])
        cell_colors.append([cell_rgba(v) for v in vals])

    col_w_in     = 0.55
    row_w_in     = 1.10
    table_w_frac = min(0.92, (row_w_in + len(col_labels) * col_w_in) / fig_width)

    tbl = ax.table(
        cellText=cell_vals,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="bottom",
        bbox=[0, -0.90, table_w_frac, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(col=list(range(-1, len(col_labels))))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.PAD = 0.18
        if r == 0:
            cell.set_facecolor("#dce6f1")
            cell.set_text_props(fontweight="bold")


def save_txt_summary(out_path: Path, results_by_anomaly: dict,
                     metrics: list[str], args) -> None:
    """Write a text summary of rolling metric averages alongside the PNG."""
    txt_path = out_path.with_suffix(".txt")
    variant_str = "_".join(args.variant) if args.variant else "all"
    col_w = 11

    with open(txt_path, "w") as fh:
        fh.write("Rolling metrics analysis\n")
        fh.write(f"  Session:  {args.session_dir}\n")
        fh.write(f"  Metric:   {', '.join(metrics)}\n")
        fh.write(f"  Variant:  {variant_str}\n")
        fh.write(f"  Window:   {args.window} steps\n")
        fh.write(f"  Smooth σ: {args.smooth} (ref {_SMOOTH_REF_N} steps)\n")
        fh.write("\n")
        fh.write("  Column buckets are anomaly-equal: each spans the time range\n")
        fh.write("  covering the same share of ground-truth anomaly steps.\n")
        fh.write("  '—'  = no GT anomalies in that interval (undefined)\n")
        fh.write("  0.0% = anomalies exist but model detected none (genuine miss)\n")

        for anomaly, series_dict in results_by_anomaly.items():
            for metric, series_by_arch in series_dict.items():
                if not series_by_arch:
                    continue
                cuts_per_arch, col_labels, dynamic, _xpcts = _resolve_cuts(series_by_arch)
                bucket_note = " (anomaly-equal buckets)" if dynamic else " (fixed % intervals)"

                fh.write(f"\n{'─' * 74}\n")
                fh.write(f"  {anomaly}  —  {METRIC_LABELS[metric]}{bucket_note}\n")
                fh.write(f"{'─' * 74}\n")
                header = f"  {'Architecture':<24}" + "".join(
                    f"{c:>{col_w}}" for c in col_labels
                )
                fh.write(header + "\n")
                fh.write("  " + "─" * (len(header) - 2) + "\n")

                multi_variant = len({v for (_, v) in series_by_arch}) > 1
                for (arch, variant), (x, mean_v, std_v, mat, _, ce_ref, ces) in series_by_arch.items():
                    cuts = cuts_per_arch[(arch, variant)]
                    vals = bucket_metric(ces, cuts, metric)
                    name = ARCH_LABELS.get(arch, arch)
                    if multi_variant:
                        name = f"{name} ({variant})"
                    cells = [
                        f"{'—':>{col_w}}" if np.isnan(v)
                        else f"{v:>{col_w}.3f}"
                        for v in vals
                    ]
                    fh.write(f"  {name:<24}" + "".join(cells) + "\n")

    print(f"Saved → {txt_path}")


def _auto_ylim(ax, skip_frac: float = 0.10, pct: float = 99,
               margin: float = 1.20, floor: float = 0.05) -> None:
    """Set y upper limit from plotted data, ignoring the first skip_frac of each line."""
    all_vals = []
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        if len(y) > 20:          # skip short lines like axvlines (2 points each)
            y = y[int(len(y) * skip_frac):]
        y = y[np.isfinite(y) & (y >= 0) & (y <= 1.0)]
        all_vals.extend(y.tolist())
    if not all_vals:
        ax.set_ylim(0, 1.0)
        return
    y_max = float(np.percentile(all_vals, pct)) * margin
    ax.set_ylim(0, float(np.clip(y_max, floor, 1.0)))


def _style_ax(ax, title, metric, window):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Stream progress (%)", fontsize=8)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=8)
    ax.set_xlim(0, 100)
    _auto_ylim(ax)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--metric",  nargs="+",
                        default=["fpr"],
                        choices=["fpr", "f1", "recall", "precision"])
    parser.add_argument("--window",  type=int, default=200)
    parser.add_argument("--smooth",  type=float, default=20,
                        help="Gaussian smoothing sigma in steps (default: 20)")
    parser.add_argument("--fig-width",  type=float, default=36.0,
                        help="Figure width in inches (default: 36)")
    parser.add_argument("--fig-height", type=float, default=6.0,
                        help="Figure height per row in inches (default: 6)")
    parser.add_argument("--lw", type=float, default=0.6,
                        help="Line width (default: 0.6)")
    parser.add_argument("--fill-nan", action="store_true", default=False,
                        help="Replace NaN with 0 instead of leaving gaps")
    parser.add_argument("--anomaly", nargs="+", default=None,
                        help="Filter by anomaly type(s), e.g. Group Contextual Clean")
    parser.add_argument("--arch",    nargs="+", default=None,
                        help="Filter by architecture name(s)")
    parser.add_argument("--variant", nargs="+", default=None,
                        help="Filter by variant name(s), e.g. s48 s12")
    parser.add_argument("--no-shade", action="store_true",
                        help="Suppress the ±std confidence shading around each line")
    parser.add_argument("--split",   action="store_true",
                        help="Produce one PNG per anomaly type instead of a combined grid")
    parser.add_argument("--out",     type=str,  default=None)
    args = parser.parse_args()

    runs = discover_runs(args.session_dir,
                         anomaly_filter=set(args.anomaly) if args.anomaly else None,
                         arch_filter=set(args.arch) if args.arch else None)

    if not runs:
        print("No _cc_summary.npz files found under", args.session_dir)
        return

    if args.variant:
        all_variants = sorted({k[2] for k in runs})
        vf = set(args.variant)
        runs = {k: v for k, v in runs.items() if k[2] in vf}
        if not runs:
            print(f"No runs matched --variant {args.variant}.")
            print(f"Available variants: {all_variants}")
            return

    anomaly_types = sorted({a for (a, _, _) in runs})
    metrics       = args.metric
    arch_order    = [a for a in ARCH_ORDER if
                     (args.arch is None or a in args.arch) and
                     any(arch == a for (_, arch, _) in runs)]
    postfix = "_" + "_".join(args.variant) if args.variant else ""

    all_results: dict = {}  # anomaly -> metric -> series_by_arch

    if args.split:
        # One PNG (+ txt) per anomaly type
        for anomaly in anomaly_types:
            runs_for_anom = {k: v for k, v in runs.items() if k[0] == anomaly}
            n_rows = len(metrics)
            fig, axes = plt.subplots(n_rows, 1,
                                     figsize=(args.fig_width, args.fig_height * n_rows),
                                     squeeze=False)
            fig.suptitle(
                f"{anomaly}  —  rolling metrics  (window = {args.window} steps)",
                fontsize=11, y=1.01)
            all_results[anomaly] = {}
            for ri, metric in enumerate(metrics):
                ax = axes[ri][0]
                series_by_arch = plot_metric(
                    ax, runs_for_anom, metric, args.window, arch_order,
                    smooth=args.smooth, shade=not args.no_shade,
                    lw=args.lw, fill_nan=args.fill_nan)
                all_results[anomaly][metric] = series_by_arch
                _style_ax(ax, METRIC_LABELS[metric], metric, args.window)
                if series_by_arch:
                    cuts_per_arch, col_labels, _, cut_xpcts = _resolve_cuts(series_by_arch)
                    for xp in cut_xpcts:
                        ax.axvline(x=xp, color="#333333", lw=1.4, ls=(0, (4, 3)),
                                   alpha=0.7, zorder=2)
                    ax.legend(fontsize=7, loc="upper left",
                              bbox_to_anchor=(1.02, 1), borderaxespad=0)
                    add_summary_table(ax, series_by_arch, metric,
                                      cuts_per_arch, col_labels,
                                      fig_width=args.fig_width)

            plt.tight_layout()
            # The table bbox extends 0.90 axes-heights below each subplot, so
            # hspace must exceed 0.90 to prevent overlap with the row below.
            # For single-row figures, bottom=0.46 reserves figure-bottom space;
            # for multi-row, large hspace handles inter-row spacing and
            # bbox_inches="tight" handles the bottom overflow.
            if n_rows == 1:
                plt.subplots_adjust(bottom=0.46, hspace=0.6)
            else:
                plt.subplots_adjust(hspace=1.2)

            if args.out:
                base = Path(args.out)
                out = base.parent / (base.stem + f"_{anomaly}" + base.suffix)
            else:
                out = args.session_dir / f"rolling_metrics{postfix}_{anomaly}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved → {out}")
            save_txt_summary(out, {anomaly: all_results[anomaly]}, metrics, args)

    else:
        # Combined grid: metrics × anomaly types
        n_rows = len(metrics)
        n_cols = len(anomaly_types)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(args.fig_width * n_cols, args.fig_height * n_rows),
                                 squeeze=False)
        fig.suptitle(f"Rolling metrics over stream  (window = {args.window} steps)",
                     fontsize=11, y=1.01)

        for ci, anomaly in enumerate(anomaly_types):
            runs_for_anom = {k: v for k, v in runs.items() if k[0] == anomaly}
            all_results[anomaly] = {}
            for ri, metric in enumerate(metrics):
                ax = axes[ri][ci]
                series_by_arch = plot_metric(
                    ax, runs_for_anom, metric, args.window, arch_order,
                    smooth=args.smooth, shade=not args.no_shade,
                    lw=args.lw, fill_nan=args.fill_nan)
                all_results[anomaly][metric] = series_by_arch
                _style_ax(ax, f"{anomaly}  —  {METRIC_LABELS[metric]}", metric, args.window)
                if series_by_arch:
                    cuts_per_arch, col_labels, _, cut_xpcts = _resolve_cuts(series_by_arch)
                    for xp in cut_xpcts:
                        ax.axvline(x=xp, color="#333333", lw=1.4, ls=(0, (4, 3)),
                                   alpha=0.7, zorder=2)
                    ax.legend(fontsize=7, loc="upper left",
                              bbox_to_anchor=(1.02, 1), borderaxespad=0)
                    add_summary_table(ax, series_by_arch, metric,
                                      cuts_per_arch, col_labels,
                                      fig_width=args.fig_width)

        plt.tight_layout()
        if n_rows == 1:
            plt.subplots_adjust(bottom=0.46, hspace=0.6, wspace=0.6)
        else:
            plt.subplots_adjust(hspace=1.2, wspace=0.6)

        if args.out:
            out = Path(args.out)
        else:
            out = args.session_dir / f"rolling_metrics{postfix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        save_txt_summary(out, all_results, metrics, args)


if __name__ == "__main__":
    main()
