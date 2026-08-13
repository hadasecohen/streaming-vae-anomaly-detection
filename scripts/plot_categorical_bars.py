#!/usr/bin/env python3
"""
plot_categorical_bars.py — Grouped vertical bar chart of a metric across a
suite's categorical (non-numeric) variants — reconstruction-loss kind, beta,
threshold, or any other `section`/`variant` pair in performance_summary.csv —
one color per architecture, with ±1-std error bars across seeds.

A port of thesis_code/scripts/generate_analysis_plots.py's
render_categorical_plot for this repo's cross_compare.py output — reads
performance_summary.csv (already mean/std-aggregated across seeds) instead
of raw per-seed rows, and drops the "default config" highlighting machinery
(not needed for a flat multi-way comparison with no baseline row).

--section is required — it selects which suite variable to plot (whatever
value that suite's config_layers used for `compare.section`, e.g.
"recon_loss", "beta", "threshold").

Usage (from project root):
    python scripts/plot_categorical_bars.py runs/regression/case06_reconstruction_loss --section recon_loss
    python scripts/plot_categorical_bars.py runs/regression/case06_reconstruction_loss --section recon_loss --metric f1
    python scripts/plot_categorical_bars.py runs/regression/case06_reconstruction_loss --section recon_loss --anomaly Contextual
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse this repo's own architecture colors/order (cross_compare.py) rather
# than a separate palette, so plots from both scripts look consistent.
ARCH_COLORS = {"MLP": "#4c72b0", "LSTM": "#dd8452",
                "TF_VAE": "#55a868", "MLP_Attn": "#c44e52", "LSTM_Attn": "#8172b2"}
ARCH_DISPLAY_NAMES = {
    "MLP": "MLP-VAE", "LSTM": "LSTM-VAE",
    "TF_VAE": "Transformer-VAE",
    "MLP_Attn": "MLP-VAE-Attn", "LSTM_Attn": "LSTM-VAE-Attn",
}
METRIC_LABELS = {
    "f1": "F1 Score", "auc": "AUC-ROC",
    "precision": "Precision", "recall": "Recall",
}
_ARCH_ORDER = ["MLP", "LSTM", "MLP_Attn", "LSTM_Attn", "TF_VAE"]

DEFAULT_VARIANT_ORDER = ["mse", "wmse", "mae", "huber"]


def _variant_sort_key(variants: list) -> list:
    known = [v for v in DEFAULT_VARIANT_ORDER if v in variants]
    unknown = sorted(v for v in variants if v not in DEFAULT_VARIANT_ORDER)
    return known + unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", help="Directory containing cross_compare/performance_summary.csv")
    parser.add_argument("--metric", default="f1", choices=["f1", "auc", "precision", "recall"])
    parser.add_argument("--section", required=True,
                         help="Value of the 'section' column to filter on, e.g. recon_loss, beta, threshold")
    parser.add_argument("--anomaly", default=None,
                         help="Filter to one anomaly_type (default: all present)")
    parser.add_argument("--output", default=None,
                         help="Output PNG path (default: <session_dir>/cross_compare/bars_<section>_<metric>_<anomaly>.png)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    csv_path = session_dir / "cross_compare" / "performance_summary.csv"
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path} (run cross_compare.py first)")

    df = pd.read_csv(csv_path)
    df = df[df["section"] == args.section]
    if df.empty:
        sys.exit(f"No rows with section == {args.section!r} in {csv_path}")

    anomalies = [args.anomaly] if args.anomaly else sorted(df["anomaly_type"].unique())

    for anomaly in anomalies:
        sub = df[df["anomaly_type"] == anomaly]
        if sub.empty:
            continue

        variants = _variant_sort_key(sorted(sub["variant"].unique()))
        arches = [a for a in _ARCH_ORDER if a in sub["arch"].unique()]
        if not arches:
            arches = sorted(sub["arch"].unique())

        n_arch = len(arches)
        bar_w = 0.65 / n_arch
        x_base = np.arange(len(variants), dtype=float)
        offsets = (np.arange(n_arch) - (n_arch - 1) / 2.0) * bar_w

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        mean_col, std_col = f"{args.metric}_mean", f"{args.metric}_std"

        cv = sub.set_index(["variant", "arch"])
        for ai, arch in enumerate(arches):
            means, errs = [], []
            for v in variants:
                key = (v, arch)
                if key in cv.index:
                    means.append(float(cv.loc[key, mean_col]))
                    errs.append(float(cv.loc[key, std_col]) if not np.isnan(cv.loc[key, std_col]) else 0.0)
                else:
                    means.append(0.0)
                    errs.append(0.0)
            color = ARCH_COLORS.get(arch, "#888")
            ax.bar(
                x_base + offsets[ai], means,
                width=bar_w * 0.88,
                color=color, yerr=errs,
                error_kw=dict(linewidth=0.7, capsize=2.0, ecolor="#444"),
                label=ARCH_DISPLAY_NAMES.get(arch, arch),
                zorder=3,
            )

        metric_label = METRIC_LABELS.get(args.metric, args.metric.upper())
        section_label = args.section.replace("_", " ").title()
        ax.set_title(f"{section_label}  ·  {metric_label}  ·  {anomaly}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels(variants, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        row_max = sub[mean_col].max()
        ax.set_ylim(0, min(1.0, float(row_max) + 0.1) if pd.notna(row_max) else 1.0)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5, linestyle="--", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, framealpha=0.95, edgecolor="#cccccc")

        fig.tight_layout()

        if args.output:
            out_path = Path(args.output)
        else:
            safe_anom = anomaly.lower()
            safe_sec = args.section.lower()
            out_path = session_dir / "cross_compare" / f"bars_{safe_sec}_{args.metric}_{safe_anom}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
