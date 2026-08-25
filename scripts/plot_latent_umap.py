"""
Post-hoc script: UMAP 3D projection of the saved latent space.

UMAP preserves both local and global topology. With large n_neighbors it
reveals the manifold's global ring structure (annual cycle). With small
n_neighbors it shows tighter local clusters.

Usage:
    python scripts/plot_latent_umap.py <run_dir> [options]

    # restrict to a date range (much faster — fits only that slice):
    python scripts/plot_latent_umap.py <run_dir> --from-date 1990-01-01 --to-date 2000-12-31

    # larger n_neighbors → more global annual-ring structure:
    python scripts/plot_latent_umap.py <run_dir> --n-neighbors 200

    # process every trial dir under a session root (e.g. one suite run
    # covering several architectures) — each trial writes to its own dir:
    python scripts/plot_latent_umap.py <session_dir> --recursive

Outputs (written to run_dir, or to each trial dir when --recursive). Each is
saved as both a fully interactive .html (rotate/zoom/hover — needs a
JS-executing viewer, e.g. Colab; GitHub's and VS Code's notebook renderers
block the external Plotly script an .html output depends on) and a static
.png snapshot (renders anywhere, no interactivity):
    latent_umap_month.html / .png   — month palette
    latent_umap_hod.html   / .png   — hour-of-day  (circular HSV)
    latent_umap_doy.html   / .png   — day-of-year  (circular HSV)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed.  pip install umap-learn")
    sys.exit(1)


MONTH_COLORS = [
    "#08306b",  # Jan
    "#6baed6",  # Feb
    "#41ab5d",  # Mar
    "#41ab5d",  # Apr
    "#fee391",  # May
    "#fd8d3c",  # Jun
    "#d73027",  # Jul
    "#d73027",  # Aug
    "#fd8d3c",  # Sep
    "#fee391",  # Oct
    "#6baed6",  # Nov
    "#08306b",  # Dec
]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(run_dir):
    """Load a trial's latent space plus matching timestamps/labels/predictions.

    Only `latent_mu.npy` (written by streaming_loop.py for every trial) and
    `trial_predictions.csv` (for timestamps/labels/predictions) are ever
    produced by this repo's own training code, so those are the only two
    files actually read.
    """
    latent_mu_path = os.path.join(run_dir, "latent_mu.npy")
    if not os.path.exists(latent_mu_path):
        print(f"ERROR: {latent_mu_path} not found.\n"
              "Re-run the trial to generate it (streaming_loop.py saves it automatically).")
        sys.exit(1)

    Z = np.load(latent_mu_path, allow_pickle=False)
    print(f"Loaded latent_mu: {Z.shape}")
    if Z.ndim == 3:
        Z = Z[:, 0, :]
    elif Z.ndim != 2:
        raise ValueError(f"Unexpected latent_mu shape: {Z.shape}")

    N = len(Z)

    csv_path = os.path.join(run_dir, "trial_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)
    with open(csv_path, "rt") as fh:
        df = pd.read_csv(fh, comment=None)
    df.columns = [c.lstrip("# ").strip() for c in df.columns]

    n = min(N, len(df))
    Z          = Z[:n]
    df         = df.iloc[:n].reset_index(drop=True)
    timestamps = pd.DatetimeIndex(pd.to_datetime(df["ts_start"]))

    labels_arr = preds_arr = None
    if "label" in df.columns and "final_pred" in df.columns:
        labels_arr = df["label"].to_numpy(dtype=int)
        preds_arr  = df["final_pred"].to_numpy(dtype=int)

    return Z, timestamps, labels_arr, preds_arr


# ── Filtering ────────────────────────────────────────────────────────────────

def apply_filters(Z, timestamps, labels_arr, preds_arr,
                  from_date=None, to_date=None,
                  focus_years=None, year_window=0):
    """Apply date-range and/or year filters. Returns filtered copies."""
    mask = np.ones(len(Z), dtype=bool)

    if from_date is not None:
        mask &= timestamps >= pd.Timestamp(from_date)
    if to_date is not None:
        mask &= timestamps <= pd.Timestamp(to_date)

    if focus_years:
        allowed = set()
        for yr in focus_years:
            for d in range(-year_window, year_window + 1):
                allowed.add(yr + d)
        mask &= np.isin(timestamps.year, list(allowed))

    if not mask.any():
        print("ERROR: no data matches the requested date filters.")
        sys.exit(1)

    Z          = Z[mask]
    timestamps = timestamps[mask]
    if labels_arr is not None: labels_arr = labels_arr[mask]
    if preds_arr  is not None: preds_arr  = preds_arr[mask]
    print(f"Date filter: {mask.sum()} / {len(mask)} samples kept "
          f"({timestamps[0].date()} → {timestamps[-1].date()})")
    return Z, timestamps, labels_arr, preds_arr


# ── Derived time arrays ──────────────────────────────────────────────────────

def time_arrays(timestamps):
    """Return month, hour-of-day, day-of-year arrays from a DatetimeIndex."""
    months = timestamps.month.to_numpy(dtype=int)
    hod    = timestamps.hour.to_numpy(dtype=int)           # 0–23
    doy    = timestamps.day_of_year.to_numpy(dtype=int)    # 1–365
    return months, hod, doy


# ── Subsampling ──────────────────────────────────────────────────────────────

def subsample(rng, months, labels_arr, preds_arr, max_normals_per_month):
    n  = len(months)
    tn = tp = fp = fn = np.zeros(n, dtype=bool)
    if labels_arr is not None and preds_arr is not None:
        tn = (preds_arr == 0) & (labels_arr == 0)
        tp = (preds_arr == 1) & (labels_arr == 1)
        fp = (preds_arr == 1) & (labels_arr == 0)
        fn = (preds_arr == 0) & (labels_arr == 1)

    anom = tp | fp | fn
    parts = []
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        if not len(idx): continue
        parts.append(idx[anom[idx]])
        norm = idx[~anom[idx]]
        k = min(max_normals_per_month, len(norm))
        parts.append(rng.choice(norm, size=k, replace=False))

    sub_idx = np.sort(np.concatenate(parts))
    print(f"Subsampled to {len(sub_idx)} display points")
    return sub_idx, tn, tp, fp, fn


# ── Plotly trace builders ────────────────────────────────────────────────────

def confusion_traces_3d(Z3, tn, tp, fp, fn, months):
    """Confusion overlay traces. Expects already-subsampled Z3/months/masks."""
    traces = []
    for mask, label, color, sz, op in [
        (tn, "TN – normal",      "steelblue",  3, 0.20),
        (tp, "TP – detected",    "darkorange", 7, 0.90),
        (fn, "FN – missed",      "magenta",    7, 0.90),
        (fp, "FP – false alarm", "crimson",    5, 0.70),
    ]:
        if not mask.any(): continue
        mo_labels = [MONTH_NAMES[m - 1] for m in months[mask]]
        traces.append(go.Scatter3d(
            x=Z3[mask, 0], y=Z3[mask, 1], z=Z3[mask, 2],
            mode="markers",
            marker=dict(size=sz, color=color, opacity=op,
                        line=dict(width=0.5, color="black")),
            name=f"{label} (n={mask.sum()})",
            text=[f"{label} | month={m}" for m in mo_labels],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            legendgroup="confusion",
        ))
    return traces


def path_trace_3d(Z3_time_sorted, path_n=2000):
    """Thin lines connecting ~path_n points in temporal order."""
    step = max(1, len(Z3_time_sorted) // path_n)
    pts  = Z3_time_sorted[::step]
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(width=1, color="rgba(80,80,80,0.18)"),
        name="temporal path",
        hoverinfo="skip",
        showlegend=True,
    )


# ── HTML/PNG writers ─────────────────────────────────────────────────────────

def _save_fig(fig, out_path_html):
    """Save an interactive .html (needs a JS-executing viewer) plus a static
    .png snapshot at the same base path (renders in GitHub/VS Code too, no
    interactivity). PNG export needs kaleido==0.2.1 specifically — kaleido>=1
    replaced the classic renderer with one that drives its own downloaded
    headless Chrome, which is prone to launch failures in containerized
    environments like Colab (missing shared libs, sandbox/shm quirks) that
    aren't worth chasing here; 0.2.1 has none of that and just works.
    """
    fig.write_html(out_path_html, include_plotlyjs="cdn")
    print(f"Saved → {out_path_html}")

    out_path_png = os.path.splitext(out_path_html)[0] + ".png"
    try:
        fig.write_image(out_path_png, width=1000, height=700, scale=2)
        print(f"Saved → {out_path_png}")
    except Exception as e:
        print(f"WARNING: couldn't save {out_path_png} ({e}). "
              "Static PNG export needs kaleido==0.2.1: pip install kaleido==0.2.1")


def write_month_html(Z3s, mos, tn, tp, fp, fn,
                     title, axis_labels, out_path,
                     draw_path=False, time_order=None):
    fig = go.Figure()
    if draw_path and time_order is not None:
        fig.add_trace(path_trace_3d(Z3s[time_order]))
    for mo in range(1, 13):
        mask = mos == mo
        if not mask.any(): continue
        fig.add_trace(go.Scatter3d(
            x=Z3s[mask, 0], y=Z3s[mask, 1], z=Z3s[mask, 2],
            mode="markers",
            marker=dict(size=3, color=MONTH_COLORS[mo - 1], opacity=0.65),
            name=MONTH_NAMES[mo - 1],
            hovertemplate=f"month={MONTH_NAMES[mo-1]}<extra></extra>",
            legendgroup="months",
        ))
    fig.add_traces(confusion_traces_3d(Z3s, tn, tp, fp, fn, mos))
    a1, a2, a3 = axis_labels
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=a1, yaxis_title=a2, zaxis_title=a3),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
    )
    _save_fig(fig, out_path)


def write_cyclic_html(Z3s, color_vals, cmin, cmax, colorscale,
                      colorbar_title, hover_extra,
                      title, axis_labels, out_path,
                      draw_path=False, time_order=None):
    """Single-trace HTML with a circular colorscale (HOD or DOY)."""
    # Sort by the color value so plotly renders the colorscale continuously
    sort_idx = np.argsort(color_vals)
    Z3s_s    = Z3s[sort_idx]
    cv_s     = color_vals[sort_idx]
    hover_s  = [hover_extra[i] for i in sort_idx]

    fig = go.Figure()
    if draw_path and time_order is not None:
        # path uses temporal order, not color-sorted order
        fig.add_trace(path_trace_3d(Z3s[time_order]))
    fig.add_trace(go.Scatter3d(
        x=Z3s_s[:, 0], y=Z3s_s[:, 1], z=Z3s_s[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=cv_s,
            cmin=cmin, cmax=cmax,
            colorscale=colorscale,
            opacity=0.70,
            colorbar=dict(title=colorbar_title),
            showscale=True,
        ),
        text=hover_s,
        hovertemplate="%{text}<extra></extra>",
        name=colorbar_title,
    ))
    a1, a2, a3 = axis_labels
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=a1, yaxis_title=a2, zaxis_title=a3),
    )
    _save_fig(fig, out_path)


# ── Recursive helpers ─────────────────────────────────────────────────────────

def find_trial_dirs(root: str) -> list[str]:
    """Return all directories under root that contain latent_mu.npy."""
    root_p = Path(root)
    dirs = sorted({p.parent for p in root_p.rglob("latent_mu.npy")})
    print(f"Found {len(dirs)} trial director{'y' if len(dirs)==1 else 'ies'} under {root}")
    return [str(d) for d in dirs]


def run_recursive(root: str, main_fn, kwargs: dict) -> None:
    """Call main_fn(run_dir=d, **kwargs) for every trial dir under root."""
    dirs = find_trial_dirs(root)
    if not dirs:
        print("No trial directories with latent_mu.npy found.")
        return

    ok = failed = 0
    for i, d in enumerate(dirs, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(dirs)}] {d}")
        try:
            main_fn(run_dir=d, **kwargs)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'═'*60}")
    print(f"Done: {ok} succeeded, {failed} failed  (total {len(dirs)})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(run_dir, out_dir=None, n_neighbors=150, min_dist=0.1,
         max_normals_per_month=5_000, draw_path=False,
         from_date=None, to_date=None, focus_years=None, year_window=0):
    out_dir = out_dir or run_dir

    Z, timestamps, labels_arr, preds_arr = load_data(run_dir)
    Z, timestamps, labels_arr, preds_arr = apply_filters(
        Z, timestamps, labels_arr, preds_arr,
        from_date=from_date, to_date=to_date,
        focus_years=focus_years, year_window=year_window,
    )
    months, hod, doy = time_arrays(timestamps)
    t_idx = np.arange(len(Z), dtype=float)

    # ── UMAP on the (filtered) full data ─────────────────────────────────
    print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"n={len(Z)}, d={Z.shape[1]}) …")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42, n_jobs=-1)
    Z3 = reducer.fit_transform(Z)
    print("UMAP done.")

    corrs = [float(np.corrcoef(t_idx, Z3[:, i])[0, 1]) for i in range(3)]
    print(f"Corr with linear time:  "
          f"U1={corrs[0]:+.3f}  U2={corrs[1]:+.3f}  U3={corrs[2]:+.3f}")

    # ── Subsample for display ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sub_idx, tn, tp, fp, fn = subsample(
        rng, months, labels_arr, preds_arr, max_normals_per_month)

    Z3s  = Z3[sub_idx]
    mos  = months[sub_idx]
    hods = hod[sub_idx]
    doys = doy[sub_idx]
    tidx_sub = t_idx[sub_idx]

    time_order = np.argsort(tidx_sub)
    ax_labels  = ("UMAP-1", "UMAP-2", "UMAP-3")
    subtitle   = (f"n_neighbors={n_neighbors}  min_dist={min_dist} | "
                  f"U1={corrs[0]:+.3f}  U2={corrs[1]:+.3f}  U3={corrs[2]:+.3f}")

    # ── HTML: month coloring ──────────────────────────────────────────────
    write_month_html(
        Z3s, mos, tn[sub_idx], tp[sub_idx], fp[sub_idx], fn[sub_idx],
        title=f"Latent – UMAP 3D · month<br><sup>{subtitle}</sup>",
        axis_labels=ax_labels,
        out_path=os.path.join(out_dir, "latent_umap_month.html"),
        draw_path=draw_path, time_order=time_order,
    )

    # ── HTML: hour-of-day coloring ────────────────────────────────────────
    write_cyclic_html(
        Z3s, hods, cmin=0, cmax=23, colorscale="HSV",
        colorbar_title="Hour of day",
        hover_extra=[f"hour={h}  month={MONTH_NAMES[m-1]}"
                     for h, m in zip(hods, mos)],
        title=f"Latent – UMAP 3D · hour of day<br><sup>{subtitle}</sup>",
        axis_labels=ax_labels,
        out_path=os.path.join(out_dir, "latent_umap_hod.html"),
        draw_path=draw_path, time_order=time_order,
    )

    # ── HTML: day-of-year coloring ────────────────────────────────────────
    write_cyclic_html(
        Z3s, doys, cmin=1, cmax=365, colorscale="HSV",
        colorbar_title="Day of year",
        hover_extra=[f"doy={d}  month={MONTH_NAMES[m-1]}"
                     for d, m in zip(doys, mos)],
        title=f"Latent – UMAP 3D · day of year<br><sup>{subtitle}</sup>",
        axis_labels=ax_labels,
        out_path=os.path.join(out_dir, "latent_umap_doy.html"),
        draw_path=draw_path, time_order=time_order,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out-dir", default=None,
                   help="Directory to write HTML files (default: same as run_dir)")
    p.add_argument("--n-neighbors", type=int, default=150,
                   help="UMAP neighborhood size. Larger = more global structure, "
                        "reveals annual ring. Smaller = tighter local clusters. (default 150)")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist. Smaller = tighter clusters. (default 0.1)")
    p.add_argument("--max-normals", type=int, default=5_000,
                   help="Max normal points per month for display (default 5000)")
    p.add_argument("--from-date", type=str, default=None,
                   help="Start of date range e.g. 1990-01-01  (applied before fitting)")
    p.add_argument("--to-date", type=str, default=None,
                   help="End of date range e.g. 2000-12-31  (applied before fitting)")
    p.add_argument("--focus-years", type=int, nargs="+", default=None)
    p.add_argument("--year-window", type=int, default=0)
    p.add_argument("--draw-path", action="store_true",
                   help="Overlay thin lines connecting points in temporal order")
    p.add_argument("--recursive", action="store_true",
                   help="Process all trial directories under run_dir that contain "
                        "latent_mu.npy. --out-dir is ignored in recursive mode.")
    args = p.parse_args()

    kwargs = dict(
        out_dir=args.out_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        max_normals_per_month=args.max_normals,
        draw_path=args.draw_path,
        from_date=args.from_date, to_date=args.to_date,
        focus_years=args.focus_years, year_window=args.year_window,
    )
    if args.recursive:
        kwargs.pop("out_dir")   # each trial writes to its own dir
        run_recursive(args.run_dir, main, kwargs)
    else:
        main(args.run_dir, **kwargs)
