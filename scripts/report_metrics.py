#!/usr/bin/env python3
"""
Report F1 and AUC across all trials suite session directories.

A session dir is any directory whose immediate children include anomaly-type
subdirs (Contextual, Group, Point, Clean) or that already has cross_compare
output.  The script walks up to 2 levels below <root> to find them.

Usage:
    # Read existing cross_compare output from all session dirs under v7/
    python scripts/report_metrics.py runs/trials_suite/v7

    # Run cross_compare first, then report
    python scripts/report_metrics.py runs/trials_suite/v7 --cc

    # Also generate analysis tables and plots
    python scripts/report_metrics.py runs/trials_suite/v7 --cc --table --plots

    # Single session dir
    python scripts/report_metrics.py runs/trials_suite/v7/q1_arch/threshold_sweep --cc

    # Pass extra flags through to cross_compare
    python scripts/report_metrics.py runs/trials_suite/v7 --cc --pca --exclude MLP
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


_ANOMALY_TYPES = {"Contextual", "Group", "Point", "Clean"}


def _is_session_dir(path: Path) -> bool:
    """A session dir directly contains anomaly-type subdirs or cross_compare output."""
    if (path / "cross_compare" / "performance_table.csv").exists():
        return True
    children = {d.name for d in path.iterdir() if d.is_dir()}
    return bool(children & _ANOMALY_TYPES)


def find_session_dirs(root: Path) -> list[Path]:
    """Find all session dirs at depth 1 and 2 below root."""
    sessions: set[Path] = set()
    for d1 in sorted(root.iterdir()):
        if not d1.is_dir() or d1.name.startswith("."):
            continue
        if _is_session_dir(d1):
            sessions.add(d1)
            continue
        for d2 in sorted(d1.iterdir()):
            if not d2.is_dir() or d2.name.startswith("."):
                continue
            if _is_session_dir(d2):
                sessions.add(d2)
    return sorted(sessions)


def _run(cmd: list[str], project_root: Path) -> bool:
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def run_cross_compare(session_dir: Path, project_root: Path,
                      extra_args: list[str]) -> bool:
    cmd = [sys.executable, str(Path("scripts") / "cross_compare.py"), str(session_dir)] + extra_args
    print(f"  [cc] {' '.join(str(a) for a in cmd)}")
    return _run(cmd, project_root)


def run_analysis_table(root: Path, project_root: Path, metrics: list[str]) -> None:
    table_script = project_root / "scripts" / "generate_analysis_table.py"
    if not table_script.exists():
        print(f"  [table] WARNING: {table_script} not found, skipping")
        return
    all_known = {"f1", "auc", "ap", "precision", "recall"}
    metric_arg = "all" if set(metrics) >= all_known else metrics[0]
    print(f"\n[table] generating analysis tables ({metric_arg}) → {root} ...")
    cmd = [sys.executable, str(table_script), str(root),
           "--metric", metric_arg, "--no-filter", "--output-dir", str(root)]
    _run(cmd, project_root)
    if len(metrics) > 1 and metric_arg != "all":
        for m in metrics[1:]:
            cmd[-3] = m
            print(f"\n[table] generating analysis tables ({m}) → {root} ...")
            _run(cmd, project_root)


def run_analysis_plots(root: Path, project_root: Path, metrics: list[str],
                       session_dirs: list[Path], smooth: float = 300) -> None:
    plots_script = project_root / "scripts" / "generate_analysis_plots.py"
    if not plots_script.exists():
        print(f"  [plots] WARNING: {plots_script} not found, skipping")
        return
    all_known = {"f1", "auc", "ap", "precision", "recall"}
    metric_arg = "all" if set(metrics) >= all_known else metrics[0]
    out_dir = root / "plots"
    print(f"\n[plots] generating analysis plots ({metric_arg}) → {out_dir} ...")
    cmd = [sys.executable, str(plots_script), str(root),
           "--metric", metric_arg, "--no-filter", "--output-dir", str(out_dir)]
    _run(cmd, project_root)
    if len(metrics) > 1 and metric_arg != "all":
        for m in metrics[1:]:
            cmd[-3] = m
            print(f"\n[plots] generating analysis plots ({m}) → {out_dir} ...")
            _run(cmd, project_root)

    # Rolling performance plots (one per session dir, one per metric)
    rolling_script = project_root / "scripts" / "plot_fpr_over_time.py"
    if not rolling_script.exists():
        print(f"  [plots] WARNING: {rolling_script} not found, skipping rolling plots")
        return
    _rolling_supported = {"fpr", "f1", "recall", "precision"}
    rolling_metrics = [m for m in metrics if m in _rolling_supported]
    if not rolling_metrics:
        rolling_metrics = ["f1"]
    for session_dir in session_dirs:
        for m in rolling_metrics:
            out = session_dir / f"rolling_{m}.png"
            print(f"\n[plots] rolling {m.upper()} → {out} ...")
            cmd = [sys.executable, str(rolling_script), str(session_dir),
                   "--metric", m, "--smooth", str(smooth), "--out", str(out)]
            _run(cmd, project_root)


def read_performance_table(session_dir: Path) -> "pd.DataFrame | None":
    csv = session_dir / "cross_compare" / "performance_table.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def _fmt(v, width=7) -> str:
    try:
        return f"{float(v):.4f}" if pd.notna(v) else "  n/a "
    except Exception:
        return "  n/a "


def print_session_summary(session_dir: Path, root: Path, df: pd.DataFrame) -> None:
    try:
        rel = session_dir.relative_to(root)
    except ValueError:
        rel = session_dir
    print(f"\n{'─' * 74}")
    print(f"  {rel}")
    print(f"{'─' * 74}")

    groups = df.groupby("anomaly", sort=True) if "anomaly" in df.columns else [("all", df)]
    for anomaly, grp in groups:
        print(f"\n  [{anomaly}]")
        for _, row in grp.sort_values(["arch", "variant", "seed"] if all(
                c in grp for c in ["arch", "variant", "seed"]) else []).iterrows():
            name = row.get("run_name", row.get("arch", "?"))
            seed_s = f"  seed={int(row['seed'])}" if "seed" in row and pd.notna(row.get("seed")) else ""
            f1  = _fmt(row.get("f1"))
            ap  = _fmt(row.get("ap"))
            auc = _fmt(row.get("auc"))
            print(f"    {str(name):<55}  F1={f1}  AP={ap}  AUC={auc}{seed_s}")


def _latest_trials_suite_root() -> Path:
    reg = Path("runs/trials_suite")
    if not reg.exists():
        reg = Path(__file__).resolve().parent.parent / "runs" / "trials_suite"
    versions = sorted(reg.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)
    return versions[-1] if versions else reg


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_root = str(_latest_trials_suite_root())
    parser.add_argument("root", nargs="?", default=_default_root,
                        help=f"Session dir or parent dir containing session dirs "
                             f"(default: {_default_root})")
    parser.add_argument("--cc", action="store_true",
                        help="Run cross_compare.py on each session dir before reading metrics")
    parser.add_argument("--pca", action="store_true",
                        help="Pass --pca to cross_compare (only with --cc)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        metavar="NAME",
                        help="Pass --exclude to cross_compare (only with --cc)")
    parser.add_argument("--table", action="store_true",
                        help="Run generate_analysis_table.py on <root> after all sessions "
                             "(reads existing cross_compare outputs — no need for --cc)")
    parser.add_argument("--table-metrics", nargs="*", default=["f1", "auc", "ap", "precision", "recall"],
                        metavar="METRIC",
                        help="Metrics to generate tables for (default: f1 auc ap precision recall)")
    parser.add_argument("--plots", action="store_true",
                        help="Run generate_analysis_plots.py on <root> after all sessions "
                             "(reads existing cross_compare outputs — no need for --cc)")
    parser.add_argument("--plots-metrics", nargs="*", default=["f1", "auc", "ap", "precision", "recall"],
                        metavar="METRIC",
                        help="Metrics to generate plots for (default: f1 auc ap precision recall)")
    parser.add_argument("--smooth", type=float, default=300,
                        help="Gaussian smoothing sigma for rolling plots (default: 300, 0=off)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List session dirs that would be processed without running anything")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    # Locate project root (the directory containing scripts/cross_compare.py) to run it from there
    project_root = root
    while not (project_root / "scripts" / "cross_compare.py").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    if not (project_root / "scripts" / "cross_compare.py").exists():
        project_root = Path.cwd()

    # Root may itself be a session dir
    if _is_session_dir(root):
        session_dirs = [root]
    else:
        session_dirs = find_session_dirs(root)

    if not session_dirs:
        sys.exit(f"No session directories found under {root}")

    print(f"Found {len(session_dirs)} session dir(s):")
    for d in session_dirs:
        try:
            rel = d.relative_to(root)
        except ValueError:
            rel = d
        status = "" if (d / "cross_compare" / "performance_table.csv").exists() else "  (no cc output yet)"
        print(f"  {rel}{status}")

    if args.dry_run:
        print("\n(dry run — exiting without processing)")
        return

    cc_extra: list[str] = []
    if args.pca:
        cc_extra.append("--pca")
    if args.exclude:
        cc_extra += ["--exclude"] + args.exclude

    failed_cc: list[Path] = []
    results: list[tuple[Path, pd.DataFrame]] = []

    for session_dir in session_dirs:
        if args.cc:
            print(f"\n[{session_dir.name}] running cross_compare ...")
            ok = run_cross_compare(session_dir, project_root, cc_extra)
            if not ok:
                print(f"  WARNING: cross_compare failed for {session_dir.name}")
                failed_cc.append(session_dir)

        df = read_performance_table(session_dir)
        if df is None:
            print(f"\n  (skip: no performance_table.csv in {session_dir.name}/cross_compare/)")
            continue
        results.append((session_dir, df))
        print_session_summary(session_dir, root, df)

    # Combined mean ± std across all sessions, grouped by anomaly
    if len(results) > 1:
        combined = pd.concat([df for _, df in results], ignore_index=True)
        m_cols = [c for c in ["f1", "auc", "ap", "precision", "recall"] if c in combined.columns]
        if m_cols and "anomaly" in combined.columns:
            print(f"\n{'═' * 74}")
            print("  COMBINED — mean ± std across all sessions, by anomaly")
            print(f"{'═' * 74}")
            summary = combined.groupby("anomaly")[m_cols].agg(["mean", "std"]).round(4)
            print(summary.to_string())

    if args.table:
        run_analysis_table(root, project_root, args.table_metrics)

    if args.plots:
        run_analysis_plots(root, project_root, args.plots_metrics, session_dirs, args.smooth)

    if failed_cc:
        print(f"\n{len(failed_cc)} cross_compare run(s) failed:")
        for d in failed_cc:
            print(f"  {d}")
        sys.exit(1)


if __name__ == "__main__":
    main()
