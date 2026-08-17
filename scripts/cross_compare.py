#!/usr/bin/env python3
"""
Post-regression cross-comparison report.

Scans a session directory, extracts metrics from all completed trial runs,
and generates cross-comparison plots grouped by anomaly type.

Usage:
     <session_dir>
    python tests/regression/cross_compare.pypython tests/regression/cross_compare.py <session_dir> --pca
    python tests/regression/cross_compare.py runs/regression/era5_golden/  # uses latest session

Outputs in <session_dir>/cross_compare/:
    performance_table.csv - one row per run: F1/AUC/Precision/Recall at your configured
                            threshold, TP/FP/TN/FN counts, and the best achievable 
                            F1+threshold found by post-hoc search.
    performance_bars.png  - F1 / AUC / Precision / Recall per anomaly group
    confusion_grid_{anomaly}.png - grid of TP/FN/TN/FP bar charts, rows = cyclic/non-cyclic, 
                            cols = arch x mode. Shows, at a glance, where false negatives 
                            or false positives are concentrated.
    metric_distributions_{anomaly}.png - paired box plots (green=normal, red=anomaly) for
                            each metric (kl, recon, elbo, etc.). Box separation shows how
                            discriminative each metric is for this anomaly type.
    score_histograms_{anomaly}.png - histogram of final confidence scores split by label=0
                           (normal) and label=1 (anomaly), with the threshold line. 
                           Good for seeing how well the final score separates the two classes.
    latent_pca_{anomaly}.png  (--pca only)

Run-name convention (extensible):
    ERA5_{ARCH}_{ANOMALY}_{cyclic_variant}[_window]
    ARCH  : MLP | LSTM | MLP_Attn | LSTM_Attn  (any new arch works automatically)
    ANOMALY: Clean | Point | Group | Contextual


Precision
  Useful when false alarms are expensive or annoying.
  Examples: fraud investigations, medical diagnoses, security alerts

  - When the model says something is anomalous, how often is it actually correct?
  - True Positives vs False Positives

  High precision: few false alarms, alerts are trustworthy
  Low precision:  model cries wolf a lot

Recall
  Useful when missing anomalies is costly.
  Examples: cancer screening, intrusion detection, fault detection in machines

  - Out of all the real anomalies that existed, how many did the model actually catch?
  - true Positives vs False Negatives

  High recall: Very few anomalies are missed
  Low recall:  more anomalies are undetected

Precision vs Recall Tradeoff - They fight each other.
 

F1 Score

F1 combines precision and recall into one number.
Useful when you want a single balanced score.

Is the model good at both catching anomalies AND avoiding false alarms?


imbalanced classification

anomaly detection / fraud/spam detection

A model only gets a high F1 if precision is high AND recall is high


"""

import os
import re
import sys
import ast
import yaml
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# ─── constants ────────────────────────────────────────────────────────────────

KNOWN_ANOMALY_TYPES = ["Contextual", "Group", "Point", "Clean"]  # longest first

# Pool-config suffixes — detected before the cyclic fallback
_POOL_CONFIGS = {
    "no_pool_no_norm": dict(pool="no_pool", norm="no_norm"),
    "no_pool_norm":    dict(pool="no_pool", norm="norm"),
    "pool_no_norm":    dict(pool="pool",    norm="no_norm"),
    "pool_norm":       dict(pool="pool",    norm="norm"),
}
# Anomaly types that use point streaming (used to infer mode when suffix is absent)
_POINT_ANOMALIES = {"Point"}

CONFUSION_COLORS  = {"TP": "#2ecc71", "FN": "#e74c3c", "TN": "#27ae60", "FP": "#c0392b"}
ARCH_COLORS       = {"MLP": "#4c72b0", "LSTM": "#dd8452",
                     "TF_VAE": "#55a868", "MLP_Attn": "#c44e52", "LSTM_Attn": "#8172b2"}
METRIC_PALETTE    = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
]


# ─── run name parsing ─────────────────────────────────────────────────────────

def _base_arch(arch: str) -> str:
    """Strip version suffix for colour lookup: 'TF_VAE_v2' → 'TF_VAE', 'LSTM_v2' → 'LSTM'."""
    return re.sub(r"_v\d+$", "", arch)


# Q1-convention arch prefixes, longest first (so TF_VAE matches before a hypothetical TF)
_Q1_ARCH_PREFIXES = ["TF_VAE", "MLP_Attn", "LSTM_Attn", "LSTM", "MLP"]


def parse_run_name_from_meta(name: str, compare_meta: dict) -> dict:
    """
    Build a run descriptor from explicit `compare:` metadata embedded in
    _suite_config.yaml.  Fields:
      anomaly  — required: Group | Contextual | Point | Clean
      arch     — required: MLP | LSTM | TF_VAE | …
      variant  — optional display label (default: everything after arch_anomaly)
      section  — optional sweep axis label (e.g. "anneal", "beta", "free_bits");
                 prepended to variant so labels read "anneal=0" instead of bare "0"
      order    — optional int sort key within the group (default: 0)
    """
    anomaly = compare_meta.get("anomaly", "Unknown")
    arch    = compare_meta.get("arch", "Unknown")
    variant = compare_meta.get("variant", name)
    section = compare_meta.get("section", "")
    order   = int(compare_meta.get("order", 0))

    variant_label = f"{section}={variant}" if section else variant
    short_label   = f"{arch}\n{variant_label}"
    col_key       = (arch, "window")
    row_key       = variant_label

    return dict(arch=arch, base_arch=_base_arch(arch), anomaly=anomaly,
                cyclic=False, mode="window", short_label=short_label,
                col_key=col_key, row_key=row_key, section=section,
                _compare_order=order)


def parse_run_name_q1(name: str) -> dict | None:
    """
    Parse Q1-convention names: {ARCH}_{ANOMALY}_{variant...}
    where ARCH ∈ {TF_VAE, MLP_Attn, LSTM_Attn, LSTM, MLP}
    and   ANOMALY ∈ KNOWN_ANOMALY_TYPES.

    Examples:
      MLP_Group_no_pool_norm_T          → arch=MLP, anomaly=Group, variant=no_pool_norm_T
      TF_VAE_Contextual_prior_var_1_5   → arch=TF_VAE, anomaly=Contextual, variant=prior_var_1_5
      LSTM_Group_pool_max_abs           → arch=LSTM, anomaly=Group, variant=pool_max_abs
    """
    arch = None
    rest = name
    for prefix in _Q1_ARCH_PREFIXES:
        if name == prefix or name.startswith(prefix + "_"):
            arch = prefix
            rest = name[len(prefix):].lstrip("_")
            break
    if arch is None:
        return None

    anomaly = None
    variant = None
    for a in KNOWN_ANOMALY_TYPES:
        if rest == a:
            anomaly, variant = a, "plain"
            break
        if rest.startswith(a + "_"):
            anomaly = a
            variant = rest[len(a):].lstrip("_") or "plain"
            break
    if anomaly is None:
        return None

    short_label = f"{arch}\n{variant}" if variant != "plain" else arch
    col_key     = (arch, "window")
    row_key     = variant

    return dict(arch=arch, base_arch=_base_arch(arch), anomaly=anomaly,
                cyclic=False, mode="window", short_label=short_label,
                col_key=col_key, row_key=row_key,
                _compare_order=0)


def parse_run_name(name: str) -> dict | None:
    """
    Supports four naming conventions, tried in order:

    1. Legacy:    ERA5_{ARCH}_{ANOMALY}_{variant}[_window|_point]
                  e.g. ERA5_LSTM_Contextual_cyclic_ms_window

    2. Pool-only: ERA5_{ARCH}_{ANOMALY}_{pool_config}
                  e.g. ERA5_TF_VAE_Contextual_no_pool_norm

    3. Combined:  ERA5_{ARCH}[_v{N}]_{ANOMALY}_{pool_config}[_{feat_eng}]
                  e.g. ERA5_TF_VAE_v2_Contextual_no_pool_norm_cyclic_ms

    4. Q1:        {ARCH}_{ANOMALY}_{variant}
                  e.g. MLP_Group_no_pool_norm_T
                  (no ERA5_ prefix; used by Q1 experiment suites)
    """
    if not name.startswith("ERA5_"):
        return parse_run_name_q1(name)
    rest = name[len("ERA5_"):]

    # Locate anomaly type (longest match wins via KNOWN_ANOMALY_TYPES order)
    anomaly = None
    for a in KNOWN_ANOMALY_TYPES:
        if f"_{a}_" in rest or rest.endswith(f"_{a}"):
            anomaly = a
            break
    if anomaly is None:
        return None

    split_tok = f"_{anomaly}"
    idx    = rest.index(split_tok)
    arch   = rest[:idx]                    # e.g. 'LSTM', 'TF_VAE_v2', 'MLP_v1'
    suffix = rest[idx + len(split_tok):]   # e.g. '_no_pool_norm_cyclic_ms'

    suffix_stripped = suffix.lstrip("_")

    # ── strip trailing _window / _point (present in some combined names) ─────
    if suffix_stripped.endswith("_window"):
        forced_mode = "window"
        suffix_stripped = suffix_stripped[: -len("_window")]
    elif suffix_stripped.endswith("_point"):
        forced_mode = "point"
        suffix_stripped = suffix_stripped[: -len("_point")]
    else:
        forced_mode = None   # infer below

    # ── pool-config runs: exact or prefix match ───────────────────────────────
    for cfg_key, cfg in _POOL_CONFIGS.items():
        if suffix_stripped == cfg_key:
            feat_eng = "plain"
        elif suffix_stripped.startswith(cfg_key + "_"):
            feat_eng = suffix_stripped[len(cfg_key) + 1:]
        else:
            continue

        mode     = forced_mode or ("point" if anomaly in _POINT_ANOMALIES else "window")
        pool_key = cfg["pool"]   # "pool" | "no_pool"
        norm_key = cfg["norm"]   # "norm" | "no_norm"
        variant  = f"{cfg_key}" if feat_eng == "plain" else f"{cfg_key}_{feat_eng}"

        short_label = f"{arch}\n{pool_key}/{norm_key}" + (f"\n{feat_eng}" if feat_eng != "plain" else "")
        col_key     = (arch, mode)
        row_key     = variant

        return dict(arch=arch, base_arch=_base_arch(arch), anomaly=anomaly,
                    cyclic=False, mode=mode, short_label=short_label,
                    col_key=col_key, row_key=row_key, pool_cfg=cfg_key, feat_eng=feat_eng)

    # ── legacy: variant[_window|_point][_cyc] ────────────────────────────────
    if suffix.endswith("_window_cyc"):
        mode        = "window"
        cyclic_part = suffix[: -len("_window_cyc")]
        cyclic      = True
    elif suffix.endswith("_point_cyc"):
        mode        = "point"
        cyclic_part = suffix[: -len("_point_cyc")]
        cyclic      = True
    elif suffix.endswith("_window"):
        mode        = "window"
        cyclic_part = suffix[: -len("_window")]
        cyclic      = False
    elif suffix.endswith("_point"):
        mode        = "point"
        cyclic_part = suffix[: -len("_point")]
        cyclic      = False
    else:
        mode        = forced_mode or ("point" if anomaly in _POINT_ANOMALIES else "window")
        cyclic_part = suffix
        cyclic      = False

    variant = cyclic_part.lstrip("_") or "plain"
    # also catch bare "cyclic" or any variant explicitly named cyclic
    if variant == "cyclic" or variant.endswith("_cyc"):
        cyclic = True

    short_label = f"{arch}\n{variant}"
    col_key     = (arch, mode)
    row_key     = variant

    return dict(arch=arch, base_arch=_base_arch(arch), anomaly=anomaly,
                cyclic=cyclic, mode=mode, short_label=short_label,
                col_key=col_key, row_key=row_key)


# ─── run discovery ────────────────────────────────────────────────────────────

def resolve_session(path: Path) -> Path:
    """Return path itself if it looks like a session; otherwise pick the latest session_* child."""
    # Immediate children named .*_\d+ (e.g. seed_42) → already a session root
    if any(re.match(r".+_\d+$", d.name) for d in path.iterdir() if d.is_dir()):
        return path
    # Any _suite_config.yaml or trial_predictions.csv/_cc_summary.npz anywhere → session root
    # (handles hierarchical layout where seed_N dirs are deeper than one level)
    if any(True for _ in path.rglob("_suite_config.yaml")):
        return path
    if any(True for _ in path.rglob("trial_predictions.csv")):
        return path
    if any(True for _ in path.rglob("_cc_summary.npz")):
        return path
    sessions = sorted(path.glob("session_*"))
    if not sessions:
        sys.exit(f"No session directories found under {path}")
    return sessions[-1]


def discover_runs(session_dir: Path, recursive: bool = False) -> list[dict]:
    """
    Find all completed runs under session_dir by locating _suite_config.yaml
    files recursively.  Falls back to trial_predictions.csv for legacy runs.
    This handles both flat and hierarchical layouts.

    The `recursive` parameter is accepted for backward compatibility but is now
    a no-op: rglob always scans the full subtree.
    """
    runs = []
    seen: set[Path] = set()

    # Prefer _suite_config.yaml — survives deletion of trial_predictions.csv.
    # Fall back to trial_predictions.csv or _cc_summary.npz for older runs.
    markers = list(session_dir.rglob("_suite_config.yaml"))
    if not markers:
        raw = (list(session_dir.rglob("trial_predictions.csv")) +
               list(session_dir.rglob("_cc_summary.npz")))
        seen_dirs: set[Path] = set()
        markers = []
        for m in sorted(raw):
            if m.parent not in seen_dirs:
                seen_dirs.add(m.parent)
                markers.append(m)

    for marker in sorted(markers):
        d = marker.parent
        # Skip anything inside a cross_compare output directory
        if any(part == "cross_compare" for part in d.relative_to(session_dir).parts):
            continue
        if d in seen:
            continue
        seen.add(d)

        # Load the suite config written by run_regression for this run.
        # Primary location: same directory as trial_predictions.csv.
        # Fallback: sibling directory without the _N suffix — handles the legacy
        # layout where resolve_unique_dir bumped seed_42/ → seed_42_1/ for outputs
        # while _suite_config.yaml remained in seed_42/.
        suite_cfg_path = d / "_suite_config.yaml"
        if not suite_cfg_path.exists():
            sib_m = re.match(r"^(.+?)_(\d+)$", d.name)
            if sib_m:
                suite_cfg_path = d.parent / sib_m.group(1) / "_suite_config.yaml"
        suite_cfg      = {}
        compare_meta   = None
        if suite_cfg_path.exists():
            try:
                with open(suite_cfg_path) as f:
                    suite_cfg = yaml.safe_load(f) or {}
                compare_meta = suite_cfg.get("_compare_meta")
            except Exception:
                pass

        # Derive a canonical run_name for display from compare_meta when available,
        # otherwise fall back to the directory name (legacy flat layout compatibility)
        if compare_meta:
            arch    = compare_meta.get("arch", "")
            anomaly = compare_meta.get("anomaly", "")
            variant = compare_meta.get("variant", "")
            run_name = f"{arch}_{anomaly}_{variant}" if arch and anomaly else d.name
        else:
            # When the trial dir is a seed directory (seed_42, seed_42_1, etc.),
            # the meaningful run name is the parent directory (the actual run name).
            if re.match(r"^seed_", d.name):
                run_name = d.parent.name
            else:
                run_name = d.name
            # Strip trailing _N suffix from legacy directory names so the name
            # is usable as a config label (e.g. "MLP_Group_no_pool_no_norm_1" → "…_no_norm")
            m = re.match(r"^(.+?)_(\d+)$", run_name)
            if m:
                run_name = m.group(1)

        # Infer trial_n from the seed directory name (seed_42 → 42) or default to 0
        trial_n = 0
        seed_m  = re.match(r"^seed_(\d+)$", d.name)
        if seed_m:
            trial_n = int(seed_m.group(1))
        else:
            # Legacy: directory ends in _N
            leg_m = re.match(r"^.+?_(\d+)$", d.name)
            if leg_m:
                trial_n = int(leg_m.group(1))

        if compare_meta is not None:
            parsed = parse_run_name_from_meta(run_name, compare_meta)
        else:
            parsed = parse_run_name(run_name)

        if parsed is None:
            continue

        # Attach the full suite config for later extraction of design columns
        runs.append({
            **parsed,
            "run_name":   run_name,
            "trial_n":    trial_n,
            "dir":        d,
            "_suite_cfg": suite_cfg,
        })
    return runs


# ─── metric extraction ────────────────────────────────────────────────────────

# ── METRICS table (configured-threshold, live performance) ───────────────────
# The METRICS block is written at end-of-run using the live anom_if_score_above
# threshold. These are the authoritative numbers that match the run's own plots.
_METRICS_F1_RE     = re.compile(r"^f1\s*\|\s*([\d.]+)",        re.MULTILINE)
_METRICS_PREC_RE   = re.compile(r"^precision\s*\|\s*([\d.]+)", re.MULTILINE)
_METRICS_RECALL_RE = re.compile(r"^recall\s*\|\s*([\d.]+)",    re.MULTILINE)
_METRICS_ACC_RE    = re.compile(r"^accuracy\s*\|\s*([\d.]+)",  re.MULTILINE)
_METRICS_AUC_RE    = re.compile(r"^conf_roc\s*\|\s*([\d.]+)",  re.MULTILINE)
_METRICS_AP_RE     = re.compile(r"^conf_pr\s*\|\s*([\d.]+)",   re.MULTILINE)
_METRICS_COUNTS_RE = re.compile(r"Counts:\s*n=\d+\s+n_finite=\d+\s+pos=(\d+)\s+neg=(\d+)")

# ── THRESHOLD TUNING blocks (post-hoc optimal — reference only) ───────────────
# These are only used to store the optimal threshold values for reference;
# they do NOT override f1/precision/recall/tp/fp/tn/fn.
_TUNING_RE = re.compile(
    r"THRESHOLD TUNING \(optimize F1\)[^\n]*\n\s+thr=([\d.]+)\n"
    r"\s+f1=[\d.]+(?:\s+f2=[\d.]+)?\s+precision=[\d.]+\s+recall=[\d.]+"
    r"\s+accuracy=[\d.]+\s+tp=\d+\s+fp=\d+\s+tn=\d+\s+fn=\d+"
)
_TUNING_F2_RE = re.compile(
    r"THRESHOLD TUNING \(optimize F2\)[^\n]*\n\s+thr=([\d.]+)\n"
    r"\s+f1=[\d.]+\s+f2=([\d.]+)\s+precision=[\d.]+\s+recall=[\d.]+"
    r"\s+accuracy=[\d.]+\s+tp=\d+\s+fp=\d+\s+tn=\d+\s+fn=\d+"
)
_TUNING_YOUDEN_RE = re.compile(
    r"THRESHOLD TUNING \(Youden J\)[^\n]*\n\s+thr=([\d.]+)\n"
)
_SEARCH_RE = re.compile(
    r"THRESHOLD SEARCH\n\s+thr=([\d.]+)\n"
    r"\s+f1=([\d.]+)\s+precision=([\d.]+)\s+recall=([\d.]+)"
)
# matches new-style key (anom_if_score_above) and old-style (ensemble_threshold)
_ENSEMBLE_THR_RE  = re.compile(r"'anom_if_score_above':\s*([\d.]+)")
_ENSEMBLE_THR_OLD = re.compile(r"'ensemble_threshold':\s*([\d.]+)")
_UPDATE_IF_RE     = re.compile(r"'update_if_score_below':\s*([\d.]+)")
_ELBO_BETA_RE     = re.compile(r"'elbo_beta':\s*([\d.eE+\-]+)")
_KL_NORM_RE       = re.compile(r"'do_kl_normalize':\s*(True|False)")
_N_PARAMS_RE      = re.compile(r"Number of Parameters:\s*(\d+)")

def load_log_metrics(run_dir: Path) -> dict:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return {}
    log = log_path.read_text(errors="replace")
    result = {}

    # ── Primary: METRICS block at the configured threshold ────────────────────
    # Parse the `final` column (always the first data column) of the METRICS table.
    mf1   = _METRICS_F1_RE.search(log)
    mprec = _METRICS_PREC_RE.search(log)
    mrec  = _METRICS_RECALL_RE.search(log)
    macc  = _METRICS_ACC_RE.search(log)
    mauc  = _METRICS_AUC_RE.search(log)
    map_  = _METRICS_AP_RE.search(log)
    mcnt  = _METRICS_COUNTS_RE.search(log)
    if mf1 and mprec and mrec and mcnt:
        f1    = float(mf1[1])
        prec  = float(mprec[1])
        rec   = float(mrec[1])
        n_pos = int(mcnt[1])
        n_neg = int(mcnt[2])
        tp = round(rec * n_pos)
        fn = n_pos - tp
        fp = round(tp * (1 - prec) / prec) if prec > 0 else n_neg
        tn = n_neg - fp
        result.update(f1=f1, precision=prec, recall=rec, tp=tp, fp=fp, tn=tn, fn=fn)
        if macc:
            result["accuracy"] = float(macc[1])
    if mauc:
        result["auc"] = float(mauc[1])
    if map_:
        result["ap"] = float(map_[1])

    # ── Reference: post-hoc optimal thresholds (do not overwrite live metrics) ─
    m = _TUNING_RE.search(log)
    if m:
        result["cfg_thr"] = float(m[1])
    m = _TUNING_F2_RE.search(log)
    if m:
        result.update(cfg_thr_f2=float(m[1]), f2=float(m[2]))
    m = _TUNING_YOUDEN_RE.search(log)
    if m:
        result["cfg_thr_youden"] = float(m[1])
    m = _SEARCH_RE.search(log)
    if m:
        result.update(best_thr=float(m[1]), best_f1=float(m[2]),
                      best_precision=float(m[3]), best_recall=float(m[4]))

    m = _ENSEMBLE_THR_RE.search(log) or _ENSEMBLE_THR_OLD.search(log)
    if m:
        result["ensemble_thr"] = float(m[1])
    m = _UPDATE_IF_RE.search(log)
    if m:
        result["update_if_thr"] = float(m[1])
    m = _ELBO_BETA_RE.search(log)
    if m:
        result["elbo_beta"] = float(m[1])
    m = _KL_NORM_RE.search(log)
    if m:
        result["do_kl_normalize"] = m[1] == "True"
    m = _N_PARAMS_RE.search(log)
    if m:
        result["n_params"] = int(m[1])
    return result


def load_perf_summary(run_dir: Path) -> dict:
    """Read _perf_summary.json written by PerfTracker.  Returns {} if absent."""
    p = run_dir / "_perf_summary.json"
    if not p.exists():
        return {}
    try:
        import json
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


_CSV_COLS = ["ts_start","ts_end","step","confusion","label","final_pred","trained","final_conf","metrics"]

_CC_CONF_DEC: dict[int, str] = {0: "TP", 1: "FP", 2: "TN", 3: "FN"}


def load_predictions_from_summary(run_dir: Path) -> "tuple[pd.DataFrame, pd.DataFrame, list[str]] | None":
    """
    Fall-back loader: reads _cc_summary.npz written by CCSummarySink during the run.
    Returns the same (df_light, df_metrics, metric_names) tuple as load_predictions(),
    or None if the file is absent or unreadable.
    """
    p = run_dir / "_cc_summary.npz"
    if not p.exists():
        return None
    try:
        data = np.load(p, allow_pickle=False)
    except Exception:
        return None

    labels       = data["labels"].astype(int)
    final_conf   = data["final_conf"].astype(float)
    confusion_enc = data["confusion_enc"].astype(int)
    confusion    = np.array([_CC_CONF_DEC.get(int(c), "") for c in confusion_enc])

    df_light = pd.DataFrame({"label": labels, "final_conf": final_conf, "confusion": confusion})

    metric_names = list(data["metric_names"].astype(str)) if "metric_names" in data.files else []
    metric_step_idx = data["metric_step_idx"].astype(int) if "metric_step_idx" in data.files else np.array([], dtype=int)

    if metric_names and len(metric_step_idx) > 0:
        safe_idx = metric_step_idx[metric_step_idx < len(labels)]
        sampled_labels = labels[safe_idx]
        df_m: dict = {"label": sampled_labels}
        n = len(safe_idx)
        for mname in metric_names:
            key = f"msc_{mname}"
            if key in data.files:
                scores = data[key].astype(float)
                df_m[mname] = scores[:n]
        df_metrics = pd.DataFrame(df_m)
    else:
        df_metrics  = pd.DataFrame({"label": np.array([], dtype=int)})
        metric_names = []

    return df_light, df_metrics, metric_names


def load_predictions(run_dir: Path, metric_sample_every: int = 100) -> "tuple[pd.DataFrame, pd.DataFrame, list[str]]":
    """
    Returns:
        df_light   — all rows, only label/final_conf/confusion (for AUC + score histograms)
        df_metrics — sampled rows with per-metric score columns extracted
        metric_names — list of detected metric names (excluding 'final')
    """
    csv = run_dir / "trial_predictions.csv"
    if not csv.exists():
        result = load_predictions_from_summary(run_dir)
        if result is not None:
            return result
        # Both files absent — caller will use log-derived metrics instead.
        return None, None, []

    df_light = pd.read_csv(csv, comment="#", header=None, names=_CSV_COLS,
                           usecols=["label", "final_conf", "confusion"])
    df_light["label"]      = pd.to_numeric(df_light["label"],      errors="coerce")
    df_light["final_conf"] = pd.to_numeric(df_light["final_conf"], errors="coerce")

    # Sample metrics column
    df_sample = pd.read_csv(csv, comment="#", header=None, names=_CSV_COLS,
                            usecols=["label", "metrics"])
    df_sample = df_sample.iloc[::metric_sample_every].copy()

    # Extract per-metric scores via regex on sampled rows
    metric_names: list[str] = []
    _score_pat = re.compile(r"'(\w+)':\s*\{'score':\s*([\d.eE+\-]+)")
    first_valid = df_sample["metrics"].dropna().iloc[0] if not df_sample["metrics"].dropna().empty else ""
    metric_names = [k for k, _ in _score_pat.findall(first_valid) if k != "final"]

    for mname in metric_names:
        pat = re.compile(rf"'{mname}'[^}}]+?'score':\s*([\d.eE+\-]+)")
        df_sample[mname] = df_sample["metrics"].str.extract(pat).astype(float)

    return df_light, df_sample[["label"] + metric_names], metric_names


def compute_auc(df_light: pd.DataFrame) -> float:
    valid = df_light.dropna(subset=["label", "final_conf"])
    if valid["label"].nunique() < 2:
        return float("nan")
    return roc_auc_score(valid["label"].astype(int), valid["final_conf"])


def compute_threshold_recommendations(df_light: pd.DataFrame) -> dict:
    """Compute F1, F2, and Youden's J optimal thresholds from the predictions CSV."""
    valid = df_light.dropna(subset=["label", "final_conf"])
    if valid["label"].nunique() < 2:
        return {}
    y = valid["label"].astype(int).values
    s = valid["final_conf"].values
    grid = np.linspace(0.05, 0.95, 91)
    best = {"cfg_thr": (None, -1.), "cfg_thr_f2": (None, -1.), "cfg_thr_youden": (None, -1.)}
    for thr in grid:
        yhat = (s >= thr).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        tnr  = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        ok   = np.isfinite(prec) and np.isfinite(rec)
        f1   = (2*prec*rec / (prec+rec))     if ok and (prec+rec) > 0     else float("nan")
        f2   = (5*prec*rec / (4*prec+rec))   if ok and (4*prec+rec) > 0   else float("nan")
        j    = (rec + tnr - 1.)              if (np.isfinite(rec) and np.isfinite(tnr)) else float("nan")
        for key, val in [("cfg_thr", f1), ("cfg_thr_f2", f2), ("cfg_thr_youden", j)]:
            if np.isfinite(val) and val > best[key][1]:
                best[key] = (float(thr), val)
    return {k: v[0] for k, v in best.items() if v[0] is not None}


# ─── column / row ordering ────────────────────────────────────────────────────
# Desired display order: MLP_point → LSTM_point → MLP_window → LSTM_window
# MLP always precedes LSTM within the same mode; point before window.

_ARCH_ORDER    = ["MLP", "LSTM", "MLP_Attn", "LSTM_Attn", "TF_VAE"]
_SECTION_ORDER = ["beta", "anneal", "free_bits"]   # section display order


def _col_sort_key(col_key: tuple[str, str]) -> tuple[int, int]:
    arch, mode = col_key
    mode_idx = 0 if mode == "point" else 1
    base = _base_arch(arch)
    arch_idx = _ARCH_ORDER.index(base) if base in _ARCH_ORDER else 99
    return (mode_idx, arch_idx)


def _section_idx(section: str) -> int:
    return _SECTION_ORDER.index(section) if section in _SECTION_ORDER else 99


def _row_sort_key(rk: str) -> tuple:
    """Sort 'section=value' row_key strings: by section order, then numeric value."""
    if "=" in rk:
        sec, val = rk.split("=", 1)
        try:
            return (_section_idx(sec), float(val))
        except ValueError:
            return (_section_idx(sec), 0.0)
    # Legacy plain row keys (cyclic, plain, …) sort after sectioned ones
    return (98, 0.0)


def _run_sort_key(r: dict) -> tuple:
    """Sort by: arch/mode, then section order, then variant value.

    When the variant (the part after '=' in row_key, or the whole row_key) is
    numeric, sort by its float value — this overrides _compare_order, which may
    have been assigned in a non-sequential order in the suite YAML.
    Non-numeric variants fall back to _compare_order then string sort.
    """
    rk = r["row_key"]
    val_str = rk.split("=", 1)[1] if "=" in rk else rk
    try:
        numeric_val = float(val_str)
        order: float = numeric_val          # numeric variants: sort by value
        rk_tiebreak: tuple = (0, numeric_val)
    except (ValueError, TypeError):
        order = float(r.get("_compare_order", 0))   # non-numeric: use YAML order
        rk_tiebreak = (1, val_str)
    return (
        _col_sort_key(r["col_key"]),
        _section_idx(r.get("section", "")),
        order,
        0 if rk == "cyclic" else 1,
        rk_tiebreak,
    )


def _is_numeric(v: str) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _extract_numeric(v: str) -> "float | None":
    """Extract the first number from a variant string.
    'w24' → 24.0,  'buf100' → 100.0,  'q0.85' → 0.85,  '0.90' → 0.90."""
    if _is_numeric(v):
        return float(v)
    m = re.search(r'\d+(?:\.\d+)?', v)
    return float(m.group()) if m else None


# ─── plotting helpers ─────────────────────────────────────────────────────────

def _arch_color(arch: str) -> str:
    return ARCH_COLORS.get(arch, ARCH_COLORS.get(_base_arch(arch), "#999999"))


def _bar_label(ax, rects, fmt="{:.0f}"):
    for r in rects:
        h = r.get_height()
        if h > 0:
            ax.text(r.get_x() + r.get_width() / 2, h * 1.01,
                    fmt.format(h), ha="center", va="bottom", fontsize=7)


# ─── plot 1: confusion grid ────────────────────────────────────────────────────

def plot_confusion_grid(runs_for_anomaly: list[dict], anomaly: str, out_dir: Path):
    """
    Table layout: rows = architectures, columns = sections.
    Each cell is labelled "{arch} — {section}" and contains two stacked bar charts:
      Top subplot    — anomaly budget per variant: TP (green bottom) + FN (red top)
      Bottom subplot — normal  budget per variant: TN (green bottom) + FP (red top)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    runs_sorted = sorted(runs_for_anomaly, key=_run_sort_key)
    archs    = list(dict.fromkeys(r["arch"] for r in runs_sorted))
    sections = list(dict.fromkeys(r.get("section", "") or "default" for r in runs_sorted))
    if not archs or not sections:
        return

    # (arch, section) → runs in display order
    cell_runs: dict[tuple, list] = {}
    for r in runs_sorted:
        key = (r["arch"], r.get("section", "") or "default")
        cell_runs.setdefault(key, []).append(r)

    n_archs = len(archs)
    n_secs  = len(sections)
    max_vars = max((len(cell_runs.get((a, s), [])) for a in archs for s in sections), default=1)
    cell_w   = max(3.0, 0.9 * max_vars)
    cell_h   = 4.5

    fig = plt.figure(figsize=(cell_w * n_secs, cell_h * n_archs))
    fig.suptitle(f"Confusion proportions — {anomaly}", fontsize=13, y=1.01)

    outer_gs = gridspec.GridSpec(n_archs, n_secs, figure=fig, hspace=0.55, wspace=0.35)

    _bp_kw = dict(patch_artist=True, manage_ticks=False, vert=True)

    for ai, arch in enumerate(archs):
        for si, sec in enumerate(sections):
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[ai, si], hspace=0.06
            )
            ax_anom = fig.add_subplot(inner_gs[0])
            ax_norm = fig.add_subplot(inner_gs[1])

            ax_anom.set_title(f"{arch} — {sec}", fontsize=8, fontweight="bold", pad=3)
            ax_anom.set_ylabel("anomalies\n(TP+FN)", fontsize=7)
            ax_norm.set_ylabel("normals\n(TN+FP)", fontsize=7)

            runs_cell = cell_runs.get((arch, sec), [])
            if not runs_cell:
                for ax in (ax_anom, ax_norm):
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=10)
                    ax.set_xticks([])
                continue

            tick_labels = []
            for vi, r in enumerate(runs_cell):
                m   = r.get("log_metrics", {})
                tp  = max(int(m.get("tp", 0) or 0), 0)
                fn  = max(int(m.get("fn", 0) or 0), 0)
                tn  = max(int(m.get("tn", 0) or 0), 0)
                fp  = max(int(m.get("fp", 0) or 0), 0)
                f1  = m.get("f1",  float("nan"))
                auc = m.get("auc", float("nan"))

                n_anom = tp + fn
                n_norm = tn + fp
                tp_pct = 100.0 * tp / n_anom if n_anom > 0 else 0.0
                fn_pct = 100.0 * fn / n_anom if n_anom > 0 else 0.0
                tn_pct = 100.0 * tn / n_norm if n_norm > 0 else 0.0
                fp_pct = 100.0 * fp / n_norm if n_norm > 0 else 0.0

                ax_anom.bar(vi, tp_pct,                color=CONFUSION_COLORS["TP"], edgecolor="white", lw=0.4)
                ax_anom.bar(vi, fn_pct, bottom=tp_pct, color=CONFUSION_COLORS["FN"], edgecolor="white", lw=0.4)
                ax_norm.bar(vi, tn_pct,                color=CONFUSION_COLORS["TN"], edgecolor="white", lw=0.4)
                ax_norm.bar(vi, fp_pct, bottom=tn_pct, color=CONFUSION_COLORS["FP"], edgecolor="white", lw=0.4)

                # Show counts inside segments only when the segment is tall enough to fit text
                _MIN_PCT = 5.0
                if tp_pct >= _MIN_PCT:
                    ax_anom.text(vi, tp_pct / 2,           f"TP={tp:,}", fontsize=5.5,
                                 va="center", ha="center", color="black")
                if fn_pct >= _MIN_PCT:
                    ax_anom.text(vi, tp_pct + fn_pct / 2,  f"FN={fn:,}", fontsize=5.5,
                                 va="center", ha="center", color="black")
                if tn_pct >= _MIN_PCT:
                    ax_norm.text(vi, tn_pct / 2,           f"TN={tn:,}", fontsize=5.5,
                                 va="center", ha="center", color="black")
                if fp_pct >= _MIN_PCT:
                    ax_norm.text(vi, tn_pct + fp_pct / 2,  f"FP={fp:,}", fontsize=5.5,
                                 va="center", ha="center", color="black")

                # Total sample count above each bar
                ax_anom.text(vi, 101, f"n={n_anom:,}", ha="center", va="bottom", fontsize=5, color="#555555")
                ax_norm.text(vi, 101, f"n={n_norm:,}", ha="center", va="bottom", fontsize=5, color="#555555")

                v = r.get("row_key", "")
                label   = v.split("=")[-1] if "=" in v else v
                f1_str  = f"F1={f1:.3f}"   if np.isfinite(f1)  else "F1=n/a"
                auc_str = f"AUC={auc:.3f}" if np.isfinite(auc) else "AUC=n/a"
                tick_labels.append(f"{label}\n{f1_str}\n{auc_str}")

            xs = list(range(len(runs_cell)))
            ax_anom.set_xticks(xs)
            ax_anom.set_xticklabels([])          # labels only on bottom subplot
            ax_norm.set_xticks(xs)
            ax_norm.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=7)
            for ax in (ax_anom, ax_norm):
                ax.set_ylim(0, 108)  # uniform 0–100 % scale; headroom for "n=…" labels
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f"{y:.0f}%")
                )
                ax.grid(axis="y", alpha=0.25)
                ax.tick_params(axis="y", labelsize=7)

    bar_legend = [
        Patch(facecolor=CONFUSION_COLORS["TP"], label="TP"),
        Patch(facecolor=CONFUSION_COLORS["FN"], label="FN"),
        Patch(facecolor=CONFUSION_COLORS["TN"], label="TN"),
        Patch(facecolor=CONFUSION_COLORS["FP"], label="FP"),
    ]
    fig.legend(handles=bar_legend, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(out_dir / "confusion_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/confusion_grid.png")


# ─── plot 2: performance bars ──────────────────────────────────────────────────

def plot_performance_bars(runs: list[dict], anomaly: str, out_dir: Path) -> None:
    """Single-column, 4-row bar chart: F1 / AUC / Recall / Precision.

    All four plots share the same x-axis and run order so bars are visually
    aligned when scanning top-to-bottom.  Saved to out_dir/performance_bars.png.
    """
    runs_sorted = sorted(runs, key=_run_sort_key)
    if not runs_sorted:
        return

    metrics_shown = [("f1", "F1"), ("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision")]
    nruns = len(runs_sorted)
    fig_w = max(8, 1.2 * nruns)
    fig, axes = plt.subplots(4, 1, figsize=(fig_w, 16), sharex=True)

    section_breaks = [i for i in range(1, nruns)
                      if runs_sorted[i].get("section", "") != runs_sorted[i - 1].get("section", "")]
    run_labels = [r["short_label"] for r in runs_sorted]

    for ax, (mkey, mlabel) in zip(axes, metrics_shown):
        vals = [r.get("log_metrics", {}).get(mkey, float("nan")) for r in runs_sorted]
        clrs = [_arch_color(r["arch"]) for r in runs_sorted]
        bars = ax.bar(range(nruns), vals, color=clrs, edgecolor="white", linewidth=0.5)
        _bar_label(ax, bars, fmt="{:.3f}")
        for brk in section_breaks:
            ax.axvline(brk - 0.5, color="gray", lw=1.0, ls="--", alpha=0.6)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        # Metric name inside the subplot (visible without scrolling to y-axis)
        ax.text(0.005, 0.97, mlabel, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color="dimgray", alpha=0.55)

    # ── Section labels: above each section block on every subplot ────────────
    boundaries = [0] + section_breaks + [nruns]
    for ax in axes:
        for s, e in zip(boundaries, boundaries[1:]):
            sec = runs_sorted[s].get("section", "")
            if sec:
                ax.text((s + e - 1) / 2, 1.04, sec, ha="center", va="bottom",
                        fontsize=8, style="italic",
                        transform=ax.get_xaxis_transform())

    # ── Bottom x-tick labels (last subplot) ──────────────────────────────────
    axes[-1].set_xticks(range(nruns))
    axes[-1].set_xticklabels(run_labels, rotation=45, ha="right", fontsize=9)

    # ── Top x-tick labels (first subplot, mirrored) ──────────────────────────
    axes[0].tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    axes[0].set_xticks(range(nruns))
    axes[0].set_xticklabels(run_labels, rotation=45, ha="left", fontsize=9)

    seen_archs = sorted(set(r["arch"] for r in runs_sorted))
    handles = [plt.Rectangle((0, 0), 1, 1, color=_arch_color(a)) for a in seen_archs]
    fig.legend(handles, seen_archs, loc="upper right", fontsize=9, title="Arch")
    fig.suptitle(f"Performance — {anomaly}", fontsize=13)
    fig.tight_layout()

    out = out_dir / "performance_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/{out.name}")


# ─── plot 2b: section line plots (new) ────────────────────────────────────────

def plot_section_lines(runs: list[dict], anomaly: str, out_dir: Path) -> None:
    """Per-section line plots: x = variant (ascending), y = metric, one line per arch.

    One file per section: section_lines_{section}.png
    Metrics shown: F1, AUC, Recall, Precision in a single-column 4-row layout.
    Runs without a `section` field are skipped.
    """
    sections_map: dict[str, list[dict]] = {}
    for r in runs:
        sec = r.get("section", "")
        if sec:
            sections_map.setdefault(sec, []).append(r)
    if not sections_map:
        return

    metrics_shown = [("f1", "F1"), ("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision")]

    for sec, sec_runs in sections_map.items():
        archs = sorted(
            set(r["arch"] for r in sec_runs),
            key=lambda a: (_ARCH_ORDER.index(_base_arch(a)) if _base_arch(a) in _ARCH_ORDER else 99),
        )

        def _variant_label(r: dict) -> str:
            v = r.get("row_key", "")
            return v.split("=")[-1] if "=" in v else v

        all_variants = sorted(
            set(_variant_label(r) for r in sec_runs),
            key=lambda v: (0, _extract_numeric(v)) if _extract_numeric(v) is not None else (1, v),
        )
        if not all_variants:
            continue

        lookup: dict[tuple[str, str], dict] = {
            (r["arch"], _variant_label(r)): r for r in sec_runs
        }

        fig_w = max(6, 1.5 * len(all_variants) + 2)
        fig, axes = plt.subplots(4, 1, figsize=(fig_w, 14), sharex=True)

        for ax, (mkey, mlabel) in zip(axes, metrics_shown):
            all_ys_finite = []
            for arch in archs:
                ys = [
                    lookup.get((arch, v), {}).get("log_metrics", {}).get(mkey, float("nan"))
                    for v in all_variants
                ]
                ax.plot(range(len(all_variants)), ys, "o-",
                        color=_arch_color(arch), label=arch,
                        lw=1.8, ms=6, markeredgecolor="white", markeredgewidth=0.5)
                all_ys_finite.extend(y for y in ys if np.isfinite(y))
            ax.set_ylabel(mlabel, fontsize=10)
            if all_ys_finite:
                lo, hi = min(all_ys_finite), max(all_ys_finite)
                rng = hi - lo if hi > lo else 0.1
                pad = 0.15 * rng
                ax.set_ylim(lo - pad, hi + pad)
            else:
                ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
            ax.grid(True, which="minor", alpha=0.1)

        axes[-1].set_xticks(range(len(all_variants)))
        axes[-1].set_xticklabels(all_variants, rotation=45, ha="right", fontsize=9)

        handles = [
            plt.Line2D([0], [0], color=_arch_color(a), lw=2, marker="o",
                       markeredgecolor="white", label=a)
            for a in archs
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=9, title="Arch")
        fig.suptitle(f"{sec} — {anomaly}", fontsize=13)
        fig.tight_layout()

        safe = re.sub(r"[^\w]", "_", sec)
        out = out_dir / f"section_lines_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_dir.name}/{out.name}")


# ─── plot 2c: arch comparison grouped bars ────────────────────────────────────

def plot_arch_comparison_bars(runs: list[dict], anomaly: str, out_dir: Path) -> None:
    """
    One file per anomaly type: arch_comparison.png

    X-axis: arch groups (MLP, LSTM, TF_VAE, …), within each group one bar per
    variant (e.g. latent_dim value).  Bars are averaged over seeds; error bars
    show ±1 std when more than one seed is available.

    Four metric rows: F1, AUC, Recall, Precision.
    """
    from collections import defaultdict

    runs_sorted = sorted(runs, key=_run_sort_key)
    if not runs_sorted:
        return

    def _variant_of(r: dict) -> str:
        rk = r.get("row_key", "")
        return rk.split("=")[-1] if "=" in rk else rk

    # Group by (arch, variant) → list of runs (seeds)
    group_data: dict[tuple[str, str], list] = defaultdict(list)
    for r in runs_sorted:
        group_data[(r["arch"], _variant_of(r))].append(r)

    # Arch display order
    archs = list(dict.fromkeys(
        r["arch"] for r in sorted(
            runs_sorted,
            key=lambda r: (_ARCH_ORDER.index(_base_arch(r["arch"]))
                           if _base_arch(r["arch"]) in _ARCH_ORDER else 99),
        )
    ))

    # Variant order within each arch (numeric ascending when possible)
    arch_variants: dict[str, list[str]] = {}
    for arch in archs:
        variants = sorted(
            set(k[1] for k in group_data if k[0] == arch),
            key=lambda v: (0, _extract_numeric(v)) if _extract_numeric(v) is not None else (1, v),
        )
        arch_variants[arch] = variants

    # Build bar x-positions: variants packed within each arch, gap between archs
    GAP   = 1.5
    BAR_W = 0.8
    positions: dict[tuple[str, str], float] = {}
    arch_spans: dict[str, tuple[float, float]] = {}
    x = 0.0
    for arch in archs:
        variants = arch_variants[arch]
        start_x = x
        for variant in variants:
            positions[(arch, variant)] = x
            x += BAR_W
        arch_spans[arch] = (start_x, x - BAR_W)
        x += GAP

    def _mean_std(values):
        vals = [float(v) for v in values if v is not None]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float("nan"), 0.0
        return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)

    metrics_shown = [("f1", "F1"), ("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision")]
    n_bars = len(positions)
    fig_w  = max(10, 0.7 * n_bars + 4)
    fig, axes = plt.subplots(4, 1, figsize=(fig_w, 16), sharex=True)
    fig.suptitle(f"Architecture comparison — {anomaly}", fontsize=13)

    for ax, (mkey, mlabel) in zip(axes, metrics_shown):
        for arch in archs:
            color    = _arch_color(arch)
            variants = arch_variants[arch]
            for variant in variants:
                cell_runs = group_data.get((arch, variant), [])
                vals = [r.get("log_metrics", {}).get(mkey, float("nan")) for r in cell_runs]
                mean_v, std_v = _mean_std(vals)
                if not np.isfinite(mean_v):
                    continue
                xpos = positions[(arch, variant)]
                ax.bar(xpos, mean_v, width=BAR_W * 0.82,
                       color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
                if std_v > 0:
                    ax.errorbar(xpos, mean_v, yerr=std_v, fmt="none",
                                color="black", capsize=3, lw=1.1, capthick=1.0)
                ax.text(xpos, mean_v + std_v + 0.01, f"{mean_v:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=45)

        # Arch group labels (above each group) and separators
        for ai, arch in enumerate(archs):
            start_x, end_x = arch_spans[arch]
            center_x = (start_x + end_x) / 2
            ax.text(center_x, 1.09, arch, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=_arch_color(arch),
                    transform=ax.get_xaxis_transform())
            if ai < len(archs) - 1:
                sep_x = end_x + GAP / 2
                ax.axvline(sep_x, color="gray", lw=1.0, ls="--", alpha=0.45)

        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_ylim(0, 1.22)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.004, 0.97, mlabel, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color="dimgray", alpha=0.5)

    # X-tick labels: variant value under each bar (shared across all 4 subplots)
    tick_xs = [positions[(arch, v)] for arch in archs for v in arch_variants[arch]]
    tick_lb = [v for arch in archs for v in arch_variants[arch]]
    axes[-1].set_xticks(tick_xs)
    axes[-1].set_xticklabels(tick_lb, rotation=45, ha="right", fontsize=8)

    seen_archs = [a for a in _ARCH_ORDER if a in set(r["arch"] for r in runs)]
    for a in set(r["arch"] for r in runs):
        if a not in seen_archs:
            seen_archs.append(a)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_arch_color(a)) for a in seen_archs]
    fig.legend(handles, seen_archs, loc="upper right", fontsize=9, title="Arch")
    fig.tight_layout()

    out = out_dir / "arch_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/{out.name}")


# ─── plot 2d: seed-stability heatmap ─────────────────────────────────────────

_STD_SEVERITY = [
    (0.005, None),        # stable — no marker
    (0.015, "#2980b9"),   # blue       (very slight)
    (0.030, "#8e44ad"),   # purple     (slight)
    (0.060, "#e67e22"),   # orange     (moderate)
    (float("inf"), "#c0392b"),  # dark red  (high)
]


def _std_dot_color(std: float) -> "str | None":
    for thr, color in _STD_SEVERITY:
        if std < thr:
            return color
    return _STD_SEVERITY[-1][1]


def plot_seed_stability_heatmap(runs: list[dict], anomaly: str, out_dir: Path) -> None:
    """
    One heatmap per section: rows = variant values, cols = archs.

    Background colour = mean F1 on a RdYlGn scale (red=low, green=high).
    Coloured dot in the bottom-right of each cell = std(F1) severity across seeds:
      no dot          : std < 0.05  (stable)
      yellow  ●  : std 0.05 – 0.10
      orange  ●  : std 0.10 – 0.20
      red     ●  : std 0.20 – 0.30
      dark-red ● : std > 0.30
    """
    from collections import defaultdict
    from copy import copy as _copy
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    def _vkey(r: dict) -> str:
        v = r.get("row_key", "")
        return v.split("=")[-1] if "=" in v else v

    # Group by section
    sec_map: dict[str, list[dict]] = {}
    for r in runs:
        sec = r.get("section", "") or "default"
        sec_map.setdefault(sec, []).append(r)

    cmap_base = _copy(plt.cm.RdYlGn)
    cmap_base.set_bad(color="#d8d8d8")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for sec, sec_runs in sec_map.items():
        archs = list(dict.fromkeys(
            r["arch"] for r in sorted(
                sec_runs,
                key=lambda r: (_ARCH_ORDER.index(_base_arch(r["arch"]))
                               if _base_arch(r["arch"]) in _ARCH_ORDER else 99),
            )
        ))
        all_variants = sorted(
            set(_vkey(r) for r in sec_runs),
            key=lambda v: (_extract_numeric(v), v) if _extract_numeric(v) is not None else (float("inf"), v),
        )
        if not archs or not all_variants:
            continue

        # Aggregate mean/std F1 per (arch, variant)
        cells_f1: dict[tuple, list] = defaultdict(list)
        for r in sec_runs:
            f1 = r.get("log_metrics", {}).get("f1", float("nan"))
            if np.isfinite(f1):
                cells_f1[(r["arch"], _vkey(r))].append(f1)

        n_rows, n_cols = len(all_variants), len(archs)
        f1_mat  = np.full((n_rows, n_cols), np.nan)
        std_mat = np.full((n_rows, n_cols), np.nan)
        for ri, variant in enumerate(all_variants):
            for ci, arch in enumerate(archs):
                vals = cells_f1.get((arch, variant), [])
                if vals:
                    f1_mat[ri, ci]  = float(np.mean(vals))
                    std_mat[ri, ci] = float(np.std(vals)) if len(vals) > 1 else 0.0

        cell_h = 0.75
        cell_w = 2.2
        fig_h  = max(2.5, cell_h * n_rows + 1.4)
        fig_w  = max(4.0, cell_w * n_cols + 2.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        masked = np.ma.masked_invalid(f1_mat)
        im = ax.imshow(masked, cmap=cmap_base, norm=norm, aspect="auto",
                       interpolation="nearest", origin="upper")

        for ri in range(n_rows):
            for ci in range(n_cols):
                mean_v = f1_mat[ri, ci]
                std_v  = std_mat[ri, ci]
                if not np.isfinite(mean_v):
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=9, color="#999999")
                    continue

                # Mean F1 (centred, slightly above mid to leave room for dot)
                ax.text(ci, ri - 0.12, f"{mean_v:.3f}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="black")

                # Std severity dot (bottom-right corner of cell)
                if np.isfinite(std_v):
                    dot_col = _std_dot_color(std_v)
                    if dot_col is not None:
                        ax.plot(ci + 0.38, ri + 0.35, "o",
                                color=dot_col, markersize=8,
                                markeredgecolor="white", markeredgewidth=0.5,
                                zorder=4, clip_on=True)
                        ax.text(ci + 0.38, ri + 0.35, f"{std_v:.2f}",
                                ha="center", va="center",
                                fontsize=4.5, color="white",
                                fontweight="bold", zorder=5)

        # Axes labels
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(archs, fontsize=9, rotation=30, ha="right")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(all_variants, fontsize=9)
        ax.set_ylabel(sec, fontsize=10)
        ax.set_title(f"Mean F1 with seed-stability — {anomaly}  [{sec}]",
                     fontsize=11, pad=16)

        # Grid lines between cells
        for x in np.arange(-0.5, n_cols, 1):
            ax.axvline(x, color="white", lw=0.8)
        for y in np.arange(-0.5, n_rows, 1):
            ax.axhline(y, color="white", lw=0.8)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_base, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("mean F1", fontsize=9)

        # Legend for dot colours
        dot_legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=9, label=lbl)
            for lbl, c in [
                ("std 0.005–0.015", "#2980b9"),
                ("std 0.015–0.030", "#8e44ad"),
                ("std 0.030–0.060", "#e67e22"),
                ("std > 0.060",     "#c0392b"),
            ]
        ]
        ax.legend(handles=dot_legend, loc="lower right",
                  bbox_to_anchor=(1.0, -0.02 - 0.04 * max(n_rows - 4, 0)),
                  fontsize=8, title="Seed instability", title_fontsize=8.5,
                  framealpha=0.92)

        fig.tight_layout()
        safe_sec = re.sub(r"[^\w]", "_", sec)
        out = out_dir / f"seed_stability_{safe_sec}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_dir.name}/{out.name}")


# ─── plot 3: metric score distributions ───────────────────────────────────────

def plot_metric_distributions(runs_for_anomaly: list[dict], anomaly: str, out_dir: Path):
    """Table of histogram+quartile plots: rows=variants, cols=archs.

    One file per metric: metric_dist_{metric}.png
    Each cell shows overlapping histograms for normal (green) and anomaly (red)
    with Q1/Q2/Q3 vertical markers and a Q1-Q3 shaded band per class.
    """
    from matplotlib.patches import Patch
    runs_sorted = sorted(runs_for_anomaly, key=_run_sort_key)

    archs = list(dict.fromkeys(
        r["arch"] for r in sorted(
            runs_sorted,
            key=lambda r: (_ARCH_ORDER.index(_base_arch(r["arch"]))
                           if _base_arch(r["arch"]) in _ARCH_ORDER else 99),
        )
    ))
    variants = list(dict.fromkeys(r["row_key"] for r in runs_sorted))

    all_metric_names: list[str] = []
    for r in runs_sorted:
        for m in r.get("metric_names", []):
            if m not in all_metric_names:
                all_metric_names.append(m)
    if not all_metric_names:
        return

    lookup = {(r["arch"], r["row_key"]): r for r in runs_sorted}

    _bp_kw = dict(
        patch_artist=True, manage_ticks=False, vert=False,
        medianprops=dict(color="black", lw=1.5),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    )

    for mname in all_metric_names:
        nrows, ncols = len(variants), len(archs)
        cell_w, cell_h = 4.0, 1.6
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(cell_w * ncols, cell_h * nrows),
                                 squeeze=False)
        fig.suptitle(f"{mname} — {anomaly}", fontsize=12, y=1.01)

        for ri, variant in enumerate(variants):
            # Show full "section=value" label on the left
            row_label = variant if variant else "—"
            for ci, arch in enumerate(archs):
                ax = axes[ri][ci]
                if ri == 0:
                    ax.set_title(arch, fontsize=9, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(row_label, fontsize=8)

                r = lookup.get((arch, variant))
                if r is None or r.get("df_metrics") is None:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=9)
                    ax.set_yticks([])
                    continue

                df_m = r["df_metrics"]
                if mname not in df_m.columns:
                    ax.set_visible(False)
                    continue

                normal = df_m.loc[df_m["label"] == 0, mname].dropna().values
                anom_d = df_m.loc[df_m["label"] == 1, mname].dropna().values

                # x range: 1st–99th pct across both classes
                all_v = np.concatenate([normal, anom_d]) if len(normal) + len(anom_d) > 0 else np.array([0.0])
                xmin = float(np.percentile(all_v, 1))
                xmax = float(np.percentile(all_v, 99))
                if xmin >= xmax:
                    xmin, xmax = xmin - 1.0, xmax + 1.0

                # Horizontal box plots: normal at y=1 (green), anomaly at y=0 (red)
                if len(normal) >= 2:
                    ax.boxplot([normal], positions=[1], widths=0.5,
                               boxprops=dict(facecolor="#27ae60", alpha=0.65), **_bp_kw)
                if len(anom_d) >= 2:
                    ax.boxplot([anom_d], positions=[0], widths=0.5,
                               boxprops=dict(facecolor="#e74c3c", alpha=0.65), **_bp_kw)

                ax.set_yticks([0, 1])
                ax.set_yticklabels(["anomaly", "normal"], fontsize=7)
                ax.set_xlim(xmin, xmax)
                ax.tick_params(axis="x", labelsize=7)
                ax.set_ylim(-0.6, 1.6)

        # Shared legend
        fig.legend(
            handles=[Patch(facecolor="#27ae60", alpha=0.65, label="normal"),
                     Patch(facecolor="#e74c3c", alpha=0.65, label="anomaly")],
            loc="upper right", fontsize=9, framealpha=0.8,
        )
        fig.tight_layout()
        safe_m = re.sub(r"[^\w]", "_", mname)
        out = out_dir / f"metric_dist_{safe_m}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_dir.name}/{out.name}")


# ─── plot 4: score histograms ──────────────────────────────────────────────────

def _score_hist_cell(ax: "plt.Axes", run: dict) -> None:
    """Fill one score-histogram cell (reused across layouts)."""
    df_l = run.get("df_light")
    if df_l is None:
        ax.set_visible(False)
        return
    for lbl, color, zorder in [(0, "#27ae60", 1), (1, "#e74c3c", 2)]:
        sub = df_l.loc[df_l["label"] == lbl, "final_conf"].dropna()
        if len(sub):
            ax.hist(sub, bins=80, alpha=0.55, color=color, density=True,
                    label=f"{'normal' if lbl == 0 else 'anomaly'}", zorder=zorder)
    lm = run.get("log_metrics", {})
    for thr_key, color, ls, fmt in [
        ("ensemble_thr", "black",      "-",  "anom={:.2f}"),
        ("update_if_thr","gray",        ":", "gate={:.2f}"),
    ]:
        v = lm.get(thr_key)
        if v is not None:
            ax.axvline(v, color=color, lw=1.2, ls=ls, label=fmt.format(v))
    ann = f"F1={lm.get('f1', float('nan')):.3f}  AUC={lm.get('auc', float('nan')):.3f}"
    ax.text(0.98, 0.97, ann, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    ax.set_xlabel("conf", fontsize=7)
    ax.tick_params(labelsize=7)

    # Clip y-axis when a saturation spike at conf≈1 dominates the scale.
    # If the tallest bar is > 3× the 90th-percentile bar, cap at 2× p90 so the
    # bulk of the distribution is visible (the spike shows as a truncated bar).
    heights = sorted(p.get_height() for p in ax.patches if p.get_height() > 0)
    if len(heights) >= 4:
        p90 = float(np.percentile(heights, 90))
        if heights[-1] > 3.0 * p90 > 0:
            ax.set_ylim(0, p90 * 2.0)
            ax.annotate("▲ clipped", xy=(1, 1), xycoords="axes fraction",
                        fontsize=6, color="gray", ha="right", va="top")


def plot_score_histograms(runs_for_anomaly: list[dict], anomaly: str, out_dir: Path):
    """Per-architecture score histogram files.

    One file per arch: score_hist_{arch}.png
    Table layout: columns = sections, rows = variants within each section.
    """
    archs = sorted(
        set(r["arch"] for r in runs_for_anomaly),
        key=lambda a: (_ARCH_ORDER.index(_base_arch(a)) if _base_arch(a) in _ARCH_ORDER else 99),
    )

    for arch in archs:
        arch_runs = sorted(
            [r for r in runs_for_anomaly if r["arch"] == arch],
            key=_run_sort_key,
        )
        if not arch_runs:
            continue

        # Collect sections in display order; runs without a section go into "default"
        sections = list(dict.fromkeys(
            r.get("section", "") or "default" for r in arch_runs
        ))

        sec_variants: dict[str, list[str]] = {}
        sec_lookup:   dict[tuple[str, str], dict] = {}
        for sec in sections:
            sec_r = [r for r in arch_runs if (r.get("section", "") or "default") == sec]
            vkeys = list(dict.fromkeys(r["row_key"] for r in sec_r))
            sec_variants[sec] = vkeys
            for r in sec_r:
                sec_lookup[(sec, r["row_key"])] = r

        ncols = len(sections)
        nrows = max(len(v) for v in sec_variants.values())

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle(f"Score histograms — {arch} — {anomaly}", fontsize=12)

        for ci, sec in enumerate(sections):
            variants = sec_variants[sec]
            for ri in range(nrows):
                ax = axes[ri][ci]
                if ri >= len(variants):
                    ax.set_visible(False)
                    continue
                variant = variants[ri]
                short_v = variant.split("=")[-1] if "=" in variant else variant
                ax.set_title(f"{sec}  |  {short_v}", fontsize=8, fontweight="bold")
                r = sec_lookup.get((sec, variant))
                if r is None:
                    ax.set_visible(False)
                    continue
                _score_hist_cell(ax, r)

        fig.tight_layout()
        safe_arch = re.sub(r"[^\w]", "_", arch)
        out = out_dir / f"score_hist_{safe_arch}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_dir.name}/{out.name}")


# ─── plot 5: latent PCA (optional) ────────────────────────────────────────────

# Seasonal colour palette — index 0 = January (matches analysis_plots.py)
_MONTH_COLORS = [
    "#08306b",  # Jan — deep blue (winter)
    "#6baed6",  # Feb — light blue
    "#41ab5d",  # Mar — green (spring)
    "#41ab5d",  # Apr — green
    "#fee391",  # May — yellow
    "#fd8d3c",  # Jun — orange (summer)
    "#d73027",  # Jul — red
    "#d73027",  # Aug — red
    "#fd8d3c",  # Sep — orange (autumn)
    "#fee391",  # Oct — yellow
    "#6baed6",  # Nov — light blue
    "#08306b",  # Dec — deep blue (winter)
]
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def _scatter3(ax, z3, mask, color, s=4, alpha=0.4, **kw):
    z = z3[mask]
    ax.scatter(z[:, 0], z[:, 1], z[:, 2] if z3.shape[1] > 2 else np.zeros(len(z)),
               c=color, s=s, alpha=alpha, **kw)


def _set_pca_axes(ax, evr, title, fontsize=9):
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(f"PC1 {evr[0]:.1%}", fontsize=7)
    ax.set_ylabel(f"PC2 {evr[1]:.1%}" if len(evr) > 1 else "", fontsize=7)
    ax.set_zlabel(f"PC3 {evr[2]:.1%}" if len(evr) > 2 else "", fontsize=7)
    ax.tick_params(labelsize=6)


def plot_latent_pca_grid(runs_for_anomaly: list[dict], anomaly: str, out_dir: Path,
                         n_samples: int = 4000):
    """Two panels per run (side-by-side rows):
      Left  — confusion coloring: green=normal, red=anomaly
      Right — seasonal month coloring
    """
    from sklearn.decomposition import PCA

    runs_for_anomaly = sorted(runs_for_anomaly, key=_run_sort_key)
    nruns = len(runs_for_anomaly)
    # Layout: nruns rows × 2 cols (left=confusion, right=month)
    fig = plt.figure(figsize=(12, 5 * nruns))
    fig.suptitle(f"Latent PCA (3D) — {anomaly}", fontsize=13, y=1.01)

    for i, run in enumerate(runs_for_anomaly):
        lat_path = run["dir"] / "latent_mu.npy"
        ts_path  = run["dir"] / "latent_timestamps.npy"
        pred_csv = run["dir"] / "trial_predictions.csv"
        if not lat_path.exists():
            continue

        # ── load data ────────────────────────────────────────────────────────
        lat = np.load(lat_path)
        if pred_csv.exists():
            df_l   = pd.read_csv(pred_csv, comment="#", header=None, names=_CSV_COLS,
                                 usecols=["label"])
            labels = pd.to_numeric(df_l["label"], errors="coerce").fillna(0).astype(int).values
        else:
            result = load_predictions_from_summary(run["dir"])
            if result is None:
                print(f"  skipping PCA for {run['run_name']} — no labels available")
                continue
            labels = result[0]["label"].values.astype(int)

        timestamps = None
        if ts_path.exists():
            try:
                timestamps = np.load(ts_path, allow_pickle=True)
            except Exception:
                pass

        # ── subsample consistently ───────────────────────────────────────────
        N   = min(len(lat), n_samples, len(labels))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(lat), N, replace=False) if len(lat) > N else np.arange(len(lat))
        lat_s = lat[idx]
        lbl_s = labels[idx]
        ts_s  = timestamps[idx] if timestamps is not None and len(timestamps) >= len(lat) else None

        # ── PCA ──────────────────────────────────────────────────────────────
        pca  = PCA(n_components=min(3, lat_s.shape[1]))
        z3   = pca.fit_transform(lat_s)
        evr  = pca.explained_variance_ratio_
        anom = lbl_s == 1

        # ── LEFT: confusion plot ─────────────────────────────────────────────
        ax_conf = fig.add_subplot(nruns, 2, i * 2 + 1, projection="3d")
        _scatter3(ax_conf, z3, ~anom, "#27ae60", s=4, alpha=0.35, label="normal")
        if anom.any():
            _scatter3(ax_conf, z3, anom, "#e74c3c", s=8, alpha=0.7, label="anomaly")
        _set_pca_axes(ax_conf, evr, f"{run['short_label']}  [normal/anomaly]")
        ax_conf.legend(fontsize=7, markerscale=2)

        # ── RIGHT: month plot ─────────────────────────────────────────────────
        ax_mon = fig.add_subplot(nruns, 2, i * 2 + 2, projection="3d")

        if ts_s is not None:
            months = np.zeros(len(ts_s), dtype=int)
            for j, t in enumerate(ts_s):
                try:
                    months[j] = pd.Timestamp(t).month
                except Exception:
                    months[j] = 0

            # Plot each month group
            plotted_months = set()
            for m in range(1, 13):
                mask = months == m
                if not mask.any():
                    continue
                label = _MONTH_NAMES[m - 1] if m not in plotted_months else None
                _scatter3(ax_mon, z3, mask, _MONTH_COLORS[m - 1],
                          s=4, alpha=0.4, label=label)
                plotted_months.add(m)

            ax_mon.legend(fontsize=6, markerscale=2, ncol=2,
                          loc="upper left", framealpha=0.6)
        else:
            ax_mon.text2D(0.5, 0.5, "no timestamps", transform=ax_mon.transAxes,
                          ha="center", va="center", fontsize=9, color="gray")

        _set_pca_axes(ax_mon, evr, f"{run['short_label']}  [month]")

    fig.tight_layout()
    out = out_dir / f"latent_pca_{anomaly.lower()}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/{out.name}")


# ─── performance table ────────────────────────────────────────────────────────

def _fmt(v, decimals=3) -> str:
    return f"{v:.{decimals}f}" if pd.notna(v) and v == v else "n/a"


def save_performance_table(all_runs: list[dict], out_dir: Path):
    rows = []
    for r in all_runs:
        m = r.get("log_metrics", {})
        p = r.get("perf", {})
        rk = r.get("row_key", "")
        variant = rk.split("=")[-1] if "=" in rk else rk
        rows.append({
            "run_name":   r["run_name"],
            "arch":       r["arch"],
            "anomaly":    r["anomaly"],
            "variant":    variant,
            "section":    r.get("section", ""),
            "seed":       r.get("trial_n", 0),
            # metrics at the live threshold — match the run's own confusion histogram
            "f1":         m.get("f1"),
            "auc":        m.get("auc"),
            "ap":         m.get("ap"),
            "precision":  m.get("precision"),
            "recall":     m.get("recall"),
            "accuracy":   m.get("accuracy"),
            "tp":         m.get("tp"),
            "fp":         m.get("fp"),
            "tn":         m.get("tn"),
            "fn":         m.get("fn"),
            "n_params":   p.get("n_params") or m.get("n_params"),
            # timing / cost
            "total_s":                p.get("total_s"),
            "warmup_s":               p.get("warmup_s"),
            "stream_s":               p.get("stream_s"),
            "stream_steps_per_sec":   p.get("stream_steps_per_sec"),
            "warmup_samples_per_sec": p.get("warmup_samples_per_sec"),
            "step_ms_mean":           p.get("step_ms_mean"),
            "step_ms_p99":            p.get("step_ms_p99"),
            "gpu_peak_mb":            p.get("gpu_peak_mb"),
            "gpu_reserved_mb":        p.get("gpu_reserved_mb"),
        })

    df = pd.DataFrame(rows)

    # Sort: within each anomaly group, by arch order then variant order then F1 desc
    anomaly_order = {a: i for i, a in enumerate(KNOWN_ANOMALY_TYPES)}
    arch_order    = {a: i for i, a in enumerate(_ARCH_ORDER)}
    df["_ao"] = df["anomaly"].map(anomaly_order).fillna(99)
    df["_ar"] = df["arch"].apply(lambda a: arch_order.get(_base_arch(a), 99))
    df["_vn"] = pd.to_numeric(df["variant"], errors="coerce")
    df = (df.sort_values(["_ao", "_ar", "section", "_vn", "variant", "f1"],
                         ascending=[True, True, True, True, True, False],
                         na_position="last")
            .drop(columns=["_ao", "_ar", "_vn"])
            .reset_index(drop=True))

    csv_out = out_dir / "performance_table.csv"
    df.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"  saved {csv_out.name}")

    # ── shared formatting helpers ────────────────────────────────────────────
    SEP  = "─" * 108
    HDR  = (f"  {'run_name':<44} {'arch':<10} {'variant':<16}"
            f" {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6}"
            f"  {'TP':>7} {'FP':>7} {'TN':>8} {'FN':>6}"
            f"  {'#params':>9}")

    def _row_line(row) -> str:
        n_params = row.get("n_params")
        npstr = f"{int(n_params):,}" if pd.notna(n_params) and n_params == n_params else "n/a"
        return (
            f"  {str(row['run_name']):<44} {str(row['arch']):<10} {str(row['variant']):<16}"
            f" {_fmt(row['f1']):>6} {_fmt(row['auc']):>6}"
            f" {_fmt(row['precision']):>6} {_fmt(row['recall']):>6} {_fmt(row['accuracy']):>6}"
            f"  {_fmt(row['tp'], 0):>7} {_fmt(row['fp'], 0):>7}"
            f" {_fmt(row['tn'], 0):>8} {_fmt(row['fn'], 0):>6}"
            f"  {npstr:>9}"
        )

    lines = []

    # ── Section 1: per-anomaly tables sorted by F1 desc ─────────────────────
    current_anomaly = None
    for _, row in df.iterrows():
        if row["anomaly"] != current_anomaly:
            current_anomaly = row["anomaly"]
            lines.append(f"\n  ══ {current_anomaly} anomalies {'═' * (100 - len(current_anomaly))}")
            lines.append(HDR)
            lines.append("  " + SEP)
        lines.append(_row_line(row))

    # ── Section 2: per-arch tables sorted by F1 desc ────────────────────────
    arches_present = [a for a in _ARCH_ORDER if a in df["arch"].values]
    for a in df["arch"].unique():
        if a not in arches_present:
            arches_present.append(a)

    for arch in arches_present:
        arch_df = (df[df["arch"] == arch]
                   .sort_values("f1", ascending=False, na_position="last"))
        if arch_df.empty:
            continue
        title = f"{arch} — all anomaly types"
        lines.append(f"\n  ══ {title} {'═' * max(0, 100 - len(title))}")
        lines.append(HDR)
        lines.append("  " + SEP)
        for _, row in arch_df.iterrows():
            lines.append(_row_line(row))

    # ── Section 3: per-arch × anomaly seed detail tables ────────────────────
    def _mean_std_str(vals) -> str:
        clean = [v for v in vals if pd.notna(v) and np.isfinite(v)]
        if not clean:
            return "   n/a  "
        mu = float(np.mean(clean))
        sd = float(np.std(clean)) if len(clean) > 1 else 0.0
        return f"{mu:.3f}±{sd:.3f}"

    SEED_SEP = "─" * 80
    SEED_HDR = (f"  {'variant':<18} {'seed':>5}"
                f"  {'F1':>10} {'AUC':>10} {'Prec':>10} {'Rec':>10}")

    for anomaly in KNOWN_ANOMALY_TYPES:
        anom_df = df[df["anomaly"] == anomaly]
        if anom_df.empty:
            continue
        for arch in arches_present:
            cell = anom_df[anom_df["arch"] == arch]
            if cell.empty:
                continue

            title = f"{arch} × {anomaly} — per-seed detail"
            lines.append(f"\n  ══ {title} {'═' * max(0, 100 - len(title))}")
            lines.append(SEED_HDR)
            lines.append("  " + SEED_SEP)

            # Determine variant ordering (numeric where possible)
            variants = list(dict.fromkeys(
                v for v in cell.sort_values(
                    ["_vn" if "_vn" in cell.columns else "variant", "variant"],
                    na_position="last",
                ).get("variant", cell["variant"])
            ))
            # fallback: sort by numeric then alpha
            def _vsort(v):
                try:    return (0, float(v))
                except: return (1, v)
            variants = sorted(set(cell["variant"].tolist()), key=_vsort)

            for variant in variants:
                vdf = cell[cell["variant"] == variant].sort_values("seed")
                seed_f1s   = vdf["f1"].tolist()
                seed_aucs  = vdf["auc"].tolist()
                seed_precs = vdf["precision"].tolist()
                seed_recs  = vdf["recall"].tolist()

                for _, srow in vdf.iterrows():
                    lines.append(
                        f"  {str(srow['variant']):<18} {int(srow['seed'] or 0):>5}"
                        f"  {_fmt(srow['f1']):>10} {_fmt(srow['auc']):>10}"
                        f" {_fmt(srow['precision']):>10} {_fmt(srow['recall']):>10}"
                    )

                # mean ± std row
                lines.append(
                    f"  {'':18} {'avg':>5}"
                    f"  {_mean_std_str(seed_f1s):>10}"
                    f" {_mean_std_str(seed_aucs):>10}"
                    f" {_mean_std_str(seed_precs):>10}"
                    f" {_mean_std_str(seed_recs):>10}"
                )
                lines.append("  " + "·" * 60)

    summary = "\n".join(lines)
    txt_out = out_dir / "performance_summary.txt"
    txt_out.write_text(summary + "\n")
    print(f"  saved {txt_out.name}")
    print(summary)


def _cfg_get(cfg: dict, *keys, default=None):
    """Safe nested dict access: _cfg_get(cfg, 'model', 'pool_latent', default=None)."""
    v = cfg
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
        if v is default:
            return default
    return v


def save_experiments_table(all_runs: list[dict], out_dir: Path):
    """
    Write experiments_all.csv — one row per run with every design-choice column
    extracted from _suite_config.yaml.  Designed for statistical analysis:
    pd.read_csv → regression / ANOVA / feature importance.

    Merge results from different seeds / sessions with pd.concat([df1, df2]).
    The `seed` column distinguishes them.

    Paper: George & McCulloch (1993) for variable selection motivation;
           Hastie et al. "Elements of Statistical Learning" Ch. 3 for regression analysis.
    """
    rows = []
    for r in all_runs:
        cfg = r.get("_suite_cfg", {})
        cm  = cfg.get("_compare_meta", {})
        m   = r.get("log_metrics", {})
        p   = r.get("perf", {})

        # ── routing / identity ────────────────────────────────────────────────
        seed_raw = _cfg_get(cfg, "common", "seed", default=None)
        # also try to infer from trial_n (set from seed_N directory name)
        seed = seed_raw if seed_raw is not None else r.get("trial_n", 0)

        # path-based rq / subsuite: derive from the run directory relative to
        # any parent — best-effort; compare_meta is authoritative when present
        d    = r["dir"]
        parts = d.parts

        rows.append({
            # ── routing ───────────────────────────────────────────────────────
            "rq":              cm.get("rq",      r.get("_rq", "")),
            "subsuite":        cm.get("subsuite", ""),
            "seed":            seed,
            "run_path":        str(d),

            # ── identity ──────────────────────────────────────────────────────
            "run_name":        r["run_name"],
            "arch":            r["arch"],
            "anomaly_type":    r["anomaly"],
            "section":         cm.get("section", ""),
            "variant":         cm.get("variant", r.get("row_key", "")),

            # ── architecture choices ──────────────────────────────────────────
            "pool_latent":     _cfg_get(cfg, "model", "pool_latent"),
            "pooling_method":  _cfg_get(cfg, "model", "pooling"),
            "kl_mode":         _cfg_get(cfg, "model", "kl_mode"),
            "divergence_kind": _cfg_get(cfg, "model", "divergence_kind", default="kl"),
            "recon_loss":      _cfg_get(cfg, "model", "recon_loss_kind"),
            "training_loss":   _cfg_get(cfg, "training", "training_loss"),
            "norm":            _cfg_get(cfg, "model", "norm"),
            "flow_type":       _cfg_get(cfg, "model", "flow_type"),
            "flow_num_steps":  _cfg_get(cfg, "model", "flow_num_steps", default=0),
            "use_multiscale":  _cfg_get(cfg, "model", "use_multiscale", default=False),
            "use_pos_enc":     _cfg_get(cfg, "model", "use_pos_enc", default=False),
            "latent_dim":      _cfg_get(cfg, "model", "latent_dim"),
            "weight_init":     _cfg_get(cfg, "model", "weight_init"),

            # ── streaming choices ─────────────────────────────────────────────
            "stream_mode":     _cfg_get(cfg, "stream", "mode"),
            "seq_len":         _cfg_get(cfg, "stream", "seq_len"),
            "stride":          _cfg_get(cfg, "stream", "stride"),
            "lr_online":       _cfg_get(cfg, "training", "lr_online"),
            "update_if_score_below": _cfg_get(cfg, "training", "update_if_score_below"),

            # ── feature engineering ───────────────────────────────────────────
            "cyclic_time_features": _cfg_get(cfg, "data", "args", "cyclic_time_features",
                                             default=False),

            # ── training hyperparameters ──────────────────────────────────────
            "elbo_beta":         _cfg_get(cfg, "training", "elbo_beta"),
            "kl_anneal_epochs":  _cfg_get(cfg, "training", "kl_anneal_epochs"),
            "kl_auto_scale":     _cfg_get(cfg, "model", "kl_auto_scale", default=False),
            "prior_variance":    _cfg_get(cfg, "model", "prior_variance"),
            "free_bits":         _cfg_get(cfg, "model", "free_bits"),
            "clamp_logvar":      _cfg_get(cfg, "model", "clamp_logvar"),
            "clamp_bound_lo":    (_cfg_get(cfg, "model", "clamp_bounds") or [None, None])[0],
            "clamp_bound_hi":    (_cfg_get(cfg, "model", "clamp_bounds") or [None, None])[1],

            # ── outcome metrics at the configured live threshold ──────────────
            "live_thr":        m.get("ensemble_thr"),
            "f1":              m.get("f1"),
            "auc":             m.get("auc"),
            "ap":              m.get("ap"),
            "precision":       m.get("precision"),
            "recall":          m.get("recall"),
            "accuracy":        m.get("accuracy"),
            "tp":              m.get("tp"),
            "fp":              m.get("fp"),
            "tn":              m.get("tn"),
            "fn":              m.get("fn"),
            # post-hoc optimal thresholds (reference only)
            "opt_f1_thr":      m.get("cfg_thr"),
            "opt_f1":          m.get("best_f1"),
            "opt_f1_precision": m.get("best_precision"),
            "opt_f1_recall":   m.get("best_recall"),
            # ── cost / runtime ────────────────────────────────────────────────
            "total_s":                p.get("total_s"),
            "warmup_s":               p.get("warmup_s"),
            "stream_s":               p.get("stream_s"),
            "stream_steps_per_sec":   p.get("stream_steps_per_sec"),
            "warmup_samples_per_sec": p.get("warmup_samples_per_sec"),
            "step_ms_mean":           p.get("step_ms_mean"),
            "step_ms_p99":            p.get("step_ms_p99"),
            "gpu_peak_mb":            p.get("gpu_peak_mb"),
            "gpu_reserved_mb":        p.get("gpu_reserved_mb"),
        })

    df = pd.DataFrame(rows)
    csv_out = out_dir / "experiments_all.csv"
    df.to_csv(csv_out, index=False, float_format="%.6f")
    print(f"  saved {csv_out.name}  ({len(df)} rows)")


def save_performance_summary_table(all_runs: list[dict], out_dir: Path) -> None:
    """
    Aggregated performance summary: one row per arch × anomaly_type × section × variant,
    with mean ± std across seeds.  Saved to performance_summary.csv.

    Columns: arch, anomaly_type, section, variant, n_seeds,
             f1_{mean,std}, auc_{mean,std}, precision_{mean,std}, recall_{mean,std},
             step_ms_mean, gpu_peak_mb.
    """
    from collections import defaultdict

    def _variant_of(r: dict) -> str:
        rk = r.get("row_key", "")
        return rk.split("=")[-1] if "=" in rk else rk

    def _mean_std(values):
        vals = [float(v) for v in values if v is not None]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)

    groups: dict[tuple, list] = defaultdict(list)
    for r in all_runs:
        key = (r["arch"], r["anomaly"], r.get("section", ""), _variant_of(r))
        groups[key].append(r)

    rows = []
    for (arch, anomaly, section, variant), grp in groups.items():
        m_all = [r.get("log_metrics", {}) for r in grp]
        p_all = [r.get("perf", {}) for r in grp]

        f1_m,   f1_s   = _mean_std([m.get("f1")        for m in m_all])
        auc_m,  auc_s  = _mean_std([m.get("auc")       for m in m_all])
        pre_m,  pre_s  = _mean_std([m.get("precision")  for m in m_all])
        rec_m,  rec_s  = _mean_std([m.get("recall")     for m in m_all])
        sms_m,  _      = _mean_std([p.get("step_ms_mean") for p in p_all])
        gpu_m,  _      = _mean_std([p.get("gpu_peak_mb")  for p in p_all])

        rows.append({
            "arch":            arch,
            "anomaly_type":    anomaly,
            "section":         section,
            "variant":         variant,
            "n_seeds":         len(grp),
            "f1_mean":         f1_m,
            "f1_std":          f1_s,
            "auc_mean":        auc_m,
            "auc_std":         auc_s,
            "precision_mean":  pre_m,
            "precision_std":   pre_s,
            "recall_mean":     rec_m,
            "recall_std":      rec_s,
            "step_ms_mean":    sms_m,
            "gpu_peak_mb":     gpu_m,
        })

    df = pd.DataFrame(rows)
    anom_ord = {a: i for i, a in enumerate(KNOWN_ANOMALY_TYPES)}
    arch_ord = {a: i for i, a in enumerate(_ARCH_ORDER)}
    df["_ao"] = df["anomaly_type"].map(anom_ord).fillna(99)
    df["_ar"] = df["arch"].apply(lambda a: arch_ord.get(_base_arch(a), 99))
    df["_vn"] = pd.to_numeric(df["variant"], errors="coerce")
    df = (df.sort_values(["_ao", "_ar", "section", "_vn", "variant"])
            .drop(columns=["_ao", "_ar", "_vn"])
            .reset_index(drop=True))

    out = out_dir / "performance_summary.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"  saved {out.name}  ({len(df)} rows)")


# ─── plot: cost / efficiency ─────────────────────────────────────────────────

def plot_cost_efficiency(runs: list[dict], out_dir: Path) -> None:
    """
    Three-panel cost overview saved to cost_efficiency.png:

      Panel 1 — Runtime breakdown bars (warmup + streaming) per run, sorted by total time
      Panel 2 — Streaming throughput (steps/sec) per run
      Panel 3 — GPU peak memory per run
      Panel 4 — F1 vs streaming steps/sec scatter (speed-accuracy tradeoff)

    Runs without _perf_summary.json are silently skipped from panels that need that data.
    """
    runs_sorted = sorted(runs, key=_run_sort_key)
    perf_runs   = [r for r in runs_sorted if r.get("perf")]

    if not perf_runs:
        return

    labels     = [r["short_label"] for r in perf_runs]
    warmup_s   = [r["perf"].get("warmup_s",  0) or 0 for r in perf_runs]
    stream_s   = [r["perf"].get("stream_s",  0) or 0 for r in perf_runs]
    sps        = [r["perf"].get("stream_steps_per_sec", float("nan")) for r in perf_runs]
    gpu_mb     = [r["perf"].get("gpu_peak_mb",          float("nan")) for r in perf_runs]
    f1_vals    = [r.get("log_metrics", {}).get("f1",    float("nan")) for r in perf_runs]
    archs      = [r["arch"] for r in perf_runs]
    clrs       = [_arch_color(a) for a in archs]
    xs         = list(range(len(perf_runs)))

    fig, axes = plt.subplots(4, 1, figsize=(max(10, 1.3 * len(perf_runs)), 18))
    fig.suptitle("Cost / efficiency overview", fontsize=13)

    # ── Panel 1: runtime breakdown ────────────────────────────────────────────
    ax = axes[0]
    ax.bar(xs, warmup_s, color="#4c72b0", label="warmup")
    ax.bar(xs, stream_s, bottom=warmup_s, color="#dd8452", label="streaming")
    for i, (w, s) in enumerate(zip(warmup_s, stream_s)):
        total = w + s
        if total > 0:
            ax.text(i, total * 1.01, f"{total:.0f}s", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Wall time (s)", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([])
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # ── Panel 2: throughput ────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(xs, sps, color=clrs, edgecolor="white", lw=0.5)
    _bar_label(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("Stream steps / s", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([])
    ax.grid(axis="y", alpha=0.25)

    # ── Panel 3: GPU peak memory ──────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(xs, gpu_mb, color=clrs, edgecolor="white", lw=0.5)
    _bar_label(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("GPU peak mem (MB)", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # ── Panel 4: F1 vs throughput scatter ────────────────────────────────────
    ax = axes[3]
    for arch in sorted(set(archs)):
        mask = [a == arch for a in archs]
        ax.scatter(
            [sps[i] for i, m in enumerate(mask) if m],
            [f1_vals[i] for i, m in enumerate(mask) if m],
            color=_arch_color(arch), s=70, label=arch, zorder=3,
            edgecolors="white", linewidths=0.5,
        )
    for i, (x_v, y_v, lbl) in enumerate(zip(sps, f1_vals, labels)):
        if np.isfinite(x_v) and np.isfinite(y_v):
            ax.annotate(lbl.replace("\n", "/"), (x_v, y_v),
                        textcoords="offset points", xytext=(4, 4), fontsize=6, alpha=0.8)
    ax.set_xlabel("Streaming throughput (steps/s)", fontsize=10)
    ax.set_ylabel("F1", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = out_dir / "cost_efficiency.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/{out.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-comparison report for a regression session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__).strip(),
    )
    parser.add_argument("session", help="Session dir (or parent dir to pick latest session)")
    parser.add_argument("--pca",    action="store_true", help="Generate latent PCA grids (heavier)")
    parser.add_argument("--sample", type=int, default=100,
                        help="Sample every Nth row for metric distributions (default: 100)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Exclude runs whose name contains any of these substrings")
    parser.add_argument("--filter-beta", action="store_true",
                        help="Exclude runs where do_kl_normalize=False but elbo_beta>=0.5 "
                             "(stale runs with wrong KL scale)")
    parser.add_argument("--recursive", action="store_true",
                        help="Also scan one level of sub-directories for trial runs. "
                             "Used by run_regression when sub-suites have their own "
                             "output directories under the session root.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write cross_compare outputs into "
                             "(default: <session_dir>/cross_compare/)")
    args = parser.parse_args()

    session_dir = resolve_session(Path(args.session))
    out_dir = Path(args.output_dir) if args.output_dir else session_dir / "cross_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cross] session : {session_dir}")
    print(f"[cross] output  : {out_dir}")

    # ── discover runs ────────────────────────────────────────────────────────
    runs = discover_runs(session_dir, recursive=args.recursive)
    if not runs:
        sys.exit("[cross] No completed trial runs found.")

    # apply --exclude name filters
    if args.exclude:
        before = len(runs)
        runs = [r for r in runs if not any(pat in r["run_name"] for pat in args.exclude)]
        print(f"[cross] --exclude: removed {before - len(runs)} run(s)")

    print(f"[cross] {len(runs)} run(s) found:")
    for r in runs:
        print(f"  {r['run_name']:50}  arch={r['arch']}  anomaly={r['anomaly']}  "
              f"cyclic={r['cyclic']}  mode={r['mode']}")

    # ── load data per run ────────────────────────────────────────────────────
    print("\n[cross] Loading run data ...")
    for r in runs:
        print(f"  {r['run_name']} ...", end=" ", flush=True)
        r["log_metrics"]  = load_log_metrics(r["dir"])
        r["perf"]         = load_perf_summary(r["dir"])
        try:
            df_l, df_m, mnames = load_predictions(r["dir"], metric_sample_every=args.sample)
        except Exception as e:
            print(f"WARN: {e}")
            df_l = df_m = None
            mnames = []

        r["df_light"]    = df_l
        r["df_metrics"]  = df_m
        r["metric_names"] = mnames

        if df_l is not None:
            # AUC from scores is more accurate than the log's conf_roc approximation.
            r["log_metrics"]["auc"] = compute_auc(df_l)
            # Exact TP/FP/TN/FN at the configured threshold: read the confusion
            # column written by the live system (overrides log-derived approximations).
            vc = df_l["confusion"].value_counts()
            tp = int(vc.get("TP", 0)); fp = int(vc.get("FP", 0))
            tn = int(vc.get("TN", 0)); fn = int(vc.get("FN", 0))
            r["log_metrics"].update(tp=tp, fp=fp, tn=tn, fn=fn)
            n_pos = tp + fn
            n_neg = tn + fp
            if n_pos > 0:
                r["log_metrics"]["recall"]    = tp / n_pos
                r["log_metrics"]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r["log_metrics"]["f1"] = (
                    2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                )
                r["log_metrics"]["accuracy"] = (tp + tn) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
            # Optimal threshold references (setdefault: only fill if not already set by log)
            for key, val in compute_threshold_recommendations(df_l).items():
                r["log_metrics"].setdefault(key, val)
        else:
            # No prediction file — keep F1/AUC/TP/FP/TN/FN parsed from run.log
            print(f"  (no predictions file; using log-derived metrics)", end=" ")
        print("ok")

    # apply --filter-beta: drop runs where do_kl_normalize=False but elbo_beta>=0.5
    if args.filter_beta:
        before = len(runs)
        def _beta_ok(r):
            lm = r.get("log_metrics", {})
            kl_norm = lm.get("do_kl_normalize", True)   # default: assume norm (safe)
            beta    = lm.get("elbo_beta", 0.0)
            return kl_norm or beta < 0.5                 # keep if norm=True OR beta is small
        runs = [r for r in runs if _beta_ok(r)]
        print(f"[cross] --filter-beta: removed {before - len(runs)} run(s) with "
              f"do_kl_normalize=False + elbo_beta>=0.5")

    # ── outputs ──────────────────────────────────────────────────────────────
    print("\n[cross] Writing outputs ...")

    # summary tables live at the session level (not per-anomaly)
    save_performance_table(runs, out_dir)
    save_experiments_table(runs, out_dir)
    save_performance_summary_table(runs, out_dir)
    plot_cost_efficiency(runs, out_dir)

    anomalies_present = [a for a in KNOWN_ANOMALY_TYPES
                         if any(r["anomaly"] == a for r in runs)]
    for anomaly in anomalies_present:
        group    = sorted([r for r in runs if r["anomaly"] == anomaly], key=_run_sort_key)
        anom_dir = out_dir / anomaly.lower()
        anom_dir.mkdir(exist_ok=True)

        plot_arch_comparison_bars(group, anomaly, anom_dir)
        plot_section_lines(group, anomaly, anom_dir)
        plot_seed_stability_heatmap(group, anomaly, anom_dir)
        plot_confusion_grid(group, anomaly, anom_dir)
        plot_metric_distributions(group, anomaly, anom_dir)
        plot_score_histograms(group, anomaly, anom_dir)
        plot_cost_efficiency(group, anom_dir)
        if args.pca:
            plot_latent_pca_grid(group, anomaly, anom_dir)

    print(f"\n[cross] Done — {out_dir}")


if __name__ == "__main__":
    main()
