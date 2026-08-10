from __future__ import annotations
import os, sys, json, time, csv
import math
import traceback
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Union, List, Tuple
import matplotlib.pyplot as plt

from src.analysis_plots import *


class Trial_Paths:

    def __init__(self, trial_dir, standalone: bool = False):

        self.trial_dir = trial_dir
        self.standalone = standalone

        # Split CSVs
        self.predictions_csv       = os.path.join(self.trial_dir, "trial_predictions.csv")
        self.humanFriendly_log     = os.path.join(self.trial_dir, "human_friendly.log")

        # Optional figures
        self.conf_hist_png_path    = os.path.join(self.trial_dir, "confusion_histograms.png")
        self.conf_stack_hist_png_path = os.path.join(self.trial_dir, "confusion_stack_histograms.png")
        self.plot_graphs_path        = os.path.join(self.trial_dir, "metric_plot_graphs.png")
        self.colored_graph_png_path        = os.path.join(self.trial_dir, "metric_colored_graphs.png")
        self.plot_scores_vs_thresholds      = os.path.join(self.trial_dir, "scores_vs_thresholds.png")
        self.plot_latent_pca_2d      = os.path.join(self.trial_dir, "latent_pca_2d.png")
        self.plot_latent_pca_3d      = os.path.join(self.trial_dir, "latent_pca_3d.png")
        self.plot_latent_pca_3d_html = os.path.join(self.trial_dir, "latent_pca_3d.html")
        self.plot_latent_pca_3d_confusion      = os.path.join(self.trial_dir, "latent_pca_3d_confusion.png")
        self.plot_latent_pca_3d_confusion_html = os.path.join(self.trial_dir, "latent_pca_3d_confusion.html")
        self.plot_latent_pca_3d_month_html     = os.path.join(self.trial_dir, "latent_pca_3d_month.html")
        # Manifold projections (generated only when enabled in plot_settings)
        self.latent_umap_month_html  = os.path.join(self.trial_dir, "latent_umap_month.html")
        self.latent_umap_hod_html    = os.path.join(self.trial_dir, "latent_umap_hod.html")
        self.latent_umap_doy_html    = os.path.join(self.trial_dir, "latent_umap_doy.html")
        self.latent_isomap_month_html = os.path.join(self.trial_dir, "latent_isomap_month.html")
        self.latent_isomap_hod_html  = os.path.join(self.trial_dir, "latent_isomap_hod.html")
        self.latent_isomap_doy_html  = os.path.join(self.trial_dir, "latent_isomap_doy.html")
        self.latent_tsne_month_html  = os.path.join(self.trial_dir, "latent_tsne_month.html")
        self.latent_tsne_hod_html    = os.path.join(self.trial_dir, "latent_tsne_hod.html")
        self.latent_tsne_doy_html    = os.path.join(self.trial_dir, "latent_tsne_doy.html")
        self.latent_mu_npy                     = os.path.join(self.trial_dir, "latent_mu.npy")
        self.latent_timestamps_npy             = os.path.join(self.trial_dir, "latent_timestamps.npy")
        self.latent_labels_npy                 = os.path.join(self.trial_dir, "latent_labels.npy")
        self.latent_scores_npy                 = os.path.join(self.trial_dir, "latent_scores.npy")
        self.plot_latent_norm_over_time      = os.path.join(self.trial_dir, "latent_norm_over_time.png")
        self.plot_latent_mahalanobis      = os.path.join(self.trial_dir, "latent_mahalanobis.png")
        self.metric_confidence_heatmap        = os.path.join(self.trial_dir, "metric_confidence_heatmap.png")
        self.decoder_sensitivity_png = os.path.join(self.trial_dir, "decoder_sensitivity.png")


DEFAULT_DISABLED_OUTPUTS: frozenset = frozenset({
    # plots disabled by default — enable individually in YAML plot_settings
    "latent_mahalanobis.png",
    "latent_norm_over_time.png",
    "latent_pca_2d.png",
    "latent_pca_3d",
    "latent_pca_3d_confusion",
    "latent_pca_3d_month",
    "scores_vs_thresholds.png",
    "metric_plot_graphs.png",
    "confusion_stack_histograms.png",
    "metric_colored_graphs.png",
    "metric_confidence_heatmap.png",
    "decoder_sensitivity.png",
    # CSV/JSON outputs disabled by default
})

class Verbosity(IntEnum):
    SILENT=0; ERRORS=1; INFO_LOW=2; INFO_HIGH=3; DEBUG=4

VERBOSITY_MAP = {
    "SILENT": Verbosity.SILENT,
    "ERRORS": Verbosity.ERRORS,
    "INFO_LOW":   Verbosity.INFO_LOW,
    "INFO_HIGH":   Verbosity.INFO_HIGH,
    "DEBUG":  Verbosity.DEBUG,
}


def _now_ts() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")

def _as_text(x: Union[str, Callable[[], str]]) -> str: return x() if callable(x) else str(x)

def _as_dict(x: Union[Dict[str, Any], Callable[[], Dict[str, Any]]]) -> Dict[str, Any]: return x() if callable(x) else dict(x)

def _timestamp_dirname() -> str:
    # e.g. 2025-11-07_1420
    return time.strftime("%Y-%m-%d_%H%M")

def resolve_unique_dir(base_path: str) -> str:
    """
    If base_path doesn't exist: create & return it.
    If exists: append _1, _2, ... (like exp -> exp_1 -> exp_2 ...)
    """
    base_path = base_path.rstrip("/\\")
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return base_path
    k = 1
    while True:
        cand = f"{base_path}_{k}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=True)
            return cand
        k += 1

# --- helper: zero-pad trial folder names ---
def _trial_dir_name(trial_number: int) -> str:
    return f"trial_{trial_number:03d}"

class Logger:
    """
    - Text lines (error/info/debug)
    - JSONL events (event / event_if)
    - Fan-out 'sinks' for structured events (e.g., write CSV rows)
    """
    def __init__(self, paths: Trial_Paths, level: Verbosity = Verbosity.INFO_LOW, stream=None):
        
        self.level = level
        self.stream = stream or sys.stdout
        self.paths = paths
        self._sinks: List[Callable[[str, Dict[str, Any]], None]] = []  # (kind, payload_dict) -> None
        self._visual_sinks: List[Callable[[str, Dict[str, Any]], None]] = []  # (kind, payload_dict) -> None

        self.main_log_path = None
        self.suffix = None

        # bind methods to either real implementation or no-op
        self.error = self._log_error if level >= Verbosity.ERRORS else self._noop
        self.info_low  = self._log_info_low  if level >= Verbosity.INFO_LOW   else self._noop
        self.info_high  = self._log_info_high  if level >= Verbosity.INFO_HIGH   else self._noop
        self.debug = self._log_debug if level >= Verbosity.DEBUG  else self._noop

        # structured events: keep callable available unless logger is fully SILENT
        self.event     = self._event if level > Verbosity.SILENT else self._noop_event
        self.event_if  = self._event_if if level > Verbosity.SILENT else self._noop_event

    # ------------- public API -------------
    
    def add_event_sink(self, sink_fn: Callable[[str, Dict[str, Any]], None]):
        self._sinks.append(sink_fn)

    def add_visualisation_sink(self, sink_fn: Callable[[str, Dict[str, Any], Trial_Paths], None]):
        self._visual_sinks.append(sink_fn)

    def set_level(self, level: Verbosity):
        self.__init__(level, self.stream)
    
    # simple text logs (support lazy message via callable)
    def _log_error(self, msg: Union[str, Callable[[], str]]):
        self.stream.write(f"{_now_ts()} [ERROR] {_as_text(msg)}\n")

    def _log_info_low(self, msg: Union[str, Callable[[], str]]):
        self.stream.write(f"{_now_ts()} [INFO_LOW ] {_as_text(msg)}\n")

    def _log_info_high(self, msg: Union[str, Callable[[], str]]):
        self.stream.write(f"{_now_ts()} [INFO_HIGH ] {_as_text(msg)}\n")

    def _log_debug(self, msg: Union[str, Callable[[], str]]):
        self.stream.write(f"{_now_ts()} [DEBUG] {_as_text(msg)}\n")

    # ---- JSONL event ----
    def _event(self, level: Verbosity, kind: str,
               payload: Union[Dict[str, Any], Callable[[], Dict[str, Any]]]) -> None:
        
        if level > self.level:
            return
       
        body = _as_dict(payload)
        # rec = {"ts": _now_ts(), "level": level.name, "kind": kind, "payload": body}
        # self.stream.write(json.dumps(rec) + "\n")
        # fan-out to sinks
        for sink in self._sinks:
            try:
                sink(kind, body)
            except Exception as e:
                # never crash on sinks
                self.stream.write(f"{_now_ts()} [ERROR] sink failed: {e}\n")

    def _event_if(self, cond: bool, level: Verbosity, kind: str,
                  payload: Union[Dict[str, Any], Callable[[], Dict[str, Any]]]) -> None:
        if cond:
            self._event(level, kind, payload)

    def visual_event(self,  kind: str, payload: Union[Dict[str, Any], Callable[[], Dict[str, Any]]]) -> None:
        
        body = _as_dict(payload)
        # rec = {"ts": _now_ts(), "level": level.name, "kind": kind, "payload": body}
        # self.stream.write(json.dumps(rec) + "\n")
        # fan-out to sinks
        for v_sink in self._visual_sinks:
            try:
                v_sink(kind, body, self.paths)
            except Exception as e:
                tb = traceback.format_exc()
                self.stream.write(f"{_now_ts()} [ERROR] visualise sink {str(v_sink)} failed: {tb}\n")
                print(f"{_now_ts()} [ERROR] visualise sink {str(v_sink)} failed: {tb}\n")
            
    # ------------- no-ops -------------
    def _noop(self, *_a, **_k): pass
    def _noop_event(self, *_a, **_k): pass


class PredictionCsvSink:
    """
    Writes per-step predictions to trial_predictions.csv
    Uses dataset timestamp from payload.
    Expected payload keys: ts, step, score, th, pred, label (label may be -1 when unknown)
    """
    FIELDS = ["ts_start", "ts_end", "step", "confusion",
              "label", "final_pred", "trained", "final_conf", "metrics"             
            ]

    def __init__(self, csv_path: str):
        # Always truncate: a trial_dir can be reused across independent reruns
        # (same run_name/seed rerun later), and appending onto a stale file from
        # a previous run silently corrupts every downstream count (fp/tn/n) —
        # this is not a crash-resume mechanism, every trial starts fresh.
        self.csv_path = csv_path
        self.stream = open(self.csv_path, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.stream.write("# " + ",".join(self.FIELDS) + "\n")

    def __call__(self, kind: str, payload: Dict[str, Any]):
        if kind != "pred_outcome":
            return
        row = {
            "ts_start": payload.get("ts_start"),
            "ts_end":   payload.get("ts_end"),
            "step":     payload.get("step"),
            "confusion":     payload.get("confusion"),
            "label":    payload.get("label", -1),            
            "final_pred":     payload.get("final_pred"),
            "trained":    payload.get("trained"),
            "final_conf" :    payload.get("final_conf"),
            "metrics"    :    payload.get("metrics")
        }
        self.writer.writerow(row)

    def close(self):
        try: self.stream.close()
        except: pass


_CC_CONF_ENC: dict[str, int] = {"TP": 0, "FP": 1, "TN": 2, "FN": 3}


class CCSummarySink:
    """
    Accumulates lightweight prediction data in memory during a run and writes
    _cc_summary.npz on close().  This file lets cross_compare.py produce all
    plots even when trial_predictions.csv has been deleted to save disk space.

    Stored arrays:
      labels           — int8, one entry per streaming step
      final_conf       — float32, one entry per step
      confusion_enc    — int8 (0=TP, 1=FP, 2=TN, 3=FN, -1=unknown)
      metric_names     — string array of metric names
      metric_step_idx  — int32, which step indices were sampled for metrics
      msc_{name}       — float32, per-metric score at each sampled step
    """

    def __init__(self, trial_dir: str, metric_sample_every: int = 50):
        self.trial_dir = trial_dir
        self.metric_sample_every = metric_sample_every
        self._labels: list[int] = []
        self._final_conf: list[float] = []
        self._confusion_enc: list[int] = []
        self._metric_names: list[str] = []
        self._metric_scores: dict[str, list[float]] = {}
        self._metric_step_idx: list[int] = []
        self._step_count = 0

    def __call__(self, kind: str, payload: Dict[str, Any]):
        if kind != "pred_outcome":
            return
        self._labels.append(int(payload.get("label", -1)))
        self._final_conf.append(float(payload.get("final_conf") or float("nan")))
        self._confusion_enc.append(_CC_CONF_ENC.get(str(payload.get("confusion", "")), -1))

        metrics = payload.get("metrics") or {}

        if not self._metric_names and metrics:
            self._metric_names = [k for k in metrics if k != "final"]
            self._metric_scores = {k: [] for k in self._metric_names}

        if self._step_count % self.metric_sample_every == 0 and self._metric_names:
            self._metric_step_idx.append(self._step_count)
            for mname in self._metric_names:
                entry = metrics.get(mname) or {}
                score = float(entry.get("score", float("nan"))) if isinstance(entry, dict) else float("nan")
                self._metric_scores[mname].append(score)

        self._step_count += 1

    def close(self):
        if not self._labels:
            return
        try:
            out = os.path.join(self.trial_dir, "_cc_summary.npz")
            arrays: dict = dict(
                labels=np.array(self._labels, dtype=np.int8),
                final_conf=np.array(self._final_conf, dtype=np.float32),
                confusion_enc=np.array(self._confusion_enc, dtype=np.int8),
                metric_names=np.array(self._metric_names),
                metric_step_idx=np.array(self._metric_step_idx, dtype=np.int32),
            )
            for mname, scores in self._metric_scores.items():
                arrays[f"msc_{mname}"] = np.array(scores, dtype=np.float32)
            np.savez_compressed(out, **arrays)
        except Exception:
            pass


class HumanFriendlyRunLog:
    """
    Writes per-step predictions to trial_predictions.csv
    Uses dataset timestamp from payload.
    Expected payload keys: ts, step, score, th, pred, label (label may be -1 when unknown)
    """
    FIELDS = ["ts_start", "ts_end", "step", "confusion",
              "final_conf", "metrics"             
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        need_header = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        self.stream = open(self.log_path, "a", newline="", buffering=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        if need_header:
            self.stream.write("# " + ",".join(self.FIELDS) + "\n")

    def __call__(self, kind: str, payload: Dict[str, Any]):
        if kind != "human_friendly":
            return
        row = {
            "ts_start": payload.get("ts_start"),
            "ts_end":   payload.get("ts_end"),
            "step":     payload.get("step"),
            "confusion":     payload.get("confusion"),
            "final_conf" :    payload.get("final_conf"),
            "metrics"    :    payload.get("metrics")
        }
        self.writer.writerow(row)

    def close(self):
        try: self.stream.close()
        except: pass

class TrialFilesObj:
    """
    Owns a single trial's files inside the parent run_dir:
      trial.log, trial_confusion.png
    Provides a Logger wired to trial.log.

    disabled_outputs: set of output names to skip. Recognised names:
      "latent_mahalanobis.png", "latent_norm_over_time.png",
      "latent_pca_2d.png", "latent_pca_3d", "latent_pca_3d_confusion",
      "latent_pca_3d_month", "latent_umap",
      "scores_vs_thresholds.png", "metric_plot_graphs.png",
    """
    def __init__(self, trial_dir: str, verbosity: Verbosity, use_this_logger: Logger = None,
                 standalone: bool = False, disabled_outputs: Optional[set] = None):

        self.trial_dir = trial_dir
        disabled_outputs = disabled_outputs or set()
        os.makedirs(self.trial_dir, exist_ok=True)

        self.paths=Trial_Paths(self.trial_dir, standalone=standalone)

        if use_this_logger is None :
            # Human-readable / JSONL text log
            self.log_path              = os.path.join(self.trial_dir, "trial.log")
            self._log_fh = open(self.log_path, "a", buffering=1)
            self.logger  = Logger(paths=self.paths, level=verbosity, stream=self._log_fh)
        else :
            self.logger = use_this_logger

        # Always point the logger at this trial's paths so visual_event()
        # picks up the correct standalone flag and file paths.
        self.logger.paths = self.paths

        # Attach event sinks (conditionally)

        self._sink_changes = None


        # pred_outcome (always enabled)
        self._sink_preds = PredictionCsvSink(self.paths.predictions_csv)
        self.logger.add_event_sink(self._sink_preds)

        # compact summary for cross_compare (always enabled; survives CSV deletion)
        self._sink_cc_summary = CCSummarySink(self.trial_dir)
        self.logger.add_event_sink(self._sink_cc_summary)

        # Attach visualisation sinks — only confusion histogram on by default
        self.logger.add_visualisation_sink(plt_confusions)
        if "confusion_stack_histograms.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_confusion_stacked)
        if "metric_plot_graphs.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_all_metrics_scores_over_time)
        if "metric_colored_graphs.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_all_metrics_scores_scatter_colored)
        if "metric_confidence_heatmap.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_all_metrics_heatmap)
        if "scores_vs_thresholds.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_metric_lines_scores_thresholds)
        if "latent_pca_2d.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_pca_2d)
        if "latent_pca_3d" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_pca_3d)
        if "latent_pca_3d_confusion" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_pca_3d_confusion)
        if "latent_pca_3d_month" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_pca_3d_month)
        if "latent_umap" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_umap)
        if "latent_norm_over_time.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_norm_over_time)
        if "latent_mahalanobis.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_latent_mahalanobis)
        if "decoder_sensitivity.png" not in disabled_outputs:
            self.logger.add_visualisation_sink(plot_decoder_sensitivity)

        


    def save_plots(self, tp, fp, tn, fn):

        plt.figure()
        labels = ["TP", "FP", "TN", "FN"]; counts = [tp, fp, tn, fn]
        plt.bar(labels, counts); plt.title("Prediction counts")
        plt.savefig(self.conf_png_path, bbox_inches="tight"); plt.close()

    
    def close(self):
        for fh in [self._sink_changes, self._sink_preds,
                   self._sink_cc_summary]:
            if fh is not None:
                try: fh.close()
                except: pass
        try: self._log_fh.close()
        except: pass

class RunFilesObj:

    def __init__(self, run_dir: str, verbosity: Verbosity, standalone: bool = False,
                 disabled_outputs: Optional[set] = None):
        # (unchanged)
        self.run_dir = run_dir
        self._disabled_outputs = disabled_outputs or set()
        os.makedirs(self.run_dir, exist_ok=True)

        self.paths=Trial_Paths(self.run_dir)

        self.log_path        = os.path.join(self.run_dir, "run.log")
        self._log_fh = open(self.log_path, "a", buffering=1)
        self.logger  = Logger(self.paths, level=verbosity, stream=self._log_fh)

        # NEW: keep track of child trials to close them later
        self._open_trials: dict[int, TrialFilesObj] = {}


    # create or return a per-trial files bundle object
    def create_trial_files_obj(self, trial_number: int, standalone: bool=False) -> TrialFilesObj:
        if not standalone :
            if trial_number in self._open_trials:
                return self._open_trials[trial_number]
            tdir = os.path.join(self.run_dir, _trial_dir_name(trial_number))
        else:
            tdir = self.run_dir

        use_this_logger = self.logger if standalone else None
        tf = TrialFilesObj(tdir, self.logger.level, use_this_logger, standalone=standalone,
                           disabled_outputs=self._disabled_outputs)
        # also announce in the run.log for discoverability
        self.logger.info_low(lambda: f"Opened {tdir}")
        self._open_trials[trial_number] = tf
        return tf

    def getStandaloneLoggers(self) :
        return self._open_trials[0]
    
    def close(self):
        # close trials first
        for t in list(self._open_trials.values()):
            try: t.close()
            except: pass
        self._open_trials.clear()
        # then close run-level sinks/handles
        #except: pass
        try: self._log_fh.close()
        except: pass


# -------------------- Single entry to build everything --------------------
# logging_cfg:
#   - run_dir (optional): base path for a run directory.
#         If omitted → ./runs/<timestamp> (unique)
#   - verbosity: "SILENT"/"ERRORS"/"INFO_LOW"/"INFO_HIGH"/"DEBUG"
# Returns: RunFilesObj
def create_run_files_obj(logging_cfg: Dict[str, Any]) -> RunFilesObj :
    
    # Verbosity
    verb = logging_cfg.get("verbosity", "INFO_LOW")
    if isinstance(verb, str):
        verbosity = VERBOSITY_MAP.get(verb.strip().upper(), Verbosity.INFO_LOW)
    elif isinstance(verb, int):
        verbosity = Verbosity(verb)
    else:
        verbosity = Verbosity.INFO_LOW

    # Decide base run_dir
    base = logging_cfg.get("run_dir", None)
    if base in [None, "", "None", "null"]:
        base = os.path.join(".", "runs", _timestamp_dirname())  # e.g., ./runs/2025-11-07_1420
        # Auto-generated paths get a uniqueness suffix (_1, _2, ...) to avoid collisions.
        run_dir = resolve_unique_dir(base)
    else:
        # Explicitly specified path (e.g. set by regression runner): use as-is so
        # outputs land in the same directory as _suite_config.yaml.
        os.makedirs(base, exist_ok=True)
        run_dir = base

    disabled_outputs = DEFAULT_DISABLED_OUTPUTS | set(logging_cfg.get("disable_outputs", []) or [])
    run_files_obj = RunFilesObj(run_dir, verbosity, disabled_outputs=disabled_outputs)

    # Let the main log know where we write things
    run_files_obj.logger.info_low(lambda: f"Run directory: {run_dir}")
    run_files_obj.logger.info_low(lambda: "Files: run.log, run_confusion.png")

    return run_files_obj
