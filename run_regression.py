#!/usr/bin/env python3
"""
Regression suite runner.

Reads a suite YAML, generates all enabled level-combinations, deep-merges
each combination on top of a base config, and runs them in parallel across
available GPUs.

Usage (run from the project root):
    python run_regression.py path/to/suite.yaml
    python run_regression.py suite.yaml --dry-run
    python run_regression.py suite.yaml --gpus 0,1
    python run_regression.py suite.yaml --session runs/regression/era5_golden/session_20260320_011131
    python run_regression.py suite.yaml --session runs/regression/era5_golden/session_20260320_011131 --skip-existing

Background (survives window close):
    nohup python run_regression.py suite.yaml --session <session_dir> --skip-existing &
    tail -f <session_dir>/suite.log      # follow progress from any terminal
    cat  <session_dir>/suite_summary.txt # clean pass/fail/skip/disabled report

Config layering
---------------
Each run in `runs:` may specify a `config_layers` list of YAML fragment files
(paths relative to the suite file).  They are merged in order on top of the
base_config and below the run-level `config:` block:

    base_config  →  config_layers[0]  →  …  →  config_layers[-1]  →  config

include_suites
--------------
A suite may delegate to other suite YAMLs via:

    include_suites:
      - path/to/sub_suite_1.yaml
      - path/to/sub_suite_2.yaml

Runs from each included suite are merged into this session.  Paths are
relative to the including suite file.  Each sub-suite may have its own
base_config, run_module, and output_dir (the session output_dir always
comes from the top-level suite or --session).

compare metadata
----------------
Each run entry may include a `compare:` block consumed by cross_compare.py:

    - name: MLP_Group_norm_T
      compare:
        anomaly: Group
        arch: MLP
        variant: norm_T
        order: 1
      config_layers: ...

This metadata is embedded in the run's _suite_config.yaml under the key
`_compare_meta` so cross_compare can read it without needing the original
suite YAML.
"""

import os
import sys
import copy
import time
import yaml
import queue
import socket
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product

_CROSS_COMPARE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "cross_compare.py")

# ── session logger ─────────────────────────────────────────────────────────────
# Mirrors all [suite] output to both stdout and <session_dir>/suite.log so the
# session can be followed with `tail -f suite.log` when running in the background.

_log_fh = None
_log_lock = threading.Lock()


def log(msg: str, flush: bool = True) -> None:
    print(msg, flush=flush)
    if _log_fh is not None:
        with _log_lock:
            _log_fh.write(msg + "\n")
            if flush:
                _log_fh.flush()


# ── config helpers ─────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def gpu_free_mb(gpu_id) -> int:
    """Return free VRAM in MB for gpu_id, or a large sentinel if unknown."""
    if gpu_id is None:
        return 999_999
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(r.stdout.strip().splitlines()[0].strip())
    except Exception:
        return 999_999


def gpu_compute_mode(gpu_id) -> str:
    """Return the compute mode string for gpu_id, e.g. 'Default' or 'Exclusive_Process'."""
    if gpu_id is None:
        return "Default"
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=compute_mode", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip().splitlines()[0].strip()
    except Exception:
        return "Default"



def available_gpu_ids() -> list:
    """Detect GPUs via nvidia-smi — never imports torch, avoids CUDA context init."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        ids = [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip().isdigit()]
        return ids if ids else [None]
    except Exception:
        return [None]


def probe_gpu(gpu_id, mem_free_threshold_mb: int = 500) -> bool:
    """Return True if gpu_id is free to accept a new CUDA context."""
    if gpu_id is None:
        return True
    try:
        mem_result = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.free,compute_mode",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if mem_result.returncode != 0:
            return True
        parts = mem_result.stdout.strip().split(",")
        if len(parts) < 2:
            return True
        free_mb = int(parts[0].strip())
        compute_mode = parts[1].strip()
        if free_mb < mem_free_threshold_mb:
            return False
        if "Exclusive" in compute_mode:
            apps_result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_id}",
                 "--query-compute-apps=pid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if apps_result.returncode == 0 and apps_result.stdout.strip():
                return False
        return True
    except Exception:
        return True


# ── suite loading ──────────────────────────────────────────────────────────────

def _load_suite_file(suite_path: str) -> tuple[dict, str]:
    """Load a YAML suite file. Returns (suite_dict, suite_dir)."""
    suite_path = os.path.abspath(suite_path)
    suite_dir  = os.path.dirname(suite_path)
    with open(suite_path) as f:
        suite = yaml.safe_load(f) or {}
    return suite, suite_dir


def _collect_runs_from_suite(suite: dict, suite_dir: str) -> tuple[list, list]:
    """
    Returns (enabled_runs, disabled_runs).

    Each item is (run_name, config_override, meta) where meta contains:
      suite_dir     — directory of the YAML that defined this run (for path resolution)
      run_module    — optional per-run module override
      base_config   — optional per-run base config override
      config_layers — list of layer paths
      compare_meta  — dict from run's `compare:` block, if present
      extra_args    — list of extra CLI args
    """
    enabled  = []
    disabled = []

    _KNOWN_KEYS = {"name", "enabled", "run_module", "base_config",
                   "config_layers", "config", "compare"}

    for run in suite.get("runs", []):
        name = run["name"]
        meta = {"suite_dir": suite_dir}
        if "run_module"    in run: meta["run_module"]    = run["run_module"]
        if "base_config"   in run: meta["base_config"]   = run["base_config"]
        if "config_layers" in run: meta["config_layers"] = run["config_layers"]
        if "compare"       in run: meta["compare_meta"]  = run["compare"]
        extra = [str(run[k]) for k in run if k not in _KNOWN_KEYS]
        if extra:
            meta["extra_args"] = extra

        entry = (name, run.get("config", {}), meta)
        if run.get("enabled", True):
            enabled.append(entry)
        else:
            disabled.append(entry)

    # levels: Cartesian product (no compare metadata support here)
    active_levels = []
    for level in suite.get("levels", []):
        if not level.get("enabled", True):
            continue
        opts = [o for o in level.get("options", []) if o.get("enabled", True)]
        if opts:
            active_levels.append((level["name"], opts))

    if active_levels:
        for combo in product(*[opts for _, opts in active_levels]):
            cname    = "__".join(o["name"] for o in combo)
            override = {}
            for o in combo:
                override = deep_merge(override, o.get("config", {}))
            enabled.append((cname, override, {"suite_dir": suite_dir}))

    return enabled, disabled


def _recursive_collect(suite: dict, suite_dir: str, suite_path: str,
                        parent_rq: str = None) -> tuple[list, list]:
    """
    Recursively collect (enabled, disabled) runs from a suite and all its
    include_suites.  Handles arbitrarily deep include chains (q_all → q1_all →
    pool_latent) so a single master YAML can compose the whole experiment tree.

    Each run's meta dict is stamped with:
      subsuite_name — filename stem of the leaf YAML that owns the run
      rq            — research-question label (from suite's `rq:` field, or
                      inherited from the nearest ancestor that declares one)
    """
    enabled_all  = []
    disabled_all = []

    suite_rq      = suite.get("rq") or parent_rq
    suite_module  = suite.get("run_module")
    suite_base    = suite.get("base_config")
    this_name     = os.path.splitext(os.path.basename(suite_path))[0]

    # ── include_suites (recursive) ────────────────────────────────────────────
    for rel_path in suite.get("include_suites", []):
        sub_path = rel_path if os.path.isabs(rel_path) else os.path.join(suite_dir, rel_path)
        sub_suite, sub_dir = _load_suite_file(sub_path)
        sub_name  = os.path.splitext(os.path.basename(sub_path))[0]

        e, d = _recursive_collect(sub_suite, sub_dir, sub_path, parent_rq=suite_rq)

        # Propagate this suite's run_module / base_config downward only when
        # the run hasn't already inherited them from a closer ancestor.
        for _, _, meta in e + d:
            if suite_module and "run_module" not in meta:
                meta["run_module"] = suite_module
            if suite_base and "base_config" not in meta:
                meta["base_config"] = (
                    suite_base if os.path.isabs(suite_base)
                    else os.path.join(suite_dir, suite_base)
                )
            # subsuite_name and rq are set by the innermost YAML first;
            # setdefault ensures outer levels cannot overwrite them.
            meta.setdefault("subsuite_name", sub_name)
            meta.setdefault("rq", suite_rq)

        enabled_all.extend(e)
        disabled_all.extend(d)

    # ── direct runs defined in this file ─────────────────────────────────────
    e, d = _collect_runs_from_suite(suite, suite_dir)
    for _, _, meta in e + d:
        if suite_module and "run_module" not in meta:
            meta["run_module"] = suite_module
        if suite_base and "base_config" not in meta:
            meta["base_config"] = (
                suite_base if os.path.isabs(suite_base)
                else os.path.join(suite_dir, suite_base)
            )
        meta.setdefault("subsuite_name", this_name)
        meta.setdefault("rq", suite_rq)

    enabled_all.extend(e)
    disabled_all.extend(d)

    return enabled_all, disabled_all


def generate_runs(suite: dict, suite_dir: str, suite_path: str) -> tuple[list, list]:
    """
    Entry point: collect all (enabled, disabled) runs from the suite tree.
    Delegates to _recursive_collect so include_suites chains are fully resolved.
    """
    enabled, disabled = _recursive_collect(suite, suite_dir, suite_path)
    if not enabled and not disabled:
        enabled = [("default", {}, {"suite_dir": suite_dir})]
    return enabled, disabled


# ── single-run worker ──────────────────────────────────────────────────────────

def run_job(run_name: str, cfg: dict, run_module: str,
            run_dir: str, gpu_id, dry_run: bool, extra_args: list = None,
            threads: int = 1):
    """Execute a single run inside run_dir (already fully resolved by the caller)."""
    os.makedirs(run_dir, exist_ok=True)

    # Respect an explicit device in the config; only inject cuda:0 if not already set.
    cfg_device = cfg.get("common", {}).get("device")
    device_override = {"device": "cuda:0"} if (gpu_id is not None and not cfg_device) else {}
    cfg = deep_merge(cfg, {"common": {"logging": {"run_dir": run_dir}, **device_override}})

    yaml_path = os.path.join(run_dir, "_suite_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, width=10000)

    env = os.environ.copy()
    # Drop MPLBACKEND: when launched from a Jupyter cell, the kernel sets this to
    # matplotlib_inline's backend, which is only registered inside a running IPython
    # kernel. The subprocess below is a plain interpreter, so inheriting it makes
    # matplotlib crash on import with "not a valid value for backend".
    env.pop("MPLBACKEND", None)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # must be set before cuBLAS init (import torch)
    # Seed-dependent vars must be in the subprocess env before the interpreter starts —
    # setting them via os.environ inside the child process is too late.
    effective_seed = str(cfg.get("common", {}).get("seed", 42))
    env["PYTHONHASHSEED"] = effective_seed
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "-m", run_module, yaml_path] + (extra_args or [])
    tag = f"GPU={gpu_id}" if gpu_id is not None else "CPU"
    log(f"[suite] START  {run_name!r}  ({tag})")

    if dry_run:
        log(f"        cmd : {' '.join(cmd)}")
        return run_name, 0

    # Shared-cluster GPUs occasionally fail a run transiently (contention from other
    # users' jobs); retry a couple of times before giving up rather than requiring
    # a manual re-run.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            break
        if attempt < max_attempts:
            log(f"[suite] RETRY  {run_name!r}  rc={result.returncode}  (attempt {attempt}/{max_attempts})")
            time.sleep(10)
    status = "OK" if result.returncode == 0 else f"FAILED(rc={result.returncode})"
    log(f"[suite] DONE   {run_name!r}  {status}")
    if result.returncode == 0:
        # Written only on success — this, not _suite_config.yaml (which exists
        # before the subprocess even starts), is what --skip-existing checks.
        # Otherwise a killed/crashed run leaves behind a config file that makes
        # a later --skip-existing resume silently treat it as complete.
        open(os.path.join(run_dir, "_DONE"), "w").close()
    return run_name, result.returncode


# ── summary file ───────────────────────────────────────────────────────────────

def write_suite_summary(
    session_dir: str,
    suite_path: str,
    results: list,           # [(name, rc), ...]  rc=0 pass, rc!=0 fail
    skipped: list,           # [name, ...]
    disabled: list,          # [(name, config, meta), ...]
) -> None:
    """
    Write <session_dir>/suite_summary.txt — clean status per run, no verbose noise.
    """
    n_pass    = sum(1 for _, rc in results if rc == 0)
    n_fail    = sum(1 for _, rc in results if rc != 0)
    n_skip    = len(skipped)
    n_dis     = len(disabled)
    n_total   = len(results) + n_skip + n_dis

    lines = []
    lines.append(f"Session : {session_dir}")
    lines.append(f"Suite   : {suite_path}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"  PASSED   {n_pass:>4}")
    lines.append(f"  FAILED   {n_fail:>4}")
    lines.append(f"  SKIPPED  {n_skip:>4}  (--skip-existing)")
    lines.append(f"  DISABLED {n_dis:>4}  (enabled: false)")
    lines.append(f"  TOTAL    {n_total:>4}")
    lines.append("")
    lines.append(f"{'STATUS':<10}  RUN NAME")
    lines.append("─" * 80)

    all_rows = (
        [(name, "PASSED"  if rc == 0 else f"FAILED rc={rc}") for name, rc in results]
        + [(name, "SKIPPED")  for name in skipped]
        + [(name, "DISABLED") for name, _, _ in disabled]
    )
    # Sort disabled by name for readability; results keep run order
    all_rows.sort(key=lambda x: (
        {"PASSED": 0, "SKIPPED": 1}.get(x[1], 2 if x[1].startswith("FAILED") else 3),
        str(x[0])
    ))
    for name, status in all_rows:
        lines.append(f"{status:<12}  {name}")

    path = os.path.join(session_dir, "suite_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[suite] summary : {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regression suite runner")
    parser.add_argument(
        "suite", nargs="?",
        default=os.path.join(os.getcwd(), "default_suite.yaml"),
        help="Path to suite YAML (default: default_suite.yaml in cwd)",
    )
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--gpus",          type=str, default=None)
    parser.add_argument("--threads-per-job", type=int, default=1,
                        help="CPU threads per job (OMP/MKL/OpenBLAS). Default: 1. "
                             "Increase for CPU-only runs; keep low for GPU runs to avoid "
                             "starving other GPU workers of CPU resources.")
    parser.add_argument("--force-gpu",     action="store_true",
                        help="Skip GPU memory probing and launch immediately on the "
                             "assigned GPU. Implied when --gpus is set explicitly.")
    parser.add_argument("--session",       type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cross-compare", action="store_true",
                        help="Run cross_compare automatically at the end of the session")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override common.seed for every run in this session. "
                             "The seed is embedded in the leaf directory name (seed_N/). "
                             "Use --session to accumulate multiple seeds in the same tree.")
    parser.add_argument("--rq", type=str, default="all",
                        help="Filter by research question: q1, q2, q3, or all (default). "
                             "Matches the rq: field in each sub-suite YAML. "
                             "Example: --rq q1  runs only suites tagged rq: q1_arch.")
    parser.add_argument("--name", type=str, default=None,
                        help="Run only tests whose name contains this substring "
                             "(case-insensitive). Comma-separate for multiple patterns: "
                             "--name MLP_Group,MLP_Point")
    parser.add_argument("--max-cpu-jobs", type=int, default=None,
                        help="Maximum number of concurrent CPU-only jobs (device: cpu). "
                             "GPU jobs are not affected — they are already gated by --gpus. "
                             "Default: unlimited. Use e.g. --max-cpu-jobs 2 to leave CPU "
                             "headroom for GPU processes and other tasks.")
    args = parser.parse_args()

    suite_path = args.suite if os.path.isabs(args.suite) else os.path.join(os.getcwd(), args.suite)
    suite, suite_dir = _load_suite_file(suite_path)

    # Load top-level base config
    base_cfg_path = suite.get("base_config")
    base_cfg = {}
    if base_cfg_path:
        if not os.path.isabs(base_cfg_path):
            base_cfg_path = os.path.join(suite_dir, base_cfg_path)
        with open(base_cfg_path) as f:
            base_cfg = yaml.safe_load(f) or {}

    run_module  = suite.get("run_module", None)
    output_base = suite.get("output_dir", "runs/regression")
    if not os.path.isabs(output_base):
        output_base = os.path.join(os.getcwd(), output_base)
    if args.session:
        output_base = args.session if os.path.isabs(args.session) else os.path.join(os.getcwd(), args.session)
    else:
        session_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(output_base, f"session_{session_ts}")

    gpus = ([int(g) for g in args.gpus.split(",")]
            if args.gpus else available_gpu_ids())
    # Always probe by default — only skip when --force-gpu is explicitly requested.
    # Passing --gpus does NOT skip probing; exclusive-mode GPUs need the probe to
    # prevent concurrent launches on the same GPU.
    skip_probe = args.force_gpu

    runs, disabled_runs = generate_runs(suite, suite_dir, suite_path)

    # ── --rq filter ───────────────────────────────────────────────────────────
    if args.rq and args.rq != "all":
        rq_prefix = args.rq.rstrip("_")  # "q1" matches "q1_arch", "q1_stream", …
        def _rq_matches(m):
            return (m.get("rq") or "").startswith(rq_prefix)
        runs          = [(n, c, m) for n, c, m in runs          if _rq_matches(m)]
        disabled_runs = [(n, c, m) for n, c, m in disabled_runs if _rq_matches(m)]

    # ── --name filter ─────────────────────────────────────────────────────────
    if args.name:
        patterns = [p.strip().lower() for p in args.name.split(",") if p.strip()]
        def _name_matches(n):
            nl = n.lower()
            return any(p in nl for p in patterns)
        runs          = [(n, c, m) for n, c, m in runs          if _name_matches(n)]
        disabled_runs = [(n, c, m) for n, c, m in disabled_runs if _name_matches(n)]

    # ── open session log ──────────────────────────────────────────────────────
    os.makedirs(output_base, exist_ok=True)
    global _log_fh
    _log_fh = open(os.path.join(output_base, "suite.log"), "a")
    log(f"[suite] log     : {os.path.join(output_base, 'suite.log')}")
    log(f"[suite] suite   : {suite_path}")
    log(f"[suite] session : {output_base}")
    if base_cfg_path:
        log(f"[suite] base cfg: {base_cfg_path}")
    if run_module:
        log(f"[suite] module  : {run_module}")
    log(f"[suite] GPUs    : {sorted(set(gpus))}  ({len(gpus)} slot(s))")
    if args.seed is not None:
        log(f"[suite] seed    : {args.seed}  (overrides YAML)")
    if args.rq != "all":
        log(f"[suite] --rq    : {args.rq}")
    if disabled_runs:
        log(f"[suite] {len(disabled_runs)} disabled run(s) (enabled: false — skipped silently):")
        for name, _, _ in disabled_runs:
            log(f"  [dis] {name}")
    log(f"[suite] {len(runs)} run(s) planned:")
    for i, (name, _, meta) in enumerate(runs, 1):
        mod = meta.get("run_module", run_module)
        rq  = meta.get("rq") or ""
        ss  = meta.get("subsuite_name") or ""
        log(f"  [{i:>3}] {name}  [{rq}/{ss}]  (module: {mod})")

    if args.dry_run:
        log("\n[suite] DRY RUN — writing configs but not launching processes")
    if args.skip_existing:
        log("[suite] --skip-existing: completed runs will be skipped")

    # ── GPU pool ──────────────────────────────────────────────────────────────
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    cpu_sem = threading.Semaphore(args.max_cpu_jobs) if args.max_cpu_jobs else None
    if args.max_cpu_jobs:
        log(f"[suite] max CPU jobs: {args.max_cpu_jobs}")

    results  = []
    skipped  = []
    lock     = threading.Lock()

    def worker(run_name, override, meta):
        run_suite_dir = meta.get("suite_dir", suite_dir)

        # ── build merged config first (fast CPU work, no GPU needed yet) ─────
        run_base_cfg = base_cfg
        if "base_config" in meta:
            bc = meta["base_config"]
            if not os.path.isabs(bc):
                bc = os.path.join(run_suite_dir, bc)
            with open(bc) as f:
                run_base_cfg = yaml.safe_load(f) or {}

        cfg = run_base_cfg
        for layer_path in meta.get("config_layers", []):
            if not os.path.isabs(layer_path):
                layer_path = os.path.join(run_suite_dir, layer_path)
            with open(layer_path) as f:
                layer = yaml.safe_load(f) or {}
            cfg = deep_merge(cfg, layer)
        cfg = deep_merge(cfg, override)

        # Apply --seed override before building the path (so the directory name
        # reflects the seed that will actually be used by the run)
        if args.seed is not None:
            cfg = deep_merge(cfg, {"common": {"seed": args.seed}})

        effective_seed = cfg.get("common", {}).get("seed", 42)
        hostname = socket.gethostname().split(".")[0]  # e.g. "cuda06"
        seed_dir = f"{hostname}_seed_{effective_seed}"

        # ── build hierarchical output path ────────────────────────────────────
        rq_label      = meta.get("rq") or "ungrouped"
        subsuite_name = meta.get("subsuite_name") or "runs"
        compare_meta  = meta.get("compare_meta", {})
        anomaly       = compare_meta.get("anomaly")
        arch          = compare_meta.get("arch")
        section       = compare_meta.get("section")   # optional grouping axis (e.g. "beta", "anneal")
        variant       = compare_meta.get("variant")

        if anomaly and arch and variant:
            # Full hierarchy: session/rq/subsuite/anomaly/arch/label/seed_N
            # Combine section + variant into one descriptive dir (e.g. "free_bits_0_0")
            safe_variant = str(variant).replace(".", "_")
            label = f"{section}_{safe_variant}" if section else safe_variant
            run_dir = os.path.join(
                output_base, rq_label, subsuite_name,
                anomaly, arch, label, seed_dir,
            )
        else:
            # Fallback for runs without compare_meta
            run_dir = os.path.join(output_base, rq_label, subsuite_name, run_name, seed_dir)

        # ── --skip-existing: marker is _DONE, written only after a successful run ──
        if args.skip_existing and not args.dry_run:
            if os.path.exists(os.path.join(run_dir, "_DONE")):
                log(f"[suite] SKIP   {run_name!r}  ({run_dir})")
                with lock:
                    results.append((run_name, 0))
                    skipped.append(run_name)
                return

        # Embed compare metadata so cross_compare can read it without the suite YAML
        if "compare_meta" in meta:
            cfg["_compare_meta"] = meta["compare_meta"]

        # ── GPU assignment (CPU jobs bypass the queue entirely) ──────────────────
        is_cpu_job = cfg.get("common", {}).get("device") == "cpu"
        if is_cpu_job:
            gpu_id = None
            if cpu_sem:
                cpu_sem.acquire()
        else:
            # Acquire ONE slot and hold it for the entire job duration.
            # This is the only guarantee that at most one job runs per GPU at a time.
            # We do NOT put the slot back while waiting — that caused a race where
            # all workers cycled through the same slots and started simultaneously.
            gpu_id = gpu_q.get()
            if not (args.dry_run or skip_probe):
                probe_interval = 30
                waited = False
                while not probe_gpu(gpu_id):
                    if not waited:
                        try:
                            mem_result = subprocess.run(
                                ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10,
                            )
                            mem_summary = "  ".join(
                                f"GPU{parts[0].strip()}:{parts[1].strip()}/{parts[2].strip()}MB free"
                                for line in mem_result.stdout.strip().splitlines()
                                if (parts := line.split(",")) and len(parts) == 3
                            )
                        except Exception:
                            mem_summary = "(nvidia-smi unavailable)"
                        log(f"[suite] {run_name!r} waiting for a free GPU  [{mem_summary}]"
                            f"  — use --force-gpu to skip this check")
                        waited = True
                    time.sleep(probe_interval)

        try:
            effective_module = meta.get("run_module") or run_module
            if not effective_module:
                raise ValueError(f"No run_module defined for run {run_name!r}")
            name, rc = run_job(run_name, cfg, effective_module, run_dir, gpu_id,
                               args.dry_run, extra_args=meta.get("extra_args"),
                               threads=args.threads_per_job)
            with lock:
                results.append((name, rc))
        finally:
            if is_cpu_job:
                if cpu_sem:
                    cpu_sem.release()
            else:
                gpu_q.put(gpu_id)

    threads = [
        threading.Thread(target=worker, args=(name, override, meta), daemon=True)
        for name, override, meta in runs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ── final summary ─────────────────────────────────────────────────────────
    log("\n[suite] ── SUMMARY ──────────────────────────────────────────────")
    ok = sum(1 for _, rc in results if rc == 0)
    for name, rc in results:
        status = "OK  " if rc == 0 else f"FAIL(rc={rc})"
        log(f"  {status}  {name}")
    log(f"[suite] {ok}/{len(results)} passed")

    # ── write clean summary file ──────────────────────────────────────────────
    write_suite_summary(
        session_dir=output_base,
        suite_path=suite_path,
        results=results,
        skipped=skipped,
        disabled=disabled_runs,
    )

    # ── run cross_compare ─────────────────────────────────────────────────────
    if not args.dry_run and args.cross_compare:
        # Collect the unique (rq, subsuite) pairs from completed runs so we can
        # run a focused cross_compare for each sub-suite directory and one for
        # each rq directory, then a final overall one.
        rq_subsuite_pairs = sorted({
            (meta.get("rq") or "ungrouped", meta["subsuite_name"])
            for _, _, meta in runs
            if "subsuite_name" in meta
        })
        rq_labels_done = sorted({rq for rq, _ in rq_subsuite_pairs})

        # Per-sub-suite: finest granularity — one experiment at a time
        for rq_lbl, ss in rq_subsuite_pairs:
            ss_dir = os.path.join(output_base, rq_lbl, ss)
            if not os.path.isdir(ss_dir):
                continue
            log(f"\n[suite] cross_compare [{rq_lbl}/{ss}] → {ss_dir}/cross_compare/")
            cc_result = subprocess.run(
                [sys.executable, _CROSS_COMPARE_PATH, ss_dir]
            )
            if cc_result.returncode != 0:
                log(f"[suite] cross_compare [{rq_lbl}/{ss}] rc={cc_result.returncode} (non-fatal)")

        # Per-rq: one cross_compare per research question (all sub-suites for that RQ)
        for rq_lbl in rq_labels_done:
            rq_dir = os.path.join(output_base, rq_lbl)
            if not os.path.isdir(rq_dir):
                continue
            log(f"\n[suite] cross_compare [{rq_lbl}] → {rq_dir}/cross_compare/")
            cc_result = subprocess.run(
                [sys.executable, _CROSS_COMPARE_PATH, rq_dir]
            )
            if cc_result.returncode != 0:
                log(f"[suite] cross_compare [{rq_lbl}] rc={cc_result.returncode} (non-fatal)")

        # Overall: everything in one pass for the best-of-all summary
        log(f"\n[suite] cross_compare [overall] → {output_base}/cross_compare/")
        cc_result = subprocess.run(
            [sys.executable, _CROSS_COMPARE_PATH, output_base]
        )
        if cc_result.returncode != 0:
            log(f"[suite] cross_compare [overall] rc={cc_result.returncode} (non-fatal)")
        else:
            log(f"[suite] cross_compare complete — see {output_base}/cross_compare/")

    if _log_fh is not None:
        _log_fh.close()
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
