#!/usr/bin/env python3
"""Streaming VAE Anomaly Detection — quick-start demo.

Runs one config from standalone_tests/baseline/ (or standalone_tests/finetuned/
with --best) end to end: offline warmup training + online streaming anomaly
detection on a real ERA5 slice.

Usage:
    python demo.py                              # MLP-Cyclic on contextual anomalies (default)
    python demo.py --anom point --arch mlp
    python demo.py --anom group --arch lstm --best
    python demo.py --anom contextual --arch transformer
    python demo.py --gpu 1                       # pin to cuda:1 (default: cuda:0; ignored for --anom point)
"""

import argparse
import os
import sys
import tempfile

import torch
import yaml

from scripts.trial.run_trial import run_trial

ANOM_ALIASES = {
    "point": "Point",
    "group": "Group",
    "contextual": "Contextual",
}

ARCH_ALIASES = {
    "mlp": "MLP",
    "mlp_cyc": "MLP_Cyclic",
    "mlp_cyclic": "MLP_Cyclic",
    "lstm": "LSTM",
    "tr": "Transformer",
    "tf": "Transformer",
    "transformer": "Transformer",
    "transtormer": "Transformer",  # common typo
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one streaming-VAE anomaly-detection experiment end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--anom", default="contextual", choices=sorted(ANOM_ALIASES),
        help="Anomaly type (default: contextual)",
    )
    parser.add_argument(
        "--arch", default="mlp_cyclic", choices=sorted(ARCH_ALIASES),
        help="Model architecture (default: mlp_cyclic)",
    )
    parser.add_argument(
        "--best", action="store_true",
        help="Run the finetuned config instead of the baseline one",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, metavar="N",
        help="GPU index to run on, e.g. --gpu 1 for cuda:1 (default: 0). "
             "Ignored for --anom point, which always runs on CPU.",
    )
    return parser.parse_args(argv)


def resolve_experiment(anom: str, arch: str, best: bool):
    anomaly = ANOM_ALIASES[anom]
    arch_dir = ARCH_ALIASES[arch]
    tuning = "finetuned" if best else "baseline"
    dirname = f"{anomaly}_{arch_dir}"
    experiment_dir = f"standalone_tests/{tuning}"

    config_path = f"{experiment_dir}/era5_{anom}_{arch_dir}_config_standalone.yaml"
    if not os.path.exists(config_path):
        sys.exit(
            f"No config for anom={anom!r}, arch={arch!r} — "
            f"{config_path} does not exist. "
            "(Point only has mlp/mlp_cyclic; Contextual and Group have all four architectures.)"
        )
    return tuning, dirname, config_path


def main(argv=None):
    args = parse_args(argv)
    tuning, dirname, config_path = resolve_experiment(args.anom, args.arch, args.best)
    tracked_config_path = config_path  # the real standalone_tests/ path, before any temp-file device override below

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_arch = cfg["model"]["type"]

    new_device = None
    if args.anom == "point":
        # Point mode streams one gradient step per row with no batching, which
        # is faster on CPU than GPU for this access pattern — every Point_*
        # config is pinned to device: cpu, and --gpu does not override that.
        pass
    elif not torch.cuda.is_available():
        # Don't write cuda:N into the config if this torch install/runtime has
        # no working CUDA — that fails deep inside run_trial with a confusing
        # "torch._C has no attribute _cuda_setDevice" instead of a clear message.
        print(
            "WARNING: no CUDA device available (torch.cuda.is_available() is "
            "False) — running on CPU instead. In Colab, check Runtime -> "
            "Change runtime type -> GPU.",
            file=sys.stderr,
        )
        new_device = "cpu"
    else:
        new_device = f"cuda:{args.gpu}"

    if new_device is not None:
        cfg["common"]["device"] = new_device
        # run_trial reads config_path from disk itself, so the override is
        # written to a temp copy rather than mutating the tracked config.
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"demo_{dirname}_", delete=False,
        )
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp.close()
        config_path = tmp.name

    print(f"Experiment : {tuning}/{dirname}")
    print(f"Config     : {config_path}")
    print(f"Model arch : {model_arch}  (cyclic_recon={cfg['model'].get('cyclic_recon', False)})")
    print(f"Device     : {cfg['common']['device']}")
    print(f"Run dir    : {cfg['common']['logging']['run_dir']}")
    print()
    print("This demo runs one fixed model/anomaly pair for a quick environment check.")
    print("To compare architectures or tune hyperparameters directly, run the same")
    print("experiment via its tracked config instead of demo.py:")
    print()
    print("    pip install -r requirements.txt")
    print(f"    python -m scripts.trial.run_trial {tracked_config_path}")
    print()

    run_trial(config_path, model_arch)


if __name__ == "__main__":
    main()
