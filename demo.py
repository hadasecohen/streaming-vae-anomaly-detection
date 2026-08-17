#!/usr/bin/env python3
"""Streaming VAE Anomaly Detection — quick-start demo.

Runs one config from experiments/baseline/ (or experiments/finetuned/ with
--best) end to end: offline warmup training + online streaming anomaly
detection on a real ERA5 slice.

Usage:
    python demo.py                              # MLP-Cyclic on contextual anomalies (default)
    python demo.py --anom point --arch mlp
    python demo.py --anom group --arch lstm --best
    python demo.py --anom contextual --arch transformer
"""

import argparse
import glob
import sys

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
    return parser.parse_args(argv)


def resolve_experiment(anom: str, arch: str, best: bool):
    anomaly = ANOM_ALIASES[anom]
    arch_dir = ARCH_ALIASES[arch]
    tuning = "finetuned" if best else "baseline"
    dirname = f"{anomaly}_{arch_dir}"
    experiment_dir = f"experiments/{tuning}/{dirname}"

    config_matches = glob.glob(f"{experiment_dir}/*_config_standalone.yaml")
    if not config_matches:
        sys.exit(
            f"No config for anom={anom!r}, arch={arch!r} — "
            f"{experiment_dir}/ does not exist. "
            "(Point only has mlp/mlp_cyclic; Contextual and Group have all four architectures.)"
        )
    return tuning, dirname, config_matches[0]


def main(argv=None):
    args = parse_args(argv)
    tuning, dirname, config_path = resolve_experiment(args.anom, args.arch, args.best)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_arch = cfg["model"]["type"]

    print(f"Experiment : {tuning}/{dirname}")
    print(f"Config     : {config_path}")
    print(f"Model arch : {model_arch}  (cyclic_recon={cfg['model'].get('cyclic_recon', False)})")
    print(f"Run dir    : {cfg['common']['logging']['run_dir']}")
    print()

    run_trial(config_path, model_arch)


if __name__ == "__main__":
    main()
