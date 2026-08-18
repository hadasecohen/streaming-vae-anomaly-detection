
import yaml
import torch
import argparse
from src.utils import set_global_seed
from src.env.streaming_env_standalone import StreamEnvStandalone


def run_trial(config_path: str, model_arch: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if model_arch is None:
        model_arch = cfg["model"]["type"]

    seed = set_global_seed(cfg["common"].get("seed", None))
    print(f"GLOBAL SEED = {seed}")

    device = cfg["common"].get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        device = "cuda:0"

    if "cuda" in device:
        torch.cuda.set_device(device)       # ensure device is active
        torch.cuda.manual_seed_all(seed)    # re-seed after device context is fully established

    env = StreamEnvStandalone(model_arch, cfg, seed, device, config_path=config_path)

    env.close_env()

    # Explicitly free GPU memory before interpreter exit so the OOM killer
    # doesn't fire during Python teardown of CUDA tensors (causes rc=-9).
    try:
        import gc
        del env
        gc.collect()
        if "cuda" in device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config",
                        help="Path to YAML config (for example: era5_clean_MLP.yaml)")
    parser.add_argument("model_arch", nargs="?", default=None,
                        help="Model architecture, e.g. MLP, LSTM, TF_VAE. "
                             "Optional: defaults to the config's own model.type.")

    args = parser.parse_args()

    run_trial(args.config, args.model_arch)
