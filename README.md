# Streaming Anomaly Detection with VAEs

A streaming anomaly detection system for ERA5 hourly climate reanalysis data
(temperature, wind, pressure, precipitation, radiation). A Variational
Autoencoder is trained offline on a warmup segment, then updated online
one sample (or window) at a time while streaming through the rest of the
series, flagging anomalies from reconstruction error, KL divergence, and
latent-space statistics with an adaptive rolling-quantile threshold.

This is a curated, from-scratch-reviewed subset of a larger research
codebase — see [`DATA_LICENSE.md`](DATA_LICENSE.md) for what data is
(and isn't) included.

## Architecture

```
CSV (ERA5 hourly reanalysis)
  -> ERA5Adapter (src/Data_tools/era5/era5_adapter.py)   # scales, windows, splits warmup/test
  -> AnomalyDetection (src/VAE_model/Streaming/streaming_loop.py)
       |-> train_warmup() -> offline_fit()                 # standard VAE training on warmup segment
       |-> run_stream() sample-by-sample                   # online step() loop
  -> StreamTrainer (src/VAE_model/Streaming/stream_trainer.py)
       |-> VAE forward pass (MLP / LSTM / TF_VAE)
       |-> multi-metric anomaly scoring (elbo, recon, kl, mu_norm, mah_diag, ...)
       |-> adaptive thresholding (rolling quantile + EMA)
       |-> online gradient update (normal-only or all samples)
  -> runs/<scenario>/<arch>/standalone/
```

Three VAE architectures are implemented in `src/VAE_model/VAE/`:
- **MLP_VAE** — per-timestep MLP encoder/decoder, asymmetric depth supported
- **LSTM_VAE** — recurrent encoder/decoder
- **TF_VAE** — pure Transformer encoder/decoder with internal sinusoidal positional encoding

Each supports point mode (`T=1`) or window mode (`T=seq_len`), optional
normalizing-flow posterior enrichment (`flow_type: planar`), and per-layer
normalization. All tensors flowing through the pipeline are strictly 3-D:
`(batch, time, features)` — see `src/VAE_model/VAE/base_VAE.py` for the
shared VAE base class (reparameterization, KL loss, pooling, reconstruction
loss dispatch).

## Anomaly scenarios

`experiments/era5/<ARCH>/<scenario>/` covers four scenarios per architecture:
`clean` (no anomalies), `point` (isolated spikes), `contextual` (locally
wrong seasonal/diurnal pattern), `group` (collective mean-shift blocks).

## Quickstart

```bash
pip install -r requirements.txt
python scripts/generate_mini_era5.py          # writes data/era5/*.csv (synthetic, no download needed)
python -m experiments.era5.MLP.group.run_era5_group_MLP_standalone
```

Swap `MLP` for `LSTM` or `TF_VAE`, and `group` for `clean` / `point` / `contextual`,
to run any of the 12 architecture × scenario combinations. Each run writes
logs and metrics to `runs/<scenario>/<arch>/standalone/run.log`.

## Colab (GPU)

[`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) clones this repo
and runs the same MLP/group experiment on a free Colab GPU runtime — no
local setup, no data download (the mini dataset is synthetic and generated
in the notebook itself).

## Multi-run tooling

- `run_regression.py` — reads a suite YAML, runs multiple configs in parallel across GPUs
- `cross_compare.py` — summarizes a completed session into a comparison table + plots
- `scripts/report_metrics.py` — walks a session tree and optionally invokes `cross_compare.py` (`--cc`)
- `regression/run_trial.py` — the single-config runner these suites launch as a subprocess (`python -m regression.run_trial <config.yaml> <ARCH>`)

No example suite YAML is bundled — compose your own suite YAML (see the
docstring at the top of `run_regression.py` for the schema) pointing
`config_layers` at any of the `experiments/era5/**/*.yaml` configs.

## Data

No real data is committed. `data/era5/` holds small, fully synthetic
ERA5-shaped CSVs (`scripts/generate_mini_era5.py`) so every experiment
config above runs standalone with zero downloads. See
[`DATA_LICENSE.md`](DATA_LICENSE.md) for details and for pointing
configs at real ERA5 data instead.

## License

MIT — see [`LICENSE`](LICENSE).
