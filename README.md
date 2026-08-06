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
  -> AnomalyDetection (src/Streaming/streaming_loop.py)
       |-> train_warmup() -> offline_fit()                 # standard VAE training on warmup segment
       |-> run_stream() sample-by-sample                   # online step() loop
  -> StreamTrainer (src/Streaming/stream_trainer.py)
       |-> VAE forward pass (MLP / LSTM / TF_VAE)
       |-> multi-metric anomaly scoring (elbo, recon, kl, mu_norm, mah_diag, ...)
       |-> adaptive thresholding (rolling quantile + EMA)
       |-> online gradient update (normal-only or all samples)
  -> runs/<scenario>/<arch>/standalone/
```

Three VAE architectures are implemented in `src/VAE/`:
- **MLP_VAE** — per-timestep MLP encoder/decoder, asymmetric depth supported
- **LSTM_VAE** — recurrent encoder/decoder
- **TF_VAE** — pure Transformer encoder/decoder with internal sinusoidal positional encoding

Each supports point mode (`T=1`) or window mode (`T=seq_len`), optional
normalizing-flow posterior enrichment (`flow_type: planar`), and per-layer
normalization. All tensors flowing through the pipeline are strictly 3-D:
`(batch, time, features)` — see `src/VAE/base_VAE.py` for the
shared VAE base class (reparameterization, KL loss, pooling, reconstruction
loss dispatch).

## Anomaly scenarios

`experiments/era5/<ARCH>/<scenario>/` covers four scenarios per architecture:
`clean` (no anomalies), `point` (isolated spikes), `contextual` (locally
wrong seasonal/diurnal pattern), `group` (collective mean-shift blocks).

## Quickstart

```bash
pip install -r requirements.txt
python -m experiments.era5.MLP.group.run_era5_group_MLP_standalone
```

Swap `MLP` for `LSTM` or `TF_VAE`, and `group` for `clean` / `point` / `contextual`,
to run any of the 12 architecture × scenario combinations. Each run writes
logs and metrics to `runs/<scenario>/<arch>/standalone/run.log`.

## Colab (GPU)

[`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) clones this repo
and runs the same MLP/group experiment on a free Colab GPU runtime — no
local setup, no data download (the mini dataset ships with the repo).

[`notebooks/cases/`](notebooks/cases/) has 10 further notebooks, each
reproducing one specific finding from the thesis this code is based on
(prediction-threshold calibration, cyclic time features, latent
dimensionality, window size/stride, ...) as a small 2–4-run comparison with
its own suite YAML — see the markdown cell at the top of each for what it
reproduces.

## Multi-run tooling

- `run_regression.py` — reads a suite YAML, runs multiple configs in parallel across GPUs
- `cross_compare.py` — summarizes a completed session into a comparison table + plots
- `scripts/report_metrics.py` — walks a session tree and optionally invokes `cross_compare.py` (`--cc`)
- `regression/run_trial.py` — the single-config runner these suites launch as a subprocess (`python -m regression.run_trial <config.yaml> <ARCH>`)

`notebooks/cases/*_suite.yaml` are worked examples of the suite YAML schema
(see the docstring at the top of `run_regression.py` for the full schema),
each composed from the shared fragments under `modules/` via `config_layers`.

## Data

`data/era5/` holds a real, 16-year slice of ERA5 hourly reanalysis data
(140,256 rows, 50:50 warmup/test) so every experiment config above runs standalone with zero
downloads. See [`DATA_LICENSE.md`](DATA_LICENSE.md) for provenance,
attribution, and how to point configs at the full-scale real export instead.

## License

MIT — see [`LICENSE`](LICENSE).
