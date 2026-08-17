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
  -> runs/standalone_tests/<baseline|finetuned>/<Anomaly>_<Arch>/
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

`standalone_tests/{baseline,finetuned}/<Anomaly>_<Arch>/` covers three
anomaly types: `Contextual` (locally wrong seasonal/diurnal pattern),
`Group` (collective mean-shift blocks), and `Point` (isolated spikes).
`Contextual` and `Group` are available for all four architectures
(`MLP`, `MLP_Cyclic`, `LSTM`, `Transformer`); `Point` only for `MLP` and
`MLP_Cyclic` (point-mode streaming isn't a meaningful test for LSTM/Transformer
— see `case07_point_vs_window.ipynb`'s Scope section for why). Each
`<Anomaly>_<Arch>` pairs a `baseline` config against a `finetuned` one with
adjusted hyperparameters.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
```

Runs `MLP_Cyclic` on `Contextual` anomalies by default. Pick any other
combination with `--anom {point,group,contextual}` and
`--arch {mlp,mlp_cyclic,lstm,transformer}`; add `--best` to run the
finetuned config instead of the baseline one, and `--gpu N` to pick a GPU
(point-mode runs always use CPU regardless of `--gpu`). See `python demo.py
--help` for the full option list. Each run writes logs and metrics to the
`run_dir` printed at the top of its output.

## Colab (GPU)

[`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) clones this repo
and runs `demo.py` on a free Colab GPU runtime — set `ANOM`/`ARCH`/`BEST`/`GPU`
in its first code cell, no local setup needed.

[`notebooks/cases/`](notebooks/cases/) has 10 further notebooks, each
reproducing one specific finding from the thesis this code is based on
(prediction-threshold calibration, cyclic time features, latent
dimensionality, window size/stride, ...) as a small 2–4-run comparison with
its own suite YAML — see the markdown cell at the top of each for what it
reproduces. Each defaults to the full-scale ERA5 data (`DATA_SOURCE =
"full_split_files"` in its setup cell) rather than the mini dataset below —
see
[`case01_clean_baseline.ipynb`](notebooks/cases/case01_clean_baseline.ipynb)
for why that choice of scale matters (it compares its own four-architecture
suite across all three dataset scales directly).

## Multi-run tooling

- `scripts/run_regression.py` — reads a suite YAML, runs multiple configs in parallel across GPUs
- `scripts/cross_compare.py` — summarizes a completed session into a comparison table + plots
- `scripts/report_metrics.py` — walks a session tree and optionally invokes `scripts/cross_compare.py` (`--cc`)
- `scripts/trial/run_trial.py` — the single-config runner these suites launch as a subprocess (`python -m scripts.trial.run_trial <config.yaml> <ARCH>`)

`notebooks/cases/*_suite.yaml` are worked examples of the suite YAML schema
(see the docstring at the top of `scripts/run_regression.py` for the full schema),
each composed from the shared fragments under `modules/` via `config_layers`.

## Data

`data/era5/full_scale/split/` holds the real ERA5 hourly reanalysis export
this work is based on, pre-split into warmup/test files — both `demo.py`'s
`standalone_tests/` configs and the `notebooks/cases/` notebooks default to
this data. `data/era5/mini_50pct/` and `data/era5/mini_28pct/` are smaller,
faster-iteration real slices the case notebooks can opt into via
`DATA_SOURCE`. See [`DATA_LICENSE.md`](DATA_LICENSE.md) for provenance,
attribution, and full details on all three.

## License

MIT — see [`LICENSE`](LICENSE).
