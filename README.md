# Streaming Anomaly Detection with VAEs

This repository contains a curated implementation of the streaming anomaly-detection framework developed for my thesis.

The thesis studies how different [VAE architectures](#vae-architectures) and streaming configurations affect [anomaly detection](#anomaly-scenarios) in temporally dependent data. The experiments use hourly ERA5 climate reanalysis data and compare four architectures:

- **MLP-VAE**
- **MLP-VAE-Cyclic**
- **LSTM-VAE**
- **Transformer-VAE**

Rather than reproducing the complete research codebase, this repository focuses on a selected set of representative test scenarios. Each scenario was chosen to demonstrate one of the main behaviours observed in the thesis, such as the effect of cyclic time features, window size and stride, latent dimensionality, prediction thresholds, or online adaptation.

The detector is first trained on a warmup segment assumed to contain normal data. It is then evaluated in streaming mode, processing either individual timesteps or temporal windows. During the stream, the model may continue updating its parameters, while anomaly scores and thresholds are adapted over time.

Anomalies are detected from VAE reconstruction behaviour, with additional latent-space and ELBO-related metrics available in the framework. The prediction threshold is based on an adaptive rolling score distribution rather than a fixed raw reconstruction-error value.

See [`DATA_LICENSE.md`](DATA_LICENSE.md) for details about the ERA5 data included in this repository.

## Quickstart

The framework may be tried by running `demo.py` from a local terminal, or by running 
[`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) on Google Colab, which 
runs the same demo on Colab GPU runtime. It needs a couple of extra one-time steps (a GitHub token secret, runtime selection), see [Colab](#colab) section below.

To execute `demo.py`:

```bash
pip install -r requirements.txt
python demo.py
```

The demo runs one fixed model on one fixed [anomaly type](#anomaly-scenarios) — it is not meant for comparing [architectures](#vae-architectures) or tuning hyperparameters, and its configuration is not user-editable. Its purpose is a quick, simple environment check: a successful run confirms the local or Colab environment is set up correctly and shows what one model's output looks like, before moving on to the added complexity of a full test suite.
By default, the demo runs **MLP-VAE-Cyclic** on the **Contextual** anomaly scenario.

The demo's output will include a command such as `python -m scripts.trial.run_trial <config>` for the trial it just ran, pointing to a config under [`standalone_tests/`](#baseline-and-fine-tuned-configurations), which is convenient for setting models for comparison or tuning configurations.

Other combinations can be selected using:

```bash
python demo.py --anom point --arch mlp
python demo.py --anom group --arch lstm
python demo.py --anom contextual --arch transformer
```

Available [anomaly types](#anomaly-scenarios):

```text
point
group
contextual
```

Available [architectures](#vae-architectures):

```text
mlp
mlp_cyclic
lstm
transformer
```

Finetuned config instead of the common baseline:

```bash
--best
```

To select a GPU:

```bash
--gpu N
```

Point-mode experiments run on CPU regardless of the selected GPU.

For the complete set of options:

```bash
python demo.py --help
```

Each run prints its output directory at startup and stores its logs and metrics there.

## Colab

The notebooks under [`notebooks/cases/`](notebooks/cases/) and [`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) can be opened directly in Google Colab.

The demo Colab notebook:

[`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb)

clones the repository and runs `demo.py`. The [anomaly type](#anomaly-scenarios), [architecture](#vae-architectures), configuration, and GPU can be selected from the first code cell.

Other notebooks in the repository can also be opened in Colab by replacing:

```text
github.com
```

with:

```text
githubtocolab.com
```

in the notebook URL.

For example:

```text
https://githubtocolab.com/hadasecohen/streaming-vae-anomaly-detection/blob/main/notebooks/demo_colab.ipynb
```

## Baseline and Fine-Tuned Configurations

The directory [`standalone_tests/`](standalone_tests/):

```text
standalone_tests/
    baseline/
    finetuned/
```

contains representative configurations for each [architecture](#vae-architectures) and [anomaly type](#anomaly-scenarios), run directly via:

```bash
python -m scripts.trial.run_trial <config.yaml>
```

Each scenario compares a common baseline configuration with a configuration adjusted according to the corresponding thesis experiments.

For example:

```text
standalone_tests/baseline/era5_contextual_MLP_Cyclic_config_standalone.yaml
standalone_tests/finetuned/era5_contextual_MLP_Cyclic_config_standalone.yaml
```

The purpose is not to claim that the fine-tuned configuration is universally optimal. The thesis shows that VAE performance depends on the interaction between architecture, anomaly type, temporal representation, and streaming configuration.

The fine-tuned configurations therefore represent settings that performed well for every model on every anomaly type, under the experimental conditions used in the thesis.

## Selected Thesis Scenarios

The directory:

[`notebooks/cases/`](notebooks/cases/)

contains a set of small experiments selected from the thesis.

These notebooks are not intended to reproduce every experiment in the thesis. Instead, they provide relatively small comparisons that demonstrate some of the more interesting findings.

Examples include:

- baseline false-positive behaviour on clean ERA5 data;
- the effect of cyclic time features on contextual anomalies;
- prediction-threshold calibration;
- point versus window streaming;
- latent dimensionality;
- online model adaptation;
- window size and stride;
- selected VAE configuration effects.

Each notebook contains a short explanation at the beginning describing:

- what thesis question or observation it represents;
- which configurations are compared;
- what result to look for;
- how the scenario relates to the full experiment.

Most notebooks perform only a small number of runs so that the behaviour can be inspected without reproducing the complete experimental sweep.

The case notebooks use their own suite YAML files and default to the full ERA5 split:

```python
DATA_SOURCE = "full_split_files"
```

Smaller ERA5 subsets are also available for faster execution.

[`case01_clean_baseline.ipynb`](notebooks/cases/case01_clean_baseline.ipynb) compares the available dataset scales directly and shows why dataset size can affect the observed streaming behaviour.


## Anomaly Scenarios

Three synthetic anomaly types are used.

### Point anomalies

Point anomalies are isolated deviations affecting a single timestep.

They are useful for examining detection when temporal context is not required and the anomaly can be identified mainly from the current observation.

### Group anomalies

Group anomalies are sustained abnormal shifts over a period of time.

In the thesis, the two MLP based models group anomalies are unusual individually. Hence the ERA5 dataset with synthetic group anomalies is less used in the repository's selected testcases.

### Contextual anomalies

Contextual anomalies contain values that may be individually plausible but occur in the wrong temporal context.

For example, a climate pattern may resemble normal behaviour from another season.

These scenarios are particularly useful for examining whether temporal context is learned from a window or supplied explicitly through cyclic calendar features.

Most testcases selected for this repository use the  injected contextual anomalies dataset.


## Framework Overview

The streaming pipeline is approximately:

```text
ERA5 hourly data
    |
    v
ERA5Adapter
    |
    |-- scaling
    |-- warmup/test split
    |-- point or window construction
    |-- add cyclic/positional features 
    |
    v
AnomalyDetection
    |
    |-- train_warmup()
    |-- run_stream()
    |
    v
StreamTrainer
    |
    |-- VAE forward pass
    |-- reconstruction / ELBO / latent metrics
    |-- confidence calculation
    |-- prediction
    |-- adaptive score threshold
    |-- optional online model update
    |
    v
Run metrics and logs
```

The main implementation is located under:

```text
src/
```


## VAE Architectures

The VAE implementations are located under:

```text
src/VAE/
```

### MLP-VAE

A feed-forward VAE in which timesteps are processed without an internal recurrent or attention mechanism.

The thesis experiments show that this relatively simple architecture can remain highly competitive when the anomaly is primarily value-based or when the required temporal information is supplied explicitly.

### MLP-VAE-Cyclic

An MLP-VAE extended with cyclic representations of known temporal variables.

This variant of MLP-VAE model was studied during the thesis experiments to determine whether explicit calendar information could compensate for the lack of a sequential architecture.

It was particularly effective for contextual anomalies in the ERA5 stream.

### LSTM-VAE

A recurrent VAE that represents temporal progression through LSTM layers.

It is useful for examining settings in which neighbouring observations and their order contribute to the anomaly representation.

### Transformer-VAE

A Transformer-based VAE using self-attention and positional encoding.

It provides the capacity to model relationships between positions within the input window. The thesis experiments examine how this architecture behaves under different streaming, latent, and threshold configurations, rather than assuming that attention is always beneficial.

All architectures use the shared VAE functionality in:

```text
src/VAE/base_VAE.py
```

including reparameterisation, KL loss, reconstruction-loss handling, and common tensor processing.

The pipeline uses tensors in the form:

```text
(batch, time, features)
```

## Streaming Behaviour

The main distinction in the framework is between **point mode** and **window mode**.

In point mode:

```text
T = 1
```

and each observation is scored independently.

In window mode:

```text
T = seq_len
```

and a complete temporal window is processed at each streaming step.

Window size and stride are important experimental parameters because they determine:

- how much temporal context is available;
- how much consecutive windows overlap;
- how much of a window is occupied by an anomaly;
- how frequently the model and threshold are updated.

During streaming, the framework provides configurations for updating both the VAE weights and the adaptive score threshold.

The raw anomaly-score threshold is estimated from recent scores using a rolling quantile and exponential smoothing. Each score is then converted to a confidence value based on scores' distance from this adaptive threshold. The final binary prediction uses a separate fixed confidence threshold.

This distinction allows the raw-score distribution to adapt over time while keeping the prediction criterion defined in a common confidence space.

## Multi-Run Experiments

The repository also contains notebooks that run small controlled experiment suites.

### Run a suite

```text
scripts/run_all_trials.py
```

Reads a suite YAML file and launches multiple experiment configurations, optionally across several GPUs.

### Compare completed runs

```text
scripts/cross_compare.py
```

Produces comparison tables and plots for a completed experiment session.

### Report metrics

```text
scripts/report_metrics.py
```

Collects metrics across a session tree and can invoke the cross-comparison tooling.

### Single trial runner

```text
scripts/trial/run_trial.py
```

Runs one configuration:

```bash
python -m scripts.trial.run_trial <config.yaml>
```

The architecture is read from the config's own `model.type` field, so it doesn't need to be passed separately (an explicit `<ARCH>` second argument is still accepted as an override, but is redundant in normal use). Unlike the demo, where configuration files are fixed, the run_trial method takes a yaml config file as input, which is used to generate the environment (data adapter, VAE, offline and online learning, results) of each trial in a trial suite.


The YAML files beside the case notebooks provide practical examples of the experiment-suite format.

Shared configuration fragments are stored under:

```text
modules/
```

and combined through `config_layers`.

### Seeds and reproducibility

Every run is controlled by `common.seed` in its config. If left unset, a
time-based seed is generated automatically, so results are not reproducible
run to run unless a seed is pinned explicitly.

A single seed is not a reliable basis for comparing architectures or
configurations. `case01_clean_baseline.ipynb` shows this directly: seven
otherwise-identical LSTM-VAE runs on the same clean baseline, differing only
by seed, produced false-positive rates ranging from 22.4% to 89.6%. Drawing
a conclusion from one seed risks mistaking run-to-run noise for a genuine
effect of the variable being tested.

`scripts/run_all_trials.py --seed` overrides `common.seed` for every run in the suite and is embedded in the output path (`seed_N/`), so results from different seeds land side by side instead of overwriting each other. `scripts/cross_compare.py` then aggregates across the `seed_N/` directories it finds, reporting mean/std rather than a single run's numbers.

## Data

The main ERA5 dataset used by the repository is stored under:

```text
data/era5/full_scale/split/
```

It contains hourly ERA5 reanalysis data pre-split into warmup and streaming-test segments.

Two smaller real-data subsets are also included:

```text
data/era5/mini_50pct/
data/era5/mini_28pct/
```

These are intended mainly for faster execution and experimentation.

The full thesis results were produced under controlled experimental configurations and should not be assumed to reproduce exactly when the dataset scale, architecture parameters, or streaming settings are changed.

For data provenance, attribution, and licensing information, see:

[`DATA_LICENSE.md`](DATA_LICENSE.md)

## Scope

This repository is intended as a reproducible demonstration of selected thesis experiments, not as a complete general-purpose anomaly-detection library.

The results represented here are specific to:

- ERA5 climate data;
- the selected variables;
- the injected anomaly definitions;
- the tested streaming configurations;
- the VAE implementations used in the thesis.

The main purpose of the repository is to make the experimental behaviour easier to inspect and reproduce, and to provide concrete examples of how architecture, temporal representation, and streaming configuration interact in VAE-based anomaly detection.

## License

MIT — see [`LICENSE`](LICENSE).