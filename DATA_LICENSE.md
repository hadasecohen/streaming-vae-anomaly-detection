# Data License

## Full-scale real ERA5 data (default for every case notebook)

`data/era5/full_scale/` holds the real ERA5 hourly reanalysis export (755,568
rows, 1940–2026) this work is based on. Every case notebook's `DATA_SOURCE`
defaults to this data (option `"full_split_files"`) via
`modules/anom_types/*.yaml`, which points directly at the split files below
— no extra data_source layer needs to be appended for the default.

Source: ERA5 hourly reanalysis, European Centre for Medium-Range Weather
Forecasts (ECMWF), distributed via the Copernicus Climate Data Store (CDS):
https://cds.climate.copernicus.eu/ — one surface grid point (Canterbury, New
Zealand). Used under the Copernicus license, which permits reuse with
attribution. Same source/license for every dataset on this page, including
the mini ones below.

`data/era5/full_scale/` is organized into two subdirectories:

- `raw_files/` — the original, un-split full-scale exports (~550MB total: the
  clean export plus the point/group/contextual anomaly-injected variants).
  Local-only, gitignored, not shipped.
- `split/` — warmup/test pairs sliced from the files in `raw_files/`, each
  individual file kept under GitHub's 100MB limit so it can be committed.
  Everything below is in this subdirectory.

### Full-scale clean (no injected anomalies) data — included

`data/era5/full_scale/split/era5_clean_warmup.csv` (302,227 rows) and
`era5_clean_test.csv` (453,341 rows) together *are* the full-scale export
above, split at the same warmup/test boundary (1974-06-23) used throughout
this repo — pre-split so each file stays under GitHub's 100MB limit and can
be committed (the single combined 755,568-row file,
`data/era5/full_scale/raw_files/reanalysis-era5-single-levels-timeseries.csv`,
is ~120MB, over that limit, and stays local-only/gitignored).

`modules/anom_types/clean.yaml` points at these two files directly —
`case01_clean_baseline.ipynb` uses this data with
`DATA_SOURCE = "full_split_files"` (default). A
`"full_single_file"` option is also available in case01 (and only case01):
one full-scale CSV (`raw_files/reanalysis-era5-single-levels-timeseries.csv`,
gitignored) plus `warmup_start`/`warmup_end` row indices, via
`modules/data_source/full_single_file_clean.yaml`. Both modes read identical
rows in identical order — see `src/Data_tools/dataset_loader.py`'s
`_read_era5_raw`.

### Full-scale contextual-anomaly data — included

`data/era5/full_scale/split/era5_contextual_test.csv` (453,341 rows) is the
test portion of the full-scale contextual-anomaly export (10,584 injected
anomalies, all in this portion). Its warmup portion is not duplicated: the
warmup half of every full-scale anomaly export is anomaly-free and
byte-identical regardless of anomaly type (verified directly), so it
reuses `era5_clean_warmup.csv` above instead of shipping a second copy.

`modules/anom_types/contextual.yaml` points at these two files directly —
`case02_cyclic_features.ipynb`, `case04_latent_dimension.ipynb`,
`case05_kl_regularisation.ipynb`, `case07_point_vs_window.ipynb`,
`case08_stride_alignment.ipynb`, and `case10_positional_encoding.ipynb` use
this data with `DATA_SOURCE = "full_split_files"` (default).

### Full-scale group- and point-anomaly data — included

`data/era5/full_scale/split/era5_group_test.csv` (453,341 rows, 22,656
injected group anomalies) and `era5_point_test.csv` (453,341 rows, 906
injected point anomalies, sliced from `raw_files/era5_point_rare.csv` —
`raw_files/era5_point_hard.csv` is also available locally as an alternative,
not yet split) are the test portions of the full-scale group- and
point-anomaly exports. Same reasoning as the contextual file above: both
warmup portions are anomaly-free and identical to `era5_clean_warmup.csv`,
so neither ships a separate warmup file.

`modules/anom_types/group.yaml` and `modules/anom_types/point.yaml` point at
these files directly — `case03_threshold_calibration.ipynb` and
`case09_ema_freezing.ipynb` use the group data (point.yaml's point data is
not currently used by any case notebook), all with
`DATA_SOURCE = "full_split_files"` (default).

## Mini ERA5 datasets — opt-in, smaller/faster alternative

Two mini datasets exist, both real slices of ERA5 hourly reanalysis data
(not synthetic, same source as above) — sized as a rough percentage of the
full-scale export's 755,568-row total, per the naming convention. Every
case notebook can select one via `DATA_SOURCE = "mini_50pct"` or
`"mini_28pct"`, each an explicit `modules/data_source/mini_{50,28}pct_{type}.yaml`
layer (one per anomaly type: `clean`, `contextual`, `group`, `point`).

**Using a mini dataset changes each architecture's own false-positive rate**
— e.g. Transformer-VAE trends toward a lower FPR at mini scale than at full
scale, because its early-miscalibration-burst mechanism needs enough warmup
data to actually manifest. See `case01_clean_baseline.ipynb`, which runs the
same suite at all three scales side by side and compares each architecture
against its own results at the other scales. Treat mini-scale results as
provisional/faster-iteration, not a substitute for the full-scale default.

### `mini_50pct/` (shipped)

The four CSVs under `data/era5/mini_50pct/` (`reanalysis-era5-single-levels-timeseries.csv`,
`era5_point_rare.csv`, `era5_contextual_anomalies.csv`, `era5_group_anomalies.csv`)
cover 1957-03-28 to 2000-05-02 (377,784 rows, ~50% of the full-scale export's
total size): 151,114 rows of warmup immediately before, and 226,670 rows of
test immediately after, the same warmup/test split point (1974-06-23) used
at full scale — a 40:60 warmup/test split, matching the full-scale export's
own proportions (rather than an even 50:50 split). `era5_point_rare.csv`,
`era5_contextual_anomalies.csv`, and `era5_group_anomalies.csv` additionally
carry synthetic point/contextual/group anomalies injected into the real
values, in the test portion only.

Produced from the full-scale source files (`data/era5/full_scale/raw_files/`)
by the same slicing logic as `scripts/slice_real_era5.py`, but with different
warmup/test bounds than that script's own defaults — `lo=151113, hi=528897`
against the full-scale files, rather than the script's default
`WARMUP_YEARS=12, TEST_YEARS=12` (which produce `mini_28pct/` below instead).

### `mini_28pct/` (shipped)

An earlier, smaller 50:50 (12y/12y, 210,384-row, ~28% of the full-scale row
count) version of the same four files, kept at `data/era5/mini_28pct/` for
`case01_clean_baseline.ipynb`'s three-way dataset-scale comparison. This is
exactly `scripts/slice_real_era5.py`'s own default output (`WARMUP_YEARS=12,
TEST_YEARS=12`, `OUT_DIR` pointed at `mini_28pct/`): 105,192 rows of warmup,
105,192 rows of test, split at the same 1974-06-23 boundary.

## `standalone_tests/`

`standalone_tests/{baseline,finetuned}/*.yaml` (run via `python demo.py` or
directly via `python -m scripts.trial.run_trial <config.yaml>`)
are independent of the `DATA_SOURCE`/data_source-layer system above — each
hardcodes `warmup_path`/`test_path` directly, pointed at the full-scale split
files under `data/era5/full_scale/split/` described above (the same files
`DATA_SOURCE = "full_split_files"` resolves to in the case notebooks).
