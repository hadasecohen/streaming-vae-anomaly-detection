# Data License

## Mini ERA5 datasets (`data/era5/`)

The four CSVs under `data/era5/` (`reanalysis-era5-single-levels-timeseries.csv`,
`era5_point_rare.csv`, `era5_contextual_anomalies.csv`, `era5_group_anomalies.csv`)
are a real, 24-year slice of ERA5 hourly reanalysis data — not synthetic. Each
file covers 1962-06-23 to 1986-06-23 (210,384 rows): 12 years of warmup
immediately before, and 12 years of test immediately after, the same
warmup/test split point (1974-06-23) used in the full-scale experiments this
work is based on — an even 50:50 warmup/test split. `era5_point_rare.csv`,
`era5_contextual_anomalies.csv`, and `era5_group_anomalies.csv` additionally
carry synthetic point/contextual/group anomalies injected into the real
values, in the test portion only.

Source: ERA5 hourly reanalysis, European Centre for Medium-Range Weather
Forecasts (ECMWF), distributed via the Copernicus Climate Data Store (CDS):
https://cds.climate.copernicus.eu/ — one surface grid point (Canterbury, New
Zealand). Used under the Copernicus license, which permits reuse with
attribution.

Produced from the full-scale source files with `scripts/slice_real_era5.py`
(requires your own copy of those source files — not included here; see that
script's docstring for the expected layout).

## Full-scale real ERA5 data

`experiments/era5/` is written for the full-scale real ERA5 export (755,568
rows, 1940–2026) used during the original experiments. This repository
includes only the 210,384-row slice described above, not the full export;
swap `data_path` and `warmup_end` in any `experiments/era5/**/*.yaml` to
point at your own full download to reproduce the original scale.
