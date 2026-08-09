#!/usr/bin/env python3
"""
Build data/era5/*.csv from a real ERA5 export by slicing a short window
centered on the same warmup/test boundary used in the full-scale thesis
experiments (row 302227, 1974-06-23) — 12 years of real warmup immediately
before the split, 12 years of real test immediately after it (50:50 split).

This requires access to the real, full-size source files (not included in
this repo — see DATA_LICENSE.md). It is NOT run automatically; it documents
how data/era5/*.csv were produced and lets you regenerate them if you have
your own copies of the source files.

Usage (from the project root):
    python scripts/slice_real_era5.py --source-dir /path/to/real/ERA5/csvs

Expects, in --source-dir, the real hourly ERA5 export plus the same-schema
anomaly-injected variants (all with the same row count/date range, one
`is_anomaly` + `anomaly_type` column pair per injected file):
    reanalysis-era5-single-levels-timeseries.csv   (clean, no injected anomalies)
    era5_point_rare.csv                             (point anomalies)
    era5_group_realistic.csv                        (group anomalies)
    era5_contextual_no_spring.csv                   (contextual anomalies)
"""

import argparse
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "era5"

# Matches the full-scale thesis split: warmup_end = row 302227 (1974-06-23).
WARMUP_END_FULL = 302227
YEAR_HOURS = 8766             # 365.25 days * 24h
WARMUP_YEARS = 12
TEST_YEARS = 12

SOURCE_TO_OUTPUT = {
    "reanalysis-era5-single-levels-timeseries.csv": "reanalysis-era5-single-levels-timeseries.csv",
    "era5_point_rare.csv":                          "era5_point_rare.csv",
    "era5_group_realistic.csv":                     "era5_group_anomalies.csv",
    "era5_contextual_no_spring.csv":                "era5_contextual_anomalies.csv",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True,
                         help="Directory holding the real, full-size ERA5 CSVs")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    lo = WARMUP_END_FULL - WARMUP_YEARS * YEAR_HOURS
    hi = WARMUP_END_FULL + TEST_YEARS * YEAR_HOURS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for src_name, out_name in SOURCE_TO_OUTPUT.items():
        src_path = source_dir / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source file: {src_path}")

        df = pd.read_csv(src_path, low_memory=False)
        sliced = df.iloc[lo:hi].reset_index(drop=True)

        out_path = OUT_DIR / out_name
        sliced.to_csv(out_path, index=False)

        n_anom = int(sliced["is_anomaly"].sum()) if "is_anomaly" in sliced.columns else 0
        print(f"{out_name}: {len(sliced)} rows "
              f"({sliced['valid_time'].iloc[0]} to {sliced['valid_time'].iloc[-1]}), "
              f"{n_anom} anomalous rows")

    print(f"\nWrote {len(SOURCE_TO_OUTPUT)} real ERA5 slices to {OUT_DIR}")
    print(f"warmup_end for these files is {WARMUP_YEARS * YEAR_HOURS} "
          "(matches the existing experiments/era5/**/*.yaml configs — no config changes needed)")


if __name__ == "__main__":
    main()
