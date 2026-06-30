#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def clean_column_name(col):
    """Clean CALVADOS/OpenMM log column names like '#"Step"'."""
    return col.strip().lstrip("#").strip('"')


def read_speed_log(log_file):
    """
    Read a CALVADOS/OpenMM tab-separated simulation log.

    Expected structure:
        #"Step"    "Potential Energy (kJ/mole)"    "Speed (ns/day)"    "Elapsed Time (s)"
        10000      ...                             0                   ...
        20000      ...                             874                 ...
    """

    df = pd.read_csv(log_file, sep="\t")
    df.columns = [clean_column_name(c) for c in df.columns]

    required = ["Step", "Speed (ns/day)", "Elapsed Time (s)"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {log_file}. "
                f"Found columns: {list(df.columns)}"
            )

    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Speed (ns/day)"] = pd.to_numeric(df["Speed (ns/day)"], errors="coerce")
    df["Elapsed Time (s)"] = pd.to_numeric(df["Elapsed Time (s)"], errors="coerce")

    df = df.dropna(subset=["Step", "Speed (ns/day)", "Elapsed Time (s)"])

    return df


def calculate_mean_speeds(
    folder_pattern="*",
    base_path=".",
    skip_zero=True,
    min_step=None,
    output_csv=None,
):
    """
    Calculate mean simulation speeds from folders matching a wildcard.

    Each folder is expected to contain:
        <foldername>/<foldername>.log
    """

    base_path = Path(base_path)
    folders = sorted([p for p in base_path.glob(folder_pattern) if p.is_dir()])

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No folders found matching pattern '{folder_pattern}' in {base_path}"
        )

    results = []
    all_speed_values = []

    for folder in folders:
        log_file = folder / f"{folder.name}.log"

        if not log_file.exists():
            print(f"Skipping {folder.name}: log file not found: {log_file}")
            continue

        df = read_speed_log(log_file)

        if skip_zero:
            df = df[df["Speed (ns/day)"] > 0]

        if min_step is not None:
            df = df[df["Step"] >= min_step]

        if len(df) == 0:
            print(f"Skipping {folder.name}: no valid speed entries after filtering.")
            continue

        mean_speed = df["Speed (ns/day)"].mean()
        median_speed = df["Speed (ns/day)"].median()
        std_speed = df["Speed (ns/day)"].std()
        n_entries = len(df)

        final_step = df["Step"].max()
        final_elapsed_s = df["Elapsed Time (s)"].max()

        results.append(
            {
                "folder": folder.name,
                "log_file": str(log_file),
                "n_entries": n_entries,
                "final_step": int(final_step),
                "final_elapsed_s": final_elapsed_s,
                "mean_speed_ns_per_day": mean_speed,
                "median_speed_ns_per_day": median_speed,
                "std_speed_ns_per_day": std_speed,
            }
        )

        all_speed_values.extend(df["Speed (ns/day)"].to_numpy())

    if len(results) == 0:
        raise RuntimeError("No valid log files were processed.")

    result_df = pd.DataFrame(results)

    mean_of_folder_means = result_df["mean_speed_ns_per_day"].mean()
    pooled_mean = pd.Series(all_speed_values).mean()

    print("\nPer-folder mean simulation speeds")
    print("-" * 100)

    for _, row in result_df.iterrows():
        print(
            f"{row['folder']}: "
            f"mean = {row['mean_speed_ns_per_day']:.2f} ns/day, "
            f"median = {row['median_speed_ns_per_day']:.2f} ns/day, "
            f"n = {int(row['n_entries'])}"
        )

    print("\nAcross folders")
    print("-" * 100)
    print(f"Mean of folder means: {mean_of_folder_means:.2f} ns/day")
    print(f"Pooled mean over all log entries: {pooled_mean:.2f} ns/day")

    if output_csv is not None:
        output_csv = Path(output_csv)
        result_df.to_csv(output_csv, index=False)
        print(f"\nWrote per-folder summary to: {output_csv}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate mean simulation speeds from CALVADOS/OpenMM log files. "
            "Each folder should contain a log file named <foldername>.log."
        )
    )

    parser.add_argument(
        "--base-path",
        default=".",
        help="Base directory containing simulation folders. Default: current directory.",
    )

    parser.add_argument(
        "--pattern",
        default="*",
        help='Wildcard pattern for folders, e.g. "NUP98_*K*" or "hpl2+lin13_*K".',
    )

    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include speed entries equal to 0. By default, zero-speed entries are ignored.",
    )

    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only include log entries with Step >= min_step.",
    )

    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV file for the per-folder summary.",
    )

    args = parser.parse_args()

    calculate_mean_speeds(
        folder_pattern=args.pattern,
        base_path=args.base_path,
        skip_zero=not args.include_zero,
        min_step=args.min_step,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()