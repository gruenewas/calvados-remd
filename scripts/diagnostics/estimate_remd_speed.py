#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def estimate_remd_speed(
    csv_file,
    segment_steps,
    dt_ps=10.0,
    time_col="time",
    segment_col="segment",
    output=None,
):
    """
    Estimate REMD simulation speed from segment number and wall-clock timestamps.

    Parameters
    ----------
    csv_file : str or Path
        REMD log CSV file.
    segment_steps : int
        Number of MD steps simulated per REMD segment.
    dt_ps : float
        Time step in ps. Default is 10 ps because this was specified by the user.
        For standard CALVADOS/OpenMM with dt = 0.01 ps, use --dt-ps 0.01.
    time_col : str
        Name of the timestamp column.
    segment_col : str
        Name of the segment column.
    output : str or Path or None
        Optional output CSV for per-segment speed estimates.

    Returns
    -------
    summary : dict
        Dictionary with overall speed statistics.
    """

    csv_file = Path(csv_file)

    df = pd.read_csv(csv_file)

    if time_col not in df.columns:
        raise ValueError(f"Could not find time column '{time_col}' in {csv_file}")

    if segment_col not in df.columns:
        raise ValueError(f"Could not find segment column '{segment_col}' in {csv_file}")

    df[time_col] = pd.to_datetime(df[time_col])
    df[segment_col] = df[segment_col].astype(int)

    # Multiple exchange attempts are logged per segment.
    # Use one representative timestamp per segment.
    seg = (
        df.groupby(segment_col, as_index=False)[time_col]
        .min()
        .sort_values(segment_col)
        .reset_index(drop=True)
    )

    if len(seg) < 2:
        raise ValueError("Need at least two logged segments to estimate a speed.")

    ns_per_segment = segment_steps * dt_ps / 1000.0

    seg["delta_segment"] = seg[segment_col].diff()
    seg["delta_time_s"] = seg[time_col].diff().dt.total_seconds()

    seg["simulated_ns"] = seg["delta_segment"] * ns_per_segment
    seg["walltime_days"] = seg["delta_time_s"] / 86400.0
    seg["speed_ns_per_day"] = seg["simulated_ns"] / seg["walltime_days"]

    # Remove first row and possible invalid rows
    speed_df = seg.dropna().copy()
    speed_df = speed_df[speed_df["delta_time_s"] > 0]
    speed_df = speed_df[speed_df["delta_segment"] > 0]

    first_segment = int(seg[segment_col].iloc[0])
    last_segment = int(seg[segment_col].iloc[-1])
    first_time = seg[time_col].iloc[0]
    last_time = seg[time_col].iloc[-1]

    total_segments = last_segment - first_segment
    total_simulated_ns = total_segments * ns_per_segment
    total_walltime_days = (last_time - first_time).total_seconds() / 86400.0
    overall_speed = total_simulated_ns / total_walltime_days

    summary = {
        "csv_file": str(csv_file),
        "first_segment": first_segment,
        "last_segment": last_segment,
        "segment_steps": segment_steps,
        "dt_ps": dt_ps,
        "ns_per_segment": ns_per_segment,
        "total_simulated_ns": total_simulated_ns,
        "total_walltime_days": total_walltime_days,
        "overall_speed_ns_per_day": overall_speed,
        "mean_interval_speed_ns_per_day": speed_df["speed_ns_per_day"].mean(),
        "median_interval_speed_ns_per_day": speed_df["speed_ns_per_day"].median(),
    }

    print("\nREMD speed estimate")
    print("-" * 80)
    print(f"CSV file:                  {csv_file}")
    print(f"Segment range:             {first_segment} -- {last_segment}")
    print(f"Segment steps:             {segment_steps}")
    print(f"Time step:                 {dt_ps:g} ps")
    print(f"Simulation per segment:    {ns_per_segment:.6g} ns")
    print(f"Total simulated time:      {total_simulated_ns:.6g} ns")
    print(f"Total wall time:           {total_walltime_days:.6g} days")
    print(f"Overall speed:             {overall_speed:.3f} ns/day")
    print(f"Mean interval speed:       {summary['mean_interval_speed_ns_per_day']:.3f} ns/day")
    print(f"Median interval speed:     {summary['median_interval_speed_ns_per_day']:.3f} ns/day")

    if output is not None:
        output = Path(output)
        speed_df.to_csv(output, index=False)
        print(f"\nWrote per-segment speed estimates to: {output}")

    return summary, speed_df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate REMD simulation speed in ns/day from segment number and timestamps."
    )

    parser.add_argument(
        "csv_file",
        help="Input REMD log CSV file.",
    )

    parser.add_argument(
        "--segment-steps",
        type=int,
        required=True,
        help="Number of MD steps per REMD segment.",
    )

    parser.add_argument(
        "--dt-ps",
        type=float,
        default=10.0,
        help=(
            "MD timestep in ps. Default: 10 ps. "
            "For standard CALVADOS/OpenMM with 10 fs timestep, use --dt-ps 0.01."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV for per-segment speed estimates.",
    )

    args = parser.parse_args()

    estimate_remd_speed(
        csv_file=args.csv_file,
        segment_steps=args.segment_steps,
        dt_ps=args.dt_ps,
        output=args.output,
    )


if __name__ == "__main__":
    main()