#!/usr/bin/env python3

import argparse
from pathlib import Path


def parse_step(line):
    """
    Parse the first column of an OpenMM StateDataReporter log line.

    Expected format:
    #"Step"    "Potential Energy (kJ/mole)"    "Speed (ns/day)"    "Elapsed Time (s)"
    50000      -12345.6                         123.4              10.2
    """
    line = line.strip()

    if not line:
        return None

    # Skip header/comment lines
    if line.startswith("#"):
        return None

    # Split tab-separated if possible; otherwise fall back to whitespace
    fields = line.split("\t")
    if len(fields) == 1:
        fields = line.split()

    if not fields:
        return None

    first = fields[0].strip().replace('"', "")

    try:
        return int(first)
    except ValueError:
        return None


def scan_log(logfile, flag_equal=False):
    logfile = Path(logfile)

    if not logfile.is_file():
        raise FileNotFoundError(f"File not found: {logfile}")

    restarts = []

    previous_step = None
    previous_line_number = None

    segment_start_line = None
    segment_start_step = None
    segments = []

    with logfile.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            step = parse_step(line)

            if step is None:
                continue

            if segment_start_line is None:
                segment_start_line = line_number
                segment_start_step = step

            if previous_step is not None:
                is_restart = step < previous_step
                is_equal_repeat = flag_equal and step == previous_step

                if is_restart or is_equal_repeat:
                    restarts.append(
                        {
                            "line": line_number,
                            "previous_line": previous_line_number,
                            "previous_step": previous_step,
                            "current_step": step,
                            "type": "decrease" if is_restart else "equal",
                        }
                    )

                    segments.append(
                        {
                            "start_line": segment_start_line,
                            "end_line": previous_line_number,
                            "start_step": segment_start_step,
                            "end_step": previous_step,
                        }
                    )

                    segment_start_line = line_number
                    segment_start_step = step

            previous_step = step
            previous_line_number = line_number

    if segment_start_line is not None:
        segments.append(
            {
                "start_line": segment_start_line,
                "end_line": previous_line_number,
                "start_step": segment_start_step,
                "end_step": previous_step,
            }
        )

    return restarts, segments


def main():
    parser = argparse.ArgumentParser(
        description="Scan an OpenMM/CALVADOS log file for restart events indicated by decreasing step numbers."
    )
    parser.add_argument("logfile", help="Path to the log file, e.g. NUP98.log")
    parser.add_argument(
        "--flag-equal",
        action="store_true",
        help="Also flag repeated identical step numbers as possible duplicate output.",
    )

    args = parser.parse_args()

    restarts, segments = scan_log(args.logfile, flag_equal=args.flag_equal)

    print(f"\nScanned log file: {args.logfile}")
    print(f"Detected {len(restarts)} restart/overlap event(s).\n")

    if restarts:
        print("Restart / overlap events:")
        print("-" * 80)
        for r in restarts:
            print(
                f"Line {r['line']}: step changed from "
                f"{r['previous_step']} at line {r['previous_line']} "
                f"to {r['current_step']} at line {r['line']} "
                f"({r['type']})"
            )
        print()

    print("Detected monotonic log segments:")
    print("-" * 80)
    for i, s in enumerate(segments, start=1):
        print(
            f"Segment {i}: "
            f"lines {s['start_line']}--{s['end_line']}, "
            f"steps {s['start_step']}--{s['end_step']}, "
            f"time [mus] {s['start_step']*1e-8:.2f}--{s['end_step']*1e-8:.2f}" 
        )


if __name__ == "__main__":
    main()
