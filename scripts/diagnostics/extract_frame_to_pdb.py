#!/usr/bin/env python3

import argparse
from pathlib import Path
import MDAnalysis as mda


def extract_frame_to_pdb(topology, trajectory, frame, output=None):
    """
    Extract one frame from a DCD trajectory and write it as a PDB.

    Parameters
    ----------
    topology : str or Path
        Input topology PDB file.
    trajectory : str or Path
        Input DCD trajectory.
    frame : int
        Frame index to extract. Uses Python-style 0-based indexing.
    output : str or Path, optional
        Output PDB file. If None, an automatic name is used.
    """

    topology = Path(topology)
    trajectory = Path(trajectory)

    if output is None:
        output = trajectory.with_name(f"{trajectory.stem}_frame{frame}.pdb")
    else:
        output = Path(output)

    u = mda.Universe(str(topology), str(trajectory))

    n_frames = len(u.trajectory)

    if frame < 0:
        frame = n_frames + frame

    if frame < 0 or frame >= n_frames:
        raise IndexError(
            f"Frame {frame} is out of range. "
            f"Trajectory contains {n_frames} frames indexed from 0 to {n_frames - 1}."
        )

    u.trajectory[frame]
    u.atoms.write(str(output))

    print(f"Wrote frame {frame} / {n_frames - 1} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a single frame from a DCD trajectory and write it as a PDB."
    )

    parser.add_argument(
        "-p", "--topology",
        required=True,
        help="Input topology PDB file, e.g. top.pdb",
    )

    parser.add_argument(
        "-t", "--trajectory",
        required=True,
        help="Input DCD trajectory file.",
    )

    parser.add_argument(
        "-f", "--frame",
        required=True,
        type=int,
        help="Frame index to extract. Uses 0-based indexing. Use -1 for the last frame.",
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PDB file. Optional.",
    )

    args = parser.parse_args()

    extract_frame_to_pdb(
        topology=args.topology,
        trajectory=args.trajectory,
        frame=args.frame,
        output=args.output,
    )