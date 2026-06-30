#!/usr/bin/env python3

from pathlib import Path
import argparse
import MDAnalysis as mda


DEFAULT_KEEP_SLICES = {
    0: [(0, 10000), (10089, None)],
    1: [(0, 10000), (10066, None)],
    2: [(0, 8000),  (9960, None)],
    3: [(0, 10000), (10086, None)],
    4: [(0, 10000), (10043, None)],
    5: [(0, 10000), (10109, None)],
}


def create_continuous_trajectory(
    folder,
    sysname,
    keep_slices,
    topology="top.pdb",
    output_suffix="_continuous",
    overwrite=False,
):
    """
    Create one cleaned continuous DCD trajectory by removing known overlap frames.

    Parameters
    ----------
    folder : Path
        Folder containing topology and trajectory.
    sysname : str
        System name. Input trajectory is expected as <sysname>.dcd.
    keep_slices : list[tuple[int, int or None]]
        Frame slices to keep, using Python-style indexing.
    topology : str
        Topology filename inside folder.
    output_suffix : str
        Suffix for output DCD.
    overwrite : bool
        Whether to overwrite existing output.
    """

    folder = Path(folder)
    top = folder / topology
    traj = folder / f"{sysname}.dcd"
    out = folder / f"{sysname}{output_suffix}.dcd"

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not top.exists():
        raise FileNotFoundError(f"Topology not found: {top}")

    if not traj.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj}")

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out}\n"
            "Use --overwrite if you want to replace it."
        )

    u = mda.Universe(str(top), str(traj))
    n_frames = len(u.trajectory)

    kept_frames = 0

    print(f"\nProcessing {sysname}")
    print(f"  Topology:   {top}")
    print(f"  Trajectory: {traj}")
    print(f"  Frames:     {n_frames}")
    print(f"  Output:     {out}")

    with mda.Writer(str(out), n_atoms=u.atoms.n_atoms) as writer:
        for start, stop in keep_slices:
            original_stop = stop

            if stop is None:
                stop = n_frames

            if start < 0 or start >= n_frames:
                raise ValueError(
                    f"Invalid start frame {start} for {sysname}. "
                    f"Trajectory has {n_frames} frames."
                )

            if stop < start:
                raise ValueError(
                    f"Invalid slice {start}:{original_stop} for {sysname}. "
                    "Stop must be larger than start."
                )

            if stop > n_frames:
                raise ValueError(
                    f"Invalid stop frame {stop} for {sysname}. "
                    f"Trajectory has only {n_frames} frames."
                )

            print(f"  Keeping frames {start}:{original_stop}")

            for ts in u.trajectory[start:stop]:
                writer.write(u.atoms)
                kept_frames += 1

    removed_frames = n_frames - kept_frames

    print(f"  Kept frames:    {kept_frames}")
    print(f"  Removed frames: {removed_frames}")

    return {
        "sysname": sysname,
        "input_frames": n_frames,
        "kept_frames": kept_frames,
        "removed_frames": removed_frames,
        "output": str(out),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create continuous NUP98 DCD trajectories by removing known overlap frames "
            "after failed checkpoint restarts."
        )
    )

    parser.add_argument(
        "--base-path",
        default=".",
        help="Directory containing the replica folders. Default: current directory.",
    )

    parser.add_argument(
        "--prefix",
        default="NUP98_300.15K",
        help="Common folder/sysname prefix. Default: NUP98_300.15K",
    )

    parser.add_argument(
        "--replicas",
        nargs="+",
        type=int,
        default=sorted(DEFAULT_KEEP_SLICES.keys()),
        help="Replica indices to process. Default: 0 1 2 3 4 5",
    )

    parser.add_argument(
        "--topology",
        default="top.pdb",
        help="Topology filename inside each replica folder. Default: top.pdb",
    )

    parser.add_argument(
        "--output-suffix",
        default="_continuous",
        help="Suffix for cleaned output DCD. Default: _continuous",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output trajectories.",
    )

    args = parser.parse_args()

    base_path = Path(args.base_path)

    summaries = []

    for replica in args.replicas:
        if replica not in DEFAULT_KEEP_SLICES:
            raise ValueError(
                f"No default keep slices defined for replica {replica}. "
                f"Available replicas: {sorted(DEFAULT_KEEP_SLICES.keys())}"
            )

        sysname = f"{args.prefix}_{replica}"
        folder = base_path / sysname

        summary = create_continuous_trajectory(
            folder=folder,
            sysname=sysname,
            keep_slices=DEFAULT_KEEP_SLICES[replica],
            topology=args.topology,
            output_suffix=args.output_suffix,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    print("\nSummary")
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['sysname']}: "
            f"input={s['input_frames']} frames, "
            f"kept={s['kept_frames']} frames, "
            f"removed={s['removed_frames']} frames, "
            f"output={s['output']}"
        )


if __name__ == "__main__":
    main()