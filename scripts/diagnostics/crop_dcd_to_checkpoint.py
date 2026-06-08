#!/usr/bin/env python3

from pathlib import Path
import argparse
import MDAnalysis as mda


def crop_to_checkpoint_frame(topology, trajectory, checkpoint_interval, output=None):
    topology = Path(topology)
    trajectory = Path(trajectory)

    if output is None:
        output = trajectory.with_name(trajectory.stem + "_checkpoint_crop.dcd")
    else:
        output = Path(output)

    if output.resolve() == trajectory.resolve():
        raise ValueError("Output file must not be the same as the input trajectory.")

    u = mda.Universe(str(topology), str(trajectory))
    n_frames = len(u.trajectory)

    n_keep = (n_frames // checkpoint_interval) * checkpoint_interval

    print(f"Input trajectory: {trajectory}")
    print(f"Topology:         {topology}")
    print(f"Total frames:     {n_frames}")
    print(f"Interval:         {checkpoint_interval} frames")
    print(f"Keeping frames:   {n_keep}")
    print(f"Removing frames:  {n_frames - n_keep}")
    print(f"Output:           {output}")

    if n_keep == 0:
        raise ValueError(
            "No complete checkpoint interval found. "
            "The trajectory is shorter than the checkpoint interval."
        )

    with mda.Writer(str(output), n_atoms=u.atoms.n_atoms) as writer:
        for ts in u.trajectory[:n_keep]:
            writer.write(u.atoms)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop a DCD trajectory to the last frame matching a checkpoint interval."
    )

    parser.add_argument(
        "-p", "--topology",
        required=True,
        help="Topology file, e.g. top.pdb",
    )

    parser.add_argument(
        "-t", "--trajectory",
        required=True,
        help="Input DCD trajectory",
    )

    parser.add_argument(
        "-i", "--checkpoint-interval",
        required=True,
        type=int,
        help="Checkpoint writing interval in trajectory frames",
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output DCD file. Default: <input>_checkpoint_crop.dcd",
    )

    args = parser.parse_args()

    crop_to_checkpoint_frame(
        topology=args.topology,
        trajectory=args.trajectory,
        checkpoint_interval=args.checkpoint_interval,
        output=args.output,
    )