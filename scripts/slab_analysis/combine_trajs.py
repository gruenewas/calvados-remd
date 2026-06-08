from pathlib import Path
import MDAnalysis as mda

def combine_nup98_trajs():
    folder = Path("/home/sgruenew/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_highT_ladder/NUP98_rt0.000sig_284.14K")

    top = folder / "top.pdb"
    traj_second = folder / "NUP98_rt0.000sig_284.14K.dcd"
    traj_first = folder / "NUP98_rt0.000sig_284.14K_first1.7mus.dcd"
    out = folder / "NUP98_rt0.000sig_284.14K_combined.dcd"

    u = mda.Universe(str(top), [str(traj_first), str(traj_second)])

    with mda.Writer(str(out), n_atoms=u.atoms.n_atoms) as W:
        for ts in u.trajectory:
            W.write(u.atoms)

    print(f"Wrote combined trajectory to: {out}")
    print(f"Total frames: {len(u.trajectory)}")
