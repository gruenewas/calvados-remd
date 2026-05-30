import MDAnalysis as mda
import numpy as np
import pandas as pd
import tqdm


def average_rg(folder, sysname, residues_csv, comp_dict, residue_key_col=None, start=None,stop=None,step=None):
    """
    Calculate the average radius of gyration for a trajectory after assigning
    coarse-grained bead masses from residue molecular weights.

    Expects:
      folder/top.pdb
      folder/{sysname}.dcd

    residues_csv must contain a column named 'MW'.
    """

    folder = Path(folder)
    top = folder / "top.pdb"
    traj = folder / f"{sysname}.dcd"

    u = mda.Universe(str(top), str(traj))

    residues = pd.read_csv(residues_csv)
    if "MW" not in residues.columns:
        raise ValueError("residues_csv must contain a column named 'MW'.")

    # Try common CALVADOS residue-name columns unless one is supplied.
    if residue_key_col is None:
        candidates = ["resname", "ResName", "three", "Three", "one", "One", "AA", "aa", "residue", "Residue"]
        residue_key_col = next((c for c in candidates if c in residues.columns), None)

    if residue_key_col is None:
        raise ValueError(
            "Could not infer residue key column. Pass residue_key_col explicitly, "
            "for example residue_key_col='one' or residue_key_col='three'."
        )

    mw_by_resname = dict(
        zip(
            residues[residue_key_col].astype(str).str.strip(),
            residues["MW"].astype(float),
        )
    )

    masses = np.empty(u.atoms.n_atoms, dtype=float)
    missing = set()

    for atom in u.atoms:
        resname = atom.resname.strip()
        if resname not in mw_by_resname:
            missing.add(resname)
            masses[atom.index] = np.nan
        else:
            masses[atom.index] = mw_by_resname[resname]

    if missing:
        raise ValueError(
            f"Missing molecular weights for residue names in topology: {sorted(missing)}"
        )

    u.add_TopologyAttr("masses", masses)

    traj_sel = u.trajectory[start:stop:step]
    nframes = len(traj_sel)

    rgs = {}
    for key,value in comp_dict.items():
        rgs[key] = np.zeros((nframes,len(u.segments[value[0]:value[1]])))
    
    for fidx,frame in tqdm.tqdm(enumerate(traj_sel),total=nframes):
        for key,value in comp_dict.items():
            for cidx,chain in enumerate(u.segments[value[0]:value[1]]):
                rgs[key][fidx,cidx] = chain.atoms.radius_of_gyration()

    return {key : {
        "mean_rg": float(np.mean(value)),
        "std_rg": float(np.std(value)),
        "rgs": value} for key,value in rgs.items()}