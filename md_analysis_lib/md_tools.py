import MDAnalysis as mda
from MDAnalysis.lib import distances
import numpy as np
import pandas as pd
import tqdm
import warnings
from pathlib import Path


def _validate_positive_float(value, name):
    """Return value as float after checking that it is finite and positive."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _prepare_histogram_edges(bins, max_distance):
    """Validate histogram settings and return bin edges when already known."""
    if bins is None:
        bins = 100

    if np.isscalar(bins):
        if isinstance(bins, (float, np.floating)) and not float(bins).is_integer():
            raise ValueError("bins must be a positive integer or a 1D array of bin edges.")
        n_bins = int(bins)
        if n_bins <= 0:
            raise ValueError("bins must be a positive integer or a 1D array of bin edges.")

        if max_distance is None:
            return None, n_bins

        max_distance = _validate_positive_float(max_distance, "max_distance")
        return np.linspace(0.0, max_distance, n_bins + 1), n_bins

    bin_edges = np.asarray(bins, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bins must be a 1D array with at least two bin edges.")
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("All histogram bin edges must be finite.")
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("Histogram bin edges must be strictly increasing.")

    if max_distance is not None:
        warnings.warn(
            "max_distance is ignored when bins is an explicit array of bin edges.",
            RuntimeWarning,
            stacklevel=2,
        )

    return bin_edges, bin_edges.size - 1


def _prefixed_output_path(output_prefix, suffix):
    """Return output_prefix with suffix appended to the path name."""
    prefix = Path(output_prefix)
    return prefix.parent / f"{prefix.name}{suffix}"


def _frame_box(ts, use_pbc):
    """Return a validated MDAnalysis box for distance calculations."""
    if not use_pbc:
        return None

    box = ts.dimensions
    if box is None or len(box) < 3 or not np.all(np.asarray(box[:3], dtype=float) > 0):
        raise ValueError(
            "use_pbc=True, but the trajectory frame has missing or invalid box "
            "dimensions. Provide unit-cell dimensions in the trajectory/topology "
            "or call with use_pbc=False."
        )
    return box


def _atom_resindices(atomgroup):
    """Return residue indices for an AtomGroup with a clear error if unavailable."""
    try:
        return np.asarray(atomgroup.resindices)
    except (AttributeError, mda.exceptions.NoDataError) as exc:
        try:
            return np.asarray([atom.residue.ix for atom in atomgroup])
        except (AttributeError, mda.exceptions.NoDataError) as fallback_exc:
            raise ValueError(
                "exclude_same_residue=True requires residue information in the topology."
            ) from fallback_exc


def _atom_segindices(atomgroup):
    """Return segment indices for an AtomGroup with a clear error if unavailable."""
    try:
        return np.asarray(atomgroup.segindices)
    except (AttributeError, mda.exceptions.NoDataError) as exc:
        try:
            return np.asarray([atom.segment.ix for atom in atomgroup])
        except (AttributeError, mda.exceptions.NoDataError) as fallback_exc:
            raise ValueError(
                "exclude_same_segment=True requires segment/chain information in the topology."
            ) from fallback_exc


def _valid_pair_indices(atomgroup, exclude_same_residue, exclude_same_segment):
    """Build unique i < j pair indices and the topology-based exclusion mask."""
    n_particles = atomgroup.n_atoms
    pair_i, pair_j = np.triu_indices(n_particles, k=1)

    pair_mask = None
    if exclude_same_residue or exclude_same_segment:
        pair_mask = np.ones(pair_i.size, dtype=bool)

        if exclude_same_residue:
            resindices = _atom_resindices(atomgroup)
            pair_mask &= resindices[pair_i] != resindices[pair_j]

        if exclude_same_segment:
            segindices = _atom_segindices(atomgroup)
            pair_mask &= segindices[pair_i] != segindices[pair_j]

    if pair_mask is None:
        return pair_i, pair_j, None, pair_i, pair_j

    return pair_i, pair_j, pair_mask, pair_i[pair_mask], pair_j[pair_mask]


def _filter_topological_exclusions(
    pair_indices,
    pair_distances,
    resindices,
    segindices,
    exclude_same_residue,
    exclude_same_segment,
):
    """Apply residue/segment exclusions to capped-distance contact pairs."""
    if pair_indices.size == 0:
        return pair_indices, pair_distances

    keep = np.ones(pair_indices.shape[0], dtype=bool)
    if exclude_same_residue:
        keep &= resindices[pair_indices[:, 0]] != resindices[pair_indices[:, 1]]
    if exclude_same_segment:
        keep &= segindices[pair_indices[:, 0]] != segindices[pair_indices[:, 1]]

    return pair_indices[keep], pair_distances[keep]


def _self_capped_pairs(positions, cutoff, box):
    """Return unique i < j pairs within cutoff using MDAnalysis capped distances."""
    if hasattr(distances, "self_capped_distance"):
        pair_indices, pair_distances = distances.self_capped_distance(
            positions,
            cutoff,
            box=box,
            return_distances=True,
        )
        return pair_indices.astype(int, copy=False), pair_distances

    pair_indices, pair_distances = distances.capped_distance(
        positions,
        positions,
        cutoff,
        box=box,
        return_distances=True,
    )
    unique_pair_mask = pair_indices[:, 0] < pair_indices[:, 1]
    return pair_indices[unique_pair_mask].astype(int, copy=False), pair_distances[unique_pair_mask]


def analyze_interparticle_distances(
    topology,
    trajectory,
    selection="all",
    cutoff=None,
    start=None,
    stop=None,
    step=None,
    exclude_same_residue=True,
    exclude_same_segment=False,
    use_pbc=True,
    bins=100,
    max_distance=None,
    return_pair_distances=False,
    output_prefix=None,
):
    """
    Analyze interparticle distances and contact/neighbor counts from a trajectory.

    This is intended for comparing CALVADOS LLPS simulations, for example asking
    whether increasing the soft-core parameter kappa shifts a condensate toward
    shorter pair distances and larger contact numbers.

    Parameters
    ----------
    topology : str or pathlib.Path
        Topology file readable by MDAnalysis, for example ``top.pdb``.
    trajectory : str or pathlib.Path
        Trajectory file readable by MDAnalysis, for example a ``.dcd`` file.
    selection : str, optional
        MDAnalysis atom selection string. Defaults to ``"all"``.
    cutoff : float
        Distance cutoff used to define neighbors/contacts. A pair is counted as a
        contact when its distance is strictly smaller than ``cutoff``.
    start, stop, step : int or None, optional
        Frame slicing arguments applied as ``u.trajectory[start:stop:step]``.
    exclude_same_residue : bool, optional
        If True, ignore pairs whose atoms belong to the same residue.
    exclude_same_segment : bool, optional
        If True, ignore pairs whose atoms belong to the same segment/chain/molecule.
        This is useful for measuring intermolecular contacts only.
    use_pbc : bool, optional
        If True, use periodic boundary conditions with the unit-cell dimensions
        stored in each trajectory frame.
    bins : int or array-like, optional
        Histogram bin count or explicit bin edges for pair distances. If an integer
        is given and ``max_distance`` is None, the trajectory is scanned once to
        determine the largest observed valid pair distance before the main pass.
    max_distance : float or None, optional
        Upper histogram edge when ``bins`` is an integer. Distances beyond this
        value are not included in the histogram, but they are still included in
        frame-level summary statistics.
    return_pair_distances : bool, optional
        If True, return all valid pair distances concatenated across analyzed
        frames. This can be very large and is disabled by default.
    output_prefix : str or pathlib.Path or None, optional
        If provided, save ``<output_prefix>_frame_summary.csv``,
        ``<output_prefix>_distance_histogram.npz``,
        ``<output_prefix>_neighbor_counts.npy``, and
        ``<output_prefix>_nearest_neighbor_distances.npy``.

    Returns
    -------
    dict
        Dictionary with:

        ``frame_summary``
            pandas DataFrame with one row per analyzed frame.
        ``distance_histogram``
            Dictionary with ``counts`` and ``bin_edges`` arrays.
        ``neighbor_counts``
            Concatenated per-particle neighbor counts over analyzed frames.
        ``nearest_neighbor_distances``
            Concatenated finite nearest-neighbor distances over analyzed frames.
        ``pair_distances``
            Only present when ``return_pair_distances=True``.

    Notes
    -----
    MDAnalysis reports distances in the coordinate units of the input files. For
    PDB/DCD workflows these are usually Angstrom, but this function deliberately
    leaves units unlabeled so simulations with different conventions can be
    labeled externally.
    """
    topology = Path(topology)
    trajectory = Path(trajectory)
    if not topology.exists():
        raise FileNotFoundError(f"Topology file does not exist: {topology}")
    if not trajectory.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {trajectory}")

    if cutoff is None:
        raise ValueError("cutoff must be provided and must be a positive number.")
    cutoff = _validate_positive_float(cutoff, "cutoff")

    if not isinstance(selection, str) or not selection.strip():
        raise ValueError("selection must be a non-empty MDAnalysis selection string.")

    if step == 0:
        raise ValueError("step must not be 0.")

    bin_edges, n_bins = _prepare_histogram_edges(bins, max_distance)

    try:
        universe = mda.Universe(str(topology), str(trajectory))
    except Exception as exc:
        raise ValueError(
            f"Could not load topology/trajectory with MDAnalysis: {topology}, {trajectory}"
        ) from exc

    try:
        atomgroup = universe.select_atoms(selection)
    except Exception as exc:
        raise ValueError(f"Invalid MDAnalysis selection: {selection!r}") from exc

    if atomgroup.n_atoms < 2:
        raise ValueError(
            f"Selection {selection!r} contains {atomgroup.n_atoms} atom(s); "
            "at least two particles are required."
        )

    trajectory_slice = universe.trajectory[start:stop:step]
    n_frames = len(trajectory_slice)
    if n_frames == 0:
        raise ValueError(
            "Frame slice selected no frames. Check start, stop, and step values."
        )

    pair_i, pair_j, pair_mask, valid_pair_i, valid_pair_j = _valid_pair_indices(
        atomgroup,
        exclude_same_residue,
        exclude_same_segment,
    )

    resindices = _atom_resindices(atomgroup) if exclude_same_residue else None
    segindices = _atom_segindices(atomgroup) if exclude_same_segment else None

    if bin_edges is None:
        observed_max_distance = 0.0
        observed_pairs = 0
        for ts in universe.trajectory[start:stop:step]:
            box = _frame_box(ts, use_pbc)
            frame_distances = distances.self_distance_array(atomgroup.positions, box=box)
            if pair_mask is not None:
                frame_distances = frame_distances[pair_mask]
            if frame_distances.size:
                observed_pairs += frame_distances.size
                observed_max_distance = max(
                    observed_max_distance,
                    float(np.max(frame_distances)),
                )

        if observed_pairs == 0:
            warnings.warn(
                "No valid particle pairs were found after applying exclusions. "
                "Distance statistics will be NaN and the distance histogram will be empty.",
                RuntimeWarning,
                stacklevel=2,
            )
            observed_max_distance = max(cutoff, 1.0)
        elif observed_max_distance <= 0:
            observed_max_distance = max(cutoff, 1.0)

        bin_edges = np.linspace(0.0, observed_max_distance, n_bins + 1)

    histogram_counts = np.zeros(bin_edges.size - 1, dtype=np.int64)
    frame_rows = []
    all_neighbor_counts = []
    all_nearest_distances = []
    all_pair_distances = [] if return_pair_distances else None

    for ts in universe.trajectory[start:stop:step]:
        box = _frame_box(ts, use_pbc)
        positions = atomgroup.positions

        frame_pair_distances = distances.self_distance_array(positions, box=box)
        if pair_mask is not None:
            valid_pair_distances = frame_pair_distances[pair_mask]
        else:
            valid_pair_distances = frame_pair_distances

        frame_histogram, _ = np.histogram(valid_pair_distances, bins=bin_edges)
        histogram_counts += frame_histogram

        nearest_distances = np.full(atomgroup.n_atoms, np.nan, dtype=float)
        if valid_pair_distances.size:
            np.minimum.at(nearest_distances, valid_pair_i, valid_pair_distances)
            np.minimum.at(nearest_distances, valid_pair_j, valid_pair_distances)
        finite_nearest = nearest_distances[np.isfinite(nearest_distances)]
        all_nearest_distances.append(finite_nearest)

        contact_pairs, contact_distances = _self_capped_pairs(positions, cutoff, box)
        if contact_pairs.size:
            strict_cutoff_mask = contact_distances < cutoff
            contact_pairs = contact_pairs[strict_cutoff_mask]
            contact_distances = contact_distances[strict_cutoff_mask]
            contact_pairs, contact_distances = _filter_topological_exclusions(
                contact_pairs,
                contact_distances,
                resindices,
                segindices,
                exclude_same_residue,
                exclude_same_segment,
            )

        neighbor_counts = np.zeros(atomgroup.n_atoms, dtype=np.int64)
        if contact_pairs.size:
            np.add.at(neighbor_counts, contact_pairs[:, 0], 1)
            np.add.at(neighbor_counts, contact_pairs[:, 1], 1)

        all_neighbor_counts.append(neighbor_counts)
        if return_pair_distances:
            all_pair_distances.append(valid_pair_distances.copy())

        if valid_pair_distances.size:
            pair_percentiles = np.percentile(
                valid_pair_distances,
                [5, 25, 50, 75, 95],
            )
            mean_pair_distance = float(np.mean(valid_pair_distances))
        else:
            pair_percentiles = np.full(5, np.nan)
            mean_pair_distance = np.nan

        if finite_nearest.size:
            mean_nearest_distance = float(np.mean(finite_nearest))
            median_nearest_distance = float(np.median(finite_nearest))
        else:
            mean_nearest_distance = np.nan
            median_nearest_distance = np.nan

        frame_rows.append(
            {
                "frame": int(ts.frame),
                "time": float(ts.time),
                "n_particles": int(atomgroup.n_atoms),
                "n_pairs": int(valid_pair_distances.size),
                "n_pairs_in_histogram": int(frame_histogram.sum()),
                "mean_pair_distance": mean_pair_distance,
                "median_pair_distance": float(pair_percentiles[2]),
                "pair_distance_p05": float(pair_percentiles[0]),
                "pair_distance_p25": float(pair_percentiles[1]),
                "pair_distance_p75": float(pair_percentiles[3]),
                "pair_distance_p95": float(pair_percentiles[4]),
                "n_particles_with_nearest_neighbor": int(finite_nearest.size),
                "mean_nearest_neighbor_distance": mean_nearest_distance,
                "median_nearest_neighbor_distance": median_nearest_distance,
                "mean_neighbor_count": float(np.mean(neighbor_counts)),
                "median_neighbor_count": float(np.median(neighbor_counts)),
                "total_contacts": int(contact_pairs.shape[0]),
            }
        )

    frame_summary = pd.DataFrame(frame_rows)
    neighbor_counts = (
        np.concatenate(all_neighbor_counts)
        if all_neighbor_counts
        else np.empty(0, dtype=np.int64)
    )
    nearest_neighbor_distances = (
        np.concatenate(all_nearest_distances)
        if all_nearest_distances
        else np.empty(0, dtype=float)
    )

    result = {
        "frame_summary": frame_summary,
        "distance_histogram": {
            "counts": histogram_counts,
            "bin_edges": bin_edges,
        },
        "neighbor_counts": neighbor_counts,
        "nearest_neighbor_distances": nearest_neighbor_distances,
    }

    if return_pair_distances:
        result["pair_distances"] = (
            np.concatenate(all_pair_distances)
            if all_pair_distances
            else np.empty(0, dtype=float)
        )

    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        frame_summary.to_csv(
            _prefixed_output_path(output_prefix, "_frame_summary.csv"),
            index=False,
        )
        np.savez_compressed(
            _prefixed_output_path(output_prefix, "_distance_histogram.npz"),
            counts=histogram_counts,
            bin_edges=bin_edges,
        )
        np.save(_prefixed_output_path(output_prefix, "_neighbor_counts.npy"), neighbor_counts)
        np.save(
            _prefixed_output_path(output_prefix, "_nearest_neighbor_distances.npy"),
            nearest_neighbor_distances,
        )

    return result


def average_rg(folder, sysname, residues_csv, comp_dict, residue_key_col=None, start=None,stop=None,step=None,dcd=None,top=None):
    """
    Calculate the average radius of gyration for a trajectory after assigning
    coarse-grained bead masses from residue molecular weights.

    Expects:
      folder/top.pdb
      folder/{sysname}.dcd

    residues_csv must contain a column named 'MW'.
    """

    folder = Path(folder)
    if top is None:
        top = folder / "top.pdb"
    if dcd is None:
        traj = folder / f"{sysname}.dcd"
    else:
        traj = folder / f"{dcd}"

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


def combine_trajectories(
    path,
    traj_names,
    output_name="combined.dcd",
    topology="top.pdb",
    step_size=1,
    start=None,
    stop=None,
    skip_first_frame_after_first=False,
):
    """
    Combine multiple DCD trajectories into one trajectory.

    Parameters
    ----------
    path : str or Path
        Folder containing topology and trajectories.
    traj_names : list[str]
        Trajectory filenames in the order they should be written.
    output_name : str
        Name of the combined output DCD.
    topology : str
        Topology filename, usually 'top.pdb' for CALVADOS.
    step_size : int
        Write every step_size-th frame.
    start, stop : int or None
        Optional frame slicing applied to each input trajectory.
    skip_first_frame_after_first : bool
        If True, skips the first frame of every trajectory after the first.
        Useful if trajectory boundaries contain duplicated frames.
    """

    path = Path(path)
    top = path / topology
    out = path / output_name

    traj_paths = [path / name for name in traj_names]

    for fp in [top, *traj_paths]:
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

    # Use first trajectory to get atom count
    u0 = mda.Universe(str(top), str(traj_paths[0]))
    n_atoms = u0.atoms.n_atoms

    n_written = 0

    with mda.Writer(str(out), n_atoms=n_atoms) as W:
        for i, traj in enumerate(traj_paths):
            u = mda.Universe(str(top), str(traj))

            if u.atoms.n_atoms != n_atoms:
                raise ValueError(f"Atom count mismatch in {traj}")

            local_start = start
            if skip_first_frame_after_first and i > 0:
                local_start = 1 if start is None else max(start, 1)

            print(f"Stitching trajectory {i}")
            for ts in tqdm.tqdm(u.trajectory[local_start:stop:step_size],total=len(u.trajectory[local_start:stop:step_size])):
                W.write(u.atoms)
                n_written += 1

    return out