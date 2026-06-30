import numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt
import pandas as pd
from MDAnalysis.coordinates.DCD import DCDWriter
from pathlib import Path
import os
import gc,glob,re
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore",category=DeprecationWarning)
plt.style.use("ggplot")


def normalize(row):
    if row >= 1:
        return 1
    else:
        return row
    

def print_acceptance_ratio_T(log_path):
    log2 = pd.read_csv(log_path)
    log2["r"] = log2["r"].apply(lambda x: normalize(x))
    n_reps = int(log2[["repid_i", "repid_j"]].max().max()) + 1
    
    acc_rates = {}
    for i in range(n_reps - 1):
        pair = f"{i}-{i+1}"
        Ti = log2[log2["pair"] == pair]["Ti"].unique()[0]
        Tj = log2[log2["pair"] == pair]["Tj"].unique()[0]
        print(f"Pair {Ti}K - {Tj}K (Delta_T = {(Tj - Ti):.2f})")
        mean = np.mean(log2[log2["pair"] == pair]["r"])
        acc = log2[log2["pair"] == pair]["accepted"].sum()/len(log2[log2["pair"] == pair]["accepted"])
        print(f"Mean r: {mean:.4f}")
        print(f"Acceptance ratio: {acc:.4f}")
        acc_rates[pair] = acc
    
    return acc_rates

def print_acceptance_ratio_H(log_path):
    
    log = pd.read_csv(log_path)
    acc_rates = {}
    for pair in np.unique(log["pair"]):
        pairlog = log[log["pair"] == pair]
        rti = np.unique(pairlog["rti"])
        rtj = np.unique(pairlog["rtj"])
        print(f"Pair: {pair} (rt: {rti} - {rtj})")
        mean_r = np.mean([min(1,i) for i in pairlog["r"]])
        acc_rate = np.sum(pairlog["accepted"])/len(pairlog["accepted"])
        print(f"Mean r: {mean_r:.2f}")
        print(f"acc_rate: {acc_rate:.2f}")
        print("")
        acc_rates[pair] = acc_rate
        
    return acc_rates
    
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fix_restarted_segments(seg):
    """Make segment numbers monotonic across appended/restarted logs."""
    seg = np.asarray(seg, dtype=int)
    fixed = np.empty_like(seg)

    offset = 0
    fixed[0] = seg[0]

    for i in range(1, len(seg)):
        if seg[i] < seg[i - 1]:
            # More general than offset += seg[i-1] + 1
            offset = fixed[i - 1] + 1 - seg[i]

        fixed[i] = seg[i] + offset

    return fixed


def _as_bool(x):
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def _get_ij_cols(df, ex_col):
    """
    Supports both Ti/Tj, rti/rtj and state_i/state_j, mode_i/mode_j style names.
    """
    candidates = [
        (f"{ex_col}i", f"{ex_col}j"),       # T -> Ti/Tj, rt -> rti/rtj
        (f"{ex_col}_i", f"{ex_col}_j"),     # state -> state_i/state_j
    ]

    for ci, cj in candidates:
        if ci in df.columns and cj in df.columns:
            return ci, cj

    raise ValueError(
        f"Could not find i/j columns for ex_col={ex_col!r}. "
        f"Tried: {candidates}"
    )


def plot_replica_histograms(
    log_path,
    ex_col="T",
    repids_are="pre_exchange",
    assume_initial_identity=True,
    validate=True,
):
    """
    Plot occupancy histograms for each replica.

    Parameters
    ----------
    log_path : str or Path
        REMD log CSV.

    ex_col : str
        Quantity to histogram. Examples:
        - "T"     -> uses Ti/Tj
        - "rt"    -> uses rti/rtj
        - "state" -> uses state_i/state_j

    repids_are : {"pre_exchange", "post_exchange"}
        Use "pre_exchange" if repid_i/j describe the replicas before the accepted
        swap is applied. This is usually the safest interpretation for exchange logs.

        Use "post_exchange" only if repid_i/j were written after the accepted
        swap was already applied.

    assume_initial_identity : bool
        If the first segment does not contain all replicas, assume missing replica r
        initially occupied state r. This is useful for odd/even neighbor schedules
        where endpoint states can be unpaired.

    validate : bool
        In pre_exchange mode, check whether the logged replica-state mapping matches
        the reconstructed mapping before applying exchanges.
    """

    log2 = pd.read_csv(log_path)

    required = {"segment", "repid_i", "repid_j"}
    missing = required - set(log2.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ex_coli, ex_colj = _get_ij_cols(log2, ex_col)

    log2["segment_fixed"] = _fix_restarted_segments(log2["segment"].to_numpy())
    seg_col = "segment_fixed"

    segs = np.sort(log2[seg_col].unique())
    n_seg = len(segs)

    n_reps = int(log2[["repid_i", "repid_j"]].max().max()) + 1

    if repids_are == "post_exchange":
        # Similar to your original method, but using NaN and categorical counts.
        occ = np.full((n_seg, n_reps), np.nan, dtype=object)

        for k, s in enumerate(segs):
            if k > 0:
                occ[k] = occ[k - 1]

            rows = log2[log2[seg_col] == s]

            for _, row in rows.iterrows():
                ri = int(row["repid_i"])
                rj = int(row["repid_j"])

                occ[k, ri] = row[ex_coli]
                occ[k, rj] = row[ex_colj]

    elif repids_are == "pre_exchange":
        required = {"state_i", "state_j", "accepted"}
        missing = required - set(log2.columns)
        if missing:
            raise ValueError(
                f"pre_exchange reconstruction needs these columns: {missing}"
            )

        # Map state index -> exchange coordinate, e.g. state 0 -> 270.15 K.
        state_value = {}

        for _, row in log2.iterrows():
            state_value[int(row["state_i"])] = row[ex_coli]
            state_value[int(row["state_j"])] = row[ex_colj]

        n_states = int(log2[["state_i", "state_j"]].max().max()) + 1

        # Initial replica -> state mapping from the first segment.
        rep_state = np.full(n_reps, -1, dtype=int)
        rows0 = log2[log2[seg_col] == segs[0]]

        for _, row in rows0.iterrows():
            rep_state[int(row["repid_i"])] = int(row["state_i"])
            rep_state[int(row["repid_j"])] = int(row["state_j"])

        missing_reps = np.where(rep_state < 0)[0]

        if len(missing_reps) > 0:
            if not assume_initial_identity:
                raise ValueError(
                    "Initial segment does not contain all replicas. "
                    f"Missing replicas: {missing_reps}"
                )

            used_states = set(rep_state[rep_state >= 0])

            for r in missing_reps:
                if r < n_states and r not in used_states:
                    rep_state[r] = r
                    used_states.add(r)
                else:
                    raise ValueError(
                        "Could not infer initial state for missing replica "
                        f"{r}. Provide a complete first segment or disable "
                        "assume_initial_identity."
                    )

        occ = np.full((n_seg, n_reps), np.nan, dtype=object)

        for k, s in enumerate(segs):
            rows = log2[log2[seg_col] == s]

            # Occupancy during this segment, before applying this segment's swaps.
            for r in range(n_reps):
                occ[k, r] = state_value[rep_state[r]]

            if validate:
                for _, row in rows.iterrows():
                    ri = int(row["repid_i"])
                    rj = int(row["repid_j"])
                    si = int(row["state_i"])
                    sj = int(row["state_j"])

                    if rep_state[ri] != si or rep_state[rj] != sj:
                        raise ValueError(
                            "Logged mapping does not match reconstructed mapping. "
                            "This usually means repid_i/j are logged after exchange, "
                            "not before. Try repids_are='post_exchange'.\n"
                            f"At segment {s}: "
                            f"replica {ri} reconstructed state {rep_state[ri]}, "
                            f"log state {si}; "
                            f"replica {rj} reconstructed state {rep_state[rj]}, "
                            f"log state {sj}."
                        )

            # Apply accepted swaps to get the state mapping for the next segment.
            for _, row in rows.iterrows():
                if _as_bool(row["accepted"]):
                    ri = int(row["repid_i"])
                    rj = int(row["repid_j"])

                    rep_state[ri], rep_state[rj] = rep_state[rj], rep_state[ri]

    else:
        raise ValueError("repids_are must be 'pre_exchange' or 'post_exchange'.")

    # Use categorical counting instead of np.histogram.
    # This avoids bin-edge issues for non-integer T or rt values.
    all_values = pd.Series(occ.reshape(-1)).dropna()

    try:
        ladder = np.array(sorted(all_values.unique(), key=float), dtype=object)
    except Exception:
        ladder = np.array(sorted(all_values.unique(), key=str), dtype=object)

    hists = {}

    for r in range(n_reps):
        fig, ax = plt.subplots(figsize=(10, 10))

        values = pd.Series(occ[:, r]).dropna()
        counts = values.value_counts().reindex(ladder, fill_value=0)

        hists[str(r)] = counts.to_numpy()

        x = [f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
             for v in ladder]

        ax.bar(x, counts.to_numpy(), width=0.8, align="center")

        ax.set_title(
            f"Number of REMD segments spent at each {ex_col} for replica {r}"
        )

        if ex_col == "T":
            ax.set_xlabel("Temperature [K]")
        elif ex_col == "rt":
            ax.set_xlabel(r"$r_t$")
        else:
            ax.set_xlabel(ex_col)

        ax.set_ylabel("Counts")
        ax.tick_params(axis="x", rotation=90)

        fig.tight_layout()

    occ_df = pd.DataFrame(
        occ,
        index=segs,
        columns=[f"replica_{r}" for r in range(n_reps)],
    )
    occ_df.index.name = seg_col

    return hists, log2, occ_df



#def plot_replica_histograms(log_path,ex_col = "T"):

    log2 = pd.read_csv(log_path)

    seg = log2["segment"].to_numpy()
    fixed = np.empty_like(seg)

    offset = 0
    fixed[0] = seg[0]

    for i in range(1, len(seg)):
        # detect restart: current segment smaller than previous
        if seg[i] < seg[i - 1]:
            offset += seg[i - 1] + 1   # add length of previous run

        fixed[i] = seg[i] + offset

    log2["segment_fixed"] = fixed

    seg_col = "segment_fixed"   # or "segment" if you use that
    segs = np.sort(log2[seg_col].unique())
    n_seg = len(segs)

    # number of replicas = max repid_i/j + 1
    n_reps = int(log2[["repid_i", "repid_j"]].max().max()) + 1

    # temps_by_seg[k, r] = temperature of replica r at segment index k (corresponding to segs[k])
    temps_by_seg = np.zeros((n_seg, n_reps))

    # map segment value -> index in temps_by_seg
    seg_to_idx = {s: k for k, s in enumerate(segs)}

    ex_coli = ex_col + "i"
    ex_colj = ex_col + "j"
    # --- initialise first segment (covers all replicas; even segment with all pairs) ---
    s0 = segs[0]
    mask0 = log2[log2[seg_col] == s0]
    for _, row in mask0.iterrows():
        ri = int(row["repid_i"])
        rj = int(row["repid_j"])
        temps_by_seg[0, ri] = row[ex_coli]
        temps_by_seg[0, rj] = row[ex_colj]

    # --- propagate over all segments ---
    for k in range(1, n_seg):
        s = segs[k]

        # start from previous segment
        temps_by_seg[k] = temps_by_seg[k - 1]

        # apply exchanges that happened in this segment
        mask = log2[log2[seg_col] == s]
        for _, row in mask.iterrows():
            ri = int(row["repid_i"])
            rj = int(row["repid_j"])
            temps_by_seg[k, ri] = row[ex_coli]
            temps_by_seg[k, rj] = row[ex_colj]

    ladder = np.sort(np.unique(temps_by_seg))
    bins = np.append(ladder, ladder[-1] + 1)
    
    hists = {str(i):None for i in range(n_reps)}
    
    for i in range(n_reps):
        fig, ax = plt.subplots(figsize=(10,10))
        
        temp_series = temps_by_seg[:, i]  # T for replica i at all segments
        counts, _ = np.histogram(temp_series, bins=bins)
        hists[str(i)] = counts
        
        x = [str(i) for i in ladder]                   # equally spaced indices
        ax.bar(x, counts, width=0.8, align='center')  # uniform bar width
        # --------------------

        ax.set_title(f"Number of REMD segments spent at each {ex_col} for replica {i}")
        if ex_col == "T":
            ax.set_xlabel("Temperature [K]")
        elif ex_col == "rt":
            ax.set_xlabel(r"$r_t$")
        ax.set_ylabel("Counts")
    
    return hists,log2

def plot_epot_per_temp(log_path):

    log2 = pd.read_csv(log_path)
    temps = sorted(set(np.concatenate((log2["Ti"], log2["Tj"]))))

    for temp in temps:
        
        fig, ax = plt.subplots()
        
        df_1 = log2[(log2["Ti"] == temp)][["Ui_xi","segment"]].rename({"Ui_xi" : "U"},axis=1)
        df_2 = log2[(log2["Tj"] == temp)][["Uj_xj","segment"]].rename({"Uj_xj" : "U"},axis=1)
        joined = pd.merge(df_1,df_2,on=["U","segment"],how = "outer")
        
        ax.scatter(joined["segment"],joined["U"],s = 1.5)
        ax.set_ylabel(r"Potential energy in $\frac{kJ}{mol}$")
        ax.set_xlabel("REMD step")
        ax.set_title(f"Potential energy across simulation at T = {temp} K")



def stitch_traj(stitched_path , log="remd_log.csv", folder_pre = "NUP98_WT",ex_col="T",overwrite=False,start=None,stop=None,step = 1,dt=None):

    cwd = os.getcwd()
    save_path = f"{cwd}/{stitched_path}"

    Path(save_path).mkdir(parents=True, exist_ok=True)


    log2 = pd.read_csv(f"{cwd}/{log}")

    seg = log2["segment"].to_numpy()
    fixed = np.empty_like(seg)

    offset = 0
    fixed[0] = seg[0]

    for i in range(1, len(seg)):
        # detect restart: current segment smaller than previous
        if seg[i] < seg[i - 1]:
            offset += seg[i - 1] + 1   # add length of previous run

        fixed[i] = seg[i] + offset

    log2["segment_fixed"] = fixed

    seg_col = "segment_fixed"   # or "segment" if you use that
    segs = np.sort(log2[seg_col].unique())
    n_seg = len(segs)

    # number of replicas = max repid_i/j + 1
    n_reps = int(log2[["repid_i", "repid_j"]].max().max()) + 1

    # temps_by_seg[k, r] = temperature of replica r at segment index k (corresponding to segs[k])
    temps_by_seg = np.zeros((n_seg, n_reps))

    # map segment value -> index in temps_by_seg
    seg_to_idx = {s: k for k, s in enumerate(segs)}

    ex_coli = ex_col + "i"
    ex_colj = ex_col + "j"
    # --- initialise first segment (covers all replicas; even segment with all pairs) ---
    s0 = segs[0]
    mask0 = log2[log2[seg_col] == s0]
    for _, row in mask0.iterrows():
        ri = int(row["repid_i"])
        rj = int(row["repid_j"])
        temps_by_seg[0, ri] = row[ex_coli]
        temps_by_seg[0, rj] = row[ex_colj]

    # --- propagate over all segments ---
    for k in range(1, n_seg):
        s = segs[k]

        # start from previous segment
        temps_by_seg[k] = temps_by_seg[k - 1]

        # apply exchanges that happened in this segment
        mask = log2[log2[seg_col] == s]
        for _, row in mask.iterrows():
            ri = int(row["repid_i"])
            rj = int(row["repid_j"])
            temps_by_seg[k, ri] = row[ex_coli]
            temps_by_seg[k, rj] = row[ex_colj]


    temps = np.sort(np.unique(temps_by_seg))
    ntemps = len(temps)
    us = {}
    print(f"Storing Universes for each {ex_col}")
    
    pattern = os.path.join(".", f"{folder_pre}*")
    folders = [p for p in sorted(glob.glob(pattern)) if os.path.isdir(p)]
    for T in temps:
        for path in folders:
            m_rt = re.search(r"_rt(\d+(?:\.\d+)?)sig", path)
            if m_rt and m_rt.group(1) == f"{T:.3f}":
                print(path)
                us[str(T)] = mda.Universe(f"{path}/top.pdb",f"{path}/{path}.dcd")
            m = re.search(r"_(\d+(?:\.\d+)?)K$", path)
            if m and m.group(1) == f"{T:.2f}":
                us[str(T)] = mda.Universe(f"{path}/equilibration_final.pdb",f"{path}/{path}.dcd")
        if str(T) not in us.keys():
            raise FileNotFoundError(f"Could not extract rt nor T value {T} from paths {folders} !")
    
    for replica_id in range(ntemps):
        print(f"Stitching trajectory for replica {replica_id}")
        Path(f"{save_path}/replica_{replica_id}").mkdir(parents=True, exist_ok=True)
        dcd_out = f"{save_path}/replica_{replica_id}/replica_{replica_id}.dcd"
        pdb_out = f"{save_path}/replica_{replica_id}/replica_{replica_id}.pdb"
        if not overwrite:
            if os.path.exists(dcd_out) and os.path.exists(pdb_out):
                print(f"Found existing trajectory and topology for replica {replica_id}. Skipping...")
                continue

        frame_indices = np.arange(len(temps_by_seg[:, replica_id]))[start:stop:step]
        temps_replica = temps_by_seg[:, replica_id][start:stop:step]

        coords = []
        times = []
        dims_list = []

        for frame_idx, T in tqdm(zip(frame_indices, temps_replica), total=len(frame_indices)):
            u = us[str(T)]
            ts = u.trajectory[frame_idx]

            coords.append(ts.positions.copy())
            times.append(ts.time)
            dims_list.append(ts.dimensions.copy())
        coords = np.asarray(coords)
        times = np.asarray(times)
        dims_list = np.asarray(dims_list)
        if dt is None:
            dt = np.median(np.diff(times))

        u_ref = us[str(temps_replica[0])]
        u_new = mda.Merge(u_ref.atoms).load_new(
            coords,
            order="fac",
            dimensions=dims_list,
            dt=dt,
        )

        u_new.atoms.write(pdb_out)

        with DCDWriter(dcd_out, u_new.atoms.n_atoms, dt=dt) as W:
            for ts, dims in zip(u_new.trajectory, dims_list):
                u_new.dimensions = dims
                W.write(u_new.atoms)

        del u_new
        gc.collect()


import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import MDAnalysis as mda
from openmm import app, unit, XmlSerializer, Platform
import openmm as mm
from tqdm import tqdm
import re

KB = 0.008314462618  # kJ / (mol*K)


def make_energy_sim(
    system_xml: Path,
    top_pdb: Path,
    T,
    platform="CPU",
    gamma=0.01,
    dt_ps=0.01,
    device_index=None,
    precision="mixed",
    cpu_threads="1",
):
    system = XmlSerializer.deserialize(system_xml.read_text())
    pdb = app.PDBFile(str(top_pdb))
    integ = mm.LangevinMiddleIntegrator(
        T * unit.kelvin,
        gamma / unit.picosecond,
        dt_ps * unit.picoseconds,
    )

    omm_platform = Platform.getPlatformByName(platform)
    properties = {}

    if platform == "CUDA":
        properties["Precision"] = precision
        if device_index is not None:
            properties["DeviceIndex"] = str(device_index)

    elif platform == "OpenCL":
        properties["Precision"] = precision
        if device_index is not None:
            properties["DeviceIndex"] = str(device_index)

    elif platform == "CPU":
        if cpu_threads is not None:
            properties["Threads"] = str(cpu_threads)

    if properties:
        sim = app.Simulation(pdb.topology, system, integ, omm_platform, properties)
    else:
        sim = app.Simulation(pdb.topology, system, integ, omm_platform)

    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        a, b, c = box
        sim.context.setPeriodicBoxVectors(a, b, c)

    return sim


def _set_frame_to_context(sim, universe):
    """
    Set current MDAnalysis frame positions and box into an OpenMM context.

    MDAnalysis coordinates/box are in Angstrom.
    OpenMM expects nm.
    """
    pos_nm = universe.atoms.positions / 10.0
    sim.context.setPositions(pos_nm)

    if universe.dimensions is not None:
        lx, ly, lz, alpha, beta, gamma = universe.dimensions
        sim.context.setPeriodicBoxVectors(
            mm.Vec3(lx / 10.0, 0.0, 0.0),
            mm.Vec3(0.0, ly / 10.0, 0.0),
            mm.Vec3(0.0, 0.0, lz / 10.0),
        )


def compute_log_r_for_current_frames(sim_i, sim_j, u_i, u_j, T_i, T_j):
    """
    Compute generalized Metropolis log-r for the current frames
    of two MDAnalysis universes.
    """
    beta_i = 1.0 / (KB * T_i)
    beta_j = 1.0 / (KB * T_j)

    # Ui(xi), Uj(xj)
    _set_frame_to_context(sim_i, u_i)
    _set_frame_to_context(sim_j, u_j)

    st_i = sim_i.context.getState(getPositions=True, getEnergy=True)
    st_j = sim_j.context.getState(getPositions=True, getEnergy=True)

    xi = st_i.getPositions()
    xj = st_j.getPositions()

    Ui_xi = st_i.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    Uj_xj = st_j.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Ui(xj), Uj(xi)
    sim_i.context.setPositions(xj)
    sim_j.context.setPositions(xi)

    # Keep each frame's own box with its own Hamiltonian
    if u_j.dimensions is not None:
        lx, ly, lz, alpha, beta, gamma = u_j.dimensions
        sim_i.context.setPeriodicBoxVectors(
            mm.Vec3(lx / 10.0, 0.0, 0.0),
            mm.Vec3(0.0, ly / 10.0, 0.0),
            mm.Vec3(0.0, 0.0, lz / 10.0),
        )
    if u_i.dimensions is not None:
        lx, ly, lz, alpha, beta, gamma = u_i.dimensions
        sim_j.context.setPeriodicBoxVectors(
            mm.Vec3(lx / 10.0, 0.0, 0.0),
            mm.Vec3(0.0, ly / 10.0, 0.0),
            mm.Vec3(0.0, 0.0, lz / 10.0),
        )

    Ui_xj = sim_i.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    Uj_xi = sim_j.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    delta = beta_i * (Ui_xj - Ui_xi) + beta_j * (Uj_xi - Uj_xj)
    log_r = -float(delta)

    return log_r, {
        "Ui_xi": Ui_xi,
        "Uj_xj": Uj_xj,
        "Ui_xj": Ui_xj,
        "Uj_xi": Uj_xi,
        "delta": delta,
    }


def accept_from_log_r(log_r, rng):
    u = float(rng.random())
    log_u = float(np.log(u))
    accepted = log_u < log_r
    return accepted, u, log_u



def build_foldername(prefix, kappa, suffix):
    return f"{prefix}{kappa}{suffix}"


def get_kappas(prefix, path="."):
    kappas = []

    patterns = [
        re.compile(rf"^{re.escape(prefix)}_kap([0-9.]+)_"),
        re.compile(rf"^{re.escape(prefix)}_rt([0-9.]+)nm_"),
    ]

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)

        if not os.path.isdir(full_path):
            continue

        for pattern in patterns:
            match = pattern.match(entry)
            if match:
                kappas.append(float(match.group(1)))
                break  # avoid double matching

    return sorted(kappas)

def get_temperatures(prefix, path="."):
    temps = []

    pattern = re.compile(rf"^{re.escape(prefix)}_([0-9.]+)K$")
    print(pattern)

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        print(full_path)
        if not os.path.isdir(full_path):
            continue

        match = pattern.match(entry)
        print(match)
        if match:
            T = float(match.group(1))
            temps.append(T)

    return sorted(temps)

def get_even_pairs(kappas):
    """
    Even adjacent pairs:
    (0,1), (2,3), (4,5), ...
    """
    return [(kappas[i], kappas[i + 1]) for i in range(0, len(kappas) - 1, 2)]

def get_odd_pairs(kappas):
    """
    Odd adjacent pairs:
    (1,2), (3,4), (5,6), ...
    """
    return [(kappas[i], kappas[i + 1]) for i in range(1, len(kappas) - 1, 2)]

def set_cpu_env_for_worker(cpu_threads_per_worker):
    """
    Limit thread-hungry libraries inside each worker.
    Useful for both CPU and GPU runs.
    """
    if cpu_threads_per_worker is None:
        return

    val = str(cpu_threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = val
    os.environ["OPENBLAS_NUM_THREADS"] = val
    os.environ["MKL_NUM_THREADS"] = val
    os.environ["NUMEXPR_NUM_THREADS"] = val


def scan_pair(
    kappa1,
    kappa2,
    *,
    base_dir=".",
    prefix="hpl2+lin13_kap",
    suffix="_260.15K",
    T1=260.15,
    T2=260.15,
    platform="CPU",
    device_index=None,
    precision="mixed",
    gamma1=0.01,
    gamma2=0.01,
    dt_ps=0.01,
    step=None,
    seed=12345,
    show_progress=False,
    progress_position=0,
    cpu_threads_per_worker=1,
    openmm_cpu_threads=1,
):
    """
    Process one kappa pair completely and return rows for the summary CSV.
    """
    set_cpu_env_for_worker(cpu_threads_per_worker)

    base_dir = Path(base_dir)

    folder1 = build_foldername(prefix, kappa1, suffix)
    folder2 = build_foldername(prefix, kappa2, suffix)

    dir1 = base_dir / folder1
    dir2 = base_dir / folder2

    pdb1 = dir1 / f"starting_config.pdb"
    #pdb1 = dir1 / f"{folder1}.pdb"
    dcd1 = dir1 / f"{folder1}.dcd"
    xml1 = dir1 / f"{folder1}.xml"

    pdb2 = dir2 / f"starting_config.pdb"
    #pdb2 = dir2 / f"{folder2}.pdb"
    dcd2 = dir2 / f"{folder2}.dcd"
    xml2 = dir2 / f"{folder2}.xml"

    for f in [pdb1, dcd1, xml1, pdb2, dcd2, xml2]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    u1 = mda.Universe(str(pdb1), str(dcd1))
    u2 = mda.Universe(str(pdb2), str(dcd2))

    traj1 = u1.trajectory[::step]
    traj2 = u2.trajectory[::step]
    n_frames = min(len(traj1), len(traj2))

    sim1 = make_energy_sim(
        system_xml=xml1,
        top_pdb=pdb1,
        T=T1,
        platform=platform,
        device_index=device_index,
        precision=precision,
        gamma=gamma1,
        dt_ps=dt_ps,
        cpu_threads=openmm_cpu_threads,
    )
    sim2 = make_energy_sim(
        system_xml=xml2,
        top_pdb=pdb2,
        T=T2,
        platform=platform,
        device_index=device_index,
        precision=precision,
        gamma=gamma2,
        dt_ps=dt_ps,
        cpu_threads=openmm_cpu_threads,
    )

    rng = np.random.default_rng(seed + int(round(kappa1 * 1000)) + 100000 * int(round(kappa2 * 1000)))
    rows = []

    iterator = enumerate(zip(traj1, traj2))
    if show_progress:
        iterator = enumerate(
            tqdm(
                zip(traj1, traj2),
                total=n_frames,
                desc=f"{kappa1} vs {kappa2}",
                position=progress_position,
                leave=True,
            )
        )

    for frame_idx, (_ts1, _ts2) in iterator:
        log_r, energies = compute_log_r_for_current_frames(
            sim_i=sim1,
            sim_j=sim2,
            u_i=u1,
            u_j=u2,
            T_i=T1,
            T_j=T2,
        )

        accepted, u_rand, log_u = accept_from_log_r(log_r, rng)

        rows.append({
            "pair_type": "even",
            "kappa1": kappa1,
            "kappa2": kappa2,
            "folder1": folder1,
            "folder2": folder2,
            "frame": frame_idx,
            "time1_ps": float(u1.trajectory.time),
            "time2_ps": float(u2.trajectory.time),
            "T1_K": T1,
            "T2_K": T2,
            "log_r": log_r,
            "r": float(np.exp(log_r)),
            "accepted": accepted,
            "u": u_rand,
            "log_u": log_u,
            "Ui_xi": energies["Ui_xi"],
            "Uj_xj": energies["Uj_xj"],
            "Ui_xj": energies["Ui_xj"],
            "Uj_xi": energies["Uj_xi"],
            "delta": energies["delta"],
            "platform": platform,
            "device_index": device_index if device_index is not None else "",
        })

    return rows


def run_all_pairs(
    *,
    base_dir=".",
    out_csv="exchange_summary_even_pairs.csv",
    prefix="hpl2+lin13_kap",
    suffix="_260.15K",
    T=260.15,
    platform="CPU",
    gpu_ids=None,
    precision="mixed",
    gamma=0.01,
    dt_ps=0.01,
    step=None,
    max_workers=None,
    cpu_threads_per_worker=1,
    openmm_cpu_threads=1,
    pair_sel = "even",
    kappas = True
):
    """
    Run all even adjacent kappa-pair comparisons in parallel and write one summary CSV.
    Works for both CPU and GPU execution.
    """
    if kappas:
        kappas = get_kappas(prefix.split("_")[0],path=".")
    else:
        #kappas = get_temperatures("NUP98_WT",path = ".")
        kappas = [270.15,271.73]
    if pair_sel == "odd":
        pairs = get_odd_pairs(kappas)
    else:
        pairs = get_even_pairs(kappas)

    if platform in {"CUDA", "OpenCL"}:
        if gpu_ids is None or len(gpu_ids) == 0:
            gpu_ids = [0]
        max_available_workers = min(len(pairs), len(gpu_ids))
    else:
        gpu_ids = [None]
        max_available_workers = len(pairs)

    if max_workers is None:
        n_workers = max_available_workers
    else:
        n_workers = min(max_workers, max_available_workers)

    fieldnames = [
        "pair_type",
        "kappa1",
        "kappa2",
        "folder1",
        "folder2",
        "frame",
        "time1_ps",
        "time2_ps",
        "T1_K",
        "T2_K",
        "log_r",
        "r",
        "accepted",
        "u",
        "log_u",
        "Ui_xi",
        "Uj_xj",
        "Ui_xj",
        "Uj_xi",
        "delta",
        "platform",
        "device_index",
    ]

    all_rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}

        for pair_idx, (k1, k2) in enumerate(pairs):
            if platform in {"CUDA", "OpenCL"}:
                device_index = gpu_ids[pair_idx % len(gpu_ids)]
            else:
                device_index = None

            # Inner progress bar only for the first submitted worker
            show_progress = (pair_idx == 0)

            fut = pool.submit(
                scan_pair,
                k1,
                k2,
                base_dir=base_dir,
                prefix=prefix,
                suffix=suffix,
                T1=T,
                T2=T,
                platform=platform,
                device_index=device_index,
                precision=precision,
                gamma1=gamma,
                gamma2=gamma,
                dt_ps=dt_ps,
                step=step,
                show_progress=show_progress,
                progress_position=1,
                cpu_threads_per_worker=cpu_threads_per_worker,
                openmm_cpu_threads=openmm_cpu_threads,
            )
            futures[fut] = (k1, k2, device_index)

        for fut in as_completed(futures):
            k1, k2, device_index = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)

            if device_index is None:
                print(f"Finished pair kappa={k1} vs {k2} on CPU -> {len(rows)} frames")
            else:
                print(f"Finished pair kappa={k1} vs {k2} on device {device_index} -> {len(rows)} frames")

    all_rows.sort(key=lambda r: (r["kappa1"], r["kappa2"], r["frame"]))

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} total comparisons to {out_csv}")

# def plot_e2e_corr_function(path,sysname,comp_dict = None,top_file = "top_reindexed.pdb", traj_file = "traj_reindexed.dcd", out_path = ".", save_fig = False):
#     u = mda.Universe(f"{path}/{sysname}/{top_file}",f"{path}/{sysname}/{traj_file}")

#     nframes = len(u.trajectory)
#     nchains = len(u.segments)
#     R = np.zeros((nframes,nchains,3))
#     for tidx,ts in enumerate(u.trajectory):
#         for cidx,chain in enumerate(u.segments):
#             R[tidx,cidx,] = chain.atoms[-1].position - chain.atoms[0].position 
            
#     rho = np.zeros((nframes,nchains))
#     R0Rt = np.zeros((nframes,nchains))
#     R2 = np.zeros((nframes,nchains))
#     for t in range(nframes):
#         for c in range(nchains):
#             R0Rt[t,c] = np.dot(R[0,c],R[t,c])
#             R2[t,c] = np.dot(R[t,c],R[t,c])
#             rho[t,c] = np.mean(R0Rt[:t+1,c])/np.mean(R2[:t+1,c])

#     if comp_dict:
#         rho_comps = {prot:rho[:,comp_dict[prot][0]:comp_dict[prot][1] + 1] for prot in comp_dict.keys()}
#     else:
#         rho_comps = {"all":rho}    
#     for i in rho_comps.keys():
#         np.savetxt(f"{out_path}/rho_{i}.txt",rho_comps[i])


#     fig,ax = plt.subplots() 
#     for key,val in rho_comps.items():
#         ax.plot([np.mean(val[i,]) for i in range(val.shape[0])], label=key)
#     ax.legend(loc="best")
#     ax.set_xlabel("Steps")
#     ax.set_ylabel("End-to-End correlation function")
#     ax.set_title(f"End-to-End correlation function for {sysname}")
#     if save_fig:
#         fig.savefig(f"{out_path}/E2E_corr_function_{sysname}.pdf")
    
    
