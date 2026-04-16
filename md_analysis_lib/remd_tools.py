import numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt
import pandas as pd
from MDAnalysis.coordinates.DCD import DCDWriter
from pathlib import Path
import os
import gc
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
plt.style.use("ggplot")


def normalize(row):
    if row >= 1:
        return 1
    else:
        return row
    

def print_acceptance_ratio(log_path):
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
        


def plot_replica_histograms(log_path):

    log2 = pd.read_csv("remd_log.csv")

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

    # --- initialise first segment (covers all replicas; even segment with all pairs) ---
    s0 = segs[0]
    mask0 = log2[log2[seg_col] == s0]
    for _, row in mask0.iterrows():
        ri = int(row["repid_i"])
        rj = int(row["repid_j"])
        temps_by_seg[0, ri] = row["Ti"]
        temps_by_seg[0, rj] = row["Tj"]

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
            temps_by_seg[k, ri] = row["Ti"]
            temps_by_seg[k, rj] = row["Tj"]

    ladder = np.sort(np.unique(temps_by_seg))
    bins = np.append(ladder, ladder[-1] + 1)
    
    hists = {str(i):None for i in range(n_reps)}
    
    for i in range(n_reps):
        fig, ax = plt.subplots(figsize=(10,10))
        
        temp_series = temps_by_seg[:, i]  # T for replica i at all segments
        counts, _ = np.histogram(temp_series, bins=bins)
        hists[str(i)] = counts
        
        x = np.arange(len(counts))                    # equally spaced indices
        ax.bar(x, counts, width=0.8, align='center')  # uniform bar width
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.2f}" for b in bins[:-1]])  # label with temps
        # --------------------

        ax.set_title(f"Number of REMD steps spent at each T for replica {i}")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Steps")
    
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



def stitch_traj(stitched_path , log="remd_log.csv", folder_pre = "NUP98_WT"):

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

    # --- initialise first segment (covers all replicas; even segment with all pairs) ---
    s0 = segs[0]
    mask0 = log2[log2[seg_col] == s0]
    for _, row in mask0.iterrows():
        ri = int(row["repid_i"])
        rj = int(row["repid_j"])
        temps_by_seg[0, ri] = row["Ti"]
        temps_by_seg[0, rj] = row["Tj"]

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
            temps_by_seg[k, ri] = row["Ti"]
            temps_by_seg[k, rj] = row["Tj"]


    temps = np.sort(np.unique(temps_by_seg))
    ntemps = len(temps)
    us = {}
    print("Storing Universes for each Temperature")
    for T in temps:
        path = f"{folder_pre}_{T:.2f}K" 
        us[str(T)] = mda.Universe(f"{path}/equilibration_final.pdb",f"{path}/{path}.dcd")

    for replica_id in range(ntemps):
        print(f"Stitching trajectory for replica {replica_id}")
        Path(f"{save_path}/replica_{replica_id}").mkdir(parents=True, exist_ok=True)
        coords = []
        for frame,T in enumerate(temps_by_seg[:,replica_id]):
            u = us[str(T)]
            coords.append(u.trajectory[frame].positions.copy())
        coords = np.array(coords)
        dims = u.dimensions.copy()
        u_new = mda.Merge(u.atoms).load_new(coords, order="fac")
        u_new.atoms.dimensions = dims
        u_new.atoms.write(f"{save_path}/replica_{replica_id}/replica_{replica_id}.pdb")
        with DCDWriter(f"{save_path}/replica_{replica_id}/replica_{replica_id}.dcd", u_new.atoms.n_atoms) as W:
            for ts in u_new.trajectory:
                u_new.dimensions = dims
                W.write(u_new.atoms)
        del u_new, u
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
    
    
