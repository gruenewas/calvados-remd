import csv, time, json, subprocess,os
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# ---- OpenMM bits ----
from openmm import app, unit, XmlSerializer, Platform
import openmm as mm

KB = 0.008314462618  # kJ/(mol*K)

# ---------- Data model ----------
@dataclass
class TFolder:
    """A fixed-temperature CALVADOS run directory."""
    path: Path          # e.g., Path("T_260.15")
    T: float            # Kelvin
    system_xml: Path    # e.g., path/"<sysname>.xml"
    top_pdb: Path       # path/"checkpoint.pdb" (latest positions/box)
    chk: Path           # path/"restart.chk" (latest checkpoint)
    gamma: float        #Friction coefficient as set in config

# ---------- Energy eval (side-effect free) ----------
def make_energy_sim(system_xml: Path, top_pdb: Path,chk: Path, T, platform="CPU",gamma=0.01,dt_ps = 0.01):
    start_build = time.time()
    system = XmlSerializer.deserialize(system_xml.read_text())
    pdb    = app.PDBFile(str(top_pdb))
    integ  = mm.LangevinMiddleIntegrator(T*unit.kelvin, gamma/unit.picosecond, dt_ps*unit.picoseconds)
    sim    = app.Simulation(pdb.topology, system, integ, Platform.getPlatformByName(platform))
    start_load = time.time()
    sim.loadCheckpoint(str(chk))
    t_load = time.time() - start_load
    print(f"Loading Checkpoint for replica at {T} took {t_load:.2f} s")
    # set PBC from PDB if present
    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        a,b,c = box
        sim.context.setPeriodicBoxVectors(a,b,c)
    t_build = time.time() - start_build
    print(f"Building system for replica at {T} took {t_build:.2f} s")
    return sim

def compute_log_r_general(fold_i: TFolder, fold_j: TFolder, platform="CPU"):
    """
    log r = -Δ for swap between two *temperature folders* (Hamiltonian/Temperature REMD).
    Uses folder i/j System XML + each folder's checkpoint.pdb for positions/box.
    Create Simulation objects for each Temperature from corresponding xml, checkpoint, temp and perform replica swap to calculate cross energies + get the swapped simulations.
    If Metrpolis gets accepted used swapped simulations to create new checkpoint. 

    """
    beta_i = 1.0/(KB*fold_i.T); beta_j = 1.0/(KB*fold_j.T)

    sim_i = make_energy_sim(fold_i.system_xml, fold_i.top_pdb, fold_i.chk, fold_i.T, platform,fold_i.gamma)
    sim_j = make_energy_sim(fold_j.system_xml, fold_j.top_pdb, fold_j.chk, fold_j.T, platform, fold_j.gamma)

    start_eval = time.time()
    st_i = sim_i.context.getState(getPositions=True,getEnergy=True)
    st_j = sim_j.context.getState(getPositions=True,getEnergy=True)

    xi = st_i.getPositions()
    xj = st_j.getPositions()

    Ui_xi = st_i.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    Uj_xj = st_j.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    sim_i.context.setPositions(xj)
    sim_j.context.setPositions(xi)

    Ui_xj = sim_i.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    Uj_xi = sim_j.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    delta = beta_i*(Ui_xj - Ui_xi) + beta_j*(Uj_xi - Uj_xj)
    log_r = -float(delta)
    t_eval = time.time() - start_eval
    print(f"Computing energies and perform coordinate swap took {t_eval:.2f} s for replicas at {fold_i.T} K and {fold_j.T} K.")
    return log_r, dict(Ui_xi=np.round(Ui_xi,2), Uj_xj=np.round(Uj_xj,2), Ui_xj=np.round(Ui_xj,2), Uj_xi=np.round(Uj_xi,2)),sim_i,sim_j

def accept_from_log_r(log_r):
    u = float(np.random.rand())
    log_u = float(np.log(u))
    acc = (log_u < log_r)
    return acc,u,log_u


# ---------- State swap (write new checkpoints under each folder's System) ----------

def read_state_from_chk(sim):
    st = sim.context.getState(getPositions = True, getVelocities=True, enforcePeriodicBox=True)
    return  st.getPositions(),st.getVelocities(), st.getPeriodicBoxVectors(), sim

def write_checkpoint(sim,positions, velocities, box,out_chk: Path,
                     out_pdb: Path=None):
    a,b,c = box
    sim.context.setPositions(positions)
    sim.context.setPeriodicBoxVectors(a,b,c)
    sim.context.setVelocities(velocities)
    sim.saveCheckpoint(str(out_chk))
    if out_pdb is not None:
        rep = app.PDBReporter(str(out_pdb), 0)
        rep.report(sim, sim.context.getState(getPositions=True, enforcePeriodicBox=True))

def swap_states_between_Tfolders(fold_i: TFolder, fold_j: TFolder,sim_i,sim_j, platform="CPU", write_pdb=False):
    """
    Accepted exchange between temperature folders:
      - i keeps its System (Ti), but takes j's positions/velocities, rescaled by sqrt(Ti/Tj), xj already stored in sim_i from energy calculation
      - j keeps its System (Tj), but takes i's positions/velocities, rescaled by sqrt(Tj/Ti), xi already stored in sim_j from energy calculation
    Overwrites restart.chk (and checkpoint.pdb if write_pdb) in each folder.
    """
    xj, vi, bi,sim_i = read_state_from_chk(sim_i)
    xi, vj, bj,sim_j = read_state_from_chk(sim_j)

    scale_j = np.sqrt(fold_i.T/fold_j.T)
    scale_i = np.sqrt(fold_j.T/fold_i.T)

    write_checkpoint(
        sim_i,xj, vj*scale_j, bj,
        fold_i.chk,
        (fold_i.path/"checkpoint.pdb") if write_pdb else None
    )
    write_checkpoint(
        sim_j,xi, vi*scale_i, bi,
        fold_j.chk,
        (fold_j.path/"checkpoint.pdb") if write_pdb else None
    )

# ---------- Pairing (even/odd neighbors over T-folders) ----------

def neighbor_pairs(n, attempt_idx):
    off = attempt_idx % 2
    return [(i, i+1) for i in range(off, n-1, 2)]

# --- ADD: parallel worker with cached Simulations (minimal) ---
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

_WORKER_CACHE = {}     # per-process cache: key = Path to T-folder; value = {"sim": Simulation, "beta": float, "top": PDBFile}



def _worker_init(platform="CPU", threads=1):
    # Called once per worker process
    os.environ.setdefault("OPENMM_CPU_THREADS", str(threads))
    _WORKER_CACHE.clear()
    _WORKER_CACHE["_platform"] = platform

def _get_cached_sim(folder: TFolder):
    ent = _WORKER_CACHE.get(folder.path)
    if ent is None:
        # Reuse your existing builder exactly once per worker+folder
        sim = make_energy_sim(folder.system_xml, folder.top_pdb, folder.chk,
                              folder.T, platform=_WORKER_CACHE["_platform"], gamma=folder.gamma)
        ent = {
            "sim": sim,
            "beta": 1.0/(KB*folder.T),
        }
        _WORKER_CACHE[folder.path] = ent
    return ent["sim"], ent["beta"]

def _eval_pair_task(args):
    """
    Worker task:
      - reuse cached Simulations per T-folder (no rebuild)
      - compute self/cross energies
      - Metropolis test
      - if accepted: write swapped checkpoints inside the worker (safe: pairs are disjoint)
      - return a compact dict for the parent to log + update rep_at_T
    """
    i, j, fold_i, fold_j = args
    sim_i, beta_i = _get_cached_sim(fold_i)
    sim_j, beta_j = _get_cached_sim(fold_j)

    # self states
    st_i = sim_i.context.getState(getPositions=True, getEnergy=True)
    st_j = sim_j.context.getState(getPositions=True, getEnergy=True)
    xi   = st_i.getPositions(); Ui_xi = st_i.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    xj   = st_j.getPositions(); Uj_xj = st_j.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # cross
    sim_i.context.setPositions(xj)
    sim_j.context.setPositions(xi)
    Ui_xj = sim_i.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    Uj_xi = sim_j.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    delta = beta_i*(Ui_xj - Ui_xi) + beta_j*(Uj_xi - Uj_xj)
    log_r = -float(delta)
    s = int(time.time_ns()*os.getpid()%(2**32-1))
    np.random.seed(s)
    u = np.random.rand()
    log_u = np.log(u)
    accepted = (log_u < log_r)
    #accepted,u,log_u = accept_from_log_r(log_r)

    if accepted:
        # swap & write fresh checkpoints for both folders (positions already swapped in the two contexts)
        swap_states_between_Tfolders(fold_i, fold_j, sim_i, sim_j, write_pdb=False)

    Es = dict(Ui_xi=np.round(Ui_xi,2), Uj_xj=np.round(Uj_xj,2),
              Ui_xj=np.round(Ui_xj,2), Uj_xi=np.round(Uj_xi,2))
    return {"i": i, "j": j, "r": np.exp(log_r), "u":u, "log_r": log_r,"log_u":log_u, "accepted": accepted, "Es": Es,
            "Ti": fold_i.T, "Tj": fold_j.T}
