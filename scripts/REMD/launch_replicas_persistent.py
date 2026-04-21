import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(line_buffering=True)

KB = 0.008314462618  # kJ/(mol*K)
CSV_FIELDS = [
    "time",
    "segment",
    "pair",
    "state_i",
    "state_j",
    "sysname_i",
    "sysname_j",
    "Ti",
    "Tj",
    "rti",
    "rtj",
    "mode_i",
    "mode_j",
    "r",
    "u",
    "log_r",
    "log_u",
    "accepted",
    "repid_i",
    "repid_j",
    "Ui_xi",
    "Uj_xj",
    "Ui_xj",
    "Uj_xi",
]


@lru_cache(maxsize=1)
def get_openmm_modules():
    import openmm
    from openmm import app, unit

    return openmm, app, unit


@lru_cache(maxsize=1)
def get_calvados_sim_module():
    from calvados import sim as calvados_sim

    return calvados_sim


def pos_unit():
    _, _, unit = get_openmm_modules()
    return unit.nanometer


def vel_unit():
    _, _, unit = get_openmm_modules()
    return unit.nanometer / unit.picosecond


def visible_cuda_devices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None
    devices = [device.strip() for device in raw.split(",") if device.strip()]
    return devices or None


@dataclass(frozen=True)
class ReplicaSpec:
    path: Path
    T: float
    rt: float | None
    sc_mode: str
    gamma: float
    sysname: str
    config: dict
    components: dict

    @property
    def state_label(self):
        rt_label = "none" if self.rt is None else f"{self.rt:.6g}"
        return f"T={self.T:.6g}|rt={rt_label}|mode={self.sc_mode}"


def neighbor_pairs(n_replicas, attempt_idx):
    offset = attempt_idx % 2
    return [(i, i + 1) for i in range(offset, n_replicas - 1, 2)]


def ensure_log(log_path: Path):
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
        return 0

    log = pd.read_csv(log_path)
    if log.empty:
        return 0
    return int(log["segment"].iloc[-1]) + 1


def reconstruct_replica_mapping(log_path: Path, n_replicas: int):
    if not log_path.exists():
        return list(range(n_replicas))

    log = pd.read_csv(log_path)
    if log.empty:
        return list(range(n_replicas))

    rep_at_state = list(range(n_replicas))
    ordered = log.sort_values(["segment", "pair"], kind="stable")
    for _, row in ordered.iterrows():
        if bool(row["accepted"]):
            i, j = map(int, str(row["pair"]).split("-"))
            rep_at_state[i], rep_at_state[j] = rep_at_state[j], rep_at_state[i]
    return rep_at_state


def discover_replicas(sysname, path):
    base = Path(path).resolve()
    replicas = []
    for folder in sorted(base.glob(f"{sysname}_*")):
        with open(folder / "config.yaml") as handle:
            config = yaml.safe_load(handle)
        with open(folder / "components.yaml") as handle:
            components = yaml.safe_load(handle)

        rt = config.get("rt")
        if rt is not None:
            rt = float(rt)

        replicas.append(
            ReplicaSpec(
                path=folder,
                T=float(config["temp"]),
                rt=rt,
                sc_mode=str(config.get("sc_mode", "none")),
                gamma=float(config["friction_coeff"]),
                sysname=folder.name,
                config=config,
                components=components,
            )
        )

    return sorted(replicas, key=lambda rep: (rep.T, rep.rt if rep.rt is not None else float("-inf"), rep.sysname))


def get_total_steps(replicas):
    total_steps = {int(rep.config["total_steps"]) for rep in replicas}
    seg_steps = {int(rep.config["steps"]) for rep in replicas}
    if len(total_steps) != 1:
        raise ValueError("Total steps vary between replicas")
    if len(seg_steps) != 1:
        raise ValueError("Segment steps vary between replicas")
    return total_steps.pop(), seg_steps.pop()


def quantity_to_numpy(quantity, target_unit):
    return np.asarray(quantity.value_in_unit(target_unit), dtype=np.float64)


def numpy_to_quantity(array, target_unit):
    return np.asarray(array, dtype=np.float64) * target_unit


class PersistentReplica:
    """
    One persistent CALVADOS/OpenMM runner owned by one worker process.

    The same Simulation is reused for:
    - propagation of REMD segments
    - self energy evaluation
    - cross energy evaluation against incoming coordinates
    - accepted state swaps
    """

    def __init__(self, spec: ReplicaSpec, platform_override=None,assigned_gpu=None):
        self.spec = spec
        self.path = spec.path
        self.config = dict(spec.config)
        self.components = spec.components
        self.assigned_gpu = assigned_gpu
        if platform_override is not None:
            self.config["platform"] = platform_override

        calvados_sim = get_calvados_sim_module()
        self.mysim = calvados_sim.Sim(str(self.path), self.config, self.components)
        self._load_or_build_system()
        self.simulation, self.append = self._initialize_simulation()
        self._attach_reporters()

    @property
    def checkpoint_path(self):
        return self.path / "restart.chk"

    @property
    def checkpoint_pdb_path(self):
        return self.path / "checkpoint.pdb"

    @property
    def system_pdb_path(self):
        return self.path / f"{self.mysim.sysname}.pdb"

    @property
    def dcd_path(self):
        return self.path / f"{self.mysim.sysname}.dcd"

    @property
    def state_log_path(self):
        return self.path / f"{self.mysim.sysname}.log"

    @property
    def xml_path(self):
        return self.path / f"{self.mysim.sysname}.xml"

    def _load_or_build_system(self):
        openmm, _, _ = get_openmm_modules()
        if self.xml_path.is_file():
            print(f"[{self.spec.sysname}] Loading existing system XML")
            try:
                self.mysim.system = openmm.XmlSerializer.deserialize(self.xml_path.read_text())
                self.mysim.pdb_cg = str(self.path / "top.pdb")
                return
            except ValueError:
                print(f"[{self.spec.sysname}] Existing XML is broken, rebuilding system")
        else:
            print(f"[{self.spec.sysname}] Building system from CALVADOS config")

        self.mysim.build_system()

    def _make_openmm_simulation(self, pdb):
        openmm, app, unit = get_openmm_modules()
        integrator = openmm.openmm.LangevinMiddleIntegrator(
            self.mysim.temp * unit.kelvin,
            self.mysim.friction_coeff / unit.picosecond,
            0.01 * unit.picosecond,
        )
        if self.mysim.random_number_seed is not None:
            integrator.setRandomNumberSeed(self.mysim.random_number_seed)

        platform = openmm.Platform.getPlatformByName(self.mysim.platform)
        if self.mysim.platform == "CPU":
            simulation = app.simulation.Simulation(
                pdb.topology,
                self.mysim.system,
                integrator,
                platform,
                dict(Threads=str(self.mysim.threads)),
            )
        else:
            properties = {}
            visible_devices = visible_cuda_devices()
            effective_device_index = None
            if self.assigned_gpu is not None:
                if visible_devices and len(visible_devices) == 1:
                    effective_device_index = "0"
                elif visible_devices and str(self.assigned_gpu) in visible_devices:
                    effective_device_index = str(visible_devices.index(str(self.assigned_gpu)))
                else:
                    effective_device_index = str(self.assigned_gpu)
                properties["DeviceIndex"] = effective_device_index
            print(
                f"[{self.spec.sysname}] Using {self.mysim.platform} "
                f"with assigned_gpu={self.assigned_gpu}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                f"effective DeviceIndex={effective_device_index}, "
                f"properties={properties}"
            )
            simulation = app.simulation.Simulation(
                pdb.topology,
                self.mysim.system,
                integrator,
                platform,
                properties,
            )
        return simulation

    def _backup_old_trajectory(self):
        if self.dcd_path.is_file():
            dt_string = time.strftime("%Y%d%m_%Hh%Mm%Ss")
            backup = self.path / f"backup_{self.mysim.sysname}_{dt_string}.dcd"
            print(f"[{self.spec.sysname}] Backing up existing DCD to {backup}")
            os.rename(self.dcd_path, backup)

    def _initialize_simulation(self):
        _, app, _ = get_openmm_modules()
        checkpoint_exists = self.checkpoint_path.is_file() and self.mysim.restart == "checkpoint"
        append = checkpoint_exists and self.dcd_path.is_file()

        if self.mysim.restart == "pdb" and Path(self.mysim.frestart).is_file():
            pdb = app.pdbfile.PDBFile(self.mysim.frestart)
        else:
            pdb = app.pdbfile.PDBFile(self.mysim.pdb_cg)

        simulation = self._make_openmm_simulation(pdb)
        print(f"[{self.spec.sysname}] Running on {simulation.context.getPlatform().getName()}")

        if checkpoint_exists:
            if append:
                print(f"[{self.spec.sysname}] Appending to existing DCD/log from checkpoint")
            else:
                print(f"[{self.spec.sysname}] No trajectory found, starting a new DCD while loading checkpoint")
            simulation.loadCheckpoint(str(self.checkpoint_path))
            self._write_checkpoint_pdb(simulation)
            return simulation, append

        if self.mysim.slab_eq or self.mysim.box_eq or self.mysim.bilayer_eq:
            print(
                f"[{self.spec.sysname}] Bootstrapping one-time CALVADOS equilibration "
                f"before switching to persistent control"
            )
            self.mysim.simulate()
            pdb = app.pdbfile.PDBFile(self.mysim.pdb_cg)
            simulation = self._make_openmm_simulation(pdb)
            simulation.loadCheckpoint(str(self.checkpoint_path))
            self._write_checkpoint_pdb(simulation)
            return simulation, self.dcd_path.is_file()

        if self.mysim.restart == "pdb":
            print(f"[{self.spec.sysname}] Reading initial coordinates from {self.mysim.frestart}")
        elif self.mysim.restart == "checkpoint":
            print(f"[{self.spec.sysname}] No checkpoint found, starting from system coordinates")
        elif self.mysim.restart is None:
            print(f"[{self.spec.sysname}] Starting from fresh system coordinates")
        else:
            raise ValueError(f"Unsupported restart mode: {self.mysim.restart}")

        self._backup_old_trajectory()
        simulation.context.setPositions(pdb.positions)
        if self.mysim.minimize_energy:
            print(f"[{self.spec.sysname}] Minimizing initial state")
            simulation.minimizeEnergy()

        simulation.saveCheckpoint(str(self.checkpoint_path))
        self._write_checkpoint_pdb(simulation)
        return simulation, False

    def _attach_reporters(self):
        _, app, _ = get_openmm_modules()
        self.simulation.reporters.append(
            app.dcdreporter.DCDReporter(str(self.dcd_path), self.mysim.wfreq, append=self.append)
        )
        self.simulation.reporters.append(
            app.statedatareporter.StateDataReporter(
                str(self.state_log_path),
                self.mysim.logfreq,
                step=True,
                speed=True,
                elapsedTime=True,
                potentialEnergy=self.mysim.report_potential_energy,
                separator="\t",
                append=self.append,
            )
        )

    def _write_checkpoint_pdb(self, simulation=None):
        _, app, _ = get_openmm_modules()
        if simulation is None:
            simulation = self.simulation
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        reporter = app.pdbreporter.PDBReporter(str(self.checkpoint_pdb_path), 0)
        reporter.report(simulation, state)

    def get_state(self):
        return self.simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )

    def state_payload(self):
        _, _, unit = get_openmm_modules()
        state = self.get_state()
        return {
            "positions_nm": quantity_to_numpy(state.getPositions(), pos_unit()),
            "velocities_nm_ps": quantity_to_numpy(state.getVelocities(), vel_unit()),
            "box_nm": quantity_to_numpy(state.getPeriodicBoxVectors(), pos_unit()),
            "energy_kj_mol": float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)),
        }

    def set_state(self, positions_nm, velocities_nm_ps, box_nm):
        positions = numpy_to_quantity(positions_nm, pos_unit())
        velocities = numpy_to_quantity(velocities_nm_ps, vel_unit())
        box = numpy_to_quantity(box_nm, pos_unit())
        a, b, c = box
        self.simulation.context.setPositions(positions)
        self.simulation.context.setPeriodicBoxVectors(a, b, c)
        self.simulation.context.setVelocities(velocities)

    def compute_cross_energy(self, positions_nm):
        _, _, unit = get_openmm_modules()
        original = self.simulation.context.getState(getPositions=True).getPositions()
        self.simulation.context.setPositions(numpy_to_quantity(positions_nm, pos_unit()))
        energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        self.simulation.context.setPositions(original)
        return float(energy)

    def save_runtime_state(self, write_system_pdb=False):
        _, app, _ = get_openmm_modules()
        self.simulation.saveCheckpoint(str(self.checkpoint_path))
        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        checkpoint_reporter = app.pdbreporter.PDBReporter(str(self.checkpoint_pdb_path), 0)
        checkpoint_reporter.report(self.simulation, state)
        if write_system_pdb:
            final_reporter = app.pdbreporter.PDBReporter(str(self.system_pdb_path), 0)
            final_reporter.report(self.simulation, state)

    def run_segment(self, nsteps):
        self.simulation.step(nsteps)
        self.save_runtime_state(write_system_pdb=False)

    def persist_system_xml(self):
        openmm, _, _ = get_openmm_modules()
        self.xml_path.write_text(openmm.XmlSerializer.serialize(self.mysim.system))


class ReplicaWorker:
    def __init__(self, spec: ReplicaSpec, platform_override=None,assigned_gpu=None):
        self.spec = spec
        self.platform_override = platform_override
        self.replica = None
        self.assigned_gpu = assigned_gpu

    def start(self):
        self.replica = PersistentReplica(self.spec, platform_override=self.platform_override,assigned_gpu=self.assigned_gpu)

    def handle(self, message):
        command = message["cmd"]
        if command == "run_segment":
            self.replica.run_segment(message["nsteps"])
            return {"ok": True}
        if command == "get_state":
            return self.replica.state_payload()
        if command == "cross_energy":
            return {"energy_kj_mol": self.replica.compute_cross_energy(message["positions_nm"])}
        if command == "set_state":
            self.replica.set_state(
                message["positions_nm"],
                message["velocities_nm_ps"],
                message["box_nm"],
            )
            self.replica.save_runtime_state(write_system_pdb=False)
            return {"ok": True}
        if command == "finalize":
            self.replica.save_runtime_state(write_system_pdb=True)
            self.replica.persist_system_xml()
            return {"ok": True}
        raise ValueError(f"Unknown worker command: {command}")


def assign_worker_gpu(worker_idx: int, n_workers: int, total_gpus: int | None):
    if total_gpus is None:
        return None
    if total_gpus <= 0:
        raise ValueError("total_gpus must be positive")
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    return min(total_gpus - 1, (worker_idx * total_gpus) // n_workers)


def replica_worker_main(conn, spec: ReplicaSpec, platform_override=None,assigned_gpu=None):
    log_path = spec.path / "run.log"
    log_handle = open(log_path, "a", buffering=1)
    sys.stdout = log_handle
    sys.stderr = log_handle
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    requested_platform = platform_override or spec.config.get("platform")
    if requested_platform == "CUDA" and assigned_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        print(
            f"[{spec.sysname}] Worker pinned to CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"(launcher-assigned GPU {assigned_gpu})"
        )

    worker = ReplicaWorker(spec, platform_override=platform_override,assigned_gpu=assigned_gpu)
    worker.start()
    while True:
        message = conn.recv()
        if message["cmd"] == "shutdown":
            conn.send({"ok": True})
            break
        try:
            conn.send(worker.handle(message))
        except Exception as exc:  # pragma: no cover - best effort transport of worker errors
            conn.send({"error": repr(exc)})
    conn.close()
    log_handle.close()


class WorkerHandle:
    def __init__(self, idx, spec: ReplicaSpec, process: mp.Process, conn):
        self.idx = idx
        self.spec = spec
        self.process = process
        self.conn = conn

    def request(self, command, **kwargs):
        self.conn.send({"cmd": command, **kwargs})

    def recv(self):
        response = self.conn.recv()
        if "error" in response:
            raise RuntimeError(f"Worker {self.idx} ({self.spec.sysname}) failed: {response['error']}")
        return response

    def close(self):
        try:
            self.request("shutdown")
            self.recv()
        finally:
            self.conn.close()
            self.process.join(timeout=5)


class ParallelPersistentREMDController:
    def __init__(self, replicas, platform_override=None, total_gpus=None):
        self.replicas = list(replicas)
        self.workers = []
        n_workers = len(self.replicas)
        for idx, spec in enumerate(self.replicas):
            parent_conn, child_conn = mp.Pipe()
            assigned_gpu = assign_worker_gpu(idx, n_workers, total_gpus)
            process = mp.Process(
                target=replica_worker_main,
                args=(child_conn, spec, platform_override, assigned_gpu),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.workers.append(WorkerHandle(idx, spec, process, parent_conn))

    def run_segments(self, nsteps):
        t0 = time.time()
        for worker in self.workers:
            worker.request("run_segment", nsteps=nsteps)
        for worker in self.workers:
            worker.recv()
        print(f"Propagated {len(self.workers)} persistent simulations in parallel in {time.time() - t0:.2f} s")

    def get_states(self, indices):
        for idx in indices:
            self.workers[idx].request("get_state")
        return {idx: self.workers[idx].recv() for idx in indices}

    def get_cross_energies(self, requests):
        for idx, positions_nm in requests:
            self.workers[idx].request("cross_energy", positions_nm=positions_nm)
        return [self.workers[idx].recv()["energy_kj_mol"] for idx, _ in requests]

    def apply_exchange(self, i, j, state_i, state_j):
        spec_i = self.replicas[i]
        spec_j = self.replicas[j]
        scale_ji = np.sqrt(spec_i.T / spec_j.T)
        scale_ij = np.sqrt(spec_j.T / spec_i.T)

        self.workers[i].request(
            "set_state",
            positions_nm=state_j["positions_nm"],
            velocities_nm_ps=state_j["velocities_nm_ps"] * scale_ji,
            box_nm=state_j["box_nm"],
        )
        self.workers[j].request(
            "set_state",
            positions_nm=state_i["positions_nm"],
            velocities_nm_ps=state_i["velocities_nm_ps"] * scale_ij,
            box_nm=state_i["box_nm"],
        )
        self.workers[i].recv()
        self.workers[j].recv()

    def evaluate_pairs(self, pairs):
        unique_indices = sorted({idx for pair in pairs for idx in pair})
        states = self.get_states(unique_indices)

        requests = []
        for i, j in pairs:
            requests.append((i, states[j]["positions_nm"]))
            requests.append((j, states[i]["positions_nm"]))
        cross_energies = self.get_cross_energies(requests)

        results = []
        cross_iter = iter(cross_energies)
        for i, j in pairs:
            spec_i = self.replicas[i]
            spec_j = self.replicas[j]
            state_i = states[i]
            state_j = states[j]
            Ui_xi = state_i["energy_kj_mol"]
            Uj_xj = state_j["energy_kj_mol"]
            Ui_xj = next(cross_iter)
            Uj_xi = next(cross_iter)
            beta_i = 1.0 / (KB * spec_i.T)
            beta_j = 1.0 / (KB * spec_j.T)
            log_r = -(beta_i * (Ui_xj - Ui_xi) + beta_j * (Uj_xi - Uj_xj))
            u = float(np.random.rand())
            log_u = float(np.log(u))
            accepted = bool(log_u < log_r)

            results.append(
                {
                    "i": i,
                    "j": j,
                    "state_i": spec_i.state_label,
                    "state_j": spec_j.state_label,
                    "sysname_i": spec_i.sysname,
                    "sysname_j": spec_j.sysname,
                    "Ti": spec_i.T,
                    "Tj": spec_j.T,
                    "rti": "" if spec_i.rt is None else spec_i.rt,
                    "rtj": "" if spec_j.rt is None else spec_j.rt,
                    "mode_i": spec_i.sc_mode,
                    "mode_j": spec_j.sc_mode,
                    "r": float(np.exp(min(log_r, 700.0))),
                    "u": u,
                    "log_r": float(log_r),
                    "log_u": log_u,
                    "accepted": accepted,
                    "Ui_xi": np.round(Ui_xi, 2),
                    "Uj_xj": np.round(Uj_xj, 2),
                    "Ui_xj": np.round(Ui_xj, 2),
                    "Uj_xi": np.round(Uj_xi, 2),
                    "_payload_i": state_i,
                    "_payload_j": state_j,
                }
            )
        return results

    def finalize(self):
        for worker in self.workers:
            worker.request("finalize")
        for worker in self.workers:
            worker.recv()

    def shutdown(self):
        for worker in self.workers:
            worker.close()


def run_remd(
    sysname="hpl-dimer",
    path=".",
    platform=None,
    total_gpus=None,
    log_csv="remd_log.csv",
    time_per_script=18,
):
    replicas = discover_replicas(sysname, path)
    if not replicas:
        raise ValueError(f"No replica folders found for prefix {sysname!r} under {path!r}")

    total_steps, seg_steps = get_total_steps(replicas)
    n_segments = total_steps // seg_steps

    log_path = Path(path) / log_csv
    curr_seg = ensure_log(log_path)
    rep_at_state = reconstruct_replica_mapping(log_path, len(replicas))

    print(
        f"Starting parallel persistent in-process REMD controller for {len(replicas)} states "
        f"({n_segments} segments total)"
    )
    controller = ParallelPersistentREMDController(
        replicas,
        platform_override=platform,
        total_gpus=total_gpus,
    )

    try:
        start_time = time.time()
        for seg in range(curr_seg, n_segments):
            if time.time() - start_time > (time_per_script * 3600 - 600):
                print("Aborted loop because less than 10 minutes remain in the controller walltime")
                break

            print(f"\n=== Segment {seg}/{n_segments} starting at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            controller.run_segments(seg_steps)

            pairs = neighbor_pairs(len(replicas), seg)
            rep_before = rep_at_state.copy()
            eval_t0 = time.time()
            results = controller.evaluate_pairs(pairs)
            for result in results:
                i = result["i"]
                j = result["j"]
                if result["accepted"]:
                    controller.apply_exchange(i, j, result["_payload_i"], result["_payload_j"])
                    rep_at_state[i], rep_at_state[j] = rep_at_state[j], rep_at_state[i]

                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "segment": seg,
                    "pair": f"{i}-{j}",
                    "state_i": result["state_i"],
                    "state_j": result["state_j"],
                    "sysname_i": result["sysname_i"],
                    "sysname_j": result["sysname_j"],
                    "Ti": result["Ti"],
                    "Tj": result["Tj"],
                    "rti": result["rti"],
                    "rtj": result["rtj"],
                    "mode_i": result["mode_i"],
                    "mode_j": result["mode_j"],
                    "r": f"{result['r']:.3f}",
                    "u": f"{result['u']:.3f}",
                    "log_r": f"{result['log_r']:.3f}",
                    "log_u": f"{result['log_u']:.3f}",
                    "accepted": result["accepted"],
                    "repid_i": rep_before[i],
                    "repid_j": rep_before[j],
                    "Ui_xi": result["Ui_xi"],
                    "Uj_xj": result["Uj_xj"],
                    "Ui_xj": result["Ui_xj"],
                    "Uj_xi": result["Uj_xi"],
                }
                with log_path.open("a", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(row)
            print(f"Parallel exchange evaluation and accepted swaps took {time.time() - eval_t0:.2f} s")
        controller.finalize()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--sysname", required=True, help="Replica folder prefix")
    parser.add_argument("--path", required=True, help="Path containing the replica folders")
    parser.add_argument(
        "--platform",
        default=None,
        help="Optional override for the OpenMM platform used by all persistent replica simulations",
    )
    parser.add_argument(
        "--total-gpus",
        type=int,
        default=None,
        help="If set for non-CPU runs, distribute workers evenly across this many GPUs via CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--log_csv", default="remd_log.csv", help="Exchange log CSV filename")
    parser.add_argument(
        "--time_per_script",
        type=float,
        default=18,
        help="Controller walltime budget in hours",
    )
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    run_remd(
        sysname=args.sysname,
        path=args.path,
        platform=args.platform,
        total_gpus=args.total_gpus,
        log_csv=args.log_csv,
        time_per_script=args.time_per_script,
    )
