import os, subprocess, glob, yaml, csv, time
from dataclasses import dataclass
from pathlib import Path

import REMD_parallel_pairs as REMD
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

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


@dataclass(frozen=True)
class HREMDReplica:
    tfolder: REMD.TFolder
    sysname: str
    rt: float | None
    sc_mode: str

    @property
    def state_label(self):
        rt_label = "none" if self.rt is None else f"{self.rt:.6g}"
        return f"T={self.tfolder.T:.6g}|rt={rt_label}|mode={self.sc_mode}"


def launch_replicas(sysname="hpl-dimer", path=".", platform="CPU",sims_per_gpu = 3):

    env = os.environ.copy()
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[k] = "1"

    procs = []
    reps = [d for d in sorted(glob.glob(f"{path}/{sysname}_rt*")) if os.path.isdir(d)]
    for i, d in enumerate(reps):
        if platform == "CUDA":
            #gpu_id = 0
            #gpu_id = i//int(sims_per_gpu)
            gpu_id = int(i%2 + 1)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            

        log = open(os.path.join(d, "run.log"), "a+", buffering=1)
        if platform == "CUDA":
            log.write(f"System {os.path.basename(d)} running on CUDA Device {gpu_id}\n")
        else:
            log.write(f"System {os.path.basename(d)} running on CPU\n")
        log.flush()
        procs.append(subprocess.Popen(["python", "run.py"], cwd=d, stdout=log, stderr=subprocess.STDOUT, env=env))

    exitcodes = [p.wait() for p in procs]
    print("Exit codes:", exitcodes)


def get_current_step(sysname="hpl-dimer", path="."):
    curr_step = []
    for d in sorted(glob.glob(f"{path}/{sysname}_rt*")):
        with open(os.path.join(d, f"{d}.log"), "r") as f:
            log = f.read().splitlines()
            last = log[-1]
            step = last.split()[0]
        curr_step.append(int(step))
    s = set(curr_step)
    if len(s) == 1:
        return s.pop()
    else:
        raise Exception("An error occured: Current Step varies between replicas")


def get_total_steps(sysname="hpl-dimer", path="."):
    tot_steps = []
    for d in sorted(glob.glob(f"{path}/{sysname}_rt*")):
        with open(os.path.join(d, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        tot_steps.append(int(cfg["total_steps"]))
        seg_steps = int(cfg["steps"])
    s = set(tot_steps)
    if len(s) == 1:
        return s.pop(), seg_steps
    else:
        raise Exception("An error occured: Total steps varies between replicas")


def discover_hremd_replicas(sysname, path):
    base = Path(path).resolve()
    replicas = []

    for d in sorted(base.glob(f"{sysname}_rt*")):
        if not d.is_dir():
            continue

        with open(d / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        rt = cfg.get("rt")
        if rt is not None:
            rt = float(rt)

        replicas.append(
            HREMDReplica(
                tfolder=REMD.TFolder(
                    path=d,
                    T=float(cfg["temp"]),
                    system_xml=d / f"{d.name}.xml",
                    top_pdb=d / "checkpoint.pdb",
                    chk=d / "restart.chk",
                    gamma=float(cfg["friction_coeff"]),
                ),
                sysname=d.name,
                rt=rt,
                sc_mode=str(cfg.get("sc_mode", "none")),
            )
        )

    return sorted(replicas, key=lambda rep: (rep.tfolder.T, rep.rt if rep.rt is not None else float("-inf"), rep.sysname))


def run_remd(sysname="hpl-dimer", platform="CPU", log_csv="remd_log.csv", path=".", time_per_script=72,sims_per_gpu = 3):

    replicas = discover_hremd_replicas(sysname, path)
    if not replicas:
        raise ValueError(f"No HREMD replica folders found for prefix {sysname!r} under {path!r}")

    tfolders = [rep.tfolder for rep in replicas]
    rep_at_T = list(range(len(tfolders)))

    log_path = Path(f"{path}/{log_csv}")
    if not log_path.exists():
        curr_seg = 0
        with log_path.open("w", newline="") as fp:
            csv.DictWriter(fp, fieldnames=CSV_FIELDS).writeheader()
    else:
        log = pd.read_csv(log_path)
        curr_seg = list(log["segment"])[-1] + 1

    total_steps, seg_steps = get_total_steps(sysname, path)
    n_segments = total_steps // seg_steps

    start_time = time.time()
    print("Starting time for Replica Exchange Simulation: {start_time}")
    for seg in range(curr_seg, n_segments + curr_seg):
        if time.time() - start_time > (time_per_script * 3600 - 600):
            print("Aborted Loop since less than 10m remain")
            break

        segment_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Segment {seg}/{n_segments} starting at {segment_start_time} ===")

        start_rep = time.time()
        launch_replicas(sysname, path, platform,sims_per_gpu)
        t_rep = time.time() - start_rep
        print(f"Simulating replicas took {t_rep:.2f} s")

        pairs = REMD.neighbor_pairs(len(tfolders), seg)
        rep_before = rep_at_T.copy()

        max_workers = min(len(pairs), os.cpu_count() or 2)
        tasks = [(i, j, tfolders[i], tfolders[j]) for (i, j) in pairs]

        start_eval = time.time()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=REMD._worker_init, initargs=(platform, 1)) as pool:
            futs = [pool.submit(REMD._eval_pair_task, t) for t in tasks]
            for fut in as_completed(futs):
                res = fut.result()
                r, u, log_r, log_u = res["r"], res["u"], res["log_r"], res["log_u"]
                i, j = res["i"], res["j"]
                ri, rj = rep_before[i], rep_before[j]
                if res["accepted"]:
                    rep_at_T[i], rep_at_T[j] = rep_at_T[j], rep_at_T[i]

                meta_i = replicas[i]
                meta_j = replicas[j]
                row = dict(
                    time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    segment=seg,
                    pair=f"{i}-{j}",
                    state_i=meta_i.state_label,
                    state_j=meta_j.state_label,
                    sysname_i=meta_i.sysname,
                    sysname_j=meta_j.sysname,
                    Ti=res["Ti"],
                    Tj=res["Tj"],
                    rti="" if meta_i.rt is None else meta_i.rt,
                    rtj="" if meta_j.rt is None else meta_j.rt,
                    mode_i=meta_i.sc_mode,
                    mode_j=meta_j.sc_mode,
                    r=f"{r:.3f}",
                    u=f"{u:.3f}",
                    log_r=f"{log_r:.3f}",
                    log_u=f"{log_u:.3f}",
                    accepted=res["accepted"],
                    repid_i=ri,
                    repid_j=rj,
                )
                row.update(res["Es"])
                with log_path.open("a", newline="") as fp:
                    csv.DictWriter(fp, fieldnames=CSV_FIELDS).writerow(row)
        t_eval = time.time() - start_eval
        print(f"Metropolis criterion evaluation, swapping and writing log file for segment took {t_eval:.2f} s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sysname", help="input system name")
    ap.add_argument("--path", help="input path name where the simulation folders are located")
    ap.add_argument("--platform", help="input simulation platform", required=False, default="CUDA")
    ap.add_argument("--log_csv", help="input name of remd log file", required=False, default="remd_log.csv")
    ap.add_argument("--max_walltime", help="input script walltime to ensure clean simulation exit",type=int)
    ap.add_argument("--sims_per_gpu",help="Number of simulations to placed on each gpu", required=True,type=int)
    args = ap.parse_args()
    run_remd(
        sysname=args.sysname,
        platform=args.platform,
        log_csv=args.log_csv,
        path=args.path,
        time_per_script=args.max_walltime,
        sims_per_gpu=args.sims_per_gpu
    )
