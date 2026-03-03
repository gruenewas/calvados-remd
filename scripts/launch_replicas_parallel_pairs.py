import os, subprocess, glob, yaml, csv, time
from pathlib import Path
import REMD_parallel_pairs as REMD
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)


def launch_replicas(sysname = "hpl-dimer",path = ".",platform="CPU"):

    env = os.environ.copy()
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        env[k] = "1"

    procs = []
    reps = [d for d in sorted(glob.glob(f"{path}/{sysname}_*")) if os.path.isdir(d)]
    for i, d in enumerate(reps):
        if platform == "CUDA":
            gpu_id = i//2
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log = open(os.path.join(d, "run.log"), "a+", buffering=1)
        procs.append(subprocess.Popen(["python", "run.py"], cwd=d, stdout=log, stderr=subprocess.STDOUT, env=env))

    # Wait for all
    exitcodes = [p.wait() for p in procs]
    print("Exit codes:", exitcodes)

def get_current_step(sysname = "hpl-dimer",path = "."):
    curr_step = []
    for d in sorted(glob.glob(f"{path}/{sysname}_*")):
        with open(os.path.join(d,f"{d}.log"),"r") as f:
            log = f.read().splitlines()
            last = log[-1]
            step = last.split()[0]
        curr_step.append(int(step))
    s = set(curr_step)
    if len(s) == 1:
        return s.pop()
    else:
        raise Exception("An error occured: Current Step varies between replicas")

def get_total_steps(sysname = "hpl-dimer",path="."):
    tot_steps = []
    for d in sorted(glob.glob(f"{path}/{sysname}_*")):
        with open(os.path.join(d,"config.yaml")) as f:
            cfg = yaml.safe_load(f)
        tot_steps.append(int(cfg["total_steps"]))
        seg_steps = int(cfg["steps"])
    s = set(tot_steps)
    if len(s) == 1:
        return s.pop(),seg_steps
    else:
        raise Exception("An error occured: Total steps varies between replicas")

def discover_tfolders(sysname,path):
    tfolders = []
    for d in sorted(glob.glob(f"{path}/{sysname}_*")):
        foldername = d.replace(f"{path}/","")
        with open(os.path.join(d,"config.yaml")) as f:
            cfg = yaml.safe_load(f)
        T = cfg["temp"]
        gamma = cfg["friction_coeff"]
        xml = Path(os.path.join(d,f"{foldername}.xml"))
        top_pdb = Path(os.path.join(d,"top.pdb"))
        chk = Path(os.path.join(d,"restart.chk"))
        tfolders.append(REMD.TFolder(d, T, xml, top_pdb, chk, gamma))
    return tfolders


def run_remd(sysname = "hpl-dimer", platform = "CPU",log_csv="remd_log.csv",path = ".",time_per_script=18):

    #read tfolders and sort them from low to high T
    tfolders = discover_tfolders(sysname,path)
    tfolders = sorted(tfolders, key=lambda f: f.T)
    rep_at_T = list(range(len(tfolders)))

    # Logging
    log_path = Path(f"{path}/{log_csv}")
    if not log_path.exists():
        curr_seg = 0
        with log_path.open("w", newline="") as fp:
            csv.DictWriter(fp, fieldnames=[
                "time","segment","pair","Ti","Tj","r","u","log_r","log_u","accepted","repid_i","repid_j","Ui_xi","Uj_xj","Ui_xj","Uj_xi"
            ]).writeheader()
    else:
        log = pd.read_csv(log_path)
        curr_seg = list(log["segment"])[-1] + 1
        
    
    total_steps,seg_steps = get_total_steps(sysname,path)

    n_segments = total_steps//seg_steps

    start_time = time.time()
    print("Starting time for Replica Exchange Simulation: {start_time}")
    for seg in range(curr_seg,n_segments+curr_seg):
        #Safety break, if less than 10 min of script time remain exit loop.
        if time.time() - start_time > (time_per_script*3600 - 600):
            print("Aborted Loop since less than 10m remain")
            break 

        segment_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Segment {seg}/{n_segments} starting at {segment_start_time} ===")
        
        # 1) run one segment at each T
        start_rep = time.time()
        launch_replicas(sysname,path,platform)
        t_rep = time.time() - start_rep
        print(f"Simulating replicas took {t_rep:.2f} s")

        # 2) attempt exchanges between neighbor T-folders (even/odd)
        pairs = REMD.neighbor_pairs(len(tfolders), seg)
        rep_before = rep_at_T.copy()

        # run pairs concurrently; cache is inside workers
        max_workers = min(len(pairs), os.cpu_count() or 2)
        tasks = [(i, j, tfolders[i], tfolders[j]) for (i, j) in pairs]

        start_eval = time.time()
        with ProcessPoolExecutor(max_workers=max_workers,
                                initializer=REMD._worker_init,
                                initargs=(platform, 1)) as pool:
            futs = [pool.submit(REMD._eval_pair_task, t) for t in tasks]
            for fut in as_completed(futs):
                res = fut.result()
                r,u,log_r,log_u = res["r"],res["u"],res["log_r"],res["log_u"]
                i, j = res["i"], res["j"]
                ri, rj = rep_before[i], rep_before[j]  # stable IDs like before
                if res["accepted"]:
                    # update the ladder mapping (swap replica IDs at those temps)
                    rep_at_T[i], rep_at_T[j] = rep_at_T[j], rep_at_T[i]

                # write one CSV row exactly as before
                row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        segment=seg, pair=f"{i}-{j}",
                        Ti=res["Ti"], Tj=res["Tj"],
                        r=f"{r:.3f}", u=f"{u:.3f}",
                        log_r=f"{log_r:.3f}",log_u=f"{log_u:.3f}",
                        accepted=res["accepted"],
                        repid_i=ri, repid_j=rj)
                row.update(res["Es"])
                with log_path.open("a", newline="") as fp:
                    csv.DictWriter(fp, fieldnames=row.keys()).writerow(row)
        t_eval = time.time() - start_eval
        print(f"Metropolis criterion evaluation, swapping and writing log file for segment took {t_eval:.2f} s")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysname", help="input system name")
    ap.add_argument("--path", help="input path name where the simulation folders are located" )
    ap.add_argument("--platform", help="input simulation platform",required=False, default="CPU")
    ap.add_argument("--log_csv", help="input name of remd log file",required=False, default="remd_log.csv")
    args = ap.parse_args()
    run_remd(sysname=args.sysname,
             platform=args.platform,
             log_csv = args.log_csv,
             path = args.path)



