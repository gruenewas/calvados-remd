import os
import re
import csv
import glob
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
import json,traceback

from md_analysis_lib import (
    calc_e2e_corr_function,
    fit_stretched_exponential_decay,
    plot_acf_with_fit,
    compute_e2e_vectors_from_bonds
)


def _set_single_thread_env():
    # Prevent BLAS/OpenMP oversubscription inside each worker
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[k] = "1"


def extract_temp_from_folder(folder):
    base = os.path.basename(os.path.normpath(folder))


    m_rt = re.search(r"_rt(\d+(?:\.\d+)?)nm", base)
    if m_rt:
        return {
            "label": f"rt{m_rt.group(1)}nm",
            "sort_value": float(m_rt.group(1)),
            "type": "rt",
        }

    m = re.search(r"_(\d+(?:\.\d+)?)K$", base)
    if m:
        return {
            "label": f"{m.group(1)}K",
            "sort_value": float(m.group(1)),
            "type": "temperature",
        }

    # fallback: replica naming
    if "replica_" in base:
        idx = int(base.split("_")[-1])
        return {
            "label": f"replica_{idx}",
            "sort_value": idx,
            "type": "replica",
        }

    # generic fallback
    return {
        "label": base,
        "sort_value": base,
        "type": "unknown",
    }

def build_positive_fit_mask(acf, min_points=10):
    """
    Keep the initial positive decay region for fitting.
    """
    acf = np.asarray(acf, dtype=float)

    valid = np.isfinite(acf)
    if not np.any(valid):
        raise ValueError("ACF contains no finite values")

    end = len(acf)
    for i in range(1, len(acf)):
        if not np.isfinite(acf[i]) or acf[i] <= 0:
            end = i
            break

    end = max(end, min_points)
    end = min(end, len(acf))

    mask = np.zeros(len(acf), dtype=bool)
    mask[:end] = True
    mask &= np.isfinite(acf)

    if np.sum(mask) < 3:
        raise ValueError("Not enough valid ACF points for fitting")

    return mask


def run_one_method(R, t, sysname, lbl, comp_dict, out_dir, method):
    """
    Run ACF + fit for one method ('fft' or 'direct').
    """
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()

    rho, rho_comps, time_axis = calc_e2e_corr_function(
        R=R,
        t = t,
        sysname=sysname,
        comp_dict=comp_dict,
        out_path=out_dir,
        plot=False,
        save_txt=True,
        save_Ree=False,
        subtract_mean=False,
        unbiased=True,
        method=method,
        wfreq = 50000,
        factor = 1
        
    )
    results = {}

    for comp in rho_comps.keys():
        acf_mean = np.nanmean(rho_comps[comp], axis=1)
        fit_mask = build_positive_fit_mask(acf_mean, min_points=10)

        popt, pcov, acf_fit,tmean_ana,terr_ana,tmean_mc,terr_mc = fit_stretched_exponential_decay(
            time_axis,
            acf_mean,
            p0=(1.0, 0.5),
            bounds=([0, 0], [np.inf, 1]),
            fit_mask=None,
        )

        elapsed = time.perf_counter() - t0

        tau, beta = popt
        perr = np.sqrt(np.diag(pcov))
        tau_err, beta_err = perr

        corr = pcov[0,1] / np.sqrt(pcov[0,0] * pcov[1,1])

        rel_tau = np.sqrt(pcov[0,0]) / tau
        rel_beta = np.sqrt(pcov[1,1]) / beta

        plot_acf_with_fit(
            time_axis[fit_mask],
            acf_mean[fit_mask],
            popt,
            acf_fit=acf_fit[fit_mask],
            label=f"{sysname} mean ACF",
            title=f"{sysname} vector ACF at {lbl} for {comp}",
            out_path=out_dir,
            filename=f"{sysname}_acf_fit_{method}_{comp}.pdf",
        )

        np.savetxt(
            os.path.join(out_dir, f"{sysname}_acf_mean_{method}_{comp}.txt"),
            np.column_stack([time_axis, acf_mean, acf_fit, fit_mask.astype(int)]),
            header="time_us acf_mean acf_fit fit_mask",
        )

        results[comp] = {
        "method": method,
        "time_axis": time_axis,
        "acf_mean": acf_mean,
        "acf_fit": acf_fit,
        "fit_mask": fit_mask,
        "tau_us": tau,
        "tau_err_us": tau_err,
        "beta": beta,
        "beta_err": beta_err,
        "t_mean_relax_ana" : tmean_ana,
        "t_mean_relax_err_ana" : terr_ana,
        "t_mean_relax_mc" : tmean_mc,
        "t_mean_relax_err_mc" : terr_mc,
        "corr" : corr,
        "rel_tau" : rel_tau,
        "rel_beta" : rel_beta,
        "runtime_s": elapsed,
        "fit_points": int(np.sum(fit_mask)),
        }

    return results

# def run_one(folder,comp_dict,compare=False):
#     _set_single_thread_env()

#     folder = os.path.normpath(folder)
#     sysname = os.path.basename(folder)
#     meta = extract_temp_from_folder(folder)
#     lbl = meta["label"]
#     sort_value = meta["sort_value"]

#     data_dir = os.path.join(folder, "data")
#     ree_file = os.path.join(data_dir, "R_ee.npy")

#     if not os.path.exists(ree_file):
#         try:
#             import MDAnalysis as mda
#             top_file = os.path.join(folder,"top_reindexed.pdb")
#             traj_file = os.path.join(folder,"traj_reindexed_full.dcd")
#             u = mda.Universe(top_file,traj_file)
#             R = compute_e2e_vectors_from_bonds(u,step = 1)
#             np.save(ree_file,R)
#         except Exception as e:
#             print(e)
#             raise FileNotFoundError(f"Missing file: {ree_file} and could not find {top_file} and/or {traj_file} to compute R_ee vectors!")

#     results_root = os.path.join(data_dir, "R_ee_results")
#     os.makedirs(results_root, exist_ok=True)

#     fft_dir = os.path.join(results_root, "fft")
#     direct_dir = os.path.join(results_root, "direct")
#     os.makedirs(fft_dir, exist_ok=True)
#     os.makedirs(direct_dir, exist_ok=True)

#     R = np.load(ree_file)
#     if R.ndim != 3 or R.shape[2] != 3:
#         raise ValueError(f"Unexpected R_ee shape {R.shape} in {ree_file}")

#     nframes, nchains, _ = R.shape
#     if comp_dict is None:
#         comp_dict = {sysname: (0, nchains - 1)}

#     res_fft = run_one_method(
#         R=R,
#         sysname=sysname,
#         lbl=lbl,
#         comp_dict=comp_dict,
#         out_dir=fft_dir,
#         method="fft",
#     )
#     if compare:
#         res_direct = run_one_method(
#             R=R,
#             sysname=sysname,
#             lbl=lbl,
#             comp_dict=comp_dict,
#             out_dir=direct_dir,
#             method="direct",
#         )
#     else:
#         res_direct = res_fft.copy()

#     if res_fft["time_axis"].shape != res_direct["time_axis"].shape:
#         raise ValueError(f"Time axes differ for {sysname}")

#     max_abs_diff_acf = np.nanmax(np.abs(res_fft["acf_mean"] - res_direct["acf_mean"]))
#     mean_abs_diff_acf = np.nanmean(np.abs(res_fft["acf_mean"] - res_direct["acf_mean"]))

#     return {
#         "sysname": sysname,
#         "label": lbl,
#         "sort_value": sort_value,
#         "nframes": nframes,
#         "nchains": nchains,
#         "fft_tau_us": res_fft["tau_us"],
#         "fft_tau_err_us": res_fft["tau_err_us"],
#         "fft_beta": res_fft["beta"],
#         "fft_beta_err": res_fft["beta_err"],
#         "fft_runtime_s": res_fft["runtime_s"],
#         "fft_fit_points": res_fft["fit_points"],
#         "direct_tau_us": res_direct["tau_us"],
#         "direct_tau_err_us": res_direct["tau_err_us"],
#         "direct_beta": res_direct["beta"],
#         "direct_beta_err": res_direct["beta_err"],
#         "direct_runtime_s": res_direct["runtime_s"],
#         "direct_fit_points": res_direct["fit_points"],
#         "delta_tau_us": abs(res_fft["tau_us"] - res_direct["tau_us"]),
#         "delta_beta": abs(res_fft["beta"] - res_direct["beta"]),
#         "max_abs_diff_acf": max_abs_diff_acf,
#         "mean_abs_diff_acf": mean_abs_diff_acf,
#         "fft_out_dir": fft_dir,
#         "direct_out_dir": direct_dir,
#     }


def run_one(folder, comp_dict, compare=False,top_name = None, traj_name = None,overwrite=False,step=None,start=None,stop=None):
    _set_single_thread_env()

    folder = os.path.normpath(folder)
    sysname = os.path.basename(folder)
    meta = extract_temp_from_folder(folder)
    lbl = meta["label"]
    sort_value = meta["sort_value"]

    data_dir = os.path.join(folder, "data")
    os.makedirs(data_dir, exist_ok=True)
    ree_file = os.path.join(data_dir, "R_ee.npy")
    t_file = os.path.join(data_dir, "t.npy")

    top_file = os.path.join(folder, str(top_name))
    traj_file = os.path.join(folder, str(traj_name))

    if top_name is None:
        top_file = os.path.join(folder, f"{folder}.pdb")

    if traj_name is None:
        traj_file = os.path.join(folder, f"{folder}.dcd")


    if not os.path.exists(ree_file) or not os.path.exists(t_file) or overwrite:
        try:
            import MDAnalysis as mda
            u = mda.Universe(top_file, traj_file)
            R,time = compute_e2e_vectors_from_bonds(u, step=step,start=start,stop=stop)
            np.save(ree_file, R)
            np.save(t_file, time)
        except Exception as e:
            raise FileNotFoundError(
                f"Missing file: {ree_file} and/or {t_file} and could not compute it from "
                f"{top_file} and/or {traj_file}. Original error: {e}"
            )

    results_root = os.path.join(data_dir, "R_ee_results")
    os.makedirs(results_root, exist_ok=True)

    if compare:
        fft_dir = os.path.join(results_root, "fft")
        direct_dir = os.path.join(results_root, "direct")
        os.makedirs(fft_dir, exist_ok=True)
        os.makedirs(direct_dir, exist_ok=True)
    else:
        fft_dir = results_root

    R = np.load(ree_file)
    t = np.load(t_file)
    if R.ndim != 3 or R.shape[2] != 3:
        raise ValueError(f"Unexpected R_ee shape {R.shape} in {ree_file}")

    if R.shape[0] != t.shape[0]:
        raise ValueError("R and t must have the same shape!")

    nframes, nchains, _ = R.shape
    if comp_dict is None:
        comp_dict = {sysname: (0, nchains - 1)}

    res_fft = run_one_method(
        R=R,
        t = t,
        sysname=sysname,
        lbl=lbl,
        comp_dict=comp_dict,
        out_dir=fft_dir,
        method="fft",
    )

    results = {}

    if compare:
        res_direct = run_one_method(
            R=R,
            sysname=sysname,
            lbl=lbl,
            comp_dict=comp_dict,
            out_dir=direct_dir,
            method="direct",
        )
        for comp in comp_dict.keys():

            if res_fft[comp]["time_axis"].shape != res_direct[comp]["time_axis"].shape:
                raise ValueError(f"Time axes differ for {sysname}")

            diff = np.abs(res_fft[comp]["acf_mean"] - res_direct[comp]["acf_mean"])
            max_abs_diff_acf = np.nan if np.all(np.isnan(diff)) else np.nanmax(diff)
            mean_abs_diff_acf = np.nan if np.all(np.isnan(diff)) else np.nanmean(diff)

        
            results[comp] = {
            "sysname": sysname,
            "label": lbl,
            "sort_value": sort_value,
            "temp_K": sort_value if isinstance(sort_value, (int, float)) else np.nan,
            "nframes": nframes,
            "nchains": nchains,
            "fft_tau_us": res_fft[comp]["tau_us"],
            "fft_tau_err_us": res_fft[comp]["tau_err_us"],
            "fft_beta": res_fft[comp]["beta"],
            "fft_beta_err": res_fft[comp]["beta_err"],
            "fft_t_mean_relax_ana" : res_fft[comp]["t_mean_relax_ana"],
            "fft_t_mean_relax_err_ana" : res_fft[comp]["t_mean_relax_err_ana"],
            "fft_t_mean_relax_mc" : res_fft[comp]["t_mean_relax_mc"],
            "fft_t_mean_relax_err_mc" : res_fft[comp]["t_mean_relax_err_mc"],
            "fft_corr" : res_fft[comp]["corr"],
            "fft_rel_tau" : res_fft[comp]["rel_tau"],
            "fft_rel_beta" : res_fft[comp]["rel_beta"],
            "fft_runtime_s": res_fft[comp]["runtime_s"],
            "fft_fit_points": res_fft[comp]["fit_points"],
            "direct_tau_us": res_direct[comp]["tau_us"],
            "direct_tau_err_us": res_direct[comp]["tau_err_us"],
            "direct_beta": res_direct[comp]["beta"],
            "direct_beta_err": res_direct[comp]["beta_err"],
            "direct_t_mean_relax_ana" : res_direct[comp]["t_mean_relax_ana"],
            "direct_t_mean_relax_err_ana" : res_direct[comp]["t_mean_relax_ana_err"],
            "direct_t_mean_relax_mc" : res_direct[comp]["t_mean_relax_mc"],
            "direct_t_mean_relax_err_mc" : res_direct[comp]["t_mean_relax_mc_err"],
            "direct_corr" : res_direct[comp]["corr"],
            "direct_rel_tau" : res_direct[comp]["rel_tau"],
            "direct_rel_beta" : res_direct[comp]["rel_beta"],
            "direct_runtime_s": res_direct[comp]["runtime_s"],
            "direct_fit_points": res_direct[comp]["fit_points"],
            "delta_tau_us": abs(res_fft[comp]["tau_us"] - res_direct[comp]["tau_us"]),
            "delta_beta": abs(res_fft[comp]["beta"] - res_direct[comp]["beta"]),
            "max_abs_diff_acf": max_abs_diff_acf,
            "mean_abs_diff_acf": mean_abs_diff_acf,
            "fft_out_dir": fft_dir,
            "direct_out_dir": direct_dir,
            "compare_performed": compare,
            }
            
    else:
        for comp in comp_dict.keys():
            results[comp] = {
            "sysname": sysname,
            "label": lbl,
            "sort_value": sort_value,
            "temp_K": sort_value if isinstance(sort_value, (int, float)) else np.nan,
            "nframes": nframes,
            "nchains": nchains,
            "tau_us": res_fft[comp]["tau_us"],
            "tau_err_us": res_fft[comp]["tau_err_us"],
            "beta": res_fft[comp]["beta"],
            "beta_err": res_fft[comp]["beta_err"],
            "t_mean_relax_ana" : res_fft[comp]["t_mean_relax_ana"],
            "t_mean_relax_err_ana" : res_fft[comp]["t_mean_relax_err_ana"],
            "t_mean_relax_mc" : res_fft[comp]["t_mean_relax_mc"],
            "t_mean_relax_err_mc" : res_fft[comp]["t_mean_relax_err_mc"],
            "corr" : res_fft[comp]["corr"],
            "rel_tau" : res_fft[comp]["rel_tau"],
            "rel_beta" : res_fft[comp]["rel_beta"],
            "runtime_s": res_fft[comp]["runtime_s"],
            "fit_points": res_fft[comp]["fit_points"],
            }
    
    return results


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--wildcard',nargs='?', default='*K', const='.', type=str)
    parser.add_argument('--comp_dict', required=True, default=None,type=str,help='Give the component dictionary as JSON string : {"comp1":"c1idx0,c1idx1" , "comp2" : "c2idx0, c2idx1", ...} ')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--top_name', required=False, default=None,help = 'Specify name of topology file (e.g. top.pdb). If not specified the default [foldername].pdb is used')
    parser.add_argument('--traj_name', required=False, default=None,help = 'Specify name of trajectory file (e.g. traj.dcd). If not specified the default [foldername].dcd is used')
    parser.add_argument('--overwrite',required=False, default=False)
    parser.add_argument("--step", required=False,default=None,type=int)
    parser.add_argument('--start',required=False,default=None,type=int)
    args = parser.parse_args()

    print(args.comp_dict)

    comps = json.loads(args.comp_dict)
    comp_dict = {}
    for key,value in comps.items():
        comp_dict[key] = (int(value.split(",")[0]),int(value.split(",")[1]))

    # pattern = os.path.join(".", f"{args.wildcard}")
    # folders = [p for p in sorted(glob.glob(pattern)) if os.path.isdir(p)]

    # if not folders:
    #     raise RuntimeError(f"No folders found for pattern: {pattern}")

    # max_workers = len(folders)
    # results_comp = []

    # with ProcessPoolExecutor(max_workers=max_workers) as ex:
    #     futures = {ex.submit(run_one, folder,comp_dict,args.compare,args.top_name, args.traj_name): folder for folder in folders}

    #     for fut in as_completed(futures):
    #         folder = futures[fut]
    #         # try:
    #         res = fut.result()
    #         results_comp.append(res)
    #         sysname = next(iter(res.values()))['sysname']
    #         print(f"done: {sysname}")
    #         # except Exception as e:
    #         #     print(f"FAILED: {folder} -> {e}")

    # if not results_comp:
    #     raise RuntimeError("All fits failed")

    pattern = os.path.join(".", f"{args.wildcard}")
    folders = [p for p in sorted(glob.glob(pattern)) if os.path.isdir(p)]

    
    if not folders:
        raise RuntimeError(f"No folders found for pattern: {pattern}")

    valid_folders = []
    skipped_folders = []

    for folder in folders:
        folder = os.path.normpath(folder)
        sysname = os.path.basename(folder)

        data_dir = os.path.join(folder, "data")
        ree_file = os.path.join(data_dir, "R_ee.npy")

        if args.top_name is None:
            top_file = os.path.join(folder, f"{sysname}.pdb")
        else:
            top_file = os.path.join(folder, args.top_name)

        if args.traj_name is None:
            traj_file = os.path.join(folder, f"{sysname}.dcd")
        else:
            traj_file = os.path.join(folder, args.traj_name)

        # Folder is runnable if:
        # 1) precomputed R_ee.npy exists, or
        # 2) topology + trajectory exist so R_ee.npy can be generated
        if os.path.exists(ree_file) or (os.path.exists(top_file) and os.path.exists(traj_file)):
            valid_folders.append(folder)
        else:
            skipped_folders.append({
                "folder": folder,
                "reason": f"missing {ree_file} and could not find usable input files ({top_file}, {traj_file})"
            })

    print(f"Found {len(folders)} folders total")
    print(f"Valid folders: {len(valid_folders)}")
    print(f"Skipped folders: {len(skipped_folders)}")

    for item in skipped_folders:
        print(f"SKIPPED: {item['folder']} -> {item['reason']}")

    if not valid_folders:
        raise RuntimeError("No valid folders found")

    # cap workers to requested Slurm CPUs if available
    max_workers = min(
        len(valid_folders),
        int(os.environ.get("SLURM_CPUS_PER_TASK", len(valid_folders)))
    )

    results_comp = []
    failed_folders = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one, folder, comp_dict, args.compare, args.top_name, args.traj_name,args.overwrite,args.step,args.start): folder
            for folder in valid_folders
        }

        for fut in as_completed(futures):
            folder = futures[fut]
            try:
                res = fut.result()
                results_comp.append(res)
                sysname = next(iter(res.values()))["sysname"]
                print(f"DONE: {sysname}")
            except Exception as e:
                failed_folders.append({"folder": folder, "reason": str(e)})
                full_traceback = traceback.format_exc()
                print(f"FAILED (Full traceback): {folder} -> {full_traceback}")

    print(f"Successful folders: {len(results_comp)}")
    print(f"Failed folders: {len(failed_folders)}")

    for item in failed_folders:
        print(f"FAILED: {item['folder']} -> {item['reason']}")

    if not results_comp:
        raise RuntimeError("All fits failed")

    for comp in comp_dict.keys():

        results = [res[comp] for res in results_comp]

        results.sort(key=lambda x: x["sort_value"])

        summary_dir = os.path.join(".", "R_ee_results")
        os.makedirs(summary_dir, exist_ok=True)

        if args.compare:
            csv_file = os.path.join(summary_dir, f"comparison_{comp}.csv")
            fieldnames = [
                "sysname",
                "label",
                "temp_K",
                "sort_value",
                "nframes",
                "nchains",
                "fft_tau_us",
                "fft_tau_err_us",
                "fft_beta",
                "fft_beta_err",
                "fft_t_mean_relax_ana" ,
                "fft_t_mean_relax_err_ana" ,
                "fft_t_mean_relax_mc" ,
                "fft_t_mean_relax_err_mc" ,
                "fft_corr",
                "fft_rel_tau",
                "fft_rel_beta",
                "fft_runtime_s",
                "fft_fit_points",
                "direct_tau_us",
                "direct_tau_err_us",
                "direct_beta",
                "direct_beta_err",
                "direct_t_mean_relax_ana" ,
                "direct_t_mean_relax_err_ana" ,
                "direct_t_mean_relax_mc" ,
                "direct_t_mean_relax_err_mc" ,
                "direct_corr",
                "direct_rel_tau",
                "direct_rel_beta",
                "direct_runtime_s",
                "direct_fit_points",
                "delta_tau_us",
                "delta_beta",
                "max_abs_diff_acf",
                "mean_abs_diff_acf",
                "fft_out_dir",
                "direct_out_dir",
                "compare_performed"
            ]

        else:
            csv_file = os.path.join(summary_dir, f"results_{comp}.csv")
            fieldnames = [
                "sysname",
                "label",
                "temp_K",
                "sort_value",
                "nframes",
                "nchains",
                "tau_us",
                "tau_err_us",
                "beta",
                "beta_err",
                "t_mean_relax_ana" ,
                "t_mean_relax_err_ana" ,
                "t_mean_relax_mc" ,
                "t_mean_relax_err_mc" ,
                "corr",
                "rel_tau",
                "rel_beta",
                "runtime_s",
                "fit_points",
            ]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nWrote comparison CSV to: {csv_file}")
