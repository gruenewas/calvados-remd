import numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.style.use("ggplot")


def minimum_image_vectors(dr, box):
    box_lengths = np.asarray(box[:3], dtype=float)
    return dr - box_lengths * np.round(dr / box_lengths)


def compute_e2e_vectors_from_bonds(u, step=None):
    if step is None:
        step = 1

    nframes = len(u.trajectory[::step])
    nchains = len(u.segments)
    R = np.zeros((nframes, nchains, 3), dtype=np.float64)

    print("Calculating end-to-end vectors from MIC-corrected bond vectors")
    for tidx, ts in tqdm(enumerate(u.trajectory[::step]), total=nframes):
        box = ts.dimensions

        for cidx, chain in enumerate(u.segments):
            pos = chain.atoms.positions

            bond_vecs = pos[1:] - pos[:-1]
            bond_vecs = minimum_image_vectors(bond_vecs, box)

            R[tidx, cidx] = np.sum(bond_vecs, axis=0)

    return R


def plot_e2e_corr_function(
    folder,
    sysname,
    temp,
    comp_dict,
    top_file="top_reindexed.pdb",
    traj_file="traj_reindexed.dcd",
    out_path=".",
    wfreq=1e5,
    factor=10,
    max_lag=None,
    eps=0.02,
    stable_n=20,
    max_frames=None,
    filename=None,
):
    """
    Calculate the lag-time end-to-end vector autocorrelation function

        phi(t) = <R(t') · R(t'+t)>_{t'}
        C(t)   = phi(t) / phi(0)

    with averaging over all valid time origins t' and over all chains
    belonging to the same component when plotting.

    Early stopping:
        If the component-averaged normalized ACF for all components stays
        within [-eps, +eps] for 'stable_n' consecutive lag frames, the
        lag loop is stopped early.

    Parameters
    ----------
    max_lag : int or None
        Maximum lag in frames. If None, uses nframes - 1.
    eps : float
        Threshold for "close to zero" used for early stopping.
    stable_n : int
        Number of consecutive lag frames within [-eps, +eps] required
        before stopping.
    """

    u = mda.Universe(f"{folder}/{top_file}", f"{folder}/{traj_file}")

    nframes = len(u.trajectory[:max_frames])
    nchains = len(u.segments)

    if max_lag is None:
        max_lag = nframes - 1
    max_lag = min(max_lag, nframes - 1)

    print("Calculating unwrapped end-to-end vectors")
    R = compute_e2e_vectors_from_bonds(u)
    np.save(f"{out_path}/Ree.npy", R)

    phi = np.full((max_lag + 1, nchains), np.nan, dtype=np.float64)

    print("Calculating lag-time end-to-end autocorrelation")
    consecutive_zero = 0
    last_valid_lag = max_lag

    for lag in tqdm(range(max_lag + 1), total=max_lag + 1):
        nvalid = nframes - lag
        if nvalid <= 0:
            last_valid_lag = lag - 1
            break

        Rt = R[:nvalid]
        Rtlag = R[lag:lag + nvalid]

        dots = np.sum(Rt * Rtlag, axis=2)

        phi[lag] = np.mean(dots, axis=0)

        if lag > 0:
            all_close = True
            for key, (start, end) in comp_dict.items():
                phi0_comp = phi[0, start:end + 1]
                philag_comp = phi[lag, start:end + 1]

                valid = np.isfinite(phi0_comp) & (phi0_comp != 0) & np.isfinite(philag_comp)
                if not np.any(valid):
                    all_close = False
                    break

                c_comp = np.nanmean(philag_comp[valid] / phi0_comp[valid])

                if np.abs(c_comp) > eps:
                    all_close = False
                    break

            if all_close:
                consecutive_zero += 1
            else:
                consecutive_zero = 0

            if consecutive_zero >= stable_n:
                last_valid_lag = lag
                print(
                    f"Early stopping at lag {lag} because all component-averaged "
                    f"ACFs stayed within ±{eps} for {stable_n} consecutive frames."
                )
                break

    phi = phi[:last_valid_lag + 1]

    rho = np.full_like(phi, np.nan)
    phi0 = phi[0]

    valid_phi0 = np.isfinite(phi0) & (phi0 != 0)
    rho[:, valid_phi0] = phi[:, valid_phi0] / phi0[valid_phi0]

    rho_comps = {
        prot: rho[:, comp_dict[prot][0]:comp_dict[prot][1] + 1]
        for prot in comp_dict.keys()
    }

    for key, val in rho_comps.items():
        np.savetxt(f"{out_path}/rho_{key}.txt", val)

    fig, ax = plt.subplots()

    total_time = last_valid_lag * factor * wfreq * 1e-8
    time = np.linspace(0, total_time, rho.shape[0])

    for key, val in rho_comps.items():
        ax.plot(time, np.nanmean(val, axis=1), label=key)

    ax.legend(loc="best")
    ax.set_xlabel(r"Lag time [$\mu$s]")
    ax.set_ylabel("Normalized end-to-end ACF")
    prot = sysname.split("_")[0]
    ax.set_title(f"End-to-End correlation function for {prot} at {temp}K")

    fig.tight_layout()
    if not filename:
        filename = f"E2E_corr_function_{sysname}.pdf"
    fig.savefig(f"{out_path}/{filename}")

    return rho, rho_comps, time


def plot_e2e_distance_autocorr(
    folder,
    sysname,
    temp,
    comp_dict,
    top_file="top_reindexed.pdb",
    traj_file="traj_reindexed.dcd",
    out_path=".",
    wfreq=1e5,
    factor=10,
    max_lag=None,
    eps=0.02,
    stable_n=20,
    max_frames=None,
    filename=None,
):
    """
    Calculate the lag-time autocorrelation of centered end-to-end distance
    fluctuations.

        d(t)      = |R_ee(t)|
        delta_d   = d(t) - <d>
        phi(tau)  = <delta_d(t') delta_d(t'+tau)>_{t'}
        C(tau)    = phi(tau) / phi(0)

    PBC handling:
        End-to-end distances are reconstructed from the sum of MIC-corrected
        consecutive bond vectors, not by endpoint MIC.

    Early stop:
        Stop once the component-averaged normalized ACF stays within ±eps
        for stable_n consecutive lag frames.
    """
    u = mda.Universe(f"{folder}/{top_file}", f"{folder}/{traj_file}")

    nframes = len(u.trajectory[:max_frames])
    nchains = len(u.segments)

    if max_lag is None:
        max_lag = nframes - 1
    max_lag = min(max_lag, nframes - 1)

    R = compute_e2e_vectors_from_bonds(u)
    np.save(f"{out_path}/Ree.npy", R)

    d = np.linalg.norm(R, axis=2)

    d_mean = np.mean(d, axis=0)
    d_centered = d - d_mean[None, :]

    phi = np.full((max_lag + 1, nchains), np.nan, dtype=np.float64)

    print("Calculating lag-time autocorrelation of centered e2e distances")
    consecutive_zero = 0
    last_valid_lag = max_lag

    for lag in tqdm(range(max_lag + 1), total=max_lag + 1):
        nvalid = nframes - lag
        if nvalid <= 0:
            last_valid_lag = lag - 1
            break

        prod = d_centered[:nvalid, :] * d_centered[lag:lag + nvalid, :]
        phi[lag, :] = np.mean(prod, axis=0)

        if lag > 0:
            all_close = True
            for key, (start, end) in comp_dict.items():
                phi0_comp = phi[0, start:end + 1]
                philag_comp = phi[lag, start:end + 1]

                valid = np.isfinite(phi0_comp) & (phi0_comp != 0) & np.isfinite(philag_comp)
                if not np.any(valid):
                    all_close = False
                    break

                c_comp = np.nanmean(philag_comp[valid] / phi0_comp[valid])

                if np.abs(c_comp) > eps:
                    all_close = False
                    break

            if all_close:
                consecutive_zero += 1
            else:
                consecutive_zero = 0

            if consecutive_zero >= stable_n:
                last_valid_lag = lag
                print(
                    f"Early stopping at lag {lag} because all component-averaged "
                    f"distance ACFs stayed within ±{eps} for {stable_n} consecutive frames."
                )
                break

    phi = phi[:last_valid_lag + 1]

    rho = np.full_like(phi, np.nan)
    phi0 = phi[0, :]
    valid_phi0 = np.isfinite(phi0) & (phi0 != 0)
    rho[:, valid_phi0] = phi[:, valid_phi0] / phi0[valid_phi0]

    rho_comps = {
        prot: rho[:, comp_dict[prot][0]:comp_dict[prot][1] + 1]
        for prot in comp_dict.keys()
    }

    d_comps = {
        prot: d[:, comp_dict[prot][0]:comp_dict[prot][1] + 1]
        for prot in comp_dict.keys()
    }

    for key, val in rho_comps.items():
        np.savetxt(f"{out_path}/rho_dist_{key}.txt", val)

    for key, val in d_comps.items():
        np.savetxt(f"{out_path}/e2e_dist_{key}.txt", val)

    fig, ax = plt.subplots()

    total_time = last_valid_lag * factor * wfreq * 1e-8
    time = np.linspace(0, total_time, rho.shape[0])

    for key, val in rho_comps.items():
        ax.plot(time, np.nanmean(val, axis=1), label=key)

    ax.legend(loc="best")
    ax.set_xlabel(r"Lag time [$\mu$s]")
    ax.set_ylabel("Normalized end-to-end ACF")
    prot = sysname.split("_")[0]
    ax.set_title(f"End-to-End distance correlation function for {prot} at {temp}K")

    fig.tight_layout()
    if not filename:
        filename = f"E2E_dist_corr_function_{sysname}.pdf"
    fig.savefig(f"{out_path}/{filename}")

    return rho, rho_comps, d, d_comps, time
