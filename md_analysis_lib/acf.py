import numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.signal import correlate
from scipy.optimize import curve_fit

plt.style.use("thesis")


def minimum_image_vectors(dr, box):
    box_lengths = np.asarray(box[:3], dtype=float)
    return dr - box_lengths * np.round(dr / box_lengths)


def compute_e2e_vectors_from_bonds(u, step=None,start=None,stop=None):
    if step is None:
        step = 1

    nframes = len(u.trajectory[start:stop:step])
    nchains = len(u.segments)
    R = np.zeros((nframes, nchains, 3), dtype=np.float64)
    t = np.zeros(nframes,dtype=np.float64)

    print("Calculating end-to-end vectors from MIC-corrected bond vectors")
    for tidx, ts in tqdm(enumerate(u.trajectory[start:stop:step]), total=nframes):
        box = ts.dimensions
        t[tidx] = ts.time*1e-6

        for cidx, chain in enumerate(u.segments):
            pos = chain.atoms.positions
            bond_vecs = pos[1:] - pos[:-1]
            bond_vecs = minimum_image_vectors(bond_vecs, box)
            R[tidx, cidx] = np.sum(bond_vecs, axis=0)

    return R,t


def _normalize_corr(corr, nframes, unbiased=True):
    if unbiased:
        corr = corr / np.arange(nframes, 0, -1, dtype=float)
    else:
        corr = corr / float(nframes)
    return corr


def autocorr_vector(x, subtract_mean=False, unbiased=True, normalize=True, method="fft"):
    """
    Autocorrelation of a single vector time series.

    Parameters
    ----------
    x : ndarray, shape (nframes, 3)
        Vector time series.
    subtract_mean : bool
        If True, subtract mean vector before correlation.
    unbiased : bool
        If True, divide lag tau by (nframes - tau).
        If False, divide all lags by nframes.
    normalize : bool
        If True, divide by corr[0].
    method : {"direct", "fft"}
        Method passed to scipy.signal.correlate.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (nframes, 3)")

    if subtract_mean:
        x = x - np.mean(x, axis=0, keepdims=True)

    nframes = x.shape[0]
    corr = None

    for dim in range(3):
        c = correlate(x[:, dim], x[:, dim], mode="full", method=method)
        c = c[c.size // 2:] #keep only positve lag part of acf

        if corr is None:
            corr = c
        else:
            corr += c

    corr = _normalize_corr(corr, nframes, unbiased=unbiased) #calculate unbiased ACF

    if normalize: #normalize with C(0)  
        if corr[0] != 0:
            corr = corr / corr[0]
        else:
            corr[:] = np.nan

    return corr


def autocorr_scalar(x, subtract_mean=False, unbiased=True, normalize=True, method="fft"):
    """
    Autocorrelation of a single scalar time series.

    Parameters
    ----------
    x : ndarray, shape (nframes,)
        Scalar time series.
    subtract_mean : bool
        If True, subtract mean before correlation.
    unbiased : bool
        If True, divide lag tau by (nframes - tau).
        If False, divide all lags by nframes.
    normalize : bool
        If True, divide by corr[0].
    method : {"direct", "fft"}
        Method passed to scipy.signal.correlate.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must have shape (nframes,)")

    if subtract_mean:
        x = x - np.mean(x)

    nframes = x.shape[0]

    corr = correlate(x, x, mode="full", method=method)
    corr = corr[corr.size // 2:]
    corr = _normalize_corr(corr, nframes, unbiased=unbiased)

    if normalize:
        if corr[0] != 0:
            corr = corr / corr[0]
        else:
            corr[:] = np.nan

    return corr


def calc_e2e_corr_function(
    folder=None,
    sysname=None,
    temp=None,
    comp_dict=None,
    top_file="top_reindexed.pdb",
    traj_file="traj_reindexed.dcd",
    out_path=".",
    wfreq=1e5,
    factor=10,
    max_frames=None,
    filename=None,
    plot=False,
    R=None,
    t = None,
    save_Ree=True,
    save_txt=True,
    subtract_mean=False,
    unbiased=True,
    method="fft",
):
    """
    Calculate normalized end-to-end vector ACF for all chains.

    Parameters
    ----------
    R : ndarray or None, shape (nframes, nchains, 3)
        Precomputed end-to-end vectors. If None, compute from trajectory.
    subtract_mean : bool
        If True, subtract mean vector before ACF.
    unbiased : bool
        If True, divide lag tau by (nframes - tau).
        If False, divide all lags by nframes.
    method : {"direct", "fft"}
        Method used by scipy.signal.correlate.
    """
    if comp_dict is None:
        raise ValueError("comp_dict must be provided")

    if R is None or t is None:
        if folder is None:
            raise ValueError("Either R or folder must be provided")
        u = mda.Universe(f"{folder}/{top_file}", f"{folder}/{traj_file}")
        if max_frames is not None:
            R,time = compute_e2e_vectors_from_bonds(u, step=1)[:max_frames]
        else:
            R,time = compute_e2e_vectors_from_bonds(u, step=1)

        if save_Ree:
            np.save(f"{out_path}/Ree.npy", R)
            np.save(f"{out_path}/t.npy", time)
    else:
        R = np.asarray(R, dtype=float)
        time = np.asarray(t,dtype=float)
        if max_frames is not None:
            R = R[:max_frames]

    nframes, nchains, ndim = R.shape
    if ndim != 3:
        raise ValueError("R must have shape (nframes, nchains, 3)")

    if nframes != time.shape[0]:
        raise ValueError("R and t must have the same shape!")

    rho = np.zeros((nframes, nchains), dtype=float)

    print(f"Calculating vector ACF with scipy.signal.correlate (method='{method}')")
    for cidx in tqdm(range(nchains), total=nchains):
        rho[:, cidx] = autocorr_vector(
            R[:, cidx, :],
            subtract_mean=subtract_mean,
            unbiased=unbiased,
            normalize=True,
            method=method,
        )

    rho_comps = {
        prot: rho[:, start:end + 1]
        for prot, (start, end) in comp_dict.items()
    }

    if save_txt:
        for key, val in rho_comps.items():
            np.savetxt(f"{out_path}/rho_{key}.txt", val)

    if plot:
        fig, ax = plt.subplots()

        for key, val in rho_comps.items():
            ax.plot(time, np.nanmean(val, axis=1), label=key)

        ax.legend(loc="best")
        ax.set_xlabel(r"Lag time [$\mu$s]")
        ax.set_ylabel("Normalized end-to-end ACF")

        if sysname is not None and temp is not None:
            prot = sysname.split("_")[0]
            ax.set_title(f"End-to-End correlation function for {prot} at {temp}K")

        fig.tight_layout()

        if not filename:
            filename = f"E2E_corr_function_{sysname}.pdf" if sysname is not None else "E2E_corr_function.pdf"
        fig.savefig(f"{out_path}/{filename}")

    return rho, rho_comps, time


def calc_e2e_distance_autocorr(
    folder=None,
    sysname=None,
    temp=None,
    comp_dict=None,
    top_file="top_reindexed.pdb",
    traj_file="traj_reindexed.dcd",
    out_path=".",
    wfreq=1e5,
    factor=10,
    max_frames=None,
    filename=None,
    plot=False,
    R=None,
    save_Ree=True,
    save_txt=True,
    subtract_mean=True,
    unbiased=True,
    method="fft",
):
    """
    Calculate normalized end-to-end distance ACF for all chains.

    Parameters
    ----------
    R : ndarray or None, shape (nframes, nchains, 3)
        Precomputed end-to-end vectors. If None, compute from trajectory.
    subtract_mean : bool
        If True, subtract mean end-to-end distance before ACF.
        For distance fluctuation ACF this is usually what you want.
    unbiased : bool
        If True, divide lag tau by (nframes - tau).
        If False, divide all lags by nframes.
    method : {"direct", "fft"}
        Method used by scipy.signal.correlate.
    """
    if comp_dict is None:
        raise ValueError("comp_dict must be provided")

    if R is None:
        if folder is None:
            raise ValueError("Either R or folder must be provided")
        u = mda.Universe(f"{folder}/{top_file}", f"{folder}/{traj_file}")
        if max_frames is not None:
            R = compute_e2e_vectors_from_bonds(u, step=1)[:max_frames]
        else:
            R = compute_e2e_vectors_from_bonds(u, step=1)

        if save_Ree:
            np.save(f"{out_path}/Ree.npy", R)
    else:
        R = np.asarray(R, dtype=float)
        if max_frames is not None:
            R = R[:max_frames]

    nframes, nchains, ndim = R.shape
    if ndim != 3:
        raise ValueError("R must have shape (nframes, nchains, 3)")

    d = np.linalg.norm(R, axis=2)
    rho = np.zeros((nframes, nchains), dtype=float)

    print(f"Calculating scalar distance ACF with scipy.signal.correlate (method='{method}')")
    for cidx in tqdm(range(nchains), total=nchains):
        rho[:, cidx] = autocorr_scalar(
            d[:, cidx],
            subtract_mean=subtract_mean,
            unbiased=unbiased,
            normalize=True,
            method=method,
        )

    rho_comps = {
        prot: rho[:, start:end + 1]
        for prot, (start, end) in comp_dict.items()
    }

    d_comps = {
        prot: d[:, start:end + 1]
        for prot, (start, end) in comp_dict.items()
    }

    if save_txt:
        for key, val in rho_comps.items():
            np.savetxt(f"{out_path}/rho_dist_{key}.txt", val)
        for key, val in d_comps.items():
            np.savetxt(f"{out_path}/e2e_dist_{key}.txt", val)

    total_time = (nframes - 1) * factor * wfreq * 1e-8
    time = np.linspace(0, total_time, nframes)

    if plot:
        fig, ax = plt.subplots()

        for key, val in rho_comps.items():
            ax.plot(time, np.nanmean(val, axis=1), label=key)

        ax.legend(loc="best")
        ax.set_xlabel(r"Lag time [$\mu$s]")
        ax.set_ylabel("Normalized end-to-end distance ACF")

        if sysname is not None and temp is not None:
            prot = sysname.split("_")[0]
            ax.set_title(f"End-to-End distance correlation function for {prot} at {temp}K")

        fig.tight_layout()

        if not filename:
            filename = (
                f"E2E_dist_corr_function_{sysname}.pdf"
                if sysname is not None else
                "E2E_dist_corr_function.pdf"
            )
        fig.savefig(f"{out_path}/{filename}")

    return rho, rho_comps, d, d_comps, time


def stretched_exponential_decay(t, tau, beta):
    return np.exp(-(t / tau) ** beta)

from scipy.special import gamma,psi

def mean_relax_time(tau,beta):

    return tau*(gamma(2/beta)/gamma(1/beta))

def mean_relax_time_err(tau,beta,var_tau,var_beta,cov_taubeta):

    g = gamma(2/beta) / gamma(1/beta)

    dydtau = g

    dydbeta = (tau/beta**2) * g * (psi(1/beta) - 2*psi(2/beta))

    var_y = dydtau**2 * var_tau + dydbeta**2 * var_beta + 2*dydtau*dydbeta*cov_taubeta

    return np.sqrt(var_y)

def mean_relax_time_mc(popt, pcov, n=100000):
    samples = np.random.multivariate_normal(popt, pcov, size=n)
    tau_s = samples[:, 0]
    beta_s = samples[:, 1]

    # keep only physically sensible samples
    mask = (tau_s > 0) & (beta_s > 0)
    vals = mean_relax_time(tau_s[mask], beta_s[mask])

    return np.mean(vals), np.std(vals, ddof=1)

def fit_stretched_exponential_decay(
    t,
    acf,
    p0=(1.0, 0.5),
    bounds=([0, 0], [np.inf, 1]),
    fit_mask=None,
):
    t = np.asarray(t, dtype=float)
    acf = np.asarray(acf, dtype=float)

    if t.ndim != 1 or acf.ndim != 1:
        raise ValueError("t and acf must both be 1D arrays")

    if t.shape[0] != acf.shape[0]:
        raise ValueError("t and acf must have the same length")

    mask = np.isfinite(t) & np.isfinite(acf)

    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != t.shape:
            raise ValueError("fit_mask must have the same shape as t")
        mask &= fit_mask

    t_fit = t[mask]
    acf_fit_data = acf[mask]

    if t_fit.size < 2:
        raise ValueError("Not enough valid data points for fitting")

    popt, pcov = curve_fit(
        stretched_exponential_decay,
        t_fit,
        acf_fit_data,
        p0=p0,
        bounds=bounds,
    )

    acf_fit = stretched_exponential_decay(t, *popt)

    t_mean_ana = mean_relax_time(*popt)
    t_err_ana = mean_relax_time_err(popt[0],popt[1],pcov[0,0],pcov[1,1],pcov[0,1])

    t_mean_mc, t_err_mc = mean_relax_time_mc(popt,pcov)

    return popt, pcov, acf_fit, t_mean_ana, t_err_ana, t_mean_mc,t_err_mc


def plot_acf_with_fit(
    t,
    acf,
    popt,
    acf_fit=None,
    label="ACF",
    fit_label=None,
    xlabel=r"Lag time [$\mu$s]",
    ylabel="Normalized ACF",
    title=None,
    out_path=".",
    filename=None,
):
    """
    Plot an ACF together with its stretched-exponential fit.

    Parameters
    ----------
    t : array-like
        Time axis.
    acf : array-like
        ACF data.
    popt : array-like
        Fitted parameters [tau, beta].
    acf_fit : array-like or None
        Precomputed fit values. If None, they are computed from popt.
    label : str
        Label for the ACF curve.
    fit_label : str or None
        Label for the fit curve. If None, a default label with tau and beta is used.
    xlabel, ylabel, title : str or None
        Plot labels/title.
    out_path : str
        Output directory.
    filename : str or None
        If given, save figure to out_path/filename.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    t = np.asarray(t, dtype=float)
    acf = np.asarray(acf, dtype=float)

    if t.ndim != 1 or acf.ndim != 1:
        raise ValueError("t and acf must both be 1D arrays")

    if t.shape[0] != acf.shape[0]:
        raise ValueError("t and acf must have the same length")

    if acf_fit is None:
        acf_fit = stretched_exponential_decay(t, *popt)
    else:
        acf_fit = np.asarray(acf_fit, dtype=float)
        if acf_fit.shape != t.shape:
            raise ValueError("acf_fit must have the same shape as t")

    tau, beta = popt

    if fit_label is None:
        fit_label = rf"Fit: $\tau={tau:.4g}$, $\beta={beta:.4g}$"

    fig, ax = plt.subplots()

    ax.plot(t, acf, label=label)
    ax.plot(t, acf_fit, label=fit_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="best")
    fig.tight_layout()

    if filename is not None:
        fig.savefig(f"{out_path}/{filename}")

    return fig, ax
