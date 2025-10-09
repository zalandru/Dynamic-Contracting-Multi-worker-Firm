# smm_tools.py
import copy
import itertools
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
# ---- Your model imports (as you already use) ----
from CRS_HMQ_full import MultiworkerContract
from simulate import Simulator

# ---- Parameter names we will vary (in a fixed order) ----
PARAM_NAMES = ['q_0', 'prod_q', 'u_bf_m', 's_job', 'alpha', 'z_corr', 'prod_var_z']

def get_results_for_p(p,all_results):
    # Create the key as a tuple
    #key = (p.num_z,p.num_v,p.num_n,p.n_bar,p.num_q,p.q_0,p.prod_q,p.hire_c,p.k_entry,p.k_f,p.prod_alpha,p.dt)
    key = (p.num_z,p.num_v,p.z_corr,p.prod_var_z,p.num_q,p.q_0,p.prod_q,p.s_job,p.alpha,p.kappa,p.dt,p.u_bf_m,p.min_wage)
    # Check if the key exists in the saved results
    if key in all_results:
        print(key)
        return all_results[key]
    else:
        print(f"No results found for p = {key}")
        return None
# -------------------------
# Core: solve & simulate
# -------------------------
def solve_and_simulate(p, n_rep: int = 5):
    """
    Given a parameter object 'p' with attributes (q_0, prod_q, ...), solve and simulate.
    Returns (moms_mean, moms_unt_mean) as 1D numpy arrays.
    """
    # Solve the model
    #mwc_J = MultiworkerContract(p)
    #model = mwc_J.J_sep(update_eq=1, s=0)
    with open("model_GE.pkl", "rb") as file:
        all_results = pickle.load(file)
    model = get_results_for_p(p,all_results)
    if model is None:
        mwc_J=MultiworkerContract(p)
        model=mwc_J.J_sep(update_eq=1,s=0)

    # Simulate moments
    sim = Simulator(model, p)
    moms_mean, _, moms_unt_mean, _ = sim.simulate_moments_rep(n_rep)
    moms_mean = np.asarray(moms_mean, dtype=float).ravel()
    moms_unt_mean = np.asarray(moms_unt_mean, dtype=float).ravel()
    return moms_mean, moms_unt_mean


def simulate_moments_for_params(p_template, theta_dict: Dict[str, float], n_rep: int = 5,
                                raise_on_fail: bool = False):
    """
    Copy 'p_template', set the parameters in 'theta_dict', then solve & simulate.
    Returns (moms_mean, moms_unt_mean). On failure, returns (np.nan arrays) unless raise_on_fail=True.
    """
    p_local = copy.deepcopy(p_template)
    try:
        for k, v in theta_dict.items():
            setattr(p_local, k, float(v))
        moms_mean, moms_unt_mean = solve_and_simulate(p_local, n_rep=n_rep)
        return moms_mean, moms_unt_mean
    except Exception as e:
        if raise_on_fail:
            raise
        warnings.warn(f"Simulation failed for {theta_dict}: {e}")
        return np.array([np.nan]), np.array([np.nan])


# -------------------------
# Grid exploration utility
# -------------------------
def _values_from_bound(bound, n_points: int) -> np.ndarray:
    """
    If bound is a tuple like (lo, hi), return np.linspace(lo, hi, n_points).
    If bound is an iterable of explicit values, return np.asarray(bound).
    """
    if isinstance(bound, tuple) and len(bound) == 2:
        lo, hi = bound
        return np.linspace(lo, hi, n_points)
    # explicit list of values
    vals = np.asarray(bound, dtype=float)
    if vals.size < 1:
        raise ValueError("Bounds list must contain at least one value.")
    return vals


def make_param_grid(bounds: Dict[str, Iterable[float] | Tuple[float, float]],
                    n_points: int = 5) -> List[Dict[str, float]]:
    """
    Create the full Cartesian grid over PARAM_NAMES.
    For each param, if bounds[param] is (lo, hi) -> linspace(lo,hi,n_points),
    else treat it as an explicit list of values.
    """
    missing = [k for k in PARAM_NAMES if k not in bounds]
    if missing:
        raise ValueError(f"Missing bounds for: {missing}")

    value_lists = [ _values_from_bound(bounds[name], n_points) for name in PARAM_NAMES ]
    grid = []
    for combo in itertools.product(*value_lists):
        grid.append({name: float(val) for name, val in zip(PARAM_NAMES, combo)})
    return grid


def grid_moments(p_template,
                 bounds: Dict[str, Iterable[float] | Tuple[float, float]],
                 n_points: int = 5,
                 n_rep: int = 5,
                 sample: Optional[int] = None,
                 random_state: int = 0,
                 save_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build a grid (5 values per param by default) and simulate moments for each case.
    WARNING: 5^7 = 78,125 cases -> heavy. Use 'sample' to randomly downsample the grid.

    Returns a DataFrame with parameter columns + separate columns for each moment (m0, m1, ...).
    """
    rng = np.random.default_rng(random_state)
    full_grid = make_param_grid(bounds, n_points=n_points)

    if sample is not None and sample < len(full_grid):
        idx = rng.choice(len(full_grid), size=sample, replace=False)
        param_dicts = [full_grid[i] for i in idx]
    else:
        param_dicts = full_grid

    # Probe once to determine number of moments
    # (use the midpoint of bounds as a baseline)
    probe_params = {k: (np.mean(bounds[k]) if isinstance(bounds[k], tuple)
                        else np.mean(_values_from_bound(bounds[k], len(bounds[k]))))
                    for k in PARAM_NAMES}
    probe_moms, _ = simulate_moments_for_params(p_template, probe_params, n_rep=n_rep)
    M = int(np.size(probe_moms))

    records = []
    for theta in param_dicts:
        moms_mean, _ = simulate_moments_for_params(p_template, theta, n_rep=n_rep)
        rec = {**theta}
        # Expand moments into columns m0..m{M-1}
        if moms_mean.size != M:
            # if it failed or dimension mismatched, pad with NaN
            row = np.full(M, np.nan)
            row[:min(M, moms_mean.size)] = moms_mean.ravel()[:min(M, moms_mean.size)]
        else:
            row = moms_mean
        for j in range(M):
            rec[f"m{j}"] = row[j]
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if save_csv_path:
        #df.to_csv(save_csv_path, index=False)
        write_header = not os.path.exists(save_csv_path)  # True if new file
        df.to_csv(save_csv_path, index=False, mode='a', header=write_header)
    return df

def grid_moments_parallel(p_template,
                 bounds,
                 n_points: int = 5,
                 n_rep: int = 5,
                 sample: Optional[int] = None,
                 random_state: int = 0,
                 save_csv_path: Optional[str] = None,
                 n_jobs: int = 1) -> pd.DataFrame:
    """
    Parallel-friendly grid evaluation. Set n_jobs>1 to use multiple processes.
    """
    rng = np.random.default_rng(random_state)
    full_grid = make_param_grid(bounds, n_points=n_points)

    if sample is not None and sample < len(full_grid):
        idx = rng.choice(len(full_grid), size=sample, replace=False)
        param_dicts = [full_grid[i] for i in idx]
    else:
        param_dicts = full_grid

    # Probe to find moment dimension
    probe_params = {k: (np.mean(bounds[k]) if isinstance(bounds[k], tuple)
                        else np.mean(np.asarray(bounds[k], dtype=float)))
                    for k in PARAM_NAMES}
    probe_moms, _ = simulate_moments_for_params(p_template, probe_params, n_rep=n_rep)
    M = int(np.size(probe_moms))

    def _one(theta_dict):
        moms_mean, _ = simulate_moments_for_params(p_template, theta_dict, n_rep=n_rep)
        row = np.full(M, np.nan)
        if moms_mean.size > 0:
            row[:min(M, moms_mean.size)] = np.asarray(moms_mean).ravel()[:min(M, moms_mean.size)]
        rec = {**theta_dict}
        for j in range(M): rec[f"m{j}"] = row[j]
        return rec

    records = []
    if n_jobs == 1:
        for theta in param_dicts:
            records.append(_one(theta))
    else:
        # (Optional) limit BLAS threads per process to avoid oversubscription
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_map = {ex.submit(_one, theta): theta for theta in param_dicts}
            for fut in as_completed(future_map):
                records.append(fut.result())

    df = pd.DataFrame.from_records(records)
    if save_csv_path:
        #df.to_csv(save_csv_path, index=False)
        write_header = not os.path.exists(save_csv_path)  # True if new file
        df.to_csv(save_csv_path, index=False, mode='a', header=write_header)
    return df
# -------------------------
# SMM objective + optimizers
# -------------------------
@dataclass
class SMMConfig:
    n_rep: int = 5                 # number of simulation reps per evaluation
    penalty: float = 1e12          # big penalty on failure/NaNs
    param_order: List[str] = None  # order of params in theta vector; defaults to PARAM_NAMES

    def __post_init__(self):
        if self.param_order is None:
            self.param_order = list(PARAM_NAMES)


def smm_objective(theta: np.ndarray,
                  p_template,
                  data_moments: np.ndarray,
                  bounds: Dict[str, Tuple[float, float] | Iterable[float]],
                  W: Optional[np.ndarray] = None,
                  cfg: SMMConfig = SMMConfig()) -> float:
    """
    SMM distance: (m_sim - m_data)' W (m_sim - m_data)
    - theta: parameter vector in cfg.param_order
    - p_template: your parameter object (will be deep-copied)
    - data_moments: 1D array of data moments
    - bounds: used to clip theta into valid range (defensive)
    - W: weighting matrix (k x k) or 1D vector of standard errors; defaults to Identity if None
    """
    # map theta -> dict
    theta_dict = {name: float(val) for name, val in zip(cfg.param_order, theta)}
    # defensive clipping into bounds if bounds are tuples
    theta_clipped = {}
    for k, v in theta_dict.items():
        b = bounds[k]
        if isinstance(b, tuple) and len(b) == 2:
            lo, hi = b
            theta_clipped[k] = float(np.clip(v, lo, hi))
        else:
            # explicit set of allowed values: project to nearest
            vals = np.asarray(_values_from_bound(b, len(b)))
            theta_clipped[k] = float(vals[np.argmin(np.abs(vals - v))])

    try:
        moms_mean, _ = simulate_moments_for_params(p_template, theta_clipped, n_rep=cfg.n_rep)
    except Exception:
        return cfg.penalty

    m_sim = np.asarray(moms_mean, dtype=float).ravel()
    m_dat = np.asarray(data_moments, dtype=float).ravel()

    if m_sim.size != m_dat.size or np.any(~np.isfinite(m_sim)):
        return cfg.penalty

    diff = m_sim - m_dat
    k = diff.size

    if W is None:
        # Identity weighting
        return float(diff @ diff)

    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        # Interpret as standard errors -> diag(1/sigma^2)
        var = np.square(W)
        with np.errstate(divide='ignore'):
            inv_var = np.where(var > 0, 1.0 / var, 0.0)
        return float(diff @ (inv_var * diff))
    else:
        # full matrix
        return float(diff @ (W @ diff))


# ---- Local optimizer (L-BFGS-B) ----
def fit_smm_local(p_template,
                  data_moments: np.ndarray,
                  bounds: Dict[str, Tuple[float, float] | Iterable[float]],
                  W: Optional[np.ndarray] = None,
                  x0: str | np.ndarray = "midpoint",
                  cfg: SMMConfig = SMMConfig()):
    """
    Run a local optimizer (L-BFGS-B) for SMM.
    - x0="midpoint": use midpoints of tuple-bounds or means of explicit lists.
    - x0 can also be a numeric array in cfg.param_order.
    """
    from scipy.optimize import minimize

    # Build numeric bounds for optimizer; for explicit lists we use (min, max)
    opt_bounds = []
    for name in cfg.param_order:
        b = bounds[name]
        if isinstance(b, tuple) and len(b) == 2:
            lo, hi = b
        else:
            vals = np.asarray(_values_from_bound(b, len(b)))
            lo, hi = float(np.min(vals)), float(np.max(vals))
        opt_bounds.append((lo, hi))

    if isinstance(x0, str) and x0 == "midpoint":
        x0_vec = []
        for name in cfg.param_order:
            b = bounds[name]
            if isinstance(b, tuple) and len(b) == 2:
                lo, hi = b
                x0_vec.append(0.5 * (lo + hi))
            else:
                vals = np.asarray(_values_from_bound(b, len(b)))
                x0_vec.append(float(np.mean(vals)))
        x0_vec = np.asarray(x0_vec, dtype=float)
    else:
        x0_vec = np.asarray(x0, dtype=float)

    obj = lambda th: smm_objective(th, p_template, data_moments, bounds, W, cfg)
    res = minimize(obj, x0_vec, method="L-BFGS-B", bounds=opt_bounds, options={"maxiter": 500})
    return res


# ---- Global optimizer (Differential Evolution) ----
def fit_smm_global(p_template,
                   data_moments: np.ndarray,
                   bounds: Dict[str, Tuple[float, float] | Iterable[float]],
                   W: Optional[np.ndarray] = None,
                   cfg: SMMConfig = SMMConfig(),
                   seed: int = 0,
                   maxiter: int = 1000,
                   popsize: int = 15):
    """
    Run a global optimizer (differential evolution) for SMM.
    This returns a result with .x (best theta) and .fun (best objective). 'polish=True' will
    do an L-BFGS-B polish automatically.
    """
    from scipy.optimize import differential_evolution

    # Numeric optimizer bounds
    opt_bounds = []
    for name in cfg.param_order:
        b = bounds[name]
        if isinstance(b, tuple) and len(b) == 2:
            lo, hi = b
        else:
            vals = np.asarray(_values_from_bound(b, len(b)))
            lo, hi = float(np.min(vals)), float(np.max(vals))
        opt_bounds.append((lo, hi))

    obj = lambda th: smm_objective(th, p_template, data_moments, bounds, W, cfg)
    res = differential_evolution(
        obj,
        bounds=opt_bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=True,
        updating="deferred",
        workers=1,  # set >1 if your model safely supports parallel sims
    )
    return res


# -------------------------
# Convenience: load data moments
# -------------------------
def load_data_moments_csv(path: str, header: bool = True) -> np.ndarray:
    """
    Load a 1D vector of data moments from CSV.
    - If header=True, expects a single row of numbers under headers (or a single column).
    - If header=False, reads raw numeric CSV and flattens.
    """
    if header:
        df = pd.read_csv(path)
        if df.shape[0] == 1:
            return df.to_numpy().ravel().astype(float)
        elif df.shape[1] == 1:
            return df.iloc[:, 0].to_numpy().ravel().astype(float)
        else:
            # Multiple rows: take the first row by default
            return df.iloc[0, :].to_numpy().ravel().astype(float)
    else:
        arr = np.loadtxt(path, delimiter=",", dtype=float)
        return np.asarray(arr).ravel()


def debug():
    # 1) Bounds for the 7 parameters (edit these to match your model’s plausible ranges)
    bounds = {
    'q_0':        (0.5, 0.85),
    'prod_q':     (0.2,  0.6),
    'u_bf_m':     (0.40, 2.50),
    's_job':      (0.2, 0.8),
    'alpha':      (0.10, 1.0),   # if really a share/prob in (0,1)
    'z_corr':     (0.80, 0.99),   # AR(1) corr
    'prod_var_z': (0.10, 0.70),
    }
    from primitives import Parameters
    p = Parameters()

    # 2) Quick landscape scan: 5 values per param (optionally sample to avoid 78,125 cases)
    #    If you *really* want full grid, drop sample=... (but it’ll be slow).
    df_grid = grid_moments(
    p, bounds, n_points=5, n_rep=5,
    sample=5,               # try ~500-2000 for a "general sense"
    random_state=42,
    save_csv_path="grid_moments_sample.csv"
    )
    print(df_grid.head())

#debug()