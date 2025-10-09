# --- smm_tools_parallel.py ---
import copy
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sqlite3
from datetime import datetime
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Iterable, Tuple
import pickle
from CRS_HMQ_full import MultiworkerContract
from simulate import Simulator

PARAM_NAMES = ['q_0','prod_q','u_bf_m','s_job','alpha','z_corr','prod_var_z']



# -----------------------
# Logging the moments
# -----------------------
def _init_log_db(db_path: str,
                 param_names: list,
                 target_keys: list):
    """
    Create (if needed) a SQLite table to log optimizer evaluations.
    Columns: ts, obj, <params...>, <moments...>
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL;")  # better concurrent writes
        cols = ["ts TEXT", "obj REAL"] \
             + [f"p__{n} REAL" for n in param_names] \
             + [f"m__{k} REAL" for k in target_keys]
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS eval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {", ".join(cols)}
            );
        """)
        con.commit()
    finally:
        con.close()

def _append_log_row(db_path: str,
                    obj_value: float,
                    theta_dict: dict,
                    moment_series: pd.Series,
                    target_keys: list):
    """
    Append one row to the log. Safe to call from multiple processes.
    """
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    # Build column/value dict
    row = {"ts": ts, "obj": float(obj_value)}
    for k, v in theta_dict.items():
        row[f"p__{k}"] = float(v)
    for k in target_keys:
        val = float(moment_series[k]) if (k in moment_series.index and np.isfinite(moment_series[k])) else np.nan
        row[f"m__{k}"] = val

    # write
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        cols = ", ".join(row.keys())
        qmarks = ", ".join(["?"] * len(row))
        con.execute(f"INSERT INTO eval_log ({cols}) VALUES ({qmarks});", list(row.values()))
        con.commit()
    finally:
        con.close()

def export_log_to_csv(db_path: str, csv_path: str):
    con = sqlite3.connect(db_path, timeout=30)
    try:
        df = pd.read_sql_query("SELECT * FROM eval_log ORDER BY id;", con)
    finally:
        con.close()
    df.to_csv(csv_path, index=False)
    return df

#Saving the result
def save_de_result(res, path_prefix: str):
    """
    Save SciPy DE result in three forms:
    - Pickle: <prefix>.pkl
    - JSON-lite summary: <prefix>.json (fun, x, nit, nfev)
    - Numpy: <prefix>_x.npy (best θ)
    """
    with open(f"{path_prefix}.pkl", "wb") as f:
        pickle.dump(res, f)
    summary = {
        "fun": float(res.fun),
        "x": [float(v) for v in np.asarray(res.x).ravel()],
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "message": str(getattr(res, "message", "")),
        "success": bool(getattr(res, "success", False)),
    }
    with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.save(f"{path_prefix}_x.npy", np.asarray(res.x, dtype=float))

def save_fit_table(df_fit: pd.DataFrame, csv_path: str):
    df_fit.to_csv(csv_path, index=False)
# -----------------------
# Core solve/sim
# -----------------------
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

def solve_and_simulate(p, n_rep=5):
    """
    Return raw moms_mean and moms_unt_mean as produced by your simulator,
    without coercing to np.array — so we keep names if they exist.
    """
    #with open("model_GE.pkl", "rb") as file:
    #    all_results = pickle.load(file)
    #model = get_results_for_p(p,all_results)
    #if model is None:
    mwc_J = MultiworkerContract(p)
    model = mwc_J.J_sep(update_eq=1, s=0)
    sim = Simulator(model, p)
    moms_mean, _, moms_unt_mean, _ = sim.simulate_moments_rep(n_rep)
    return moms_mean, moms_unt_mean


# -----------------------
# Moment name extraction
# -----------------------
def extract_values_and_names(moms):
    """
    Robustly extract (values, names) from various possible moment types.

    Supports:
      - pandas.Series (preferred): names from index
      - mapping/dict-like: names from keys (in insertion order)
      - structured numpy array: names from dtype.names
      - plain array-like: no names (returns None)
    """
    # pandas Series
    if isinstance(moms, pd.Series):
        vals = np.asarray(moms.values, dtype=float).ravel()
        names = [str(x) for x in moms.index.tolist()]
        return vals, names

    # numpy structured array
    if isinstance(moms, np.ndarray) and moms.dtype.names is not None:
        names = list(moms.dtype.names)
        vals = np.array([moms[name] for name in names], dtype=float).ravel()
        return vals, names

    # generic mapping (dict-like)
    if hasattr(moms, "keys") and hasattr(moms, "__getitem__") and not isinstance(moms, (np.ndarray, list, tuple, pd.DataFrame)):
        keys = list(moms.keys())
        vals = np.array([moms[k] for k in keys], dtype=float).ravel()
        names = [str(k) for k in keys]
        return vals, names

    # fallback: plain array-like
    vals = np.asarray(moms, dtype=float).ravel()
    return vals, None


# -----------------------
# Grid construction
# -----------------------
def _values_from_bound(bound, n_points):
    if isinstance(bound, tuple) and len(bound) == 2:
        lo, hi = bound
        return np.linspace(lo, hi, n_points)
    return np.asarray(bound, dtype=float)

def make_param_grid(bounds, n_points=5):
    import itertools
    value_lists = [_values_from_bound(bounds[name], n_points) for name in PARAM_NAMES]
    grid = []
    for combo in itertools.product(*value_lists):
        grid.append({name: float(val) for name, val in zip(PARAM_NAMES, combo)})
    return grid


# -----------------------
# Worker (top-level, picklable)
# -----------------------
def _grid_eval_one(args):
    """
    args: (theta_dict, n_rep, use_factory, p_template, p_factory)
    Returns: (theta_dict, moms_values)  where moms_values is 1D float array.
    """
    theta_dict, n_rep, use_factory, p_template, p_factory = args
    try:
        if use_factory and (p_factory is not None):
            p_local = p_factory()
        else:
            p_local = copy.deepcopy(p_template)

        for k, v in theta_dict.items():
            setattr(p_local, k, float(v))

        moms_mean, _ = solve_and_simulate(p_local, n_rep=n_rep)
        vals, _ = extract_values_and_names(moms_mean)
    except Exception:
        vals = np.array([np.nan], dtype=float)

    return theta_dict, vals


# -----------------------
# Parallel grid evaluation
# -----------------------
def grid_moments_parallel(p_template,
                          bounds,
                          n_points: int = 5,
                          n_rep: int = 5,
                          sample: int | None = None,
                          random_state: int = 0,
                          save_csv_path: str | None = None,
                          n_jobs: int = 1,
                          p_factory=None):
    """
    Parallel grid evaluation that preserves moment names as DataFrame columns.
    If your Params object isn't trivially picklable, pass p_factory (top-level function)
    to construct a fresh Params in each worker.
    """
    # Avoid BLAS oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    rng = np.random.default_rng(random_state)
    full_grid = make_param_grid(bounds, n_points=n_points)
    if sample is not None and sample < len(full_grid):
        idx = rng.choice(len(full_grid), size=sample, replace=False)
        param_dicts = [full_grid[i] for i in idx]
    else:
        param_dicts = full_grid

    # --- Probe once to determine #moments and their names ---
    if p_factory is not None:
        p_probe = p_factory()
    else:
        p_probe = copy.deepcopy(p_template)

    # Midpoint probe params
    probe_params = {k: (np.mean(bounds[k]) if isinstance(bounds[k], tuple)
                        else float(np.mean(np.asarray(bounds[k], dtype=float))))
                    for k in PARAM_NAMES}
    for k, v in probe_params.items():
        setattr(p_probe, k, float(v))

    probe_moms, _ = solve_and_simulate(p_probe, n_rep=n_rep)
    probe_vals, probe_names = extract_values_and_names(probe_moms)
    M = int(probe_vals.size)

    # Column names for the DataFrame
    if probe_names is None:
        moment_cols = [f"m{j}" for j in range(M)]
    else:
        # ensure simple string columns (avoid MultiIndex objects etc.)
        moment_cols = [str(n) for n in probe_names]

    # --- Build job args ---
    job_args = [(theta, n_rep, p_factory is not None, p_template, p_factory) for theta in param_dicts]

    # --- Run serial/parallel and assemble rows ---
    records = []
    def _pack_row(theta, vals):
        rec = {**theta}
        row = np.full(M, np.nan)
        row[:min(M, vals.size)] = vals[:min(M, vals.size)]
        for j, col in enumerate(moment_cols):
            rec[col] = row[j]
        return rec

    if n_jobs == 1:
        for ja in job_args:
            theta, vals = _grid_eval_one(ja)
            records.append(_pack_row(theta, vals))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for theta, vals in ex.map(_grid_eval_one, job_args, chunksize=1):
                records.append(_pack_row(theta, vals))

    df = pd.DataFrame.from_records(records)

    if save_csv_path:
        # Append-friendly write (preserves header once)
        write_header = not os.path.exists(save_csv_path)
        df.to_csv(save_csv_path, index=False, mode='a', header=write_header)

    return df

# --- Config for percent-SSE objective ---
@dataclass
class SMMPercentConfig:
    n_rep: int = 5                  # simulation reps per objective eval
    penalty: float = 1e12           # big penalty for failures / missing moments
    param_order: List[str] = None   # default to the global PARAM_NAMES
    eps: float = 1e-10              # to avoid division by zero in % errors
    average: bool = False           # True -> mean of squared % errors; False -> sum

    def __post_init__(self):
        if self.param_order is None:
            self.param_order = list(PARAM_NAMES)

# --- Helper: clip/project theta into bounds (tuple range or nearest in explicit list) ---
def _clip_to_bounds(theta_dict: Dict[str, float],
                    bounds: Dict[str, Tuple[float, float] | Iterable[float]]) -> Dict[str, float]:
    theta_c = {}
    for k, v in theta_dict.items():
        b = bounds[k]
        if isinstance(b, tuple) and len(b) == 2:
            lo, hi = b
            theta_c[k] = float(np.clip(v, lo, hi))
        else:  # explicit candidate values
            vals = np.asarray(b, dtype=float)
            theta_c[k] = float(vals[np.argmin(np.abs(vals - v))])
    return theta_c

# --- Helper: build a named Series of simulated moments ---
def simulate_named_moments(p, n_rep: int = 5) -> pd.Series:
    moms_mean, _ = solve_and_simulate(p, n_rep=n_rep)
    vals, names = extract_values_and_names(moms_mean)
    if names is None:
        names = [f"m{j}" for j in range(len(vals))]
    return pd.Series(vals, index=[str(n) for n in names], dtype=float)

# --- Top-level objective for differential_evolution (picklable) ---
def _smm_percent_obj_top(theta: np.ndarray,
                         p_template,
                         p_factory,
                         p_overrides,
                         bounds,
                         data_moments: Dict[str, float],
                         target_keys: Optional[List[str]],
                         cfg: SMMPercentConfig,
                         log_db_path: Optional[str] = None):
    theta_dict = {name: float(val) for name, val in zip(cfg.param_order, theta)}
    theta_dict = _clip_to_bounds(theta_dict, bounds)

    try:
        p_local = p_factory() if p_factory is not None else copy.deepcopy(p_template)
        if p_overrides:
            for k, v in p_overrides.items():
                setattr(p_local, k, v)
        for k, v in theta_dict.items():
            setattr(p_local, k, v)

        s = simulate_named_moments(p_local, n_rep=cfg.n_rep)

        keys = target_keys if target_keys is not None else list(data_moments.keys())
        errs = []
        for k in keys:
            if k not in s.index or not np.isfinite(s[k]):
                obj = cfg.penalty
                # still log the attempt (with NaNs) if requested
                if log_db_path:
                    _append_log_row(log_db_path, obj, theta_dict, s, keys)
                return obj
            dat = float(data_moments[k])
            denom = max(abs(dat), cfg.eps)
            errs.append(((float(s[k]) - dat) / denom) ** 2)

        obj = float(np.mean(errs) if cfg.average else np.sum(errs))

        if log_db_path:
            _append_log_row(log_db_path, obj, theta_dict, s, keys)

        return obj

    except Exception:
        if log_db_path:
            # Log failure with NaNs for moments
            try:
                s = pd.Series(dtype=float)
                keys = target_keys if target_keys is not None else list(data_moments.keys())
                _append_log_row(log_db_path, cfg.penalty, theta_dict, s, keys)
            except Exception:
                pass
        return cfg.penalty

# --- Global optimizer wrapper (parallelizable) ---
def fit_smm_global_percent(p_template,
                           data_moments: Dict[str, float],
                           bounds,
                           target_keys: Optional[List[str]] = None,
                           cfg: SMMPercentConfig = SMMPercentConfig(),
                           seed: int = 0,
                           maxiter: int = 200,
                           popsize: int = 10,
                           n_jobs: int = 1,
                           p_factory=None,
                           p_overrides: dict | None = None,
                           log_db_path: Optional[str] = None,
                           polish: bool = True):
    from scipy.optimize import differential_evolution

    # numeric bounds
    opt_bounds = []
    for name in cfg.param_order:
        b = bounds[name]
        if isinstance(b, tuple) and len(b) == 2:
            lo, hi = b
        else:
            vals = np.asarray(b, dtype=float)
            lo, hi = float(np.min(vals)), float(np.max(vals))
        opt_bounds.append((lo, hi))

    # default target set
    keys = target_keys if target_keys is not None else list(data_moments.keys())

    # prepare logging db (once, on the driver)
    if log_db_path:
        _init_log_db(log_db_path, cfg.param_order, keys)

    res = differential_evolution(
        _smm_percent_obj_top,
        bounds=opt_bounds,
        args=(p_template, p_factory, p_overrides, bounds, data_moments, keys, cfg, log_db_path),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=polish,
        updating="deferred",
        workers=n_jobs,
    )
    return res

# --- Convenience: evaluate + compare at a given theta ---
def report_fit(theta: np.ndarray,
               data_moments: Dict[str, float],
               bounds: Dict[str, Tuple[float, float] | Iterable[float]],
               target_keys: Optional[List[str]] = None,
               cfg: SMMPercentConfig = SMMPercentConfig(),
               p_template=None,
               p_factory=None,
               p_overrides: dict | None = None) -> pd.DataFrame:
    """Return a tidy comparison table for the selected moments at theta."""
    theta_dict = {name: float(val) for name, val in zip(cfg.param_order, theta)}
    theta_dict = _clip_to_bounds(theta_dict, bounds)

    p_local = p_factory() if p_factory is not None else copy.deepcopy(p_template)

    # <--- apply coarse/fine overrides here (tolerances, sim sizes, etc.)
    if p_overrides:
        for k, v in p_overrides.items():
            setattr(p_local, k, v)

    # then apply theta on top
    for k, v in theta_dict.items():
        setattr(p_local, k, v)

    s = simulate_named_moments(p_local, n_rep=cfg.n_rep)

    keys = target_keys if target_keys is not None else list(data_moments.keys())
    rows = []
    for k in keys:
        sim = np.nan if k not in s.index else float(s[k])
        dat = float(data_moments[k])
        denom = max(abs(dat), cfg.eps)
        pct_err_sq = ((sim - dat) / denom) ** 2 if np.isfinite(sim) else np.nan
        rows.append({"moment": k, "data": dat, "sim": sim, "pct_sq_err": pct_err_sq})
    df = pd.DataFrame(rows)
    df["abs_pct_err_%"] = 100.0 * np.sqrt(df["pct_sq_err"])
    return df

def run():

    from params_factory import make_params_coarse, make_params_fine

    # bounds for your 7 parameters (edit as needed)
    bounds = {
    'q_0':        (0.50, 0.85),
    'prod_q':     (0.20, 0.60),
    'u_bf_m':     (0.20, 0.95),
    's_job':      (0.20, 0.80),
    'alpha':      (0.10, 1.00),   # if really a share/prob in (0,1)
    'z_corr':     (0.80, 0.99),   # AR(1) corr
    'prod_var_z': (0.10, 0.70),
    }

    # your target data moments (names must match your simulator's moment names)
    moms_data = {
    'pr_j2j_an': 0.063,                       # for s_job. but wait: this is YEARLY, not QUARTERLY
    'pr_new_hire': 0.128,                  # for alpha
    'layoffs_share_tercile_0': 0.039,      # for q_0/prod_q #what about this one? in the data it's yearly layoff rate of firms
    # 'layoffs_share_tercile_1': ...
    'layoffs_share_tercile_2': 0.030,      # for q_0/prod_q
    'avg_w_growth_10': 0.33,                # for b
    'sd_dypw': 0.39,                       # for sigma_y
    'autocov_ypw_alt': 0.79,               # for lambda_y
    }



    # --- where to log every evaluation ---
    log_db = "runs/quickscan/eval_log.sqlite"

    # --- quick coarse config ---
    cfg = SMMPercentConfig(n_rep=3, average=False)
    quick_overrides = {'tol_simple_model': 1e-4, 'tol_full_model': 1e-4, 'sim_ni': 3000, 'sim_nrep': 3}

    res = fit_smm_global_percent(
    p_template=None,
    p_factory=make_params_coarse,
    p_overrides=quick_overrides,
    data_moments=moms_data,
    bounds=bounds,
    target_keys=list(moms_data.keys()),
    cfg=cfg,
    seed=123,
    maxiter=35,
    popsize=6,
    n_jobs=24,  #for 12 cores
    log_db_path=log_db,     # <-- enable per-evaluation logging
    polish=True,
    )

    # save the optimizer result
    save_de_result(res, "runs/quickscan/de_result")

    # verify/present a fit table (e.g., at higher fidelity)
    df_fit = report_fit(
    res.x, moms_data, bounds, list(moms_data.keys()),
    cfg=SMMPercentConfig(n_rep=8, average=False),
    p_factory=make_params_fine,
    p_overrides={'tol_simple_model': 1e-6, 'tol_full_model': 1e-6, 'sim_ni': 15000, 'sim_nrep': 8}
    )
    save_fit_table(df_fit, "runs/quickscan/fit_table.csv")

    # export the full evaluation log (all tried θ and their moments/objective)
    df_log = export_log_to_csv(log_db, "runs/quickscan/eval_log.csv")
    print("Logged rows:", len(df_log))

run()