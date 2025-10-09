# params_factory.py
from primitives import Parameters

def make_params_for_grid():
    # You can also shrink simulation sizes here to speed up the grid
    return Parameters(overwrite={
        # keep all your defaults, only change what helps scans
        'sim_nrep': 5,
        # leave the SMM seven to be overridden per-point by the grid runner
    })

def make_params_coarse():
    # coarser solver + smaller sim for quick global sweep
    return Parameters(overwrite={
        'tol_simple_model': 1e-4,   # looser than 1e-6
        'tol_full_model':   1e-4,   # looser than 1e-6
        'sim_ni':           3000,   # shrink for speed
        'sim_nrep':         3,      # fewer reps for coarse pass
    })

def make_params_fine():
    # tight solver + bigger sim for verification/polish
    return Parameters(overwrite={
        'tol_simple_model': 1e-6,   # or even 1e-7 if you like
        'tol_full_model':   1e-6,
        'sim_ni':           15000,
        'sim_nrep':         8,
    })