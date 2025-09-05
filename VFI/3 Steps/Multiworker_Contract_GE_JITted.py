import numpy as np
import logging
from scipy.stats import lognorm as lnorm
from typing import Sequence, Optional, Tuple
import opt_einsum as oe

#For printing
import matplotlib.pyplot as plt
import subprocess
import shlex

from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search_GE import JobSearchArray
from valuefunction_multi import PowerFunctionGrid
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import splrep
from scipy.interpolate import splev
from itertools import product #To clean up the code: use nested loops but without actual ugly nesting
import numba as nb
from numba import cuda, float64, prange

import pickle
import datetime
from time import time
import math

ax = np.newaxis
dtype = np.float32

# Set up the basic configuration for the logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
    logging.FileHandler("app.log", mode='w'),  # Log to a file
    #logging.StreamHandler()  # Log to console
    ])
    # Suppress debug logs from the numba library
logging.getLogger('numba').setLevel(logging.WARNING)

def load_pickle_file(new_p_value, pickle_file="results_GE.pkl"):
    # Step 1: Load the existing data from the pickle file
    try:
        with open(pickle_file, "rb") as file:
            all_results = pickle.load(file)
    except FileNotFoundError:
        # If file doesn't exist, start with an empty dictionary
        all_results = {}
        print("No existing file found. Creating a new one.")

def impose_decreasing(M):
    if len(M.shape)==1:
        nv = M.shape[0]
        for v in reversed(range(nv-1)):
            M[v] = np.maximum(M[v],M[v+1])    
    elif len(M.shape)==2:
        nv = M.shape[1]
        for v in reversed(range(nv-1)):
            M[:,v] = np.maximum(M[:,v],M[:,v+1])
    elif len(M.shape)==5:
        nv = M.shape[3]
        for v in reversed(range(nv-1)):
            M[:, :, :, v, :] = np.maximum(M[:, :, :, v, :],M[:, :, :, v+1, :])        
    else:
        nv = M.shape[1]        
        for v in reversed(range(nv-1)):
            M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
    return M
@nb.njit(cache=True)
def impose_increasing(A0):
    A = np.copy(A0)
    nv = len(A)
    for v in range(1,nv):
        A[v] = np.maximum(A[v],A[v-1])
    return A
@nb.njit(cache=True)
def impose_increasing_policy(A0):
    A = np.copy(A0)
    for v in range(1,A.shape[3]):
        A[...,v,:] = np.maximum(A[...,v,:],A[...,v-1,:])
    return A
@nb.njit(cache=True)
def impose_increasing_foc(A0):
    A = np.copy(A0)
    for v in range(1,A.shape[3]):
        A[...,v,:,:] = np.maximum(A[...,v,:,:],A[...,v-1,:,:])
    return A
@nb.njit(cache=True)
def impose_increasing_fsep(A0):
    A = np.copy(A0)
    for v in range(1,A.shape[-1]):
        A[...,v] = np.maximum(A[...,v],A[...,v-1])
    return A

def impose_decreasing_policy(A0):
    A = np.copy(A0)
    for v in reversed(range(A.shape[3]-1)):
        A[...,v,:] = np.maximum(A[...,v,:],A[...,v+1,:])
    return A
@nb.njit(cache=True)
def impose_increasing_W(A0):
    A = np.copy(A0)
    for v in range(1,A.shape[3]):
        A[...,v,:,1] = np.maximum(A[...,v,:,1],A[...,v-1,:,1]+1e-8)
    return A

def array_exp_dist(A,B,h):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    # log_weight = - 0.5*np.power(B/h,2) 
    # # handling underflow gracefully
    # log_weight = log_weight - log_weight.max()
    # weight = np.exp( np.maximum( log_weight, -100))
    # return  (np.power( A-B,2) * weight ).mean() / ( np.power(B,2) * weight ).mean() 
    weight = np.exp( - 0.5*np.power(B/h,2))
    return  (np.power( A-B,2) * weight ).mean() / ( np.power(B,2) * weight ).mean() 
#@nb.njit()
def array_dist(A,B):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    return  (np.power( A-B,2) ).mean() / ( np.power(B,2) ).mean()

#Manual linear interpolator
@nb.njit(cache=True)
def interp( point,x, y):
    if point < x[0]:
        # Point is below the lower bound, return the value at the lower bound
        return y[0]
    elif point > x[-1]:
        # Point is above the upper bound, return the value at the upper bound
        return y[-1]
    else:
        # Point is within bounds, perform linear interpolation
        idx = np.searchsorted(x, point) - 1
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        return y0 + (point - x0) * (y1 - y0) / (x1 - x0)

#Takes a set of multidimensional points and corresponding multi-dimensional y and interpolates over them along the 0-th dimension
@nb.njit(cache=True)
def interp_multidim(points, x,y):
    points = np.asarray(points)
    x = np.asarray(x)

    # Initialize the array for the results
    z = np.zeros_like(points)

    # Compute indices for each point in the flattened array
    for i in np.ndindex(y.shape[1:]):
        p = points[i]
        if p < x[0]:
            z[i] = y[(0,) + i]
        elif p > x[-1]:
            z[i] = y[(-1,) + i]
        else:
            ind = np.searchsorted(x, p) - 1
            x0 = x[ind]
            x1 = x[ind+1]
            y0 = y[(ind,) + i]
            y1 = y[(ind+1,) + i]
            z[i] = y0 + (p - x0) * (y1 - y0) / (x1 - x0)

    return z

@nb.njit(cache=True)
def interp_multidim_extra_dim(points, x,y):
    points = np.asarray(points)
    x = np.asarray(x)

    # Initialize the array for the results
    z = np.zeros_like(points)

    #Loop over the additional dimension that is only in points (for the case, for example, where we're interpolating onto rho_star, which itself depends on iv)
    for v in range(points.shape[0]):
     pointss = points[v,...]
    # Compute indices for each point in the flattened array
     for i in np.ndindex(y.shape[1:]):
        p = pointss[i]
        if p < x[0]:
            z[(v,) + i] = y[(0,) + i]
        elif p > x[-1]:
            z[(v,) + i] = y[(-1,) + i]
        else:
            ind = np.searchsorted(x, p) - 1
            x0 = x[ind]
            x1 = x[ind+1]
            y0 = y[(ind,) + i]
            y1 = y[(ind+1,) + i]
            z[(v,) + i] = y0 + (p - x0) * (y1 - y0) / (x1 - x0)

    return z


@cuda.jit(device=True)
def compute_indices(pos, num_n, num_n1, num_v, num_q):
    # Given the flat index 'pos' and the dimensions,
    # compute the multi-dimensional indices (iz, in0, in1, iv, iq)
    iq = pos % num_q
    pos //= num_q
    iv = pos % num_v
    pos //= num_v
    in1 = pos % num_n1
    pos //= num_n1
    in0 = pos % num_n
    pos //= num_n
    iz = pos  # remaining value
    return iz, in0, in1, iv, iq
@cuda.jit(device=True)
def interp_gpu(point, x, y, n):
    # If the point is outside the bounds, return the corresponding edge value.
    if point <= x[0]:
        return y[0]
    elif point >= x[n-1]:
        return y[n-1]
    else:
        idx = 0
        # Manually search for the index where x[idx] <= point < x[idx+1]
        for i in range(n - 1):
            if x[i] <= point and point < x[i+1]:
                idx = i
                break
        # Avoid division by zero.
        if x[idx+1] == x[idx]:
            return y[idx]
        return y[idx] + (point - x[idx]) * (y[idx+1] - y[idx]) / (x[idx+1] - x[idx])
@cuda.jit
def rho_star_kernel(foc, rho_grid, N_grid, N_grid1, n_bar1, rho_star_out, num_z, num_n, num_n1, num_v, num_q):
    pos = cuda.grid(1)
    #num_z, num_n, num_n1, num_v, num_q = foc.shape
    total = num_z * num_n * num_n1 * num_v * num_q
    if pos >= total:
        return
    # Compute multi-dimensional indices in one go
    iz, in0, in1, iv, iq = compute_indices(pos, num_n, num_n1, num_v, num_q)    

    # Skip if condition not met.
    if N_grid[in0] + N_grid1[in1] > n_bar1:
        rho_star_out[iz, in0, in1, iv, iq] = 0.0
        return

    # Find where foc crosses zero and interpolate.
    rho_star_out[iz, in0, in1, iv, iq] = interp_gpu(0,foc[iz,in0,in1,:,iv,iq],rho_grid,num_v)


@nb.njit(cache=True,parallel=True)
def solve_rho(rho_grid, N_grid, N_grid1, foc, rho_star, num_z, num_n, num_n1, n_bar1, num_v, num_q):
    for in0 in prange(num_n): 
        for in1 in prange(num_n1):
         if (N_grid[in0] + N_grid1[in1] > n_bar1):
          continue
         for iz in prange(num_z):
          for iv in prange(num_v):
            for iq in prange(num_q):
                rho_star[iz,in0, in1, iv, iq] = interp(0,
                                                    foc[iz, in0, in1, :, iv, iq],
                                                    rho_grid[:])  
    return rho_star
@nb.njit(cache=True,parallel=True)
def n0(Rhod0_diff, n0_star, N_grid, Ihire, hire_c):
    for idx in np.argwhere(Ihire):
        n0_star[idx[0], idx[1], idx[2], idx[3], idx[4]] = interp( -hire_c ,Rhod0_diff[idx[0], idx[1], idx[2], idx[3], idx[4],:],N_grid[1:]) #oh shit, should we also account for how that affects the worker value???
    print("n0_star borders", n0_star.min(), n0_star.max())   
    return n0_star 
@nb.njit(cache=True,parallel=True)
def EJs(EJ_star, EW_star, Jd0, Wd0, n0_star, N_grid, num_z, num_n, num_v, num_q):
    for iz in prange(num_z):
        for in0 in prange(num_n):
            for in1 in prange(num_n):
                for iv in prange(num_v):
                 for iq in prange(num_q):
                    EJ_star[iz,in0,in1,iv,iq] = interp(n0_star[iz,in0,in1,iv,iq],N_grid,Jd0[iz,in0,in1,iv,iq,:])
                    EW_star[iz,in0,in1,iv,iq] = interp(n0_star[iz,in0,in1,iv,iq],N_grid,Wd0[iz,in0,in1,iv,iq,:])
    return EJ_star, EW_star
@nb.njit(cache=True)
def ERhoDerivative(Jd0,Wd0,ceiln1,floorn1,n1_star,rho_star,N_grid1,num_z,num_n,n_bar,num_v,num_q):
                ERhoDeriv = np.zeros_like(ceiln1)
                for iz in range(num_z):
                 for in0 in range(num_n):
                  for in1 in range(num_n):
                    for iv in range(num_v):
                        for iq in range(num_q):
                         if ceiln1[iz,in0,in1,iv,iq]==0:
                            continue
                         if N_grid1[floorn1[iz,in0,in1,iv,iq]]>=n_bar:
                            continue
                         if floorn1[iz,in0,in1,iv,iq] == ceiln1[iz,in0,in1,iv,iq]:
                            ERhoDeriv[iz,in0,in1,iv,iq] = (Jd0[iz,in0,in1,iv,iq,ceiln1[iz,in0,in1,iv,iq]+1]-Jd0[iz,in0,in1,iv,iq,floorn1[iz,in0,in1,iv,iq]-1]) / (N_grid1[ceiln1[iz,in0,in1,iv,iq]+1]-N_grid1[floorn1[iz,in0,in1,iv,iq]-1])
                         else:
                            ERhoDeriv[iz,in0,in1,iv,iq] = (Jd0[iz,in0,in1,iv,iq,ceiln1[iz,in0,in1,iv,iq]]-Jd0[iz,in0,in1,iv,iq,floorn1[iz,in0,in1,iv,iq]]) / (N_grid1[ceiln1[iz,in0,in1,iv,iq]]-N_grid1[floorn1[iz,in0,in1,iv,iq]])
                return ERhoDeriv
@nb.njit(cache=True,parallel=True)
def solve_sep(sep_star,foc_sep,sep_grid,num_z,num_v,num_n,num_n1,num_q,tenure):
                #foc_sep = - J_fut_deriv_n * pc_temp[...,ax] * size[...,ax, 0] + J_fut_deriv_q * q_deriv_s - size[...,ax, 0] * (re_star[...,ax]+EW_star[...,ax] - EU) / inv_util_1d  
                #Note that, if juniors aren't fired, seniors aren't either: the size impact is the same, the quality impact (at 0) is worse, the firing cost is higher
                #Overall, the total 
                #foc_sep = J_s_deriv - size[...,ax,0] * (re_star[...,ax]+EW_star[...,ax] - EU) / inv_util_1d + size[...,ax,0] * pc_temp[...,ax] * rho_star[...,ax] * EW_star[...,ax]

                #print("foc_sep difference", np.max(np.abs(foc_sep-foc_sep_diff)))
                #NOTE: EW_star and re_star here are constant, not dependent on s at all (although they kinda should be)
                #Calculating the separations. Can drop this into jit surely
                for in0 in prange(num_n):
                    if (in0==0) & (tenure==0): #Give a chance for senior firings. 
                        sep_star[:,in0,...] = 0
                        continue
                    for iz in prange(num_z):
                     for in1 in prange(num_n1):
                      for iv in prange(num_v):
                       for iq in prange(num_q):                         
                        sep_star[iz,in0,in1,iv,iq] = interp(0,foc_sep[iz,in0,in1,iv,iq,:],sep_grid) 
                        
                return sep_star
#Interpolation functions instead of RegularGridInterpolator
@nb.njit(cache=True,parallel=True)
def precompute_interp_params(points, grid):
    """
    Precompute interpolation parameters for multiple points.
    
    Parameters:
      points : (n_points, n_dims) array of query points.
      grid   : tuple of 1D arrays, one per dimension (each sorted in ascending order).
      
    Returns:
      indices  : (n_points, n_dims) array of lower index for each dimension.
      fractions: (n_points, n_dims) array of fractional distances within each interval.
    """
    n_points, n_dims = points.shape
    indices = np.empty((n_points, n_dims), dtype=np.int64)
    fractions = np.empty((n_points, n_dims), dtype=np.float64)
    
    for i in prange(n_points):
        for d in prange(n_dims):
            g = grid[d]
            x = points[i, d]
            if x <= g[0]:
                indices[i, d] = 0
                fractions[i, d] = 0.0
            elif x >= g[-1]:
                indices[i, d] = len(g) - 2
                fractions[i, d] = 1.0
            else:
                j = 0
                # Find the right interval: g[j] <= x < g[j+1]
                while j < len(g) - 1 and x >= g[j+1]:
                    j += 1
                indices[i, d] = j
                fractions[i, d] = (x - g[j]) / (g[j+1] - g[j])
    return indices, fractions
@nb.njit(cache=True)
def multi_index_to_flat(idx, shape):
    """
    Convert a multi-dimensional index (given as a 1D array) into a flat index,
    assuming C-contiguous storage.
    
    Parameters:
      idx   : 1D array of indices.
      shape : shape of the multidimensional array.
    
    Returns:
      flat_index corresponding to idx.
    """
    flat_index = 0
    stride = 1
    n_dims = len(shape)
    for d in range(n_dims-1, -1, -1):
        flat_index += idx[d] * stride
        stride *= shape[d]
    return flat_index
@nb.njit(cache=True,parallel=True)
def multilinear_interp_at_points(indices, fractions, values):
    """
    Evaluate multilinear interpolation at multiple points.
    
    Parameters:
      indices  : (n_points, n_dims) array of lower grid indices (precomputed).
      fractions: (n_points, n_dims) array of fractional distances (precomputed).
      values   : N-dimensional array of values defined on the grid.
      
    Returns:
      result   : 1D array of interpolated values (one per query point).
    """
    n_points, n_dims = indices.shape
    result = np.empty(n_points, dtype=values.dtype)
    n_corners = 1 << n_dims  # 2**n_dims
    
    shape = values.shape  # shape of the grid values array
    
    for i in prange(n_points):
        res = 0.0
        # Loop over all 2^n_dims corners of the hypercube surrounding the point.
        for corner in prange(n_corners):
            weight = 1.0
            idx = np.empty(n_dims, dtype=np.int64)
            for d in prange(n_dims):
                # Test the d-th bit of corner:
                if (corner >> d) & 1:
                    weight *= fractions[i, d]
                    idx[d] = indices[i, d] + 1
                else:
                    weight *= (1.0 - fractions[i, d])
                    idx[d] = indices[i, d]
            flat_idx = multi_index_to_flat(idx, shape)
            res += weight * values.flat[flat_idx]
        result[i] = res
    return result
@nb.njit(cache=True)
def multilinear_interp(points, grid, values):
    """
    Wrapper function to perform multilinear interpolation for multiple points.
    
    Parameters:
      points : (n_points, n_dims) array of query points.
      grid   : tuple of 1D arrays (one per dimension).
      values : N-dimensional array defined on the grid.
      
    Returns:
      Interpolated values at the provided points.
    """
    indices, fractions = precompute_interp_params(points, grid)
    return multilinear_interp_at_points(indices, fractions, values)
def tuple_into_2darray(query_points):
    # Assume each element in the tuple is a NumPy array of the same shape.
    #shape = query_points[0].shape
    n_points = query_points[0].size
    n_dims = len(query_points)
    pts = np.empty((n_points, n_dims), dtype=np.float64)
    for d in range(n_dims):
        pts[:, d] = query_points[d].ravel()
    return pts
    #flat_result = multilinear_interp(pts, grid, values)
    #return flat_result.reshape(shape)
@nb.njit(cache=True, parallel=True)
def solve_policies(ite_num, rho_star, n0_star, sep_star, foc_sep, sep_grid, hire_c, Ihire, Rhod0_diff,
                   rho_grid, N_grid, N_grid1, foc, num_z, num_n, num_n1, n_bar1, num_v, num_q, tenure):
    # Precompute constants that do not change within the loops
    ite_ge_20 = (ite_num >= 20)
    ite_ge_10 = (ite_num >= 10)
    tenure_nonzero = (tenure != 0)

    for in0 in prange(num_n):
        # This condition is invariant over iz, in1, iv, iq
        cond_sep = ((in0 > 0) or tenure_nonzero) and ite_ge_20
        for in1 in prange(num_n1):
            # Compute condition dependent only on in0 and in1 once per iteration
            cond_rho = (N_grid[in0] + N_grid1[in1] <= n_bar1)
            for iz in prange(num_z):
                for iv in prange(num_v):
                    for iq in prange(num_q):
                        #if cond_rho:
                        rho_star[iz, in0, in1, iv, iq] = interp(0,
                                                                    foc[iz, in0, in1, :, iv, iq],
                                                                    rho_grid)
                        if cond_sep:
                            sep_star[iz, in0, in1, iv, iq] = interp(0,
                                                                    foc_sep[iz, in0, in1, iv, iq, :],
                                                                    sep_grid)
                        if ite_ge_10 and Ihire[iz, in0, in1, iv, iq]:
                            n0_star[iz, in0, in1, iv, iq] = interp(-hire_c,
                                                                   Rhod0_diff[iz, in0, in1, iv, iq, :],
                                                                   N_grid[1:])
    return rho_star, n0_star, sep_star

@nb.njit(cache=True,parallel=True)
def Rhod0_int(Rhod0,ERho,N_grid1,rho_grid,Q_grid,points,num_z,num_n):
    shape = Rhod0[0,...,0].shape
    for iz in prange(num_z):
     for in00 in prange(num_n):
        flat_result = multilinear_interp(points[iz], (( N_grid1, rho_grid, Q_grid)), ERho[iz, in00, ...])
        Rhod0[iz,...,in00] =  flat_result.reshape(shape)
    return Rhod0
@nb.njit(cache=True,parallel=True)
def Js_int(Js,ERho_s,N_grid,N_grid1,rho_grid,Q_grid,points,num_z):
    shape = Js[0,...].shape
    for iz in prange(num_z):
        flat_result = multilinear_interp(points[iz], ((N_grid, N_grid1, rho_grid, Q_grid)), ERho_s[iz, ...])
        Js[iz,...] =  flat_result.reshape(shape)
    return Js
@nb.njit(cache=True,parallel=True)
def Values_int(ERho_star,EW_star,ERho,EW,N_grid,N_grid1,rho_grid,Q_grid,points,num_z):
    shape = ERho_star[0,...].shape
    for iz in prange(num_z):
        flat_result = multilinear_interp(points[iz], ((N_grid, N_grid1, rho_grid, Q_grid)), ERho[iz, ...])
        ERho_star[iz,...] =  flat_result.reshape(shape)
        flat_result = multilinear_interp(points[iz], ((N_grid, N_grid1, rho_grid, Q_grid)), EW[iz, ...])
        EW_star[iz,...] =  flat_result.reshape(shape)        
    return ERho_star,EW_star
#Get indices from a flat index
@nb.njit(cache=True)
def decode_flat_index(pos, dims):
    # dims: 1D int64 array of sizes
    nd = dims.size
    idx = np.empty(nd, np.int64)
    for d in range(nd-1, -1, -1):
        idx[d] = pos % dims[d]
        pos //= dims[d]
    return idx
#Layout class to handle the dimensions
class Layout:
    def __init__(self, K):
        # Order: [z] + [n (K-1 dims)] + [n1 (1 dim)] + [v (K-1 dims)] + [q (K-1 dims)]
        # indices:
        self.ax_z = 0
        self.ax_ns = list(range(1, K))         # juniors at each step before seniors
        self.ax_n1 = K                         # seniors
        self.ax_vs = list(range(K+1, K+(K-1)+1))
        self.ax_qs = list(range(K+(K-1)+1, K+2*(K-1)+1))

        # Convenience – “primary” dims (first v/q step) when you conceptually had one
        self.ax_v = self.ax_vs[0] if self.ax_vs else None
        self.ax_q = self.ax_qs[0] if self.ax_qs else None

def dims_for_K(p, K):
    dims = [p.num_z]
    dims.extend([p.num_n]*(K-1))
    dims.extend([p.num_n1])            # seniors
    dims.extend([p.num_v]*(K-1))
    dims.extend([p.num_q]*(K-1))
    return dims

#Smoothing functions
def _sort_if_needed(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure 1D; x and y must be 1D views
    if np.all(np.diff(x) > 0):
        return x, y
    order = np.argsort(x)
    return x[order], y[order]

def _smooth_series(x, y, s=0.0, k=3):
    # x, y are 1D
    x1, y1 = _sort_if_needed(x, y)
    tck = splrep(x1, y1, s=s, k=k)
    return splev(x, tck)  # evaluate back on the original (unsorted) grid

def smooth_along_single_v_axis_inplace(
    J: np.ndarray,
    W: np.ndarray,         # assumes an extra trailing "step" axis of size K
    Rho: np.ndarray,
    rho_grid: np.ndarray,  # 1D, shared x for W and Rho smoothing
    v_axis: int,           # which axis in J/Rho/W[..., step] is "v"
    step_idx: int,         # which step slice of W to pair with this v_axis
    s: float = 0.0,
    k: int = 3,
    *,
    also_smooth_J_vs_W: bool = True,
    also_smooth_W_vs_rho: bool = True,
    also_smooth_Rho_vs_rho: bool = True,
):
    """
    In-place smoothing along the v_axis:
      - J(·) as a function of W(·, step_idx)
      - W(·, step_idx) as a function of rho_grid
      - Rho(·) as a function of rho_grid
    """

    # Bring the v-axis to the end to get clean 1D series slices
    Jv   = np.moveaxis(J,   v_axis, -1)
    Wv   = np.moveaxis(W[..., step_idx], v_axis, -1)  # fix step, then move v to the end
    Rhov = np.moveaxis(Rho, v_axis, -1)

    # The leading shape (all dims except the last "v" dim)
    lead_shape = Jv.shape[:-1]
    nv = Jv.shape[-1]

    # Basic sanity
    assert Wv.shape[-1] == nv and Rhov.shape[-1] == nv
    assert rho_grid.ndim == 1 and rho_grid.size == nv

    # Iterate over all other indices; operate on 1D vectors along the last axis
    it = np.ndindex(lead_shape)
    for idx in it:
        # 1) J vs W (per-slice x varies)
        if also_smooth_J_vs_W:
            xW = Wv[idx]          # shape (nv,)
            yJ = Jv[idx]          # shape (nv,)
            Jv[idx] = _smooth_series(xW, yJ, s=s, k=k)

        # 2) W vs rho_grid (shared x)
        if also_smooth_W_vs_rho:
            yW = Wv[idx]
            tckW = splrep(rho_grid, yW, s=s, k=k)
            Wv[idx] = splev(rho_grid, tckW)

        # 3) Rho vs rho_grid (shared x)
        if also_smooth_Rho_vs_rho:
            yR = Rhov[idx]
            tckR = splrep(rho_grid, yR, s=s, k=k)
            Rhov[idx] = splev(rho_grid, tckR)

    # Move axes back in place (in-place update)
    J =                 np.moveaxis(Jv,   -1, v_axis)
    W[..., step_idx] =  np.moveaxis(Wv,   -1, v_axis)
    Rho =               np.moveaxis(Rhov, -1, v_axis)

def smooth_all_v_axes_inplace(
    J: np.ndarray,
    W: np.ndarray,
    Rho: np.ndarray,
    rho_grid: np.ndarray,
    v_axes: Sequence[int],
    step_indices: Optional[Sequence[int]] = None,
    s: float = 0.0,
    k: int = 3,
):
    """
    Smooth along every v-axis. By convention, map v_axes[i] to step_indices[i].
    If step_indices is None, use step_indices = [1, 2, ..., len(v_axes)].
    """
    if step_indices is None:
        step_indices = list(range(1, len(v_axes)+1))
    for v_ax, st in zip(v_axes, step_indices):
        smooth_along_single_v_axis_inplace(
            J, W, Rho, rho_grid, v_axis=v_ax, step_idx=st, s=s, k=k
        )
class MultiworkerContract:
    """
        This solves a contract model with DRS production, hirings, and heterogeneous match quality.
    """
    def __init__(self, K=2, input_param=None, js=None):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
    
        self.log = logging.getLogger('MWF with Hiring')
        self.log.setLevel(logging.INFO)
        self.p = input_param
        self.K = K
        #K-based layofs
        self.layout = Layout(K)
        dimensions = dims_for_K(self.p, K)
        L = self.layout
        #Deep loops
        self.indices = list(product(range(self.p.num_z), range(self.p.num_n), range(self.p.num_n1), range(self.p.num_v) ,range(self.p.num_q)))
        self.indices_no_v = list(product(range(self.p.num_z), range(self.p.num_n), range(self.p.num_n1),range(self.p.num_q)))

        self.deriv_eps = 1e-4 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid
        self.Q_grid = np.linspace(self.p.q_0,1,self.p.num_q, dtype=dtype) # Create worker productivity grid

        #Size grid:
        self.N_grid=np.linspace(0,self.p.n_bar,self.p.num_n, dtype=dtype)
        self.N_grid1 = np.linspace(0,self.p.n_bar1,self.p.num_n1, dtype=dtype) #Separate grid for seniors, since everyone ends up there. For more tenure levels, keep everyone besides seniors on the same grids, and larger grid for seniors.
        self.N_grid1[0] = 1e-2 #So that it's not exactly zero and I thus can keep my interpretation

        #self.N_grid=np.linspace(0,1,self.p.num_n)
        # Unemployment Benefits across Worker Productivities

        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        self.J_grid = np.zeros(dimensions, dtype=dtype)
        self.grid = np.ogrid[[slice(dim) for dim in self.J_grid.shape]]
        # Production Function in the Model
        self.fun_prod_onedim = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = self.fun_prod_onedim.reshape((self.p.num_z,) + (1,) * (self.J_grid.ndim - 1))

        self.unemp_bf = np.ones(self.p.num_x) * 0.5 * self.fun_prod.min() #Half of the lowest productivity. Kinda similar to Shimer-like estimates who had 0.4 of the average

        # Wage and Shadow Cost Grids
        self.w_grid = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v , dtype=dtype) #Note that this is not the true range of possible wages as this excludes the size part of the story
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)


        #Total firm size for each possible state

        # Calculate the sum size for each element in the matrix
        self.sum_size     = np.zeros_like(self.J_grid) #Sum size
        self.sum_sizeadj  = np.zeros_like(self.J_grid) #Sum size ADJUSTED FOR QUALITY
        self.sum_wage     = np.zeros_like(self.J_grid) #Calculate the total wage paid for every state
        #Start with juniors layers
        self.sum_size[...] = self.N_grid[self.grid[1]]
        self.sum_sizeadj[...] = self.N_grid[self.grid[1]] * (self.p.prod_q +self.p.q_0*(1.0-self.p.prod_q))
        #add medium workers
        if K > 2:
            for k in range(1,K-1): #Ignoring the original junior layer
                self.sum_size += self.N_grid[self.grid[L.ax_ns[k]]]
                self.sum_sizeadj += self.N_grid[self.grid[L.ax_ns[k]]] * (self.p.prod_q + self.Q_grid[self.grid[L.ax_qs[k-1]]] * (1.0 - self.p.prod_q))
                self.sum_wage += self.w_grid[self.grid[L.ax_vs[k-1]]] * self.N_grid1[self.grid[L.ax_ns[k]]]
        #Add seniors
        #for i in range(2, K + 1):
        #    self.sum_size += self.N_grid1[self.grid[i]]
        #    self.sum_sizeadj += self.N_grid1[self.grid[i]] * (self.p.prod_q + self.Q_grid[self.grid[self.J_grid.ndim - (K-1) + (i-2)]] * (1.0 - self.p.prod_q))
        #for i in range(K+1,self.J_grid.ndim - (K-1)):
        #    self.sum_wage += self.w_grid[self.grid[i]]*self.N_grid1[self.grid[i-K+1]] #We add +1 because the wage at the very first step is semi-exogenous, and I will derive it directly
        self.sum_size += self.N_grid1[self.grid[L.ax_n1]] #Add the senior size to the sum size
        self.sum_sizeadj += self.N_grid1[self.grid[L.ax_n1]] * (self.p.prod_q + self.Q_grid[self.grid[L.ax_qs[-1]]] * (1.0 - self.p.prod_q)) 
        self.sum_wage += self.w_grid[self.grid[L.ax_vs[-1]]] * self.N_grid1[self.grid[L.ax_n1]]
        #Setting up production grids
        self.prod = self.production(self.sum_sizeadj) #F = sum (n* (prod_q+q_1*(1-prod_q)))
        #self.prod_diff = self.production_diff(self.sum_sizeadj)
        self.prod_1d = self.fun_prod_1d(self.sum_sizeadj)
        for a in L.ax_qs: #This excludes the junior cohort, for which q_0 is used
            self.prod_nd = self.prod_1d * (self.p.prod_q + self.Q_grid[self.grid[a]] * (1.0-self.p.prod_q)) #\partial F / \partial n_1 = q_1 * (prod_q+q_1*(1-prod_q)) F'(nq)
        #I'm not using this guy anywhere anyway
        #self.prod_qd = self.prod_1d * self.N_grid1[self.grid[L.ax_ns]] * (1.0-self.p.prod_q) #\partial F / \partial q_1 = n_1 * (1-prod_q) * F'(nq) #Andrei Aug'25: wait why??? Why is n still here?? F = (sum n * (prod_q+q_1 * (1-prod_q)))^α. ∂ F / ∂ q = F' * sum_n * (1-prod_q). ah lol k


        #Job value and GE first
        self.v_grid = np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod_onedim.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #Value promised to the worker at the bottom step
        self.v_0 = self.v_grid[0]#-1.0
        
        self.simple_J=np.divide(self.fun_prod_onedim[:,ax] - self.w_grid[ax,:],1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        #This is eqUvalent to marginal value of a firm of size 1 at the lowest step
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        

        if js is None:
            self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
        #    self.js.update(self.v_grid[:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        else:
            self.js = js       


        #Create a guess for the MWF value function
        #self.J_grid1 = self.J_grid1+np.divide(self.fun_prod*production(self.sum_size)-self.w_grid[0]*self.N_grid[ax,:,ax,ax]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #self.J_grid1 = np.zeros_like(self.J_grid)
        #self.J_grid = self.J_grid+np.divide(self.fun_prod*self.prod-self.p.k_f-self.p.beta*self.w_grid[ax,ax,ax,:,ax]*self.N_grid[self.grid[1]]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #The guess above is problematic because it overvalues rho for junior workers. Here, even when there's way more juniors than seniors, rho matters a lot.
        self.J_grid = self.J_grid+np.divide(self.fun_prod*self.prod-self.w_grid[0]*self.N_grid[self.grid[1]]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #Alternatively, here rho is undervalued, as juniors will essentially be forever juniors, being paid nothing
    
        #print("J_grid_diff:", np.max(abs(self.J_grid-self.J_grid1)))
        #The two methods are eqUvalent!! grid[1] really does capture the right value!!!


        #Guess for the Worker value function
        self.W = np.zeros(self.J_grid.shape + (self.K,), dtype=dtype)

        #Creating the wage matrix manually
        self.w_matrix = np.zeros_like(self.W)
        self.w_matrix[...,0] = 0 #The workers at the bottom step will have special wages, derived endogenously through their PK
        #Actually, do I then need to add that step to the worker value? Not really, but useful regardless.
        # Can say that the bottom step really is step zero, with a fixed value owed to the worker.
        # And then all the actually meaningful steps are 1,2... etc, so when K=2 with just have 1 meaningful step            
        for k in range(1,K):
            self.w_matrix[...,k] = self.w_grid[self.grid[L.ax_vs[k-1]]]

        self.W[...,1:] = self.W[...,1:] + self.pref.utility(self.w_matrix[...,1:])/(1-self.p.beta) #skip the first K-1 columns, as they don't correspond to the wage state. Then, pick the correct step, which is hidden in the last dimension of the grid
        self.W[...,0] = self.W[...,0] + self.pref.utility(self.unemp_bf.min())/(1-self.p.beta)

        #Setting up size and quality grids already in the matrix for
        #self.size = np.zeros_like(self.W)
        #self.q = np.zeros_like(self.W)
        #self.size[...,0] = self.N_grid[self.grid[1]]
        #self.q[...,0] = self.p.q_0
        #for i in range(2,K + 1):
        #    self.size[...,i-1] = self.N_grid1[self.grid[i]]
        #for k in range(1,K):
        #    self.q[...,k] +=  self.Q_grid[self.grid[L.ax_qs[k-1]]] #Will this work? Worried about the fact that L.ax_qs is a list.
        del self.w_matrix, 
    def J_sep(self,Jg=None,Wg=None,Ug=None,Rhog=None,P=None,kappa=None,n_g = None, sep_g = None,update_eq=1,s=1.0):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        sum_wage = self.sum_wage
        rho_grid = self.rho_grid
        N_grid = self.N_grid
        N_grid1 = self.N_grid1
        Q_grid = self.Q_grid
        grid = self.grid
        #size = self.size
        #q = self.q
        indices = self.indices
        indices_no_v = self.indices_no_v
        layout = self.layout
        if Jg is None:
            J = np.copy(self.J_grid)
        else:
            J = np.copy(Jg)
        if Wg is None:
            W = np.copy(self.W)
        else:
            W = impose_increasing_W(Wg)
        if Ug is None:
            U = self.pref.utility(self.unemp_bf) / (1 - self.p.beta)
        else:
            U = np.copy(Ug)
        if Rhog is None:
            Rho = J
            for k in range(1,self.K):
                if k < self.K - 1:
                    Rho += self.N_grid[self.grid[layout.ax_ns[k]]] * self.rho_grid[self.grid[layout.ax_vs[k-1]]] * W[...,k]
                else:
                    Rho += self.N_grid1[self.grid[layout.ax_n1]] * self.rho_grid[self.grid[layout.ax_vs[k-1]]] * W[...,k]
        else:
            Rho = np.copy(Rhog) 
        if n_g is None:
            n_star = np.zeros_like(W)   
        else:
            n0_star = n_g
        if sep_g is None:
            sep_star = np.zeros_like(W)
        else:
            sep_star = sep_g
        # create representation for J1p
        Jp = np.zeros_like(J)
        Wp = np.zeros_like(J)
        Rhop = np.zeros_like(J)
        # Updating J1 representation
        #for iz, in0, in1, iq in indices_no_v:
        #    W_inc = W[iz,in0,in1,:,iq,1]
        #    Jp[iz,in0,in1,:,iq] = splev(W_inc, splrep(W[iz,in0,in1,:,iq,1],J[iz,in0,in1,:,iq],s=s))
        #    Wp[iz,in0,in1,:,iq] = splev(rho_grid, splrep(rho_grid,W[iz,in0,in1,:,iq,1],s=s))
        #   Rhop[iz,in0,in1,:,iq] = splev(rho_grid, splrep(rho_grid,Rho[iz,in0,in1,:,iq],s=s))
       
        #This is slow af, BUT IT WORKS
        smooth_all_v_axes_inplace(
        J=J, W=W, Rho=Rho,
        rho_grid=rho_grid,
        v_axes=layout.ax_vs,           # smooth along ALL v-axes
        step_indices=None,             # defaults to [1, 2, ..., K-1]
        s=s,                        # your smoothing strength
        k=3,                    # spline order
        )
        Jp = Rhop 
        for k in range(1,self.K):
            if k < self.K - 1:
                Jp += - self.N_grid[self.grid[layout.ax_ns[k]]] * self.rho_grid[self.grid[layout.ax_vs[k-1]]] * Wp[...,k]
            else:
                Jp += - self.N_grid1[self.grid[layout.ax_n1]] * self.rho_grid[self.grid[layout.ax_vs[k-1]]] * Wp[...,k]        
        #- size[...,1] * rho_grid[ax,ax,ax,:,ax] * Wp
        print("J shape", J.shape)
        print("W shape", W.shape)        


        #These are SOOOOO MANY THOUGH. I don't think I have ram space for all this...
        EW_star = np.copy(W) #These are all W shape now in order to be useable for every cohort. The zero index will usually be empty tho
        #EJ_star = np.copy(J)
        ERho_star = np.copy(J)
        #EJderiv = np.zeros_like(J)
        Jderiv = np.zeros_like(W)
        rho_star = np.zeros_like(W) #except the zero dimension here. That one will be empty
        #sep_star1 = np.zeros_like(J) #probably better to just make it multidimensional
        #n1_star = np.zeros_like(J)   
        q_star  = np.zeros_like(W)   

        Rhoderiv = np.zeros_like(J)
        Rhod0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n1, self.p.num_v, self.p.num_q, self.p.num_n))
        #Rhod1 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n1, self.p.num_v, self.p.num_q, self.p.num_n1)) 
        Rhod0_diff = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n1, self.p.num_v, self.p.num_q, self.p.num_n-1))  
        Ihire = np.zeros_like(J,dtype=bool)     
        #Wd0 = np.zeros_like(Rhod0)

        #Separations related variables
        sep_grid = np.linspace(0,1,20)
        n1_s = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n1, self.p.num_v, self.p.num_q, sep_grid.shape[0]))
        q_s = np.zeros_like(n1_s)
        foc_sep = np.zeros_like(n1_s)
        J_s = np.zeros_like(n1_s)
        J_s_deriv = np.zeros_like(J_s)
        #sep_reshaped = np.zeros_like(J_s) + sep_grid[ax,ax,ax,ax,ax,:]
        #sep_grid_exp = sep_grid[ax,ax,ax,ax,ax,:]

        # prepare expectation call
        Ez = oe.contract_expression('anmvq,az->znmvq', J.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', U.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW_star)

        ite_num = 0
        error_js = 1
        

        #First U + GE solve
        critU = 1
        while critU > 1e-3:
            # General equilibrium first time
            self.v_0 = U
            self.v_grid = np.linspace(U.min(),W[self.p.z_0-1, 0, 1, :, 0, 1].max(),self.p.num_v)
            if P is None:
                kappa, P = self.GE(Ez(Jp, self.Z_trans_mat),Ez(W[...,1], self.Z_trans_mat)[self.p.z_0-1,0,1,:,0])
            self.js.update(self.v_grid,P)
            U2 = U
            _, ru, _ = self.getWorkerDecisions(EU, employed=False)
            U = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EU)
            U = 0.2 * U + 0.8 * U2
            critU = np.abs(U-U2)
        critU = 1
        print("kappa", kappa)
        print("P", P)
        for ite_num in range(self.p.max_iter):
            J2 = J
            W2 = np.copy(W)
            U2 = U
            Rho2 = np.copy(Rho)
            rho_star2 = np.copy(rho_star)
            n_star2 =  np.copy(n_star)            
            q_star2 =   np.copy(q_star)                       
            # we compute the expected value next period by applying the transition rules
            EW = Ez(Wp, self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJ = Ez(Jp, self.Z_trans_mat)
            ERho = Ez(Rhop, self.Z_trans_mat)
            EU = U

            # get worker decisions
            _, re, pc = self.getWorkerDecisions(EW)
            # get worker decisions at EW + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW + self.deriv_eps) 
        
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff = np.zeros_like(pc)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value

            
        
            Rhoderiv[:, :, 0, ...] = (Rho[:, :, 1,  ...] - Rho[:, :, 0, ...]) / (N_grid1[1] - N_grid1[0])
            Rhoderiv[:, :, -1, ...] = Rho[:, :, -1,  ...] - Rho[:, :, -2,  ...]/ (N_grid1[-1] - N_grid1[-2])
            Rhoderiv[:, :, 1:-1, ...] = (Rho[:, :, 2:,  ...] - Rho[:, :, :-2, ...]) / (N_grid1[ax, ax, 2:, ax, ax] - N_grid1[ax, ax, :-2, ax, ax])  
            Jderiv = Rhoderiv-rho_grid[ax,ax,ax,:,ax]*W[...,1]
            EJinv=(Jderiv+self.w_grid[ax,ax,ax,:, ax]-self.fun_prod*self.prod_nd)/self.p.beta #creating expected job value as a function of today's value            
            #Hey... is this EJinv correct? Doesn't it require also sep_star1?

            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modified with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1

            #FOC for future Rho
            inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[...,ax,:,:]*EU+(1-sep_star[...,ax,:,:])*(EW[..., ax, :]+re[..., ax, :])))
            foc_2ndpart = - size[..., ax, :, 1] * (1-sep_star1[...,ax,:,:])*rho_grid[ax, ax, ax, ax, :, ax] -\
                 size[..., ax, :,0] * (1-sep_star[...,ax,:,:]) / inv_utility_1d                         
            foc = rho_grid[ax, ax, ax, :, ax, ax] - (EJinv[:, :, :, ax, :, :] / (pc[...,ax,:] * (1 - sep_star1[...,ax,:])))* (log_diff[...,ax,:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
            foc = foc*(size[..., ax, :,0] * (1-sep_star[...,ax,:,:])+size[..., ax, :,1] * (1-sep_star1[...,ax,:])) + foc_2ndpart
            foc = impose_increasing_foc(foc)
            assert (np.isnan(foc) & (pc[..., ax, :] > 0)).sum() == 0, "foc has NaN values where p>0"

            #beg_cpu=time()
            #rho_star = solve_rho(
            #    rho_grid, N_grid, N_grid1, foc, rho_star, self.p.num_z, self.p.num_n, self.p.num_n1, self.p.n_bar1, self.p.num_v, self.p.num_q) 
            #end_cpu=time()
            #rho_star = impose_increasing_policy(rho_star)
            #print("Parallel time", end_cpu-beg_cpu)
            #FOC for Separations
            tenure = ite_num % 2
            if ite_num>=20:
                #WHAT IF. We just do a direct derivative wrt s??? Like, we know what q_s and n1_s are. Inteprolate directly onto them, which will already give us the total derivative of J wrt s, no?
                if tenure == 0:  
                    sen_el = np.argmin(np.abs(sep_grid - sep_star1[..., None]), axis=-1)
                    sen_sep = sep_grid[sen_el]
                    n1_s = np.maximum((size[...,0,ax]*(1-sep_grid_exp)+size[...,1,ax] * (1-sen_sep[...,ax])) * pc_star[...,ax],N_grid1[0])
                    n1_s = np.minimum(n1_s,N_grid1[-1])
                    q_s = np.fmin((size[...,0,ax] * np.minimum(self.p.q_0,1-sep_grid_exp)+size[...,1,ax] * np.minimum(q[...,ax],1-sen_sep[...,ax])) / (size[...,0,ax]*(1-sep_grid_exp)+size[...,1,ax]*(1-sen_sep[...,ax])),1)               
                    sep_star0 = np.copy(sep_star)
                    sep_star[...] = 0
                else:
                    jun_el = np.argmin(np.abs(sep_grid - sep_star[..., None]), axis=-1)
                    jun_sep = sep_grid[jun_el]
                    n1_s = np.maximum((size[...,0,ax]*(1-jun_sep[...,ax])+size[...,1,ax] * (1-sep_grid_exp)) * pc_star[...,ax],N_grid1[0])
                    q_s = np.fmin((size[...,0,ax] * np.minimum(self.p.q_0,1-jun_sep[...,ax])+size[...,1,ax] * np.minimum(q[...,ax],1-sep_grid_exp)) / (size[...,0,ax]*(1-jun_sep[...,ax])+size[...,1,ax]*(1-sep_grid_exp)),1)               
                    sep_star0 = np.copy(sep_star1)
                    sep_star1[...] = 0
                  
                ERho_s = ERho - size[...,1] * rho_grid[ax,ax,ax,:,ax] * EW
                points = []
                n0_sz = np.repeat(n0_star2[...,ax], sep_grid.shape[0], axis=-1)
                rho_sz = np.repeat(rho_star2[...,ax], sep_grid.shape[0], axis=-1)
                for iz in range(self.p.num_z):
                    points.append( tuple_into_2darray((n0_sz[iz, ...], n1_s[iz,...], rho_sz[iz, ...], q_s[iz,...])))
                J_s = Js_int(J_s,ERho_s,N_grid,N_grid1,rho_grid,Q_grid,points,self.p.num_z)

                J_s_deriv[...,0] = (J_s[...,1] - J_s[...,0]) / (sep_grid[1] - sep_grid[0])
                J_s_deriv[...,-1] = (J_s[...,-1] - J_s[...,-2]) / (sep_grid[-1] - sep_grid[-2]) 
                J_s_deriv[..., 1:-1]    = (J_s[...,2:] - J_s[...,:-2]) / (sep_reshaped[...,2:] - sep_reshaped[...,:-2]) 

                if tenure == 0:
                    inv_util_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_reshaped * EU[...,ax] + (1-sep_reshaped) * (EW_star[...,ax] + re_star[...,ax])))
                    foc_sep = J_s_deriv - size[...,ax,0] * (re_star[...,ax]+EW_star[...,ax] - EU) / inv_util_1d
                else:
                    foc_sep = J_s_deriv - size[...,ax,1] * (re_star[...,ax]+EW_star[...,ax] - EU) * rho_grid[ax,ax,ax,:,ax,ax]
                foc_sep = impose_increasing_fsep(-foc_sep)

            #FOC for hiring
            points = []
            for iz in range(self.p.num_z):
                points.append( tuple_into_2darray((n1_star2[iz, ...], rho_star2[iz, ...], q_star2[iz, ...])))
            Rhod0 = Rhod0_int(Rhod0,ERho,N_grid1,rho_grid,Q_grid,points,self.p.num_z,self.p.num_n)
            n0_star[...] = 0
            Ihire = ((Rhod0[...,1] - Rhod0[...,0]) / (N_grid[1] - N_grid[0]) > kappa/self.p.beta) & (size[...,0] + size[...,1] < self.p.n_bar)
            Rhod0_diff = impose_increasing_fsep( - (Rhod0[..., 1:] - Rhod0[..., :-1]) / (N_grid[1:]-N_grid[:-1])) 
                #n0_star = n0(Rhod0_diff, n0_star, N_grid, Ihire, kappa / self.p.beta)
                #n0_star = impose_decreasing_policy(n0_star)

            rho_star,n0_star,sep = solve_policies(ite_num,rho_star,n0_star,sep_star,foc_sep,sep_grid, kappa / self.p.beta,Ihire,Rhod0_diff,rho_grid, N_grid, N_grid1, foc, self.p.num_z, self.p.num_n, self.p.num_n1, self.p.n_bar1, self.p.num_v, self.p.num_q,tenure)
            rho_star = impose_increasing_policy(rho_star)
            n0_star = impose_decreasing_policy(n0_star)
            if ite_num >= 20:
                if tenure==0:
                    sep_star = 0.4 * impose_increasing_policy(sep) + 0.6 * sep_star0
                else:
                    sep_star1 = 0.4 * impose_decreasing_policy(sep) + 0.6 * sep_star0
                    sep_star1 = np.minimum(sep_star1, 1-1e-2) #Cannot fire more than 99% of seniors so that we do not reach literal zero
            print("n0_star", n0_star.min(),n0_star.max())    
            print("sep jun", sep_star.min(),sep_star.max())
            print("sep sen", sep_star1.min(),sep_star1.max())

            #Getting n1_star and q_star
            pc_trans = np.moveaxis(pc,3,0)
            rho_trans = np.moveaxis(rho_star,3,0)            
            pc_temp = np.moveaxis(interp_multidim_extra_dim(rho_trans,rho_grid,pc_trans),0,3)
            n1_star = np.maximum((size[...,0]*(1-sep_star)+size[...,1]*(1-sep_star1))*pc_temp,N_grid1[0])
            n1_star = np.minimum(n1_star,N_grid1[-1])
            q_star = np.fmin((size[...,0]* np.minimum(self.p.q_0,1-sep_star)+size[...,1]*np.minimum(q,1-sep_star1))/(size[...,0]*(1-sep_star)+size[...,1]*(1-sep_star1)),1)
            print("q_star", q_star[self.p.z_0-2,self.p.num_n-1,0,50, :])

            #Future optimal expectations
            #ERho_star = interp_multidim(n0_star,N_grid,np.moveaxis(Rhod0,-1,0))
            points = []
            for iz in range(self.p.num_z):
                points.append( tuple_into_2darray((n0_star[iz,...],n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...])))
            #ERho_star = Js_int(ERho_star,ERho,N_grid,N_grid1,rho_grid,Q_grid,points,self.p.num_z) #Do it like this so as to use all the new policies
            #EW_star = Js_int(EW_star,EW,N_grid,N_grid1,rho_grid,Q_grid,points,self.p.num_z)
            ERho_star, EW_star =  Values_int(ERho_star,EW_star,ERho,EW,N_grid,N_grid1,rho_grid,Q_grid,points,self.p.num_z)



            _, ru, _ = self.getWorkerDecisions(EU, employed=False)
            U = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EU)
            U = 0.2 * U + 0.8 * U2

            _, re_star, pc_star = self.getWorkerDecisions(EW_star)
            # Update firm value function 
            wage_jun = self.pref.inv_utility(self.v_0-self.p.beta*(sep_star*EU+(1-sep_star)*(EW_star+re_star)))
            J= self.fun_prod*self.prod - self.p.k_f - sum_wage - kappa*n0_star - \
                wage_jun*size[...,0]  + self.p.beta *  (ERho_star - rho_star*n1_star*EW_star) #Note it's not EJ_star here anymore. To make the function faster
            #J = impose_decreasing(J)
            assert np.isnan(J).sum() == 0, "J has NaN values"

            Rho = self.fun_prod*self.prod - self.p.k_f - sum_wage - kappa*n0_star - \
                wage_jun*size[...,0] + \
                rho_grid[ax,ax,ax,:,ax]*size[...,1]*W[...,1] + self.p.beta * (ERho_star - rho_star*n1_star*EW_star)
            #Rho_alt = J + size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]                    
            assert np.isnan(Rho).sum() == 0, "Rho has NaN values"  
            # Update worker value function
            W[...,1] = self.pref.utility(self.w_matrix[...,1]) + \
                self.p.beta * (sep_star1 * EU + (1 - sep_star1) * (EW_star + re_star)) #For more steps the ax at the end won't be needed as EW_star itself will have multiple steps
            W = impose_increasing_W(W)
            assert np.isnan(W).sum() == 0, "W has NaN values"

            #W[...,1] = W[...,1] * (J > 0) + U * (J <= 0)
            #Rho= Rho * (J > 0) + 0 * (J <= 0)
            #J[J <= 0] = 0
            #comparison_range = (size[...,0]+size[...,1] <= self.p.n_bar) & (size[...,0]+size[...,1] >= N_grid[1])
            #print("Diff Rho:", np.mean(np.abs((Rho_alt[comparison_range]-Rho[comparison_range])/Rho[comparison_range])))
            
            Rho = .2 * Rho + .8 * Rho2
            J= .2 * J + .8 * J2
            W[...,1:] = .2 * W[...,1:] + .8 * W2[...,1:] #we're completely ignoring the 0th step

            # Updating J1 representation
            st= time()
            for iz, in0, in1, iq in indices_no_v:
                Wp[iz,in0,in1,:,iq] = splev(rho_grid, splrep(rho_grid,W[iz,in0,in1,:,iq,1],s=s))
                Rhop[iz,in0,in1,:,iq] = splev(rho_grid, splrep(rho_grid,Rho[iz,in0,in1,:,iq],s=s))
            end=time()
            if (ite_num % 100 == 0):
                print("Time to fit the spline", end - st)
            
            #TRYING AGAIN WITH THE BASIC RHO
            Jp = Rhop - size[...,1]*rho_grid[ax,ax,ax,:,ax]*Wp      

            # Compute convergence criteria
            error_j1i = array_exp_dist(Rho,Rho2,100) #np.power(J - J2, 2).mean() / np.power(J2, 2).mean()  
            error_w1 = array_dist(W[...,1:], W2[...,1:])

            # update worker search decisions
            if (((ite_num % 100) == 0) & (ite_num>10)):
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    print("Errors:",  error_j1i, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model and error_js < self.p.tol_search
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            self.v_0 = U
                            print("U",U)
                            EJ = Ez(Jp, self.Z_trans_mat)
                            #self.v_grid = np.linspace(W[self.p.z_0-1, 0, 1, :, :,1].min(),W[self.p.z_0-1, 0, 1, :, :,1].max(),self.p.num_v)
                            self.v_grid = np.linspace(U.min(),W[self.p.z_0-1, 0, 1, :, :, 1].max(),self.p.num_v)
                            kappa, P = self.GE(EJ,W,kappa,J,n0_star)
                            #P_xv = self.matching_function(J1p.eval_at_W1(W)[self.p.z_0-1, 0, 1, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(self.v_grid, P, type=1, relax=relax)
                            #error_js = self.js.update(self.v_grid, P)
                            #self.js.update(self.v_grid,P)

                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[...,1], Ji)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    print("Errors:",  error_j1i,  error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
            #Comparing Ejinv to the future deriv
            #if (ite_num % 100) == 0:
            # #Getting the derivative of the future job value wrt n1:
            # floorn1=np.floor(np.interp( n1_star, N_grid1, range(self.p.num_n1))).astype(int)
            # ceiln1=np.ceil(np.interp( n1_star, N_grid1, range(self.p.num_n1))).astype(int)            
            # for iz in range(self.p.num_z):
            #    for in11 in range(self.p.num_n1): 
                    
            #        Rho_interpolator = RegularGridInterpolator((N_grid, rho_grid, Q_grid), ERho[iz, :, in11, ...], bounds_error=False, #fill_value=None)
                    #W_interpolator = RegularGridInterpolator((N_grid, rho_grid, Q_grid), EW[iz, :, in11, ...], bounds_error=False, fill_value=None)
            #        Rhod1[iz, ..., in11] = Rho_interpolator((n0_star[iz, ...], rho_star[iz,...], q_star[iz, ...]))
                    #Wd0[iz, ..., in11] = W_interpolator((n0_star[iz, ...], rho_star[iz,...], q_star[iz, ...]))
            # ERhoderiv = ERhoDerivative(Rhod1,Wd0,ceiln1,floorn1,n1_star,rho_star,N_grid1,self.p.num_z,self.p.num_n,self.p.n_bar,self.p.num_v,self.p.num_q)
            # EJderiv = ERhoderiv - rho_star * EW_star
            # print("EJinv", EJinv[self.p.z_0-1,1,2,50, 0]/pc_star[self.p.z_0-1,1,2,50, 0])
            # print("EJderiv", EJderiv[self.p.z_0-1,1,2,50, 0])
            # j = np.where(N_grid==1)
            # s = np.where(N_grid1==2)
            # print("EJinv diff 1j 2s:", np.mean(np.abs((EJinv[:,j,s,:, 0]/pc_star[:,j,s,:, 0] - EJderiv[:,j,s,:, 0]) / EJderiv[:,j,s,:, 0])))
            # print("EJinv diff 1 sen:", np.mean(np.abs((EJinv[:,0,1,:, 0]/pc_star[:,0,1,:, 0] - EJderiv[:,0,1,:, 0]) / EJderiv[:,0,1,:, 0])))
            # print("EJinv diff 2 sen:", np.mean(np.abs((EJinv[:,0,s,:, 0]/pc_star[:,0,s,:, 0] - EJderiv[:,0,s,:, 0]) / EJderiv[:,0,s,:, 0])))

            if (((ite_num % 200)  == 0) & (ite_num>10)):   
                plt.plot(W[self.p.z_0-2, 0, 1, :, 0 ,1], J[self.p.z_0-2, 0, 1, :, 0], label='1 senior value function')
                plt.plot(W[self.p.z_0-2, 0, 1, :, 0 ,1], Jp[self.p.z_0-2, 0, 1, :, 0], label='1 senior value function') 
                #plt.show() # this will load image to console before executing next line of code
                #plt.plot(W[self.p.z_0-1, 0, 1, :, 0, 1], 1-pc_star[self.p.z_0-1, 0, 1, :, 0], label='Probability of the worker leaving across submarkets')      
                plt.show()

        # --------- wrapping up the model ---------

        # find rho_j2j
        rho_j2j = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n1, self.p.num_v, self.p.num_q))
        ve_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n1, self.p.num_v, self.p.num_q))
        #for ix in range(self.p.num_x):
        for iz, in0, in1, iq in indices_no_v:
        #for iz in range(self.p.num_z):
            ve_star[iz, in0, in1, :, iq] = self.js.ve( EW_star[iz, in0, in1, :, iq])
            rho_j2j[iz, in0, in1, :, iq] = np.interp(ve_star[iz, in0, in1, :, iq], W[iz, in0, in1, :, iq, 1], rho_grid)

        # find rho_u2e
        self.ve = self.js.ve(EU)
        Pr_u2e = self.js.pe(EU) # this does not include the inefficiency of search for employed

        # value functions
        self.Vf_J = J
        self.Vf_W = W
        self.Vf_U = U
        self.Vf_Rho = Rho
        self.Jp = Jp
        self.EW_star  = EW_star

        # policies
        self.rho_j2j = rho_j2j
        #self.rho_u2e = rho_u2e
        self.w_jun = wage_jun
        self.rho_star = rho_star
        self.sep_star = sep_star
        self.sep_star1 = sep_star1
        self.n0_star = n0_star
        self.n1_star = n1_star
        self.q_star = q_star
        self.pe_star = 1-pc_star
        self.ve_star = ve_star
        self.Pr_u2e = Pr_u2e

        self.error_w1 = error_w1
        self.error_j = error_j1i
        #self.error_j1p = error_j1g
        self.error_js = error_js
        self.niter = ite_num

        #GE values
        self.P = P
        self.kappa = kappa

        self.append_results_to_pickle(J, W, U, Rho, P, kappa, EW_star, sep_star, n0_star, n1_star)

        #Saving the entire model
        self.save("model_GE.pkl")
        return self



    def save(self,filename):
        # Load the existing data from the pickle file
        try:
            with open(filename, "rb") as file:
                all_results = pickle.load(file)
        except FileNotFoundError:
            all_results = {}
            print("No existing file found. Creating a new one.")
        # Use a tuple as the key
        key = (self.p.num_z,self.p.num_v,self.p.num_n,self.p.n_bar,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.hire_c,self.p.prod_alpha,self.p.dt,self.p.u_bf_m)
        
        all_results[key] = self
        #Save the updated dictionary back to the pickle file        
        with open(filename, "wb") as output_file:
            pickle.dump(all_results, output_file)
        print(f"Results for p = {key} have been appended to {filename}.")
    def GE(self,EJ,W,kappa_old=None,J=None,n0_star=None):
        #Find kappa, which is the hiring cost firms have to pay per worker unit
        #BIG NOTE: For now I'm assuming that all the firms start at the same productivity level, p.z_0-1, rather than the Schaal assumption of them drawing their productivity upon entering.
        #Quick method: Envelope Theorem
        if (kappa_old is not None) and (n0_star[self.p.z_0-1,0,0,0,0] > 0) and (n0_star[self.p.z_0-1,0,0,0,0] <= 0):
            print("Fast kappa method")
            kappa = np.divide( -self.p.k_entry + J[self.p.z_0-1,0,0,0,0] + n0_star[self.p.z_0-1,0,0,0,0] * kappa_old, n0_star[self.p.z_0-1,0,0,0,0] )
        else:
        #Slow precise method: Directly compute all this
            print("Slow kappa method")
            kappa_grid = np.linspace(0,50,50)
            n0_k = np.zeros((50))
            J_diff = (EJ[self.p.z_0-1,1:,0,0,0] - EJ[self.p.z_0-1,:-1,0,0,0] ) / (self.N_grid[1:] - self.N_grid[:-1])
            n0_k[:] = np.interp(-kappa_grid / self.p.beta,impose_increasing(-J_diff),self.N_grid[1:])
            n0_k[(EJ[self.p.z_0-1,1,0,0,0] - EJ[self.p.z_0-1,0,0,0,0]) / (self.N_grid[1] - self.N_grid[0]) - kappa_grid / self.p.beta <= 0] = 0
            entry = -n0_k * kappa_grid -self.p.k_f + self.p.beta * np.interp(n0_k,self.N_grid,EJ[self.p.z_0-1,:,0,0,0])
            entry = - kappa_grid - self.p.k_f + self.p.beta *EJ[self.p.z_0-1,1,0,0,0] #Simplifying assumption: entering firms always just hire 1 guy
            kappa = np.interp(-self.p.k_entry,impose_increasing(-entry),kappa_grid)

        #Smoothing the kappa
        if kappa_old is not None:
            smooth = 0.1
            kappa = smooth * kappa + (1-smooth) * kappa_old
        

        print("kappa", kappa)
        #if kappa_old is not None:
        #    print("difference between the two methods")
        #Find the sign-on bonus
        # Sign-on formula: u(w*(1-beta))/(1-beta)+beta*v_0=v_m. This is basically allowing the worker to split their bonus forever
        # The issue with this is that, theoretically, this should then impact all worker future utility...
        # Man I didn't expect this to be such an annoying bottleneck
        # For now, try this still: u(w*(1-beta)) = (v_m - beta*v_0) * (1-beta), so w=u^{-1}[(v_m - beta*v_0) * (1-beta)] / (1-beta)
        # Okay, this didn't work lmao
        # Instead, augment v_0 somehow??? util( inv_util[v_0 * (1-beta)] + signon * (1-beta) ) / (1-beta) = v_m
        #assert np.all((EW - self.p.beta * self.v_0) * (1-self.p.u_rho) + 1 >= 0)
        #signon_bonus = self.pref.inv_utility(self.v_grid - self.p.beta * self.v_0)
        signon_bonus = self.pref.inv_utility(self.v_grid - self.v_0) - self.pref.inv_utility(self.v_grid[0] - self.v_0) #-1 to compensate the initial difference
        #Andrei: WTF is this sign_on bonus?? I can't even back out the initial formula here. The 2nd part is super confusing
        #u(w_bon)+v_0=v_m makes sense. the 2nd part doesnt. I guess it's some kind of correction?? In case v_grid[0]<v_0? But I don't see how it could be theoretically founded...
        #u(w_bon+u^-1(v_grid[0]-v_0))+v_0=v_m is the original formula here #Is the idea here to normalize w_bon to 0? I think so, right???
        #So this guy isn't tooooooooooo bad... but it still blows up super quickly
        #An alternative could be that, upon being signed, they get the bonus wage in addition to the unemp production
        # u(w_bon+unemp.bf)+beta*v_0=v_m, so w_bon = u^{-1}(v_m-beta*v_0)-unemp.bf
        #signon_bonus = self.pref.inv_utility(self.v_grid - self.p.beta * self.v_0)-self.unemp_bf
        #By the way, is the timing... working out? Because like this I'm very explicitly stating that, upon hiring, workers have 1 period of being useless.
        #Moreover, that period is not part of the unemployment period, like this I would genuinely have a worker be useless for 1 period.


        #signon_bonus = self.pref.inv_utility((self.v_grid - self.p.beta * self.v_0) * (1 - self.p.beta)) / (1 - self.p.beta) #This is the bonus firms have to pay right upon hiring
        #signon_bonus = (self.pref.inv_utility(self.v_grid * (1 - self.p.beta)) - self.pref.inv_utility(self.v_0 * (1 - self.p.beta))) / (1 - self.p.beta)
        signon_bonus[signon_bonus < 0] = 0
        #Another signon bonus alternative: just linear!!!
        #signon_bonus = self.v_grid - self.p.beta * self.v_0
        #print("signon", signon_bonus)

        # Given kappa, find the tightness
        q=np.minimum(self.p.hire_c/(kappa-signon_bonus),1)
        #print("q",q)
        theta = self.pref.q_inv(q)
        theta[signon_bonus>kappa-self.p.hire_c]=0 #Hiring cost should be lower now, since chance of hiring<1 AND firms have to pay the sign-on

        #This is eqUvalent to kappa < sign-on + hire_c/1, suggesting that we need q>1. OR, if sign-on>kappa, that we need q<0.
        #Get the job-finding probability for each submarket
        P = q * theta
        print("P",P)
        #plt.plot(self.v_grid, P, label='Probability of finding a job across submarkets')       
        #plt.show() # this will load image to console before executing next line of code
 
        return kappa, P
    def append_results_to_pickle(self, J, W, U, Rho, P, kappa, EW_star, sep_star, n0_star, n1_star, pickle_file="results_GE.pkl"):
        # Step 1: Load the existing data from the pickle file
        try:
            with open(pickle_file, "rb") as file:
                all_results = pickle.load(file)
        except FileNotFoundError:
            all_results = {}
            print("No existing file found. Creating a new one.")

        # Step 2: Create results for the multi-dimensional p
        new_results = self.save_results_for_p(J, W, U, Rho, P, kappa, EW_star, sep_star, n0_star, n1_star)

        # Step 3: Use a tuple (p.num_z, p.num_v, p.num_n) as the key
        key = (self.p.num_z,self.p.num_v,self.p.num_n,self.p.n_bar,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.hire_c,self.p.prod_alpha,self.p.dt,self.p.u_bf_m)

        # Step 4: Add the new results to the dictionary
        all_results[key] = new_results

        # Step 5: Save the updated dictionary back to the pickle file
        with open(pickle_file, "wb") as file:
            pickle.dump(all_results, file)

        print(f"Results for p = {key} have been appended to {pickle_file}.")
    def save_results_for_p(self, J, W, U, Rho, P, kappa, EW_star, sep_star, n0_star, n1_star):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
        'date': current_date,
        'J': J,
        'W': W,
        'U': U,
        'Rho': Rho,
        'P': P,
        'kappa': kappa,
        'EW_star': EW_star,
        'sep_star': sep_star,
        'n0_star': n0_star,
        'n1_star': n1_star,
        'p_value': (self.p.num_z,self.p.num_v,self.p.num_n,self.p.n_bar,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.hire_c,self.p.prod_alpha,self.p.dt,self.p.u_bf_m)
    }    
    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)
    def production(self,sum_n):
        return np.power(sum_n, self.p.prod_alpha)
    def production_diff(self,sum):

        diff = (self.production(np.minimum(sum+1,self.K*(self.p.num_n-1))) - self.production(sum) + self.production(sum) - self.production(np.maximum(sum-1,0))) / (np.minimum(sum+1,self.K*(self.p.num_n-1)) - np.maximum(sum-1,0))
        
        return diff
    def fun_prod_1d(self,sum_n):
        return  self.p.prod_alpha*np.power(sum_n,self.p.prod_alpha-1)*(sum_n>0)
    def getWorkerDecisions(self, EW1, employed=True): #Andrei: Solves for the entire matrices of EW1 and EU
        """
        :param EW1: Expected value of employment
        :param EU:  Expected value of unemployment
        :param employed: whether the worker is employed (in which case we multiply by efficiency)
        :return: pe,re,qi search decision and associated return, as well as qUt decision.
        """
        pe, re = self.js.solve_search_choice(EW1) #Uses the job search array to solve for the search choice
        assert (~np.isnan(pe)).all(), "pe is not NaN"
        assert (pe <= 1).all(), "pe is not less than 1"
        assert (pe >= -1e-10).all(), "pe is not larger than 0"

        if employed:
            pe = pe * self.p.s_job
            re = re * self.p.s_job
        #print("Shape of pe:", pe.shape)
        # construct the continuation probability. #Andrei: probability the worker doesn't get fired and also doesn't leave
        pc = (1 - pe)

        return pe, re, pc
    def matching_function(self,J1): 
        return self.p.alpha * np.power(1 - np.power( 
            np.divide(self.p.kappa, np.maximum(J1, self.p.kappa)), self.p.sigma),
                                1 / self.p.sigma) #Andrei: the formula of their matching function, applied to each particula job value J1       
        
        
         
#from primitives import Parameters
#p = Parameters()

#mwc_GE_J = objects['mwc_Rho_J']
#mwc_GE_W = objects['mwc_Rho_W']
#mwc_GE=MultiworkerContract(K=3,input_param=p)
#model=mwc_GE.J_sep(update_eq=1,s=20.0)