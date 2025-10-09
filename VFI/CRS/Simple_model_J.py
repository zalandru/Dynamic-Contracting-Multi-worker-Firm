import numpy as np
import logging
from scipy.stats import lognorm as lnorm

import opt_einsum as oe

#For printing
import matplotlib.pyplot as plt
import subprocess
import shlex
import os
from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search_GE import JobSearchArray
from valuefunction_multi import PowerFunctionGrid
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import splrep
from scipy.interpolate import splev
from itertools import product #To clean up the code: use nested loops but without actual ugly nesting

import numba as nb
from numba import njit, cuda, float64, prange
import numpy as np


#from numba import cuda, float64, prange

import pickle
import datetime
from time import time
import math

ax = np.newaxis

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
    for v in range(1,A.shape[1]):
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
    for v in range(1,A.shape[1]):
        A[...,v,:] = np.maximum(A[...,v,:],A[...,v-1,:]+1e-12*np.maximum(1.0, np.abs(A[..., v-1, :])))
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


@nb.njit(cache=True,parallel=False)
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
@nb.njit(cache=True,parallel=False)
def n0(Rhod0_diff, n0_star, N_grid, Ihire, hire_c):
    for idx in np.argwhere(Ihire):
        n0_star[idx[0], idx[1], idx[2], idx[3], idx[4]] = interp( -hire_c ,Rhod0_diff[idx[0], idx[1], idx[2], idx[3], idx[4],:],N_grid[1:]) #oh shit, should we also account for how that affects the worker value???
    print("n0_star borders", n0_star.min(), n0_star.max())   
    return n0_star 
@nb.njit(cache=True,parallel=False)
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
@nb.njit(cache=True,parallel=False)
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
@nb.njit(cache=True,parallel=False)
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
@nb.njit(cache=True,parallel=False)
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
@nb.njit(cache=True, parallel=False)
def solve_policies(ite_num, layoff_iter, sep_star, foc_sep, mask, sep_grid,
                    num_z, num_v, num_q):
    # Precompute constants that do not change within the loops
    ite_ge_20 = (ite_num >= layoff_iter)
    #ite_ge_10 = (ite_num >= 10)
    #tenure_nonzero = (tenure != 0)


    for iz in prange(num_z):
        for iv in prange(num_v):
            for iq in prange(num_q):
                        if ite_ge_20:
                            sep_star[iz, iv, iq] = interp(0,
                                                                    foc_sep[iz, iv, iq, :],
                                                                    sep_grid)
    return sep_star

@nb.njit(cache=True,parallel=False)
def Rhod0_int(Rhod0,ERho,N_grid1,rho_grid,Q_grid,points,num_z,num_n):
    shape = Rhod0[0,...,0].shape
    for iz in prange(num_z):
     for in00 in prange(num_n):
        flat_result = multilinear_interp(points[iz], (( N_grid1, rho_grid, Q_grid)), ERho[iz, in00, ...])
        Rhod0[iz,...,in00] =  flat_result.reshape(shape)
    return Rhod0
@nb.njit(cache=True,parallel=False)
def Js_int(Js,ERho_s,rho_grid,Q_grid,points,num_z):
    shape = Js[0,...].shape
    for iz in prange(num_z):
        flat_result = multilinear_interp(points[iz], ((rho_grid, Q_grid)), ERho_s[iz, ...])
        Js[iz,...] =  flat_result.reshape(shape)
    return Js
@nb.njit(cache=True,parallel=False)
def Values_int(ERho_star,EW_star,ERho,EW,rho_grid,Q_grid,points,num_z):
    shape = ERho_star[0,...].shape
    for iz in prange(num_z):
        flat_result = multilinear_interp(points[iz], (( rho_grid, Q_grid)), ERho[iz, ...])
        ERho_star[iz,...] =  flat_result.reshape(shape)
        flat_result = multilinear_interp(points[iz], (( rho_grid, Q_grid)), EW[iz, ...])
        EW_star[iz,...] =  flat_result.reshape(shape)        
    return ERho_star,EW_star
@nb.njit(cache=True)
def get_EJpi(EJpi,q_star2,Q_grid,EJ,num_z,num_v,num_q):
            for iz in prange(num_z):
                for iv_current in prange(num_v):
                    for iq in prange(num_q):
                        for iv_future in prange(num_v):
                            EJpi[iz,iv_current,iv_future,iq] = interp(q_star2[iz,iv_current,iq],Q_grid, EJ[iz,iv_future,:])
            return EJpi
class SimpleModel:
    """
        This solves a contract model with CRS production, heterogeneous match quality, and constant wages (no OJS!)
    """
    def __init__(self, input_param=None, js=None):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """

        self.log = logging.getLogger('MWF with CRS')
        self.log.setLevel(logging.INFO)
        self.K = 2
        K = 2
        self.p = input_param
        #Deep loops
        self.indices = list(product(range(self.p.num_z), range(self.p.num_v) ,range(self.p.num_q))) 
        self.indices_no_v = list(product(range(self.p.num_z),range(self.p.num_q)))

        self.deriv_eps = 1e-4 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid
        self.Q_grid = np.linspace(self.p.q_0,1,self.p.num_q) # Create worker productivity grid


        #self.N_grid=np.linspace(0,1,self.p.num_n)
        # Unemployment Benefits across Worker Productivities

        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        dimensions=[self.p.num_z]
        dimensions.extend([self.p.num_v] * (K - 1))  
        dimensions.extend([self.p.num_q] * (K - 1))   
        self.J_grid   = np.zeros(dimensions) #grid of job values, first productivity, then size for each step, then value level for each step BESIDES FIRST
        # Production Function in the Model
        self.fun_prod_onedim = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = self.fun_prod_onedim.reshape((self.p.num_z,) + (1,) * (self.J_grid.ndim - 1))
        self.qual_prod = 1 * self.Q_grid[ax,ax,:]  + self.p.prod_q * (1 - self.Q_grid[ax,ax,:]) #Quality adjustment for the productivity
        self.unemp_bf = self.p.u_bf_m #Half of the lowest productivity. Kinda similar to Shimer-like estimates who had 0.4 of the average

        # Wage and Shadow Cost Grids
        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max(), self.p.num_v ) #Note that this is not the true range of possible wages as this excludes the size part of the story
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)


        #Total firm size for each possible state
        self.grid = np.ogrid[[slice(dim) for dim in self.J_grid.shape]]


        #Job value and GE first
        self.v_grid = np.linspace(np.divide(self.pref.utility(self.unemp_bf),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod_onedim.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #Value promised to the worker at the bottom step
        
        self.simple_J=np.divide(self.fun_prod_onedim[:,ax,ax]*self.qual_prod - self.w_grid[ax,:,ax],1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        #This is eqUvalent to marginal value of a firm of size 1 at the lowest step

        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        



        self.J_grid = self.J_grid+np.divide(self.fun_prod * self.qual_prod-self.w_grid[ax,:,ax],1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #Alternatively, here rho is undervalued, as juniors will essentially be forever juniors, being paid nothing
    

        #Guess for the Worker value function
        self.W = np.zeros_like(self.J_grid)
        self.w_matrix = np.zeros(self.W.shape)
        
        self.w_matrix = self.w_grid[ax,:,ax]

        self.W += self.pref.utility(self.w_matrix)/(1-self.p.beta) #skip the first K-1 columns, as they don't correspond to the wage state. Then, pick the correct step, which is hidden in the last dimension of the grid


        self.q = np.zeros_like(self.J_grid) + self.Q_grid[ax,ax,:]

    def J_sep(self,Jg=None,Wg=None,Ug=None,Rhog=None,P=None,kappa=None,n0_g = None, sep_g = None,update_eq=1,s=1.0,layoff_iter=1):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        rho_grid = self.rho_grid
        Q_grid = self.Q_grid
        q = self.q

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
        if sep_g is None:
            sep_star = np.zeros_like(J)
        else:
            sep_star = sep_g
        # create representation for J1p
        Rho = J + rho_grid[ax,:,ax] * W    



        EW_star = np.copy(J)
        EJ_star = np.copy(J)
        ERho_star = np.copy(J)

        q_star  = self.q  
        #EJpi = np.zeros_like(J)

        #Separations related variables
        sep_grid = np.linspace(0,0.5,20)
        n1_s = np.zeros((self.p.num_z, self.p.num_v, self.p.num_q, sep_grid.shape[0]))
        q_s = np.zeros_like(n1_s)
        foc_sep = np.zeros_like(n1_s)
        J_s = np.zeros_like(n1_s)
        J_s_deriv = np.zeros_like(J_s)
        sep_reshaped = np.zeros_like(J_s) + sep_grid[ax,ax,ax,:]

        # prepare expectation call
        Ez = oe.contract_expression('avq,az->zvq', J.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', U.shape, self.X_trans_mat.shape)

        ite_num = 0        

        for ite_num in range(self.p.max_iter):
            J2 = J
            W2 = np.copy(W)
            U2 = U
            Rho2 = Rho
            # we compute the expected value next period by applying the transition rules
            EW = Ez(W, self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJ = Ez(J, self.Z_trans_mat)
            ERho = Ez(Rho, self.Z_trans_mat)
            EU = U


            #FOC for Separations
            if ite_num>=layoff_iter:
                #WHAT IF. We just do a direct derivative wrt s??? Like, we know what q_s and n1_s are. Inteprolate directly onto them, which will already give us the total derivative of J wrt s, no?
                q_s = np.fmin( self.Q_grid[:,ax] / (1-sep_grid[ax,:]),1)               
                #sep_star0 = np.copy(sep_star)
                sep_star[...] = 0
                
                #ERho_s = ERho
                W_s = np.zeros_like(J_s_deriv)
                Rho_s = np.zeros_like(J_s_deriv)
                for iz in range(self.p.num_z):
                 for iv in range(self.p.num_v):
                    for iq in range(self.p.num_q):                
                        W_s[iz,iv,iq,:] = np.interp(q_s[iq,:],self.Q_grid,EW[iz,iv,:])
                        Rho_s[iz,iv,iq,:] = np.interp(q_s[iq,:],self.Q_grid,ERho[iz,iv,:])

                J_s_deriv[...,0] = (Rho_s[...,1] - Rho_s[...,0]) / (sep_grid[1] - sep_grid[0])
                J_s_deriv[...,-1] = (Rho_s[...,-1] - Rho_s[...,-2]) / (sep_grid[-1] - sep_grid[-2]) 
                J_s_deriv[..., 1:-1]    = (Rho_s[...,2:] - Rho_s[...,:-2]) / (sep_reshaped[...,2:] - sep_reshaped[...,:-2]) 

                #foc_sep = J_s_deriv - (EW_star[...,ax] - EU) * rho_grid[ax,:,ax,ax] #Basically, here I take the derivative of (1-sep) * pc_star * EJ wrt sep. Since pc_star is a "coonstant" here, I look at how (1-sep) * EJ changes
                foc_sep = J_s_deriv * (1-sep_reshaped) - (Rho_s - rho_grid[ax,:,ax,ax] * W_s) - (W_s - EU) * rho_grid[ax,:,ax,ax] #Basically, here I take the derivative of (1-sep) * pc_star * EJ wrt sep. Since pc_star is a "coonstant" here, I look at how (1-sep) * EJ changes
                
                mask = np.where(foc_sep[...,0] < 0) #within the mask layoffs must be zero
                
                foc_sep = impose_increasing_fsep(-foc_sep)
            #how does prod-ty develop with layoffs right now (1-sep) * y * ( prod_q + q/(1-sep)*(1- prod_q) ) =  y * (prod_q * (1-sep) + q * (1-prod_q)) #so yep, layoffs just shed bad matches, and those are more productive in high periods. so you want to fire less!

                sep = solve_policies(ite_num,layoff_iter, sep_star,foc_sep, mask, sep_grid, self.p.num_z, self.p.num_v, self.p.num_q)
                assert np.all(sep[mask] <= 1e-8)
                #sep_star[mask] = 0 #this shouldn't be necessary, good check though!
                sep_star = - impose_increasing_policy(-sep)
                assert np.all(sep_star[mask] <= 1e-8)
 


            #Getting n1_star and q_star        
            q_star = np.fmin(q/(1-sep_star),1)

            #Future optimal expectations
            #ERho_star = interp_multidim(n0_star,N_grid,np.moveaxis(Rhod0,-1,0))
            #EJ_star = interp_multidim(q_star,self.Q_grid,np.moveaxis(J_s,-1,0))
            for iz in range(self.p.num_z):
                for iv in range(self.p.num_v):
                    ERho_star[iz,iv,:] = np.interp(q_star[iz,iv,:],self.Q_grid,ERho[iz,iv,:])
                    EW_star[iz,iv,:] = np.interp(q_star[iz,iv,:],self.Q_grid,EW[iz,iv,:])

            # Update worker value function
            W = self.pref.utility(self.w_matrix) + \
                self.p.beta * (sep_star * EU + (1 - sep_star) * EW_star ) 
            W = impose_increasing_W(W)
            assert np.isnan(W).sum() == 0, "W has NaN values"

            # Update firm value function 
            Rho = self.fun_prod * self.qual_prod - self.w_grid[ax,:,ax] + rho_grid[ax,:,ax] * W + self.p.beta * (1-sep_star) * (ERho_star - rho_grid[ax,:,ax] * EW_star)
            J = Rho - rho_grid[ax,:,ax] * W
            #J= self.fun_prod * self.qual_prod - self.w_grid[ax,:,ax] + self.p.beta * (1-sep_star) * EJ_star
            #J = impose_decreasing(J)
            assert np.isnan(J).sum() == 0, "J has NaN values"

            # Apply the matching function
            ite_prob_vx = self.p.alpha * np.power(1 - np.power(
                np.divide(self.p.kappa, np.maximum(J[self.p.z_0 - 1, :, 0], 1.0)), self.p.sigma), 1/self.p.sigma)
            
            # Update the guess for U given p
            U = np.max( self.pref.utility(self.unemp_bf) + self.p.beta * ite_prob_vx *
                               (W[self.p.z_0 - 1, :, 0] - EU) + self.p.beta * EU, axis=0)            


            # Compute the norm-inf between the two iterations of U(x)
            error_u  = np.max(abs(U - U2))
            error_j  = np.max(abs(Rho - Rho2))
            error_w1 = np.max(abs(W - W2))

            if np.array([error_u, error_w1, error_j]).max() < self.p.tol_simple_model and ite_num>10:
                break
            #else:
                #print(np.array([error_u, error_w1, error_j]))
                #J = 0.2 * J + 0.8 * J2
                #W = 0.2 * W + 0.8 * W2
                #U = 0.2 * U + 0.8 * U2
                #Rho = 0.2 * Rho + 0.8 * Rho2
                #Rho = J + rho_grid[ax,:,ax] * W
        # --------- wrapping up the model ---------

        # extract U2E probability
        usearch = np.argmax( self.pref.utility(self.unemp_bf) + self.p.beta * ite_prob_vx *
                     (W[self.p.z_0 - 1, :, 0] - EU) + self.p.beta * EU, axis=0)
        Pr_u2e = ite_prob_vx[usearch]
    
        self.Vf_J    = J
        self.Vf_W   = W
        self.Vf_Rho  = Rho
        self.sep_star  = sep_star
        self.Fl_wage = self.w_grid
        self.Vf_U    = U
        self.Pr_u2e  = Pr_u2e
        self.prob_find_vx = ite_prob_vx
        

        return self



    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)
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
        
        
         
def debug():     
    from primitives import Parameters
    p = Parameters()


    mwc_GE=SimpleModel(p)
    simple_model=mwc_GE.J_sep(update_eq=1,s=10.0)

#debug()