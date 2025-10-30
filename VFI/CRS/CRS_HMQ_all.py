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
from Simple_model_J_sev import SimpleModel
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
def impose_increasing_policy_q(A0):
    A = np.copy(A0)
    for v in range(1,A.shape[-1]):
        A[...,v] = np.maximum(A[...,v],A[...,v-1])
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
        A[...,v,:] = np.maximum(A[...,v,:],A[...,v-1,:]+1e-8)
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
def solve_policies(ite_num, rho_star, sep_star, foc_sep, sep_grid,
                   rho_grid, foc, num_z, num_v, num_q):
    # Precompute constants that do not change within the loops
    ite_ge_20 = (ite_num >= 1)
    #ite_ge_10 = (ite_num >= 10)
    #tenure_nonzero = (tenure != 0)


    for iz in prange(num_z):
        for iv in prange(num_v):
            for iq in prange(num_q):
                        rho_star[iz, iv, iq] = interp(0,
                                                                    foc[iz, iv, :, iq],
                                                                    rho_grid)
                        if ite_ge_20:
                            sep_star[iz, iv, iq] = interp(0,
                                                                    foc_sep[iz, iv, iq, :],
                                                                    sep_grid)
    return rho_star, sep_star

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
def get_EJpi(EJpi,EWpi,q_star2,Q_grid,EJ,EW,num_z,num_v,num_q):
            for iz in prange(num_z):
                for iv_current in prange(num_v):
                    for iq in prange(num_q):
                        for iv_future in prange(num_v):
                            EJpi[iz,iv_current,iv_future,iq] = interp(q_star2[iz,iv_current,iq],Q_grid, EJ[iz,iv_future,:])
                            EWpi[iz,iv_current,iv_future,iq] = interp(q_star2[iz,iv_current,iq],Q_grid, EW[iz,iv_future,:])
            return EJpi,EWpi
@nb.njit(cache=True)
def get_fin_diff(grid,values,out):

    out[...,0] = (values[...,1] - values[...,0]) / (grid[...,1] - grid[...,0])
    out[...,-1] = (values[...,-1] - values[...,-2]) / (grid[...,-1] - grid[...,-2]) 
    out[..., 1:-1]    = (values[...,2:] - values[...,:-2]) / (grid[...,2:] - grid[...,:-2])
    return out

class MultiworkerContract:
    """
        This solves a contract model with DRS production, hirings, and heterogeneous match quality.
    """
    def __init__(self, input_param=None, js=None, log=True,print_choice_simple=False):
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
        self.pref = Preferences(input_param=self.p,log=log)

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
        self.qual_prod_onedim =  1 * self.Q_grid  + self.p.prod_q * (1 - self.Q_grid)
        self.qual_prod = 1 * self.Q_grid[ax,ax,:]  + self.p.prod_q * (1 - self.Q_grid[ax,ax,:]) #Quality adjustment for the productivity
        self.unemp_bf = self.p.u_bf_m 

        # Wage and Shadow Cost Grids
        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max(), self.p.num_v ) #Note that this is not the true range of possible wages as this excludes the size part of the story
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)
        #Add the minimum wage
        self.w_grid[self.w_grid < self.p.min_wage] = self.p.min_wage
        #self.sev_grid = np.linspace(0, self.unemp_bf/2, 100 ) #note that, if I intriduce min wage here as well this will become incorrect!!! also maybe I'll want a higher upper bound

        #Total firm size for each possible state
        self.grid = np.ogrid[[slice(dim) for dim in self.J_grid.shape]]



        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        #print("Training simple model")
        mod_s = SimpleModel(self.p)
        self.simple_model = mod_s.J_sep(print_choice=print_choice_simple)
        self.sev_grid = self.simple_model.sev_grid #To make sure the two stay consistent
        self.rho_grid = self.simple_model.rho_grid
        self.w_grid = self.simple_model.w_grid
        #print("Simple training done")

        if js is None:
            self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
            self.js.update(self.simple_model.Vf_W[self.p.z_0-1,:,0], self.simple_model.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        else:
            self.js = js       

        self.w_matrix = np.zeros_like(self.simple_model.Vf_J) + self.w_grid[ax,:,ax]
        self.q = np.zeros_like(self.J_grid) + self.Q_grid[ax,ax,:]

    def J_sep(self,update_eq=1,s=1.0, print_choice = False):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        rho_grid = self.rho_grid
        Q_grid = self.Q_grid
        sev_grid = self.sev_grid
        q = self.q
        indices_no_v = self.indices_no_v
        indices = self.indices

        J = self.simple_model.Vf_J
        W = self.simple_model.Vf_W
        Rho = self.simple_model.Vf_Rho 
        sep_star = self.simple_model.sep_star
        q_star  = self.simple_model.q_star 
        U   = self.simple_model.Vf_U
        # create representation for J1p
        Jp = np.zeros_like(J)
        Wp = np.zeros_like(J)
        Rhop = np.zeros_like(J)
        # Updating J1 representation
        for iz, iq in indices_no_v:
            Wp[iz,:,iq] = splev(rho_grid, splrep(rho_grid,W[iz,:,iq],s=s))
            Rhop[iz,:,iq] = splev(rho_grid, splrep(rho_grid,Rho[iz,:,iq],s=s))
       
        Jp = Rhop - rho_grid[ax,:,ax] * Wp
        if print_choice:
            print("J shape", J.shape)
            print("W shape", W.shape)        



        EW_star = np.copy(J)
        #EJ_star = np.copy(J)
        ERho_star = np.copy(J)
        rho_star = np.zeros_like(J) + rho_grid[ax,:,ax]
        EJpi = np.zeros((self.p.num_z, self.p.num_v, self.p.num_v, self.p.num_q))
        EWpi = np.zeros((self.p.num_z, self.p.num_v, self.p.num_v, self.p.num_q))

        #Separations related variables
        sep_grid = np.linspace(0,0.95,19)
        n1_s = np.zeros((self.p.num_z, self.p.num_v, self.p.num_q, sep_grid.shape[0]))
        q_s = np.zeros_like(n1_s)
        foc_sep = np.zeros_like(n1_s)
        J_s = np.zeros_like(n1_s)
        J_s_deriv = np.zeros_like(J_s)
        sep_reshaped = np.zeros_like(J_s) + sep_grid[ax,ax,ax,:]
        #Severance values
        pu_deriv = np.zeros_like(sev_grid)
        # prepare expectation call
        Ez = oe.contract_expression('avq,az->zvq', J.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', U.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW_star)

        ite_num = 0
        error_js = 1

        for ite_num in range(self.p.max_iter):
            J2 = J
            W2 = np.copy(W)
            U2 = U
            Rho2 = np.copy(Rho)
            rho_star2 = np.copy(rho_star)
            q_star2 =   np.copy(q_star)                       
            # we compute the expected value next period by applying the transition rules
            EW = Ez(Wp, self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJ = Ez(Jp, self.Z_trans_mat)
            ERho = Ez(Rhop, self.Z_trans_mat)
            EU = U

            #Getting severance pay. The reason I want this here is so that new EU star shows for both sep_star and W
            #Formula: u'(b+sev)/u'(w)= 1- (beta * sev * (\partial pe/\artial sev))/(1-beta(1-pe))
            #NONE of this is state dependent except for rho: basically, for every value of rho, we already have a severance pay
            # get unemployed decisions for each severance value
            pu, ru, _ = self.getWorkerDecisions(EU, employed=False)
            #We need an actual derivative here, wrt sev
            pu_deriv = get_fin_diff(grid=sev_grid,values=pu,out=pu_deriv)
            #Or, we could try pe_deriv from the js
            #pu_deriv = self.js.pe_deriv(EU) #note that this would be derivative wrt EU!!! We still need a derivative of EU wrt sev
            #deriv_EU_sev = self.pref.utility_1d(self.unemp_bf+self.sev_grid) / (1-self.p.beta * (1-pu))
            #So the pu_deriv would be one times the other
            #pu_deriv = pu_deriv * deriv_EU_sev
            if print_choice:
                print("pu deriv range", pu_deriv.min(),pu_deriv.max())
            sev_check = self.pref.utility_1d(self.unemp_bf+self.sev_grid[:,ax]) * rho_grid[ax,:] - (1 - self.p.beta * self.sev_grid[:,ax] * pu_deriv[:,ax] /  (1-self.p.beta * (1-pu[:,ax])))
            #So... is this guy positive anywhere?
            mask = sev_check[0,:] > 0
            sev_star = np.interp(rho_grid,
                            (1 - self.p.beta * self.sev_grid * pu_deriv / (1-self.p.beta * (1-pu))) / self.pref.utility_1d(self.unemp_bf+self.sev_grid) ,
                            self.sev_grid)
            EU_star = np.interp(sev_star,self.sev_grid,EU)
            pu_star, _, _ = self.getWorkerDecisions(EU_star, employed=False)
            assert np.all(sev_star[mask]>0)


            EJpi,EWpi = get_EJpi(EJpi,EWpi,q_star2,Q_grid,EJ,EW,self.p.num_z,self.p.num_v,self.p.num_q)
            # get worker decisions
            _, _, pc = self.getWorkerDecisions(EWpi)
            # get worker decisions at EW + epsilon
            _, _, pc_d = self.getWorkerDecisions(EWpi + self.deriv_eps) 
        
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff = np.zeros_like(pc)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value

            foc = rho_grid[ax, ax, :, ax] - rho_grid[ax, :, ax, ax] - EJpi * log_diff / self.deriv_eps 
            

            #FOC for Separations
            if ite_num>=1:
                #WHAT IF. We just do a direct derivative wrt s??? Like, we know what q_s and n1_s are. Inteprolate directly onto them, which will already give us the total derivative of J wrt s, no?
                q_s = np.fmin(q[...,ax] / (1-sep_reshaped),1)
                #q_s = np.fmin(self.Q_grid[:,ax] / (1-sep_grid[ax,:]),1)               
                sep_star0 = np.copy(sep_star)
                sep_star[...] = 0
                  
                #ERho_s = ERho - rho_grid[ax,:,ax] * EW
                points = []
                rho_sz = np.repeat(rho_star2[...,ax], sep_grid.shape[0], axis=-1)
                for iz in range(self.p.num_z):
                    points.append( tuple_into_2darray((rho_sz[iz, ...], q_s[iz,...])))
                Rho_s = np.zeros_like(J_s_deriv)
                W_s = np.zeros_like(J_s_deriv)
                Rho_s,W_s = Values_int(Rho_s,W_s,ERho,EW,rho_grid,Q_grid,points,self.p.num_z)

                J_s_deriv = get_fin_diff(grid=sep_reshaped,values=Rho_s,out=J_s_deriv)
                _, re_s, pc_s = self.getWorkerDecisions(W_s)
                #Assert that severance makes sense
                assert np.all(EU_star * rho_grid - sev_star / ( 1 - self.p.beta*(1-pu_star))> EU[0] * rho_grid)
                foc_sep = (J_s_deriv * (1-sep_reshaped) - (Rho_s - rho_star[...,ax] * W_s) ) * pc_s - (re_s+W_s - EU_star[ax,:,ax,ax]) * rho_grid[ax,:,ax,ax] - sev_star[ax,:,ax,ax] / ( 1 - self.p.beta*(1-pu_star[ax,:,ax,ax]))
                #What do we do with EU here? it should be EU_star, correct?
                #foc_sep = J_s_deriv * pc_s - (re_s+W_s - EU) * rho_grid[ax,:,ax,ax]
                foc_sep = impose_increasing_fsep(-foc_sep)

            rho_star,sep = solve_policies(ite_num,rho_star,sep_star,foc_sep,sep_grid, rho_grid, foc, self.p.num_z, self.p.num_v, self.p.num_q)
            rho_star = impose_increasing_policy(rho_star)
            rho_star = impose_increasing_policy_q(rho_star) #Ensure that rho_star also rises with q
            if ite_num >= 1:
                sep_star = 0.4 * (- impose_increasing_policy(-sep)) + 0.6 * sep_star0

            #Getting q_star        
            q_star = np.fmin(q/(1-sep_star),1) #so 1-sep_star = q/q_star
            q_star = impose_increasing_policy_q(q_star)
            sep_star = 1 - q/q_star #this way sep_star is consistent with the q_star change
            points = []
            for iz in range(self.p.num_z):
                points.append( tuple_into_2darray((rho_star[iz, ...], q_star[iz, ...])))


            ERho_star, EW_star =  Values_int(ERho_star,EW_star,ERho,EW,rho_grid,Q_grid,points,self.p.num_z)
            _, re_star, pc_star = self.getWorkerDecisions(EW_star)

            #Unemployed value
            U = self.pref.utility_gross(self.unemp_bf + self.sev_grid) + self.p.beta * (ru + EU)
            U = 0.2 * U + 0.8 * U2

            # Update worker value function
            W = self.pref.utility(self.w_matrix) + \
                self.p.beta * (sep_star * EU_star[ax,:,ax] + (1 - sep_star) * (EW_star + re_star)) #For more steps the ax at the end won't be needed as EW_star itself will have multiple steps
            W = impose_increasing_W(W)
            assert np.isnan(W).sum() == 0, "W has NaN values"

            # Update firm value function 
            Rho = self.fun_prod * self.qual_prod - self.w_grid[ax,:,ax] + self.p.beta * pc_star * (1-sep_star) * (ERho_star - rho_star*EW_star) + \
                rho_grid[ax,:,ax] * W - self.p.beta * sep_star * sev_star[ax,:,ax] / ( 1 - self.p.beta*(1-pu_star[ax,:,ax]))   #I've now introduce the cost to layoffs        
            J= Rho - rho_grid[ax,:,ax] * W
            J = impose_increasing_policy_q(J) 
            assert np.isnan(J).sum() == 0, "J has NaN values"
            assert np.isnan(Rho).sum() == 0, "Rho has NaN values"  
    
            
            Rho = .2 * Rho + .8 * Rho2
            J= .2 * J + .8 * J2
            W = .2 * W + .8 * W2 #we're completely ignoring the 0th step

            # Updating J representation
            for iz, iq in indices_no_v:
                Wp[iz,:,iq] = splev(rho_grid, splrep(rho_grid,W[iz,:,iq],s=s))
                Rhop[iz,:,iq] = splev(rho_grid, splrep(rho_grid,Rho[iz,:,iq],s=s)) #Should I go back to the BL-style smoothing? This would have value at least due to fitting the same kind of polynomial every time

            
            #TRYING AGAIN WITH THE BASIC RHO
            Jp = Rhop - rho_grid[ax,:,ax]*Wp    
            Jp = impose_increasing_policy_q(Jp)  

            # Compute convergence criteria
            error_j1i = array_exp_dist(Rho,Rho2,100) #np.power(J - J2, 2).mean() / np.power(J2, 2).mean()  
            error_w1 = array_dist(W, W2)

            # update worker search decisions
            if (((ite_num % 10) == 0)):
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    if print_choice:
                        print("Errors:",  error_j1i, error_w1, error_js)
                        print("q_star", q_star[self.p.z_0-2,50, :])
                        print("sep", sep_star.min(),sep_star.max())   
                        print("sev", sev_star.min(),sev_star.max())                
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model and error_js < self.p.tol_search
                            and ite_num > 100):
                        break
                    # ------ or update search fsunction parameter using relaxation ------
                    else:
                            P_xv = self.matching_function(J[self.p.z_0-1, :, 0])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            #error_js = self.js.update(self.v_grid, P, type=1, relax=relax)
                            error_js = self.js.update(W[self.p.z_0-1, :, 0], P_xv, type=1, relax=relax)                       

                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[...,1], Ji)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    if print_choice:
                        print("Errors:",  error_j1i,  error_w1, error_js)
                        print("q_star", q_star[self.p.z_0-2,50, :])
                        print("sep", sep_star.min(),sep_star.max())    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            if print_choice:
             if (((ite_num % 1000)  == 0) & (ite_num>10)):
                   
                plt.plot(W[self.p.z_0-2, :, 0], J [self.p.z_0-2, :, 0], label='1 senior value function')
                plt.plot(W[self.p.z_0-2, :, 0], Jp[self.p.z_0-2, :, 0], label='1 senior value function') 
                #plt.show() # this will load image to console before executing next line of code
                #plt.plot(W[self.p.z_0-1, 0, 1, :, 0, 1], 1-pc_star[self.p.z_0-1, 0, 1, :, 0], label='Probability of the worker leaving across submarkets')      
                plt.show()


        # --------- wrapping up the model ---------

        # find rho_j2j
        rho_j2j = np.zeros((self.p.num_z, self.p.num_v, self.p.num_q))
        ve_star = np.zeros((self.p.num_z, self.p.num_v, self.p.num_q))
        #for ix in range(self.p.num_x):
        for iz,  iq in indices_no_v:
        #for iz in range(self.p.num_z):
            ve_star[iz,  :, iq] = self.js.ve( EW_star[iz,  :, iq])
            rho_j2j[iz,  :, iq] = np.interp(ve_star[iz,  :, iq], W[self.p.z_0-1,  :, 0], rho_grid) #Is this W correct? Or should it be W[p.z_0-1,  :, 0]? Before I had W[iz,:,iq] here

        # find rho_u2e
        self.ve = self.js.ve(EU)
        Pr_u2e = self.js.pe(EU) # this does not include the inefficiency of search for employed
        self.rho_u2e = np.interp(self.ve, W[self.p.z_0-1,  :, 0], rho_grid) #note that this u2e will be a grid. which is fine, but in simulation I will need to keep track of it!!
        # value functions
        self.Vf_J = J
        self.Vf_W = W
        self.Vf_U = U
        self.Jp = Jp
        self.EW_star  = EW_star

        #Getting the target wage and value
        target_w   = np.zeros((self.p.num_z, self.p.num_q))
        target_rho = np.zeros((self.p.num_z, self.p.num_q))
        for iq in range(self.p.num_q):
            for iz in range(self.p.num_z):
                rho1 = rho_grid.max()
                rho0 = rho_grid.min()
                for _ in range(20):
                    rhog = (rho0 + rho1) / 2
                    val = np.interp(rhog, rho_grid, EJ[iz, :, iq])
                    if val > 0:
                        rho0 = rhog
                    else:
                        rho1 = rhog
                target_rho[iz, iq] = rhog
                target_w[iz, iq] = np.interp(rhog, rho_grid, self.w_grid)
        self.target_w = target_w
        self.target_rho = target_rho
        
        # policies
        self.rho_j2j = rho_j2j
        #self.rho_u2e = rho_u2e
        self.sev_star = sev_star
        self.rho_star = rho_star
        self.sep_star = sep_star
        self.re_star = re_star
        self.q_star = q_star
        self.pe_star = 1-pc_star
        self.ve_star = ve_star
        self.Pr_u2e = Pr_u2e

        self.error_w1 = error_w1
        self.error_j = error_j1i
        #self.error_j1p = error_j1g
        self.error_js = error_js
        self.niter = ite_num


        #self.append_results_to_pickle(J, W, U, Rho,EW_star, sep_star)

        #Saving the entire model
        self.save("model_GE_all.pkl")
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
        key = (self.p.num_z,self.p.num_v,self.p.z_corr,self.p.prod_var_z,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.s_job,self.p.alpha,self.p.kappa,self.p.dt,self.p.u_bf_m,self.p.min_wage)
        
        all_results[key] = self
        #Save the updated dictionary back to the pickle file        
        with open(filename, "wb") as output_file:
            pickle.dump(all_results, output_file)
        print(f"Results for p = {key} have been appended to {filename}.")

    def append_results_to_pickle(self, J, W, U, Rho, EW_star, sep_star, pickle_file="results_GE_all.pkl"):
        # Step 1: Load the existing data from the pickle file
        try:
            with open(pickle_file, "rb") as file:
                all_results = pickle.load(file)
        except FileNotFoundError:
            all_results = {}
            print("No existing file found. Creating a new one.")

        # Step 2: Create results for the multi-dimensional p
        new_results = self.save_results_for_p(J, W, U, Rho, EW_star, sep_star)

        # Step 3: Use a tuple (p.num_z, p.num_v, p.num_n) as the key
        key = (self.p.num_z,self.p.num_v,self.p.z_corr,self.p.prod_var_z,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.s_job,self.p.alpha,self.p.kappa,self.p.dt,self.p.u_bf_m,self.p.min_wage)
        # Step 4: Add the new results to the dictionary
        all_results[key] = new_results

        # Step 5: Save the updated dictionary back to the pickle file
        with open(pickle_file, "wb") as file:
            pickle.dump(all_results, file)

        print(f"Results for p = {key} have been appended to {pickle_file}.")
    def save_results_for_p(self, J, W, U, Rho, EW_star, sep_star):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
        'date': current_date,
        'J': J,
        'W': W,
        'U': U,
        'Rho': Rho,
        'EW_star': EW_star,
        'sep_star': sep_star,
        'p_value': (self.p.num_z,self.p.num_v,self.p.z_corr,self.p.prod_var_z,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.s_job,self.p.alpha,self.p.kappa,self.p.dt,self.p.u_bf_m,self.p.min_wage)
    }    
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
        :return: pe,re,pc search decision and associated return
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
    new_baseline = {
   'q_0': 0.56602, 'prod_q':	0.50425,'u_bf_m':	2.34264/4,'s_job':	0.779616,'alpha':	0.72,'z_corr':	0.946006,'prod_var_z':	0.646317, 'min_wage': 2.34264/4
    }
    p = Parameters(overwrite=new_baseline)


    mwc_GE=MultiworkerContract(p)
    model=mwc_GE.J_sep(update_eq=1,s=0.0,print_choice=True)

#debug()