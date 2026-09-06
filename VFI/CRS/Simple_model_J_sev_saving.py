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

def load_pickle_file(new_p_value, pickle_file="results_GE_sev.pkl"):
    # Step 1: Load the existing data from the pickle file
    try:
        with open(pickle_file, "rb") as file:
            all_results = pickle.load(file)
    except FileNotFoundError:
        # If file doesn't exist, start with an empty dictionary
        all_results = {}
        print("No existing file found. Creating a new one.")
@nb.njit(cache=True)
def impose_increasing_z(A0):
    A = np.copy(A0)
    nv = A.shape[0]
    for v in range(1,nv):
        A[v,...] = np.maximum(A[v,...],A[v-1,...])
    return A
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

#Interpolation functions instead of RegularGridInterpolator
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
        self.w_grid = np.linspace(self.unemp_bf / 2, self.fun_prod.max(), self.p.num_v ) #Note that this is not the true range of possible wages as this excludes the size part of the story
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)
        #Add the minimum wage
        self.w_grid[self.w_grid < self.p.min_wage] = self.p.min_wage        
        self.ass_grid = np.linspace(0,  4 * self.fun_prod.max(), 100 ) #This is now a grid of SAVINGS that the unemployed worker may get from the firm
        #The highest value is a year's worth of highest possible salary. that's quite a lot

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

    def J_sep(self,Jg=None,Wg=None,Ug=None,Rhog=None,P=None,kappa=None,n0_g = None, sep_g = None,update_eq=1,s=1.0,layoff_iter=1,print_choice=False):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        rho_grid = self.rho_grid
        Q_grid = self.Q_grid
        q = self.q
        ass_grid = self.ass_grid

        if Jg is None:
            J = np.copy(self.J_grid)
        else:
            J = np.copy(Jg)
        if Wg is None:
            W = np.copy(self.W)
        else:
            W = impose_increasing_W(Wg)
        if Ug is None:
            U = self.pref.utility(self.unemp_bf + ass_grid * (1-self.p.beta)) / (1 - self.p.beta) #Basic assumption: assets are smoothed the whole way
        else:
            U = np.copy(Ug)
        if sep_g is None:
            sep_star = np.zeros_like(J)
        else:
            sep_star = sep_g
        # create representation for J1p
        Rho = J + rho_grid[ax,:,ax] * W    

        # First matching function
        ite_prob_vx = self.p.alpha * np.power(1 - np.power(
                np.divide(self.p.kappa, np.maximum(J[self.p.z_0 - 1, :, 0], self.p.kappa)), self.p.sigma), 1/self.p.sigma)

        EW_star = np.copy(J)
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
        #Unemp worker variables
        a_star = self.ass_grid - self.ass_grid*(1-self.p.beta) #optimal savings guess; basically perfect smoothing
        c=ass_grid[:,ax]+self.unemp_bf - self.p.beta * ass_grid[ax,:]  #To be fair tis one is always the case, I don't need to recalculate this guy
        c_star = np.zeros_like(ass_grid)
        for a in range(ass_grid.shape[0]):
            c_star[a] = np.interp(a_star[a],ass_grid,c[a,:])
        # prepare expectation call
        Ez = oe.contract_expression('avq,az->zvq', J.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', U.shape, self.X_trans_mat.shape)

        ite_num = 0        

        for ite_num in range(self.p.max_iter):
            W2 = np.copy(W)
            U2 = U
            Rho2 = np.copy(Rho)
            J2 = np.copy(J)
            # we compute the expected value next period by applying the transition rules
            EW = Ez(W, self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            ERho = Ez(Rho, self.Z_trans_mat)
            EU = U

            #Getting optimal unemployed search: 1st dimension is FUTURE assets, second dimension is where the worker chooses to search
            usearch = np.argmax( ite_prob_vx[:] * (W[ self.p.z_0 - 1, :, 0] - EU[0]) + self.p.beta * EU[0], axis=-1)
            #usearch = np.argmax( ite_prob_vx[ax,:] * (W[ax, self.p.z_0 - 1, :, 0] - EU[:,ax]) + self.p.beta * EU[:,ax], axis=1)
            pu = ite_prob_vx[usearch]
            search_value = W[self.p.z_0 - 1, usearch, 0]
            #Optimal unemployed savings: the Euler equation is 1-pu(a'(a)) = u'(c(a'))/u'(c(a''(a')))
            #So now we'll do 2 dimensions: first dim is current a, second is the future one we're optimizing over. 


            #foc_a = (1-pu[ax,:]) * self.pref.utility_1d(c_star[ax,:]) - self.pref.utility_1d(c)

            #for a in range(ass_grid.shape[0]):
            #    a_star[a] = np.interp(0,
            #                   impose_increasing(-foc_a[a,:]),
            #                   ass_grid)  
            #    c_star[a] = np.interp(a_star[a],ass_grid,c[a,:])  #Could we have done it with c_star from the get-go? mayybe...     
            #Now from this, we know the optimal c upon layoffs, or, equivalently, the severance we need to pay optimally: 1/u'(w)=1/u'(c(sev))
            #sev_star= np.interp(rho_grid,1/self.pref.utility_1d(c_star),ass_grid) #Note that sev_star is on a same grid as rho, net assets anymore
            #EU_star = np.interp(sev_star,ass_grid,EU)
            #Alternatively do this directly?
            #mask = (c > 0) #cannot introduce buts where consumption is negative
            #saving_value = self.pref.utility(np.fmax(c,1e-10)) + self.p.beta * ( (1-pu[ax,:]) * EU[ax,:] + pu[ax,:] * search_value[ax,:])
            saving_value = self.pref.utility(np.fmax(c,1e-10)) + self.p.beta * ( (1-pu) * EU[ax,:] + pu * search_value)
            #opt_sav_idx = np.argmax(saving_value,axis=1)
            #a_star = ass_grid[opt_sav_idx]
            # #Not necessary for now but makes this stuff very easy still
            #for a in range(ass_grid.shape[0]):
            #    c_star[a] = np.interp(a_star[a],ass_grid,c[a,:])
            sev_profit = rho_grid[:,ax] * EU[ax,:] - ass_grid[ax,:] #Note that to properly calculate EU we still need stuff like c_star, right?
            opt_sev_idx = np.argmax(sev_profit,axis=1)
            sev_star = ass_grid[opt_sev_idx]
            EU_star = EU[opt_sev_idx]
            #Assert that severance makes sense. This way AT LEAST the severance is not fucking us over
            assert np.all(EU_star * rho_grid - sev_star >= EU[0] * rho_grid) #This is a very basic comparison, but basically checking whether the current option is better than paying zero


            #FOC for Separations
            if ite_num>=layoff_iter:
                #WHAT IF. We just do a direct derivative wrt s??? Like, we know what q_s and n1_s are. Inteprolate directly onto them, which will already give us the total derivative of J wrt s, no?
                q_s = np.fmin( Q_grid[:,ax] / (1-sep_grid[ax,:]),1)               
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
                #EJ_s = Rho_s - rho_grid[ax,:,ax,ax] * W_s
                #J_s_deriv[...,0] = (Rho_s[...,1] * (1-sep_grid[1]) - Rho_s[...,0] * (1-sep_grid[0])) / (sep_grid[1] - sep_grid[0])
                #J_s_deriv[...,-1] = (Rho_s[...,-1] * (1-sep_grid[-1]) - Rho_s[...,-2] * (1-sep_grid[-2])) / (sep_grid[-1] - sep_grid[-2]) 
                #J_s_deriv[..., 1:-1]    = (Rho_s[...,2:] * (1-sep_reshaped[...,2:]) - Rho_s[...,:-2] * (1-sep_reshaped[...,:-2])) / (sep_reshaped[...,2:] - sep_reshaped[...,:-2]) 
                J_s_deriv[...,0] = (Rho_s[...,1] - Rho_s[...,0] ) / (sep_grid[1] - sep_grid[0])
                J_s_deriv[...,-1] = (Rho_s[...,-1] - Rho_s[...,-2] ) / (sep_grid[-1] - sep_grid[-2]) 
                J_s_deriv[..., 1:-1]    = (Rho_s[...,2:] - Rho_s[...,:-2] ) / (sep_reshaped[...,2:] - sep_reshaped[...,:-2]) 

                #foc_sep = J_s_deriv - (EW_star[...,ax] - EU) * rho_grid[ax,:,ax,ax] #Basically, here I take the derivative of (1-sep) * pc_star * EJ wrt sep. Since pc_star is a "coonstant" here, I look at how (1-sep) * EJ changes
                foc_sep = J_s_deriv * (1-sep_reshaped) - (Rho_s - rho_grid[ax,:,ax,ax] * W_s) - (W_s - EU_star[ax,:,ax,ax]) * rho_grid[ax,:,ax,ax] - sev_star[ax,:,ax,ax] #Basically, here I take the derivative of (1-sep) * pc_star * EJ wrt sep. Since pc_star is a "coonstant" here, I look at how (1-sep) * EJ changes
                #foc_sep = J_s_deriv + rho_grid[ax,:,ax,ax] * W_s - (W_s - EU_star[ax,:,ax,ax]) * rho_grid[ax,:,ax,ax] - sev_star[ax,:,ax,ax]
                mask = np.where(foc_sep[...,0] < 0) #within the mask layoffs must be zero
                #mask_pos = np.where((foc_sep[...,0] > 0) & (foc_sep[...,1] > 0))
                foc_sep = impose_increasing_fsep(-foc_sep)
                foc_sep = impose_increasing_z(foc_sep) #this is a negative of the original foc_sep so this should be ok

            #how does prod-ty develop with layoffs right now (1-sep) * y * ( prod_q + q/(1-sep)*(1- prod_q) ) =  y * (prod_q * (1-sep) + q * (1-prod_q)) #so yep, layoffs just shed bad matches, and those are more productive in high periods. so you want to fire less!

                sep = solve_policies(ite_num,layoff_iter, sep_star,foc_sep, mask, sep_grid, self.p.num_z, self.p.num_v, self.p.num_q)
                assert np.all(sep[mask] <= 1e-8)
                #assert np.all(sep[mask_pos] > 0)
                #sep_star[mask] = 0 #this shouldn't be necessary, good check though!
                sep_star = - impose_increasing_policy(-sep)
                assert np.all(sep_star[mask] <= 1e-8)
                #assert np.all(sep_star[1:,...]-sep_star[:-1,...]<=0) #Here this MUST be true! W_s only ever changes because of layoffs anyway!

 


            #Getting q_star        
            q_star = np.fmin(q/(1-sep_star),1)
            q_star = impose_increasing_policy_q(q_star)
            sep_star = 1 - q/q_star #this way sep_star is consistent with the q_star change
            for iz in range(self.p.num_z):
                for iv in range(self.p.num_v):
                    ERho_star[iz,iv,:] = np.interp(q_star[iz,iv,:],self.Q_grid,ERho[iz,iv,:])
                    EW_star[iz,iv,:] = np.interp(q_star[iz,iv,:],self.Q_grid,EW[iz,iv,:])

            # Update firm value function 
            #Rho = self.fun_prod * self.qual_prod - self.w_grid[ax,:,ax] + rho_grid[ax,:,ax] * W - self.p.beta * sep_star * sev_star[ax,:,ax] + self.p.beta * (1-sep_star) * (ERho_star - rho_grid[ax,:,ax] * EW_star)
            J = self.fun_prod * self.qual_prod - self.w_grid[ax,:,ax] - self.p.beta * (sep_star+self.p.delta) * sev_star[ax,:,ax] +\
                self.p.beta * (1-sep_star-self.p.delta) * (ERho_star - rho_grid[ax,:,ax] * EW_star)

            #J = Rho - rho_grid[ax,:,ax] * W
            #J= self.fun_prod * self.qual_prod - self.w_grid[ax,:,ax] + self.p.beta * (1-sep_star) * EJ_star
            #J = impose_decreasing(J)
            assert np.isnan(J).sum() == 0, "J has NaN values"

            # Update worker value function
            W = self.pref.utility(self.w_matrix) + \
                self.p.beta * ((sep_star+self.p.delta) * EU_star[ax,:,ax] + (1 - sep_star - self.p.delta) * EW_star ) 
            W = impose_increasing_W(W)
            assert np.isnan(W).sum() == 0, "W has NaN values"


            Rho = J + rho_grid[ax,:,ax] * W
            # Apply the matching function
            ite_prob_vx = self.p.alpha * np.power(1 - np.power(
                np.divide(self.p.kappa, np.maximum(J[self.p.z_0 - 1, :, 0], self.p.kappa)), self.p.sigma), 1/self.p.sigma)
            
            # Update the guess for U given p
            #EU_fut = np.interp(a_star,ass_grid,EU)
            #U = np.max( self.pref.utility(c_star[:,ax]) + self.p.beta * ite_prob_vx[ax,:] *
            #                   (W[ax, self.p.z_0 - 1, :, 0] - EU_fut[:,ax]) + self.p.beta * EU_fut[:,ax], axis=1)
            U = np.max(saving_value,axis=1)
            #pu_star = np.interp(a_star,ass_grid,pu)
            #U = self.pref.utility(c_star[:,ax]) + self.p.beta * (pu_star * np.interp(ass_star,ass_grid,EU) + (1-pu_star))
            # Compute the norm-inf between the two iterations of U(x)
            error_u  = np.max(abs(U - U2))
            error_j  = np.max(abs(Rho - Rho2))
            error_w1 = np.max(abs(W - W2))
            Rho = 0.4 * Rho + 0.6 * Rho2
            J = 0.4 * J + 0.6 * J2
            W = 0.4 * W + 0.6 * W2

            if print_choice:
                print(error_j,error_w1,error_u, sep_star.max())
            #print(sev_star.min(),sev_star.max())

            if np.array([error_u, error_w1, error_j]).max() < self.p.tol_simple_model and ite_num>10:
                break
        # --------- wrapping up the model ---------

        # extract U2E probability
        #usearch = np.argmax( self.pref.utility(self.unemp_bf) + self.p.beta * ite_prob_vx *
        #             (W[self.p.z_0 - 1, :, 0] - EU[0,ax]) + self.p.beta * EU[0,ax], axis=1)
        #Pr_u2e = ite_prob_vx[usearch]
        #saving_value = self.pref.utility(np.fmax(c,1e-10)) + self.p.beta * ( (1-pu[ax,:]) * EU[ax,:] + pu[ax,:] * search_value[ax,:])
        #opt_sav_idx = np.argmax(saving_value,axis=1)
        #a_star = ass_grid[opt_sav_idx]    
        self.Vf_J    = J
        self.Vf_W   = W
        self.Vf_Rho  = Rho
        self.sep_star  = sep_star
        #self.a_star    = a_star
        self.q_star    = q_star
        self.Fl_wage = self.w_grid
        self.Vf_U    = U
        #self.Pr_u2e  = Pr_u2e
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