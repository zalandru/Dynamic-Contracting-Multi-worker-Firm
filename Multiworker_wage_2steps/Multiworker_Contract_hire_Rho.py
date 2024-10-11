import numpy as np
import logging
from scipy.stats import lognorm as lnorm

import opt_einsum as oe

from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search import JobSearchArray
from valuefunction_multi import PowerFunctionGrid
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interpn
from regulargrid.cartesiangrid import CartesianGrid
import numba as nb

import pickle
import datetime
import time

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

def load_pickle_file(new_p_value, pickle_file="results_hmq_sep.pkl"):
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
#@nb.njit()
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


#Maybe create my own version of this that could adapt to multidimensional functions?? Although Ig I can already do it, just write parallel everywhere else
#At the very least, the boundary condition could be universal, no? Do the point check first and then everything else. That only works in stuff like optimized_loop though. Most others go to different points
#Solve for rho_star

#Okay new perspective: focus on the fact that most points are under the same axis.

@nb.njit(cache=True)
def optimized_loop(pc, rho_grid, N_grid1, foc, rho_star, num_z, num_n, n_bar, num_v, num_q):
    for in0 in range(num_n): #Not this: we don't do the case for max juniors. for some reason separations fail otherwise
        for in1 in range(num_n):
         if (N_grid1[in0] + N_grid1[in1] > n_bar):
          continue
         for iz in range(num_z):
            for iv in range(num_v):
             for iq in range(num_q):
                
                rho_star[iz,in0, in1, iv,iq] = interp(0,
                                                    impose_increasing(foc[iz, in0, in1, :, iv,iq]),
                                                    rho_grid[:])  
                #rho_min = np.min(rho_grid[pc[iz, in0, in1, :, iq] > 0])  # Lowest promised rho with continuation > 0
                #Isearch = (pc[iz, in0, in1, :, iq] > 0)
                
                #if np.any(Isearch):
                #    Isearch_indices = np.where(Isearch)[0]
                #    for iv in Isearch_indices:
                 #       rho_star[iz, in0, in1, iv,iq] = interp(
                #            0, impose_increasing(foc[iz, in0, in1, Isearch, iv, iq]), rho_grid[Isearch]
                #        )
                
                #IqUt = ~(pc[iz, in0, in1, :, iq] > 0)
                #if np.any(IqUt):
                #    rho_star[iz, in0, in1, IqUt, iq] = rho_min

    return rho_star

@nb.njit(cache=True,parallel=True)
def n0(Jd0, n0_star, N_grid, Ihire, hire_c):
    for idx in np.argwhere(Ihire):
        #slice_Jd0 = Jd0[idx[0], idx[1], idx[2], idx[3], idx[4], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], idx[4], :-1]+n1_star[idx[0], idx[1], idx[2], idx[3], idx[4]]*rho_star[idx[0], idx[1], idx[2], idx[3], idx[4]]*(Wd0[idx[0], idx[1], idx[2], idx[3], idx[4],1:]-Wd0[idx[0], idx[1], idx[2], idx[3], idx[4],:-1]) 
        slice_Jd0 = (Jd0[idx[0], idx[1], idx[2], idx[3], idx[4], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], idx[4], :-1]) / (N_grid[1:]-N_grid[:-1])
        n0_star[idx[0], idx[1], idx[2], idx[3], idx[4]] = interp( -hire_c ,impose_increasing(-slice_Jd0),N_grid[1:]) #oh shit, should we also account for how that affects the worker value???
    print("n0_star borders", n0_star.min(), n0_star.max())   
    return n0_star 
@nb.njit(cache=True)
def EJs(EJ_star, EW_star, Jd0, Wd0, n0_star, N_grid, num_z, num_n, num_v, num_q):
    for iz in range(num_z):
        for in0 in range(num_n):
            for in1 in range(num_n):
                for iv in range(num_v):
                 for iq in range(num_q):
                    EJ_star[iz,in0,in1,iv,iq] = interp(n0_star[iz,in0,in1,iv,iq],N_grid,Jd0[iz,in0,in1,iv,iq,:])
                    EW_star[iz,in0,in1,iv,iq] = interp(n0_star[iz,in0,in1,iv,iq],N_grid,Wd0[iz,in0,in1,iv,iq,:])
    return EJ_star, EW_star
@nb.njit(cache=True)
def solve_everything(pc, rho_grid, N_grid, N_grid1, foc, rho_star, sep_star, n1, num_z, num_n, n_bar, num_v, num_q):
    for iz in range(num_z):
        for in0 in range(num_n):
            for in1 in range(num_n):
                if (in0 +in1 > n_bar):
                    continue
                for iv in range(num_v):
                 for iq in range(num_q):
                  rho_star[iz,in0, in1, iv,iq] = interp(0,
                                                    impose_increasing(foc[iz, in0, in1, :, iv,iq]),
                                                    rho_grid[:])      
                  n1[iz, in0, in1, iv, iq] = (N_grid[in0]*(1-sep_star[iz,in0,in1,iv, iq])+N_grid1[in1])*interp(rho_star[iz, in0, in1, iv, iq], rho_grid, pc[iz,in0,in1,:,iq])         
    return rho_star,n1
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
@nb.njit(cache=True)
def sep_solve_1(n1_s,q_s,q_deriv_s,pc_temp,sep_grid,N_grid1,size,q1,Q_grid,num_n,num_q,q_0):
              
                for s in range(sep_grid.shape[0]):
                    n1_s[...,s] = (size[...,0]*(1-sep_grid[s])+size[...,1]) * pc_temp
                    q_s[...,s] = (size[...,0] * np.minimum(q_0,1-sep_grid[s])+q1*size[...,1]) / (size[...,0]*(1-sep_grid[s])+size[...,1])
                #Calculating derivative of future q wrt s
                    if sep_grid[s] < 1 - q_0:
                        q_deriv_s[...,s] = size[...,0]* q_s[...,s] / (size[...,1]+(1-sep_grid[s])*size[...,0])
                    else:
                        q_deriv_s[...,s] = (( q1 - 1 ) * size[...,1] * size[...,0]) / np.power(size[...,0]*(1-sep_grid[s])+size[...,1],2)
                assert np.isnan(n1_s).sum() == 0, "n1_s has NaN values"
                assert np.isnan(q_s).sum() == 0, "q_s has NaN values"
                #Calculating closest grid points to each future size and quality
                range_n = np.arange(num_n)
                n1_s_ceil = np.ceil(np.interp( n1_s, N_grid1, range_n)).astype(np.int32)
                n1_s_floor = np.floor(np.interp( n1_s, N_grid1, range_n)).astype(np.int32)
                range_q = np.arange(num_q)  
                q_s_ceil = np.ceil(np.interp( q_s, Q_grid, range_q)).astype(np.int32)
                q_s_floor = np.floor(np.interp( q_s, Q_grid, range_q)).astype(np.int32)
              
                return n1_s_ceil,n1_s_floor,q_s_ceil,q_s_floor#,J_fut_deriv_n,J_fut_deriv_q
@nb.njit(cache=True)
def sep_solve_2(sep_star,J_fut_deriv_n,J_fut_deriv_q,J_s_deriv,q_deriv_s,pc_temp,inv_util_1d,re_star,EW_star,EU,sep_grid,size,num_z,num_v,num_n,num_q):
                #foc_sep = - J_fut_deriv_n * pc_temp[...,ax] * size[...,ax, 0] + J_fut_deriv_q * q_deriv_s - size[...,ax, 0] * (re_star[...,ax]+EW_star[...,ax] - EU) / inv_util_1d  
                foc_sep = J_s_deriv - size[...,ax,0] * (re_star[...,ax]+EW_star[...,ax] - EU) / inv_util_1d
                #print("foc_sep difference", np.max(np.abs(foc_sep-foc_sep_diff)))
                #NOTE: EW_star and re_star here are constant, not dependent on s at all (although they kinda should be)
                #Calculating the separations. Can drop this into jit surely
                for in0 in range(num_n):
                 if in0 == 0:
                  continue
                 for iz in range(num_z):
                    for in1 in range(num_n):
                     for iv in range(num_v):
                      for iq in range(num_q):
                        if np.all(foc_sep[iz,in0,in1,iv,iq,:] <= 0):
                           sep_star[iz,in0,in1,iv,iq] = 0
                        elif np.all(foc_sep[iz,in0,in1,iv,iq,:] > 0):
                           sep_star[iz,in0,in1,iv,iq] = 1
                        else:
                            #print("check, reached interpolation")
                            sep_star[iz,in0,in1,iv,iq] = interp(0,impose_increasing(-foc_sep[iz,in0,in1,iv,iq,:]),sep_grid) 
                return sep_star
class MultiworkerContract:
    """
        This solves a contract model with DRS production, hirings, and heterogeneous match quality.
    """
    def __init__(self, input_param=None, js=None):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
    
        self.log = logging.getLogger('MWF with Hiring')
        self.log.setLevel(logging.INFO)
        self.K = 2
        K = 2
        self.p = input_param
        self.deriv_eps = 1e-4 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid
        self.Q_grid = np.linspace(self.p.q_0,1,self.p.num_q) # Create worker productivity grid

        #Size grid:
        self.N_grid=np.linspace(0,self.p.n_bar,self.p.num_n)
        self.N_grid1 = np.copy(self.N_grid)
        self.N_grid1[0] = 1e-100 #So that it's not exactly zeor and I thus can keep my interpretation

        #self.N_grid=np.linspace(0,1,self.p.num_n)


        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        dimensions=[self.p.num_z]
        dimensions.extend([self.p.num_n] * K)
        dimensions.extend([self.p.num_v] * (K - 1))  
        dimensions.extend([self.p.num_q] * (K - 1))   
        self.J_grid   = np.zeros(dimensions) #grid of job values, first productivity, then size for each step, then value level for each step BESIDES FIRST
        # Production Function in the Model
        self.fun_prod_onedim = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = self.fun_prod_onedim.reshape((self.p.num_z,) + (1,) * (self.J_grid.ndim - 1))

        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = np.ones(self.p.num_x) * 0.5 * self.fun_prod.min()

        # Wage and Shadow Cost Grids
        self.w_grid = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v ) #Note that this is not the true range of possible wages as this excludes the size part of the story
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)


        #Total firm size for each possible state
        self.grid = np.ogrid[[slice(dim) for dim in self.J_grid.shape]]
        # Calculate the sum size for each element in the matrix
        self.sum_size = np.zeros(self.J_grid.shape) #Sum size
        self.sum_sizeadj = np.zeros(self.J_grid.shape) #Sum size ADJUSTED FOR QUALITY
        self.sum_wage=np.zeros(self.J_grid.shape) #Calculate the total wage paid for every state
        self.sum_size[...] = self.N_grid[self.grid[1]]
        self.sum_sizeadj[...] = self.N_grid[self.grid[1]] * (self.p.prod_q +self.p.q_0*(1.0-self.p.prod_q))
        for i in range(2, K + 1):
            self.sum_size += self.N_grid1[self.grid[i]]
            self.sum_sizeadj += self.N_grid1[self.grid[i]] * (self.p.prod_q + self.Q_grid[self.grid[self.J_grid.ndim - (K-1) + (i-2)]] * (1.0 - self.p.prod_q))
        for i in range(K+1,self.J_grid.ndim - (K-1)):
            self.sum_wage += self.w_grid[self.grid[i]]*self.N_grid1[self.grid[i-K+1]] #We add +1 because the wage at the very first step is semi-exogenous, and I will derive it directly

                

        #Setting up production grids
        self.prod = self.production(self.sum_sizeadj) #F = sum (n* (prod_q+q_1*(1-prod_q)))
        self.prod_diff = self.production_diff(self.sum_sizeadj)
        self.prod_1d = self.fun_prod_1d(self.sum_sizeadj)
        self.prod_nd = self.prod_1d * (self.p.prod_q + self.Q_grid[self.grid[4]] * (1.0-self.p.prod_q)) #\partial F / \partial n_1 = (prod_q+q_1*(1-prod_q)) F'(nq)
        self.prod_qd = self.prod_1d * self.N_grid1[self.grid[2]] * (1.0-self.p.prod_q) #\partial F / \partial q_1 = n_1 * (1-prod_q) * F'(nq)


        #Job value and GE first
        self.v_grid = np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod_onedim.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #Value promised to the worker at the bottom step
        self.v_0 = self.v_grid.min()#-1.0
        
        self.simple_J=np.divide(self.fun_prod_onedim[:,ax] - self.w_grid[ax,:],1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        #This is eqUvalent to marginal value of a firm of size 1 at the lowest step
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        

        if js is None:
            self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
            self.js.update(self.v_grid[:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        else:
            self.js = js       


        #Create a guess for the MWF value function
        #self.J_grid1 = self.J_grid1+np.divide(self.fun_prod*production(self.sum_size)-self.w_grid[0]*self.N_grid[ax,:,ax,ax]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #self.J_grid1 = np.zeros_like(self.J_grid)
        self.J_grid = self.J_grid+np.divide(self.fun_prod*self.prod-self.p.beta*self.w_grid[ax,ax,ax,:,ax]*self.N_grid[self.grid[1]]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #self.J_grid = self.J_grid+np.divide(self.fun_prod*self.prod-self.w_grid[0]*self.N_grid[self.grid[1]]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        
    
        #print("J_grid_diff:", np.max(abs(self.J_grid-self.J_grid1)))
        #The two methods are eqUvalent!! grid[1] really does capture the right value!!!


        #Guess for the Worker value function
        self.W = np.zeros_like(self.J_grid)
        self.W = np.expand_dims(self.W, axis=-1) #adding an extra dimension to W
        self.W = np.repeat(self.W, self.K, axis=-1)

        #Creating the wage matrix manually
        self.w_matrix = np.zeros(self.W.shape)
        self.w_matrix[...,0] = 0 #The workers at the bottom step will have special wages, derived endogenously through their PK
        #Actually, do I then need to add that step to the worker value? Not really, but useful regardless.
        # Can say that the bottom step really is step zero, with a fixed value owed to the worker.
        # And then all the actually meaningful steps are 1,2... etc, so when K=2 with just have 1 meaningful step            
        self.w_matrix[...,1] = self.w_grid[ax,ax,ax,:,ax]

        self.W[...,1] = self.W[...,1] + self.pref.utility(self.w_matrix[...,1])/(1-self.p.beta) #skip the first K-1 columns, as they don't correspond to the wage state. Then, pick the correct step, which is hidden in the last dimension of the grid
        self.W[...,0] = self.W[...,0] + self.pref.utility(self.unemp_bf.min())/(1-self.p.beta)

        #Setting up size and quality grids already in the matrix for
        self.size = np.zeros_like(self.W)
        self.size[...,0] = self.N_grid[self.grid[1]]
        for i in range(2,K + 1):
            self.size[...,i-1] = self.N_grid1[self.grid[i]]

        self.q = self.Q_grid[self.grid[4]]

    def J(self,Jg=None,Wg=None,update_eq=0):    
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
        size = self.size
        q = self.q

        if Jg is None:
            J = np.copy(self.J_grid)
        else:
            J = np.copy(Jg)
        if Wg is None:
            W = np.copy(self.W)
        else:
            W = np.copy(Wg)
        U = self.pref.utility(self.unemp_bf) / (1 - self.p.beta)
        Rho = J + size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]        

        print("Jshape", J.shape)
        print("W shape", W.shape)        
        # create representation for J1p
        #J1p = PowerFunctionGrid(W, J) #From valueFunction.py


        EW_star = np.copy(J)
        EJ_star = np.copy(J)
        EJderiv = np.zeros_like(J)
        EWderiv = np.zeros_like(J)
        #EW_tilde = np.copy(J)
        Jderiv = np.zeros_like(J)
        rho_star = np.zeros_like(J)
        sep_star = np.zeros_like(J)
        
        n0_star = np.zeros_like(J)      
        n1_star = np.zeros_like(J)   

        Rhoderiv = np.zeros_like(J)
        Wderiv = np.zeros_like(J)

        Rhod0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n)) #two extra size dimensions corresponding to future (arbitrary) sizes
        Wd0 = np.zeros_like(Rhod0)


        # prepare expectation call
        Ez = oe.contract_expression('anmvq,az->znmvq', J.shape, self.Z_trans_mat.shape)
        log_diff = np.zeros_like(EW_star)

        ite_num = 0
        error_js = 1
        
        # evaluate J1 tomorrow using our approximation
        #Jpi = J1p.eval_at_W1(W[...,1])
        for ite_num in range(self.p.max_iter):
            J2 = np.copy(J)
            W2 = np.copy(W)
            U2 = np.copy(U)
            Rho2 = np.copy(Rho)
            if ite_num>1:
             print("EJinv", EJinv[self.p.z_0-1,1,2,50, 0]/pc_star[self.p.z_0-1,1,2,50, 0])
             print("EJderiv", EJderiv[self.p.z_0-1,1,2,50, 0])
             j = np.where(N_grid==1)
             s = np.where(N_grid1==2)
             print("EJinv diff 1j 2s:", np.mean(np.abs((EJinv[:,j,s,:, 0]/pc_star[:,j,s,:, 0] - EJderiv[:,j,s,:, 0]) / EJderiv[:,j,s,:, 0])))
             print("EJinv diff 1 sen:", np.mean(np.abs((EJinv[:,0,1,:, 0]/pc_star[:,0,1,:, 0] - EJderiv[:,0,1,:, 0]) / EJderiv[:,0,1,:, 0])))
             print("EJinv diff 2 sen:", np.mean(np.abs((EJinv[:,0,s,:, 0]/pc_star[:,0,s,:, 0] - EJderiv[:,0,s,:, 0]) / EJderiv[:,0,s,:, 0])))


            # we compute the expected value next period by applying the transition rules
            EW = Ez(W[...,1], self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJ = Ez(J, self.Z_trans_mat)
            ERho = Ez(Rho, self.Z_trans_mat)

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

            # First boundary condition: forward difference            
            Rhoderiv[:, :, 0, ...] = (Rho[:, :, 1,  ...] - Rho[:, :, 0, ...]) / (N_grid1[1] - N_grid1[0])
            #Wderiv[:, :, 0, ...]     = (W[:, :, 1, :, :, 1] - W[:, :, 0, :, :, 1]) / (N_grid1[1] - N_grid1[0])
            # Last boundary condition: backward difference
            Rhoderiv[:, :, -1, ...] = Rho[:, :, -1,  ...] - Rho[:, :, -2,  ...]/ (N_grid1[-1] - N_grid1[-2])
            #Wderiv[:, :, -1, ...]     = W[:, :, -1, :, :, 1] - W[:, :, -2, :, :, 1]/ (N_grid1[-1] - N_grid1[-2])
            # Central differences: average of forward and backward differences
            Rhoderiv[:, :, 1:-1, ...] = (Rho[:, :, 2:,  ...] - Rho[:, :, :-2, ...]) / (N_grid1[ax, ax, 2:, ax, ax] - N_grid1[ax, ax, :-2, ax, ax])
            #Wderiv[:, :, 1:-1, ...]     = (W[:, :, 2:, :, :, 1] - W[:, :, :-2, :, :, 1]) / (N_grid1[ax, ax, 2:, ax, ax] - N_grid1[ax, ax, :-2, ax, ax])

            Jderiv = Rhoderiv-rho_grid[ax,ax,ax,:,ax]*W[...,1]
            #Jderiv = Rhoderiv+N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:, ax]*Wderiv #accounting for the fact that size change also impacts W

            #EJinv=(Jderiv+self.w_grid[ax,ax,ax,:]-self.fun_prod*self.prod_diff)/self.p.beta #creating expected job value as a function of today's value
            EJinv=(Jderiv+self.w_grid[ax,ax,ax,:, ax]-self.fun_prod*self.prod_nd)/self.p.beta #creating expected job value as a function of today's value            
            #EJinv[:,0,0,:] = (Jderiv[:,0,0,:]+self.w_grid[ax,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta
            
            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modified with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1

            #dim 0 is prod, dim 1 and 2 are size, dim 3 is future v, dim 4 is today's v, dim 5 is hmq
            foc = rho_grid[ax, ax, ax, :, ax, ax] - (EJinv[:, :, :, ax, :, :] / pc[...,ax,:])* (log_diff[...,ax,:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
            foc = foc*self.sum_size[..., ax, :] - size[..., ax, :,1]*rho_grid[ax, ax, ax, ax, :, ax] - size[..., ax, :,0]/self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW[..., ax, :]+re[..., ax, :]))
            assert (np.isnan(foc) & (pc[..., ax, :] > 0)).sum() == 0, "foc has NaN values where p>0"


            #Future senior wage
            rho_star = optimized_loop(
                pc, rho_grid, N_grid1, foc, rho_star, self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q) 
            
            #Future senior size
            pc_trans = np.moveaxis(pc,3,0)
            rho_trans = np.moveaxis(rho_star,3,0)
            
            n1_star = (size[...,0]*(1-sep_star)+size[...,1])*np.moveaxis(interp_multidim_extra_dim(rho_trans,rho_grid,pc_trans),0,3)
            q_star = (size[...,0]* np.minimum(self.p.q_0,1-sep_star)+q*size[...,1])/(size[...,0]*(1-sep_star)+size[...,1])
            
            #Getting hiring decisions
            n0_star[...] = 0
            for iz in range(self.p.num_z):
                for in00 in range(self.p.num_n):

                    Rho_interpolator = RegularGridInterpolator((N_grid1, rho_grid, Q_grid), ERho[iz, in00, ...], bounds_error=False, fill_value=None)
                    #W_interpolator = RegularGridInterpolator((N_grid1, rho_grid, Q_grid), EW[iz, in00, ...], bounds_error=False, fill_value=None)
                    Rhod0[iz, ..., in00] = Rho_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                    #Wd0[iz, ..., in00] = W_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
            if ite_num >= 0:
                #Ihire = ((Jd0[...,1]-Jd0[...,0]+rho_star*n1_star*(Wd0[...,1]-Wd0[...,0])) > self.p.hire_c/self.p.beta) & (N_grid[self.grid[1]]+N_grid1[self.grid[2]] < self.p.n_bar )
                Ihire = ((Rhod0[...,1]-Rhod0[...,0]) / (N_grid[1]-N_grid[0]) > self.p.hire_c/self.p.beta) & (size[...,0]+size[...,1] < self.p.n_bar)
                n0_star = n0(Rhod0, n0_star, N_grid, Ihire, self.p.hire_c / self.p.beta)



            #Future optimal expectations
            ERho_star = interp_multidim(n0_star,N_grid,np.moveaxis(Rhod0,-1,0))
            #EW_star = interp_multidim(n0_star,N_grid,np.moveaxis(Wd0,-1,0))
            for iz in range(self.p.num_z):            
                EJ_star[iz,...] = RegularGridInterpolator((N_grid,N_grid1, rho_grid, Q_grid), EJ[iz, ...], bounds_error=False, fill_value=None) ((n0_star[iz,...],n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                EW_star[iz,...] = RegularGridInterpolator((N_grid,N_grid1, rho_grid, Q_grid), EW[iz, ...], bounds_error=False, fill_value=None) ((n0_star[iz,...],n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
            #EJ_star = ERho_star - rho_star * n1_star * EW_star


            #Getting the derivative of the future job value wrt n1:
            floorn1=np.floor(np.interp( n1_star, N_grid1, range(self.p.num_n))).astype(int)
            ceiln1=np.ceil(np.interp( n1_star, N_grid1, range(self.p.num_n))).astype(int)            
            for iz in range(self.p.num_z):
                for in11 in range(self.p.num_n): 
                    
                    Rho_interpolator = RegularGridInterpolator((N_grid, rho_grid, Q_grid), ERho[iz, :, in11, ...], bounds_error=False, fill_value=None)
                    #W_interpolator = RegularGridInterpolator((N_grid, rho_grid, Q_grid), EW[iz, :, in11, ...], bounds_error=False, fill_value=None)
                    Rhod0[iz, ..., in11] = Rho_interpolator((n0_star[iz, ...], rho_star[iz,...], q_star[iz, ...]))
                    #Wd0[iz, ..., in11] = W_interpolator((n0_star[iz, ...], rho_star[iz,...], q_star[iz, ...]))
            ERhoderiv = ERhoDerivative(Rhod0,Wd0,ceiln1,floorn1,n1_star,rho_star,N_grid1,self.p.num_z,self.p.num_n,self.p.n_bar,self.p.num_v,self.p.num_q)
            EJderiv = ERhoderiv - rho_star * EW_star
            
            assert np.isnan(EW_star).sum() == 0, "EW_star has NaN values"

            _, re_star, pc_star = self.getWorkerDecisions(EW_star)

            _, ru, _ = self.getWorkerDecisions(EU, employed=False)
            U = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EU)
            U = 0.4 * U + 0.6 * U2

            # Update firm value function 
            J= self.fun_prod*self.prod - sum_wage - self.p.hire_c * n0_star - \
                self.pref.inv_utility(self.v_0-self.p.beta*(EW_star+re_star))*size[...,0]  + self.p.beta * EJ_star
            #Update the dual value function rho
            Rho = self.fun_prod*self.prod - sum_wage - self.p.hire_c * n0_star - \
                self.pref.inv_utility(self.v_0-self.p.beta*(EW_star+re_star))*size[...,0] + \
                rho_grid[ax,ax,ax,:,ax]*size[...,1]*W[...,1] + self.p.beta * (ERho_star - rho_star*n1_star*EW_star)
            # Update worker value function
            W[...,1] = self.pref.utility(self.w_matrix[...,1]) + \
                self.p.beta * (EW_star + re_star) #For more steps the ax at the end won't be needed as EW_star itself will have multiple steps
            Rho_alt = J+size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]

            # Updating Jrepresentation
            #J= Rho - size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]
            #W[...,1] = W[...,1] * (J>= 0) + U * (J< 0)
            #Ji[J< 0] = 0
            comparison_range = (size[...,0]+size[...,1] <= self.p.n_bar) & (size[...,0]+size[...,1] >= N_grid[1])
            print("Diff Rho:", np.mean(np.abs((Rho_alt[comparison_range]-Rho[comparison_range])/Rho[comparison_range])))
            #print("Max diff point", np.where(np.abs((Rho_alt-Rho)/Rho)==np.max(np.abs((Rho_alt-Rho)/Rho))))

            Rho = .2 * Rho + .8 * Rho2
            J= .2 * J + .8 * J2
            W[...,1:] = .4 * W[...,1:] + .6 * W2[...,1:] #we're completely ignoring the 0th step
            #Rho = J+ size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]


            #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[:,:,:,:,1], J)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Rho,Rho2,100) #np.power(J- Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_w1 = array_dist(W[...,1:], W2[...,1:])

            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    print("Errors:",  error_j1i, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            #P_xv = self.matching_function(J1p.eval_at_W1(W)[self.p.z_0-1, 0, 1, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            #error_js = self.js.update(W[self.p.z_0-1, 0, 0, :, 1], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[...,1], J)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    print("Errors:",  error_j1i,  error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break


        return J,W,Rho,EW_star,pc_star,n0_star, n1_star

    def J_sep_dir(self,Jg=None,Wg=None,update_eq=0):    
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
        size = self.size
        q = self.q

        if Jg is None:
            J= np.copy(self.J_grid)
        else:
            J= np.copy(Jg)
        if Wg is None:
            W = np.copy(self.W)
        else:
            W = np.copy(Wg)
        U = self.pref.utility_gross(self.unemp_bf)/(1-self.p.beta)
        Rho = J+ size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]        

        print("Jshape", J.shape)
        print("W shape", W.shape)        
        # create representation for J1p
        #J1p = PowerFunctionGrid(W, J) #From valueFunction.py


        EW_star = np.copy(J)
        EJ_star = np.copy(J)
        EJderiv = np.zeros_like(J)
        Jderiv = np.zeros_like(J)
        rho_star = np.zeros_like(J)
        sep_star = np.zeros_like(J)
        
        n0_star = np.zeros_like(J)      
        n1_star = np.zeros_like(J)   

        Rhoderiv = np.zeros_like(J)

        Rhod0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
        Wd0 = np.zeros_like(Rhod0)

        #Separations related variables
        sep_grid = np.linspace(0,1,20)
        n1_s = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, sep_grid.shape[0]))
        q_s = np.zeros_like(n1_s)
        q_deriv_s = np.zeros_like(n1_s)
        J_fut_deriv_n = np.zeros_like(n1_s)
        J_fut_deriv_q = np.zeros_like(n1_s)

        # prepare expectation call
        Ez = oe.contract_expression('anmvq,az->znmvq', J.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', U.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW_star)

        ite_num = 0
        error_js = 1
        
        # evaluate J1 tomorrow using our approximation
        #Jpi = J1p.eval_at_W1(W[:,:,:,:,1])

        for ite_num in range(self.p.max_iter):
            J2 = np.copy(J)
            W2 = np.copy(W)
            U2 = np.copy(U)
            Rho2 = np.copy(Rho)

            if ite_num>1:
             print("EJinv", EJinv[self.p.z_0-1,1,2,50, 0]/pc_star[self.p.z_0-1,1,2,50, 0])
             print("EJderiv", EJderiv[self.p.z_0-1,1,2,50, 0])
             j = np.where(N_grid==1)
             s = np.where(N_grid1==2)
             print("EJinv diff 1j 2s:", np.mean(np.abs((EJinv[:,j,s,:, 0]/pc_star[:,j,s,:, 0] - EJderiv[:,j,s,:, 0]) / EJderiv[:,j,s,:, 0])))
             print("EJinv diff 1 sen:", np.mean(np.abs((EJinv[:,0,1,:, 0]/pc_star[:,0,1,:, 0] - EJderiv[:,0,1,:, 0]) / EJderiv[:,0,1,:, 0])))
             print("EJinv diff 2 sen:", np.mean(np.abs((EJinv[:,0,s,:, 0]/pc_star[:,0,s,:, 0] - EJderiv[:,0,s,:, 0]) / EJderiv[:,0,s,:, 0])))

            # we compute the expected value next period by applying the transition rules
            EW = Ez(W[...,1], self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJ = Ez(J, self.Z_trans_mat)
            ERho = Ez(Rho, self.Z_trans_mat)
            EU = U

            # get worker decisions
            _, re, pc = self.getWorkerDecisions(EW)
            # get worker decisions at EW + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW + self.deriv_eps) 
           
            if ite_num>100000000:
                EW1_tild = EW_tild(n1_star,EW,N_grid,self.p.num_z,self.p.num_n,self.p.num_v)
                _, re, pc = self.getWorkerDecisions(EW1_tild)
                # get worker decisions at EW + epsilon
                _, _, pc_d = self.getWorkerDecisions(EW1_tild+self.deriv_eps)
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff = np.zeros_like(pc)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value

            
            #Calculating all the value function derivatives (manually of course)
            #Rho = J+ size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1] #This is the full rho

            # First boundary condition: forward difference            
            Rhoderiv[:, :, 0, ...] = (Rho[:, :, 1,  ...] - Rho[:, :, 0, ...]) / (N_grid1[1] - N_grid1[0])
            # Last boundary condition: backward difference
            Rhoderiv[:, :, -1, ...] = Rho[:, :, -1,  ...] - Rho[:, :, -2,  ...]/ (N_grid1[-1] - N_grid1[-2])
            # Central differences: average of forward and backward differences
            Rhoderiv[:, :, 1:-1, ...] = (Rho[:, :, 2:,  ...] - Rho[:, :, :-2, ...]) / (N_grid1[ax, ax, 2:, ax, ax] - N_grid1[ax, ax, :-2, ax, ax])
            
            Jderiv = Rhoderiv-rho_grid[ax,ax,ax,:,ax]*W[...,1]
            #Jderiv = Rhoderiv+size[...,1]*rho_grid[ax,ax,ax,:, ax]*Wderiv #accounting for the fact that size change also impacts W
            #Jderiv0 = Jderiv0+size[...,1]*rho_grid[ax,ax,ax,:]*Wderiv0 #accounting for the fact that size change also impacts W

            EJinv=(Jderiv+self.w_grid[ax,ax,ax,:, ax]-self.fun_prod*self.prod_nd)/self.p.beta #creating expected job value as a function of today's value            
            #if ite_num>1: #I'm using previous guesses for sep_star and EW_star. This way, it is still as if EJinv0 is a function of today's states only, even though that's not exactly correct
            # EJinv0 = (Jderiv0+wage_jun- (self.p.prod_q + self.p.q_0 * (1-self.p.prod_q)) * self.fun_prod*self.prod_1d)/self.p.beta
             #EJinv0[:,0,0,:] = (Jderiv0[:,0,0,:]+wage_jun[:,0,0,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta


            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modified with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1

             #Main foc, in the absence for separations
            inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[...,ax,:,:]*EU+(1-sep_star[...,ax,:,:])*(EW[..., ax, :]+re[..., ax, :])))
            foc_2ndpart = - size[..., ax, :, 1]*rho_grid[ax, ax, ax, ax, :, ax] -\
                 size[..., ax, :,0] * (1-sep_star[...,ax,:,:]) / inv_utility_1d             
             
            foc = rho_grid[ax, ax, ax, :, ax, ax] - (EJinv[:, :, :, ax, :, :] / pc[...,ax,:])* (log_diff[...,ax,:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
            foc = foc*(size[..., ax, :,0] * (1-sep_star[...,ax,:,:])+size[..., ax, :,1]) + foc_2ndpart
            #There are no separations here as this FOC is in the case of NO separations (although, what if s=1?)
            #Foc for wages if separations are positive
            # foc_rho_s = rho_grid[ax, ax, ax, :,ax, ax]+((EW[..., ax, :]+re[..., ax, :]-EU) / inv_utility_1d)*(log_diff[..., ax,:] / self.deriv_eps)/(pc[..., ax,:])
            # foc_rho_s =  foc_rho_s*(N_grid[self.grid[1][..., ax,:]] * (1-sep_star[...,ax,:,:]) + N_grid1[self.grid[2][..., ax,:]]) + foc_2ndpart

            
            
            assert (np.isnan(foc) & (pc[..., ax, :] > 0)).sum() == 0, "foc has NaN values where p>0"


            #assert np.all(EW[iz, in0, in1, 1:] >= EW[iz, in0, in1, :-1]) #Andrei: check that worker value is increasing in v
            #if ite_num<=1:
            rho_star = optimized_loop(
                pc, rho_grid, N_grid1, foc, rho_star, self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q) 
            
            #MINIMUM WAGE ADDITION! DON'T ALLOW RHO_GRID TO GO BELOW RELATED MIN WAGE
            #Would this work tho? I dunno if it's the correct way of doing this... anyway let's try
            #rho_cutoff = interp(self.p.min_wage,self.w_grid,rho_grid)
            #rho_star[rho_star < rho_cutoff] = rho_cutoff

            pc_trans = np.moveaxis(pc,3,0)
            rho_trans = np.moveaxis(rho_star,3,0)            
            pc_temp = np.moveaxis(interp_multidim_extra_dim(rho_trans,rho_grid,pc_trans),0,3)
            #Diff sep approach: interpolate the whole fucking foc!!!

            sep_star[...] = 0
            
            if ite_num>0:
                

                #WHAT IF. We just do a direct derivative wrt s??? Like, we know what q_s and n1_s are. Inteprolate directly onto them, which will already give us the total derivative of J wrt s, no?
                for s in range(sep_grid.shape[0]):
                    n1_s[...,s] = (size[...,0]*(1-sep_grid[s])+size[...,1]) * pc_temp
                    q_s[...,s] = (size[...,0] * np.minimum(self.p.q_0,1-sep_grid[s])+self.q*size[...,1]) / (size[...,0]*(1-sep_grid[s])+size[...,1])
                
                J_s = np.zeros_like(n1_s)
                for iz in range(self.p.num_z):
                #       J_s[iz,...,s] = RegularGridInterpolator((N_grid, N_grid1, rho_grid, Q_grid), EJ[iz, ...], bounds_error=False, fill_value=None) ((n0_star[iz, ...], n1_s[iz,...,s], rho_star[iz, ...], q_s[iz, ...,s]))
                        J_s[iz,...] = RegularGridInterpolator((N_grid, N_grid1, rho_grid, Q_grid), ERho[iz, ...], bounds_error=False, fill_value=None) ((n0_star[iz, ...,ax], n1_s[iz,...], rho_star[iz, ...,ax], q_s[iz, ...]))

                sep_reshaped = sep_grid.reshape((1,) * (J.ndim) + (-1,))

                J_s_deriv = np.zeros_like(J_s)
                J_s_deriv[...,0] = (J_s[...,1] - J_s[...,0]) / (sep_grid[1] - sep_grid[0])
                J_s_deriv[...,-1] = (J_s[...,-1] - J_s[...,-2]) / (sep_grid[-1] - sep_grid[-2]) 
                J_s_deriv[..., 1:-1]    = (J_s[...,2:] - J_s[...,:-2]) / (sep_reshaped[...,2:] - sep_reshaped[...,:-2]) 
                #J_s_deriv[..., 1:-1]    = (J_s[...,1:-1] - J_s[...,0:-2]) / (sep_reshaped[...,1:-1] - sep_reshaped[...,0:-2])
                #Maybe make it all be forward difference??? That way we should actually have seprations plateau exactly at 0.5, no?
                #Gott be somewhat careful around the borders no tho

                inv_util_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_reshaped * EU[...,ax] + (1-sep_reshaped) * (EW_star[...,ax] + re_star[...,ax])))
                sep_star = sep_solve_2(sep_star,J_fut_deriv_n,J_fut_deriv_q,J_s_deriv,q_deriv_s,pc_temp,inv_util_1d,re_star,EW_star,EU,sep_grid,size,self.p.num_z,self.p.num_v,self.p.num_n,self.p.num_q)
                print("sep borders", sep_star.min(),sep_star.max())

            #Getting n1_star
            n1_star = (size[...,0]*(1-sep_star)+size[...,1])*pc_temp
            q_star = (size[...,0]* np.minimum(self.p.q_0,1-sep_star)+q*size[...,1])/(size[...,0]*(1-sep_star)+size[...,1])
            print("q_star", q_star[self.p.z_0-1,1,0,50, :])
            


            #Getting hiring decisions
            n0_star[...] = 0
            for iz in range(self.p.num_z):
                for in00 in range(self.p.num_n):

                    Rho_interpolator = RegularGridInterpolator((N_grid1, rho_grid, Q_grid), ERho[iz, in00, ...], bounds_error=False, fill_value=None)
                    Rhod0[iz, ..., in00] = Rho_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
            if ite_num >= 0:
                Ihire = ((Rhod0[...,1]-Rhod0[...,0]) / (N_grid[1]-N_grid[0]) > self.p.hire_c/self.p.beta) & (size[...,0]+size[...,1] < self.p.n_bar)
                n0_star = n0(Rhod0, n0_star, N_grid, Ihire, self.p.hire_c / self.p.beta)



            #Future optimal expectations
            ERho_star = interp_multidim(n0_star,N_grid,np.moveaxis(Rhod0,-1,0))
            for iz in range(self.p.num_z):            
                EJ_star[iz,...] = RegularGridInterpolator((N_grid,N_grid1, rho_grid, Q_grid), EJ[iz, ...], bounds_error=False, fill_value=None) ((n0_star[iz,...],n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                EW_star[iz,...] = RegularGridInterpolator((N_grid,N_grid1, rho_grid, Q_grid), EW[iz, ...], bounds_error=False, fill_value=None) ((n0_star[iz,...],n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))


            #Getting the derivative of the future job value wrt n1:
            floorn1=np.floor(np.interp( n1_star, N_grid1, range(self.p.num_n))).astype(int)
            ceiln1=np.ceil(np.interp( n1_star, N_grid1, range(self.p.num_n))).astype(int)            
            for iz in range(self.p.num_z):
                for in11 in range(self.p.num_n): 
                    Rho_interpolator = RegularGridInterpolator((N_grid, rho_grid, Q_grid), ERho[iz, :, in11, ...], bounds_error=False, fill_value=None)
                    Rhod0[iz, ..., in11] = Rho_interpolator((n0_star[iz, ...], rho_star[iz,...], q_star[iz, ...]))
            ERhoderiv = ERhoDerivative(Rhod0,Wd0,ceiln1,floorn1,n1_star,rho_star,N_grid1,self.p.num_z,self.p.num_n,self.p.n_bar,self.p.num_v,self.p.num_q)
            EJderiv = ERhoderiv - rho_star * EW_star

            assert np.isnan(EW_star).sum() == 0, "EW_star has NaN values"

            _, re_star, pc_star = self.getWorkerDecisions(EW_star)
            #print("states at which worker qUts:", np.where(~(pc_star[self.p.z_0-1,1,1,:]==0)))
            
            _, ru, _ = self.getWorkerDecisions(EU, employed=False)
            U = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EU)
            U = 0.4 * U + 0.6 * U2

            # Update firm value function 
            wage_jun = self.pref.inv_utility(self.v_0-self.p.beta*(sep_star*EU+(1-sep_star)*(EW_star+re_star)))
            J= self.fun_prod*self.prod - sum_wage - self.p.hire_c*n0_star - \
                wage_jun*size[...,0]  + self.p.beta * EJ_star
            
            Rho = self.fun_prod*self.prod - sum_wage - self.p.hire_c * n0_star - \
                wage_jun*size[...,0] + \
                rho_grid[ax,ax,ax,:,ax]*size[...,1]*W[...,1] + self.p.beta * (ERho_star - rho_star*n1_star*EW_star)
            Rho_alt = J + size[...,1]*rho_grid[ax,ax,ax,:,ax]*W[...,1]            
            
            # Update worker value function
            W[...,1] = self.pref.utility(self.w_matrix[...,1]) + \
                self.p.beta * (EW_star + re_star) #For more steps the ax at the end won't be needed as EW_star itself will have multiple steps
            

            comparison_range = (size[...,0]+size[...,1] <= self.p.n_bar) & (size[...,0]+size[...,1] >= N_grid[1])
            print("Diff Rho:", np.mean(np.abs((Rho_alt[comparison_range]-Rho[comparison_range])/Rho[comparison_range])))

            Rho = .2 * Rho + .8 * Rho2
            J= .2 * J + .8 * J2
            W[...,1:] = .4 * W[...,1:] + .6 * W2[...,1:] #we're completely ignoring the 0th step


            # Updating J1 representation
            #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[:,:,:,:,1], J)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Rho,Rho2,100) #np.power(J- Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_w1 = array_dist(W[...,1:], W2[...,1:])

            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    print("Errors:",  error_j1i, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            #P_xv = self.matching_function(J1p.eval_at_W1(W)[self.p.z_0-1, 0, 1, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            #error_js = self.js.update(W[self.p.z_0-1, 0, 0, :, 1], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[...,1], J)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                    print("Errors:",  error_j1i,  error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            #if (ite_num % 25) == 0:
                # Updating J1 representation
                #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W[...,1], J)
                #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W[...,1]), 100)
                #print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)    
                #self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                #                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        #self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
        #                             ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        self.append_results_to_pickle(J, W, EW_star, sep_star, n0_star, n1_star)

        return J,W,Rho,EW_star,sep_star, n0_star, n1_star




    def append_results_to_pickle(self, J, W, EW_star, sep_star, n0_star, n1_star, pickle_file="results_Rho_sep.pkl"):
        # Step 1: Load the existing data from the pickle file
        try:
            with open(pickle_file, "rb") as file:
                all_results = pickle.load(file)
        except FileNotFoundError:
            all_results = {}
            print("No existing file found. Creating a new one.")

        # Step 2: Create results for the multi-dimensional p
        new_results = self.save_results_for_p(J, W, EW_star, sep_star, n0_star, n1_star)

        # Step 3: Use a tuple (p.num_z, p.num_v, p.num_n) as the key
        key = (self.p.num_z,self.p.num_v,self.p.num_n,self.p.n_bar,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.hire_c,self.p.prod_alpha,self.p.dt)

        # Step 4: Add the new results to the dictionary
        all_results[key] = new_results

        # Step 5: Save the updated dictionary back to the pickle file
        with open(pickle_file, "wb") as file:
            pickle.dump(all_results, file)

        print(f"Results for p = {key} have been appended to {pickle_file}.")
    def save_results_for_p(self, J, W, EW_star, sep_star, n0_star, n1_star):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
        'date': current_date,
        'J': J,
        'W': W,
        'EW_star': EW_star,
        'sep_star': sep_star,
        'n0_star': n0_star,
        'n1_star': n1_star,
        'p_value': (self.p.num_z,self.p.num_v,self.p.num_n,self.p.n_bar,self.p.num_q,self.p.q_0,self.p.prod_q,self.p.hire_c,self.p.prod_alpha,self.p.dt)
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
    def matching_function(self,J1): #Andrei: the formula of their matching function, applied to each particula job value J1
        return self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(J1, self.p.kappa)), self.p.sigma),
                                1 / self.p.sigma)

#from primitives import Parameters
#p = Parameters()
#from ContinuousContract import ContinuousContract
#cc=ContinuousContract(p)
#(cc_J,cc_W,cc_Wstar,cc_Jpi,cc_pc)=cc.J(1)
#mwc_hmq=MultiworkerContract(p,cc.js)
#(mwc_hmq_sd_J,mwc_hmq_sd_W,mwc_hmq_sd_Wstar,mwc_hmq_sd_sep,mwc_hmq_sd_n0,mwc_hmq_sd_n1)=mwc_hmq.J()