import numpy as np
import logging
from scipy.stats import lognorm as lnorm

import opt_einsum as oe

from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search import JobSearchArray
from valuefunction_multi import PowerFunctionGrid
from scipy.interpolate import RegularGridInterpolator
from numba import jit

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


@jit(nopython=True)
def impose_decreasing(M):
    nv = M.shape[1]
    if len(M.shape)==2:
        for v in reversed(range(nv-1)):
            M[:,v] = np.maximum(M[:,v],M[:,v+1]+1e-10)
    else:
        for v in reversed(range(nv-1)):
            M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
    return M
@jit(nopython=True, cache=True)
def impose_increasing(A0):
    A = np.copy(A0)
    nv = len(A)
    for v in range(1,nv):
        A[v] = np.maximum(A[v],A[v-1])
    return A
#@jit(nopython=True)
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
#@jit(nopython=True)
def array_dist(A,B):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    return  (np.power( A-B,2) ).mean() / ( np.power(B,2) ).mean()
#Solve for rho_star
@jit(nopython=True, cache=True)
def optimized_loop(pc, rho_grid, N_grid1, foc, rho_star, num_z, num_n, n_bar, num_q):
    for iz in range(num_z):
        for in0 in range(num_n): #Not this: we don't do the case for max juniors. for some reason separations fail otherwise
            for in1 in range(num_n):
             if (N_grid1[in0] + N_grid1[in1] > n_bar):
              continue
             for iq in range(num_q):
                rho_min = np.min(rho_grid[pc[iz, in0, in1, :, iq] > 0])  # Lowest promised rho with continuation > 0
                Isearch = (pc[iz, in0, in1, :, iq] > 0)
                
                if np.any(Isearch):
                    Isearch_indices = np.where(Isearch)[0]
                    for iv in Isearch_indices:
                        rho_star[iz, in0, in1, iv,iq] = np.interp(
                            0, impose_increasing(foc[iz, in0, in1, Isearch, iv, iq]), rho_grid[Isearch]
                        )
                
                Iquit = ~(pc[iz, in0, in1, :, iq] > 0)
                if np.any(Iquit):
                    rho_star[iz, in0, in1, Iquit, iq] = rho_min

    return rho_star
@jit(nopython=True, cache=True) #To be done: corerect the inv_utility issue, it doesn't work with numba!
def optimized_loop_sep(rho_grid, foc, rho_star, sep_star, num_z, num_n, num_v, num_q):
    for iz in range(num_z):
        for in0 in range(num_n - 1):
            for in1 in range(num_n):
                if (in0 +in1 > num_n-1):
                    continue
                for iv in range(num_v):
                 for iq in range(num_q):
                  rho_star[iz,in0, in1, iv,iq] = np.interp(0,
                                                    impose_increasing(foc[iz, in0, in1, :, iv,iq]),
                                                    rho_grid[:])                  
    return rho_star
@jit(nopython=True, cache=True)
def optimized_loop_tilde(pc, rho_grid, foc, rho_star, num_z, num_n, num_v, num_q):
    for iz in range(num_z):
        for in0 in range(num_n - 1):
            for in1 in range(num_n):
                if (in0 + in1 > num_n - 1):
                    continue
                for iv in range(num_v):
                 for iq in range(num_q):
                    #rho_star[iz, in0, in1, iv,iq] = np.interp(
                    #        0, impose_increasing(foc[iz, in0, in1, :, iv,iq]), rho_grid[:]
                    #    )
                    rho_min = np.min(rho_grid[pc[iz, in0, in1, :, iv,iq] > 0])  # Lowest promised rho with continuation > 0
                    Isearch = (pc[iz, in0, in1, :, iv,iq] > 0)
                
                    if np.any(Isearch):
                        Isearch_indices = np.where(Isearch)[0]
                        for iv in Isearch_indices:
                            rho_star[iz, in0, in1, iv,iq] = np.interp(
                                0, impose_increasing(foc[iz, in0, in1, Isearch, iv,iq]), rho_grid[Isearch]
                                )
                
                    Iquit = ~(pc[iz, in0, in1, :, iv,iq] > 0)
                    if np.any(Iquit):
                        rho_star[iz, in0, in1, Iquit] = rho_min
    return rho_star
#Given rho_star, find n1_star
@jit(nopython=True, cache=True)
def n1(pc, rho_grid, rho_star, sep_star, N_grid, N_grid1, num_z, num_n, num_v, num_q):
    n1 = np.zeros((num_z, num_n, num_n, num_v, num_q))
    for iz in range(num_z):
     for in0 in range(num_n):
        for in1 in range(num_n):
         for iq in range(num_q):
            n1[iz, in0, in1, :, iq] = (N_grid[in0]*(1-sep_star[iz,in0,in1,:, iq])+N_grid1[in1])*np.interp(rho_star[iz, in0, in1, :, iq], rho_grid, pc[iz,in0,in1,:,iq])
    return n1
@jit(nopython=True, cache=True)
def n1_tilde(n1,pc,rho_grid,rho_star, sep_star, N_grid,num_z, num_n, num_v, num_q):
    for iz in range(num_z):
     for in0 in range(num_n):
      for in1 in range(num_n):         
        for iv in range(num_v):
         for iq in range(num_q):
            n1[iz, in0, in1, :, iq] = (N_grid[in0]*(1-sep_star[iz,in0,in1,:, iq])+N_grid[in1])*np.interp(rho_star[iz, in0, in1, iv,iq], rho_grid, pc[iz,in0,in1,:,iv,iq])
    return n1
#Given rho_star and n1_star, calculate the future derivative
@jit(nopython=True, cache=True)
def EJderivative(EJpi,floorn1,ceiln1, n0_star,Ederiv,rho_grid, N_grid, N_grid1,rho_star,num_z, num_n, n_bar, num_v, num_q):
        EJc= np.zeros((num_z,num_n,num_n,num_v,num_v, num_q))
        EJf= np.zeros((num_z,num_n,num_n,num_v,num_v, num_q))           
        for iz in range(num_z):
            for in0 in range(num_n):
                for in1 in range(num_n):
                    for iv in range(num_v):
                        for iq in range(num_q):
                         if ceiln1[iz,in0,in1,iv,iq]==0:
                            continue
                         if N_grid1[floorn1[iz,in0,in1,iv,iq]]>=n_bar:
                            continue
                         for iv1 in range(num_v):
                           EJc[iz,in0,in1,iv,iv1,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,EJpi[iz,:,ceiln1[iz,in0,in1,iv,iq],iv1,iq]) #We first interpolate future value to the correct size at step 0. Then we compute the derivative wrt size at step 1 around the optimal size and shadow cost
                           EJf[iz,in0,in1,iv,iv1,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,EJpi[iz,:,floorn1[iz,in0,in1,iv,iq],iv1,iq])
                         Ederiv[iz,in0,in1,iv,iq] = (np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EJc[iz,in0, in1, iv, :,iq])-np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EJf[iz,in0, in1, iv, :,iq]))/(N_grid1[ceiln1[iz,in0,in1,iv,iq]]-N_grid1[floorn1[iz,in0,in1,iv,iq]])
        return Ederiv
@jit(nopython=True, cache=True)
def EJderivative3(EJpi,EW1i, floorn1,ceiln1,Ederiv,rho_grid,rho_star,num_z, num_n, num_v, num_q):
            for iz in range(num_z):
                for in0 in range(num_n):
                    for in1 in range(num_n):
                        for iv in range(num_v):
                         for iq in range(num_q):
                          if ceiln1[iz,in0,in1,iv,iq]==0:
                            continue
                          if floorn1[iz,in0,in1,iv,iq]>=num_n-1:
                            continue
                          Ederiv[iz,in0,in1,iv,iq] = (np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EJpi[iz,0, ceiln1[iz,in0,in1,iv,iq],:,iq])+ceiln1[iz,in0,in1,iv,iq]*rho_star[iz,in0,in1,iv,iq]*np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EW1i[iz,0, ceiln1[iz,in0,in1,iv,iq],:,iq])- \
                                                  (np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EJpi[iz,0, floorn1[iz,in0,in1,iv,iq],:,iq]))+floorn1[iz,in0,in1,iv,iq]*rho_star[iz,in0,in1,iv,iq]*np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EW1i[iz,0, floorn1[iz,in0,in1,iv,iq],:,iq])) \
                                                    /(ceiln1[iz,in0,in1,iv,iq]-floorn1[iz,in0,in1,iv,iq])
            return Ederiv
@jit(nopython=True, cache=True)
def EWderivative(EW1i,floorn1,ceiln1, n0_star,Ederiv,rho_grid, N_grid, N_grid1,rho_star,num_z, num_n, n_bar, num_v, num_q):
        EWc= np.zeros((num_z,num_n,num_n,num_v,num_v,num_q))
        EWf= np.zeros((num_z,num_n,num_n,num_v,num_v,num_q))
        for iz in range(num_z):
            for in0 in range(num_n):
                for in1 in range(num_n):
                    for iv in range(num_v):
                        for iq in range(num_q):
                         if ceiln1[iz,in0,in1,iv,iq]==0:
                            continue
                         if N_grid1[floorn1[iz,in0,in1,iv,iq]]>=n_bar:
                            continue
                         for iv1 in range(num_v):
                           EWc[iz,in0,in1,iv,iv1,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,EW1i[iz,:,ceiln1[iz,in0,in1,iv,iq],iv1,iq]) #We first interpolate future value to the correct size at step 0. Then we compute the derivative wrt size at step 1 around the optimal size and shadow cost
                           EWf[iz,in0,in1,iv,iv1,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,EW1i[iz,:,floorn1[iz,in0,in1,iv,iq],iv1,iq])
                         Ederiv[iz,in0,in1,iv,iq] = (np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EWc[iz,in0, in1,iv, :,iq])-np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,EWf[iz,in0, in1, iv, :,iq]))/(N_grid1[ceiln1[iz,in0,in1,iv,iq]]-N_grid1[floorn1[iz,in0,in1,iv,iq]])
        return Ederiv
#Gives us future worker value as a function of promised value, but with updated size, taken based on a guess of n1_star

def RhoQderiv(EJpi,floorn1,ceiln1, n0_star,Ederiv,rho_grid, N_grid, N_grid1,rho_star,num_z, num_n, n_bar, num_v, num_q):
        ERc= np.zeros((num_z,num_n,num_n,num_v,num_v, num_q))
        ERf= np.zeros((num_z,num_n,num_n,num_v,num_v, num_q))           
        for iz in range(num_z):
            for in0 in range(num_n):
                for in1 in range(num_n):
                    for iv in range(num_v):
                        for iq in range(num_q):
                         if ceiln1[iz,in0,in1,iv,iq]==0:
                            continue
                         if N_grid1[floorn1[iz,in0,in1,iv,iq]]>=n_bar:
                            continue
                         for iv1 in range(num_v):
                           ERc[iz,in0,in1,iv,iv1,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,EJpi[iz,:,ceiln1[iz,in0,in1,iv,iq],iv1,iq]) #We first interpolate future value to the correct size at step 0. Then we compute the derivative wrt size at step 1 around the optimal size and shadow cost
                           ERf[iz,in0,in1,iv,iv1,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,EJpi[iz,:,floorn1[iz,in0,in1,iv,iq],iv1,iq])
                         Ederiv[iz,in0,in1,iv,iq] = (np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,ERc[iz,in0, in1, iv, :,iq])-np.interp(rho_star[iz,in0,in1,iv,iq],rho_grid,ERf[iz,in0, in1, iv, :,iq]))/(N_grid1[ceiln1[iz,in0,in1,iv,iq]]-N_grid1[floorn1[iz,in0,in1,iv,iq]])
        return Ederiv
@jit(nopython=True, cache=True, parallel=True)
def EW_tild(n1_star,EW1i,N_grid,num_z,num_n,num_v, num_q):
    EW_tild = np.zeros((num_z, num_n, num_n, num_v, num_v))
    for iz in range(num_z):
     for in0 in range(num_n):
      for in1 in range(num_n):
       for iv in range(num_v):
        for iv1 in range(num_v):
         for iq in range(num_q):
            EW_tild[iz,in0,in1,iv1,iv,iq] = np.interp(n1_star[iz,in0,in1,iv,iq],N_grid,EW1i[iz,0,:,iv1])
    return EW_tild
@jit(nopython=True, cache=True)
def n0(Jd0, n0_star, N_grid, Ihire, hire_c):
    for idx in np.argwhere(Ihire):
        slice_Jd0 = (Jd0[idx[0], idx[1], idx[2], idx[3], idx[4], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], idx[4], :-1]) / (N_grid[1:]-N_grid[:-1])# Shape should be (5,)
        n0_star[idx[0], idx[1], idx[2], idx[3], idx[4]] = np.interp( -hire_c ,impose_increasing(-slice_Jd0),N_grid[1:]) #oh shit, should we also account for how that affects the worker value???
    print("n0_star borders", n0_star.min(), n0_star.max())   
    return n0_star 
@jit(nopython=True, cache=True)
def EJs(EJ1_star, EW1_star, Jd0, Wd0, n0_star, N_grid, num_z, num_n, num_v, num_q):
    for iz in range(num_z):
        for in0 in range(num_n):
            for in1 in range(num_n):
                for iv in range(num_v):
                 for iq in range(num_q):
                    EJ1_star[iz,in0,in1,iv,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,Jd0[iz,in0,in1,iv,:])
                    EW1_star[iz,in0,in1,iv,iq] = np.interp(n0_star[iz,in0,in1,iv,iq],N_grid,Wd0[iz,in0,in1,iv,:])
    return EJ1_star, EW1_star
@jit(nopython=True, cache=True)
def EQs(EJq,EWq,EJpi,EW1i,q_star,Q_grid,num_z,num_n,num_v,num_q):
    for iz in range(num_z):
        for in0 in range(num_n):
         for in1 in range(num_n):
            for iv in range(num_v):
             for iq in range(num_q):
                EJq[iz,in0,in1,iv,iq] = np.interp(q_star[iz,in0,in1,iv,iq],Q_grid, EJpi[iz,in0,in1,iv,:]) 
                EWq[iz,in0,in1,iv,iq] = np.interp(q_star[iz,in0,in1,iv,iq],Q_grid, EW1i[iz,in0,in1,iv,:])
    return EJq, EWq
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
        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = np.ones(self.p.num_x) * self.p.u_bf_m

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

        #self.unemp_bf = self.fun_prod_onedim[3]

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
        self.sum_sizeadj[...] = self.N_grid[self.grid[1]] * self.p.q_0
        for i in range(2, K + 1):
            self.sum_size += self.N_grid1[self.grid[i]]
            self.sum_sizeadj += self.N_grid1[self.grid[i]] * self.Q_grid[self.grid[self.J_grid.ndim - (K-1) + (i-2)]]
        for i in range(K+1,self.J_grid.ndim - (K-1)):
            self.sum_wage += self.w_grid[self.grid[i]]*self.N_grid1[self.grid[i-K+1]] #We add +1 because the wage at the very first step is semi-exogenous, and I will derive it directly

        #Setting up production grids
        self.prod = self.production(self.sum_sizeadj)
        self.prod_diff = self.production_diff(self.sum_sizeadj)
        self.prod_1d = self.fun_prod_1d(self.sum_sizeadj)
        self.prod_nd = self.prod_1d * self.Q_grid[self.grid[4]] #\partial F / \partial n_1 = q_1 F'(nq)
        self.prod_qd = self.prod_1d * self.N_grid1[self.grid[2]] #\partial F / \partial q_1 = n_1 F'(nq)


        #Job value and GE first
        self.v_grid = np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod_onedim.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #Value promised to the worker at the bottom step
        self.v_0 = self.v_grid.min()#-1.0
        
        self.simple_J=np.divide(self.fun_prod_onedim[:,ax] - self.w_grid[ax,:],1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        #This is equivalent to marginal value of a firm of size 1 at the lowest step
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
        #The two methods are equivalent!! grid[1] really does capture the right value!!!


        #Guess for the Worker value function
        self.W1i = np.zeros_like(self.J_grid)
        self.W1i = np.expand_dims(self.W1i, axis=-1) #adding an extra dimension to W1i
        self.W1i = np.repeat(self.W1i, self.K, axis=-1)

        #Creating the wage matrix manually
        self.w_matrix = np.zeros(self.W1i.shape)
        self.w_matrix[...,0] = 0 #The workers at the bottom step will have special wages, derived endogenously through their PK
        #Actually, do I then need to add that step to the worker value? Not really, but useful regardless.
        # Can say that the bottom step really is step zero, with a fixed value owed to the worker.
        # And then all the actually meaningful steps are 1,2... etc, so when K=2 with just have 1 meaningful step            
        self.w_matrix[...,1] = self.w_grid[ax,ax,ax,:,ax]

        self.W1i[...,1] = self.W1i[...,1] + self.pref.utility(self.w_matrix[...,1])/(1-self.p.beta) #skip the first K-1 columns, as they don't correspond to the wage state. Then, pick the correct step, which is hidden in the last dimension of the grid
        self.W1i[...,0] = self.W1i[...,0] + self.pref.utility(self.unemp_bf.min())/(1-self.p.beta)


    def J(self,Jg=None,Wg=None,update_eq=0):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        sum_wage = self.sum_wage
        rho_grid = self.rho_grid

        if Jg is None:
            Ji = np.copy(self.J_grid)
        else:
            Ji = np.copy(Jg)
        if Wg is None:
            W1i = np.copy(self.W1i)
        else:
            W1i = np.copy(Wg)
        
        print("Ji shape", Ji.shape)
        print("W1i shape", W1i.shape)        
        # create representation for J1p
        #J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py


        EW1_star = np.copy(Ji)
        EJ1_star = np.copy(Ji)
        EJderiv = np.zeros_like(EJ1_star)
        EWderiv = np.zeros_like(EW1_star)
        EJq = np.zeros_like(EJ1_star)
        EWq = np.zeros_like(EW1_star)
        #EW_tilde = np.copy(Ji)
        Jderiv = np.zeros_like(Ji)
        rho_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))
        sep_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))
        
        n0_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))        
        n1_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))   

        Jfullderiv = np.zeros_like(Ji)
        Wderiv = np.zeros_like(Ji)


        # prepare expectation call
        Ez = oe.contract_expression('anmvq,az->znmvq', Ji.shape, self.Z_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1
        
        # evaluate J1 tomorrow using our approximation
        #Jpi = J1p.eval_at_W1(W1i[...,1])
        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = np.copy(W1i)

            if ite_num>1:
             print("EJinv", EJinv[self.p.z_0-1,1,2,50, 0]/pc_star[self.p.z_0-1,1,2,50, 0])
             print("EJderiv", EJderiv[self.p.z_0-1,1,2,50, 0])
             j = np.where(self.N_grid==1)
             s = np.where(self.N_grid1==2)
             print("EJinv diff 1j 2s:", np.mean(np.abs((EJinv[:,j,s,:, 0]/pc_star[:,j,s,:, 0] - EJderiv[:,j,s,:, 0]) / EJderiv[:,j,s,:, 0])))
             print("EJinv diff 1 sen:", np.mean(np.abs((EJinv[:,0,1,:, 0]/pc_star[:,0,1,:, 0] - EJderiv[:,0,1,:, 0]) / EJderiv[:,0,1,:, 0])))
             print("EJinv diff 2 sen:", np.mean(np.abs((EJinv[:,0,s,:, 0]/pc_star[:,0,s,:, 0] - EJderiv[:,0,s,:, 0]) / EJderiv[:,0,s,:, 0])))


            # we compute the expected value next period by applying the transition rules
            EW1i = Ez(W1i[...,1], self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJpi = Ez(Ji, self.Z_trans_mat)

            # get worker decisions
            _, re, pc = self.getWorkerDecisions(EW1i)
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps) 
           
            if ite_num>100000000:
                EW1_tild = EW_tild(n1_star,EW1i,self.N_grid,self.p.num_z,self.p.num_n,self.p.num_v)
                _, re, pc = self.getWorkerDecisions(EW1_tild)
                # get worker decisions at EW1i + epsilon
                _, _, pc_d = self.getWorkerDecisions(EW1_tild+self.deriv_eps)
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff = np.zeros_like(pc)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value

            Ji3 = Ji + self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:,ax]*W1i[...,1] #This is the full rho
            # First boundary condition: forward difference            
            Jfullderiv[:, :, 0, ...] = (Ji3[:, :, 1,  ...] - Ji3[:, :, 0, ...]) / (self.N_grid1[1] - self.N_grid1[0])
            Wderiv[:, :, 0, ...]     = (W1i[:, :, 1, :, :, 1] - W1i[:, :, 0, :, :, 1]) / (self.N_grid1[1] - self.N_grid1[0])
            # Last boundary condition: backward difference
            Jfullderiv[:, :, -1, ...] = Ji3[:, :, -1,  ...] - Ji3[:, :, -2,  ...]/ (self.N_grid1[-1] - self.N_grid1[-2])
            Wderiv[:, :, -1, ...]     = W1i[:, :, -1, :, :, 1] - W1i[:, :, -2, :, :, 1]/ (self.N_grid1[-1] - self.N_grid1[-2])
            # Central differences: average of forward and backward differences
            Jfullderiv[:, :, 1:-1, ...] = (Ji3[:, :, 2:,  ...] - Ji3[:, :, :-2, ...]) / (self.N_grid1[ax, ax, 2:, ax, ax] - self.N_grid1[ax, ax, :-2, ax, ax])
            Wderiv[:, :, 1:-1, ...]     = (W1i[:, :, 2:, :, :, 1] - W1i[:, :, :-2, :, :, 1]) / (self.N_grid1[ax, ax, 2:, ax, ax] - self.N_grid1[ax, ax, :-2, ax, ax])

            Jderiv = Jfullderiv-rho_grid[ax,ax,ax,:,ax]*W1i[...,1]
            #Jderiv = Jfullderiv+self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:, ax]*Wderiv #accounting for the fact that size change also impacts W

            #EJinv=(Jderiv+self.w_grid[ax,ax,ax,:]-self.fun_prod*self.prod_diff)/self.p.beta #creating expected job value as a function of today's value
            EJinv=(Jderiv+self.w_grid[ax,ax,ax,:, ax]-self.fun_prod*self.prod_nd)/self.p.beta #creating expected job value as a function of today's value            
            #EJinv[:,0,0,:] = (Jderiv[:,0,0,:]+self.w_grid[ax,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta
            
            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modified with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1
            if ite_num<=100000000:
             #dim 0 is prod, dim 1 and 2 are size, dim 3 is future v, dim 4 is today's v, dim 5 is hmq
             foc = rho_grid[ax, ax, ax, :, ax, ax] - (EJinv[:, :, :, ax, :, :] / pc[...,ax,:])* (log_diff[...,ax,:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
             foc = foc*self.sum_size[..., ax, :] - self.N_grid1[self.grid[2][..., ax, :]]*rho_grid[ax, ax, ax, ax, :, ax] - self.N_grid[self.grid[1][..., ax, :]]/self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW1i[..., ax, :]+re[..., ax, :]))
            
            if ite_num>100000000:
             foc = rho_grid[ax, ax, ax, :, ax] - (EJinv[:, :, :, ax, :] / pc)* (log_diff / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
             foc = foc*self.sum_size[..., ax] - self.N_grid1[self.grid[2][..., ax]]*rho_grid[ax, ax, ax, ax, :] - self.N_grid[self.grid[1][:, :, :, ax, :]]/self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW1_tild+re))
            if ite_num<=100000000:
             assert (np.isnan(foc) & (pc[..., ax, :] > 0)).sum() == 0, "foc has NaN values where p>0"
            else:
             assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"

                #assert np.all(EW1i[..., 1:] >= EW1i[..., :-1]) #Andrei: check that worker value is increasing in v
            if ite_num<=100000000:
                rho_star = optimized_loop(
                    pc, rho_grid, self.N_grid1, foc, rho_star, self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_q) 
            else:
                rho_star = optimized_loop_tilde(
                    pc, rho_grid, foc, rho_star, self.p.num_z, self.p.num_n, self.p.num_v)
     
            
            #Getting n1_star
            if ite_num<=100000000:            
                n1_star = n1(pc, rho_grid, rho_star, sep_star, self.N_grid, self.N_grid1, self.p.num_z, self.p.num_n, self.p.num_v, self.p.num_q)
            else:
                n1_star = n1_tilde(n1_star,pc,rho_grid,rho_star, sep_star, self.N_grid,self.p.num_z, self.p.num_n, self.p.num_v)
            q_star = (self.p.q_0*self.N_grid[self.grid[1]]+self.Q_grid[self.grid[4]]*self.N_grid1[self.grid[2]])/(self.N_grid[self.grid[1]]*(1-sep_star)+self.N_grid1[self.grid[2]])
            #Getting hiring decisions
            Jd0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
            Wd0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
            n0_star[...] = 0
            if ite_num > 1:
                for iz in range(self.p.num_z):
                    for in00 in range(self.p.num_n):

                        J_interpolator = RegularGridInterpolator((self.N_grid1, rho_grid, self.Q_grid), EJpi[iz, in00, ...], bounds_error=False, fill_value=None)
                        W_interpolator = RegularGridInterpolator((self.N_grid1, rho_grid, self.Q_grid), EW1i[iz, in00, ...], bounds_error=False, fill_value=None)
                        Jd0[iz, ..., in00] = J_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                        Wd0[iz, ..., in00] = W_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                #Ihire = ((Jd0[...,1]-Jd0[...,0]+rho_star*n1_star*(Wd0[...,1]-Wd0[...,0])) > self.p.hire_c) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] < self.p.n_bar - 1)
                Ihire = ((Jd0[...,1]-Jd0[...,0]) / (self.N_grid[1]-self.N_grid[0]) > self.p.hire_c/self.p.beta) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] < self.p.n_bar - 1)
                #for idx in np.argwhere(Ihire):
                    #slice_Jd0 = Jd0[idx[0], idx[1], idx[2], idx[3], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], :-1]+n1_star[idx[0], idx[1], idx[2], idx[3]]*rho_star[idx[0], idx[1], idx[2], idx[3]]*(Wd0[idx[0], idx[1], idx[2], idx[3],1:]-Wd0[idx[0], idx[1], idx[2], idx[3],:-1])  # Shape should be (5,)
                #    slice_Jd0 = (Jd0[idx[0], idx[1], idx[2], idx[3], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], :-1]) / (self.N_grid[1:]-self.N_grid[:-1])# Shape should be (5,)
                #    n0_star[idx[0], idx[1], idx[2], idx[3]] = np.interp( -self.p.hire_c/self.p.beta ,impose_increasing(-slice_Jd0),self.N_grid[1:]) #oh shit, should we also account for how that affects the worker value???
             #print("n0_star borders", n0_star.min(), n0_star.max())    
                n0_star = n0(Jd0, n0_star, self.N_grid, Ihire, self.p.hire_c / self.p.beta)

            EW1i_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), EW1i[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
            EJpi_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), EJpi[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
            # Prepare points for interpolation
            rho_n_star_points = np.stack((n0_star, n1_star, rho_star, q_star), axis=-1)  # Shape: (num_z, ..., 2)
            # Vectorized interpolation over all iz
            EW1_star = np.array([interpolator(rho_n_star_points[iz, ...]) for iz, interpolator in enumerate(EW1i_interpolators)])
            EJ1_star = np.array([interpolator(rho_n_star_points[iz, ...]) for iz, interpolator in enumerate(EJpi_interpolators)])
            
            #EJ1_star, EW1_star = EJs(EJ1_star, EW1_star, Jd0, Wd0, n0_star, self.N_grid, self.p.num_z, self.p.num_n, self.p.num_v)
            #I'm interpolating EJ and EW functions on future q's in order to then use them for calculating future derivatives
            #Btw still super confused as to why the EJs function was somehow slower than the full complex interpolation
            EJq, EWq = EQs(EJq,EWq,EJpi,EW1i,q_star,self.Q_grid,self.p.num_z,self.p.num_n,self.p.num_v,self.p.num_q)
            #EW1_star = np.interp(n0_star, self.N_grid, Jd0[ax, ax, :, ax, :])
            #Getting the derivative of the future job value wrt n1:
            floorn1=np.floor(np.interp( n1_star, self.N_grid1, range(self.p.num_n))).astype(int)
            ceiln1=np.ceil(np.interp( n1_star, self.N_grid1, range(self.p.num_n))).astype(int)            
            #EJpi3 = EJpi+(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc*rho_grid[ax,ax,ax,:]*EW1i
            #EJderiv0 = EJderivative3(EJpi,EW1i, floorn1,ceiln1,EJderiv,rho_grid,rho_star,self.p.num_z, self.p.num_n, self.p.num_v)
            EJderiv0 = EJderivative(EJq, floorn1,ceiln1,n0_star, EJderiv,rho_grid, self.N_grid, self.N_grid1, rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
            EWderiv = EWderivative(EWq,floorn1,ceiln1, n0_star, EWderiv,rho_grid, self.N_grid, self.N_grid1, rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
            EJderiv = EJderiv0+n1_star*rho_star*EWderiv
            #Ejderiv = EJderiv0#-rho_star*EW1_star

            
            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            _, re_star, pc_star = self.getWorkerDecisions(EW1_star)
            #EJderiv = EJderiv0-pc_star*rho_star*EW1_star

            # Update firm value function 
            Ji = self.fun_prod*self.prod - sum_wage - self.p.hire_c * n0_star - \
                self.pref.inv_utility(self.v_0-self.p.beta*(EW1_star+re_star))*self.N_grid[self.grid[1]]  + self.p.beta * EJ1_star
            
            Ji = .2 * Ji + .8 * Ji2


            # Update worker value function
            W1i[...,1] = self.pref.utility(self.w_matrix[...,1]) + \
                self.p.beta * (EW1_star + re_star) #For more steps the ax at the end won't be needed as EW1_star itself will have multiple steps
            #W1i[:,:,0,:,1] = W1i[:,:,1,:,1]

            W1i[...,1:] = .4 * W1i[...,1:] + .6 * W1i2[...,1:] #we're completely ignoring the 0th step

            
            # Updating J1 representation
            #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[:,:,:,:,1], Ji)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_w1 = array_dist(W1i[...,1:], W1i2[...,1:])

            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                    print("Errors:",  error_j1i, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            #P_xv = self.matching_function(J1p.eval_at_W1(W1i)[self.p.z_0-1, 0, 1, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            #error_js = self.js.update(W1i[self.p.z_0-1, 0, 0, :, 1], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[...,1], Ji)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                    print("Errors:",  error_j1i,  error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            #if (ite_num % 25) == 0:
                # Updating J1 representation
                #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[...,1], Ji)
                #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                #print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)    
                #self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                #                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        #self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
        #                             ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        return Ji,W1i,EW1_star,pc_star,n0_star, n1_star

    def J_sep(self,Jg=None,Wg=None,update_eq=0):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        sum_wage = self.sum_wage
        rho_grid = self.rho_grid

        if Jg is None:
            Ji = np.copy(self.J_grid)
        else:
            Ji = np.copy(Jg)
        if Wg is None:
            W1i = np.copy(self.W1i)
        else:
            W1i = np.copy(Wg)
        Ui = self.pref.utility_gross(self.unemp_bf)/(1-self.p.beta)
        print("Ji shape", Ji.shape)
        print("W1i shape", W1i.shape)        
        # create representation for J1p
        #J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py


        EW1_star = np.copy(Ji)
        EJ1_star = np.copy(Ji)
        EJderiv = np.zeros_like(EJ1_star)
        EWderiv = np.zeros_like(EW1_star)
        EJq = np.zeros_like(EJ1_star)
        EWq = np.zeros_like(EW1_star)
        #EW_tilde = np.copy(Ji)
        Jderiv = np.zeros_like(Ji)
        rho_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))
        sep_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))
        
        n0_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))        
        n1_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))    

        Jfullderiv = np.zeros_like(Ji)
        Wderiv = np.zeros_like(Ji)
        Jderiv0 = np.zeros_like(Ji)
        Qderiv = np.zeros_like(Ji)


        # prepare expectation call
        Ez = oe.contract_expression('anmvq,az->znmvq', Ji.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1
        
        # evaluate J1 tomorrow using our approximation
        #Jpi = J1p.eval_at_W1(W1i[...,1])
        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = np.copy(W1i)
            Ui2 = Ui

            if ite_num>1:
             print("EJinv", EJinv[self.p.z_0-1,1,2,50, 0]/pc_star[self.p.z_0-1,1,2,50, 0])
             print("EJderiv", EJderiv[self.p.z_0-1,1,2,50, 0])
             j = np.where(self.N_grid==1)
             s = np.where(self.N_grid1==2)
             print("EJinv diff 1j 2s:", np.mean(np.abs((EJinv[:,j,s,:, 0]/pc_star[:,j,s,:, 0] - EJderiv[:,j,s,:, 0]) / EJderiv[:,j,s,:, 0])))
             print("EJinv diff 1 sen:", np.mean(np.abs((EJinv[:,0,1,:, 0]/pc_star[:,0,1,:, 0] - EJderiv[:,0,1,:, 0]) / EJderiv[:,0,1,:, 0])))
             print("EJinv diff 2 sen:", np.mean(np.abs((EJinv[:,0,s,:, 0]/pc_star[:,0,s,:, 0] - EJderiv[:,0,s,:, 0]) / EJderiv[:,0,s,:, 0])))

            # we compute the expected value next period by applying the transition rules
            EW1i = Ez(W1i[...,1], self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJpi = Ez(Ji, self.Z_trans_mat)
            EUi = Ui
            # get worker decisions
            _, re, pc = self.getWorkerDecisions(EW1i)
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps) 
           
            if ite_num>100000000:
                EW1_tild = EW_tild(n1_star,EW1i,self.N_grid,self.p.num_z,self.p.num_n,self.p.num_v)
                _, re, pc = self.getWorkerDecisions(EW1_tild)
                # get worker decisions at EW1i + epsilon
                _, _, pc_d = self.getWorkerDecisions(EW1_tild+self.deriv_eps)
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff = np.zeros_like(pc)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value

            Ji3 = Ji + self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:,ax]*W1i[...,1] #This is the full rho

            # First boundary condition: forward difference            
            Jfullderiv[:, :, 0, ...] = (Ji3[:, :, 1,  ...] - Ji3[:, :, 0, ...]) / (self.N_grid1[1] - self.N_grid1[0])
            Wderiv[:, :, 0, ...]     = (W1i[:, :, 1, :, :, 1] - W1i[:, :, 0, :, :, 1]) / (self.N_grid1[1] - self.N_grid1[0])
            Jderiv0[:, 0, :, :]    = Ji[:, 1, ...] - Ji[:, 0, ...] / (self.N_grid[1] - self.N_grid[0])
            Qderiv[...,0] = (Ji3[...,1]-Ji3[...,0]) / (self.Q_grid[1]-self.Q_grid[0])
            # Last boundary condition: backward difference
            Jfullderiv[:, :, -1, ...] = Ji3[:, :, -1,  ...] - Ji3[:, :, -2,  ...]/ (self.N_grid1[-1] - self.N_grid1[-2])
            Wderiv[:, :, -1, ...]     = W1i[:, :, -1, :, :, 1] - W1i[:, :, -2, :, :, 1]/ (self.N_grid1[-1] - self.N_grid1[-2])
            Jderiv0[:, -1, :, :]    = Ji[:, -1, ...] - Ji[:, -2, ...]/ (self.N_grid[-1] - self.N_grid[-2])
            Qderiv[...,-1] = (Ji3[...,-1]-Ji3[...,-2]) / (self.Q_grid[-1]-self.Q_grid[-2])
            # Central differences: average of forward and backward differences
            Jfullderiv[:, :, 1:-1, ...] = (Ji3[:, :, 2:,  ...] - Ji3[:, :, :-2, ...]) / (self.N_grid1[ax, ax, 2:, ax, ax] - self.N_grid1[ax, ax, :-2, ax, ax])
            Wderiv[:, :, 1:-1, ...]     = (W1i[:, :, 2:, :, :, 1] - W1i[:, :, :-2, :, :, 1]) / (self.N_grid1[ax, ax, 2:, ax, ax] - self.N_grid1[ax, ax, :-2, ax, ax])
            Jderiv0[:, 1:-1, ...]    = (Ji[:, 2:, ...] - Ji[:, :-2, ...]) / (self.N_grid[ax, 2:, ax, ax, ax] - self.N_grid[ax, :-2, ax, ax, ax])
            Qderiv[...,1:-1] = (Ji3[...,2:] - Ji3[...,:-2]) / (self.Q_grid[2:] - self.Q_grid[:-2])
            Jderiv = Jfullderiv-rho_grid[ax,ax,ax,:,ax]*W1i[...,1]
            #Jderiv = Jfullderiv+self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:]*Wderiv #accounting for the fact that size change also impacts W
    	    
            #Jderiv0 = Jderiv0+self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:]*Wderiv0 #accounting for the fact that size change also impacts W

            EJinv=(Jderiv+self.w_grid[ax,ax,ax,:, ax]-self.fun_prod*self.prod_nd)/self.p.beta #creating expected job value as a function of today's value            
            #EJinv[:,0,0,:] = (Jderiv[:,0,0,:]+self.w_grid[ax,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta
            if ite_num>1: #I'm using previous guesses for sep_star and EW1_star. This way, it is still as if EJinv0 is a function of today's states only, even though that's not exactly correct
             EJinv0 = (Jderiv0+wage_jun- self.p.q_0 * self.fun_prod*self.prod_1d)/self.p.beta #Multiply by q_0 to gt the correct derivative
             #EJinv0[:,0,0,:] = (Jderiv0[:,0,0,:]+wage_jun[:,0,0,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta


            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modified with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1
            if ite_num<=100000000:
             #Main foc, in the absence for separations
             inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[...,ax,:,:]*EUi+(1-sep_star[...,ax,:,:])*(EW1i[..., ax, :]+re[..., ax, :])))
             foc_2ndpart = - self.N_grid1[self.grid[2][..., ax, :]]*rho_grid[ax, ax, ax, ax, :, ax] -\
                 self.N_grid[self.grid[1][..., ax, :]] * (1-sep_star[...,ax,:,:]) / inv_utility_1d             
             
             foc = rho_grid[ax, ax, ax, :, ax, ax] - (EJinv[:, :, :, ax, :, :] / pc[...,ax,:])* (log_diff[...,ax,:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
             foc = foc*(self.N_grid[self.grid[1][..., ax, :]] * (1-sep_star[...,ax,:,:])+self.N_grid1[self.grid[2][..., ax, :]]) + foc_2ndpart
            #There are no separations here as this FOC is in the case of NO separations (although, what if s=1?)
            #Foc for wages if separations are positive
             foc_rho_s = rho_grid[ax, ax, ax, :,ax, ax]+((EW1i[..., ax, :]+re[..., ax, :]-EUi) / inv_utility_1d)*(log_diff[..., ax,:] / self.deriv_eps)/(pc[..., ax,:])
             foc_rho_s =  foc_rho_s*(self.N_grid[self.grid[1][..., ax,:]] * (1-sep_star[...,ax,:,:]) + self.N_grid1[self.grid[2][..., ax,:]]) + foc_2ndpart

            if ite_num>100000000:
             foc = rho_grid[ax, ax, ax, :, ax] - (EJinv[..., ax, :] / pc)* (log_diff / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
             foc = foc*self.sum_size[..., ax] - self.N_grid1[self.grid[2][..., ax]]*rho_grid[ax, ax, ax, ax, :] - self.N_grid[self.grid[1][:, :, :, ax, :]]/self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW1_tild+re))
            
            
            if ite_num<=100000000:
             assert (np.isnan(foc) & (pc[..., ax, :] > 0)).sum() == 0, "foc has NaN values where p>0"
            else:
             assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"


            #assert np.all(EW1i[iz, in0, in1, 1:] >= EW1i[iz, in0, in1, :-1]) #Andrei: check that worker value is increasing in v
            #if ite_num<=1:
            rho_star = optimized_loop(
                pc, rho_grid, self.N_grid1, foc, rho_star, self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_q) 

   
            if ite_num>1:
                sep_star[...] = 0  
                Ifire = (EJinv0 < 0) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] <= self.p.n_bar) & (self.N_grid[self.grid[1]] > 0)
                worker_future_value = np.zeros_like(EW1i)
                for iz in range(self.p.num_z):
                 for in0 in range(self.p.num_n):
                  for in1 in range(self.p.num_n):
                   for iv in range(self.p.num_v):
                    for iq in range(self.p.num_q):
                     if Ifire[iz,in0,in1,iv,iq]:
                        worker_future_value[iz,in0,in1,iv,iq] = np.maximum(np.interp(rho_star[iz, in0, in1, iv,iq], rho_grid, re[iz,in0,in1,:,iq]+EW1i[iz,in0,in1,:,iq]),EUi)
                        #print("Worker future value:", worker_future_value[iz,in0,in1,iv,iq])
                #sep_star[Ifire] = 1-(EJinv0[Ifire]/((EUi-worker_future_value[Ifire]) / self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[Ifire]*EUi+(1-sep_star[Ifire])*worker_future_value[Ifire])))) #The thing is, this wouldn't work: t he corner case of ultra negative EJinv would suggest us negative separations, rather than 1       
                #New formula where I use EJinv rather than EJinv0. That way, I only have separations via the HMQ part
                for idx in np.argwhere(Ifire):
                
                    sep_star[idx[0], idx[1], idx[2], idx[3], idx[4]] = 1- ( (self.Q_grid[idx[4]]*self.N_grid1[idx[2]]+self.p.q_0*self.N_grid[idx[1]]) * (Qderiv[idx[0], idx[1], idx[2], idx[3], idx[4]]-self.prod_qd[idx[0], idx[1], idx[2], idx[3], idx[4]]) / \
                    (((worker_future_value[idx[0], idx[1], idx[2], idx[3], idx[4]]-EUi) / (self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[idx[0], idx[1], idx[2], idx[3], idx[4]]*EUi+(1-sep_star[idx[0], idx[1], idx[2], idx[3], idx[4]])*worker_future_value[idx[0], idx[1], idx[2], idx[3], idx[4]])))+ EJinv[idx[0], idx[1], idx[2], idx[3], idx[4]]) * self.p.beta * self.N_grid1[idx[2]]) - self.N_grid1[idx[2]]) / self.N_grid[idx[1]]
                #NOTE: gotta adjust the derivatives. If my production function is F(\sum n*z), then F'n = z F' and F'z = n F'	
                sepneg = (EJinv0 < 0) & (sep_star < 0)
                sep_star[sepneg] = 1
                sep_star = np.minimum(sep_star, 0.5)
            #if ite_num>1:
            #    rho_star,sep_star = optimized_loop_sep(
            #        re, pc, EJinv0, EW1i, EUi, rho_grid, foc, rho_star, sep_star, self.p.num_z, self.p.num_n, self.v_0, self.pref.inv_utility_1d, self.p.beta)
            #else:
            #    rho_star = optimized_loop_tilde(
            #        pc, rho_grid, foc, rho_star, self.p.num_z, self.p.num_n, self.p.num_v)
            #sep_star[:,0,...] = 0 #This is only for now, as we're not considering separations for seniors
    
            n1_star = n1(pc, rho_grid, rho_star, sep_star, self.N_grid, self.N_grid1, self.p.num_z, self.p.num_n, self.p.num_v, self.p.num_q)
            q_star = (self.p.q_0*self.N_grid[self.grid[1]]+self.Q_grid[self.grid[4]]*self.N_grid1[self.grid[2]])/(self.N_grid[self.grid[1]]*(1-sep_star)+self.N_grid1[self.grid[2]])

            #Getting hiring decisions
            #Getting hiring decisions
            Jd0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
            Wd0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
            n0_star[...] = 0
            if ite_num > 1:
                for iz in range(self.p.num_z):
                    for in00 in range(self.p.num_n):

                        J_interpolator = RegularGridInterpolator((self.N_grid1, rho_grid, self.Q_grid), EJpi[iz, in00, ...], bounds_error=False, fill_value=None)
                        W_interpolator = RegularGridInterpolator((self.N_grid1, rho_grid, self.Q_grid), EW1i[iz, in00, ...], bounds_error=False, fill_value=None)
                        Jd0[iz, ..., in00] = J_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                        Wd0[iz, ..., in00] = W_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                #Ihire = ((Jd0[...,1]-Jd0[...,0]+rho_star*n1_star*(Wd0[...,1]-Wd0[...,0])) > self.p.hire_c) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] < self.p.n_bar - 1)
                Ihire = ((Jd0[...,1]-Jd0[...,0]) / (self.N_grid[1]-self.N_grid[0]) > self.p.hire_c/self.p.beta) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] < self.p.n_bar - 1)
                #for idx in np.argwhere(Ihire):
                    #slice_Jd0 = Jd0[idx[0], idx[1], idx[2], idx[3], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], :-1]+n1_star[idx[0], idx[1], idx[2], idx[3]]*rho_star[idx[0], idx[1], idx[2], idx[3]]*(Wd0[idx[0], idx[1], idx[2], idx[3],1:]-Wd0[idx[0], idx[1], idx[2], idx[3],:-1])  # Shape should be (5,)
                #    slice_Jd0 = (Jd0[idx[0], idx[1], idx[2], idx[3], 1:] - Jd0[idx[0], idx[1], idx[2], idx[3], :-1]) / (self.N_grid[1:]-self.N_grid[:-1])# Shape should be (5,)
                #    n0_star[idx[0], idx[1], idx[2], idx[3]] = np.interp( -self.p.hire_c/self.p.beta ,impose_increasing(-slice_Jd0),self.N_grid[1:]) #oh shit, should we also account for how that affects the worker value???
             #print("n0_star borders", n0_star.min(), n0_star.max())    
                n0_star = n0(Jd0, n0_star, self.N_grid, Ihire, self.p.hire_c / self.p.beta)


            EW1i_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), EW1i[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
            EJpi_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), EJpi[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
            # Prepare points for interpolation
            rho_n_star_points = np.stack((n0_star, n1_star, rho_star, q_star), axis=-1)  # Shape: (num_z, ..., 2)
            # Vectorized interpolation over all iz
            EW1_star = np.array([interpolator(rho_n_star_points[iz, ...]) for iz, interpolator in enumerate(EW1i_interpolators)])
            EJ1_star = np.array([interpolator(rho_n_star_points[iz, ...]) for iz, interpolator in enumerate(EJpi_interpolators)])
            
            #EJ1_star, EW1_star = EJs(EJ1_star, EW1_star, Jd0, Wd0, n0_star, self.N_grid, self.p.num_z, self.p.num_n, self.p.num_v)
            #I'm interpolating EJ and EW functions on future q's in order to then use them for calculating future derivatives
            #Btw still super confused as to why the EJs function was somehow slower than the full complex interpolation
            EJq, EWq = EQs(EJq,EWq,EJpi,EW1i,q_star,self.Q_grid,self.p.num_z,self.p.num_n,self.p.num_v,self.p.num_q)
            #EW1_star = np.interp(n0_star, self.N_grid, Jd0[ax, ax, :, ax, :])
            #Getting the derivative of the future job value wrt n1:
            floorn1=np.floor(np.interp( n1_star, self.N_grid1, range(self.p.num_n))).astype(int)
            ceiln1=np.ceil(np.interp( n1_star, self.N_grid1, range(self.p.num_n))).astype(int)            
            #EJpi3 = EJpi+(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc*rho_grid[ax,ax,ax,:]*EW1i
            #EJderiv0 = EJderivative3(EJpi,EW1i, floorn1,ceiln1,EJderiv,rho_grid,rho_star,self.p.num_z, self.p.num_n, self.p.num_v)
            EJderiv0 = EJderivative(EJq, floorn1,ceiln1,n0_star, EJderiv,rho_grid, self.N_grid, self.N_grid1, rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
            EWderiv = EWderivative(EWq,floorn1,ceiln1, n0_star, EWderiv,rho_grid, self.N_grid, self.N_grid1, rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
            EJderiv = EJderiv0+n1_star*rho_star*EWderiv

            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            _, re_star, pc_star = self.getWorkerDecisions(EW1_star)
            #print("states at which worker quits:", np.where(~(pc_star[self.p.z_0-1,1,1,:]==0)))
            # Update firm value function
            wage_jun = self.pref.inv_utility(self.v_0-self.p.beta*(sep_star*EUi+(1-sep_star)*(EW1_star+re_star)))
            Ji = self.fun_prod*self.prod - sum_wage - self.p.hire_c*n0_star  -\
                wage_jun*self.N_grid[self.grid[1]]  + self.p.beta * EJ1_star
            Ji = .2*Ji + .8*Ji2

            # Update worker value function
            W1i[...,1] = self.pref.utility(self.w_matrix[...,1]) + \
                self.p.beta * (re_star + EW1_star) #For more steps the ax at the end won't be needed as EW1_star itself will have multiple steps

            W1i[...,1:] = .4*W1i[...,1:] + .6*W1i2[...,1:] #we're completely ignoring the 0th step

            #print("Worker Value diff:", np.max(np.abs(W1i[:,:,:,:,1:]-W1i2[:,:,:,:,1:])))   
            _, ru, _ = self.getWorkerDecisions(EUi, employed=False)
            Ui = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EUi)
            Ui = 0.4*Ui + 0.6*Ui2
            if ite_num>1:
                print("sep borders", sep_star.min(), sep_star.max())
            # Updating J1 representation
            #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[:,:,:,:,1], Ji)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  

            error_w1 = array_dist(W1i[...,1:], W1i2[...,1:])



            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                    print("Errors:",  error_j1i, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            #P_xv = self.matching_function(J1p.eval_at_W1(W1i)[self.p.z_0-1, 0, 1, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            #error_js = self.js.update(W1i[self.p.z_0-1, 0, 0, :, 1], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[...,1], Ji)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                    print("Errors:",  error_j1i,  error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            #if (ite_num % 25) == 0:
                # Updating J1 representation
                #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[...,1], Ji)
                #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                #print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)    
                #self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                #                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        #self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
        #                             ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        return Ji,W1i,EW1_star,sep_star, n0_star, n1_star

    def J_sep_dir(self,Jg=None,Wg=None,update_eq=0):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        sum_wage = self.sum_wage
        rho_grid = self.rho_grid

        if Jg is None:
            Ji = np.copy(self.J_grid)
        else:
            Ji = np.copy(Jg)
        if Wg is None:
            W1i = np.copy(self.W1i)
        else:
            W1i = np.copy(Wg)
        Ui = self.pref.utility_gross(self.unemp_bf)/(1-self.p.beta)
        print("Ji shape", Ji.shape)
        print("W1i shape", W1i.shape)        
        # create representation for J1p
        #J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py


        EW1_star = np.copy(Ji)
        EJ1_star = np.copy(Ji)
        EJderiv = np.zeros_like(EJ1_star)
        EWderiv = np.zeros_like(EW1_star)
        EJq = np.zeros_like(EJ1_star)
        EWq = np.zeros_like(EW1_star)
        #EW_tilde = np.copy(Ji)
        Jderiv = np.zeros_like(Ji)
        rho_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))
        sep_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))
        
        n0_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))        
        n1_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v,self.p.num_q))   

        Jfullderiv = np.zeros_like(Ji)
        Wderiv = np.zeros_like(Ji)
        Jderiv0 = np.zeros_like(Ji)
        Qderiv = np.zeros_like(Ji)


        # prepare expectation call
        Ez = oe.contract_expression('anmvq,az->znmvq', Ji.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1
        
        # evaluate J1 tomorrow using our approximation
        #Jpi = J1p.eval_at_W1(W1i[:,:,:,:,1])

        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = np.copy(W1i)
            Ui2 = Ui

            if ite_num>1:
             print("EJinv", EJinv[self.p.z_0-1,1,2,50, 0]/pc_star[self.p.z_0-1,1,2,50, 0])
             print("EJderiv", EJderiv[self.p.z_0-1,1,2,50, 0])
             j = np.where(self.N_grid==1)
             s = np.where(self.N_grid1==2)
             print("EJinv diff 1j 2s:", np.mean(np.abs((EJinv[:,j,s,:, 0]/pc_star[:,j,s,:, 0] - EJderiv[:,j,s,:, 0]) / EJderiv[:,j,s,:, 0])))
             print("EJinv diff 1 sen:", np.mean(np.abs((EJinv[:,0,1,:, 0]/pc_star[:,0,1,:, 0] - EJderiv[:,0,1,:, 0]) / EJderiv[:,0,1,:, 0])))
             print("EJinv diff 2 sen:", np.mean(np.abs((EJinv[:,0,s,:, 0]/pc_star[:,0,s,:, 0] - EJderiv[:,0,s,:, 0]) / EJderiv[:,0,s,:, 0])))

            # we compute the expected value next period by applying the transition rules
            EW1i = Ez(W1i[...,1], self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJpi = Ez(Ji, self.Z_trans_mat)
            EUi = Ui
            # get worker decisions
            _, re, pc = self.getWorkerDecisions(EW1i)
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps) 
           
            if ite_num>100000000:
                EW1_tild = EW_tild(n1_star,EW1i,self.N_grid,self.p.num_z,self.p.num_n,self.p.num_v)
                _, re, pc = self.getWorkerDecisions(EW1_tild)
                # get worker decisions at EW1i + epsilon
                _, _, pc_d = self.getWorkerDecisions(EW1_tild+self.deriv_eps)
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff = np.zeros_like(pc)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value

            Ji3 = Ji + self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:,ax]*W1i[...,1] #This is the full rho

            # First boundary condition: forward difference            
            Jfullderiv[:, :, 0, ...] = (Ji3[:, :, 1,  ...] - Ji3[:, :, 0, ...]) / (self.N_grid1[1] - self.N_grid1[0])
            Wderiv[:, :, 0, ...]     = (W1i[:, :, 1, :, :, 1] - W1i[:, :, 0, :, :, 1]) / (self.N_grid1[1] - self.N_grid1[0])
            Jderiv0[:, 0, :, :]    = Ji[:, 1, ...] - Ji[:, 0, ...] / (self.N_grid[1] - self.N_grid[0])
            Qderiv[...,0] = (Ji[...,1]-Ji[...,0]) / (self.Q_grid[1]-self.Q_grid[0])
            # Last boundary condition: backward difference
            Jfullderiv[:, :, -1, ...] = Ji3[:, :, -1,  ...] - Ji3[:, :, -2,  ...]/ (self.N_grid1[-1] - self.N_grid1[-2])
            Wderiv[:, :, -1, ...]     = W1i[:, :, -1, :, :, 1] - W1i[:, :, -2, :, :, 1]/ (self.N_grid1[-1] - self.N_grid1[-2])
            Jderiv0[:, -1, :, :]    = Ji[:, -1, ...] - Ji[:, -2, ...]/ (self.N_grid[-1] - self.N_grid[-2])
            Qderiv[...,-1] = (Ji[...,-1]-Ji[...,-2]) / (self.Q_grid[-1]-self.Q_grid[-2])
            # Central differences: average of forward and backward differences
            Jfullderiv[:, :, 1:-1, ...] = (Ji3[:, :, 2:,  ...] - Ji3[:, :, :-2, ...]) / (self.N_grid1[ax, ax, 2:, ax, ax] - self.N_grid1[ax, ax, :-2, ax, ax])
            Wderiv[:, :, 1:-1, ...]     = (W1i[:, :, 2:, :, :, 1] - W1i[:, :, :-2, :, :, 1]) / (self.N_grid1[ax, ax, 2:, ax, ax] - self.N_grid1[ax, ax, :-2, ax, ax])
            Jderiv0[:, 1:-1, ...]    = (Ji[:, 2:, ...] - Ji[:, :-2, ...]) / (self.N_grid[ax, 2:, ax, ax, ax] - self.N_grid[ax, :-2, ax, ax, ax])
            Qderiv[...,1:-1] = (Ji[...,2:] - Ji[...,:-2]) / (self.Q_grid[2:] - self.Q_grid[:-2])
            Jderiv = Jfullderiv-rho_grid[ax,ax,ax,:,ax]*W1i[...,1]
            
            #Jderiv = Jfullderiv+self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:, ax]*Wderiv #accounting for the fact that size change also impacts W
    	    
            #Jderiv0 = Jderiv0+self.N_grid1[self.grid[2]]*rho_grid[ax,ax,ax,:]*Wderiv0 #accounting for the fact that size change also impacts W


            EJinv=(Jderiv+self.w_grid[ax,ax,ax,:, ax]-self.fun_prod*self.prod_nd)/self.p.beta #creating expected job value as a function of today's value            
            #EJinv[:,0,0,:] = (Jderiv[:,0,0,:]+self.w_grid[ax,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta
            if ite_num>1: #I'm using previous guesses for sep_star and EW1_star. This way, it is still as if EJinv0 is a function of today's states only, even though that's not exactly correct
             EJinv0 = (Jderiv0+wage_jun-self.p.q_0 * self.fun_prod*self.prod_1d)/self.p.beta
             #EJinv0[:,0,0,:] = (Jderiv0[:,0,0,:]+wage_jun[:,0,0,:]-self.fun_prod[:,0,0,:]*self.prod_diff[:,0,0,:])/self.p.beta


            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modified with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1
            if ite_num<=100000000:
             #Main foc, in the absence for separations
             inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[...,ax,:,:]*EUi+(1-sep_star[...,ax,:,:])*(EW1i[..., ax, :]+re[..., ax, :])))
             foc_2ndpart = - self.N_grid1[self.grid[2][..., ax, :]]*rho_grid[ax, ax, ax, ax, :, ax] -\
                 self.N_grid[self.grid[1][..., ax, :]] * (1-sep_star[...,ax,:,:]) / inv_utility_1d             
             
             foc = rho_grid[ax, ax, ax, :, ax, ax] - (EJinv[:, :, :, ax, :, :] / pc[...,ax,:])* (log_diff[...,ax,:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
             foc = foc*(self.N_grid[self.grid[1][..., ax, :]] * (1-sep_star[...,ax,:,:])+self.N_grid1[self.grid[2][..., ax, :]]) + foc_2ndpart
            #There are no separations here as this FOC is in the case of NO separations (although, what if s=1?)
            #Foc for wages if separations are positive
             foc_rho_s = rho_grid[ax, ax, ax, :,ax, ax]+((EW1i[..., ax, :]+re[..., ax, :]-EUi) / inv_utility_1d)*(log_diff[..., ax,:] / self.deriv_eps)/(pc[..., ax,:])
             foc_rho_s =  foc_rho_s*(self.N_grid[self.grid[1][..., ax,:]] * (1-sep_star[...,ax,:,:]) + self.N_grid1[self.grid[2][..., ax,:]]) + foc_2ndpart

            if ite_num>100000000:
             foc = rho_grid[ax, ax, ax, :, ax] - (EJinv[..., ax, :] / pc)* (log_diff / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
             foc = foc*self.sum_size[..., ax] - self.N_grid1[self.grid[2][..., ax]]*rho_grid[ax, ax, ax, ax, :] - self.N_grid[self.grid[1][:, :, :, ax, :]]/self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW1_tild+re))
            
            
            if ite_num<=100000000:
             assert (np.isnan(foc) & (pc[..., ax, :] > 0)).sum() == 0, "foc has NaN values where p>0"
            else:
             assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"


            #assert np.all(EW1i[iz, in0, in1, 1:] >= EW1i[iz, in0, in1, :-1]) #Andrei: check that worker value is increasing in v
            #if ite_num<=1:
            rho_star = optimized_loop(
                 pc, rho_grid, self.N_grid1, foc, rho_star, self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_q) 	
            if ite_num>1:
                EJderiv0 = np.zeros_like(EJ1_star)
                EWderiv0 = np.zeros_like(EW1i)
                EJderiv1 = np.zeros_like(EJ1_star)
                EWderiv1 = np.zeros_like(EW1i)
                q0 = np.zeros_like(EJ1_star)
                q1 = np.zeros_like(EJ1_star)
                q0 = (self.p.q_0*self.N_grid[self.grid[1]]+self.Q_grid[self.grid[4]]*self.N_grid1[self.grid[2]])/(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]]) + (self.grid[3] - self.grid[3])
                q1 = (self.p.q_0*self.N_grid[self.grid[1]]+self.Q_grid[self.grid[4]]*self.N_grid1[self.grid[2]])/(self.N_grid[self.grid[1]]*(1- 0.5)+self.N_grid1[self.grid[2]]) + (self.grid[3] - self.grid[3])
                EJ0, EW0 = EQs(EJq,EWq,EJpi,EW1i,q0,self.Q_grid,self.p.num_z,self.p.num_n,self.p.num_v,self.p.num_q)
                EJ1, EW1 = EQs(EJq,EWq,EJpi,EW1i,q1,self.Q_grid,self.p.num_z,self.p.num_n,self.p.num_v,self.p.num_q)

                EJderiv0 = EJderivative(EJ0,np.floor((self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star).astype(int),np.ceil((self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star).astype(int),n0_star, EJderiv0,rho_grid, self.N_grid, self.N_grid1,rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
                EWderiv0 = EWderivative(EW0,np.floor((self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star).astype(int),np.ceil((self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star).astype(int),n0_star, EWderiv0,rho_grid, self.N_grid, self.N_grid1,rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
                EJderiv1 = EJderivative(EJ1,np.floor((self.N_grid[self.grid[1]] * self.p.q_0+self.N_grid1[self.grid[2]])*pc_star).astype(int),np.ceil((self.N_grid[self.grid[1]] * self.p.q_0+self.N_grid1[self.grid[2]])*pc_star).astype(int),n0_star, EJderiv1,rho_grid, self.N_grid, self.N_grid1,rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
                EWderiv1 = EWderivative(EW1,np.floor((self.N_grid[self.grid[1]] * self.p.q_0+self.N_grid1[self.grid[2]])*pc_star).astype(int),np.ceil((self.N_grid[self.grid[1]] * self.p.q_0+self.N_grid1[self.grid[2]])*pc_star).astype(int),n0_star, EWderiv1,rho_grid, self.N_grid, self.N_grid1,rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
                
                ERho = Ez(Ji, self.Z_trans_mat)    #Ez(Ji3, self.Z_trans_mat)                
                Rho_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), ERho[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
                # Prepare points for interpolation
                rho_n_star_points_0 = np.stack((n0_star, (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star, rho_star, q0), axis=-1)  # Shape: (num_z, ..., 2)
                rho_n_star_points_1 = np.stack((n0_star, (self.N_grid[self.grid[1]]*self.p.q_0+self.N_grid1[self.grid[2]])*pc_star, rho_star, q1), axis=-1)  # Shape: (num_z, ..., 2)

                # Vectorized interpolation over all iz
                ERho_star_0 = np.array([interpolator(rho_n_star_points_0[iz, ...]) for iz, interpolator in enumerate(Rho_interpolators)])
                ERho_star_1 = np.array([interpolator(rho_n_star_points_1[iz, ...]) for iz, interpolator in enumerate(Rho_interpolators)])
                RhoderivQ0 = np.zeros_like(rho_star)
                RhoderivQ1 = np.zeros_like(rho_star)
                floorq0=np.floor(np.interp( q0, self.Q_grid, range(self.p.num_q))).astype(int)
                floorq1=np.floor(np.interp( q1, self.Q_grid, range(self.p.num_q))).astype(int)
                ceilq0=np.ceil(np.interp( q1, self.Q_grid, range(self.p.num_q))).astype(int)
                ceilq1=np.ceil(np.interp( q1, self.Q_grid, range(self.p.num_q))).astype(int)
                for iz in range(self.p.num_z):
                 for in0 in range(self.p.num_n):
                  if in0 == 0:
                     continue
                  for in1 in range(self.p.num_n):
                   for iv in range(self.p.num_v):
                     for iq in range(self.p.num_q):
                        if (floorq0[iz,in0,in1,iv,iq] == ceilq0[iz,in0,in1,iv,iq]):
                            if floorq0[iz,in0,in1,iv,iq] == 0:
                                RhoderivQ0[iz,in0,in1,iv,iq] = (ERho_star_0[iz,in0,in1,iv,1] - ERho_star_0[iz,in0,in1,iv,0]) / (self.Q_grid[1] - self.Q_grid[0])                             
                            else:
                                if floorq0[iz,in0,in1,iv,iq] == self.p.num_q-1:
                                      RhoderivQ0[iz,in0,in1,iv,iq] = (ERho_star_0[iz,in0,in1,iv,-1] - ERho_star_0[iz,in0,in1,iv,-2]) / (self.Q_grid[-1] - self.Q_grid[-2])
                                else:
                                        RhoderivQ0[iz,in0,in1,iv,iq] = (ERho_star_0[iz,in0,in1,iv,floorq0[iz,in0,in1,iv,iq]+1] - ERho_star_0[iz,in0,in1,iv,floorq0[iz,in0,in1,iv,iq]-1]) / (self.Q_grid[floorq0[iz,in0,in1,iv,iq]+1] - self.Q_grid[floorq0[iz,in0,in1,iv,iq]-1])
                            continue
                        if (floorq1[iz,in0,in1,iv,iq] == ceilq1[iz,in0,in1,iv,iq]):
                            if floorq1[iz,in0,in1,iv,iq] == 0:
                                RhoderivQ1[iz,in0,in1,iv,iq] = (ERho_star_1[iz,in0,in1,iv,1] - ERho_star_1[iz,in0,in1,iv,0]) / (self.Q_grid[1] - self.Q_grid[0])                             
                            else:
                                if floorq1[iz,in0,in1,iv,iq] == self.p.num_q-1:
                                      RhoderivQ1[iz,in0,in1,iv,iq] = (ERho_star_1[iz,in0,in1,iv,-1] - ERho_star_1[iz,in0,in1,iv,-2]) / (self.Q_grid[-1] - self.Q_grid[-2])
                                else:
                                        RhoderivQ1[iz,in0,in1,iv,iq] = (ERho_star_1[iz,in0,in1,iv,floorq1[iz,in0,in1,iv,iq]+1] - ERho_star_1[iz,in0,in1,iv,floorq1[iz,in0,in1,iv,iq]-1]) / (self.Q_grid[floorq1[iz,in0,in1,iv,iq]+1] - self.Q_grid[floorq1[iz,in0,in1,iv,iq]-1])
                            continue
                        RhoderivQ0[iz,in0,in1,iv,iq] = np.interp(ceilq0[iz,in0,in1,iv,iq],self.Q_grid, ERho_star_0[iz,in0,in1,iv,:]) - np.interp(floorq0[iz,in0,in1,iv,iq],self.Q_grid, ERho_star_0[iz,in0,in1,iv,:]) / (self.Q_grid[ceilq0[iz,in0,in1,iv,iq]]-self.Q_grid[floorq0[iz,in0,in1,iv,iq]])
                        RhoderivQ1[iz,in0,in1,iv,iq] = np.interp(ceilq1[iz,in0,in1,iv,iq],self.Q_grid, ERho_star_1[iz,in0,in1,iv,:]) - np.interp(floorq1[iz,in0,in1,iv,iq],self.Q_grid, ERho_star_1[iz,in0,in1,iv,:]) / (self.Q_grid[ceilq1[iz,in0,in1,iv,iq]]-self.Q_grid[floorq1[iz,in0,in1,iv,iq]]) 
                
                
                sep_star[...] = 0 #That way all the separations are normalized to zero first.
                Ifire = ( (-EJderiv0-rho_star*(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star*EWderiv0) + RhoderivQ0 * q0 / (self.N_grid[self.grid[1]]+self.N_grid[self.grid[2]]) - (EW1_star+re_star-EUi)/ self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW1_star + re_star)) > 0) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] <= self.p.n_bar) & (-(EJderiv1+rho_star*self.N_grid1[self.grid[2]]*pc_star*EWderiv1)*pc_star - (EW1_star+re_star-EUi)/ self.pref.inv_utility_1d(self.v_0-self.p.beta*(0.5 * EUi + 0.5 * (EW1_star + re_star))) + RhoderivQ1 * q1 / (self.N_grid[self.grid[1]] * self.p.q_0+self.N_grid1[self.grid[2]]) < 0 )
                worker_future_value = np.zeros_like(EW1i)
                for iz in range(self.p.num_z):
                 for in0 in range(self.p.num_n):
                  for in1 in range(self.p.num_n):
                   for iv in range(self.p.num_v):
                     for iq in range(self.p.num_q):
                      if Ifire[iz,in0,in1,iv,iq]:
                        worker_future_value[iz,in0,in1,iv,iq] = np.maximum(np.interp(rho_star[iz, in0, in1, iv,iq], rho_grid, re[iz,in0,in1,:,iq]+EW1i[iz,in0,in1,:,iq]),EUi)
                        #print("Worker future value:", worker_future_value[iz,in0,in1,iv,iq])
                #sep_star[Ifire] = 1-(EJinv0[Ifire]/((EUi-worker_future_value[Ifire]) / self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[Ifire]*EUi+(1-sep_star[Ifire])*worker_future_value[Ifire])))) #The thing is, this wouldn't work: t he corner case of ultra negative EJinv would suggest us negative separations, rather than 1       
                for idx in np.argwhere(Ifire):
                
                    sep_star[idx[0], idx[1], idx[2], idx[3], idx[4]] = 1- ( (self.Q_grid[idx[4]]*self.N_grid1[idx[2]]+self.p.q_0*self.N_grid[idx[1]]) * (Qderiv[idx[0], idx[1], idx[2], idx[3], idx[4]]-self.prod_qd[idx[0], idx[1], idx[2], idx[3], idx[4]]) / \
                    (((worker_future_value[idx[0], idx[1], idx[2], idx[3], idx[4]]-EUi) / (self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[idx[0], idx[1], idx[2], idx[3], idx[4]]*EUi+(1-sep_star[idx[0], idx[1], idx[2], idx[3], idx[4]])*worker_future_value[idx[0], idx[1], idx[2], idx[3], idx[4]])))+ EJinv[idx[0], idx[1], idx[2], idx[3], idx[4]]) * self.p.beta * self.N_grid1[idx[2]]) - self.N_grid1[idx[2]]) / self.N_grid[idx[1]]

                Icompletefire = ((-EJderiv0-rho_star*(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star*EWderiv0) + RhoderivQ0 * (self.N_grid[self.grid[1]]*q0) / (self.N_grid[self.grid[1]]+self.N_grid[self.grid[2]]) - (EW1_star+re_star-EUi)/ self.pref.inv_utility_1d(self.v_0-self.p.beta*(EW1_star + re_star)) > 0) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] <= self.p.n_bar) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] <= self.p.n_bar) & (-(EJderiv1+rho_star*self.N_grid1[self.grid[2]]*pc_star*EWderiv1)*pc_star - (EW1_star+re_star-EUi)/ self.pref.inv_utility_1d(self.v_0-self.p.beta*(0.5 * EUi + 0.5 * (EW1_star + re_star))) + RhoderivQ1 * q1 / (self.N_grid[self.grid[1]] * self.p.q_0+self.N_grid1[self.grid[2]]) >= 0)
                sep_star[Icompletefire] = 0.5

                #assert np.all(rho_star[Ifire]>rho_min)
                #assert np.all(worker_future_value[Ifire] > EUi)
                sepneg = ((-EJderiv0-rho_star*(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc_star*EWderiv0) + RhoderivQ0 * (self.N_grid[self.grid[1]]*q0) / (self.N_grid[self.grid[1]]+self.N_grid[self.grid[2]]) > 0) & (sep_star < 0)
                sep_star[sepneg] = 1
                sep_star = np.maximum(0.5, sep_star)
                seplarge = (sep_star>0.5)
                sep_star[seplarge] = 0.5
                sep_star[:,0,...] = 0 #This is only for now, as we're not considering separations for seniors
            #Getting n1_star
            if ite_num<=100000000:            
                n1_star = n1(pc, rho_grid, rho_star, sep_star, self.N_grid, self.N_grid1, self.p.num_z, self.p.num_n, self.p.num_v, self.p.num_q)
            else:
                n1_star = n1_tilde(n1_star,pc,rho_grid,rho_star, sep_star, self.N_grid,self.p.num_z, self.p.num_n, self.p.num_v)
            q_star = (self.p.q_0*self.N_grid[self.grid[1]]+self.Q_grid[self.grid[4]]*self.N_grid1[self.grid[2]])/(self.N_grid[self.grid[1]]*(1-sep_star)+self.N_grid1[self.grid[2]])
            print("q_star", q_star[self.p.z_0-1,1,0,50, :])
            #Getting hiring decisions
            Jd0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
            Wd0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v, self.p.num_q, self.p.num_n))
            n0_star[...] = 0
            if ite_num > 1:
                for iz in range(self.p.num_z):
                    for in00 in range(self.p.num_n):

                        J_interpolator = RegularGridInterpolator((self.N_grid1, rho_grid, self.Q_grid), EJpi[iz, in00, ...], bounds_error=False, fill_value=None)
                        W_interpolator = RegularGridInterpolator((self.N_grid1, rho_grid, self.Q_grid), EW1i[iz, in00, ...], bounds_error=False, fill_value=None)
                        Jd0[iz, ..., in00] = J_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                        Wd0[iz, ..., in00] = W_interpolator((n1_star[iz, ...], rho_star[iz, ...], q_star[iz, ...]))
                #Ihire = ((Jd0[...,1]-Jd0[...,0]+rho_star*n1_star*(Wd0[...,1]-Wd0[...,0])) > self.p.hire_c) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] < self.p.n_bar - 1)
                Ihire = ((Jd0[...,1]-Jd0[...,0]) / (self.N_grid[1]-self.N_grid[0]) > self.p.hire_c/self.p.beta) & (self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]] < self.p.n_bar - 1)

                n0_star = n0(Jd0, n0_star, self.N_grid, Ihire, self.p.hire_c / self.p.beta)

            EW1i_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), EW1i[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
            EJpi_interpolators = [RegularGridInterpolator((self.N_grid, self.N_grid1, rho_grid, self.Q_grid), EJpi[iz, ...], bounds_error=False, fill_value=None) for iz in range(self.p.num_z)]
            # Prepare points for interpolation
            rho_n_star_points = np.stack((n0_star, n1_star, rho_star, q_star), axis=-1)  # Shape: (num_z, ..., 2)
            # Vectorized interpolation over all iz
            EW1_star = np.array([interpolator(rho_n_star_points[iz, ...]) for iz, interpolator in enumerate(EW1i_interpolators)])
            EJ1_star = np.array([interpolator(rho_n_star_points[iz, ...]) for iz, interpolator in enumerate(EJpi_interpolators)])
            
            #EJ1_star, EW1_star = EJs(EJ1_star, EW1_star, Jd0, Wd0, n0_star, self.N_grid, self.p.num_z, self.p.num_n, self.p.num_v)
            #I'm interpolating EJ and EW functions on future q's in order to then use them for calculating future derivatives
            #Btw still super confused as to why the EJs function was somehow slower than the full complex interpolation
            EJq, EWq = EQs(EJq,EWq,EJpi,EW1i,q_star,self.Q_grid,self.p.num_z,self.p.num_n,self.p.num_v,self.p.num_q)
            #EW1_star = np.interp(n0_star, self.N_grid, Jd0[ax, ax, :, ax, :])
            #Getting the derivative of the future job value wrt n1:
            floorn1=np.floor(np.interp( n1_star, self.N_grid1, range(self.p.num_n))).astype(int)
            ceiln1=np.ceil(np.interp( n1_star, self.N_grid1, range(self.p.num_n))).astype(int)            
            #EJpi3 = EJpi+(self.N_grid[self.grid[1]]+self.N_grid1[self.grid[2]])*pc*rho_grid[ax,ax,ax,:]*EW1i
            #EJderiv0 = EJderivative3(EJpi,EW1i, floorn1,ceiln1,EJderiv,rho_grid,rho_star,self.p.num_z, self.p.num_n, self.p.num_v)
            EJderiv0 = EJderivative(EJq, floorn1,ceiln1,n0_star, EJderiv,rho_grid, self.N_grid, self.N_grid1, rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
            EWderiv = EWderivative(EWq,floorn1,ceiln1, n0_star, EWderiv,rho_grid, self.N_grid, self.N_grid1, rho_star,self.p.num_z, self.p.num_n, self.p.n_bar, self.p.num_v, self.p.num_q)
            EJderiv = EJderiv0+n1_star*rho_star*EWderiv
            #Ejderiv = EJderiv0#-rho_star*EW1_star


            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            _, re_star, pc_star = self.getWorkerDecisions(EW1_star)
            #print("states at which worker quits:", np.where(~(pc_star[self.p.z_0-1,1,1,:]==0)))
            # Update firm value function
            wage_jun = self.pref.inv_utility(self.v_0-self.p.beta*(sep_star*EUi+(1-sep_star)*(EW1_star+re_star)))
            print("wage_jun", wage_jun[self.p.z_0-1,1,0,50,0])
            print("wage jun no sep", self.pref.inv_utility(self.v_0-self.p.beta*(EW1_star[self.p.z_0-1,1,0,50,0]+re_star[self.p.z_0-1,1,0,50,0])))
            Ji = self.fun_prod*self.prod - sum_wage - self.p.hire_c*n0_star - \
                wage_jun*self.N_grid[self.grid[1]]  + self.p.beta * EJ1_star
            Ji = .2*Ji + .8*Ji2

            # Update worker value function
            W1i[...,1] = self.pref.utility(self.w_matrix[...,1]) + \
                self.p.beta * (re_star + EW1_star) #For more steps the ax at the end won't be needed as EW1_star itself will have multiple steps
            #W1i[:,:,0,:,1] = W1i[:,:,1,:,1]

            W1i[...,1:] = .4*W1i[...,1:] + .6*W1i2[...,1:] #we're completely ignoring the 0th step
            #print("Worker Value diff:", np.max(np.abs(W1i[:,:,:,:,1:]-W1i2[:,:,:,:,1:])))   
            _, ru, _ = self.getWorkerDecisions(EUi, employed=False)
            Ui = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EUi)
            Ui = 0.4*Ui + 0.6*Ui2

            if ite_num>1:
                print("sep borders", sep_star.min(), sep_star.max())
            # Updating J1 representation
            #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[:,:,:,:,1], Ji)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_w1 = array_dist(W1i[...,1:], W1i2[...,1:])

            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------

                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                    print("Errors:",  error_j1i, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            #P_xv = self.matching_function(J1p.eval_at_W1(W1i)[self.p.z_0-1, 0, 1, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            #error_js = self.js.update(W1i[self.p.z_0-1, 0, 0, :, 1], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[...,1], Ji)
                    #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                    print("Errors:",  error_j1i,  error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            #if (ite_num % 25) == 0:
                # Updating J1 representation
                #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[...,1], Ji)
                #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[...,1]), 100)
                #print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)    
                #self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                #                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        #self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
        #                             ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        return Ji,W1i,EW1_star,sep_star, n0_star, n1_star


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
        :return: pe,re,qi search decision and associated return, as well as quit decision.
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
#mwc=MultiworkerContract(p)
#(mwc_J,mwc_W,mwc_Wstar,mwc_pc,mwc_n1)=mwc.J()