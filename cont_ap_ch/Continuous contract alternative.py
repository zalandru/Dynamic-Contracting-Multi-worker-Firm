import numpy as np
import logging
from scipy.stats import lognorm as lnorm
import matplotlib.pyplot as plt

import opt_einsum as oe

from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search import JobSearchArray
from scipy.optimize import minimize
ax = np.newaxis

def impose_decreasing(M):
    nv = M.shape[1]
    for v in reversed(range(nv-1)):
        M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
    return M

class BasicContract:
    """
        This solves a classic contract model.
    """

    def __init__(self, input_param=None):

        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        self.log = logging.getLogger('BasicContract')
        self.log.setLevel(logging.INFO)

        self.p = input_param
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid

        # Production Function in the Model
        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = np.ones(self.p.num_x) * self.p.u_bf_m

        # Transition matrices
        self.X_trans_mat = createBlockPoissonTransitionMatrix(self.p.num_x/self.p.num_np,self.p.num_np, self.p.x_corr)
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        self.J_grid   = -10 * np.ones(self.p.num_z, self.p.num_v) #grid of job values, first productivity, then starting value, then tenure level

        #Gotta fix the tightness+re functions somehow. Ultra simple J maybe?
        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v_simple ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        self.simple_J=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
        self.js.update(self.simple_v_grid[ax,:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        #self.re=self.js.re
        #self.pc = self.getWorkerDecisions(self.simple_v_grid[ax, :,ax]) #shit, re is an array, not a function!! why???

    def J(self):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        v_grid = self.v_grid

        # Setting up the initial values for the VFI
        J =  self.J_grid

        #w_grid is outside of the iteration becaause we're in a partial equailibrium: pe and re are exogenous, and thus don't depend on J.    
       
        #w_grid[0,:]=self.pref.inv_utility(v_grid[:]-self.p.beta*(v_grid[:,ax]*r_shifted[ax,:]+self.js.re(v_grid[:,ax]*r_shifted[ax,:]))) #so R is the search continuation value, depends on both the original v and the r.
        #print("w_grid:", w_grid)

        for ite_num in range(2*self.p.max_iter):
            J2 = J


            J_shifted=self.shift(J)
            J=self.fun_prod[:,ax,ax] - self.pref.inv_utility(v_grid[:]-self.p.beta*(v_grid[:,ax]*r_shifted[ax,:]+self.js.re(v_grid[:,ax]*r_shifted[ax,:]))) + self.p.beta  *(1-self.js.pe(v_grid_0[ax,:,ax]*r_shifted[ax,ax,:]))*J_shifted
            #J = impose_decreasing(J)


            error_j  = np.max(abs(J - J2))


            if np.array([error_j]).max() < self.p.tol_simple_model and ite_num>10:
                break #Andrei: Break if the error is small enough

            if (ite_num % 25 ==0): #Andrei: Log every 25 iterations
                self.log.debug('[{}]  Error_J = {:2.4e}'.format(ite_num, error_j))

            self.log.info('[{}]  Error_J = {:2.4e}' .format(ite_num, error_j))
        
        return -J


    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)