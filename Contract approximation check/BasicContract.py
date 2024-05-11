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
        This solves a version of the model with no aggregate risk, no endogenous job destruction, and just one worker (CRS production!).
    """

    def __init__(self,num_K, js, input_param=None):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        self.log = logging.getLogger('BasicContract')
        self.log.setLevel(logging.INFO)

        self.p = input_param
        self.num_K = num_K
        self.js = js
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
        self.J_grid   = -10 * np.ones((self.p.num_z, self.p.num_v0, self.num_K)) #grid of job values, first productivity, then starting value, then tenure level
        self.v_grid_0 = np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v0 ) #grid of possible starting job values, based on a grid of possibe wages, assuming those are constant over time
        self.v_grid_0 = np.linspace(50.0, 90.0, self.p.num_v0 ) #grid of possible starting job values, based on a grid of possibe wages, assuming those are constant over time

        #Gotta fix the tightness+re functions somehow. Ultra simple J maybe?
        #self.simple_v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v_simple ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #self.simple_J=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.simple_v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        #self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
        #    np.divide(self.p.kappa, np.maximum(self.simple_J[0, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        #self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
        #self.js.update(self.simple_v_grid[ax,:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        #self.re=self.js.re
        #self.pc = self.getWorkerDecisions(self.simple_v_grid[ax, :,ax]) #shit, re is an array, not a function!! why???
    
    def J_r(self,r):
        """
        Computes the value of a job with K tenure levels with ratios r
        :param K: number of tenure steps
        :param r: ratio of values between the steps, allowed different values depending on different intial promise. Thus r.shape=(num_v0,num_K)
        :return: value of the job
        """
        v_grid_0 = self.v_grid_0

        # Setting up the initial values for the VFI
        J =  self.J_grid
        #w_grid is outside of the iteration becaause we're in a partial equailibrium: pe and re are exogenous, and thus don't depend on J.    
        w_grid=np.zeros((self.p.num_z,self.p.num_v0,self.num_K)) #first productivity, then starting value, then tenure level
        r_shifted=self.shift(r)
        w_grid[0,:,:]=self.pref.inv_utility(v_grid_0[:,ax]*r[ax,:]-self.p.beta*(v_grid_0[:,ax]*r_shifted[ax,:]+self.js.re(v_grid_0[:,ax]*r_shifted[ax,:]))) #so R is the search continuation value, depends on both the original v and the r.
        #print("w_grid:", w_grid) #so r is iultra big because re is still positive somehow??? for superlarge r that thing should be zero, but is somehow positive. Why?
                 #Need to introduce (1-pc(W1i[:,1]*r_k)) here. But wait, what do I do do with the other W1i's? Guess I never need them huh...
        #print("fun_prod shape:", self.fun_prod.shape)
        #print("w_grid shape:", w_grid.shape)
        #print("r_shifted shape:", r_shifted.shape)
        J[:,:,-1]=np.divide(self.fun_prod[:,ax] - w_grid[:,:,-1],1-self.p.beta  *(1-self.js.pe(v_grid_0[ax,:]*r[ax,ax,-1])))
        for k in range(J.shape[2]-2, -1, -1):
            J[:,:,k]=self.fun_prod[:,ax] - w_grid[:,:,k] + self.p.beta  *(1-self.js.pe(v_grid_0[ax,:]*r[ax,ax,k+1]))*J[:,:,k+1]

        #Introduce optimal r right here!
        return -J
    

    #Alternative: I don't think I can start with a preset wage, especially one that's independent from r. (Is that always an issue? Or only here?)
    #Okay, that's an important quesiton. In the big model, can I work with a grid of u(w) values that are independent of r?
    #I guess with v'moving about it's still possible? That each wage on the grid is optimal for some current v. And those v's will be affected by r, so I guess that's fine.
    #Will the idea still make sense though? Like the r optimization. How do we optimize r if the first grid point is always the same?
    #Maybe that condition I had in mind? That the first promised value is always the same? So that r only gives us ratios, not the first bottom value.
    #I can try that later.
    #Wait, here... I can do w_grid, it's just that... it ain't an actual grid lmao. We're just connecting wages across k points.
    #or... we can make it a grid, thus creating this contract for a variety of starting wages.
    #in that case, define w_grid_0=np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v ). That is simply a set of positive starting values. May start with just 1 point though.
    #then get the full w_grid from the formula below:
    # w_grid[:,k+1]=u-1(u([w_grid[:,k])+Wi[:,1]*(r[k+1]-r[k])*(1-beta*(1-pc[:,k+1])))
    #wtf do we do with pc here though? guess it's also defined on the grid? How exactly though?
    #Also, even though v'and v are the same here, they still affect that expression. Hencewhy I add Wi[:,1], that's the v=v'.

    

    def J_K(self):
        """
        Computes the value of a job with K tenure levels, by choosing the ratios r
        :param K: number of tenure steps
        :return: value of the job
        """
        initial_r = np.ones((self.num_K)) #first starting value, then tenure level
        #initial_r[0:]=1.5
        def J_0(r,v0): #Provides value of the firm at the starting point(s)
            result=self.J_r(r)[0,v0,0]
            #print("Result:", result)  # Check what is being returned
            #print("Type of result:", type(result))  # This should be 'numpy.float64' or similar for a scalar
            return result
        #print(type(J_0))  # This should output <class 'function'>
        bounds = [(1, 2) for _ in range(self.num_K)]
        r_star=np.zeros((self.v_grid_0.size,self.num_K))
        for v0 in range(self.p.num_v0):
            r_star[v0,:]=minimize(lambda x,v=v0: J_0(x,v),initial_r,bounds=bounds).x #we assume the worker starts at the bottom (thus the last dimenision is 0) and we want to optimize firm's value of holding that worker
        
        return (r_star,[-self.J_r(r_star[v0,:])[:,v0,:] for v0 in range(self.p.num_v0)]) #Andrei: Return the optimal r and the value of the job
    
    def shift(self,f):
        f_shifted=np.zeros(f.shape)
        if len(f.shape)==3: #to shift J
            f_shifted[:,:,:-1]=f[:,:,1:]
            f_shifted[:,:,-1]=f[:,:,-1]
        else: #to shift r
            f_shifted[:-1]=f[1:]
            f_shifted[-1]=f[-1]        
        return f_shifted
    
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

        # construct the continuation probability. #Andrei: probability the worker doesn't get fired and also doesn't leave
        pc = (1 - pe)

        return re, pc
    def construct_z_grid(self):


        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)
    
        for ite_num in range(2*self.p.max_iter):
            J2 = J


            J_shifted=self.shift(J)
            J=self.fun_prod[:,ax,ax] - w_grid + self.p.beta  *(1-self.js.pe(v_grid_0[ax,:,ax]*r_shifted[ax,ax,:]))*J_shifted
            #J = impose_decreasing(J)

            #Also is this correct? We fix the wage in the first point, but not the future promise... is that ok? 
            #No, it's not because we directly minimize wrt r instead of using some kinda FOCs. Therefore, in the current setting, r literally lowers workers current value, which may be profitable.
            #So how do I get around that? Do I fix the first value? I guess I essentially do that already.
            #So, here I iterate over both worker and firm values?
            error_j  = np.max(abs(J - J2))


            if np.array([error_j]).max() < self.p.tol_simple_model and ite_num>10:
                break #Andrei: Break if the error is small enough

            if (ite_num % 25 ==0): #Andrei: Log every 25 iterations
                self.log.debug('[{}]  Error_J = {:2.4e}'.format(ite_num, error_j))

            self.log.info('[{}]  Error_J = {:2.4e}' .format(ite_num, error_j))
        
        #print("Shape of J:", J.shape)  # This should print something like (p.num_z, p.num_v0, num_K)