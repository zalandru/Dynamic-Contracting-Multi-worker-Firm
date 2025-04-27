import numpy as np
import logging
from scipy.stats import lognorm as lnorm
import matplotlib.pyplot as plt

import opt_einsum as oe

from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search import JobSearchArray
from valuefunction2d import PowerFunctionGrid
from scipy.optimize import minimize
ax = np.newaxis

def impose_decreasing(M):
    nv = M.shape[1]
    for v in reversed(range(nv-1)):
        M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
    return M
def impose_increasing(A0):
    A = np.copy(A0)
    nv = len(A)
    for v in range(1,nv):
        A[v] = np.maximum(A[v],A[v-1])
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
def array_dist(A,B):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    return  (np.power( A-B,2) ).mean() / ( np.power(B,2) ).mean()

class ContinuousContract:
    """
        This solves a classic contract model.
    """
    def __init__(self, input_param=None):

        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        self.log = logging.getLogger('ContinuousContract')
        self.log.setLevel(logging.INFO)

        self.p = input_param
        self.deriv_eps = 1e-4 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid

        # Production Function in the Model
        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = np.ones(self.p.num_x) * self.p.u_bf_m

        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        self.J_grid   = -10 * np.ones((self.p.num_z, self.p.num_v)) #grid of job values, first productivity, then starting value, then tenure level
        self.w_grid = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v )
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)
        #Gotta fix the tightness+re functions somehow. Ultra simple J maybe?
        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        self.simple_J=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
        self.js.update(self.v_grid[:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        #self.re=self.js.re
        #self.pc = self.getWorkerDecisions(self.simple_v_grid[ax, :,ax]) #shit, re is an array, not a function!! why???
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

    def J(self,update_eq=1):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
                # create representation for J1p
        w_grid=self.w_grid
        rho_grid=self.rho_grid
        Ji= self.simple_J
        W1i=np.zeros((self.p.num_z, self.p.num_v))
        W1i=W1i+self.v_grid[ax,:]

        J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py

        EW1_star = np.copy(Ji)
        EJ1_star = np.copy(Ji)

        rho_bar = np.zeros((self.p.num_z))
        rho_star = np.zeros((self.p.num_z, self.p.num_v))

        # prepare expectation call
        Exz = oe.contract_expression('av,az->zv', W1i.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1
        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = W1i

            # evaluate J1 tomorrow using our approximation
            Jpi = J1p.eval_at_W1(W1i)
            #print("Jpi-Ji max:", np.max(np.abs(Jpi-Ji)))
            # we compute the expected value next period by applying the transition rules
            EW1i = Exz(W1i, self.Z_trans_mat)
            EJpi = Exz(Jpi, self.Z_trans_mat)
            #EW1i = W1i
            #EJpi = Jpi
            # get worker decisions
            _, _, pc = self.getWorkerDecisions(EW1i)
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps) 

            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            #print("Shape of pc:", pc.shape)
            #print("Shape of pc_d:", pc_d.shape if 'pc_d' in locals() else "pc_d not defined")
            #print("Shape of log_diff:", log_diff.shape if 'log_diff' in locals() else "log_diff not defined")
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
            foc = rho_grid[ax, :] - EJpi * log_diff / self.deriv_eps #So the FOC wrt promised value is: pay shadow cost lambda today (rho_grid), but more likely that the worker stays tomorrow
            assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"


            for iz in range(self.p.num_z):

                assert np.all(EW1i[iz, 1:] >= EW1i[iz, :-1]) #Andrei: check that worker value is increasing in v
                    # find highest V with J2J search
                rho_bar[iz] = np.interp(self.js.jsa.e0, EW1i[iz, :], rho_grid) #Andrei: interpolate the rho_grid, aka the shadow cost, to the point where the worker no longer searches
                rho_min = rho_grid[pc[iz, :] > 0].min()  # lowest promised rho with continuation > 0
                    #Andrei: so we look for the shadow cost that will satisfy the foc? Yes, look for u'(w'), with u'(w) given, so that the foc is satisfied
                    # look for FOC below  rho_0
                Isearch = (rho_grid <= rho_bar[iz]) & (pc[iz, :] > 0) #Okay, I think this is the set of points (of promised value v) such that these conditions hold
                if Isearch.sum() > 0:
                      rho_star[iz, Isearch] = np.interp(rho_grid[Isearch],
                                                              impose_increasing(foc[iz, Isearch]),
                                                              rho_grid[Isearch], right=rho_bar[iz])

                    # look for FOC above rho_0
                Ieffort = (rho_grid > rho_bar[iz]) & (pc[iz, :] > 0)
                if Ieffort.sum() > 0:
                        #assert np.all(foc[iz, Ieffort, ix][1:] > foc[iz, Ieffort, ix][:-1])
                         rho_star[iz, Ieffort] = np.interp(rho_grid[Ieffort],
                                                              foc[iz, Ieffort], rho_grid[Ieffort])
                    #Andrei: so this interpolation is: find the rho_grid value such that foc=rho_grid?
                    #Let's try to be more precise here: for each v_0 in Ieffort, we want rho_star=rho_grid[v'] such that foc[v']=rho_grid[v_0]
                    # set rho for quits to the lowest value
                Iquit = ~(pc[iz, :] > 0) 
                if Iquit.sum() > 0:
                           rho_star[iz, Iquit] = rho_min

                    # get EW1_Star and EJ1_star
                EW1_star[iz, :] = np.interp(rho_star[iz, :], rho_grid, EW1i[iz, :])
                EJ1_star[iz, :] = np.interp(rho_star[iz, :], rho_grid, EJpi[iz, :]) #Andrei: how does interpolating the shadow cost give us the future Value?
                    #Andrei: rather, we're interpolating the Job value at the point of the optimal shadow cost. still confused as to why its a shadow cost rather than lambda
                    #Or, more like, we're interpolating EJpi to the value where the shadow cost is the optimal one, aka rho_star/
                    #Basically, fixing today's promised value, we find the future value that will be optimal via  the shadow cost, and interpolate the expected value at the point of the optimal shadow cost
            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            # get pstar, qstar
            pe_star, re_star, _ = self.getWorkerDecisions(EW1_star)
            #print("Expectation diff:", np.max(np.abs(EJ1_star-(Ji-self.fun_prod[:,ax]+w_grid[ax,:])/(self.p.beta*(1-pe_star)))))
            # Update firm value function 
            #Andrei: why is the w_grid still preset? Doesn't it depend on what you promised to the worker?
            #Andrei: also, why do we still use this EJ1_star as the future value rather than just the actual value?
            Ji = self.fun_prod[:, ax] - w_grid[ax, :] + self.p.beta * (1 - pe_star) * EJ1_star
            #print("Value diff:", np.max(np.abs(Ji-Ji2)))
            # Update worker value function
            W1i = self.pref.utility(w_grid)[ax, :] + \
                self.p.beta * (re_star + EW1_star)
            W1i = .2*W1i + .8*W1i2
            #Ji = .4*Ji+.6*Ji2
            # Updating J1 representation
            error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i, Ji)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i), 100)
            error_w1 = array_dist(W1i, W1i2)
            #print("Error:", error_j1i)
            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------
                    if (np.array([error_w1, error_js, error_j1p_chg]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            P_xv = self.matching_function(J1p.eval_at_W1(W1i)[self.p.z_0-1, :])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(W1i[self.p.z_0-1, :], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    if (np.array([error_w1, error_j1g]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            if (ite_num % 25) == 0:
                self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        return Ji,W1i,EW1_star,Jpi


    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)