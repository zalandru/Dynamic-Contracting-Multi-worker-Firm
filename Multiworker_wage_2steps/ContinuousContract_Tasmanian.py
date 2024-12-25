import numpy as np
import logging
from scipy.stats import lognorm as lnorm
import matplotlib.pyplot as plt

import opt_einsum as oe
import Tasmanian

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
       # self.log = logging.getLogger('ContinuousContract')
       # self.log.setLevel(logging.INFO)

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
        
        self.w_grid_bas = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v )
        self.rho_grid_bas=1/self.pref.utility_1d(self.w_grid_bas)

        #Tasmanian Setup
        a = (self.rho_grid_bas[0])
        b = (self.rho_grid_bas[-1])
        # Create a global sparse grid
        dim = 1
        outputs = 1
        depth = 6
        type = "level"
        rule = "rleja"
        self.grid = {}
        self.gridE = {}

        self.grid[self.p.z_0-1] = Tasmanian.TasmanianSparseGrid()
            # Using a global polynomial rule for demonstration:
        #self.grid[self.p.z_0-1].makeGlobalGrid(dim, outputs, depth, type, rule)
        self.grid[self.p.z_0-1].makeLocalPolynomialGrid(dim, outputs, depth)
            #Can also consider trying different depth options rather than "level", see https://ornl.github.io/TASMANIAN/stable/group__SGEnumerates.html#ga145e27d5ae92acdd5f74149c6d4f2ca2
            #Good methods: "clenshaw-curtis", rleja" maybe?, "gauss-patterson" kinda overdoes it, "leja" is good but requires higher depth (like 10)... still not many points though!
        self.grid[self.p.z_0-1].setDomainTransform(np.column_stack((a, b)))
        #Tring out just a single grid for the expectation
        for iz in range(self.p.num_z):
            self.gridE[iz] = Tasmanian.TasmanianSparseGrid()
        # Using a global polynomial rule for demonstration:
            #self.gridE[iz].makeGlobalGrid(dim, outputs, depth, type, rule)
            self.gridE[iz].makeLocalPolynomialGrid(dim, outputs, depth)
            self.gridE[iz].setDomainTransform(np.column_stack((a, b)))
        # Get the points where Tasmanian wants J evaluated
        self.points = self.grid[self.p.z_0-1].getPoints()  # shape = (N,4)
        print(self.points.shape)
        self.sorted_indices = np.argsort(self.points[:,0])
        self.inverse_indices = np.zeros_like(self.sorted_indices)
        self.inverse_indices[self.sorted_indices] = np.arange(len(self.sorted_indices))
        self.points_sorted = self.points[self.sorted_indices]
        #So now, we already have the values rho_grid at each of these points! They're points[:,0]
        #But should I work with Rho instead of J? Since the Tasmanian should operate on a constant grid, defining it through J may be bad... but let's try it! lol

        # Value Function Setup
        self.J_grid   = -10 * np.ones((self.p.num_z, self.points.shape[0])) #grid of job values, first productivity, then starting value, then tenure level

        self.rho_grid = self.points_sorted[:,0]
        self.w_grid = self.rho_grid #Due to log utlity!

        #Gotta fix the tightness+re functions somehow. Ultra simple J maybe?
        #self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        self.v_grid = np.divide(self.pref.utility(self.w_grid),1-self.p.beta)
        self.simple_J=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)

        self.v_grid_bas = np.divide(self.pref.utility(self.w_grid_bas),1-self.p.beta)
        self.simple_J_bas=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.v_grid_bas[ax,:]*(1-self.p.beta)),1-self.p.beta)    
        #Loading the grid
        # Load these values into the Tasmanian grid
        #for iz in range(self.p.num_z):        
        self.grid[self.p.z_0-1].loadNeededPoints(self.simple_J[self.p.z_0-1,self.inverse_indices,ax])
        
        #Apply the matching function: take the simple function and consider its different values across v.
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J_bas[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
        self.js.update(self.v_grid_bas[:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        print(self.js.jsa.e0)
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

        # construct the continuation probability. #Andrei: probability the worker doesn't get fired and also doesn't leave
        pc = (1 - pe)

        return pe, re, pc
    def matching_function(self,J1): #Andrei: the formula of their matching function, applied to each particula job value J1
        return self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(J1, self.p.kappa)), self.p.sigma),
                                1 / self.p.sigma)

    def J(self,update_eq=0):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        # create representation for J1p
        w_grid=self.w_grid
        rho_grid=self.rho_grid
        Ji= self.simple_J
        W1i=np.zeros((self.p.num_z, self.rho_grid.shape[0]))
        W1i=W1i+self.v_grid[ax,:]
        U = self.pref.utility(self.unemp_bf) / (1 - self.p.beta)
        #J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py
        #W1i[ Ji < 0 ] = U
        #Ji[ Ji < 0 ] = 0  
        print(Ji.shape)
        print(W1i.shape)

        EW1_star = np.copy(Ji)
        EJ1_star = np.copy(Ji)
        Jpi = np.copy(Ji)


        rho_bar = np.zeros((self.p.num_z))
        rho_star = np.zeros((self.p.num_z, self.rho_grid.shape[0]))

        # prepare expectation call
        Exz = oe.contract_expression('av,az->zv', W1i.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1
        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = W1i
            U2 = np.copy(U)

            # evaluate J1 tomorrow using our approximation
            #if ite_num>=0:
                #for iz in range(self.p.num_z):
                    #Jpi[iz,:] = self.grid[iz].evaluateBatch(self.points_sorted)[:,0]
                #    Jpi[iz,:] = self.grid[iz].getLoadedValues()[self.sorted_indices,0]
            #    Jpi[iz,:] = self.grid[iz].getValues()
            #Jpi = J1p.eval_at_W1(W1i)
            #print("Jpi-Ji max:", np.max(np.abs(Jpi-Ji)))
            # we compute the expected value next period by applying the transition rules
            EW1i = Exz(W1i, self.Z_trans_mat)
            EJpi = Exz(Ji, self.Z_trans_mat)
            EU = U

            for iz in range(self.p.num_z):
                self.gridE[iz].loadNeededPoints(EJpi[iz, self.inverse_indices, ax])

            #EW1i = W1i
            #EJpi = Jpi
            # get worker decisions
            _, _, pc = self.getWorkerDecisions(EW1i)
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps) 

            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
            foc = rho_grid[ax, :] - EJpi * log_diff / self.deriv_eps #So the FOC wrt promised value is: pay shadow cost lambda today (rho_grid), but more likely that the worker stays tomorrow
            # foc = 1/u'(w')-eta(v')EJpi = 1/u'(w)
            assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"


            for iz in range(self.p.num_z):

                assert np.all(EW1i[iz, 1:] >= EW1i[iz, :-1]) #Andrei: check that worker value is increasing in v
                    # find highest V with J2J search
                rho_bar[iz] = np.interp(self.js.jsa.e0, EW1i[iz, :], rho_grid) #Andrei: interpolate the rho_grid, aka the shadow cost, to the point where the worker no longer searches
                #print("e0", self.js.jsa.e0)
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
                #EJ1_star[iz, :] = np.interp(rho_star[iz, :], rho_grid, EJpi[iz, :]) #Andrei: how does interpolating the shadow cost give us the future Value?
                #Trying evaluate instead! AHHHHHHHHHH shit! Only works with EJ! Okay, then for now let's just do EJ.
                #J_d = np.zeros_like(Ji)
                #J_int = np.zeros_like(Ji)
                #for izz in range(self.p.num_z):
                #    J_d[izz,:] = self.grid[izz].evaluateBatch(rho_star[iz,:,ax])[:,0]
                #    J_int[izz,:] = np.interp(rho_star[iz, :], rho_grid, Ji[izz, :])
                #print("Difference btw batch eval and interpolation", np.abs(J_d-J_int).max())
                #EJ1_star_d = Exz(J_d, self.Z_trans_mat)
                #print("Expectation diff", np.abs((self.gridE[iz].evaluateBatch(rho_star[iz,:,ax])[:,0]-EJ1_star[iz,:])).max())
                #print("Expectation diff of interpolation order", np.abs((Exz(J_int, self.Z_trans_mat)[iz,:]-EJ1_star[iz,:])).max())

                EJ1_star[iz, :] = self.gridE[iz].evaluateBatch(rho_star[iz,:,ax])[:,0]
                

            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            # get pstar, qstar
            pe_star, re_star, pc_star = self.getWorkerDecisions(EW1_star)

            _, ru, _ = self.getWorkerDecisions(EU, employed=False)
            U = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EU)
            U = 0.4 * U + 0.6 * U2

            # Update firm value function 
            #Andrei: why is the w_grid still preset? Doesn't it depend on what you promised to the worker?
            #Andrei: also, why do we still use this EJ1_star as the future value rather than just the actual value?
            Ji = self.fun_prod[:, ax] - w_grid[ax, :] + self.p.beta * (1 - pe_star) * EJ1_star
            Ji = .2 * Ji + .8 * Ji2
            #print("Value diff:", np.max(np.abs(Ji-Ji2)))
            # Update worker value function
            W1i = self.pref.utility(w_grid)[ax, :] + \
                self.p.beta * (re_star + EW1_star)
            #plt.plot(W1i[self.p.z_0-1, :], pe_star[self.p.z_0-1, :], label='Probability of the worker leaving across submarkets')      
            #plt.show()
            W1i = .4 * W1i + .6 * W1i2

            #Firm exit:
            #W1i[ Ji < 0 ] = U
            #Ji[ Ji < 0 ] = 0
            # Updating J1 representation
  
            J_inverted = Ji[self.p.z_0-1, self.inverse_indices, ax]
            self.grid[self.p.z_0-1].loadNeededPoints(J_inverted)
            #error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i, Ji)

            #Surplus refinement: could be useful, but because here I'm working with both J and EJ, this may become a mess.
            #self.grid[self.p.z_0-1].setSurplusRefinement(1e-20,-1,'classic')
            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            #error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i), 100)
            error_w1 = array_dist(W1i, W1i2)
            #print("Error:", error_j1i)
            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------
                    if (np.array([error_j1i,error_w1, error_js]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        Wrange = np.interp(self.rho_grid_bas,rho_grid,W1i[self.p.z_0-1,:])
                        P_xv = self.matching_function(self.grid[self.p.z_0-1].evaluateBatch(self.rho_grid_bas[:,ax])[:,0]) #This currently isn't evaluating outside the original points!
                        plt.plot(Wrange, P_xv, label='Probability of finding a job across submarkets')      
                        plt.show()
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            Wrange = np.interp(self.rho_grid_bas,rho_grid,W1i[self.p.z_0-1,:])
                            P_xv = self.matching_function(self.grid[self.p.z_0-1].evaluateBatch(self.rho_grid_bas[:,ax])[:,0]) #This currently isn't evaluating outside the original points!

                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(Wrange, P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    #plt.plot(W1i[self.p.z_0-1, :],Ji[self.p.z_0-1,:], label='Value function')      
                    #plt.show()
                    if (np.array([error_j1i,error_w1]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            #if (ite_num % 25) == 0:
             #   self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
            #                         ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        #self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
        #                             ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        return Ji,W1i,EW1_star, rho_star, Jpi,pc_star


    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)