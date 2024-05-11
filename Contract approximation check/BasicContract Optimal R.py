import numpy as np
import logging
from scipy.stats import lognorm as lnorm
import matplotlib.pyplot as plt

import opt_einsum as oe
from valuefunction import PowerFunctionGrid
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
        # Wage Grid
        #self.w_grid=np.zeros((self.p.num_z,self.p.num_v0,self.num_K)) #first productivity, then starting value, then tenure level
        self.w_grid = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v )
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)
        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v_simple ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!

        # Transition matrices
        self.X_trans_mat = createBlockPoissonTransitionMatrix(self.p.num_x/self.p.num_np,self.p.num_np, self.p.x_corr)
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        self.J_grid   = -10 * np.ones((self.p.num_z, self.p.num_v0, self.num_K)) #grid of job values, first productivity, then starting value, then tenure level
        self.v_grid_0 = np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v0 ) #grid of possible starting job values, based on a grid of possibe wages, assuming those are constant over time
        self.v_grid_0 = np.linspace(50.0, 90.0, self.p.num_v0 ) #grid of possible starting job values, based on a grid of possibe wages, assuming those are constant over time
        self.simple_J=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)
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
    
    def J_K(self,update_eq=False):
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
                # create representation for J1p
        w_grid=self.w_grid
        rho_grid=self.rho_grid
        Ji=np.zeros((self.p.num_z,self.p.num_v0,self.num_K))
        W1i=np.zeros((self.p.num_z,self.p.num_v0,self.num_K))
        Ji= self.simple_J[:,:,ax]
        W1i=self.v_grid[ax,:,ax]

        J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py

        EW1_star = np.copy(Ji)
        EJ1_star = np.copy(Ji)

        rho_bar = np.zeros((self.p.num_z,self.num_K))
        rho_star = np.zeros((self.p.num_z, self.p.num_v,self.num_K))
        log_diff = np.zeros_like(EW1_star)


        ite_num = 0
        error_js = 1
        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = W1i
            Jpi = J1p.eval_at_W1(W1i)

            # we compute the expected value next period by applying the transition rules
            EW1i = W1i
            EJpi = Jpi
            #print("Shape of EW1i:", EW1i.shape)
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
            foc = rho_grid[ax, :,ax] - EJpi * log_diff / self.deriv_eps #So the FOC wrt promised value is: pay shadow cost lambda today (rho_grid), but more likely that the worker stays tomorrow
            #Andrei: do we need the k-dimension for FOC? do we need it for rho_grid?
            assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"


            for iz in range(self.p.num_z):
             for ik in range(self.num_K):
                assert np.all(EW1i[iz, 1:,ik] >= EW1i[iz, :-1,ik]) #Andrei: check that worker value is increasing in v
                    # find highest V with J2J search
                rho_bar[iz,ik] = np.interp(self.js.jsa.e0, EW1i[iz, :,ik], rho_grid) #Andrei: interpolate the rho_grid, aka the shadow cost, to the point where the worker no longer searches
                rho_min = rho_grid[pc[iz, :,ik] > 0].min()  # lowest promised rho with continuation > 0
                    #Andrei: so we look for the shadow cost that will satisfy the foc? Yes, look for u'(w'), with u'(w) given, so that the foc is satisfied
                    # look for FOC below  rho_0       
                Isearch = (rho_grid <= rho_bar[iz,ik]) & (pc[iz, :,ik] > 0) #Okay, I think this is the set of points (of promised value v) such that these conditions hold
                if ik==0:
                 if Isearch.sum() > 0:
                      rho_star[iz, Isearch,ik] = np.interp(rho_grid[Isearch],
                                                              impose_increasing(foc[iz, Isearch,ik]),
                                                              rho_grid[Isearch], right=rho_bar[iz,ik])

                    # look for FOC above rho_0
                 Ieffort = (rho_grid > rho_bar[iz,ik]) & (pc[iz, :,ik] > 0)
                 if Ieffort.sum() > 0:
                        #assert np.all(foc[iz, Ieffort, ix][1:] > foc[iz, Ieffort, ix][:-1])
                         rho_star[iz, Ieffort,ik] = np.interp(rho_grid[Ieffort],
                                                              foc[iz, Ieffort,ik], rho_grid[Ieffort])
                    # set rho for quits to the lowest value
                 Iquit = ~(pc[iz, :] > 0) 
                 if Iquit.sum() > 0:
                           rho_star[iz, Iquit,ik] = rho_min
                else:
                 if Isearch.sum() > 0:
                      rho_star[iz, Isearch,ik] = np.interp(rho_star[iz,Isearch,ik-1],
                                                              impose_increasing(foc[iz, Isearch]),
                                                              rho_grid[Isearch], right=rho_bar[iz,ik])

                    # look for FOC above rho_0
                 Ieffort = (rho_grid > rho_bar[iz,ik]) & (pc[iz, :,ik] > 0)
                 if Ieffort.sum() > 0:
                        #assert np.all(foc[iz, Ieffort, ix][1:] > foc[iz, Ieffort, ix][:-1])
                         rho_star[iz, Ieffort,ik] = np.interp(rho_star[iz,Isearch,ik-1],
                                                              foc[iz, Ieffort], rho_grid[Ieffort])
                    #Andrei: so this interpolation is: find the rho_grid value such that foc=rho_grid?
                    #Let's try to be more precise here: for each v_0 in Ieffort, we want rho_star=rho_grid[v'] such that foc[v']=rho_grid[v_0]
                    # set rho for quits to the lowest value
                 Iquit = ~(pc[iz, :] > 0) 
                 if Iquit.sum() > 0:
                           rho_star[iz, Iquit,ik] = rho_min                
                    # get EW1_Star and EJ1_star
                w_star=np.zeros((self.p.num_z,self.p.num_v,self.num_K))
                w_star[iz, :,ik] = np.interp(rho_star[iz, :,ik], rho_grid, w_grid)
            Ji[:,:,-1]=np.divide(self.fun_prod[:,ax] - w_star[:,:,-1],1-self.p.beta  *(1-self.js.pe(EW1i[:,:,-1])))
            W1i[:,:,-1]=np.divide(self.pref.utility(w_star)[:,:,-1],1- self.p.beta * (1-self.js.pe(EW1i[:,:,-1])))
            for k in range(J.shape[2]-2, -1, -1):
                Ji[:,:,k]=self.fun_prod[:,ax] - w_star[:,:,k] + self.p.beta  *(1-self.js.pe(v_grid_0[ax,:]*r[ax,ax,k+1]))*J[:,:,k+1]
                W1i[:,:,k]=self.pref.utility(w_star[:,:,k]) + self.p.beta * (1-self.js.pe(v_grid_0[ax,:]*r[ax,ax,k+1]))*W1i[:,:,k+1]





                
                EW1_star[iz, :,ik] = np.interp(rho_star[iz, :,ik], rho_grid, EW1i[iz, :,ik])
                EJ1_star[iz, :,ik] = np.interp(rho_star[iz, :,ik], rho_grid, EJpi[iz, :,ik]) #Andrei: how does interpolating the shadow cost give us the future Value?
                
            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            # get pstar, qstar
            pe_star, re_star, _ = self.getWorkerDecisions(EW1_star)

            # Update firm value function 
            EJ1_star = self.shift(EJ1_star) #I am shifting the future value to the next step of k!
            EW1_star = self.shift(EW1_star)
            Ji[:,:,0] = self.fun_prod[:, ax] - w_grid[ax, :] + self.p.beta * (1 - pe_star) * EJ1_star #we v_grid state is for the starting wage so we keep that.
            #However, all the future wages are determined optimally, and we can determine that through the shadow cost. Only weird part here is that we still... interpolate the expected values???
            Ji[:,:,1:] = self.fun_prod[:, ax] - np.power(rho_star[:, :,:],self.p.u_rho) + self.p.beta * (1 - pe_star) * EJ1_star #so rho=1/u'(w), so w=u'^(-1)(1/rho)
            # Update worker value function
            W1i[:,:,0] = self.pref.utility(w_grid)[ax, :] + \
                self.p.beta * (re_star + EW1_star)
            W1i[:,:,1:] = self.pref.utility(np.power(rho_star,self.p.u_rho))[:, :,:] + \
                self.p.beta * (re_star + EW1_star)            
            W1i = .2*W1i + .8*W1i2

            # Updating J1 representation
            error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i, Ji)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i), 100)
            error_w1 = array_dist(W1i, W1i2)
        
            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------
                    if (np.array([error_w1, error_js, error_j1p_chg]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            P_xv = self.matching_function(J1p.eval_at_W1(W1i)[0, :])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(W1i[0, :], P_xv, type=1, relax=relax)
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
        return Ji,W1i
        #Introduce optimal r right here!
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