import numpy as np
import logging
from scipy.stats import lognorm as lnorm
import matplotlib.pyplot as plt

import opt_einsum as oe

from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search import JobSearchArray
from valuefunction_multi import PowerFunctionGrid
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

ax = np.newaxis

def impose_decreasing(M):
    nv = M.shape[1]
    if len(M.shape)==2:
        for v in reversed(range(nv-1)):
            M[:,v] = np.maximum(M[:,v],M[:,v+1]+1e-10)
    else:
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

def production(sum_n):
    return np.power(sum_n, 0.5)
def fun_prod_1d(sum_n):
    return 0.5*np.power(sum_n+1e-10,-0.5) #1e-10 added to avoid division by zero in the lowest size state.
    #Still kinda insane though, makes it look like the future derivate at zero size is minus infty
    #Should I do a manual derivative instead?? Like a diff between zero and 1???



class MultiworkerContract:
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
        self.K = 2
        K = 2
        self.p = input_param
        self.deriv_eps = 1e-4 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid

        #Size grid:
        self.N_grid=np.linspace(0,self.p.num_n-1,self.p.num_n)
        #self.N_grid=np.linspace(0,1,self.p.num_n)
        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = np.ones(self.p.num_x) * self.p.u_bf_m

        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        dimensions=[self.p.num_z]
        dimensions.extend([self.p.num_n] * K)
        dimensions.extend([self.p.num_v] * (K - 1))        
        self.J_grid   = np.zeros(dimensions) #grid of job values, first productivity, then size for each step, then value level for each step BESIDES FIRST
        # Production Function in the Model
        self.fun_prod_onedim = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = self.fun_prod_onedim.reshape((self.p.num_z,) + (1,) * (self.J_grid.ndim - 1))

        # Wage and Shadow Cost Grids
        self.w_grid = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v )
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)


        #Total firm size for each possible state
        self.grid = np.ogrid[[slice(dim) for dim in self.J_grid.shape]]
        # Calculate the sum size for each element in the matrix
        self.sum_size = np.zeros(self.J_grid.shape)
        self.sum_wage=np.zeros(self.J_grid.shape) #Calculate the total wage paid for every state
        for i in range(1, K + 1):
            self.sum_size += self.N_grid[self.grid[i]]
        for i in range(K+1,self.J_grid.ndim):
            self.sum_wage += self.w_grid[self.grid[i]]*self.N_grid[self.grid[i-K+1]] #We add +1 because the wage at the very first step is semi-exogenous, and I will derive it directly

        #Setting up production grids
        self.prod = production(self.sum_size)
        self.prod_diff = self.production_diff(self.sum_size)

        #Job value and GE first
        self.v_grid = np.linspace(np.divide(self.pref.utility(self.unemp_bf.min()),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod_onedim.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #Value promised to the worker at the bottom step
        self.v_0 = self.v_grid.min()
        
        self.simple_J=np.divide(self.fun_prod_onedim[:,ax] -self.pref.inv_utility(self.v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)
        #Apply the matching function: take the simple function and consider its different values across v.
        #This is equivalent to marginal value of a firm of size 1 at the lowest step
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
        self.js.update(self.v_grid[:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        


        #Create a guess for the MWF value function
        #self.J_grid1 = self.J_grid1+np.divide(self.fun_prod*production(self.sum_size)-self.w_grid[0]*self.N_grid[ax,:,ax,ax]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #self.J_grid1 = np.zeros_like(self.J_grid)
        self.J_grid = self.J_grid+np.divide(self.fun_prod*production(self.sum_size)-self.w_grid[0]*self.N_grid[self.grid[1]]-self.sum_wage,1-self.p.beta) #Andrei: this is the guess for the value function, which is the production function times the square root of the sum of the sizes of the markets the worker could search in
        #print("J_grid_diff:", np.max(abs(self.J_grid-self.J_grid1)))
        #The two methods are equivalent!! grid[1] really does capture the right value!!!


        #Guess for the Worker value function
        self.W1i = np.zeros_like(self.J_grid)
        self.W1i = np.expand_dims(self.W1i, axis=-1) #adding an extra dimension to W1i
        self.W1i = np.repeat(self.W1i, self.K, axis=-1)

        #Creating the wage matrix manually
        self.w_matrix = np.zeros(self.W1i.shape)
        self.w_matrix[:,:,:,:,0] = 0 #The workers at the bottom step will have special wages, derived endogenously through their PK
        #Actually, do I then need to add that step to the worker value? Not really, but useful regardless.
        # Can say that the bottom step really is step zero, with a fixed value owed to the worker.
        # And then all the actually meaningful steps are 1,2... etc, so when K=2 with just have 1 meaningful step            
        self.w_matrix[:,:,:,:,1] = self.w_grid[ax,ax,ax,:]

        self.W1i = self.W1i + self.pref.utility(self.w_matrix/(1-self.p.beta)) #skip the first K-1 columns, as they don't correspond to the wage state. Then, pick the correct step, which is hidden in the last dimension of the grid
        self.W1i[:,:,:,:,0] = self.W1i[:,:,:,:,0] + self.pref.utility(self.unemp_bf.min()/(1-self.p.beta))
    def production_diff(self,sum):

        #diff = fun_prod(np.minimum(sum+1,self.K*(self.p.num_n-1))) - production(sum) + production(sum) - fun_prod(np.maximum(sum-1,0)) / 2*((np.logical_and(sum - 1>= 0, sum + 1 <= self.K * (self.p.num_n - 1))))
        diff = (production(np.minimum(sum+1,self.K*(self.p.num_n-1))) - production(sum) + production(sum) - production(np.maximum(sum-1,0))) / (np.minimum(sum+1,self.K*(self.p.num_n-1)) - np.maximum(sum-1,0))
        
        return diff
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

    def J(self,update_eq=0,Ji=None,W1i=None):    
        """
        Computes the value of a job for each promised value v
        :return: value of the job
        """
        sum_wage = self.sum_wage
        rho_grid = self.rho_grid
        if Ji is None:
            Ji = self.J_grid
        if W1i is None:
            W1i = self.W1i
        Ui = self.pref.utility_gross(self.unemp_bf)/(1-self.p.beta)
        print("Ji shape", Ji.shape)
        print("W1i shape", W1i.shape)        
        # create representation for J1p
        J1p = PowerFunctionGrid(W1i, Ji) #From valueFunction.py


        EW1_star = np.copy(Ji)
        EW1_star_1 = np.copy(Ji)
        EJ1_star = np.copy(Ji)
        EJ1_star_0 = np.copy(Ji)
        EJ1_star_1 = np.copy(Ji)
        Jderiv = np.zeros_like(W1i)
        rho_bar = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n))
        rho_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v))
        re_star = np.zeros_like(Ji)
        sep_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v))
        n0_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v))        
        n1_star = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v))
        n1_star0 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v))
        n1_star1 = np.zeros((self.p.num_z, self.p.num_n, self.p.num_n, self.p.num_v))  
        EJinv0 = np.zeros_like(Ji) 
        # prepare expectation call
        Ez = oe.contract_expression('anmv,az->znmv', Ji.shape, self.Z_trans_mat.shape)
        #Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1
        for ite_num in range(self.p.max_iter):
            Ji2 = Ji
            W1i2 = np.copy(W1i)
            Ui2 = Ui
            #print("J with 1 junior", Ji[0,1,0,0])
            # evaluate J1 tomorrow using our approximation
            Jpi = J1p.eval_at_W1(W1i[:,:,:,:,1])

            # we compute the expected value next period by applying the transition rules
            EW1i = Ez(W1i[:,:,:,:,1], self.Z_trans_mat) #Later on this should be a loop over all the k steps besides the bottom one.
            #Will also have to keep in mind that workers go up the steps! Guess it would just take place in the expectation???
            EJpi = Ez(Ji, self.Z_trans_mat)
            EUi = Ui
            # Define the interpolators for EW1i and EJpi
            #EW1i_interpolator = RegularGridInterpolator((self.Z_grid, self.N_grid, self.N_grid,rho_grid), EW1i, bounds_error=False, fill_value=None)
            #EJpi_interpolator = RegularGridInterpolator((self.Z_grid, self.N_grid, self.N_grid,rho_grid), EJpi, bounds_error=False, fill_value=None)

            # get worker decisions
            _, re, pc = self.getWorkerDecisions(EW1i)
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps) 
           
            # compute derivative where continuation probability is >0
            #Andrei: continuation probability is pc, that the worker isn't fired and doesn't leave
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
            

            # First boundary condition: forward difference
            Jderiv[:, :, 0, :, 1] = Ji[:, :, 1, :] - Ji[:, :, 0, :]
            Jderiv[:, 0, :, :, 0] = Ji[:, 1, :, :] - Ji[:, 0, :, :]

            # Last boundary condition: backward difference
            Jderiv[:, :, -1, :, 1] = Ji[:, :, -1, :] - Ji[:, :, -2, :]
            Jderiv[:, -1, :, :, 0] = Ji[:, -1, :, :] - Ji[:, -2, :, :]
            # Central differences: average of forward and backward differences
            Jderiv[:, :, 1:-1, :, 1] = (Ji[:, :, 2:, :] - Ji[:, :, 1:-1, :] + Ji[:, :, 1:-1, :] - Ji[:, :, :-2, :]) / 2
            Jderiv[:, 1:-1, :, :, 0] = (Ji[:, 2:, :, :] - Ji[:, 1:-1, :, :] + Ji[:, 1:-1, :, :] - Ji[:, :-2, :, :]) / 2

            
            EJinv=(impose_decreasing(Jderiv[:,:,:,:,1]+self.w_grid[ax,ax,ax,:])-self.fun_prod*self.prod_diff)/self.p.beta #creating expected job value as a function of today's value
            #So this EJinv, if I want it for the step zero, I should take the correct wage!
            if ite_num>1: #I'm using previous guesses for sep_star and EW1_star. This way, it is still as if EJinv0 is a function of today's states only, even though that's not exactly correct
             EJinv0 = (impose_decreasing(Jderiv[:,:,:,:,0]+self.pref.inv_utility(self.v_0-self.p.beta*(sep_star*EUi+(1-sep_star)*(EW1_star+re_star))))-self.fun_prod*self.prod_diff)/self.p.beta
            
            #Checking whether the inverted expectation is indeed correct
            if ite_num>1:
             sep_inn = (sep_star[0,1,1,:] < 1)
             print("EJinv diff:", np.mean(np.power((EJinv[0,1,1,sep_inn]/(pc_star[0,1,1,sep_inn]*(1-sep_star[0,1,1,sep_inn])) - (EJ1_star[0,1,1,sep_inn]-EJ1_star[0,1,0,sep_inn])) / (EJ1_star[0,1,1,sep_inn]-EJ1_star[0,1,0,sep_inn]),2)))
             #print("EJinv diff:", np.mean(np.power((EJinv[:,1,1,sep_inn]/(pc_star[0,1,1,sep_inn]*(1-sep_star[0,1,1,sep_inn])) - (EJ1_star[:,1,2,sep_inn]-EJ1_star[:,1,0,sep_inn])/2)/(EJ1_star[:,1,2,sep_inn]/2-EJ1_star[:,1,0,sep_inn]/2),2)))

            #Andrei: this is a special foc for the 1st step only! As both the 0th and the 1st steps are affected
            #Because of this, the values are modofied with size according to the following formula:
            #(n_0+n_1)*rho'_1-EJderiv*eta*(n_0+n_1)-n_0*rho_0-n_1*rho_1

            #Main foc, in the absence for separations
            foc = rho_grid[ax, ax, ax, :, ax] - (EJinv[:, :, :, ax, :] / pc[:, :, :, :, ax])* (log_diff[:, :, :, :, ax] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
            foc = foc*(self.N_grid[self.grid[1][:, :, :, ax, :]] * (1-sep_star[:,:,:,ax,:])+self.N_grid[self.grid[2][:, :, :, ax, :]]) - self.N_grid[self.grid[2][:, :, :, ax, :]]*rho_grid[ax, ax, ax, ax, :] -\
                 self.N_grid[self.grid[1][:, :, :, ax, :]] * (1-sep_star[:,:,:,ax,:]) / self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[:,:,:,ax,:]*EUi+(1-sep_star[:,:,:,ax,:])*(EW1i[:, :, :, :, ax]+re[:, :, :, :, ax])))
            #There are no separations here as this FOC is in the case of NO separations (although, what if s=1?)
            #For for wages if separations are positive
            foc_rho_s = rho_grid[ax, ax, ax, :,ax]+((re[:, :, :, :, ax]+EW1i[:, :, :, :, ax]-EUi) / self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[:,:,:,ax,:]*EUi+(1-sep_star[:,:,:,ax,:])*(EW1i[:, :, :, :, ax]+re[:, :, :, :, ax]))))*(log_diff[:, :, :, :, ax] / self.deriv_eps)/(pc[:, :, :, :, ax])
            foc_rho_s =  foc_rho_s*(self.N_grid[self.grid[1][:, :, :, ax, :]] * (1-sep_star[:,:,:,ax,:]) + self.N_grid[self.grid[2][:, :, :, ax, :]]) - self.N_grid[self.grid[2][:, :, :, ax, :]]*rho_grid[ax, ax, ax, ax, :]  -\
                 self.N_grid[self.grid[1][:, :, :, ax, :]] * (1-sep_star[:,:,:,ax,:]) / self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[:,:,:,ax,:]*EUi+(1-sep_star[:,:,:,ax,:])*(EW1i[:, :, :, :, ax]+re[:, :, :, :, ax])))
            assert (np.isnan(foc) & (pc[:, :, :, :, ax] > 0)).sum() == 0, "foc has NaN values where p>0"
            #So, the issue is that the wage the firm pays to the bottom guy depends on the separation rate, creating an additional interdependence

            for iz in range(self.p.num_z):
                # get EW1_Star and EJ1_star
             EW1i_interpolator = RegularGridInterpolator(
             (self.N_grid, rho_grid), EW1i[iz, 0, :, :], bounds_error=False, fill_value=None) #2nd dim set to 0 because n0_star=0
             EJpi_interpolator = RegularGridInterpolator(
             (self.N_grid, rho_grid), EJpi[iz, 0, :, :], bounds_error=False, fill_value=None)

             for in0 in range(self.p.num_n):
              for in1 in range(self.p.num_n):

               rho_min = rho_grid[pc[iz, in0, in1, :] > 0].min()  # lowest promised rho with continuation > 0
               if ite_num>10:
                assert np.all(EW1i[iz, in0, in1, 1:] >= EW1i[iz, in0, in1, :-1]) #Andrei: check that worker value is increasing in v

                for iv in range(self.p.num_v):
                #Alternative corner: not EJinv0, but EJpi interpolated ot the point of no separations
                #And similarly, EJPi at the point of s=1
                #Everywhere else, interior solution
                
                 n1_star0[iz,in0,in1,:]=(in0*+in1)*np.interp(rho_star[iz, in0, in1, :], rho_grid, pc[iz,in0,in1,:])
                 n1_star1[iz,in0,in1,:]=in1*np.interp(rho_star[iz, in0, in1, :], rho_grid, pc[iz,in0,in1,:])                
                 rho_n_star_points_0 = np.array([n1_star0[iz, in0, in1, iv], rho_star[iz, in0, in1, iv]])
                 rho_n_star_points_1 = np.array([n1_star1[iz, in0, in1, iv], rho_star[iz, in0, in1, iv]])                 
                 EJ1_star_0[iz, in0, in1, iv] = EJpi_interpolator(rho_n_star_points_0)
                 EW1_star_1[iz, in0, in1, iv] = EW1i_interpolator(rho_n_star_points_1)
                 EJ1_star_1[iz, in0, in1, iv] = EJpi_interpolator(rho_n_star_points_1)
                Ikeep = (pc[iz, in0, in1, :] > 0) & (EJ1_star_0[iz, in0, in1, :] >= 0) #If marginal value is positive, firm keeps the worker
                if Ikeep.sum() > 0:
                    Ikeep_indices = np.where(Ikeep)[0]
                    for iv in Ikeep_indices:
                        
                      rho_star[iz,in0, in1, iv] = np.interp(0,
                                                    impose_increasing(foc[iz, in0, in1, :, iv]),
                                                    rho_grid[:])
                      sep_star[iz,in0, in1, iv] = 0

                Icompletefire = (pc[iz, in0, in1, :] > 0) & (-EJ1_star_1[iz, in0, in1, :]- (EW1_star_1[iz,in0,in1,:]+re_star[iz,in0,in1,:]-EUi)/ self.pref.inv_utility_1d(self.v_0-self.p.beta*(EUi))> 0 )
                if Icompletefire.sum() > 0:
                    sep_star[iz,in0, in1, Icompletefire] = 1
                    Icompletefire_indices = np.where(Icompletefire)[0]
                    for iv in Icompletefire_indices:
                        rho_star[iz,in0, in1, iv] = np.interp(0,
                                                    impose_increasing(foc[iz, in0, in1, :, iv]),
                                                    rho_grid[:])

                
                Ifire = (pc[iz, in0, in1, :] > 0) & (EJ1_star_0[iz, in0, in1, :] < 0) & (-EJ1_star_1[iz, in0, in1, :]- (EW1_star_1[iz,in0,in1,:]+re_star[iz,in0,in1,:]-EUi)/ self.pref.inv_utility_1d(self.v_0-self.p.beta*(EUi)) < 0 ) #If foc at zero sep>0 and at 1 sep<0, fire partially
                if Ifire.sum() > 0:
                    Ifire_indices = np.where(Ifire)[0]
                    for iv in Ifire_indices:
                        
                        rho_star[iz,in0, in1, iv] = np.interp(0,
                                                    impose_increasing(foc_rho_s[iz, in0, in1, :, iv]),
                                                    rho_grid[:])  
                        worker_future_value = np.interp(rho_star[iz, in0, in1, iv], rho_grid, re[iz,in0,in1,:]+EW1i[iz,in0,in1,:])
                      #print("Worker future value:", worker_future_value)
                      #print("EUi:", EUi)
                      #print("EJinv0:", EJinv0[iz,in0,in1,iv])
                        assert worker_future_value > EUi
                        sep_star[iz,in0, in1, iv] = 1-(EJinv0[iz,in0,in1,iv]/((EUi-worker_future_value) / self.pref.inv_utility_1d(self.v_0-self.p.beta*(sep_star[iz,in0, in1, iv]*EUi+(1-sep_star[iz,in0, in1, iv])*worker_future_value)))) #The thing is, this wouldn't work: t he corner case of ultra negative EJinv would suggest us negative separations, rather than 1       

                Iquit = ~(pc[iz, in0, in1, :] > 0) 
                if Iquit.sum() > 0:
                        rho_star[iz, in0, in1, Iquit] = rho_min
                        sep_star[iz,in0, in1, Iquit] = 0
                #Update the future size for each given size.
                #Issue is: ideally I would use pe_star, but that is only available after I get EW1i. Is there a way around this?
                n0_star[iz, in0, in1, :] = 0 #For now, I'm basically assuming that someone extra will come. Can this fuck up the inverse expectation thing?
                n1_star[iz, in0, in1, :] = (in0*(1-sep_star[iz,in0, in1, :])+in1)*np.interp(rho_star[iz, in0, in1, :], rho_grid, pc[iz,in0,in1,:])
                



                for iv in range(self.p.num_v):
                 #rho_n_star_points = np.array([iz, in0,  n1_star[iz, in0, in1, iv], rho_star[iz, in0, in1, iv]])
                 rho_n_star_points = np.array([n1_star[iz, in0, in1, iv], rho_star[iz, in0, in1, iv]])
                 EW1_star[iz, in0, in1, iv] = EW1i_interpolator(rho_n_star_points)
                 EJ1_star[iz, in0, in1, iv] = EJpi_interpolator(rho_n_star_points)
                
                #EW1_star[iz, in0, in1, :] = np.interp(rho_star[iz, in0, in1, :], rho_grid, EW1i[iz, 0, in1, :])
                #EJ1_star[iz, in0, in1, :] = np.interp(rho_star[iz, in0, in1, :], rho_grid, EJpi[iz, in0, in1, :]) #Andrei: how does interpolating the shadow cost give us the future Value?
                #We're interpolating EJpi to the value where the shadow cost is the optimal one, aka rho_star/
                #Basically, fixing today's promised value, we find the future value that will be optimal via  the shadow cost, and interpolate the expected value at the point of the optimal shadow cost
               else: 
                Isearch = (pc[iz, in0, in1, :] > 0) #Okay, I think this is the set of points (of promised value v) such that these conditions hold
                if Isearch.sum() > 0:
                    Isearch_indices = np.where(Isearch)[0]
                    for iv in Isearch_indices:

                      rho_star[iz,in0, in1, iv] = np.interp(0,
                                                    impose_increasing(foc[iz, in0, in1, Isearch, iv]),
                                                    rho_grid[Isearch])                
                Iquit = ~(pc[iz, in0, in1, :] > 0) 
                if Iquit.sum() > 0:
                           rho_star[iz, in0, in1, Iquit] = rho_min

                #Update the future size for each given size.
                #Issue is: ideally I would use pe_star, but that is only available after I get EW1i. Is there a way around this?
                n0_star[iz, in0, in1, :] = 0 #For now, I'm basically assuming that someone extra will come. Can this fuck up the inverse expectation thing?
                n1_star[iz, in0, in1, :] = (in0+in1)*np.interp(rho_star[iz, in0, in1, :], rho_grid, pc[iz,in0,in1,:])


                for iv in range(self.p.num_v):
                 #rho_n_star_points = np.array([iz, in0,  n1_star[iz, in0, in1, iv], rho_star[iz, in0, in1, iv]])
                 rho_n_star_points = np.array([n1_star[iz, in0, in1, iv], rho_star[iz, in0, in1, iv]])
                 EW1_star[iz, in0, in1, iv] = EW1i_interpolator(rho_n_star_points)
                 EJ1_star[iz, in0, in1, iv] = EJpi_interpolator(rho_n_star_points)


            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            _, re_star, pc_star = self.getWorkerDecisions(EW1_star)
            # Update firm value function 
            Ji = self.fun_prod*self.prod - sum_wage -\
                self.pref.inv_utility(self.v_0-self.p.beta*(sep_star*EUi+(1-sep_star)*(EW1_star+re_star)))*self.N_grid[self.grid[1]]  + self.p.beta * EJ1_star
            Ji = .2*Ji + .8*Ji2
            #print("Value diff:", np.max(np.abs(Ji-Ji2)))

            # Update worker value function
            W1i[:,:,:,:,1] = self.pref.utility(self.w_matrix[:,:,:,:,1]) + \
                self.p.beta * (re_star + EW1_star) #For more steps the ax at the end won't be needed as EW1_star itself will have multiple steps
            #print("Max search value", re_star.max())
            W1i[:,:,:,:,1:] = .4*W1i[:,:,:,:,1:] + .6*W1i2[:,:,:,:,1:] #we're completely ignoring the 0th step
            #print("Worker Value diff:", np.max(np.abs(W1i[:,:,:,:,1:]-W1i2[:,:,:,:,1:])))   
            _, ru, _ = self.getWorkerDecisions(EUi, employed=False)
            Ui = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EUi)
            Ui = 0.2*Ui + 0.8*Ui2


            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  

            error_w1 = array_dist(W1i[:,:,:,:,1:], W1i2[:,:,:,:,1:])


            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------
                     # Updating J1 representation
                    error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[:,:,:,:,1], Ji)
                    error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[:,:,:,:,1]), 100)
                    print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)                   
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            P_xv = self.matching_function(J1p.eval_at_W1(W1i)[self.p.z_0-1, 0, 0, :, 1])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(W1i[self.p.z_0-1, 0, 0, :, 1], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    # Updating J1 representation
                    error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[:,:,:,:,1], Ji)
                    error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[:,:,:,:,1]), 100)
                    print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)    
                    if (np.array([error_w1, error_j1i]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            if (ite_num % 25) == 0:
                # Updating J1 representation
                error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i[:,:,:,:,1], Ji)
                error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i[:,:,:,:,1]), 100)
                print("Errors:", error_j1p_chg, error_j1i, error_j1g, error_w1, error_js)    
                self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}   rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))

        self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, self.js.rsq(), rsq_j1p ))
        return Ji,W1i,EW1_star,sep_star


    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)