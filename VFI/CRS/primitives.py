"""
    Set of functions and classes that help dealing with creating, modifying and saving primitives, such as parameters or the utility functions.
"""

import numpy as np
import logging
import json


class Parameters:

    """
        Class whose objects store the parameters used in the model.
    """

    def __init__(self, overwrite={}):
        """
            We can either initialize the object using a pre-defined set of parameters, or pass in a user-defined set
             of parameters.
            :param model_input: Dict or None
        """

        # Points in the Model
        self.num_l  = 101     # Number of points of evaluation
        self.num_v  = 100     # Number of points in the grid for V
        self.num_x  = 1      # Number of points of support for worker productivity #Andrei: removed worker heterogeneity, require both num_x and num_np to be 1
        self.num_z  = 7      # Number of points for match productivity
        self.num_q = 10      #Number of avg match quality levels
        # Time periods in the Model
        self.dt     = 0.25 #0.25    # Time as a Fraction of Year


        #HMQ
        self.q_0 = 0.70 #Starting match q
        self.prod_q = 0.3 #Relative prodctitivity of a low q match. So, total productivity is sum (prod_q+q_grid*(1-prod_q))*N_grid #Under no HMQ firm doesnt fire

        # Unemployment Parameters
        self.u_bf_m = 0.5        #1.0 * self.dt  #0.05?? sooo low # Intercept of benefit function for unemployed(x)
        #Min wage
        self.min_wage = 0.5
        # Severance min bound <- scales with tenure??? how tf do I do that in crs tho
        self.min_sev = 0 #So not clear how to do this. Maybe I don't do this for now, and instead just compare the realized optima severance with the legal bound
        #Variable set
        # 2 HMQ
        # unemp value b
        # 2 search: s_job and alpha (or kappa)
        # 2 productivity: variance and corr



        #Utility shifter
        self.util_shift = 1.0
        # Utility Function Parameters
        self.u_rho = 1.5       # Risk aversion coefficient
        self.u_a   = 1.0
        self.u_b   = 1.0

        # Search Environment
        self.z_0      = int(self.num_z/2+0.5)         # Slice of value function of firms (index starts at 1)
        self.s_job    = 0.30        # Relative Efficiency of Search on the Job #0.53 in BL, but this is a bit of a pain at lower values since worker value is then below then unemp
        self.alpha    = 1.0        # Parameter for probability of finding a job #If I'm playing with kappa, I can fix this too 1
        self.sigma    = 0.8         # Parameter for probability of finding a job #PRESET, DON'T RE-ESTIMATE
        self.kappa    = 1.0         # Vacancy cost parameter


        # Productivity shocks
        self.x_corr = 0.95  # Correlation in worker productivity
        self.z_corr = 0.95 #0.8 #was 0.95, but 0.95^4=0.81  # Correlation in match productivity #This is a probability of productivity changing, should be adjusted to time!!!

        # Productivity Function Parameters
        self.prod_var_x  = 1.0           # Variance of X (permanent)
        self.prod_var_x2 = 1.0           # Variance of X (non-permanent)
        self.prod_var_z  = 0.49          # Variance of Z # was 0.49 in BL, I'm toning this fucker way down
        #self.prod_z      = 0.5           # Production function parameter #Andrei: where does this come in???
        self.prod_rho    = 1.0           # Production function parameter #This is like the curvature of fun_prod wrt z... do I need this? Seems to play the same purpose as var_z
        #self.prod_mu     = 0.2           # Worker contribution
        #self.prod_px     = 1.0           # Worker power (non linear in type)
        #self.prod_py     = 1.0           # Firm power (nonlinear in type)
        self.prod_a      = 4 * self.dt  # Factor for output function #Questioon is whether i want to noormalize the prooductivity or the wage to 1
        #Also setting prod_a to 1 doesn't actually normalize productivity to 1, only upon improvements. So let's raise this boy a bit
        self.prod_a     = 1.3
        self.prod_err_w  = 0.0           # Measurement error on wages
        self.prod_err_y  = 0.0           # Measurement error on wages

        # Discounting Rates
        self.beta     = 1 - (1 - 0.95) * self.dt  # Impatience
        self.int_rate = 1 / self.beta - 1         # Period interest rate


        # Unemployment Parameters w_net = tau * w ^ lambda
        self.tax_lambda = 1.0       # curvature of the tax system 
        self.tax_tau    = 1.0       # proportion of take home
        self.tax_expost_lambda = 1.0  # this is for counterfactuals, allows to only apply taxes expost
        self.tax_expost_tau = 1.0     # this is for counterfactuals, allows to only apply taxes expost

        # Computational Parameters
        self.chain            = 1         # Chain id when running in parallel
        self.max_iter         = 10000
        self.max_iter_fb      = 5000
        self.verbose          = 5
        self.iter_display     = 25
        self.tol_simple_model = 1e-5
        self.tol_full_model   = 5e-8
        self.tol_search       = 1e-2
        self.eq_relax_power   = 0.4       #  we relax the equilibrium constrain using an update rule based
        self.eq_relax_margin  = 500       #  on mumber of iterations
        self.eq_weighting_at0 = 0.01      # fitting J function with weight around 0

        # simulation parameters
        self.sim_ni      = 20000  # number of workers
        self.sim_nt      = 30     # time periods on top of nt_burn
        self.sim_nt_burn = 10     # periods to discard at begining
        self.sim_nh      = 200    # length of the firm history
        self.sim_nrep    = 20     # number of replication samples
        self.sim_net_earnings = False # whether to use net or gross earnings in the simulation
        #Simulate_val values
        #ni=int(1e4),nt=100,burn=20,nh=100


        for key, val in overwrite.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = val
            else:
                logging.warning("this key does not exists:" + key)

    @staticmethod
    def load(filename) -> 'Parameters':
        with open(filename, "r") as infile:
            pdict =  json.load(infile)
        p = Parameters(pdict)    
        return p

    def save(self,filename, append_dict = {}):
        temp_dict = self.__dict__.copy()
        temp_dict.update(append_dict)
        with open(filename, "w") as fp:
            json.dump(temp_dict, fp)

    def __getstate__(self):
        """ defines how the model is pickled """
        return self.__dict__     # get attribute dictionary
    
    def __setstate__(self, dict):
        """ defines how the model is unpickled """
        self.__dict__ = dict     # make dict our attribute dictionary

    def to_dict(self):
        return self.__dict__.copy()
    
    def get_x_components(self):
        """ give the values of x0 and x1 for each of the values of x """
        num_x0 = int(self.num_x / self.num_np)

        # the fixed heterogeneity component
        x0 = np.arange(num_x0)
        xt = np.arange(self.num_np)
        x0 = np.kron(x0,np.ones(self.num_np)) # permanent is slow moving
        xt = np.kron(np.ones(num_x0),xt) # permanent is slow moving
        return x0,xt

class Preferences():

    """
        Class whose methods represent the preferences and their derivatives, taking a Parameters object as input.
    """

    def __init__(self, input_param=None,log=True):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        if input_param is None:
            self.p = Parameters()
        else:
            self.p = input_param
        self.log = log

    def q_inv(self,q):
         """
            Computes the tightness function at a particular vacancy-filling probability.
            :param  q: Argument of the function.
            :return: Output of the function.
        """       
         return np.power( np.power(self.p.alpha/q,self.p.sigma) - np.power(self.p.alpha,self.p.sigma), 1 / self.p.sigma )

    def utility(self, wage):
        """
            Computes the utility function at a particular wage.
            :param wage: Argument of the function.
            :return: Output of the function.
        """
        if not self.log:
            aa = self.p.u_a * np.power(self.p.tax_tau, 1 - self.p.u_rho) 
            return np.divide(aa * np.power( wage, self.p.tax_lambda * (1.0 - self.p.u_rho)) - self.p.u_b,
                         1 - self.p.u_rho)
        else:
            return np.log(self.p.util_shift * wage)

    def utility_gross(self, wage): #Standard CRRA utility
        """
            Computes the utility function at a particular wage, not applying the tax function
            :param wage: Argument of the function.
            :return: Output of the function.
        """
        if not self.log:
            return np.divide(self.p.u_a * np.power(wage, 1 - self.p.u_rho) - self.p.u_b,
                         1 - self.p.u_rho)
        else:
            return np.log(self.p.util_shift * wage)

    def inv_utility(self, value):
        """
            Computes the inverse utility function at a particular value.
            :param value: Argument of the function.
            :return: Output of the function.
        """
        if not self.log:
            aa = self.p.u_a * np.power(self.p.tax_tau, 1.0 - self.p.u_rho) 
            return np.power(np.divide((1.0 - self.p.u_rho) * value + self.p.u_b, aa),
                        (np.divide(1.0, self.p.tax_lambda * (1.0 - self.p.u_rho))))
        else:
            return np.exp(value) / self.p.util_shift

    def utility_1d(self, wage):
        """
            Computes the first derivative of the utility function at a particular wage.
            :param wage: Argument of the function.
            :return: Output of the function.
        """
        if not self.log:
            return self.p.u_a * np.power(wage, - self.p.u_rho)
        else:
            return 1/wage

    def inv_utility_1d(self, value):
        """
            Computes the first derivative of the inverse utility function at a particular value. u'(u^{-1}(v))
            :param value: Argument of the function.
            :return: Output of the function.
        """
        if not self.log:
            aa = self.p.u_a * np.power(self.p.tax_tau, 1 - self.p.u_rho) 
            pow_arg = ( (1 - self.p.u_rho) * value + self.p.u_b   ) / aa
            return np.power( pow_arg, 1.0/( self.p.tax_lambda * (1 - self.p.u_rho) ) - 1.0) / ( self.p.tax_lambda * aa )
        #return np.power( pow_arg, -self.p.u_rho / ( 1 - self.p.u_rho )) / ( self.p.tax_lambda * aa )
        # return self.utility_1d(self.inv_utility(value))
        else:
            return self.p.util_shift / np.exp(value)


    def log_consumption_eq(self, V):
        """
            Returns the log wage/consumption equivalent associated with a present value of the worker.
        """
        return(np.log(self.inv_utility( (1-self.p.beta) * V )))

    def log_profit_eq(self,J):
        """
            Returns the log profit equivalent associated with the firm present value
        """
        return( np.log( (1-self.p.beta) * J))


    def consumption_eq(self, V):
        """
            Returns the log wage/consumption equivalent associated with a present value of the worker.
        """
        return((self.inv_utility( (1-self.p.beta) * V )))

    def profit_eq(self,J):
        """
            Returns the log profit equivalent associated with the firm present value
        """
        return(( (1-self.p.beta) * J))
