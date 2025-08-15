""" 
    Simulates panel data from the model    
"""
from time import time
import numpy as np
import logging
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
import gc
#from scipy.interpolate import RegularGridInterpolator
from pyfixest import feols
from scipy.ndimage import map_coordinates
def bool_index_combine(I,B):
    """ returns an index where elements of I have been updated using B
    I,B are boolean, and len(B)==I.sum() """
    I2 = np.copy(I)
    I2[I]=B
    return I2


class Event:
    uu  = 0
    ee  = 1
    u2e = 2
    e2u = 3
    j2j = 4

class RegularGridInterpolator:
    def __init__(self, points, values, method='linear'):
        self.limits = np.array([[min(x), max(x)] for x in points])
        self.values = np.asarray(values, dtype=float)
        self.order = {'linear': 1, 'cubic': 3, 'quintic': 5}[method]

    def __call__(self, xi):
        """
        `xi` here is an array-like (an array or a list) of points.

        Each "point" is an ndim-dimensional array_like, representing
        the coordinates of a point in ndim-dimensional space.
        """
        # transpose the xi array into the ``map_coordinates`` convention
        # which takes coordinates of a point along columns of a 2D array.
        xi = np.asarray(xi).T

        # convert from data coordinates to pixel coordinates
        ns = self.values.shape
        coords = [(n-1)*(val - lo) / (hi - lo)
                  for val, n, (lo, hi) in zip(xi, ns, self.limits)]

        # interpolate
        return map_coordinates(self.values, coords,
                               order=self.order,
                               cval=np.nan,mode='nearest')  # fill_value

def create_year_lag(df,colnames,lag):
    """ the table should be index by i,year
    """
    # prepare names
    if lag>0:
        s = "_l" + str(lag)
    else:
        s = "_f" + str(-lag)

    values = [n + s for n in colnames]
    rename = dict(zip(colnames, values))

    # create lags
    dlag = df.reset_index() \
             .assign(year=lambda d: d['year'] + lag) \
             .rename(columns=rename)[['i','year'] + values] \
             .set_index(['i','year'])

    # join and return
    return(df.join(dlag))


def create_lag_i(df,time_col,colnames,lag):
    """ the table should be index by i,year
    """
    # prepare names
    if lag>0:
        s = "_l" + str(lag)
    else:
        s = "_f" + str(-lag)

    values = [n + s for n in colnames]
    rename = dict(zip(colnames, values))

    # create lags
    dlag = df.reset_index() \
             .assign(t=lambda d: d[time_col] + lag) \
             .rename(columns=rename)[['i',time_col] + values] \
             .set_index(['i',time_col])

    # join and return
    return(df.join(dlag))


def create_lag(df,time_col,colnames,lag):
    """ the table should be index by i,year
    """
    # prepare names
    if lag>0:
        s = "_l" + str(lag)
    else:
        s = "_f" + str(-lag)

    values = [n + s for n in colnames]
    rename = dict(zip(colnames, values))
    assign_arg = {time_col : lambda d: d[time_col] + lag}

    # create lags
    dlag = df.reset_index() \
             .assign(**assign_arg) \
             .rename(columns=rename)[[time_col] + values] \
             .set_index([time_col])

    # join and return
    return(df.join(dlag))

def discretize_n(n):
    # Step 1: Compute floor and ceil hiring for each firm
    n_floor = np.floor(n).astype(int)
    n_ceil = np.ceil(n).astype(int)
    alpha = n - n_floor  # Residual fractions

    # Step 2: Randomly assign firms to ceil with probability alpha
    ceil_firms = np.random.binomial(1, alpha).astype(bool)

    # Step 3: Ensure the sum is correct
    new_sum = np.sum(n_ceil[ceil_firms]) + np.sum(n_floor[~ceil_firms])
    original_sum = np.sum(n)
    delta = int(original_sum - new_sum)  # Discrepancy


    if delta > 0:
        # Need to increase hiring: randomly move delta firms from floor to ceil
        floor_firms = np.where(~ceil_firms)[0]  # Indices of firms hiring floor
        adjust_indices = np.random.choice(floor_firms, size=delta, replace=False)
        ceil_firms[adjust_indices] = True
    elif delta < 0:
        # Need to decrease hiring: randomly move delta firms from ceil to floor
        ceil_firms_indices = np.where(ceil_firms)[0]  # Indices of firms hiring ceil
        adjust_indices = np.random.choice(ceil_firms_indices, size=abs(delta), replace=False)
        ceil_firms[adjust_indices] = False

    # Step 4: Update n0[t, :]
    n[ ceil_firms] = n_ceil[ceil_firms]
    n[ ~ceil_firms] = n_floor[~ceil_firms]
    return n

def allocate_workers_to_vac_rand(n_hire,F_set,total_workers):

    # Split into integer and fractional parts
    integer_parts = np.floor(n_hire).astype(int)
    fractional_parts = n_hire - integer_parts

    # Generate integer vacancies (1.0 probability)
    integer_firms = np.repeat(F_set, integer_parts) #This WILL REMOVE firms not hiring at all.
    integer_probs = np.ones_like(integer_firms, dtype=np.float64)
    # Generate fractional vacancies (prob = fractional part)
    has_fractional = fractional_parts >= 1e-8  # Tolerance for floating point precision
    fractional_firms = F_set[has_fractional]
    fractional_probs = fractional_parts[has_fractional]

    # Combine all vacancies and probabilities
    all_firms = np.concatenate([integer_firms, fractional_firms])
    all_probs = np.concatenate([integer_probs, fractional_probs])

    # Normalize probabilities
    prob_sum = all_probs.sum()
    if prob_sum == 0:
        raise ValueError("No vacancies available for allocation.")
    all_probs_normalized = all_probs / prob_sum

    # Create vacancy indices (0, 1, 2, ...)
    vacancy_indices = np.arange(len(all_probs))
    # Allocate workers without replacement
    selected_indices = np.random.choice(
        vacancy_indices, 
        size=total_workers, 
        replace=False, 
        p=all_probs_normalized
    )
    selected_firms = all_firms[selected_indices]
    return selected_firms

def allocate_workers_to_vac_determ(n_hire,F_set,F,Id,I_find_job):
    # Split into integer and fractional parts
    integer_parts = np.floor(n_hire).astype(int)

    # Generate integer vacancies (1.0 probability)
    integer_firms = np.repeat(F_set, integer_parts)
    integer_vacancies = np.arange(len(integer_firms))
    #Randomly select workers to be allocated to these vacancies
    Id_to_int_vacancies = np.random.choice(Id[I_find_job], size=len(integer_firms),replace=False) #This gives us literal worker id's. Still need to bring this back to boolean
    I_to_int_vacancies = np.isin(Id,Id_to_int_vacancies) #This gives us a boolean, set to 1 workers who were selected in the line above
    selected_vacancies = np.random.choice(integer_vacancies, size=len(Id_to_int_vacancies),replace=False)
    F[I_to_int_vacancies] = integer_firms[selected_vacancies]
    
    #Now take the allocate of the workers to the fractional vacancies
    I_to_frac_vacancies = (I_find_job * (1 -np.isin(Id,Id_to_int_vacancies))).astype(bool)
    if I_to_frac_vacancies.sum()>0:
        fractional_parts = n_hire - integer_parts
        has_fractional = fractional_parts >= 1e-8  # Tolerance for floating point precision
        fractional_firms = F_set[has_fractional]
        fractional_vacancies = np.arange(len(fractional_firms))

        # Set and Normalize probabilities
        fractional_probs = fractional_parts[has_fractional]
        prob_sum = fractional_probs.sum()
        probs_normalized = fractional_probs / prob_sum

        selected_vacancies = np.random.choice(fractional_vacancies,size = I_to_frac_vacancies.sum(),replace=False,p=probs_normalized)
        F[I_to_frac_vacancies] = fractional_firms[selected_vacancies]

    return F

class Simulator:
    """
    Simulates data from the model and computes moments on simulated data.
    """

    def __init__(self,model,p):
        self.sdata = pd.DataFrame()
        self.model = model
        self.p = p
        self.moments = {}
        self.Zhist = np.zeros((p.num_z,p.sim_nh),dtype=int)
        self.log = logging.getLogger('Simulator')
        self.log.setLevel(logging.INFO)

    def simulate(self,redraw_zhist=True,ignore=[]):
        return(self.simulate_val(
            self.p.sim_ni, 500, #500 is for nf
            self.p.sim_nt_burn + self.p.sim_nt, 
            self.p.sim_nt_burn,
            self.p.sim_nh,
            redraw_zhist=redraw_zhist,
            ignore=ignore))

    def simulate_val(self,ni=int(1e3),nf=20,nt=40,burn=20,nl=100,redraw_zhist=True,ignore=[]):
        """ we simulate a panel using a solved model

            ni (1e4) : number of individuals
            nt (40)  : number of time period
            nl (100) : length of the firm shock history

            returns a data.frame available at self.sdata with the following columns:
                i: worker id
                t: time
                e: employment status
                h: firm history (where in the common history of shocks)
                x: worker productivity
                z: match productivity
                r: value of rho
                d: even associated with current period
                w: wage
                y: firm present value
                s: tenure at current firm

             the timing is such that the event is what leads to the current state, so the current wage reflects
             the current productivity, current event realized, and rho/wage has been updated. In other words, the event
             happens at the begining of the period, hence U2E are associated with a wage, but E2U are not.

                1) even_t realizes with new firm if necessary
                2) X,Z are drawn
                3) wage is evaluated
        """

        model = self.model
        p     = self.p

        # prepare the ignore shocks
        INCLUDE_E2U = not ('e2u' in ignore)
        INCLUDE_J2J = not ('j2j' in ignore)
        INCLUDE_XCHG = not ('xshock' in ignore)
        INCLUDE_ZCHG = not ('zshock' in ignore)
        INCLUDE_WERR = not ('werr' in ignore)

        # we store the current state into an array
        Id = np.arange(1,ni+1,dtype=int) #Worker id. Only important for determenistic allocation of workers to vacancies
        #X  = np.zeros(ni,dtype=int)  # current value of the X shock
        #Z  = np.zeros(ni,dtype=int)  # current value of the Z shock
        R  = np.zeros(ni)            # current value of rho
        E  = np.zeros(ni,dtype=int)  # employment status (0 for unemployed, 1 for employed)
        #PROD  = np.zeros(ni,dtype=int)  # firm total production
        #Andrei: so should I create F as well?
        F = np.zeros(ni,dtype=int) # Firm identifier for each worker
        F_set  = np.arange(1,nf+1,dtype=int) #Set of all the unique firms
        #N0 = np.zeros(ni,dtype=int)
        #N1 = np.zeros(ni,dtype=int)
        #N  = np.zeros(ni,dtype=int)

        D  = np.zeros(ni,dtype=int)  # event
        W  = np.zeros(ni)            # log-wage
        #P  = np.zeros(ni)            # firm profit
        S  = np.ones(ni,dtype=int)  # number of periods in current spell
        #pr = np.zeros(ni)            # probability, either u2e or e2u

        #Prepare firm info
        extra = 5000 #1k extra firms allowed to enter at most before an error appears
        rho = np.zeros((nt,nf+1+extra)) #Extra 1 is added, so that rho[:,0] ends up unused
        w = np.zeros((nt,nf+1+extra))
        n_hire = np.zeros((nt,nf+1)) #No extra here as we're gonna be concatenating n_hire
        q = np.zeros_like(w) #Should q be int or no?
        sep_rate = np.zeros_like(w)
        pr_j2j = np.zeros_like(w)
        ve_star = np.zeros_like(w)
        bon_leave = np.zeros_like(w)
        w = np.zeros((nt,nf+1+extra,2))
        prod = np.zeros((nt,nf+1+extra))
        z = np.ones((nt,nf+1+extra),dtype=int) * (p.z_0-1) #So the productivity always starts at the starting value
        n0 = np.zeros_like(z)
        n1 = np.zeros_like(z)
        rho_diff_idx = np.zeros(nf+1+extra,dtype=int)

        #Initialize interpolators
        rho_interpolator = np.empty((p.num_z, p.num_n, p.num_n), dtype=object)
        n_interpolator = np.empty_like(rho_interpolator)
        j2j_interpolator = np.empty_like(rho_interpolator)
        sep_interpolator = np.empty_like(rho_interpolator)
        q_interpolator = np.empty_like(rho_interpolator)
        ve_interpolator = np.empty_like(rho_interpolator)
        for iz in range(p.num_z):
            for in0 in range(p.num_n):
                for in1 in range(p.num_n):        
                    rho_interpolator[iz,in0,in1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[iz, in0,in1, ...]) 
                    n_interpolator[iz,in0,in1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[iz, in0,in1, ...]) 
                    j2j_interpolator[iz,in0,in1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[iz, in0,in1, ...]) 
                    sep_interpolator[iz,in0,in1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[iz, in0,in1, ...]) 
                    q_interpolator[iz,in0,in1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[iz, in0,in1, ...]) 
                    ve_interpolator[iz,in0,in1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[iz, in0,in1, ...]) 

        """
        Schaal's GE algorithm:
        1. Fixing  the cost of entry k_e, calculate hiring cost kappa (equal across submarkets) such that the value of an entrant firm E_y J(y,0,0,0,0)=k_e
        2. Calculate theta st kappa=x+c/q(theta) (or, equivalently, theta = q^{-1}(c/(kappa-x))), where x is the suubmarket-specific value. Set theta=0 if x>kappa-c (aka never worth it to post there)
        I need to adjust step 2 since the worker is risk averse. Rather than x this is smth like u^{-1}(x)??? Have to clarify it a bit later
        3. Update the workers value functions and repeat the loop if necessary
        In terms of adapting this to my code:
        Given my guesses, have step 1 be the exact same. Ah! Except that I need n0_star here! Though tbf n0_star changes with kappa... so this this is kinda complex eh? Maybe envelope?
        tbf this is still pretty easy, no? hiring just increases tomorrow's value... but nah, since I don't have a closed-form solution for n0_star, this is still kind of a mess
        Using the envelope, this would be smth like k_e = E_y (J(y,0,0,0,0)-k_f-n0_star(y,0,0,0,0) * (kappa_new - kappa_old)). This should work if kappa changes are small
        Instead, the direct approach would be: k_e = E_y (F(0,0)-w(0,0)-kappa*n0_star+beta*E_{y'|y}J(y',n0_star,0,0,0)), where n0_star is a function of kappa. This
        We can use the foc for n0_star then: kappa = beta E_{y'|y}J'_{n0}. The nice thing here is that the RHS is constant in kappa. So... can calculate it once, but still gotta interpolate to get n0_star!!!
        Good news though is that... I caninterpolate the same function onto many points.
        So this would require:
        a. Get n0_star for many kappa values
        b. Get J'(n0_star) for all those kappa values (J'[kappa]=interp(n0_star[kappa],N_grid,J'))
        c. Find kappa such that the original equation: k_e = E_y (J-k_f-n0_star[y,kappa]*kappa + beta* E_{y'|y}J(y',n0_star[kappa],0,0,0))
        So kind of a pain, esp since I also got do this for every y. But def feasible
        Can try coding both and compare the new kappas. Which one ends up better??? The simpler one might be very shit early on, esp since Id need a separate guess for it. 
        Maybe start with the full one, and later switch to the simpler version. Once I actually start calibrating, can start with a better guess
        BUT WHAT IF ITS A GLOBAL OPTIMIZATION??? Maybe a bad idea then to have the same guess??? Oh shit, is this what happened to my cyclicality paper? Should I remove that guess part???
        Maybe, but kinda unclear tbh. Worth a try tho, esp on a 3-submarket case

        Okay coming back to the algorithm... I can do the same thing, right? The difference to BL here is that I would start with the GE part rather than end with it
        Like this I would need to have an original guess for the search function/kappa.
        """

        # we initialize worker types
        #X = np.random.choice(range(p.num_x),ni)

        df_all = pd.DataFrame()
        # looping over time
        for t in range(1,nt):
            # save the state when starting the period
            E0 = np.copy(E)
            #Z0 = np.copy(Z)
            F0 = np.copy(F) #Note that this F is the set of firms, not the assignment of workers to firms
            S0 = np.copy(S)

            #Check destruction:
            self.close_pr = 0.05 #Percent of closing firms. For now, without aggregate movements, keep it constant
            close = np.random.binomial(1, self.close_pr, F_set.shape[0])==1  
            #F_set     = F_set * (~close) #All the closed firms have index 0, the rest keep their indeces
            #F_set = np.where(close, 0, F_set) #All the closed firms are now called 0, to avoid confusion with unemployed workers who have F=-1
            F_set = F_set[~close] #removing the closed firms from the set

            #Loop over firm states: so that firms aren't repeated
            for iz in range(p.num_z):
                for in0 in range(p.num_n):
                    for in1 in range(p.num_n1):
                        F_spec = F_set[(z[t-1,F_set]==iz)  & (n0[t,F_set]==np.floor(model.N_grid[in0]).astype(int)) & (n1[t,F_set]==np.floor(model.N_grid1[in1]).astype(int))]
                        if len(F_spec)==0:
                            continue
                        coords = np.stack((rho[t-1, F_spec], q[t-1, F_spec]), axis=1)
                        rho[t,F_spec] = rho_interpolator[iz,in0,in1] (coords)
                        n_hire[t,F_spec] = n_interpolator[iz,in0,in1] (coords)
                        pr_j2j[t,F_spec] = j2j_interpolator[iz,in0,in1] (coords)
                        sep_rate[t,F_spec] = sep_interpolator[iz,in0,in1] (coords)
                        q[t,F_spec] = q_interpolator[iz,in0,in1] (coords)
                        ve_star[t,F_spec] = ve_interpolator[iz,in0,in1] (coords)
            w[t,F_set,1] = rho[t,F_set]
            bon_leave[t,F_set] = model.pref.inv_utility(ve_star[t,F_set] - model.v_0) - model.pref.inv_utility(model.v_grid[0] - model.v_0) #Bonus wage if leave the current firm            
            bon_unemp = model.pref.inv_utility(model.ve - model.v_0) - model.pref.inv_utility(model.v_grid[0] - model.v_0) #Bonus wage if find a job
            # print(n_hire[t,F_set[F_set == 0]].sum())
            #print("n_hire for that period", n_hire[t,:])
            #print("F_set of closed firms", F_set[F_set == 0])
            #assert (n_hire[t,F_set[F_set == 0]].sum() <= 1e-1), "Closed firms are hiring"
            #Answer: this thing is >0 because n_hire[t,0] is an actual thing, just like n_hire[t,-1]. SO GOTTA MAKE NOT TO INCLUDE THEM ANYWHERE
            #1. Make employed workers of closed firms unemployed
            I_close = (E0==1) & ~np.isin(F0,F_set)

            E[I_close]  = 0           # make the worker unemployed
            D[I_close]  = Event.e2u
            W[I_close]  = 0           # no wage
            F[I_close]  = 0
            S[I_close]  = 1
            R[I_close]  = 0

            #2. Fire employed workers. WAIT. I'm FIRING EVERYONE HERE, EVEN SENIORS!!!
            I_remain = (E0==1) & np.isin(F0,F_set) & (S0==1)
            F_remain = F0 * I_remain #This one isn't actually necessary, right? Correct, when we do bool_index_combine, the workers with technically real firms still get left out
            # we check the for separation
            rng = np.random.default_rng()
            sep = INCLUDE_E2U *  rng.binomial(1, sep_rate[t,F_remain]) == 1  #This should work, if slightly inefficient. 
            #All the workers outside of I_remain are assigned F_remain=0, so their sep_rate[t,0]=0. For the rest, they're assigned their firm's actual sep_rate
            # workers who got fired
            I_e2u      = I_remain * sep
            E[I_e2u]   = 0
            D[I_e2u]   = Event.e2u
            W[I_e2u]   = 0  # no wage
            F[I_e2u]  = 0 # no firm
            S[I_e2u]   = 1
            R[I_e2u]   = 0

            #3. Unemp workers
            # a) Search shock for unemp workers
            Iu = (E0==0)
            # get whether match a firm
            meet_u2e = np.random.binomial(1, model.Pr_u2e, Iu.sum())==1
            #pr = model.Pr_u2e        
            #Deal with the unlucky unemployed
            Iu_uu = bool_index_combine(Iu,~meet_u2e)
            E[Iu_uu]   = 0
            D[Iu_uu]   = Event.uu
            W[Iu_uu]   = 0  # no wage
            F[Iu_uu]  = 0 # no firm
            S[Iu_uu]   = S[Iu_uu] + 1
            R[Iu_uu]   = 0
            #Unemployed that did find a job
            Iu_u2e     = bool_index_combine(Iu,meet_u2e)
            
            #4. Emp workers            
            Ie_jun = I_remain * (1 - sep) #Before this was bool_index_combine(I_remain,~sep), but that didn't work when I_remain was all zeroes. I think this is due to how sep is created
            Ie_sen = (E0==1) & np.isin(F0,F_set) & (S0 > 1) #Seniors never at risk of firing (for now)
            Ie = (Ie_jun + Ie_sen).astype(bool)
            #If their firm closed, I_remain=0. If they got separated, ~sep=0.
            F_e = F0 * Ie #All the workers outside of this set are assigned "firm' 0, where all the policies are zero.
            # search decision for those not fired    
            rng = np.random.default_rng()           
            meet = INCLUDE_J2J *  rng.binomial(1, pr_j2j[t,F_e]) == 1  

            # Employed workers that didn't transition
            #Ie_stay = bool_index_combine(Ie,~meet)
            Ie_stay = (Ie * (~meet)).astype(bool)
            E[Ie_stay]  = 1          
            D[Ie_stay]  = Event.ee
            W[Ie_stay]  = w[t,F[Ie_stay],1]           # senior wage
            F[Ie_stay]  = F0[Ie_stay]
            S[Ie_stay]  = S[Ie_stay] + 1
            R[Ie_stay]  = rho[t,F[Ie_stay]]                
            
            #Emp workers that did find a job
            #Ie_e2e = bool_index_combine(Ie,meet)
            Ie_e2e = (Ie * meet).astype(bool)
            #5. Total search and allocation of workers to firms
            I_find_job = (Iu_u2e + Ie_e2e.astype(bool)).astype(bool) #Note that I'm simply summing up non-overlapping boolean values here
            #Sum up expected hiring across all firms
            hire_total = n_hire[t,F_set].sum()
            #If not enough expected vacancies, update:
            if np.ceil(hire_total) < I_find_job.sum():
                #See how much new firms want to hire
                new_firm_hiring = model.n0_star[p.z_0-1,0,0,0,0]
                #Note that, if there're 4 workers searching and 3.5 vacancies, no firms are added.
                #Add new firms
                n_new = np.ceil(np.floor(I_find_job.sum() - hire_total)/new_firm_hiring).astype(int) 
                #And, if we have 4w/2.5v, we add 1 firm
                new_firms = np.arange(nf + 1, nf + n_new + 1)
                F_set = np.concatenate((F_set, new_firms)) #or is this too soon? let these firms find a job first mayb?
                nf = nf+n_new #that way, even if we update F_set by removing the closed firms, we can keep the numerator going
                #Add these firms into aggregate hiring
                n_hire_new = np.zeros((nt,n_new))
                n_hire_new[t,:] = new_firm_hiring
                n_hire = np.concatenate((n_hire, n_hire_new),axis=1)
                hire_total = n_hire[t,F_set].sum()
                #print("Number of workers looking for job vs total hiring (diff caps)", I_find_job.sum(), n_hire[t,1:len(F_set)+1].sum(), n_hire[t,F_set].sum())
                assert (np.ceil(hire_total) >= I_find_job.sum()), "Not enough expected hirings"
            #First, a basic version: allocating workers across vacancies completely randomly
            #F[I_find_job] = allocate_workers_to_vac_rand(n_hire[t,F_set],F_set,I_find_job.sum()) #should it be +1 or +2???
            #Now try satisfying all the prob 1 vacancies first
            F = allocate_workers_to_vac_determ(n_hire[t,F_set],F_set,F,Id,I_find_job) #Am I sure it's the entire F??? seems a little sus ngl. yes, it's ok, it only updates for I_find_job rows
            E[I_find_job]   = 1
            S[I_find_job]   = 1
            R[I_find_job]   = model.v_0 #Bonus noted in the actual wage, computed below
            D[Ie_e2e]   = Event.j2j
            D[Iu_u2e]   = Event.u2e  

            #6. Wrapping the period up: updating firm info 
            #Sum up the number of jun and sen workers
            n0[t,F_set] = np.bincount(F[S == 1], minlength=n0.shape[1])[F_set]
            n1[t,F_set] = np.bincount(F[S > 1], minlength=n1.shape[1])[F_set]
            n1[n1> model.N_grid1[-1]] = model.N_grid1[-1] #So that we don't have to worry about the last bin
            n0[n0> model.N_grid[-1]] = model.N_grid[-1] #So that we don't have to worry about the last bin


            #Update firm prod shock
            for iz in range(p.num_z):
                Fz =  F_set[(z[t-1,F_set]==iz) & (F_set != 0)]
                #print(len(Fz))
                z[t,Fz] = np.random.choice(p.num_z, len(Fz), p=model.Z_trans_mat[iz])
            #Update firm production + jun wage (both are interpolations, so put in the same loop)
            for iz in range(p.num_z):
                for in0 in range(p.num_n):
                    for in1 in range(p.num_n1):
                        F_spec = F_set[(z[t,F_set]==iz) & (n0[t,F_set]==np.floor(model.N_grid[in0]).astype(int)) & (n1[t,F_set]==np.floor(model.N_grid1[in1]).astype(int))]
                        if len(F_spec)==0:
                            continue
                        #Jun wage requires 2d intepolation so perform it the same way. Note also that it takes today's states, not previous ones. It's the only policy variable that works like that  
                        coords = np.stack((rho[t-1, F_spec], q[t-1, F_spec]), axis=1)              
                        w[t,F_spec,0] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[iz, in0, in1, ...]) (coords) #Is this time inconsistent??? Given that prod is decided later?
                        prod[t,F_spec] = np.interp(q[t,F_spec],model.Q_grid,model.prod[iz,in0,in1,0,:])
            assert prod[t,F_set[F_set!=0]].min() > 0, "Zero production for open firms"
            #Update new hire wages not jun wages are known
            #W[I_find_job]   = w[t,F[I_find_job],0]   # jun wage. potentially could add the wage bonus here as well. Important for tracking j2j wage growth
            W[Iu_u2e]   = w[t,F[Iu_u2e],0] + bon_unemp
            W[Ie_e2e]   = w[t,F[Ie_e2e],0] + bon_leave[t,F[Ie_e2e]]

            # append to data
            if (t>burn):
                #print("Shapes of all the arrays", E.shape, S.shape, F.shape, X.shape, Z.shape, R.shape, D.shape, W.shape, P.shape, pr.shape)
                df     = pd.DataFrame({ 'i':range(ni),'t':np.ones(ni) * t, 'e':E, 's':S, 'f':F, 'r':R, 'd':D, 'w':W } )
                df_all = pd.concat([df_all, df], axis =0) #Andrei: merge current dataset with the previous ones based on the i axis. So then each worker just has a single row assigned to them?

        # append match output
        df_all['z'] = z[df_all['t'].values.astype(int), df_all['f'].values.astype(int)]
        df_all['n0'] = n0[df_all['t'].values.astype(int), df_all['f'].values.astype(int)]
        df_all['n1'] = n1[df_all['t'].values.astype(int), df_all['f'].values.astype(int)]
        df_all['prod'] = prod[df_all['t'].values.astype(int), df_all['f'].values.astype(int)]
        df_all['n'] = df_all['n0'].values + df_all['n1'].values
        #Unemployed don't get any firm info
        df_all.loc[df_all.e==0,['z','n0','n1', 'prod', 'n']] = 0        

        # sort the data
        df_all = df_all.sort_values(['i', 't'])
        #print("Cumul time gain, good if >0", cumul_time_gain)
        self.sdata = df_all
        return(self)

    def simulate_firm(self,z,n0,n1,rho,q,nt, allow_hiring=True,allow_fire=True,allow_leave=True,update_z=False, z_dir=None,seed=False,disable_fire=False):
        """
        simulates a path of a particular firm from initial state [z,n0,n1,rho,q]
        one can choose to allow the firm to expand or allow the workers to leave using allow_hiring and allow_leave
        one can choose to update z using update_z. choosing z_dir=1 or -1 will result in simulating just a single, deterministic, shock in the middle of the simulation
        one can choose to fix the seed using seed
        for the aggregate version (multiple firms), simulate_force_ee is a better comparison
        """
        if seed:
            np.random.seed(42)
        model = self.model
        all_df = []
        extra = 100 #extra workers to be potentially hired
        if allow_hiring == False:
            extra=0
        ni=n0+n1 #Number of workers employed
        W  = np.zeros(ni+extra)     # log-wage
        W1 = np.zeros(ni+extra)     # value to the worker
        Vs = np.zeros(ni+extra)     # search decision
        S = np.zeros(ni+extra)      # tenure at the firm
        D  = np.zeros(ni+extra,dtype=int)  # event  

        S[:n0] = 1          
        S[n0:ni] = 2
        Y = np.zeros(ni+extra)  # log-output
        P = np.zeros(ni+extra)      # firm profit
        pr_sep_array = np.zeros(ni+extra)  # probability, either u2e or e2u
        pr_j2j_array = np.zeros(ni+extra)  # probability, either u2e or e2u
        F = np.zeros(ni+extra) #==1 if worker is in the firm, zero otherwise    
        F[:ni] = 1
        N0_array = np.zeros(ni+extra)  
        N0_array[F==1] = n0 
        N1_array = np.zeros(ni+extra) 
        N1_array[F==1] = n1
        Z = np.zeros(nt,dtype=int) + z #if z is not updated, always keep it at the same value
        Z_array = np.zeros(ni+extra)
        Z_array[F==1] = z
        #If performing a deterministic prod shock
        if update_z and z_dir is not None:
            #Shock a
            Z[np.floor(nt/2).astype(int):] = z+z_dir
        N0 = np.zeros_like(Z)
        N0[0] = n0
        N1 = np.zeros_like(Z)
        N1[0] = n1
        RHO = np.zeros(nt)
        RHO[0] = rho
        Q = np.zeros_like(RHO)
        Q[0] = q
        prod = np.zeros(nt)
        prod[0] = np.interp(q,model.Q_grid,model.prod[z,n0,n1,0,:])
        w = np.zeros((nt,2))

        #Time 0 append with no update
        W[(F==1) & (S==1)] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[z, n0, n1, ...], bounds_error=False, fill_value=None) ((rho,q)) )
        W[(F==1) & (S>1)] = np.log(rho)
        Y[F==1] = np.log(prod[0])
        W1[(F==1) & (S>1)] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[z, n0, n1, ...,1], bounds_error=False, fill_value=None) (((rho,q)))
        #Firm info, added for every employed worker
        P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[z, n0, n1, ...], bounds_error=False, fill_value=None) ((rho,q))

        all_df.append(pd.DataFrame({ 'i':range(ni+extra),'t':0, 'f':F, 
                'z':Z_array, 'w':W , 'Pi':P, 'D': D, 'S': S, 'n0': N0_array, 'n1': N1_array,
                'pr_e2u':pr_sep_array, 'pr_j2j':pr_j2j_array , 'y':Y, 'W1':W1, 'vs':Vs, 
                }))
        #if pb:
        #    rr = tqdm(range(1,nt))
        #else:
        rr = range(1,nt) 

        for t in rr:
            #Update firm decisions first
            RHO[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            pr_j2j = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            pr_sep = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            pr_sep1 = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star1[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))            
            Q[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            Vs[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            w[t,1] = RHO[t]
            #if (disable_fire & (t>np.floor(nt/2).astype(int))):
            #    Q[t] = Q[t-1]
                #Q[t] = (N0[t-1]* np.minimum(self.p.q_0,1)+N1[t-1]*np.minimum(Q[t-1],1))/(N0[t-1]+N1[t-1])
            #Update probabilities for the data
            pr_j2j_array[F==1] = pr_j2j
            pr_sep_array[(F==1) & (S==1)] = pr_sep
            pr_sep_array[(F==1) & (S>1)] = pr_sep1            
            #Question: is it okay that I put these values before firing/hiring happens? I do this because I want to focus on the guys that do indeed have a chance to leave/be fired etc
            if update_z and z_dir is None:
                Z[t] = np.random.choice(model.p.num_z, 1, p=model.Z_trans_mat[Z[t-1]])

            #Workers leave. Once they leave, they're no longer tracked
            if allow_fire & ((not disable_fire) | (t<np.floor(nt/2).astype(int))):
                #Firing first:
                I_jun = (S==1) & (F==1)
                sep = np.random.binomial(1, pr_sep, I_jun.sum()) == 1
                I_sep_jun = bool_index_combine(I_jun,sep)
                I_sen = (S>1) & (F==1)
                sep1 = np.random.binomial(1, pr_sep1, I_sen.sum()) == 1
                I_sep_sen = bool_index_combine(I_sen,sep1)                
                I_sep = (I_sep_jun + I_sep_sen).astype(bool)
                F[I_sep] =0
                D[I_sep] = Event.e2u
            if allow_leave:
                #Next j2j
                I_rest = (F==1) #all the employed workers have a chance to leave
                j2j = np.random.binomial(1, pr_j2j, I_rest.sum())
                I_j2j = bool_index_combine(I_rest,j2j)
                F[I_j2j] = 0
                D[I_j2j] = Event.j2j
            #Update seniority of survivors
            S[F==1] = S[F==1] + 1

            #Update firm size
            N1[t] = ((F==1) & (S>1)).sum()
            if allow_hiring:
                N0[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
                #Discretize n0:
                n0_int = np.floor(N0[t])
                N0[t] = n0_int + np.random.binomial(1, N0[t]-n0_int)
                if update_z and z_dir is not None and (t==np.floor(nt/2).astype(int)):
                    #Force a hire that could potentially get fired
                    N0[t] = N0[t] + 1
                F[ni:ni+N0[t]] = 1 #these guys now employed
                S[ni:ni+N0[t]] = 1 #they're juniors
                D[ni:ni+N0[t]] = Event.u2e #they don't actually have to be hired from unemp btw           
            w[t,0] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[Z[t], N0[t], N1[t], ...], bounds_error=False, fill_value=None) ((RHO[t],Q[t])) ) #Is this time inconsistent??? Given that prod is decided later?
            W[ni:ni+N0[t]] = w[t,0] #junior wage paid
            ni = ni + N0[t] #add the extra workers 

            prod[t] = model.fun_prod_onedim[Z[t]] * np.interp(Q[t],model.Q_grid,model.prod[Z[t],N0[t],N1[t],0,:])
            #Update the employed worker info
            I_sen = (F==1) & (S>1) #Seniors that didn't leave
            W[I_sen] = np.log(w[t,1])
            D[I_sen] = Event.ee
            W1[I_sen] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[Z[t], N0[t], N1[t], ...,1], bounds_error=False, fill_value=None) ((RHO[t],Q[t]))
            #Firm info, added for every employed worker
            P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[Z[t], N0[t], N1[t], ...], bounds_error=False, fill_value=None) ((RHO[t],Q[t]))
            Y[F==1] = np.log(prod[t])
            N0_array[F==1] = N0[t]
            N1_array[F==1] = N1[t]
            Z_array[F==1] = Z[t]
            #print("shapes", F.shape,Z_array.shape,W.shape,S.shape,D.shape,W.shape, P.shape, N0_array.shape, N1_array.shape)
            all_df.append(pd.DataFrame({ 'i':range(n0+n1+extra),'t':t, 'f':F, 
                'z':Z_array, 'w':W , 'Pi':P, 'D': D, 'S': S, 'n0': N0_array, 'n1': N1_array,
                'pr_e2u':pr_sep_array, 'pr_j2j':pr_j2j_array , 'y':Y, 'W1':W1, 'vs':Vs, 
                }))
        all_df = pd.concat(all_df).sort_values(['i','t'])
        all_df.loc[all_df.f==0,['z','n0','n1', 'y', 'w', 'Pi', 'S', 'pr_e2u', 'pr_j2j', 'W1', 'vs']] = 0
        all_df['n'] = all_df['n0'].values + all_df['n1'].values              
        return all_df

    def simulate_firm_sep(self,z,n0,n1,rho,q,nt, force_sep=True,allow_hiring=True,allow_fire=True,allow_leave=True,update_z=False, z_dir=None,seed=False):
        """
        simulates a path of a particular firm from initial state [z,n0,n1,rho,q], forcing a separation at some point
        one can choose to allow the firm to expand or allow the workers to leave using allow_hiring and allow_leave
        one can choose to update z using update_z. choosing z_dir=1 or -1 will result in simulating just a single, deterministic, shock in the middle of the simulation
        one can choose to fix the seed using seed
        for the aggregate version (multiple firms), simulate_force_ee is a better comparison
        """
        if seed:
            np.random.seed(42)
        model = self.model
        all_df = []
        extra = 100 #extra workers to be potentially hired
        if allow_hiring == False:
            extra=0
        ni=n0+n1 #Number of workers employed
        W  = np.zeros(ni+extra)     # log-wage
        W1 = np.zeros(ni+extra)     # value to the worker
        Vs = np.zeros(ni+extra)     # search decision
        S = np.zeros(ni+extra)      # tenure at the firm
        D  = np.zeros(ni+extra,dtype=int)  # event  

        S[:n0] = 1          
        S[n0:ni] = 2
        Y = np.zeros(ni+extra)  # log-output
        P = np.zeros(ni+extra)      # firm profit
        pr_sep_array = np.zeros(ni+extra)  # probability, either u2e or e2u
        pr_j2j_array = np.zeros(ni+extra)  # probability, either u2e or e2u
        F = np.zeros(ni+extra) #==1 if worker is in the firm, zero otherwise    
        F[:ni] = 1
        N0_array = np.zeros(ni+extra)  
        N0_array[F==1] = n0 
        N1_array = np.zeros(ni+extra) 
        N1_array[F==1] = n1
        Z = np.zeros(nt,dtype=int) + z #if z is not updated, always keep it at the same value
        Z_array = np.zeros(ni+extra)
        Z_array[F==1] = z
        #If performing a deterministic prod shock
        if update_z and z_dir is not None:
            #Shock a
            Z[np.floor(nt/2).astype(int):] = z+z_dir
        N0 = np.zeros_like(Z)
        N0[0] = n0
        N1 = np.zeros_like(Z)
        N1[0] = n1
        RHO = np.zeros(nt)
        RHO[0] = rho
        Q = np.zeros_like(RHO)
        Q[0] = q
        prod = np.zeros(nt)
        prod[0] = np.interp(q,model.Q_grid,model.prod[z,n0,n1,0,:])
        w = np.zeros((nt,2))

        #Time 0 append with no update
        W[(F==1) & (S==1)] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[z, n0, n1, ...], bounds_error=False, fill_value=None) ((rho,q)) )
        W[(F==1) & (S>1)] = np.log(rho)
        Y[F==1] = np.log(prod[0])
        W1[(F==1) & (S>1)] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[z, n0, n1, ...,1], bounds_error=False, fill_value=None) (((rho,q)))
        #Firm info, added for every employed worker
        P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[z, n0, n1, ...], bounds_error=False, fill_value=None) ((rho,q))

        all_df.append(pd.DataFrame({ 'i':range(ni+extra),'t':0, 'f':F, 
                'z':Z_array, 'w':W , 'Pi':P, 'D': D, 'S': S, 'n0': N0_array, 'n1': N1_array,
                'pr_e2u':pr_sep_array, 'pr_j2j':pr_j2j_array , 'y':Y, 'W1':W1, 'vs':Vs, 
                }))
        #if pb:
        #    rr = tqdm(range(1,nt))
        #else:
        rr = range(1,nt) 

        for t in rr:
            #Update firm decisions first
            RHO[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            pr_j2j = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            pr_sep = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            pr_sep1 = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star1[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))            
            Q[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            Vs[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
            w[t,1] = RHO[t]
            #Update probabilities for the data
            pr_j2j_array[F==1] = pr_j2j
            pr_sep_array[(F==1) & (S==1)] = pr_sep
            pr_sep_array[(F==1) & (S>1)] = pr_sep1            
            #Question: is it okay that I put these values before firing/hiring happens? I do this because I want to focus on the guys that do indeed have a chance to leave/be fired etc
            if update_z and z_dir is None:
                Z[t] = np.random.choice(model.p.num_z, 1, p=model.Z_trans_mat[Z[t-1]])

            #Workers leave. Once they leave, they're no longer tracked
            if force_sep and (t==np.floor(nt/2).astype(int)): #Easy quick route: raise sep probability
                    #fired_workers = np.random.choice(
                    #vacancy_indices, 
                    #size=1, 
                    #replace=False)
                I_f = F==1
                sep = np.random.binomial(1, 0.25, I_f.sum()) == 1
                I_sep = bool_index_combine(I_f,sep)  
                F[I_sep] =0
                D[I_sep] = Event.e2u           

            if allow_fire:
                #Firing first:
                I_jun = (S==1) & (F==1)
                sep = np.random.binomial(1, pr_sep, I_jun.sum()) == 1
                I_sep_jun = bool_index_combine(I_jun,sep)
                I_sen = (S>1) & (F==1)
                sep1 = np.random.binomial(1, pr_sep1, I_sen.sum()) == 1
                I_sep_sen = bool_index_combine(I_sen,sep1)                
                I_sep = (I_sep_jun + I_sep_sen).astype(bool)
                F[I_sep] =0
                D[I_sep] = Event.e2u
            if allow_leave:
                #Next j2j
                I_rest = (F==1) #all the employed workers have a chance to leave
                j2j = np.random.binomial(1, pr_j2j, I_rest.sum())
                I_j2j = bool_index_combine(I_rest,j2j)
                F[I_j2j] = 0
                D[I_j2j] = Event.j2j
            #Update seniority of survivors
            S[F==1] = S[F==1] + 1

            #Update firm size
            N1[t] = ((F==1) & (S>1)).sum()
            if allow_hiring:
                N0[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[Z[t-1], N0[t-1],N1[t-1], ...], bounds_error=False, fill_value=None) ((RHO[t-1],Q[t-1]))
                #Discretize n0:
                n0_int = np.floor(N0[t])
                N0[t] = n0_int + np.random.binomial(1, N0[t]-n0_int)
                if update_z and z_dir is not None and (t==np.floor(nt/2).astype(int)):
                    #Force a hire that could potentially get fired
                    N0[t] = N0[t] + 1
                F[ni:ni+N0[t]] = 1 #these guys now employed
                S[ni:ni+N0[t]] = 1 #they're juniors
                D[ni:ni+N0[t]] = Event.u2e #they don't actually have to be hired from unemp btw           
            w[t,0] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[Z[t], N0[t], N1[t], ...], bounds_error=False, fill_value=None) ((RHO[t],Q[t])) ) #Is this time inconsistent??? Given that prod is decided later?
            W[ni:ni+N0[t]] = w[t,0] #junior wage paid
            ni = ni + N0[t] #add the extra workers 

            prod[t] = model.fun_prod_onedim[Z[t]] * np.interp(Q[t],model.Q_grid,model.prod[Z[t],N0[t],N1[t],0,:])
            #Update the employed worker info
            I_sen = (F==1) & (S>1) #Seniors that didn't leave
            W[I_sen] = np.log(w[t,1])
            D[I_sen] = Event.ee
            W1[I_sen] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[Z[t], N0[t], N1[t], ...,1], bounds_error=False, fill_value=None) ((RHO[t],Q[t]))
            #Firm info, added for every employed worker
            P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[Z[t], N0[t], N1[t], ...], bounds_error=False, fill_value=None) ((RHO[t],Q[t]))
            Y[F==1] = np.log(prod[t])
            N0_array[F==1] = N0[t]
            N1_array[F==1] = N1[t]
            Z_array[F==1] = Z[t]
            #print("shapes", F.shape,Z_array.shape,W.shape,S.shape,D.shape,W.shape, P.shape, N0_array.shape, N1_array.shape)
            all_df.append(pd.DataFrame({ 'i':range(n0+n1+extra),'t':t, 'f':F, 
                'z':Z_array, 'w':W , 'Pi':P, 'D': D, 'S': S, 'n0': N0_array, 'n1': N1_array,
                'pr_e2u':pr_sep_array, 'pr_j2j':pr_j2j_array , 'y':Y, 'W1':W1, 'vs':Vs, 
                }))
        all_df = pd.concat(all_df).sort_values(['i','t'])
        all_df.loc[all_df.f==0,['z','n0','n1', 'y', 'w', 'Pi', 'S', 'pr_e2u', 'pr_j2j', 'W1', 'vs']] = 0
        all_df['n'] = all_df['n0'].values + all_df['n1'].values              
        return all_df

    def simulate_force_ee(self,X0,Z0,H0,R0,nt,update_x=True, update_z=True, pb=False):
        """
        init should give the vector of initial values of X,Z,rho
        we start from this initial value and simulate forward
        one can choose to update x, z using update_z and update_x
        one can choose to show a progress bar with pb=True
        """
        X  = X0.copy() # current value of the X shock
        R  = R0.copy() # current value of rho
        H  = H0.copy() # location in the firm shock history (so that workers share common histories)
        Z  = Z0.copy() # location in the firm shock history (so that workers share common histories)

        ni = len(X)
        W  = np.zeros(ni)     # log-wage
        W1 = np.zeros(ni)     # value to the worker
        Ef = np.zeros(ni)     # effort
        Vs = np.zeros(ni)     # search decision
        tw = np.zeros(ni)     # target wage

        Y = np.zeros(ni)  # log-output
        P = np.zeros(ni)      # firm profit
        pr_sep = np.zeros(ni)  # probability, either u2e or e2u
        pr_j2j = np.zeros(ni)  # probability, either u2e or e2u

        model = self.model
        nl = self.Zhist.shape[1]
        all_df = []

        if pb:
            rr = tqdm(range(nt))
        else:
            rr = range(nt) 

        for t in rr:

            # we store the outcomes at the current state
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    Ixz_ee = (X == ix) & (Z == iz)
                    if Ixz_ee.sum() == 0: continue

                    Y[Ixz_ee] = np.log(model.fun_prod[iz,ix])
                    pr_sep[Ixz_ee] = np.interp( R[Ixz_ee], model.rho_grid , model.qe_star[iz,:,ix])
                    pr_j2j[Ixz_ee] = np.interp( R[Ixz_ee], model.rho_grid , model.pe_star[iz,:,ix])
                    W[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                    W1[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.Vf_W1[iz, :, ix] )  # value to the worker
                    P[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.Vf_J[iz, :, ix])  # interpolate firm Expected profit 
                    Vs[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.ve_star[iz, :, ix])  # interpolate firm Expected profit 
                    tw[Ixz_ee] = np.log(model.target_w[iz,ix])

            ef = np.log(model.pref.inv_utility(model.pref.effort_cost(pr_sep)))
            all_df.append(pd.DataFrame({ 'i':range(ni),'t':t, 'h':H, 
                'x':X , 'z':Z, 'r':R, 'w':W , 'Pi':P, 
                'pr_e2u':pr_sep, 'pr_j2j':pr_j2j , 'y':Y, 'W1':W1, 'vs':Vs, 
                'target_wage':tw, 'effort': ef }))

            # we update the different shocks
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    Ixz_ee = (X == ix) & (Z == iz)
                    if Ixz_ee.sum() == 0: continue
                    R[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.rho_star[iz,:,ix]) # find the rho using law of motion

            if update_x:
                for ix in range(self.p.num_x):
                    Ixz_ee = (X == ix) 
                    if Ixz_ee.sum() == 0: continue
                    X[Ixz_ee] = np.random.choice(self.p.num_x, Ixz_ee.sum(), p=model.X_trans_mat[:,ix])

            if update_z:
                for iz in range(self.p.num_z):
                    Ixz_ee = (Z == iz)
                    if Ixz_ee.sum() == 0: continue
                    Z[Ixz_ee] = self.Zhist[ (Z[Ixz_ee] , H[Ixz_ee]) ] # extract the next Z from the pre-computed histories
                    H[Ixz_ee] = (H[Ixz_ee] + 1) % nl                  # increment the history by 1
            
        return pd.concat(all_df).sort_values(['i','t'])

    def get_sdata(self):
        return(self.sdata)

    def get_yearly_data(self):

        sdata = self.sdata

        # compute firm output and sizes at year level
        hdata = (sdata.set_index(['i', 't'])
                      .pipe(create_lag_i, 't', ['d'], -1)
                      .reset_index()
                      .query('f>=0')
                      .assign(c_e2u=lambda d: d.d_f1 == Event.e2u,
                              c_j2j=lambda d: d.d_f1 == Event.j2j)
                      .groupby(['f'])
                      .agg( {'prod': 'sum', 'i': "count", 'c_e2u': 'sum', 'c_j2j': 'sum'}))
        hdata['prod_year'] = hdata.prod + np.roll(hdata.prod, -1) + np.roll(hdata.prod, 0) + np.roll(hdata.prod, -3)
        hdata['c_year'] = hdata.i + np.roll(hdata.i, -1) + np.roll(hdata.i, 0) + np.roll(hdata.i, -3) #Total number of workers at the firm in a year. How do they do this though?
        #i is an indicator of a worker, no? So aren't they just summing up the indicators here?
        hdata['c_e2u_year'] = hdata.c_e2u + np.roll(hdata.c_e2u, -1) + np.roll(hdata.c_e2u, 0) + np.roll(hdata.c_e2u, -3)
        hdata['c_j2j_year'] = hdata.c_j2j + np.roll(hdata.c_j2j, -1) + np.roll(hdata.c_j2j, 0) + np.roll(hdata.c_j2j, -3)
        hdata['ypw'] = np.log(hdata.prod_year/hdata.c_year) #Output per worker at the firm level within a year
        hdata['lsize'] = np.log(hdata.c_year/4) # log number of worker in the year

        # create year on year growth at the firm level
        hdata['le2u'] = np.log(hdata['c_e2u_year'] / hdata['c_year'])
        hdata['lj2j'] = np.log(hdata['c_j2j_year'] / hdata['c_year'])
        hdata['lsep'] = np.log((hdata['c_j2j_year'] + hdata['c_e2u_year']) / hdata['c_year'])
        hdata = hdata.drop(columns='i')

        # add measurement error to ypw
        hdata_sep = (hdata.assign(ypwe=lambda d: d.ypw + self.p.prod_err_y * np.random.normal(size=len(d.ypw)))
                          .pipe(create_lag, 'h', ['ypw', 'ypwe', 'le2u', 'lj2j', 'lsep'], 4)
                          .assign(dlypw=lambda d: d.ypw - d.ypw_l4,
                                  dlypwe=lambda d: d.ypwe - d.ypwe_l4,
                                  dle2u=lambda d: d.le2u - d.le2u_l4,
                                  dlsep=lambda d: d.lsep - d.lsep_l4,
                                  dlj2j=lambda d: d.lj2j - d.lj2j_l4)[['dlypw', 'dlypwe', 'dle2u', 'dlj2j', 'dlsep', 'c_year']])

        # compute wages at the yearly level, for stayers
        sdata['s2'] = sdata['s']
        sdata['es'] = sdata['e']
        #sdata['w_exp'] = np.exp(sdata['w'])

        sdata_y = sdata.groupby(['i', 'year']).agg({'w': 'sum', 'f': 'min', 's': 'min', 's2': 'max', 'e': 'min', 'es': 'sum'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's'], -1).pipe(create_year_lag, ['e', 'es'], 1)
        # make sure we stay in the same spell, and make sure it is employment
        sdata_y = sdata_y.query('f>=0').query('s+3==s2')
        #sdata_y['w'] = np.log(sdata_y['w'])

        # attach firm output, compute lags and growth
        sdata_y = (sdata_y.join(hdata.ypw, on="f")
                          .pipe(create_year_lag, ['ypw', 'w', 's', 'f'], 1)
                          .assign(dw=lambda d: d.w - d.w_l1,
                                  dypw=lambda d: d.ypw - d.ypw_l1))


        return(sdata_y)

    def computeMoments(self):
        """
        Computes the simulated moments using the simulated data
        :return:
        """
        sdata = self.sdata
        moms = {}
 
        # extract total output
        moms['total_output'] = sdata.query('f>0')['prod'].sum()/len(sdata)
        moms['total_wage'] = sdata.query('f>0')['w'].sum()/len(sdata) #Used to have an exponent here, but I was not taking a log of wages anywhere
        #moms['total_wage_net'] = np.exp(sdata.query('h>0')['w_net']).sum()/len(sdata)
        moms['total_uben'] = self.p.u_bf_m * sdata.eval('f==0').sum()/len(sdata)

        # ------  transition rates   -------
        # compute unconditional transition probabilities
        u2e = (sdata['d'] == Event.u2e).sum()
        eu  = (sdata['d'] == Event.e2u).sum()
        j2j = (sdata['d'] == Event.j2j).sum()
        #moms['pr_u2e'] = sdata.eval('d==@Event.u2e').sum() / sdata.eval('d==@Event.u2e | d==@Event.uu').sum()
        #moms['pr_j2j'] = sdata.eval('d==@Event.j2j').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        #moms['pr_e2u'] = sdata.eval('d==@Event.e2u').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()

        #moms['pr_u2e'] = sdata.eval('d==@Event.u2e').sum() / sdata.eval('d==@Event.u2e | d==@Event.uu').sum()
        #moms['pr_j2j'] = sdata.eval('d==@Event.j2j').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        #moms['pr_e2u'] = sdata.eval('d==@Event.e2u').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        # ------  earnings and value added moments at yearly frequency  -------
        # compute firm output and sizes at year level
        hdata = (sdata.set_index(['i', 't'])
                      .pipe(create_lag_i, 't', ['d'], -1) #This is a forward lag, so that d appears on the employed workers
                      .reset_index()
                      .query('e>0') #That (using f>0) would still include closed firms though! So I should do e>0 instead?
                      .assign(e2u=lambda d: d.d_f1 == Event.e2u,
                              c_e2u=lambda d: d.d_f1 == Event.e2u,
                              c_j2j=lambda d: d.d_f1 == Event.j2j)
                      .groupby(['f','t'])
                      .agg( {'prod': 'max', 'i': "count", 'c_e2u': 'sum', 'c_j2j': 'sum', 'n' : 'max', 'w': 'mean'})) #Also we don't want to sum prod! It is already at the firm level so don't sum it up.
        #Also do we want to check? Whether i == n? i is the empirical count of workers, n is the theoretical count of workers, right? So they should be equal?
        #assert (hdata['i'] == hdata['n']).all() #Also a good check to see if I understand the data correctly
        #This assertion is not correctionb because I am clamping the size of the firm.
        #These ones below not needed, since the data is already at the yearly level
        #hdata['prod_year'] = hdata.prod + np.roll(hdata.prod, -1) + np.roll(hdata.prod, 0) + np.roll(hdata.prod, -3)
        #hdata['c_year'] = hdata.i + np.roll(hdata.i, -1) + np.roll(hdata.i, 0) + np.roll(hdata.i, -3) #wait why is this not averaged? what if the workers are repeated? then we count them multiple times?
        #hdata['c_e2u_year'] = hdata.c_e2u + np.roll(hdata.c_e2u, -1) + np.roll(hdata.c_e2u, 0) + np.roll(hdata.c_e2u, -3)
        #hdata['c_j2j_year'] = hdata.c_j2j + np.roll(hdata.c_j2j, -1) + np.roll(hdata.c_j2j, 0) + np.roll(hdata.c_j2j, -3)
        #hdata['ypw'] = np.log(hdata.prod_year/hdata.c_year)
        hdata = hdata.sort_values(['f', 't'])  
        hdata['ypw'] = np.log(hdata['prod']/hdata['n'])
        hdata = hdata.sort_values(['f', 't']) #Sort by firm and time, so that the lags work correctly
        hdata['dypw'] = hdata['ypw'] - hdata.groupby('f')['ypw'].shift(1) #I want the growth in logs, right? CHECK IN THE DATA. Souchier does it in logs.
        hdata['id_shock_diff'] = hdata['dypw']
        hdata['id_shock_sum'] = hdata.groupby('f')['ypw'].shift(-1) - hdata.groupby('f')['ypw'].shift(2)
        hdata['id_shock_sum_lag'] = hdata['ypw'] - hdata.groupby('f')['ypw'].shift(3)
        #hdata['id_shock_sum'] = hdata.pipe(create_lag, 'f', ['ypw'], -1)['ypw_f1'] - hdata.pipe(create_lag, 'f', ['ypw'], 2)['ypw_l2'] #Cumulative log sum over the 3 years, 1 forward and 2 back
        #hdata['id_shock_sum_lag'] = hdata['ypw'] - hdata.pipe(create_lag, 'f', ['ypw'], 3)['ypw_l3']

        #hdata['id_shock_sum_lag'] = hdata.pipe(create_lag, 'f', ['id_shock_sum'], 1)['id_shock_sum_l1'] #This is the lagged cumulative log sum over the 3 years, 3 back
        #And to confirm
        #assert np.all(hdata.dropna(subset=['id_shock_sum','id_shock_sum_lag']).groupby('f')['id_shock_sum_lag'].shift(0) == hdata.dropna(subset=['id_shock_sum','id_shock_sum_lag']).groupby('f')['id_shock_sum'].shift(1)) #Weird assertion error
        #hdata['lsize'] = np.log(hdata.c_year/4) # log number of worker in the year
        #hdata['c_year_mean'] = hdata.c_year / 4  # average number of workers in the year

        # create year on year growth at the firm level
        #hdata['le2u'] = np.log(hdata['c_e2u_year'] / hdata['c_year'])
        #hdata['lj2j'] = np.log(hdata['c_j2j_year'] / hdata['c_year'])
        #hdata['lsep'] = np.log((hdata['c_j2j_year'] + hdata['c_e2u_year']) / hdata['c_year'])
        #hdata = hdata.drop(columns='i')

        #Andrei: Now Adding my own moments
        #1.Aggregate transitions
        #a) hiring rate = moms['pr_u2e'](~ average duration of a non-employment spell)
        #Wait, or did I do a rate of new hires, aka sum(new==1)/sum(new==1 | neww==0)? I think that's what I did
        # So here it would be (s==1 & e==1)/(e==1). This is not yearly, but I do not do this yearly in the data either!
        moms['pr_new_hire'] = sdata.eval('s==1 & e==1').sum() / sdata.eval('e==1').sum()
        #b) annual e2u rate = moms['pr_e2u']
        moms['pr_e2u'] = (sdata['d'] == Event.e2u).sum() / ((sdata['d'] == Event.j2j).sum() + (sdata['d'] == Event.ee).sum() + (sdata['d'] == Event.e2u).sum())

        #moms['pr_e2u'] = sdata.eval('d==@Event.e2u').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        
        #c) annuanl j2j rate = moms['pr_j2j']
        moms['pr_j2j'] = (sdata['d'] == Event.j2j).sum() / ((sdata['d'] == Event.j2j).sum() + (sdata['d'] == Event.ee).sum() + (sdata['d'] == Event.e2u).sum())
        #moms['pr_j2j'] = sdata.eval('d==@Event.j2j').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()

        #2. Tenure profile of wages at 7.5 years (how did Souchier do that?)
        # In my data, I took the average wage growth for each tenure group, and then summed/multiplied them all.
        # So here, I could take the average wage growth for each s[e==1], and then combine them all. Do I want to have low wages then?
        #sdata.query('e==1 & s==1')['w'].log() - sdata.query('e==1 & s==1').pipe(create_year_lag, ['w'], 7).log() #This is the wage growth over 7 years
        # Step 1: Keep only employed individuals
        employed = sdata.sort_values(['i', 't']).copy()
        employed['e2u_emp'] = (employed.groupby('i')['d'].shift(1) == Event.e2u)
        employed = employed.query('e == 1')

        # Step 2: Sort by individual and time
        employed = employed.sort_values(['i', 't'])

        # Step 3: Compute log wage growth (log difference)
        employed['w_growth_rate'] = (np.log(employed['w']) - np.log(employed.groupby('i')['w'].shift(1)))   
        #sdata['w_growth_rate'] = np.log(sdata.query('e==1').groupby('i')['w']) - np.log(sdata.query('e==1').groupby('i')['w'].shift(1)) #No this is kinda trash, it just a single growth rate 7 years back
        #So I need to take the average wage growth for each tenure group, and then sum them up
        #So first, I need to group by s, and then take the mean of w_growth_rate
        #But I also need to make sure that I only take the ones with e==1, so I can do it in a query
        #sdata['w_growth_rate_tenure']=sdata.query('e==1').groupby('s')['w_growth_rate'].mean()
        #Now I wanna take the average wage growth for each tenure group then sum them up
        moms['avg_w_growth_7'] = employed.groupby('s')['w_growth_rate'].mean().sum()
        #So what I want is the following: get the growth rate
        #3. Productivity moments
        #a) s.d. of firm productivity growth (take sd of dypw?) yep, this is exactly what we do in the data
        moms['sd_dypw'] = hdata['dypw'].std()
        #b) annual persistence of firm productivity (take autocovariance of ypw?)
        #moms['autocov_dypw'] = hdata.pipe(create_year_lag, ['ypw'], 1).pipe(create_year_lag, ['ypw'], 2)[['ypw', 'ypw_l1', 'ypw_l2']].cov()['ypw']['ypw_l1'] #This is kinda weird? This was suggested code, but I don't get it. How does it work? Detailed answer: it takes the lagged ypw, and then computes the covariance between the current and lagged ypw. So it is like a persistence measure, but not really an autocorrelation, because it is not normalized by variance
        hdata['ypw_l1'] = hdata.groupby('f')['ypw'].shift(1)
        moms['autocov_ypw'] = hdata.cov()['ypw']['ypw_l1']
        #4. Firm dynamics moments
        #a) average firm size (mean of c_year?)
        moms['avg_firm_size'] = hdata['i'].mean() #Average number of workers in a firm in a year
        #b) ratio of jobs created by opening firms (need to denote opening firms, then sum up c_u2e_year (doesn't exist yet) for them divide by c_u2e_year for everyone)
        #Wait, do I want u2e? I don't think that's what I did in the data, did I? It may have been jut a ratio of new==1 between opening firms and all firms
        #First denote opening firms, that is, firms that for the first time (not the second or third) have a non-zero t
        # Get the first period each firm appears in
        firm_entry = sdata.query('i > 0').groupby('f')['t'].min()

        # Map this back to each row in sdata
        sdata['firm_entry_time'] = sdata['f'].map(firm_entry)

        # Compare: Is this the firm's first period?
        sdata['opening_firm'] = sdata['t'] == sdata['firm_entry_time']
        moms['ratio_jobs_opening'] = sdata.query('opening_firm').eval('s==1 & e==1').sum() / sdata.eval('s==1 & e==1').sum()
        #c?) proportion of jobs created by firms of certain size (>10/>100 etc. do I need this? if I fix the DRS factor, then not)
        #5. HMQ moments
        #a) Most basic approach: passthrough of productivity wrt layoffs. That's just one moment though. Also this is quite close to one of my untargeted regressions, so maybe I shouldn't use it
        #b) response of sen/junior wage ratio to layoffs. doesn't work well with the 2 tenure steps imo, given that juniors have the bonus wage as well, which might mess things up
        #Can consider it in a 3-step scenario though? And compare this across 2nd and 3rd steps? Or maybe consider wages without the bonus part?
        #c) Labor share approach: I don't have any capital though??? So what would I look for? I guess my production IS the labor share? So look at how prod and ypw change with layoffs?
        # Yeah, maybe do the labor share approach, gives me exactly two moments
        # Do I do the real labor share here? Or just the prod and ypw? Because in the data it was also kinda meh
        # First, do the fake one, just working with prod and ypw. So need to regress prod and ypw on layoffs, and then take the coefficients
        # Also though, do I regress on EU, or on EU_rate? Let's try regressing on EU
        #Can I take a forward lag of e2u to get the e2u for employed workers? I think I can, so let's do that
        #hdata['e2u'] = ( sdata['d'] == Event.e2u ).astype(int)
        #hdata['e2u_emp'] = hdata.pipe(create_lag, 'i', ['c_e2u'], -1)['c_e2u'] #This is the e2u for employed workers, so it is the forward lag of e2u
        #hdata_emp = sdata.query('e==1').copy() #Keep only employed workers. But wait, where does even e2u reside? Doesn't it reside only with unemployed workers? Yep, so the issue that I have is that I don't have e2u for employed workers, like I do in the data
        #Now we regress ypw on e2u_emp
        #employed['e2u_emp'] = (employed['d'] == Event.e2u) #ahhh fuck, need to get a lag first!!!
        employed['ypw'] = np.log(employed['prod']/employed['n'])
        model_ypw = feols('ypw ~ 1 + e2u_emp', data=employed)
        moms['ypw_layoffs'] = model_ypw.coef()['e2u_emp']
        model_prod = feols('prod ~ e2u_emp', data=employed)
        moms['prod_layoffs'] = model_prod.coef()['e2u_emp']

        #d) Fabrice suggestion: distribution of layoffs across firm productivity. What would the firm productivity be? I guess the firm output per worker, so ypw?
        # So in the data, I could take terciles of productivity, and then compute the share of layoffs in each tercile. And then same here. Then, if I do terciles, can get rid of the aggregate moment
        #So here, first get the terciles of ypw 
        hdata['ypw_tercile'] = pd.qcut(hdata['ypw'], 3, labels=False)
        #Then compute the share of layoffs in each tercile
        hdata['layoff_rate'] = hdata['c_e2u'] / hdata['i']
        moms['layoffs_share_tercile'] = (hdata.groupby('ypw_tercile')['layoff_rate']
                                                .mean().rename('layoffs_share_tercile'))

        #del wid_2spells 
        #del sdata_y 
        self.hdata = hdata
        self.employed = employed
        self.moments = moms
        return self

    def model_evaluation(self):
        """
        Computes the simulated NONTARGETED moments using the simulated data
        I might also want to produce the plots here, if I intend to replicate any
        :return:
        """
        employed = self.employed
        moms_untarg = {}

        #Now I want to run my regressions. First, let's run basic wages and layoffs across worker tenure in first-differences.
        #Get log wage growth
        #self.hdata['w_growth'] = np.log(self.hdata['w']) - np.log(self.hdata.pipe(create_lag_i, 't', ['w'], 1)['w_l1'])
        # Merge hdata['id_shock_sum'] onto employed using firm ID and time
        employed = employed.merge(self.hdata.reset_index()[['f', 't', 'id_shock_sum', 'id_shock_diff']], on=['f', 't'], how='left')

        # Regress log wage growth on the interaction between tenure and firm productivity shock 
        #We get hdata['id_shock_sum'] as the cumulative log shock over the 3 years
        employed['tenure'] = employed['s']
        model_wage_ten = feols('w_growth_rate ~ i(tenure, id_shock_sum)', data=employed)
        moms_untarg['wage_pass_ten'] = model_wage_ten.coef() #This is HUGE so far. more than 1 for S==2!!!  Even bigger if I use id_shock_diff??? Surprising ngl
        model_wage_ten = feols('w_growth_rate ~ id_shock_sum + tenure * id_shock_sum', data=employed)
        moms_untarg['wage_pass_ten_simple'] = model_wage_ten.coef()  #positive baseline effect of id_shock_sum, POSITIVE effect of tenure (WRONG!), negative interaction (WRONG!!!!!!!)    
        #It's also very weird after the first 2 tenures for some reason
        #Ah shit, should I actually be looking at the first 2 guys?? 
        # The very first tenure they've got the bonus wage
        # The second year they become seniors, which is a huge huge jump in wages, too
        # But past that it's immediately negative??? I guess, a positive shock now means new hirings?
        # And then you don't wanna raise wages much in anticipation of future seniors? That kinda sucks ngl. 
        # Is there any way to deal with this outside of more steps? Maybe I don't let firms internalize the combination???
        # But then the value function is quite literally incorrect! Okay gotta focus some more on Neural Nets ig
        model_e2u_ten = feols('e2u_emp ~ C(tenure)', data=employed)
        moms_untarg['sep_ten'] = model_e2u_ten.coef() #So far it's only juniors getting fired. Surprisingly, their firing rates are not very high, only 8%      
        model_e2u_pass_ten = feols('e2u_emp ~ i(tenure, id_shock_sum)', data=employed)
        moms_untarg['sep_pass_ten'] = model_e2u_pass_ten.coef()        
        model_e2u_pass_ten = feols('e2u_emp ~ id_shock_sum + tenure * id_shock_sum', data=employed)
        moms_untarg['sep_pass_ten_simple'] = model_e2u_pass_ten.coef()  #exactly as we would expect! Negative id_shock_sum coef (-0.08), negative basic tenure coef (very small though? -0.002), POSITIVE interaction term, meaning that layoffs of seniors respond less to shock     
        self.moments_untargeted = moms_untarg
        return self

    def clean(self):
        del self.sdata
        gc.collect()

    def compute_growth_var_by_xz(self):
        """ 
        returns wage and match output growth variance for each (x,z) types.

        this function is useful for the coutnerfactual decomposition of wage
        and output growth """

        sdata = self.sdata
        sdata['w_exp'] = np.exp(sdata['w'])
        sdata['s2'] = sdata['s']

        sdata_y = sdata.groupby(['i', 'year']).agg({'w_exp': 'sum', 'f':'sum', 
                    'h':'min', 's': 'min', 's2': 'max', 
                    'e': 'min', 'x':'first', 'z':'first'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's', 'f'], 1)
        #sdata_y = sdata_y.pipe(create_year_lag, ['e', 's', 'f'], 2)
        sdata_y = sdata_y.query('h>=0').query('s+3==s2')
        sdata_y['w'] = np.log(sdata_y['w_exp'])
        sdata_y['lf'] = np.log(sdata_y['f'])
        sdata_y = sdata_y.pipe(create_year_lag, ['w', 'lf'], 1)

        dd = sdata_y.assign( dw = lambda d: d.w - d.w_l1,
                             df = lambda d: d.lf - d.lf_l1
                             ).groupby(['x','z']).agg(
                                 dw_m=('dw','mean'),
                                 dw_v=('dw','var'),
                                 df_m=('df','mean'),
                                 df_v=('df','var'),
                                 e_count=('e','count'))

        return dd 

    def get_moments(self):
        return self.moments, self.moments_untargeted

    def simulate_moments_rep(self, nrep):
        """
        simulates moments from the model, running it multiple times
        :param nrep: number of replications
        :return:
        """

        moms = pd.DataFrame()
        moms_unt = pd.DataFrame()
        self.log.info("Simulating {} reps".format(nrep))
        for i in range(nrep):
            self.log.debug("Simulating rep {}/{}".format(i+1, nrep))
            mom,mom_unt = self.simulate_val().computeMoments().model_evaluation().get_moments()
            moms = pd.concat([ moms, pd.DataFrame({ k:[v] for k,v in mom.items() })] , axis=0)
            moms_unt = pd.concat([ moms_unt, pd.DataFrame({ k:[v] for k,v in mom_unt.items() })] , axis=0)
            self.clean()
        self.log.info("done simulating")
        moms_mean = moms.mean().rename('value_model')
        moms_var = moms.var().rename('value_model_var')
        moms_unt_mean = moms_unt.mean('regressions')
        moms_unt_var = moms_unt.var().rename('regressions_var')

        return(moms_mean, moms_var,moms_unt_mean,moms_unt_var)

#Okay, gotta debug
def get_results_for_p(p,all_results):
    # Create the key as a tuple
    #key = (p.num_z,p.num_v,p.num_n,p.n_bar,p.num_q,p.q_0,p.prod_q,p.hire_c,p.k_entry,p.k_f,p.prod_alpha,p.dt)
    key = (p.num_z,p.num_v,p.num_n,p.n_bar,p.num_q,p.q_0,p.prod_q,p.hire_c,p.prod_alpha,p.dt,p.u_bf_m)
    # Check if the key exists in the saved results
    if key in all_results:
        print(key)
        return all_results[key]
    else:
        print(f"No results found for p = {key}")
        return None
from primitives import Parameters
p = Parameters()
import pickle
import matplotlib.pyplot as plt
import numpy as np
from plots import Plots
import cProfile
import pstats
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model_GE.pkl")
print("Loading model from:", model_path)
with open(model_path, "rb") as file:
    all_results = pickle.load(file)
model = get_results_for_p(p,all_results)
sim = Simulator(model,p)
sim.simulate_moments_rep(1)


""" 
Old Notes
#Old 2d, slightly (just barely) slower, because it needed to re-initialize every time:
            beg_old=time()
            for iz in range(p.num_z):
                for in0 in range(p.num_n):
                    for in1 in range(p.num_n):
                        F_spec = F_set[(z[t-1,F_set]==iz) & (n0[t-1,F_set]==in0) & (n1[t-1,F_set]==in1)]
                        if len(F_spec)==0:
                            continue
                        rho[t,F_spec] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[iz, in0,in1, ...], bounds_error=False, fill_value=None) ((rho[t-1,F_spec],q[t-1,F_spec]))
                        n_hire[t,F_spec] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[iz, in0,in1, ...], bounds_error=False, fill_value=None) ((rho[t-1,F_spec],q[t-1,F_spec]))
                        pr_j2j[t,F_spec] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[iz, in0,in1, ...], bounds_error=False, fill_value=None) ((rho[t-1,F_spec],q[t-1,F_spec]))
                        sep_rate[t,F_spec] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[iz, in0,in1, ...], bounds_error=False, fill_value=None) ((rho[t-1,F_spec],q[t-1,F_spec]))
                        q[t,F_spec] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[iz, in0,in1, ...], bounds_error=False, fill_value=None) ((rho[t-1,F_spec],q[t-1,F_spec]))
                        ve_star[t,F_spec] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[iz, in0,in1, ...], bounds_error=False, fill_value=None) ((rho[t-1,F_spec],q[t-1,F_spec]))
            end_old=time()


            #1d approach: here I do not interpolate for rho, instead I assign it to the closest value. Cost: max diff between the two rho's is 0.4, which is quite a bit imo
            #rho_diff = np.zeros(nf+1+extra)
            #for f in F_set:
                #rho_diff[f] = np.interp(q[t-1,f],model.Q_grid,model.rho_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:])
                #This rho_diff uses idx from the previous period! so the comparisons are always valid
            #    rho_diff[f] = model.rho_grid[np.argmin(np.abs(model.rho_grid - np.interp(q[t-1,f],model.Q_grid,model.rho_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho_diff_idx[f],:]))).astype(int)]
            #    rho_diff_idx[f] = np.argmin(np.abs(model.rho_grid - np.interp(q[t-1,f],model.Q_grid,model.rho_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho_diff_idx[f],:]))).astype(int)
            #    w[t,f,1] = model.pref.inv_utility_1d(np.interp(q[t-1,f],model.Q_grid,model.rho_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:]))

            #    n_hire[t,f] = np.interp(q[t-1,f],model.Q_grid,model.n0_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:])
            #   pr_j2j[t,f] = np.interp(q[t-1,f],model.Q_grid,model.pe_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:])
            #    sep_rate[t,f] = np.interp(q[t-1,f],model.Q_grid,model.sep_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:])
            #    q[t,f] = np.interp(q[t-1,f],model.Q_grid,model.q_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:])
            #    ve_star[t,f] = np.interp(q[t-1,f],model.Q_grid,model.ve_star[z[t-1,f],n0[t-1,f],n1[t-1,f],rho[t-1,f],:])           
            
            
            #for f in F_set: 
            #    #Via 2d interpolation (may be too slow, we'll see). Could I do this in numba?
            #    rho[t,f] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[z[t-1,f], n0[t-1,f],n1[t-1,f], ...], bounds_error=False, fill_value=None) ((rho[t-1,f],q[t-1,f]))
            #    w[t,f,1] = model.pref.inv_utility_1d(rho[t,f])

            #    n_hire[t,f] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[z[t-1,f], n0[t-1,f],n1[t-1,f], ...], bounds_error=False, fill_value=None) ((rho[t-1,f],q[t-1,f]))
            #    pr_j2j[t,f] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[z[t-1,f], n0[t-1,f],n1[t-1,f], ...], bounds_error=False, fill_value=None) ((rho[t-1,f],q[t-1,f]))
            #    sep_rate[t,f] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[z[t-1,f], n0[t-1,f],n1[t-1,f], ...], bounds_error=False, fill_value=None) ((rho[t-1,f],q[t-1,f]))
            #    q[t,f] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[z[t-1,f], n0[t-1,f],n1[t-1,f], ...], bounds_error=False, fill_value=None) ((rho[t-1,f],q[t-1,f]))
            #    ve_star[t,f] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[z[t-1,f], n0[t-1,f],n1[t-1,f], ...], bounds_error=False, fill_value=None) ((rho[t-1,f],q[t-1,f]))
            #    bon_leave[t,f] = model.pref.inv_utility(ve_star[t,f] - model.v_0) - model.pref.inv_utility(model.v_grid[0] - model.v_0) #Bonus wage if leave the current firm
            #print("len fset",len(F_set))



            #Andrei: can start by updating everything for firms
            #for ifirm in range() #Probably easier to do it via a loop? So yes, iterate over all the firms and add the crucial information: 
            # today's size (or tomorrow's? depends on the timing)
            # whether it hires: if so, how many
            # whether it fires: if so, how many
            # maybe I do it per state rather than per firm? that would make randomization across firms in the same state easier also probably
            # BUT WAIT! WE KNOW PER STATE INFO ALREADY. IT'S IN THE MODEL. So maybe we preempt randomization even somehow? 
            # Not sure, since the allocation of hiring across firms should not be the same every time, no? Ah, but the randomization part we can indeed redo every time. Or make it dynamic from the start
            # But still, we could def do at least part of this stuff in __init__. And then here we just update the firm info based on their current state
            # So yes, here we should indeed loop over firms I'd say
                #But for t=2, where rho may be not on the grid. for now, assume it's just rho, and q gets put back on the grid somehow
                #rho[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.rho_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #Junior wage
                #w[t,f,0] = np.interp(rho[t-1,f],model.rho_grid,model.wage_jun[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #Senior wage, calculated by knowing the rho
                #w[t,f,1] = p.pref.inv_utility_1d(rho[t,f])

                #Get the hiring rate and quality
                #n_hire[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.n0_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #n1[t,f] = np.interp(rho[t-1,f],rho_grid,n1_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #q_v[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.q_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])  
                #Get the J2J rate based on last period's rho
                #pr_j2j[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.pe_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])

                #Also the sep rate to actually get the sense of firing:
                #sep_rate[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.sep_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])  
                #The ACTUAL n0 and n1 will be calculated BASED on these!
            
                
                #Or we could deduce it from the other variables + OJS:
                # n1[t,f] =  (n0[t-1,f] * (1-sep) +n1[t-1,f]) *(1 - p(rho[t,f]))
                #sep[t,f] = 1 - (n1[t,f]/(1-p) - n1[t-1,f])/n0[t-1,f] #Okay yeah, that's way better
                #Also note that these n0,n1,q should be points on the grid rather than their actual values! Not hard to change to that tho
                #But also... this seems to be slow af once we also interpolate over q, no?
                
                #Size_discr = #This takes the optimal size and discretizes it using randomization
                #CAN THE RANDOMIZATION BE AGGREGATED??? In that no firm, unless no hiring/sep/ojs, would have a discrete future size.
                #Ah but wait: if the total hiring sum is not discrete, we would genuinely have to discrete, even at the aggregate. Since so far we only allow entering firms to hire exactly 1 worker
                #So no choice there, in the aggregate it's only gonna be precise up to the discretization. 
                #I think I aggregate AFTER doing all the individual firms decisions
                #Quality = Quality.transition(State0)
                #State = np.array(Size,Quality,rho_star(State0)) #Thats the annoying thing, we gotta track everything
                #BUT WAIT. How... do they update rho_star? Because, once you get out of the grid, we don't have the values. Do they interpolate twice?
                #In a way yes. Given today's rho, whatever it is, they can immediately get w (new rho_star not needed), and they get rho_star by interpolating it's state into today's rho.
            #Aggregating the decisions, introducing entering firms. AH IMPORTANT POINT. The number of firms is not fixed! Not a problem though, even with num_f = num_f + num_entering - num_leaving at the end
            #This aggregation should include:
            #1. Closing firms (completely randomly) #Do this before?? To not have to deal with all da decisions??
                

            #Also, this ideally should NOT BE invariant! Even if I don't wanna make it completely endogenous, AT LEAST gotta make it cycle-variant
            #2. Summing up all the hiring of remaining firms
            #F_hiring = (n0[t,:] > 0) & (F_stay_open==1) #Gotta ensure the timing's correct
            #n0_total = np.sum(n0[t,F_hiring])
            #3. Adding entering firms to clear the market (so that sum of n0_star = U * p + all incumbents from last period * (1-s) * p)
            #Formula is N_newfirms = U*p + (1-F_close) * (1-s) * p - n0_total
            #Ix = (E0==0) #Unemployed workers
            #pr_unemp = model.Pr_u2e
            #N_newfirms = Ix.sum * pr_unemp + (1 - sep[t,F_stay_open]) * p - n0_total
            #F_new = np.ones(N_newfirms)
            #This clearing formula aint gonna be easy though...
            #5. Updating the firm set
            #F = (F_stay_open==1) #This doesn't work correctly. This should be a merging process!!!
            #Append new firms
            #F = np.append(F,F_stay_open) #Literally just made this shit up, 99% wrong
            #4. Deciding whether firms will ultimately hire the floor or ceil of n0_star for each firm
            #n0[t,:] = discretize_n(n0[t,:])
            #n1[t,:] = discretize_n(n1[t,:]) #May be better to merge these. Have a single function for both. Not much faster, def not at 2 steps, but is indeed a bit more fast

            #4.5 Shit, I need to do the same with n1! Moreover, depending on whether it's floor or ceil, I'd need to either let some guy go or keep them!!!
            #Even worse, I need to know whether the guy left due to separation of OJS (or both???)
            #Maybe I do smth about this more directly?
            #Like literally deduce n1 FROM the separations and from p?
            #Essentially, we know p and we know s. Both of these can realize or not. 
            #But what if p=0.6 and s=0.4. At least one of these has gotta realize, right?
            #Maybe I SHOULD do this at the worker level? 
            #Essentially, here I just sum up in order to gather how many new firms I need
            #But the actual allocation (what happens to whom, how many workers each firm ends up having) I do at the worker side?
            #Essentially, randomly allocate workers among the firms that are indeed hiring.
            #It's like selection without replacement: firm with n>0 is still in contention for that particular worker as long as it has not already hired too much.
            #Though tbh this process seems kinda slow... With the constant check and updating and allat
            #Maybe I do that for only n1 then? And keep the n0 strategy as above?
            #5. Randomly allocating workers across the hiring firms (should that be done in the worker section though mayb? guess not necessarily. can do this here, and just append the outcome in the loop)
            
            #Okay, so this shit DOES seem fairly complicated... but not impossible, just gotta tackle this step-by-step, starting with individual firm decisions.
            #Once I have that, I can proceed to aggregating.
            #Remove empty firms!!! Might not actually be good tbh. Forces more and more new firms in when unneeded
            #F_set = F_set[n0[t,F_set]+n1[t,F_set] > 0]

            
            
            #Z[E==1] = z[t,F[E==1]]
            #N0[E==1] = n0[t,F[E==1]]
            #N0[E==0] = 0 #MUST DO THIS!!! Because these guys don't reset on their own!
            #N1[E==1] = n1[t,F[E==1]]
            #N[E==1] = N0[E==1] + N1[E==1]
            #PROD[E==1] = prod[t,F[E==1]] #Can I add all of these later???
            #Also gotta initialize productivies for firms??? That's the job for tomorrow. PAST THAT, WE GOT IT!!! THE SIMULATION'S THERE, CAN BUG TEST AND MOVE ON TO MOMENTS

            # we shock the type of workers
            #for ix in range(p.num_x):
            #    Ix    = (X==ix)
            #    if INCLUDE_XCHG:
            #        X[Ix] = np.random.choice(p.num_x, Ix.sum(), p=model.X_trans_mat[:,ix])
            
            """