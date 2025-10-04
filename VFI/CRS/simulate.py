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

def tercile_labels(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    # 1) try qcut with tie-handling
    try:
        t = pd.qcut(s, 3, labels=False, duplicates='drop')
        if pd.Series(t).nunique() == 3:
            return t.astype('Int64')
    except ValueError:
        pass
    # 2) fallback: percentile-rank → terciles (always 0/1/2)
    pr = s.rank(method='average', pct=True)           # in (0,1]
    t = np.floor(pr * 3).astype(int)                  # 0,1,2,3
    t[t == 3] = 2                                     # clamp edge case
    return t.astype('Int64')
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
            self.p.sim_ni, 
            self.p.sim_nt_burn + self.p.sim_nt, 
            self.p.sim_nt_burn,
            self.p.sim_nh,
            redraw_zhist=redraw_zhist,
            ignore=ignore))

    def simulate_val(self,ni=int(1e4),nt=100,burn=20,nl=100,redraw_zhist=True,ignore=[]):
        """ we simulate a panel using a solved model

            ni (1e4) : number of individuals
            nt (20)  : number of time period
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

        #Initialize interpolators
        rho_interpolator = np.empty(p.num_z, dtype=object)
        j2j_interpolator = np.empty_like(rho_interpolator)
        sep_interpolator = np.empty_like(rho_interpolator)
        q_interpolator = np.empty_like(rho_interpolator)
        ve_interpolator = np.empty_like(rho_interpolator)
        vf_interpolator = np.empty_like(rho_interpolator)
        rhoj2j_interpolator = np.empty_like(rho_interpolator)
        for iz in range(p.num_z):     
                    rho_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[iz, ...]) 
                    j2j_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[iz, ...]) 
                    sep_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[iz, ...]) 
                    q_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[iz, ...]) 
                    ve_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[iz, ...])
                    vf_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[iz, ...])
                    rhoj2j_interpolator[iz] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_j2j[iz, ...])

        # we store the current state into an array
        X  = np.zeros(ni,dtype=int)  # current value of the X shock
        Z  = np.zeros(ni,dtype=int)  # current value of the Z shock
        R  = np.zeros(ni)            # current value of rho
        E  = np.zeros(ni,dtype=int)  # employment status (0 for unemployed, 1 for employed)
        H  = np.zeros(ni,dtype=int)  # location in the firm shock history (so that workers share common histories)
        D  = np.zeros(ni,dtype=int)  # event
        W  = np.zeros(ni)            # log-wage
        Q  = np.zeros(ni)            # cohort quality
        P  = np.zeros(ni)            # firm profit
        S  = np.zeros(ni,dtype=int)  # number of periods in current spell
        pr = np.zeros(ni)            # probability, either u2e or e2u

        # we create a long sequence of firm innovation shocks where we store
        # a sequence of realized Z, we store realized Z_t+1 | Z_t for each
        # value of Z_t.
        if (redraw_zhist):
            Zhist = np.zeros((p.num_z,nl),dtype=int)
            for i in range(1,nl):
                # at each time we draw a uniform shock
                u = np.random.uniform(0,1,1)
                # for each value of Z we find the draw given that shock
                for z in range(p.num_z):
                   Zhist[z,i] = np.argmax( model.Z_trans_mat[ z , : ].cumsum() >= u  )
            self.Zhist = Zhist

        # we initialize worker types
        X = np.random.choice(range(p.num_x),ni)

        df_all = pd.DataFrame()
        # looping over time
        for t in range(nt):

            # save the state when starting the period
            E0 = np.copy(E)
            Z0 = np.copy(Z)

            # first we look at the unemployed of a given type X
            for ix in range(p.num_x): 
                Ix = (E0==0) & (X==ix)

                if Ix.sum() == 0: continue

                # get whether match a firm
                meet_u2e = np.random.binomial(1, model.Pr_u2e, Ix.sum())==1
                pr[Ix] = model.Pr_u2e

                # workers finding a job
                Ix_u2e     = bool_index_combine(Ix,meet_u2e)
                H[Ix_u2e]  = np.random.choice(nl, Ix_u2e.sum()) # draw a random location in the shock history
                E[Ix_u2e]  = 1                                  # make the worker employed
                R[Ix_u2e]  = model.rho_u2e                      # find the firm and the initial rho
                Z[Ix_u2e]  = p.z_0-1                            # starting z_0 for new matches
                Q[Ix_u2e]  = p.q_0                              # startomg q_0 for new matches
                D[Ix_u2e]  = Event.u2e
                W[Ix_u2e]  = np.interp(R[Ix_u2e], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                coords = np.column_stack((R[Ix_u2e], Q[Ix_u2e]))
                P[Ix_u2e]  = vf_interpolator[p.z_0-1] (coords)
                #np.interp(R[Ix_u2e], model.rho_grid, model.Vf_J[p.z_0-1,:,0])  # interpolate wage
                S[Ix_u2e]  = 1

                # workers not finding a job
                Ix_u2u     = bool_index_combine(Ix,~meet_u2e)
                E[Ix_u2u]  = 0           # make the worker unemployed
                W[Ix_u2u]  = 0           # no wage
                D[Ix_u2u]  = Event.uu
                H[Ix_u2u]  = -1
                S[Ix_u2u]  = S[Ix_u2u] + 1 # increase spell of unemployment
                R[Ix_u2u]  = 0
                S[Ix_u2u]  = 0

            # next we look at employed workers of quality q,prod Z
            for iz in range(p.num_z):
                    Ixz = (E0 == 1) & (X == ix) & (Z0 == iz)

                    if Ixz.sum() == 0: continue

                    # we check the probability to separate
                    coords = np.column_stack(( R[Ixz], Q[Ixz]))
                    pr_sep  = sep_interpolator[iz] (coords)
                    #np.interp( R[Ixz], model.rho_grid , model.qe_star[iz,:,iq])
                    sep     = INCLUDE_E2U * np.random.binomial(1, pr_sep, Ixz.sum() )==1
                    pr[Ixz] = pr_sep

                    # workers who quit
                    Ix_e2u      = bool_index_combine(Ixz,sep)
                    E[Ix_e2u]   = 0
                    D[Ix_e2u]   = Event.e2u
                    W[Ix_e2u]   = 0  # no wage
                    H[Ix_e2u]   = -1
                    S[Ix_e2u]   = 1
                    R[Ix_e2u]   = 0

                    # search decision for non-quiters
                    Ixz     = bool_index_combine(Ixz,~sep)
                    coords = np.column_stack(( R[Ixz], Q[Ixz]))
                    pr_meet = INCLUDE_J2J * j2j_interpolator[iz] (coords)
                    meet    = np.random.binomial(1, pr_meet, Ixz.sum() )==1

                    # workers with j2j
                    Ixz_j2j      = bool_index_combine(Ixz,meet)
                    coords = np.column_stack((R[Ixz_j2j], Q[Ixz_j2j]))
                    H[Ixz_j2j]   = np.random.choice(nl, Ixz_j2j.sum()) # draw a random location in the shock history

                    R[Ixz_j2j]   = rhoj2j_interpolator[iz] (coords)
                    #np.interp(R[Ixz_j2j], model.rho_grid, model.rho_j2j[iz,:,iq]) # find the rho that delivers the v2 applied to
                    if INCLUDE_ZCHG:
                        Z[Ixz_j2j]   = p.z_0-1                        # starting z_0 for new matches
                    else:
                        Z[Ixz_j2j]   = np.random.choice(range(p.num_z),Ixz_j2j.sum()) # this is for counterfactual simulations
                    D[Ixz_j2j]   = Event.j2j
                    W[Ixz_j2j]   = np.log(R[Ixz_j2j])
                    #np.interp(R[Ixz_j2j], model.rho_grid, np.log(model.w_grid)) # interpolate wage
                    P[Ixz_j2j]   = vf_interpolator[iz] (coords)
                    #np.interp(R[Ixz_j2j], model.rho_grid, model.Vf_J[iz, :, iq])  # interpolate wage
                    S[Ixz_j2j]   = 1
                    Q[Ixz_j2j]   = p.q_0

                    # workers with ee
                    Ixz_ee      = bool_index_combine(Ixz,~meet)
                    coords = np.column_stack((R[Ixz_ee], Q[Ixz_ee]))
                    R[Ixz_ee]   = rho_interpolator[iz] (coords)
                    #np.interp(R[Ixz_ee], model.rho_grid, model.rho_star[iz,:,iq]) # find the rho using law of motion
                    if INCLUDE_ZCHG:
                        Z[Ixz_ee]   = Zhist[ (Z[Ixz_ee] , H[Ixz_ee]) ] # extract the next Z from the pre-computed histories
                    H[Ixz_ee]   = (H[Ixz_ee] + 1) % nl             # increment the history by 1
                    D[Ixz_ee]   = Event.ee
                    W[Ixz_ee]   = np.log(R[Ixz_ee])
                    #np.interp(R[Ixz_ee], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                    P[Ixz_ee]   = vf_interpolator[iz] (coords)
                    #np.interp(R[Ixz_ee], model.rho_grid, model.Vf_J[iz, :, iq])  # interpolate firm Expected profit @fixme this done at past X not new X
                    S[Ixz_ee]   = S[Ixz_ee] + 1
                    Q[Ixz_ee]   = q_interpolator[iz] (coords)

            # we shock the type of the worker
            #for ix in range(p.num_x):
            #    Ix    = (X==ix)
            #    if INCLUDE_XCHG:
            #        X[Ix] = np.random.choice(p.num_x, Ix.sum(), p=model.X_trans_mat[:,ix])

            # append to data
            if (t>burn):
                df     = pd.DataFrame({ 'i':range(ni),'t':np.ones(ni) * t, 'e':E, 's':S, 'h':H, 'x':X , 'z':Z, 'r':R, 'd':D, 'w':W , 'q':Q ,'Pi':P, 'pr':pr} )
                df_all = pd.concat([df_all, df], axis =0)

        # append match output
        df_all['f'] = model.fun_prod[(df_all.z, df_all.x)]
        df_all.loc[df_all.e==0,'f'] = 0

        # construct a year variable called t4
        df_all['year'] = (df_all['t'] - (df_all['t'] % 4))//4

        # make earnings net of taxes (w is in logs here)
        df_all['w_gross'] = df_all['w']      
        df_all['w_net'] = np.log(self.p.tax_tau) + self.p.tax_lambda * df_all['w']  

        # apply expost tax transform
        df_all['w'] = np.log(self.p.tax_expost_tau) + self.p.tax_expost_lambda * df_all['w']  

        # add log wage measurement error
        # measurement error is outside the model, so we apply it after the taxes
        if INCLUDE_WERR:
            df_all['w'] = df_all['w'] + p.prod_err_w * np.random.normal(size=len(df_all['w']))

        # sort the data
        df_all = df_all.sort_values(['i', 't'])

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
        coords = np.array([[rho,q]])
        W[(F==1) & (S==1)] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[z, n0, n1, ...]) (coords) )
        W[(F==1) & (S>1)] = np.log(rho)
        Y[F==1] = np.log(prod[0])
        W1[(F==1) & (S>1)] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[z, n0, n1, ...,1]) (coords)
        #Firm info, added for every employed worker
        P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[z, n0, n1, ...]) (coords)

        all_df.append(pd.DataFrame({ 'i':range(ni+extra),'t':0, 'f':F, 
                'z':Z_array, 'w':W , 'Pi':P, 'D': D, 'S': S, 'n0': N0_array, 'n1': N1_array,
                'pr_e2u':pr_sep_array, 'pr_j2j':pr_j2j_array , 'y':Y, 'W1':W1, 'vs':Vs, 
                }))
        #if pb:
        #    rr = tqdm(range(1,nt))
        #else:
        rr = range(1,nt) 

        for t in rr:
            coords = np.array([[RHO[t-1],Q[t-1]]])
            #Update firm decisions first
            RHO[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)
            pr_j2j = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)
            pr_sep = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)
            pr_sep1 = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star1[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)            
            Q[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)
            Vs[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)
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
                N0[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[Z[t-1], N0[t-1],N1[t-1], ...]) (coords)
                #Discretize n0:
                n0_int = np.floor(N0[t])
                N0[t] = n0_int + np.random.binomial(1, N0[t]-n0_int)
                if update_z and z_dir is not None and (t==np.floor(nt/2).astype(int)):
                    #Force a hire that could potentially get fired
                    N0[t] = N0[t] + 1
                F[ni:ni+N0[t]] = 1 #these guys now employed
                S[ni:ni+N0[t]] = 1 #they're juniors
                D[ni:ni+N0[t]] = Event.u2e #they don't actually have to be hired from unemp btw           
            coords = np.array([[RHO[t],Q[t]]])
            w[t,0] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[Z[t], N0[t], N1[t], ...]) (coords) ) #Is this time inconsistent??? Given that prod is decided later?
            W[ni:ni+N0[t]] = w[t,0] #junior wage paid
            ni = ni + N0[t] #add the extra workers 

            prod[t] = model.fun_prod_onedim[Z[t]] * np.interp(Q[t],model.Q_grid,model.prod[Z[t],N0[t],N1[t],0,:])
            #Update the employed worker info
            I_sen = (F==1) & (S>1) #Seniors that didn't leave
            W[I_sen] = np.log(w[t,1])
            D[I_sen] = Event.ee
            W1[I_sen] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[Z[t], N0[t], N1[t], ...,1]) (coords)
            #Firm info, added for every employed worker
            P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[Z[t], N0[t], N1[t], ...]) (coords)
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
        W[(F==1) & (S==1)] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[z, n0, n1, ...]) ((rho,q)) )
        W[(F==1) & (S>1)] = np.log(rho)
        Y[F==1] = np.log(prod[0])
        W1[(F==1) & (S>1)] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[z, n0, n1, ...,1]) (((rho,q)))
        #Firm info, added for every employed worker
        P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[z, n0, n1, ...]) ((rho,q))

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
            RHO[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.rho_star[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))
            pr_j2j = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.pe_star[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))
            pr_sep = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))
            pr_sep1 = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.sep_star1[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))            
            Q[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.q_star[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))
            Vs[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.ve_star[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))
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
                N0[t] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.n0_star[Z[t-1], N0[t-1],N1[t-1], ...]) ((RHO[t-1],Q[t-1]))
                #Discretize n0:
                n0_int = np.floor(N0[t])
                N0[t] = n0_int + np.random.binomial(1, N0[t]-n0_int)
                if update_z and z_dir is not None and (t==np.floor(nt/2).astype(int)):
                    #Force a hire that could potentially get fired
                    N0[t] = N0[t] + 1
                F[ni:ni+N0[t]] = 1 #these guys now employed
                S[ni:ni+N0[t]] = 1 #they're juniors
                D[ni:ni+N0[t]] = Event.u2e #they don't actually have to be hired from unemp btw           
            w[t,0] = np.log(RegularGridInterpolator((model.rho_grid, model.Q_grid), model.w_jun[Z[t], N0[t], N1[t], ...]) ((RHO[t],Q[t])) ) #Is this time inconsistent??? Given that prod is decided later?
            W[ni:ni+N0[t]] = w[t,0] #junior wage paid
            ni = ni + N0[t] #add the extra workers 

            prod[t] = model.fun_prod_onedim[Z[t]] * np.interp(Q[t],model.Q_grid,model.prod[Z[t],N0[t],N1[t],0,:])
            #Update the employed worker info
            I_sen = (F==1) & (S>1) #Seniors that didn't leave
            W[I_sen] = np.log(w[t,1])
            D[I_sen] = Event.ee
            W1[I_sen] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_W[Z[t], N0[t], N1[t], ...,1]) ((RHO[t],Q[t]))
            #Firm info, added for every employed worker
            P[F==1] = RegularGridInterpolator((model.rho_grid, model.Q_grid), model.Vf_J[Z[t], N0[t], N1[t], ...]) ((RHO[t],Q[t]))
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
                      .query('h>=0')
                      .assign(c_e2u=lambda d: d.d_f1 == Event.e2u,
                              c_j2j=lambda d: d.d_f1 == Event.j2j)
                      .groupby(['h'])
                      .agg( {'f': 'sum', 'i': "count", 'c_e2u': 'sum', 'c_j2j': 'sum'}))
        hdata['f_year'] = hdata.f + np.roll(hdata.f, -1) + np.roll(hdata.f, -2) + np.roll(hdata.f, -3)
        hdata['c_year'] = hdata.i + np.roll(hdata.i, -1) + np.roll(hdata.i, -2) + np.roll(hdata.i, -3)
        hdata['c_e2u_year'] = hdata.c_e2u + np.roll(hdata.c_e2u, -1) + np.roll(hdata.c_e2u, -2) + np.roll(hdata.c_e2u, -3)
        hdata['c_j2j_year'] = hdata.c_j2j + np.roll(hdata.c_j2j, -1) + np.roll(hdata.c_j2j, -2) + np.roll(hdata.c_j2j, -3)
        hdata['ypw'] = np.log(hdata.f_year/hdata.c_year)
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
        sdata['w_exp'] = np.exp(sdata['w'])

        sdata_y = sdata.groupby(['i', 'year']).agg({'w_exp': 'sum', 'h': 'min', 's': 'min', 's2': 'max', 'e': 'min', 'es': 'sum'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's'], -1).pipe(create_year_lag, ['e', 'es'], 1)
        # make sure we stay in the same spell, and make sure it is employment
        sdata_y = sdata_y.query('h>=0').query('s+3==s2')
        sdata_y['w'] = np.log(sdata_y['w_exp'])

        # attach firm output, compute lags and growth
        sdata_y = (sdata_y.join(hdata.ypw, on="h")
                          .pipe(create_year_lag, ['ypw', 'w', 's', 'h'], 1)
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
        moms['total_output'] = sdata.query('h>0')['f'].sum()/len(sdata)
        moms['total_wage_gross'] = np.exp(sdata.query('h>0')['w_gross']).sum()/len(sdata)
        moms['total_wage_net'] = np.exp(sdata.query('h>0')['w_net']).sum()/len(sdata)
        moms['total_uben'] = self.p.u_bf_m * sdata.eval('h==0').sum()/len(sdata)

        # ------  transition rates   -------
        # compute unconditional transition probabilities
        #moms['pr_u2e'] = sdata.eval('d==@Event.u2e').sum() / sdata.eval('d==@Event.u2e | d==@Event.uu').sum()
        #moms['pr_j2j'] = sdata.eval('d==@Event.j2j').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        #moms['pr_e2u'] = sdata.eval('d==@Event.e2u').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()

        d = sdata['d']

        moms['pr_u2e'] = (d.eq(Event.u2e)).sum() / (d.isin([Event.u2e, Event.uu])).sum()
        moms['pr_j2j'] = (d.eq(Event.j2j)).sum() / (d.isin([Event.j2j, Event.ee, Event.e2u])).sum()
        moms['pr_e2u'] = (d.eq(Event.e2u)).sum() / (d.isin([Event.j2j, Event.ee, Event.e2u])).sum()

        # ------  earnings and value added moments at yearly frequency  -------
        # compute firm output and sizes at year level
        hdata = (sdata.set_index(['i', 't'])
                      .pipe(create_lag_i, 't', ['d'], -1)
                      .reset_index()
                      .query('h>=0')
                      .assign(c_e2u=lambda d: d.d_f1 == Event.e2u,
                              c_j2j=lambda d: d.d_f1 == Event.j2j)
                      .groupby(['h'])
                      .agg( {'f': 'sum', 'i': "count", 'c_e2u': 'sum', 'c_j2j': 'sum'}))
        hdata['f_year'] = hdata.f + np.roll(hdata.f, -1) + np.roll(hdata.f, -2) + np.roll(hdata.f, -3)
        hdata['c_year'] = hdata.i + np.roll(hdata.i, -1) + np.roll(hdata.i, -2) + np.roll(hdata.i, -3)
        hdata['c_e2u_year'] = hdata.c_e2u + np.roll(hdata.c_e2u, -1) + np.roll(hdata.c_e2u, -2) + np.roll(hdata.c_e2u, -3)
        hdata['c_j2j_year'] = hdata.c_j2j + np.roll(hdata.c_j2j, -1) + np.roll(hdata.c_j2j, -2) + np.roll(hdata.c_j2j, -3)
        hdata['ypw'] = np.log(hdata.f_year/hdata.c_year)
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

        # covaraince between change in log separation and log value added per worker
        moms['cov_dydsep'] = hdata_sep.cov()['dlypw']['dlsep'] #Adrei: also a valuable moment for me!!!

        # moments of the process of value added a the firm level
        cov = hdata_sep.pipe(create_lag, 'h', ['dlypwe'], 4)[['dlypwe', 'dlypwe_l4']].cov()
        moms['var_dy'] = cov['dlypwe']['dlypwe']
        moms['cov_dydy_l4'] = cov['dlypwe']['dlypwe_l4']

        # compute wages at the yearly level, for stayers
        sdata['s2'] = sdata['s']
        sdata['es'] = sdata['e']
        sdata['w_exp'] = np.exp(sdata['w'])
        sdata['e2u'] = sdata['d'].eq(Event.e2u).astype(int)

        sdata_y = sdata.groupby(['i', 'year']).agg({'w_exp': 'sum', 'h': 'min', 's': 'min', 's2': 'max', 'e': 'min', 'es': 'sum', 'e2u': 'max'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's'], -1).pipe(create_year_lag, ['e', 'es','e2u'], 1)
        # make sure we stay in the same spell, and make sure it is employment
        sdata_y = sdata_y.query('h>=0').query('s+3==s2')
        sdata_y['w'] = np.log(sdata_y['w_exp'])

        # attach firm output, compute lags and growth
        sdata_y = (sdata_y.join(hdata.ypw, on="h")
                          .pipe(create_year_lag, ['ypw', 'w', 's'], 1)
                          .assign(dw=lambda d: d.w - d.w_l1,
                                  dypw=lambda d: d.ypw - d.ypw_l1))

        # make sure that workers stays in same firm for 2 periods
        cov = sdata_y.query('s == s_l1 + 4')[['dw', 'dypw']].cov()
        moms['cov_dydw'] = cov['dypw']['dw']

        # Extract 2 U2E transitions within individual
        wid_2spells = (sdata_y.query('e_l1<1')
                            .assign(w1=lambda d: d.w, w2=lambda d: d.w, count=lambda d: d.h)
                            .groupby('i')
                            .agg({'count':'count','w1':'first','w2':'last'})
                            .query('count>1'))
        cov = wid_2spells[['w1','w2']].cov()
        moms['var_w_longac'] = cov['w1']['w2']

        cov = sdata_y.pipe(create_year_lag, ['w'], 4)[['w', 'w_l4']].cov()
        moms['var_w'] = sdata_y['w'].var()

        # lag wage growth auto-covariance
        cov = sdata_y.pipe(create_year_lag, ['dw'], 1).pipe(create_year_lag, ['dw'], 2)[['dw', 'dw_l1', 'dw_l2']].cov()
        moms['cov_dwdw_l4'] = cov['dw']['dw_l1']
        moms['cov_dwdw_l8'] = cov['dw']['dw_l2']
        moms['var_dw'] = cov['dw']['dw']

        # compute wage growth J2J and unconditionaly
        sdata_y.query('s == s_l1 + 4')['dw'].mean()
        moms['mean_dw'] = sdata_y['dw'].mean()
        sdata_y.pipe(create_year_lag, ['w'], 2).eval('w - w_l2').mean()

        # compute u2e, ee gap
        moms['w_u2e_ee_gap'] = sdata_y['w'].mean() - sdata_y.query('es_l1==0')['w'].mean()

        # compute wage growth given employer change
        moms['mean_dw_j2j_2'] = (sdata_y
                                    .pipe(create_year_lag, ['w', 'h', 'e'], 2)
                                    .query('e_l2 == 1').query('h_l2 + 8 != h')
                                    .assign(diff=lambda d: d.w - d.w_l2)['diff'].mean())

        #Andrei: extra moments of my own
        #1. Rate of new hires out of all the employed workers
        moms['pr_new_hire'] = sdata.eval('s==1 & e==1').sum() / sdata.eval('e==1').sum()
        #2. Tenure profile of wages at 7.5 years (how did Souchier do that?)
        # Step 1: Keep only employed individuals
        #employed = sdata_y.sort_values(['i', 'year']).copy()
        #employed['e2u_emp'] = (employed.groupby('i')['d'].shift(1) == Event.e2u)
        #employed = employed.query('e == 1')
        # Step 2: Sort by individual and time
        #employed = employed.sort_values(['i', 't'])
        #employed = employed.sort_values(['i', 'f', 't']).copy()
        #employed['lw'] = np.log(employed['w'])

        # previous time within (i,f)
        #employed['t_prev'] = employed.groupby(['i','f'])['t'].shift(1)

        # Step 3: Compute log wage growth (log difference):
        #sdata_y['w_growth_rate'] = sd.groupby(['i','f'])['lw'].diff()

        # optional: drop growth if last observation wasn’t the immediately prior period
        #sdata_y.loc[employed['t_prev'] != employed['t'] - 1, 'w_growth_rate'] = np.nan
        employed = sdata_y.copy().query(' s<= 32') #4 periods per year, 8 years
        moms['avg_w_growth_7'] = employed.groupby('s')['dw'].mean().sum() #But wait, this is the average for all tenure
        #3. Productivity moments
        #a) s.d. of firm productivity growth (take sd of dypw?) yep, this is exactly what we do in the data
        moms['sd_dypw'] = hdata_sep['dlypw'].std()
        #b) annual persistence of firm productivity (take autocovariance of ypw?)
        #moms['autocov_dypw'] = hdata.pipe(create_year_lag, ['ypw'], 1).pipe(create_year_lag, ['ypw'], 2)[['ypw', 'ypw_l1', 'ypw_l2']].cov()['ypw']['ypw_l1'] #This is kinda weird? This was suggested code, but I don't get it. How does it work? Detailed answer: it takes the lagged ypw, and then computes the covariance between the current and lagged ypw. So it is like a persistence measure, but not really an autocorrelation, because it is not normalized by variance
        #hdata['ypw_l1'] = hdata.groupby('f')['ypw'].shift(1)
        moms['autocov_ypw'] = hdata.pipe(create_lag, 'h',['ypw'], 4).cov()['ypw']['ypw_l4']
        #hdata_sep.cov()['ypw']['ypw_l4']
        #Step 4: HMQ moments
        #d) Fabrice suggestion: distribution of layoffs across firm productivity. What would the firm productivity be? I guess the firm output per worker, so ypw?
        # So in the data, I could take terciles of productivity, and then compute the share of layoffs in each tercile. And then same here. Then, if I do terciles, can get rid of the aggregate moment
        #So here, first get the terciles of ypw 
        #hdata['ypw_tercile'] = pd.qcut(hdata['ypw'], 3, labels=False)
        #Then compute the share of layoffs in each tercile
        #hdata['layoff_rate'] = hdata['c_e2u_year'] / hdata['c_year']
        #moms['layoffs_share_tercile'] = (hdata.groupby('ypw_tercile')['layoff_rate']
        #                                        .mean().rename('layoffs_share_tercile'))     
        #layoffs_by_tercile = hdata.groupby('ypw_tercile')['layoff_rate'].mean()
        #for k, v in layoffs_by_tercile.items():
        #    moms[f'layoffs_share_tercile_{int(k)}'] = float(v)   
        # --- after (robust) ---
        hdata['ypw_tercile'] = tercile_labels(hdata['ypw'])

        # share of layoffs per tercile → expand into scalars
        hdata['layoff_rate'] = hdata['c_e2u_year'] / hdata['c_year']
        layoffs_by_tercile = hdata.groupby('ypw_tercile', observed=True)['layoff_rate'].mean()

        # ensure keys 0,1,2 exist even if a tercile is empty (rare but possible)
        for k in range(3):
            moms[f'layoffs_share_tercile_{k}'] = float(layoffs_by_tercile.get(k, np.nan))
        
        del wid_2spells 
        #del sdata_y 
        self.sdata_y = sdata_y

        self.moments = moms
        return self

    def model_evaluation(self):
        """
        Computes the simulated NONTARGETED moments using the simulated data
        I might also want to produce the plots here, if I intend to replicate any
        :return:
        """
        sdata_y = self.sdata_y
        moms_untarg = {}

        #Now I want to run my regressions. First, let's run basic wages and layoffs across worker tenure in first-differences.
        #Get log wage growth
        #self.hdata['w_growth'] = np.log(self.hdata['w']) - np.log(self.hdata.pipe(create_lag_i, 't', ['w'], 1)['w_l1'])
        # Merge hdata['id_shock_sum'] onto employed using firm ID and time
        #employed = employed.merge(self.hdata.reset_index()[['f', 't', 'id_shock_sum', 'id_shock_diff']], on=['f', 't'], how='left')

        # Regress log wage growth on the interaction between tenure and firm productivity shock 
        #We get hdata['id_shock_sum'] as the cumulative log shock over the 3 years
        #employed['tenure'] = employed['s']
        model_wage_ten = feols('dw ~ i(s, dypw)', data=sdata_y)
        moms_untarg['wage_pass_ten'] = model_wage_ten.coef() #This is HUGE so far. more than 1 for S==2!!!  Even bigger if I use id_shock_diff??? Surprising ngl
        model_wage_ten = feols('dw ~ dypw + s * dypw', data=sdata_y) #the dypw coefficient is negative????
        moms_untarg['wage_pass_ten_simple'] = model_wage_ten.coef()  
        #It's also very weird after the first 2 tenures for some reason
        #Ah shit, should I actually be looking at the first 2 guys?? 
        # The very first tenure they've got the bonus wage
        # The second year they become seniors, which is a huge huge jump in wages, too
        # But past that it's immediately negative??? I guess, a positive shock now means new hirings?
        # And then you don't wanna raise wages much in anticipation of future seniors? That kinda sucks ngl. 
        # Is there any way to deal with this outside of more steps? Maybe I don't let firms internalize the combination???
        # But then the value function is quite literally incorrect! Okay gotta focus some more on Neural Nets ig
        model_e2u_ten = feols('e2u_l1 ~ C(s)', data=sdata_y)
        moms_untarg['sep_ten'] = model_e2u_ten.coef() #So far it's only juniors getting fired. Surprisingly, their firing rates are not very high, only 8%      
        model_e2u_pass_ten = feols('e2u_l1 ~ i(s, dypw)', data=sdata_y)
        moms_untarg['sep_pass_ten'] = model_e2u_pass_ten.coef()        
        model_e2u_pass_ten = feols('e2u_l1 ~ dypw + s * dypw', data=sdata_y)
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
            mom, mom_unt = self.simulate_val().computeMoments().model_evaluation().get_moments()
            moms = pd.concat([ moms, pd.DataFrame({ k:[v] for k,v in mom.items() })] , axis=0)
            #moms_unt = pd.concat([ moms_unt, pd.DataFrame({ k:[v] for k,v in mom_unt.items() })] , axis=0)
            # untargeted regression outputs (dict of Series) → flatten to a single row
            # mom_unt like: {'wage_pass_ten': Series(...), 'wage_pass_ten_simple': Series(...), ...}
            flat = pd.concat(mom_unt, names=['regression', 'coef'])  # MultiIndex Series
            moms_unt = pd.concat([moms_unt, flat.to_frame().T], axis=0)

            self.clean()
        self.log.info("done simulating")
        moms_mean = moms.mean().rename('value_model')
        moms_var = moms.var().rename('value_model_var')
        moms_unt_mean = moms_unt.mean().rename('regressions_mean')
        moms_unt_var  = moms_unt.var().rename('regressions_var')

        return(moms_mean, moms_var, moms_unt_mean, moms_unt_var)

#Okay, gotta debug
def debug(load):
    def get_results_for_p(p,all_results):
        # Create the key as a tuple
        #key = (p.num_z,p.num_v,p.num_n,p.n_bar,p.num_q,p.q_0,p.prod_q,p.hire_c,p.k_entry,p.k_f,p.prod_alpha,p.dt)
        key = (p.num_z,p.num_v,p.num_n,p.n_bar,p.num_q,p.q_0,p.prod_q,p.hire_c,p.prod_alpha,p.dt,p.u_bf_m)
        #    # Check if the key exists in the saved results
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
    if load:
        print("Loading model from:", model_path)
        with open(model_path, "rb") as file:
            all_results = pickle.load(file)
            model = get_results_for_p(p,all_results)
    else:
        from Multiworker_Contract_J import MultiworkerContract
        mwc_J=MultiworkerContract(p)
        model=mwc_J.J_sep(update_eq=0,s=40)

    sim = Simulator(model,p)
    sim.simulate_val().computeMoments().model_evaluation()

debug(load=True)