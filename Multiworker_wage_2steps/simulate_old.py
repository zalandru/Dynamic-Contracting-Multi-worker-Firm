""" 
    Simulates panel data from the model    
"""

import numpy as np
import logging
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
import gc

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
    has_fractional = fractional_parts > 1e-2  # Tolerance for floating point precision
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
    I_to_frac_vacancies = bool_index_combine(I_find_job,~np.isin(Id,Id_to_int_vacancies))
    if I_to_int_vacancies.sum()>0:
        fractional_parts = n_hire - integer_parts
        has_fractional = fractional_parts >= 1e-2  # Tolerance for floating point precision
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

    def simulate_val(self,ni=int(1e4),nf=500,nt=40,burn=20,nl=100,redraw_zhist=True,ignore=[]):
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
        Id = range(ni) #Worker id. Only important for determenistic allocation of workers to vacancies
        X  = np.zeros(ni,dtype=int)  # current value of the X shock
        Z  = np.zeros(ni,dtype=int)  # current value of the Z shock
        R  = np.zeros(ni)            # current value of rho
        E  = np.zeros(ni,dtype=int)  # employment status (0 for unemployed, 1 for employed)
        H  = np.zeros(ni,dtype=int)  # location in the firm shock history (so that workers share common histories)
        #Andrei: so should I create F as well?
        F = np.zeros(ni,dtype=int) # Firm identifier for each worker
        F_set  = np.arange(1,nf,dtype=int) #Set of all the unique firms

        D  = np.zeros(ni,dtype=int)  # event
        W  = np.zeros(ni)            # log-wage
        P  = np.zeros(ni)            # firm profit
        S  = np.zeros(ni,dtype=int)  # number of periods in current spell
        pr = np.zeros(ni)            # probability, either u2e or e2u

        #Prepare firm info
        extra = 1000 #1k extra firms allowed to enter at most before an error appears
        rho = np.zeros((nt,nf+1+extra)) #Extra 1 is added, so that rho[:,0] ends up unused
        w = np.zeros_like(rho)
        n_hire = np.zeros_like(rho)
        q = np.zeros_like(rho) #Should q be int or no?
        sep_rate = np.zeros_like(rho)
        pr_j2j = np.zeros_like(rho)
        w = np.zeros_like(rho)
        prod = np.zeros((nt,nf+1+extra),dtype=int)
        n0 = np.zeros_like(prod)
        n1 = np.zeros_like(prod)
        
        # we create a long sequence of firm innovation shocks where we store
        # a sequence of realized Z, we store realized Z_t+1 | Z_t for each
        # value of Z_t.
        if (redraw_zhist): #Andrei: this doesn't work for me as the set of firms is not preset. 
            #Moreover, since new hire firms always start at the same prod level (p.z_0-1), can't initialize extra firms in advance
            Zhist = np.zeros((p.num_z,nf+1),dtype=int)
            for i in range(1,nf+1):
                # at each time we draw a uniform shock
                u = np.random.uniform(0,1,1)
                # for each value of Z we find the draw given that shock
                for z in range(p.num_z):
                   Zhist[z,i] = np.argmax( model.Z_trans_mat[ z ].cumsum() >= u  )
            self.Zhist = Zhist
        #Andrei: good time as any to ask the big question
        # How do I assign workers to firms??? Guess it doesn't matter since workers are homogenous, no?
        # So workers are randomly assigned to the firms hiring today.
        # This is where GE comes in: pr_u2e is inconsistent with n0_star rn
        #But how WOULD it work in full GE? we need that U*pr_u2e + (1-U)(1-s)*pr_e2e = \sum n0_star
        #So Schaal deals with this by having entrants hire as well. Then N_entrants = N_hires - N_hirings
        #What if there's too many hiring though?? Is that possible????
        #Also interesting note: in Schaal, although there is free-entry of firms, the nu,ber of entering firms DOESN'T depend on this condition
        #Instead, the firm free-entry identifies the hiring cost, and then the number of entrants is set to fix the GE inflows/outflows of labor

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
        for t in range(nt):

            # save the state when starting the period
            E0 = np.copy(E)
            Z0 = np.copy(Z)
            F0 = np.copy(F) #Note that this F is the set of firms, not the assignment of workers to firms
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

            #Check destruction:
            self.close_pr = 0.05 #Percent of closing firms. For now, without aggregate movements, keep it constant
            close = np.random.binomial(1, self.close_pr, F_set.shape[0])==1  
            #F_set     = F_set * (~close) #All the closed firms have index 0, the rest keep their indeces
            F_set = np.where(close, -2, F_set) #All the closed firms are now called -2, to avoid confusion with unemployed workers who have F=-1
            for f in F_set[F_set != -2]: 
                #But for t=2, where rho may be not on the grid. for now, assume it's just rho, and q gets put back on the grid somehow
                rho[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.rho_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #Junior wage
                w[t,f,0] = np.interp(rho[t-1,f],model.rho_grid,model.wage_jun[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #Senior wage, calculated by knowing the rho
                w[t,f,1] = p.pref.inv_utility_1d(rho[t,f])

                #Get the hiring rate and quality
                n_hire[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.n0_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                #n1[t,f] = np.interp(rho[t-1,f],rho_grid,n1_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])
                q[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.q_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])  

                #Get the J2J rate based on last period's rho
                pr_j2j[t,f] = 1 - np.interp(rho[t-1,f],model.rho_grid,model.pe_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])

                #Also the sep rate to actually get the sense of firing:
                sep_rate[t,f] = np.interp(rho[t-1,f],model.rho_grid,model.sep_star[prod[t-1,f],n0[t-1,f],n1[t-1,f],:,q[t-1,f]])  
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
            
            #1. Make employed workers of closed firms unemployed
            I_close = (E0==1) & ~np.isin(F0,F_set)

            E[I_close]  = 0           # make the worker unemployed
            D[I_close]  = Event.e2u
            W[I_close]  = 0           # no wage
            F[I_close]  = -1
            S[I_close]  = 1
            R[I_close]  = 0

            #2. Fire employed workers
            I_remain = (E0==1) & np.isin(F0,F_set)
            F_remain = F0 * I_remain #This one isn't actually necessary, right? Correct, when we do bool_index_combine, the workers with technically real firms still get left out
            # we check the for separation
            rng = np.random.default_rng()
            sep = INCLUDE_E2U *  rng.binomial(1, sep_rate[t,F_remain]) == 1  #This should work, if slightly inefficient. 
            #All the workers outside of I_remain are assigned F_remain=0, so their sep_rate[t,0]=0. For the rest, they're assigned their firm's actual sep_rate

            # workers who got fired
            I_e2u      = bool_index_combine(I_remain,sep)
            E[I_e2u]   = 0
            D[I_e2u]   = Event.e2u
            W[I_e2u]   = 0  # no wage
            F[I_e2u]  = -1 # no firm
            S[I_e2u]   = 1
            R[I_e2u]   = 0

            #3. Unemp workers
            # a) Search shock for unemp workers
            Iu = (E0==0)
            # get whether match a firm
            meet_u2e = np.random.binomial(1, model.Pr_u2e, Iu.sum())==1
            pr = model.Pr_u2e        
            #Deal with the unlucky unemployed
            Iu_uu = bool_index_combine(Iu,~meet_u2e)
            E[Iu_uu]   = 0
            D[Iu_uu]   = Event.uu
            W[Iu_uu]   = 0  # no wage
            F[Iu_uu]  = -1 # no firm
            S[Iu_uu]   = S[Iu_uu] + 1
            R[Iu_uu]   = 0
            #Unemployed that did find a job
            Iu_u2e     = bool_index_combine(Iu,meet_u2e)
            #4. Emp workers            
            Ie = bool_index_combine(I_remain,~sep) #If their firm closed, I_remain=0. If they got separated, ~sep=0.
            F_e = F0 * Ie #All the workers outside of this set are assigned "firm' 0, where all the policies are zero.
            # search decision for those not fired    
            rng = np.random.default_rng()           
            meet = INCLUDE_J2J *  rng.binomial(1, pr_j2j[t,F_e]) == 1  

            # Employed workers that didn't transition
            Ie_stay =bool_index_combine(Ie,~meet)
            E[Ie_stay]  = 1           # make the worker unemployed
            D[Ie_stay]  = Event.ee
            W[Ie_stay]  = w[t,F[Ie_stay],1]           # senior wage
            F[Ie_stay]  = F0[Ie_stay]
            S[Ie_stay]  = S[Ie_stay] + 1
            R[Ie_stay]  = rho[t,F[Ie_stay]]                
            
            #Emp workers that did find a job
            Ie_e2e = bool_index_combine(Ie,meet)
            #5. Total search and allocation of workers to firms
            I_find_job = Iu_u2e + Ie_e2e #Note that I'm simply summing up boolean values here
            #Sum up expected hiring across all firms
            hire_total = n_hire[t,:].sum()
            #If not enough expected vacancies, update:
            if np.ceil(hire_total) < I_find_job.sum():
                #Note that, if there're 4 workers searching and 3.5 vacancies, no firms are added.
                #Add new firms
                n_new = np.floor(I_find_job.sum() - hire_total).astype(int)
                #And, if we have 4w/2.5v, we add 1 firm
                new_firms = np.arange(nf + 1, nf + n_new + 1)
                F_set = np.concatenate((F_set, new_firms)) #or is this too soon? let these firms find a job first mayb?
                #Add these firms into aggregate hiring
                n_hire_new = np.zeros((nt,n_new))
                n_hire_new[t,:] = 1
                n_hire = np.concatenate((n_hire, n_hire_new))
                hire_total = n_hire[t,:].sum()
                assert (np.ceil(hire_total) >= I_find_job.sum()), "Not enough expected hirings"
            #First, a basic version: allocating workers across vacancies completely randomly
            F[I_find_job] = allocate_workers_to_vac_rand(n_hire,F_set,I_find_job.sum())
            #Now try satisfying all the prob 1 vacancies first. DO THIS AND CAN GO HOME!!!
            F[I_find_job] = allocate_workers_to_vac_determ(n_hire,F_set,F,Id,I_find_job)

            #Tomorrow: now that searching workers are allocated, give the rest of their values. Also update the firm set F_set? Ah, no, shouldn't do that, would mess up the indices
            #Ah, it's only non-moving workers left! Update them. THEN update the actual firm sizes!!!
            E[I_find_job]   = 1
            #The rest is different across different workers??? Or not because all workers start at the same v_0 anyway?
            W[I_find_job]   = w[t,F[I_e2u],0]  # jun wage. potentially could add the wage bonus here as well
            S[I_find_job]   = 1
            R[I_find_job]   = model.v_0 #v_0? Not sure actually. Main question for tomorrow!

            D[Ie_e2e]   = Event.j2j
            D[Iu_u2e]   = Event.u2e  

            #6. Wrapping the period up: updating firm info 
            for f in F_set[F_set != -2]: 
                #Sum up the number of jun and sen workers
                n0[t,f] = len(F[(F==f) & (S==1)])
                n1[t,f] = len(F[(F==f) & (S > 1)])
            #Also gotta initialize productivies for firms??? That's the job for tomorrow. PAST THAT, WE GOT IT!!! THE SIMULATION'S THERE, CAN BUG TEST AND MOVE ON TO MOMENTS
            
            #########################OLDDDDD################################################# 
            # first we look at the unemployed of a given type X
            for ix in range(p.num_x):
                Ix = (E0==0) & (X==ix)

                if Ix.sum() == 0: continue

                # get whether match a firm
                meet_u2e = np.random.binomial(1, model.Pr_u2e[ix], Ix.sum())==1
                pr[Ix] = model.Pr_u2e[ix]

                # workers finding a job
                Ix_u2e     = bool_index_combine(Ix,meet_u2e)
                H[Ix_u2e]  = np.random.choice(nl, Ix_u2e.sum()) # draw a random location in the shock history
                #Andrei: can do np.random.choice(nf_inc, Ix_u2e.sum()+Ix_e2e.sum(),replace=0,p=n0) #Ah wait no, that don't work. 
                # Here it's as if every firm would hire just 1 worker. So gotta allocate to open vacations instead!
                #Essentionally, if n0[t,f]=1.2, we open one full location and one location with p=0.2!
                #Thing is though, adding new firms until sum of n0=sum of u2e+e2e is not enough, exactly because of these p=0.2 vacancies!
                #So still, add firms until sum of n0+new_firms=u2e+e2e.
                #Meaning that order is: 
                #1. Incumbent firm decisions (before discretizing) (just how necessary is this btw?)  (this will tell us also q, which will be necessary for production!)
                # For the next two points... do I loop over firms' size? To determine J2J probability??? Or just over firms??
                # I think... I loop over firms??? Or I can do it in advance maybe??? 
                # OHHHHH FUCKKKK! I NEED TO EXPORT THE WAGES FOR THE NEW HIRES! RHO DON WORK THERE. Not a big deal tho
                #2. Destroy firms + separate other incumbents (all these guys just end up unemployed. BUT WAIT. GOTTA CHANGE THEIR STATUSES AFTER THE UNEMP SEARCH? ah no, we're using E0 to define U workers)
                #3. Search for unemployed+employed. Ix_u2e.sum()+Ix_e2e.sum() = n0[t,:].sum() (of remaining firms) + n_new_firms (this will give us a # of new firms)
                    #Create vacancies with weights corresponding in n0: v=np.arange(n0[t,:].ceil.sum()+n_new_firms). (corr firm) f_v = ,(prob of filling) p_v=
                    #Allocate both unemp and emp workers using np.random.choice(nf_inc, Ix_u2e.sum()+Ix_e2e.sum(),replace=0,p=n0)
                #4. The remaining unemp guys get updated. The remaining emp guys update their tenure and get updated
                #Next time: DO THIS!!!
                #a) Destroy the firms. Then fire employed workers (each emp worker (E0==1) has a chance of being fired based on their firm (sep[t,f])). Update these workers right away
                #b) Take unemp workers (E0==0) and not fired workers (E0==0 & ~sep).
                E[Ix_u2e]  = 1                                  # make the worker employed
                R[Ix_u2e]  = model.rho_u2e[ix]                  # find the firm and the initial rho
                Z[Ix_u2e]  = p.z_0-1                            # starting z_0 for new matches
                D[Ix_u2e]  = Event.u2e
                W[Ix_u2e]  = np.interp(R[Ix_u2e], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                P[Ix_u2e]  = np.interp(R[Ix_u2e], model.rho_grid, model.Vf_J[p.z_0-1,:,ix])  # interpolate wage
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

            # next we look at employed workers of type X,Z
            for ix in range(p.num_x):
                for iz in range(p.num_z):
                    Ixz = (E0 == 1) & (X == ix) & (Z0 == iz)

                    if Ixz.sum() == 0: continue

                    # we check the probability to separate
                    pr_sep  = np.interp( R[Ixz], model.rho_grid , model.qe_star[iz,:,ix])
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
                    pr_meet = INCLUDE_J2J * np.interp( R[Ixz], model.rho_grid , model.pe_star[iz,:,ix]) #Andrei: What is this R[Ixz]? What is its value??? That's their current Rho! For me, I'll need to also account for their firms' size and stuff
                    meet    = np.random.binomial(1, pr_meet, Ixz.sum() )==1

                    # workers with j2j
                    Ixz_j2j      = bool_index_combine(Ixz,meet)
                    H[Ixz_j2j]   = np.random.choice(nl, Ixz_j2j.sum()) # draw a random location in the shock history
                    R[Ixz_j2j]   = np.interp(R[Ixz_j2j], model.rho_grid, model.rho_j2j[iz,:,ix]) # find the rho that delivers the v2 applied to
                    #Andrei: what is this rho_j2j? Guess it's the optimal rho that the worker finds in the market that he searches at?
                    #Andrei: this R is still confusing me, because at period 1 this suggests they're starting with rho=0?
                    #Andrei: my R for j2j will always be the same, v_0. HOWEVER, would be good to track the sign-on wage mayb
                    if INCLUDE_ZCHG:
                        Z[Ixz_j2j]   = p.z_0-1                        # starting z_0 for new matches
                    else:
                        Z[Ixz_j2j]   = np.random.choice(range(p.num_z),Ixz_j2j.sum()) # this is for counterfactual simulations
                    D[Ixz_j2j]   = Event.j2j
                    W[Ixz_j2j]   = np.interp(R[Ixz_j2j], model.rho_grid, np.log(model.w_grid)) # interpolate wage
                    P[Ixz_j2j]   = np.interp(R[Ixz_j2j], model.rho_grid, model.Vf_J[iz, :, ix])  # interpolate wage #Andrei: this is interpolate Job value, no?
                    S[Ixz_j2j]   = 1

                    # workers with ee #Andrei: wtf is ee? Job stayers?
                    Ixz_ee      = bool_index_combine(Ixz,~meet) #Andrei: yes, this is workers who didn't quit and didn't find a job elsewhere
                    R[Ixz_ee]   = np.interp(R[Ixz_ee], model.rho_grid, model.rho_star[iz,:,ix]) # find the rho using law of motion
                    #Andrei: so, fixing today's rho (which we now based on the history), what is tomorrow's rho? Really nice
                    #Andrei: for me, I'll also need to track the firm (or its state) for each particular worker, since rho_star depends on more things than just iz,R,ix
                    if INCLUDE_ZCHG:
                        Z[Ixz_ee]   = Zhist[ (Z[Ixz_ee] , H[Ixz_ee]) ] # extract the next Z from the pre-computed histories
                    H[Ixz_ee]   = (H[Ixz_ee] + 1) % nl             # increment the history by 1
                    D[Ixz_ee]   = Event.ee
                    W[Ixz_ee]   = np.interp(R[Ixz_ee], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                    P[Ixz_ee]   = np.interp(R[Ixz_ee], model.rho_grid, model.Vf_J[iz, :, ix])  # interpolate firm Expected profit @fixme this done at past X not new X
                    S[Ixz_ee]   = S[Ixz_ee] + 1

            # we shock the type of the worker
            #for ix in range(p.num_x):
            #    Ix    = (X==ix)
            #    if INCLUDE_XCHG:
            #        X[Ix] = np.random.choice(p.num_x, Ix.sum(), p=model.X_trans_mat[:,ix])

            # append to data
            if (t>burn):
                df     = pd.DataFrame({ 'i':range(ni),'t':np.ones(ni) * t, 'e':E, 's':S, 'h':H, 'x':X , 'z':Z, 'r':R, 'd':D, 'w':W , 'Pi':P, 'pr':pr} )
                df_all = pd.concat([df_all, df], axis =0) #Andrei: merge current dataset with the previous ones based on the i axis. So then each worker just has a single row assigned to them?

        # append match output
        df_all['f'] = model.fun_prod[(df_all.z, df_all.x)] #for me this will be self.prod? Exactly! Exact same thing except at these values (meaning, for me, z,n0,n1,any_v,q)
        #Can I do other stuff like this, too? Some easy stuff to append?
        df_all.loc[df_all.e==0,'f'] = 0 #unemployed get output of 0?

        # construct a year variable called t4 #Andrei: this is because the data generated is of quarterly frequency
        #df_all['year'] = (df_all['t'] - (df_all['t'] % 4))//4

        # make earnings net of taxes (w is in logs here)
        #df_all['w_gross'] = df_all['w']      
        #df_all['w_net'] = np.log(self.p.tax_tau) + self.p.tax_lambda * df_all['w']  

        # apply expost tax transform
        #df_all['w'] = np.log(self.p.tax_expost_tau) + self.p.tax_expost_lambda * df_all['w']  

        # add log wage measurement error
        # measurement error is outside the model, so we apply it after the taxes
        #if INCLUDE_WERR:
        #    df_all['w'] = df_all['w'] + p.prod_err_w * np.random.normal(size=len(df_all['w']))

        # sort the data
        df_all = df_all.sort_values(['i', 't'])

        self.sdata = df_all
        return(self)

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
        moms['pr_u2e'] = sdata.eval('d==@Event.u2e').sum() / sdata.eval('d==@Event.u2e | d==@Event.uu').sum()
        moms['pr_j2j'] = sdata.eval('d==@Event.j2j').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        moms['pr_e2u'] = sdata.eval('d==@Event.e2u').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()

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
        moms['cov_dydsep'] = hdata_sep.cov()['dlypw']['dlsep']

        # moments of the process of value added a the firm level
        cov = hdata_sep.pipe(create_lag, 'h', ['dlypwe'], 4)[['dlypwe', 'dlypwe_l4']].cov()
        moms['var_dy'] = cov['dlypwe']['dlypwe']
        moms['cov_dydy_l4'] = cov['dlypwe']['dlypwe_l4']

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

        del wid_2spells 
        del sdata_y 

        self.moments = moms
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
        return self.moments

    def simulate_moments_rep(self, nrep):
        """
        simulates moments from the model, running it multiple times
        :param nrep: number of replications
        :return:
        """

        moms = pd.DataFrame()
        self.log.info("Simulating {} reps".format(nrep))
        for i in range(nrep):
            self.log.debug("Simulating rep {}/{}".format(i+1, nrep))
            mom = self.simulate().computeMoments().get_moments()
            moms = pd.concat([ moms, pd.DataFrame({ k:[v] for k,v in mom.items() })] , axis=0)
            self.clean()
        self.log.info("done simulating")
        moms_mean = moms.mean().rename('value_model')
        moms_var = moms.var().rename('value_model_var')

        return(moms_mean, moms_var)
