import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_results_for_p(p,all_results):
    # Create the key as a tuple
    #key = (p.num_z,p.num_v,p.num_n,p.n_bar,p.num_q,p.q_0,p.prod_q,p.hire_c,p.k_entry,p.k_f,p.prod_alpha,p.dt)
    key = (p.num_z,p.num_v,p.z_corr,p.prod_var_z,p.num_q,p.q_0,p.prod_q,p.s_job,p.alpha,p.kappa,p.dt,p.u_bf_m,p.min_wage)
    # Check if the key exists in the saved results
    if key in all_results:
        print(key)
        return all_results[key]
    else:
        print(f"No results found for p = {key}")
        return None
def cf_model_to_life(p,first_best, update_prod=False, pr_cache=False):
    """
    We simulate the response of several variables to a shock to z and x.
    We fixed the cross-section distribution of (X,Z) and set rho to rho_start
    We apply a permanent shock to either X or Z, and fix the employment relationship, as well as (X,Z)
    We then simulate forward the Rho, and the wage, and report several different variable of interest.
    """
    import pandas as pd
    from scipy.ndimage import map_coordinates

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
    
    nt = 20*4
    # we load the model
    #with open("model_GE.pkl", "rb") as file:
    #    all_results = pickle.load(file)
    #model = get_results_for_p(p,all_results)
    #if model is None:
    from CRS_HMQ_full import MultiworkerContract
    mwc_J=MultiworkerContract(p)
    model=mwc_J.J_sep(update_eq=1,s=0)


    # we simulate from the model to get a cross-section
    from simulate import Simulator
    sim = Simulator(model, p)
    sdata = sim.simulate().get_sdata()
    print("Solved and simulated")
    # we construct the different starting values
    tm = sdata['t'].max()
    d0 = sdata.query('e==1 & t==@tm')[['q','z','h','r']]
    z_grid = np.linspace(0,p.num_z-1,p.num_z)
    # we start at target rho
    #rho_target_interpolator = RegularGridInterpolator((z_grid, model.Q_grid), model.target_rho) 
    #points = np.column_stack((d0['z'],d0['q']))
    #Or, without the double interpolation...
    R0 = np.zeros_like(d0['z'])
    for iz in range(p.num_z):
        Ixz = np.where(d0['z']==iz)
        R0[Ixz] = np.interp(d0[Ixz]['z'],model.Q_grid,model.target_rho[iz,:])
    #np.interp(d0['q'],model.Q_grid,model.target_rho[d0['z'],:])#[ (d0['z'],d0['q']) ]
    #R0 = rho_target_interpolator (points)
    # starting with Z shocks
    def get_z_pos(pr):
        Z1_pos = np.minimum(sdata['z'].max(), d0['z'] + 1)
        Z1_pos = np.where(np.random.uniform(size=len(Z1_pos)) > pr, Z1_pos, d0['z']  )
        return(Z1_pos)

    def get_z_neg(pr):
        Z1_neg = np.maximum(0, d0['z'] - 1)
        Z1_neg = np.where(np.random.uniform(size=len(Z1_neg)) > pr, Z1_neg, d0['z']  )
        return(Z1_neg)

    # simulate a control group
    var_name = {'q':r'worker productivity $q$', 
                'w':r'log earnings $\log w$', 
                'W1':'worker promised value $V$', 
                'lceq':'worker cons. eq.', 
                'Pi':r'firm present value $J(x,z,V)$', 
                'y':r'log match output $\log f(x,z)$', 
                'pr_j2j':'J2J probability', 
                'pr_e2u':'E2U probability',
                'target_wage':r'log of target wage $\log w^*(x,z)$',
                'vs':'worker search decision $v_1$'}
    var_list = { k:'mean' for k in var_name.keys()  }

    def sim_agg(dd):
        # compute consumption equivalent for W1
        dd['lceq'] = model.pref.log_consumption_eq(dd['W1'])
        dd['lpeq'] = model.pref.log_profit_eq(dd['W1'])
        return(dd.groupby('t').agg(var_list))

    sdata0 = sim_agg(sim.simulate_force_ee(d0['q'],d0['z'],d0['h'],R0, nt,  update_z=False, pb=True))
    print("Control group simulated")
    # we run for a grid of probabilities
    if pr_cache:
        with open("res_cf_pr_fb{}.json".format(first_best)) as f:
            all = json.load(f)
    else:
        all = []
        vec = np.linspace(0,1,10)
        for i in range(len(vec)):
            res = {}
            res['pr'] = vec[i]
            pr = vec[i]
            res['z_pos'] = sim.simulate_force_ee(
                    d0['q'], get_z_pos(pr), d0['h'],R0, nt, 
                     update_z=False, pb=True)['y'].mean() 
            res['z_neg'] = sim.simulate_force_ee(
                    d0['q'], get_z_neg(pr), d0['h'],R0, nt, 
                     update_z=False, pb=True)['y'].mean() 
            all.append(res)

        # save to file!
        # with open("res_cf_pr_fb{}.json".format(first_best), 'w') as fp:
        #     json.dump(all, fp)
    
    df = pd.DataFrame(all)

    df = df.sort_values(['z_pos'])
    pr_z_pos = np.interp( sdata0['y'].mean() + 0.1, df['z_pos'] , df['pr'] )
    df = df.sort_values(['z_neg'])
    pr_z_neg = np.interp( sdata0['y'].mean() - 0.1, df['z_neg'] , df['pr'] )
    

 
    sdata0 = sim_agg(sim.simulate_force_ee(d0['q'],d0['z'],d0['h'],R0, nt, update_z=update_prod, pb=True))
    print("Ran for a grid of probabilities?")
    # finaly we simulate at the probabilities that we have chosen.
    sdata_z_pos = sim_agg(sim.simulate_force_ee(
        d0['q'],get_z_pos(pr_z_pos),d0['h'],R0, nt, 
        update_z=update_prod,pb=True))

    sdata_z_neg = sim_agg(sim.simulate_force_ee(
        d0['q'],get_z_neg(pr_z_neg),d0['h'],R0, nt, 
        update_z=update_prod,pb=True))

    # preparing the lead and lag plots
    pp0 = lambda v : np.concatenate([ np.zeros(5), v ])
    ppt = lambda v : np.concatenate([ [-4,-3,-2,-1,0], v ])
    
    to_plot = {'w','pr_j2j','pr_e2u','vs','effort','Pi','y','W1','target_wage'}
    to_plot = {k:v for k,v in var_name.items() if k in to_plot}
    print("Plotting")
    # Z shock response
    plt.clf()
    # plt.rcParams["figure.figsize"]=12,12
    plt.figure(figsize=(12, 12), dpi=80)
    for i,name in enumerate(to_plot.keys()):
        plt.subplot(3, 3, i+1)
        plt.plot( ppt (sdata0.index/4) , pp0(sdata_z_pos[name] - sdata0[name]) )
        plt.plot( ppt (sdata0.index/4) , pp0(sdata_z_neg[name] - sdata0[name]), linestyle='--')
        plt.axhline(0,linestyle=':',color="black")
        plt.xlabel(var_name[name])
        #plt.xlabel('years')
        plt.xticks(range(0,21,5))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-3,5))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if first_best:
        plt.savefig('../figures/figurew6-ir-zshock-fb.pdf', bbox_inches='tight')
    else:
        plt.savefig('../figures/figure4-ir-zshock.pdf', bbox_inches='tight')

new_baseline = {
   'q_0': 0.56602, 'prod_q':	0.50425,'u_bf_m':	2.34264/4,'s_job':	0.779616,'alpha':	0.79,'z_corr':	0.946006,'prod_var_z':	0.646317
}
from primitives import Parameters
p = Parameters(overwrite=new_baseline)
cf_model_to_life(p,first_best=False, update_prod=False, pr_cache=False)