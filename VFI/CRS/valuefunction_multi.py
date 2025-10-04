"""
    We implement a power represenation for the value function,
    together with methods to initialize and update it
"""

import numpy as np
from scipy.optimize import minimize,nnls, differential_evolution
import matplotlib.pyplot as plt
import warnings
def curve_fit_search_and_grad(gamma, Xi, Yi, Xmax): #Andrei: Xi is worker's value, Yi is job's value
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/ 100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    Ri     = gamma[0] + gamma[1] * Xi_pow - Yi
    val    = np.power(Ri, 2).mean()
    #Andrei: complete function to minimize is 0.01*(gamma[0]+gamma[1]*(Wmax+exp(gamma[3])-Wi)^gamma[2]-Ji)^2
    # the optimizer can handle invalid returns for gradient
    # with np.errstate(divide='ignore'):
    #     with np.errstate(invalid='ignore'):
    g1     = 2 * Ri.mean() #Andrei: Derivative wrt gamma[0]
    g2     = 2 * ( Ri * Xi_pow ).mean() #Derivative wrt gamma[1]
    g3     = 2 * ( Ri * np.log( Xi_arg ) * Xi_pow * gamma[1] ).mean() #Derivative wrt gamma[2]
    g4     = 2 * ( Ri * gamma[1] * gamma[2] * np.exp(gamma[3]) * np.power( Xi_arg , gamma[2] - 1 ) ).mean() #Derivative wrt gamma[3]

    return val, np.array([g1,g2,g3,g4])

def curve_fit_search_terms(gamma, Xi, Yi, Xmax):
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    return Xi_pow,Yi

def curve_eval(gamma, Xi, Xmax):
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    return gamma[0] + gamma[1] * Xi_pow

class PowerFunctionGrid:
    """ Class that represents the value function using a power function representation.

        The different parameters are stored in gamma_all
        Y =   g0 + g1*(g4 + exp(g3) - X)^g2

        note, instead of using Vmax here, it might be better to use a more stable value, ie the
        actual max value of promisable utility to the worker which is u(infty)/r
        we might then be better of linearly fitting functions of the sort g1 * ( Vmax - X)^ g2 for
        a list of g2.
        Andrei: now need a different functional form! Shit
        Y = g0*n_sum^g5+sum_k g(1*k)*n_k*(g(4*k)+exp(g3*k)-X_k)^g(2*k)+n_0*gamma(6)... or smth like that. Very ugly tbh
        OR: if we loop over the size and have only 2 steps, it's essentially the same form!
    """

    def __init__(self,W1,J1,weight=0.01):
        self.num_z, _ , self.num_q = J1.shape #Calling num_z the shape of the 1st dimension, num_n the shape of the 2nd etc.
        #Need only the shapes that I'm looping over
        self.gamma_all = np.zeros( (self.num_z, 5, self.num_q) )
        self.rsqr  = np.zeros( (self.num_z, self.num_q))
        self.weight = weight

        # we fit for each (z,x). Andrei: So the only function inside is v? What should I do with size here?
        #Inputting size inside will def make it faster, but will it make it more precise?
        #For now, do the loop over size.
        p0 = [0, -1, -1, np.log(0.1)] #Andrei: starting guesses
        #Andrei: 0th dimension value function ignored in the curve_fit_search_and_grad since it's fixed
        #Thus, in the 2-step case, I don't even need to change anything else do I?
        for iz in range(self.num_z):
               #again num_n as the size grid si the same for the two steps
                     for iq in range(self.num_q):
                        p0[0] = J1[iz, 0, iq] #The first guess is set to J1 at the lowest promise
                        #res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                        #                options={'gtol': 1e-10, 'disp': False, 'maxiter': 10000},
                        #                args=(W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max())) 
                        bounds = [(p0[0] - 100, p0[0] + 100), (-100000, 0), (-100, 0), (np.log(0.001), np.log(100.0))]  # Example bounds for gamma[0], gamma[1], etc.
                        res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                                options={'gtol': 1e-8, 'disp': False, 'maxiter': 2000},
                                args=(W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max()))
                        #differential_evolution(
                               # lambda g: curve_fit_search_and_grad(g, W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max())[0],
                               # bounds=bounds,
                               # strategy='best1bin',
                               # maxiter=1000,
                               # tol=1e-7)
                        p0 = res2.x
                        #print("Gammas after init", iz, iq, res2.x)
                        self.gamma_all[iz, 0:4, iq] = res2.x
                        self.gamma_all[iz, 4, iq]   = W1[iz, :, iq].max() #Andrei: I'm confused. Is this not part of the [0:4]?
                        self.rsqr[iz, iq] = res2.fun / np.power(J1[iz, :, iq],2).mean()

    def eval_at_zxv(self,z, v, iq): #Andrei: for fixed z,x,v, give the exact prediction for J. this is after J has already been fitted
        return curve_eval(self.gamma_all[z,0:4, iq],v,self.gamma_all[z,4, iq])

    def get_vmax(self,z, iq):
        return self.gamma_all[z, 4, iq] + np.exp(self.gamma_all[z, 3, iq]) #Andrei: this is the Wbar from the appendix

    def eval_at_W1(self,W1): #Once J has been fitted, evaluate it at all the grid values
        J1_hat = np.zeros(W1.shape)
        for iz in range(self.num_z):
    
              for iq in range(self.num_q):
                J1_hat[iz, :, iq] = self.eval_at_zxv(iz,W1[iz, :, iq], iq)
        # make a for loop on x,z
        return(J1_hat)


    def mse(self,W1,J1):
        mse_val = 0

        for iz in range(self.num_z):
             for iq in range(self.num_q):
                val,_ = curve_fit_search_and_grad( self.gamma_all[iz, 0:4, iq], W1[iz, :, iq], J1[iz, :, iq], self.gamma_all[iz, 4, iq] )
                mse_val = mse_val + val

        return(mse_val)

    def update(self,W1,J1,lr):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        mean_update = 0


        for iz in range(self.num_z):
#again num_n as the size grid si the same for the two steps
              for iq in range(self.num_q):                
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4, iq], W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max() )
                self.gamma_all[iz, 0:4, iq] = self.gamma_all[iz,0:4, iq] - lr * grad
                self.gamma_all[iz, 4, iq]   = W1[iz, :, iq].max()
                mean_update = mean_update + np.abs(lr * grad).mean()

        return(mean_update/(self.num_z))

    def update_cst(self,W1,J1,lr):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0


        for iz in range(self.num_z):
#again num_n as the size grid si the same for the two steps
             for iq in range(self.num_q):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4, iq], W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max() )
                self.gamma_all[iz, 0:2, iq] = self.gamma_all[iz,0:2, iq] - lr * grad[0:2]
                self.gamma_all[iz, 4, iq]   = W1[iz, :, iq].max()
                tot_update_chg += np.abs(lr * grad[0:2]).mean()

        return(tot_update_chg/(self.num_z))

    def update_cst_ls(self,W1,J1):
        """
        Updates the parameters intercept and slope parameters of the representative
        function using lease square. Also stores the highest value to g4.

        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0
        pj_last = np.copy(self.gamma_all)
        for iz in range(self.num_z):

             for iq in range(self.num_q):
                Xi,Yi = curve_fit_search_terms( self.gamma_all[iz, 0:4, iq], W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max() )
                # W = np.exp(- self.weight * np.power(Yi,2))
                #print("Yi", Yi)
                W = 1.0 * (Yi >= -50) #Andrei: why -50 here? what's the point? IS this why the thing is nan? Because W=1 for every value so gamma[0],gamma[1] end up divided by 0? THE OPPOSITE! It's because Yi was below -50 everywhere
                W = W / W.sum()
                xbar       = ( Xi * W ).sum()
                ybar       = ( Yi * W ).sum()
                self.gamma_all[iz, 1, iq] = ( (Xi-xbar) * (Yi-ybar) * W ).sum() / (  (Xi-xbar) * (Xi-ybar) * W ).sum()
                        #print("W and gamma_all[1]", W, self.gamma_all[iz, 1])
                self.gamma_all[iz, 0, iq] = ( (Yi - self.gamma_all[iz, 1, iq]* Xi) * W ).sum()
                self.gamma_all[iz, 4, iq]   = W1[iz, :, iq].max()

        rsq = 1 - self.mse(W1,J1)/ np.power(J1,2).sum()
        chg = (np.power(pj_last - self.gamma_all,2).mean(axis=(0,1)) / np.power(pj_last,2).mean(axis=(0,1))).mean() #What's this axis thing?? Is this correct? Before it was 0,3, so it's as if 3 meant the rho dimension...
        return(chg,rsq)

    def update_local_min(self, W1, J1):
        for iz in range(self.num_z):
               #again num_n as the size grid si the same for the two steps
                     for iq in range(self.num_q):       
                        p0 = self.gamma_all[iz,:-1,iq]
                        res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                                        options={'gtol': 1e-5, 'disp': False, 'maxiter': 300},
                                        args=( W1[iz, :, iq], J1[iz, :, iq], W1[iz, :, iq].max())) 
                        self.gamma_all[iz, 0:4, iq] = res2.x
                        self.gamma_all[iz, 4, iq]   = W1[iz, :, iq].max()


class PowerFunctionGrid2:
    """ Class that represents the value function using a power function representation.

        The different parameters are stored in gamma_all
        Y =   1 - sum_k g0k*(gm - X)^(-g1k)

    """

    def __init__(self,W1,J1,vmax, gpow= np.arange(0.0,20.0,1) ,weight=0.01):
        self.num_z, _ = J1.shape
        self.num_g = len(gpow)
        self.gpow = np.array(gpow) # the sequence of power to use
        self.gamma_all = np.zeros( (self.num_z,self.num_g) )
        self.rsqr = np.zeros( (self.num_z))
        self.weight = weight
        self.vmax = vmax

        # we fit for each (z,x)

        for iz in range(self.num_z):

                self.gpow = np.exp( np.arange(-4,4))
                Yi = J1[iz, :]
                # compute the design matrix
                XX1 = - np.power(self.vmax - W1[iz, :][:,np.newaxis] , - self.gpow[np.newaxis,:])

                # constant plus linear
                XX2 = - np.power(W1[iz, :][:,np.newaxis] , np.arange(2)[np.newaxis,:])
                XX2[:,0] = - XX2[:,0]
                XX = np.concatenate([XX1,XX2],axis=1)

                # prepare weights
                W = np.sqrt(1.0 * (Yi >= -50))

                # fit parameters imposing non-negativity
                par,norm = nnls(XX * W[:,np.newaxis], Yi * W)
                rsq = np.power( W * np.matmul(XX,par) , 2).mean() / np.power( W * Yi, 2).mean()

                I = W>0
                plt.plot( W1[iz, I], J1[iz, I],'blue')
                #for k in range(1,len(self.gpow)):
                #    plt.plot( W1[iz, I],  XX[I,k] * par[k],"--")
                plt.plot( W1[iz, I], np.matmul(XX[I,:],par),'red')
                plt.show()
                p0 = 0

                res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                                options={'gtol': 1e-8, 'disp': False, 'maxiter': 2000},
                                args=(W1[iz, :], J1[iz, :], W1[iz, :].max()))
                self.gamma_all[iz, 0:4] = res2.x
                self.rsqr[iz] = 0

    def eval_at_zxv(self,z,v):
        return curve_eval(self.gamma_all[z,0:4],v,self.gamma_all[z,4])

    def eval_at_W1(self,W1):
        J1_hat = np.zeros(W1.shape)
        for iz in range(self.num_z):
                J1_hat[iz,:] = self.eval_at_zxv(iz,W1[iz,:])
        # make a for loop on x,z
        return(J1_hat)

    def mse(self,W1,J1):
        mse_val = 0

        for iz in range(self.num_z):
            val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], self.gamma_all[iz, 4] )
            mse_val = mse_val + val

        return(mse_val)

    def update(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        mean_update = 0

        for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], W1[iz, :].max() )
                self.gamma_all[iz, 0:4] = self.gamma_all[iz,0:4] - lr * grad
                self.gamma_all[iz, 4]   = W1[iz, :].max()
                mean_update = mean_update + np.abs(lr * grad).mean()

        return(mean_update/(self.num_z))

    def update_cst(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], W1[iz, :].max() )
                self.gamma_all[iz, 0:2] = self.gamma_all[iz,0:2] - lr * grad[0:2]
                self.gamma_all[iz, 4]   = W1[iz, :].max()
                tot_update_chg += np.abs(lr * grad[0:2]).mean()

        return(tot_update_chg/(self.num_z))

    def update_cst_ls(self,W1,J1):
        """
        Updates the parameters intercept and slope parameters of the representative
        function using lease square. Also stores the highest value to g4.

        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        pj_last = np.copy(self.gamma_all)
        for iz in range(self.num_z):
                Xi,Yi = curve_fit_search_terms( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], W1[iz, :].max() )
                # W = np.exp(- self.weight * np.power(Yi,2))
                W = 1.0 * (Yi >= -50)
                W = W / W.sum()
                xbar       = ( Xi * W ).sum()
                ybar       = ( Yi * W ).sum()
                self.gamma_all[iz, 1] = ( (Xi-xbar) * (Yi-ybar) * W ).sum() / (  (Xi-xbar) * (Xi-ybar) * W ).sum()
                self.gamma_all[iz, 0] = ( (Yi - self.gamma_all[iz, 1]* Xi) * W ).sum()
                self.gamma_all[iz, 4]   = W1[iz, :].max()

        rsq = 1 - self.mse(W1,J1)/ np.power(J1,2).sum()
        chg = (np.power(pj_last - self.gamma_all,2).mean(axis=(0,1)) / np.power(pj_last,2).mean(axis=(0,1))).mean()
        return(chg,rsq)

class PowerFunctionGridold:
    """ Class that represents the value function using a power function representation.

        The different parameters are stored in gamma_all
        Y =   g0 + g1*(g4 + exp(g3) - X)^g2

        note, instead of using Vmax here, it might be better to use a more stable value, ie the
        actual max value of promisable utility to the worker which is u(infty)/r
        we might then be better of linearly fitting functions of the sort g1 * ( Vmax - X)^ g2 for
        a list of g2.
    """

    def __init__(self,W1,J1,weight=0.01):
        self.num_z, _  = J1.shape
        self.gamma_all = np.zeros( (self.num_z,5) )
        self.rsqr  = np.zeros( (self.num_z))
        self.weight = weight

        # we fit for each (z,x)
        p0 = [0, -1, -1, np.log(0.1)] #Andrei: starting guesses

        for iz in range(self.num_z):
                p0[0] = J1[iz, 0]
                res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                                options={'gtol': 1e-8, 'disp': False, 'maxiter': 2000},
                                args=(W1[iz, :], J1[iz, :], W1[iz, :].max()))
                p0 = res2.x
                self.gamma_all[iz, 0:4] = res2.x
                self.gamma_all[iz, 4]   = W1[iz, :].max() #Andrei: I'm confused. Is this not part of the [0:4]?
                self.rsqr[iz] = res2.fun / np.power(J1[iz, :],2).mean()

    def eval_at_zxv(self,z,v): #Andrei: for fixed z,x,v, give the exact prediction for J. this is after J has already been fitted
        return curve_eval(self.gamma_all[z,0:4],v,self.gamma_all[z,4])

    def get_vmax(self,z):
        return self.gamma_all[z, 4] + np.exp(self.gamma_all[z, 3]) #Andrei: this is the Wbar from the appendix

    def eval_at_W1(self,W1): #Once J has been fitted, evaluate it at all the grid values
        J1_hat = np.zeros(W1.shape)
        for iz in range(self.num_z):
                J1_hat[iz,:] = self.eval_at_zxv(iz,W1[iz,:])
        # make a for loop on x,z
        return(J1_hat)

    def mse(self,W1,J1):
        mse_val = 0

        for iz in range(self.num_z):
                val,_ = curve_fit_search_and_grad( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], self.gamma_all[iz, 4] )
                mse_val = mse_val + val

        return(mse_val)

    def update(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        mean_update = 0


        for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], W1[iz, :].max() )
                self.gamma_all[iz, 0:4] = self.gamma_all[iz,0:4] - lr * grad
                self.gamma_all[iz, 4]   = W1[iz, :].max()
                mean_update = mean_update + np.abs(lr * grad).mean()

        return(mean_update/(self.num_z))

    def update_cst(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0


        for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], W1[iz, :].max() )
                self.gamma_all[iz, 0:2] = self.gamma_all[iz,0:2] - lr * grad[0:2]
                self.gamma_all[iz, 4]   = W1[iz, :].max()
                tot_update_chg += np.abs(lr * grad[0:2]).mean()

        return(tot_update_chg/(self.num_z))

    def update_cst_ls(self,W1,J1):
        """
        Updates the parameters intercept and slope parameters of the representative
        function using lease square. Also stores the highest value to g4.

        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        pj_last = np.copy(self.gamma_all)
        for iz in range(self.num_z):
                Xi,Yi = curve_fit_search_terms( self.gamma_all[iz,0:4], W1[iz, :], J1[iz, :], W1[iz, :].max() )
                # W = np.exp(- self.weight * np.power(Yi,2))
                W = 1.0 * (Yi >= -50)
                W = W / W.sum()
                xbar       = ( Xi * W ).sum()
                ybar       = ( Yi * W ).sum()
                self.gamma_all[iz, 1] = ( (Xi-xbar) * (Yi-ybar) * W ).sum() / (  (Xi-xbar) * (Xi-ybar) * W ).sum()
                self.gamma_all[iz, 0] = ( (Yi - self.gamma_all[iz, 1]* Xi) * W ).sum()
                self.gamma_all[iz, 4]   = W1[iz, :].max()

        rsq = 1 - self.mse(W1,J1)/ np.power(J1,2).sum()
        chg = (np.power(pj_last - self.gamma_all,2).mean(axis=(0,1)) / np.power(pj_last,2).mean(axis=(0,1))).mean()
        return(chg,rsq)