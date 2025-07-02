#Neural Network solver for the CRS model, with both policy and value networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap
from tqdm import tqdm
import numpy as np
from scipy.stats import lognorm as lnorm
from ContinuousContract import ContinuousContract
from primitives import Parameters
from primitives_CRS import Parameters as p_crs
import opt_einsum as oe
from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search_tensor import JobSearchArray
import matplotlib.pyplot as plt
from time import time
import math
import copy
from ranger21 import Ranger21 as RangerOptimizer
from plotter import LossPlotter
p = Parameters()
tensor = torch.tensor
# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
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
ax = np.newaxis
#Tensor type
type = torch.float32
class StateBoundsProcessor:
    def __init__(self, lower_bounds, upper_bounds):
        """
        Initialize with lower and upper bounds for each state dimension
        
        Args:
            lower_bounds: List or tensor of lower bounds [x_1, x_2, ..., x_20]
            upper_bounds: List or tensor of upper bounds [y_1, y_2, ..., y_20]
        """
        self.lower_bounds = tensor(lower_bounds, dtype=type)
        self.upper_bounds = tensor(upper_bounds, dtype=type)
        self.range = self.upper_bounds - self.lower_bounds
        
    def normalize(self, states):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[ax, :]) / self.range[ax, :]
        #Example: lower-upper is [0,1]. So normalize(0.5) = 2 * (0.5 - 0) /1 -1 = 0. Ok correct
        #Another example. lower-upper is [0,30]. Sn normalize 15= 2 * 15 / 30 -1 = 0 Ok good.
        # And normalize (20) = 40/30 - 1 = 1/3 yup
        # Now denormalize(1/3) = 0.5 ( 1/3 +1 ) * 30 + 0 = 2/3*30 = 20

        #Normalizing to (0,1) now. Checking: 15 in (0,30) is 15/30=0.5. 15 in (10,20) is 5/10=0.5       
    def normalize_rho(self, states):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[ax, K_n:]) / self.range[ax, K_n:]    #Note: this will need to be adjusted at multiple steps    
    def denormalize_rho(self, normalized_states):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range[ax,K_n:] + self.lower_bounds[ax,K_n:] #Check: 0.5 * 10 + 10= 15. 
    def denormalize(self, normalized_states):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range[ax, :] + self.lower_bounds[ax, :] #Check: 0.5 * 10 + 10= 15.
    def normalize_dim(self, states,dim):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[dim]) / self.range[dim]    #Note: this will need to be adjusted at multiple steps      
    def denormalize_dim(self, normalized_states,dim):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range[dim] + self.lower_bounds[dim] #Check: 0.5 * 10 + 10= 15. 
    def denormalize_size(self, normalized_states):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range[ax,:K_n] + self.lower_bounds[ax,:K_n] #Check: 0.5 * 10 + 10= 15.

#Neural Nets. Note: alternatively, I could code give them the same trunk. Either way, this is called an Actor-Critic architecture.
class ValueFunctionNN(nn.Module):
    """Neural network to approximate the value function"""
    def __init__(self, state_dim, num_y, hidden_dims=[40, 30, 20, 10]):
        super(ValueFunctionNN, self).__init__()
        
        # Build layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Consider adding layer normalization for stability
            #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())  # LeakyReLU activation function: allows slightly negative values
            input_dim = hidden_dim
        
        # Final layer: one output per discrete state y'
        layers.append(nn.Linear(input_dim, num_y)) #was input_dim instead of 16
        self.model = nn.Sequential(*layers)
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
    def forward(self, x):
        return self.model(x)
class ReLUPlusEps(nn.Module):
    def __init__(self, eps=1e-2):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.relu(x) + self.eps

class PolicyNN(nn.Module):
    """Neural network to approximate a multi-dimensional policy:
       - wages: multiple values per productivity state y across a predefined set of K_v
       - hiring decision: probability per productivity state y
    """
    def __init__(self, state_dim, num_y, K_v, hidden_dims=[40, 30, 20, 10], cc=None):
        super(PolicyNN, self).__init__()
        self.K_v = K_v #Number of Value-related policies
        self.num_y = num_y

        # shared trunk
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # wage head: output num_y * num_Kv wage values, then reshape
        self.wage_head = nn.Sequential(
            nn.Linear(input_dim, num_y * self.K_v),
            nn.ReLU()
        )
        # future value v' head: output num_y * num_Kv values, then reshape
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, num_y * self.K_v),
            nn.ReLU()
        )
        #NOTE: I use this value to directly get EW_star, so that I do not need to call a gradient during the simulation stage
        #Instead, I can use this during the simulation and train it based on the MSE between this and the actual gradient
        #Question though: will the wages and values both move toward each other? That part may be a bit risky tbh.
        # Maybe I detach the gradient once I take it out of the foc_optimizer and use that for training!!!
        # hiring head: probability of hiring per discrete state y
        self.hiring_head = nn.Sequential(
            nn.Linear(input_dim, num_y),
            nn.ReLU()
        )
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        for seq in (self.trunk, self.value_head, self.wage_head, self.hiring_head):
         for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
    def forward(self, x):
        # x: [B, state_dim]
        B = x.size(0)
        features = self.trunk(x)  # [B, hidden_dims[-1]]

        # wages: reshape to [B, num_y, num_Kv]
        wages_flat = self.wage_head(features)            # [B, num_y * num_Kv]
        wages = wages_flat.view(B, self.num_y, self.K_v)  # [B, num_y, num_Kv]

        # values: reshape to [B, num_y, num_Kv]
        values_flat = self.value_head(features)            # [B, num_y * num_Kv]
        values = values_flat.view(B, self.num_y, self.K_v)  # [B, num_y, num_Kv]        
        # hire probabilities: [B, num_y]
        hiring = self.hiring_head(features)          # [B, num_y]

        return {
            'wages': wages,
            'values': values,
            'hiring': hiring
        }

#Function to create mini-batches
def random_mini_batches(X, minibatch_size=64, seed=0):
    """Generate random minibatches from X."""
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    # Step 1: Shuffle X
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]

    # Step 2: Partition shuffled_X. Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(math.floor(m / minibatch_size))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[(k * minibatch_size):((k+1) * minibatch_size), :]
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)

    return mini_batches

#Gradient functions
def get_batch_gradients(states, value_model, num_y, range_tensor=None):
    """
    Computes per-sample gradient of E_{y'|y} V(y', s) for all y ∈ {0, 1, ..., num_y-1}

    Args:
        states:        [B, D] — input states (normalized), requires_grad NOT required
        value_model:   neural net mapping [B, D] → [B, num_y]
        P_mat:         [num_y, num_y] — transition matrix with rows P(y → y′)
        range_tensor:  [D] or scalar, optional — rescale gradients (e.g., if states were normalized)

    Returns:
        expectation_grads: [B, num_y, D] — ∇_s E[V(y′|y, s)] for each y or fixed y
    """
    states = states.requires_grad_(True)
    B, D = states.shape
    eps = 1e-3
    # [B, D] → [B, 1, D], then broadcast-add an eye matrix [1, D, D]*eps → [B, D, D]
    eye = torch.eye(D, device=states.device) * eps        # [D, D]
    perturbs = states.unsqueeze(1) + eye.unsqueeze(0)      # [B, D, D]

    # flatten back to a big batch of size B*D
    flat_plus = perturbs.reshape(-1, D)                   # [B*D, D]

    # forward through value_model
    V_plus = value_model(flat_plus)          # [B*D, num_y]
    V_plus = V_plus.view(B, D, num_y).permute(0, 2, 1)     # → [B, num_y, D]

    V_minus = value_model(states).unsqueeze(-1) # [B, num_y, 1]
    grads      = (V_plus - V_minus) / eps                          # [B, D, num_y]

    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = grads / range_tensor[ax,ax,:]  # broadcast over D

    return expectation_grads  # [B, y, D] or [B, D] if current_y is not None

def get_expectation_gradients(states, value_model, P_mat,  i, range_tensor=None, current_y=None):
    """
    Computes per-sample gradient of E_{y'|y} V(y', s) for all y ∈ {0, 1, ..., num_y-1}

    Args:
        states:        [B, D] — input states (normalized), requires_grad NOT required
        value_model:   neural net mapping [B, D] → [B, num_y]
        P_mat:         [num_y, num_y] — transition matrix with rows P(y → y′)
        range_tensor:  [D] or scalar, optional — rescale gradients (e.g., if states were normalized)

    Returns:
        expectation_grads: [B, num_y, D] — ∇_s E[V(y′|y, s)] for each y or fixed y
    """
    states = states.requires_grad_(True)
    B, D = states.shape
    num_y = P_mat.shape[0]
    eps = 1e-3
    # [B, D] → [B, 1, D], then broadcast-add an eye matrix [1, D, D]*eps → [B, D, D]
    eye = torch.eye(D, device=states.device) * eps        # [D, D]
    perturbs = states.unsqueeze(1) + eye.unsqueeze(0)      # [B, D, D]

    # flatten back to a big batch of size B*D
    flat_plus = perturbs.reshape(-1, D)                   # [B*D, D]

    # forward through value_model
    V_plus = value_model(flat_plus)             # [B*D, num_y]
    V_plus = V_plus.view(B, D, num_y).permute(0, 2, 1)     # → [B, num_y, D]

    V_minus = value_model(states).unsqueeze(-1) # [B, num_y, 1]
    grads      = (V_plus - V_minus) / eps                          # [B, num_y, D]
    # Multiply by P^T to get: ∇_s E_{y'|y} V(y', s)
    # jacobian: [B, y', D];     P^T: [y', y]
    expectation_grads = torch.einsum("byd,zy->bzd", grads, P_mat)  # [B, y, D]
    #QUESTION: should I be using the transpose of P_mat here? No, without transpose, we will indeed get E_{y'|y} ∂V(y', s)/∂s_b
    
    #Optional: pick current y for the expectation
    if current_y is not None:
        #i=torch.arange(expectation_grads.shape[0])
        expectation_grads = expectation_grads[i,current_y.long(),:]
    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = expectation_grads / range_tensor[ax,:]  # broadcast over D

    #Temp: check that it's the same value as with the loop
    #ch = get_expectation_gradients_loop(states, value_model, P_mat, range_tensor=range_tensor, current_y=current_y)
    #assert ((expectation_grads-ch).abs().max() < 1e-8)
    return expectation_grads  # [B, y, D] or [B, D] if current_y is not None

def get_expectation_gradients_loop(states, value_model, P_mat,  range_tensor=None, current_y=None):
    """
    Computes per-sample gradient of E_{y'|y} V(y', s) for all y ∈ {0, 1, ..., num_y-1}

    Args:
        states:        [B, D] — input states (normalized), requires_grad NOT required
        value_model:   neural net mapping [B, D] → [B, num_y]
        P_mat:         [num_y, num_y] — transition matrix with rows P(y → y′)
        range_tensor:  [D] or scalar, optional — rescale gradients (e.g., if states were normalized)

    Returns:
        expectation_grads: [B, num_y, D] — ∇_s E[V(y′|y, s)] for each y or fixed y
    """

    B, D = states.shape
    num_y = P_mat.shape[0]
    jacobian = torch.zeros((B, num_y, D), dtype=type)  # Initialize Jacobian tensor
    #print(states.is_leaf)           # False
    # Detach any prior graph, ensure float precision
    #states = states.requires_grad_(True)
    V_minus = value_model(states)
    for dim in range(D):
        eps = 1e-3
        delta = torch.zeros_like(states)
        delta[:, dim] = eps

        V_plus = value_model(states + delta)
        jacobian[:,:, dim] = (V_plus - V_minus) / ( eps)

    # Multiply by P^T to get: ∇_s E_{y'|y} V(y', s)
    # jacobian: [B, y', D];     P^T: [y', y]
    expectation_grads = torch.einsum("byd,zy->bzd", jacobian, P_mat)  # [B, y, D]
    #QUESTION: should I be using the transpose of P_mat here? No, without transpose, we will indeed get E_{y'|y} ∂V(y', s)/∂s_b
    
    #Optional: pick current y for the expectation
    if current_y is not None:
        i=torch.arange(expectation_grads.shape[0])
        expectation_grads = expectation_grads[i,current_y.long(),:]
    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = expectation_grads / range_tensor  # broadcast over D

    return expectation_grads  # [B, y, D] or [B, D] if current_y is not None
#Class to compute the FOC residuals
class FOCresidual:
    """Class to compute the FOC residual for the CRS model"""
    def __init__(self, bounds_processor, K, p, cc):
        self.bounds_processor = bounds_processor  # Store bounds_processor
        self. K = K
        self.p = p
        self.deriv_eps = 1e-3 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)
        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid

        # Production Function in the Model
        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = tensor(self.fun_prod,dtype=type)
        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = self.p.u_bf_m

        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)
        self.Z_trans_tensor = tensor(self.Z_trans_mat, dtype=torch.float32)
        # Value Function Setup
        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max(), self.p.num_v )
        self.rho_grid=tensor(1/self.pref.utility_1d(self.w_grid), dtype = type)
        # Normalize rho_grid to tensor for model input
        self.rho_normalized = self.bounds_processor.normalize_rho(self.rho_grid).unsqueeze(1).requires_grad_(True)

        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        self.v_0 = self.v_grid[0]

        self.simple_J = torch.zeros((self.p.num_z,self.p.num_v), dtype = type)
        self.simple_J= (self.fun_prod[:,ax] -self.pref.inv_utility(tensor(self.v_grid[ax,:]*(1-self.p.beta),dtype = type)))/ (1-self.p.beta)
        self.simple_Rho = self.simple_J + self.rho_grid[ax,:] * tensor(self.v_grid[ax,:], dtype = type)#We do indeed need to work with Rho here since we're taking W via its derivatives
        #Apply the matching function: take the simple function and consider its different values across v.
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        if cc is None:
            self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
            self.js.update(self.v_grid[:], self.prob_find_vx.numpy()) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        else:
            self.js = cc.js

    def production(self,sum_n):
        return sum_n ** self.p.prod_alpha
    def production_1d_n(self,sum_n):
        return self.p.prod_alpha * (sum_n ** (self.p.prod_alpha - 1))    
    def getWorkerDecisions(self, EW1, employed=True): #Andrei: Solves for the entire matrices of EW1 and EU
        """
        :param EW1: Expected value of employment
        :param EU:  Expected value of unemployment
        :param employed: whether the worker is employed (in which case we multiply by efficiency)
        :return: pe,re,qi search decision and associated return, as well as quit decision.
        """
        pe, re = self.js.solve_search_choice(EW1) #Uses the job search array to solve for the search choice
        assert (~torch.isnan(pe)).all(), "pe is NaN"
        assert (pe <= 1).all(), "pe is not less than 1"
        assert (pe >= -1e-10).all(), "pe is not larger than 0"
        #ve = self.js.ve(EW1)
        if employed:
            pe = pe * self.p.s_job
            re = re * self.p.s_job
        #print("Shape of pe:", pe.shape)
        # construct the continuation probability. #Andrei: probability the worker doesn't get fired and also doesn't leave
        pc = (1 - pe)

        return re, pc #ve is vhat, the value the worker gets upon finding a job    
    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)    
    def future_values(self, states, prod_states, fut_states, rho_star, hiring, value_net):
        """Compute the expected value of the value function for the given states and policy"""
        #policies_norm[:,K_v] = self.bounds_processor.normalize(fut_states) #No need to normalize, these are already correct.
        #Need to attach policies to EW_star here. Use fut_states as a way to get future size here in the derivative. Alternatively, I can loop here a couple of times, updating size based on pc(EW_star) and then calling it again
        i=torch.arange(fut_states.shape[0])
        fut_size = self.bounds_processor.denormalize_size(fut_states[:,:K_n])
        fut_states_adj = torch.cat((hiring.unsqueeze(1),fut_size[:,1:K_n],rho_star),dim=-1)
        fut_states_adj = self.bounds_processor.normalize(fut_states_adj)
        EW_star = get_expectation_gradients(fut_states_adj, value_net, self.Z_trans_tensor, i=i, range_tensor=self.bounds_processor.range,current_y = prod_states)[:,K_n:]
        size = self.bounds_processor.denormalize_size(states[:,:K_n])
        pos_sen = fut_size[:,1] > 0 #Only if we have no future seniors, does EW_star stop making sense
        EW_star[~pos_sen,:] = self.v_grid[0] #Just a temp thing to avoid NaNs
        EW_star[pos_sen,:] = EW_star[pos_sen,:]/fut_size[pos_sen,1].unsqueeze(1)
        re, pc = self.getWorkerDecisions(EW_star[pos_sen,:], employed=True) #Get the decisions for the seniors only
        n_1 = torch.zeros((fut_states.shape[0]),1, dtype=type)
        n_1[pos_sen,:] = ((size[pos_sen,0]+size[pos_sen,1]) * pc.squeeze(1)).unsqueeze(1)
        #Now redo this to incorporate pc
        fut_states_adj = torch.cat((hiring.unsqueeze(1),n_1,rho_star) ,dim=-1)
        fut_states_adj = self.bounds_processor.normalize(fut_states_adj)
        future_grad = get_expectation_gradients(fut_states_adj, value_net, self.Z_trans_tensor, i=i, range_tensor=self.bounds_processor.range,current_y = prod_states)        

        #Should I update EW_star again? I think yes
        EW_star = future_grad[:,K_n:] #not an actual EW_star, but v'*n_1
        EW_star[~pos_sen,:] = self.v_grid[0] #Just a temp thing to avoid NaNs
        EW_star[pos_sen,:] = EW_star[pos_sen,:]/n_1[pos_sen,:] 
        #re, pc = self.getWorkerDecisions(EW_star)
        V_all = value_net(fut_states_adj)
        ERho_star= torch.einsum("by,zy->bz", V_all, self.Z_trans_tensor) #Should this be transposed? Doesn't matter now but will later.
        ERho_star = ERho_star[i,prod_states.long()]
        #n_1 = self.bounds_processor.denormalize_dim(fut_size,1)
        EJ_star = ERho_star.unsqueeze(1) - rho_star * n_1 * EW_star #This should all be policy shape
        #EW_star = EW_star / fut_size #Normalizing, because ∂ 𝒫(y,n_k,ρ_k,z_k)/∂ ρ_k = n_k W_k. So to get true W_k gotta divide by the size
        return  EJ_star, EW_star, future_grad, re, pc, n_1   
    def values(self, states, prod_states, EJ_star, EW_star, pc, re, hiring, rho_star, future_grad):
        """Compute the value function for the given states and policy"""
        states_d = self.bounds_processor.denormalize(states)
        size = states_d[:, :K_n]
        #s_pos = states_d[:,1] > 0 #For wages, look only at the cases where we have a positive number of seniors 
        rho = states_d[:,K_n:].squeeze(1)
        wages = torch.zeros(rho.shape[0],K_n) 

        wages[:,1] = tensor(np.interp(rho.detach().numpy(),self.rho_grid,self.w_grid),dtype = type)
        worker_values = self.pref.utility(wages[:,1:]) + self.p.beta * (EW_star + re)
        wages[:,:1] = self.pref.inv_utility(self.v_0 - self.p.beta*((EW_star+re)))
        tot_size = 0
        tot_wage = 0
        for k in range(K_n):
            tot_size += size[:,k]
            tot_wage += size[:,k] * wages[:,k]
        values = self.fun_prod[prod_states.detach().long()] * self.production(tot_size) - self.p.hire_c * hiring - \
            tot_wage + rho * size[:,1] * worker_values.squeeze(1) + self.p.beta * EJ_star.squeeze(1)
        grad = torch.zeros((states.shape[0],K_n+K_v), dtype=type)
        #grad[:,:K_n] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,:] + self.p.beta * pc_star * (future_grad[:,1:K_n] - omega * v_prime)# notice that future_grad is the same for juniors and seniors, because both end up becoming seniors
        grad[:,:1] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,:1] + self.p.beta * pc * (future_grad[:,1:K_n] - rho_star * EW_star)
        grad[:,1:K_n] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,1:] + self.p.beta * pc * (future_grad[:,1:K_n] - rho_star * EW_star) + worker_values * rho.unsqueeze(1)    
        grad[:,K_n:] = worker_values * size[:,1:] #The gradient is multiplied by the size
        #values = tensor(values, dtype=type) #Convert to tensor
        #print("Diff btw two versions", torch.abs(pc_star*EJ_star - (pc_star*ERho_star - policies.unsqueeze(1) * pc_star * EW_star) ).mean().item())
        return values, grad
    def FOC_loss(self, states, rho_star, hiring, pc, re, EW_star, future_grad):
        """Given the Value network and the policy network,
        Compute the FOC residuals for given set of states
        Requires: EW_star for each point, derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star ? Right?
        Note: for hirings and layoffs, will need to ensure the loss is zero when the policy is consistent: sep=0 => FOC<0, sep=1 => FOC>0 etc
        """
        # Derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star
        EJ_deriv = future_grad[:,1:K_n] - rho_star * EW_star #Should it be like this??? Because, if I write 𝒫(y,n_k,̃ρ_k) = max_{v_k} J(y,n_k,v_k) + ̃ρ_k v_k, then v_k has no role to play in the derivative wrt n_k!
        size=self.bounds_processor.denormalize_size(states[:,:K_n])
        #Get worker's search decisions and associated return
        re, pc = self.getWorkerDecisions(EW_star)
        _, pc_d = self.getWorkerDecisions(EW_star + self.deriv_eps) 
        # After computing pc and pc_d:
        assert not torch.isnan(pc).any(), "NaN in pc"
        assert not torch.isnan(pc_d).any(), "NaN in pc_d"
        # Clamp pc and pc_d to avoid log(0)
        log_diff = torch.zeros_like(EW_star)
        ##log_diff[:] = torch.nan
        log_diff[pc > 0] = torch.log(pc_d[pc > 0]) - torch.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
        assert torch.all(log_diff >= 0)
        assert torch.all(~torch.isnan(log_diff))
        # After computing log_diff:
        assert not torch.isnan(log_diff).any(), "NaN in log_diff"
        #Neeed an updated foc that includes size, same as in my 2-tenure step model!!! Check how I write this theoretically in order to adapt. Gotta make sure
        inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*((EW_star+re)))
        assert torch.all(inv_utility_1d > 0)
        # After computing inv_utility_1d:
        assert not torch.isnan(inv_utility_1d).any(), "NaN in inv_utility_1d"
        # Clamp inv_utility_1d to avoid division by zero
        inv_utility_1d = torch.clamp(inv_utility_1d, min=1e-8)

        focs_rho_2ndpart = - self.bounds_processor.denormalize_rho(states[:, K_n:]) * size[:,1:] -   size[:,:1]  / inv_utility_1d #BIG NOTE: I am taking ρ_1 * n_1 directly here from states. As this is all that I need, I do not need ρ separately. BUT! I WILL NEED IT IN ALL THE OTHER STEPS!!! So I will need to either keep track of it or... remove cases where size is zero. I already did that in the VFI approach, too. That makes more sense tbh. Then I can deduce ρ=(ρ * n) / n. For this I will need to take a subset at some point. s = size[:,1:] > 0 or smth like that.                 
        focs_rho = rho_star - EJ_deriv[:,0:] * (log_diff / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
        focs_rho = focs_rho*(size[ :,:1] + size[ :,1:]) + focs_rho_2ndpart    
        assert torch.all(~torch.isnan(focs_rho))
        assert not torch.isnan(focs_rho).any(), "NaN in inv_utility_1d"

        #Now hiring FOC
        #fg_hire = future_grad[:, 0].clamp(-1e2, 1e2)
        focs_hire = self.p.beta * future_grad[:, 0] - self.p.hire_c #Maybe Rho instead of J??? No, should be J I think.
        focs_hire[hiring <= 0] = torch.relu(focs_hire[hiring <= 0]) #This is the case where the firm is not hiring. So we only keep the loss if the FOC is positive
        assert not torch.isnan(focs_hire).any(), "NaN in inv_utility_1d"
        return focs_rho, focs_hire
    def initiation(self, states, prod_states, fut_states, rho_star, hiring, value_net):
        EJ_star, EW_star, future_grad, re, pc, n_1 = self.future_values(states, prod_states, fut_states, rho_star, hiring, value_net)

        return EJ_star, EW_star, future_grad, re, pc, n_1        


def simulate(starting_states, policy_net, bounds_processor, simulation_steps):
    """Simulate the firm path using the policy network
    Track the reached states and the corresponding values (values may be too hard to track)
    Output: set of (reached) states and corresponding values
    Args:
        starting_states: [B, D] — input states (normalized), requires_grad NOT required
        prod_states:     [B]   — production states (discrete), requires_grad NOT required
        value_net:       neural net mapping [B, D] → [B, num_y]
        policy_net:      neural net mapping [B, D] → [B, num_y]
        simulation_steps: number of steps to simulate
    """
    B = starting_states.shape[0] #initial batch size
    D = starting_states.shape[1] #number of state variables
    his_size_total=0
    for t in range(simulation_steps):
        his_size_total+= B * (p.num_z**(t+1)) #How many states we can reach in simulation_steps steps

    S = torch.zeros(his_size_total, D, dtype=type) #Vector of all reached states
    Fut_states = torch.zeros(his_size_total, D, dtype=type) #Vector off all future states (we know them all besides y'). It's just policies for each state 
    #V = torch.zeros(his_size_total, dtype=type) #Vector off all reached values. It's just values for each state (productivity alrdy included via P)
    P = torch.arange(S.shape[0]) % p.num_z #Vector off all production states

    his_start = 0
    his_end = his_start + p.num_z * B
    his_size = his_end - his_start
    states = starting_states
    for t in range(simulation_steps):
        #    S[his_start:his_end,:] = states.repeat(p.num_z, 1) #We repeat the policy from the previous step   
        S[his_start:his_end,:] = states.repeat(p.num_z, 1)
        policy = policy_net(states) 
        hiring = policy['hiring'] #Shape [B,num_y]
        values = policy['values']
        wages = policy['wages']
        #Fut_val[his_start:his_end,:, :] = values.view(B*p.num_z,D) #Given today's y and state S, we get ρ*(S,y)
        #V[his_start:his_end] = value_net(states).view(-1)
        #Get probability of staying... wait. In that case, if I wanna do it later, I can compile the entire W = u(w_state_today) + beta * (EW_star=future_value + re_star=re from getWorkerDecisions)
        _, pc = foc_optimizer.getWorkerDecisions(values.view(his_size,K_v))
        sizes = bounds_processor.denormalize_size(states[:,:K_n])
        tot_size = sizes[:,0]+sizes[:,1] #This is ok only if the sizes have the same range. So gotta be careful here
        #Set up new states
        states = torch.zeros(his_size, D, dtype= type)
        states[:,1:K_n] = ((tot_size.repeat(p.num_z)) * pc.squeeze(1)).unsqueeze(1)#Future size, based on n'= n * pc(v') * (1 - s). Extreme case here as n'_1=(n'_0+n'_1) * pc
        states[:,0] = hiring.view(-1) #Future jun size, based on hiring
        states[:,K_n:] = wages.view(his_size,K_v) # state ρ_1*n_1
        # In simulate(), after updating states:
        assert not torch.isnan(states).any(), "NaN in simulated states"
        assert torch.all(states[:, K_n:] >= 0), "Negative rho*n in states"
        states = bounds_processor.normalize(states) #Now all the states are normalized together
        Fut_states[his_start:his_end,:] = states
        his_start = his_end
        his_end = his_start + his_size * p.num_z
        his_size = his_end - his_start
    assert (his_start == S.shape[0])
    #Append P to S. That way, when I sample minibatches, I can just sample S and P together.
    #S=torch.cat((S, P.unsqueeze(1)), dim=1)

    return S, P, Fut_states #Doing values here may be not as efficient since some of them may not even be sampled.

def initialize(bounds_processor, state_dim, K_n, K_v, hidden_dims, learning_rate, weight_decay, pre_training_steps, num_epochs, minibatch_num):
    #Initializations:
    
    # Initialize value function neural network
    value_net = ValueFunctionNN(state_dim, p.num_z, hidden_dims)
    policy_net = PolicyNN(state_dim, p.num_z, K_v, hidden_dims, cc)
    # Initialize neural network optimizer
    #optimizer_value = optim.AdamW(value_net.parameters(), lr=learning_rate[0], weight_decay=weight_decay[0])
    #optimizer_policy = optim.AdamW(policy_net.parameters(), lr=learning_rate[1], weight_decay=weight_decay[1])

    # Use Ranger21 for all three networks:
    optimizer_value = RangerOptimizer(
        params=value_net.parameters(),
        lr=learning_rate[0],
        weight_decay=weight_decay[0],
        num_epochs=num_epochs,       # for built‑in warmup + scheduler
        num_batches_per_epoch=minibatch_num * (3/4),
        num_warmup_iterations=int(0.05 * num_epochs)  # 5% of training
    )
    optimizer_policy = RangerOptimizer(
        params=policy_net.parameters(),
        lr=learning_rate[1],
        weight_decay=weight_decay[1],
        num_epochs=num_epochs,
        num_batches_per_epoch= (minibatch_num / 4),
        num_warmup_iterations=int(0.05 * num_epochs)
    )
    # Initialize FOC computer
    foc_optimizer = FOCresidual(bounds_processor, K=K_n, p=p, cc=None)    

    #Step 0: basic guess
    value_net, policy_net, optimizer_value, optimizer_policy = pre_training(optimizer_value,optimizer_policy,value_net,policy_net,foc_optimizer,bounds_processor, K_n, K_v, pre_training_steps)   

    return value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer
def pre_training(optimizer_value,optimizer_policy,value_net,policy_net,foc_optimizer,bounds_processor, K_n, K_v, pre_training_steps):
    rho_states = bounds_processor.normalize_rho(tensor(cc.rho_grid[:,ax],dtype=type))
    assert torch.all(rho_states[1:] > rho_states[:-1]), "States are not increasing"
    crs_Rho = foc_optimizer.simple_Rho.t()
    #Train also on the gradient
    target_W = torch.zeros_like(crs_Rho) + tensor(foc_optimizer.v_grid[ax,:], dtype=type).t()
    #Let the policy just be today's rho.
    target_wages =  torch.zeros_like(crs_Rho) + tensor(foc_optimizer.rho_grid[ax,:], dtype=type).t()
    assert not torch.isnan(crs_Rho).any(), "NaN in crs_Rho"
    assert not torch.isnan(target_wages).any(), "NaN in target_wages"
    assert not torch.isnan(target_W).any(), "NaN in target_W"

    #Multiply the target gradient by size
    #target_W_grad = target_W[...,ax] * size[:,ax,1:] #Careful about the dimensions here. At the end this should be [B,num_y, D], where D=K_n
    print("Pre-training...")
    for _ in range(pre_training_steps):

        #Now randomize size... wait HOW!!! I don't have an upper bound for size!
        size = torch.rand(rho_states.shape[0], K_n,dtype=type) #Random sizes for each size state. Shape [B,K_n]
        states = torch.zeros(rho_states.shape[0],K_n+K_v)
        states[:,:K_n] = size #First size states, then rho, then quality
        size = bounds_processor.denormalize_size(size)
        states[:,K_n:] = bounds_processor.normalize_dim(tensor(cc.rho_grid,dtype=type) * size[:,1],-1).unsqueeze(1) # State is ρ_k n_k
        target_values = torch.zeros_like(crs_Rho,dtype=type)
        for k in range(K_n):
            target_values += crs_Rho * size[:,ax,k] #final shape [B, num_y] 
        predicted_values = value_net(states)
        predicted_grads = get_batch_gradients(states, value_net,  num_y=predicted_values.shape[1], range_tensor=bounds_processor.range)[:,:,K_n:] #These are gradients wrt rho_k forall k besides the bottom one
        value_loss = nn.MSELoss()(predicted_values, target_values) + nn.MSELoss()(predicted_grads, target_W[...,ax]) 
        value_loss.backward() #Backpropagation
        optimizer_value.step() #Update the weights
        optimizer_value.zero_grad()
        #Policy loss: very specific here bcs its not a FOC loss. EVEN THOUGH I COULD MAKE IT A FOC LOSS.
        policy = policy_net(states)
        predicted_wages = policy['wages'] #I am not training hiring here, only rho_star
        predicted_values = policy['values']
        wage_loss = nn.MSELoss()(predicted_wages, target_wages[...,ax])
        future_value_loss = nn.MSELoss()(predicted_values, target_W[...,ax])
        violation = torch.relu(policy['hiring'][:,:-1] - policy['hiring'][:,1:])#Should hire more in better states. Tbf the ρ' and EW* should also be increasing, no?
        mon_loss = (violation ** 2).mean()
        policy_loss = 2 * wage_loss + future_value_loss + mon_loss
        policy_loss.backward()
        optimizer_policy.step()
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
    return value_net, policy_net, optimizer_value, optimizer_policy
def train(state_dim, value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer, bounds_processor, num_episodes=20, minibatch_num=8, starting_points_per_iter=100, simulation_steps=5, 
    minibatch_size=512, λ=1.0, target_values=None, target_W=None, use_saved_nets = False):
    """
    Main training loop for value function approximation
    
    Args:
        state_dim: Dimension of state space
        num_iterations: Number of training iterations
        starting_points_per_iter: Number of starting points per iteration
        simulation_steps: Steps to simulate for each starting point
        learning_rate: Learning rate for neural network optimizer
        discount_factor: Discount factor for future rewards
    
    Returns:
        Trained value function model
    """
    if use_saved_nets:
        print("Loading saved nets")
        value_net.load_state_dict(torch.load("trained_value_function.pt"))
        policy_net.load_state_dict(torch.load("trained_policy_function.pt"))
    #Initialize a target network
    target_value_net = copy.deepcopy(value_net)
    # Ensure it's not updated by the optimizer
    for param in target_value_net.parameters():
        param.requires_grad = False
    # Define which losses you’ll track
    loss_names = ['value','value_grad','FOC_wage','FOC_hire','fut_value']

    # Initialize once before training
    plotter = LossPlotter(loss_names, pause=0.001, update_interval=10)
    
    print("Training...")
    # Training loop
    for episode in tqdm(range(num_episodes)):
        ep= episode + 1
        #torch.autograd.set_detect_anomaly(True)
        #Starting states. First with a single junior and nothing else. State = (y, {1,0},{rho_min,rho_min})
        state_start = torch.zeros(state_dim,dtype=type)
        state_start[0] = bounds_processor.normalize_dim(1,0) # 1 junior worker
        #Or if randomized. 
        states= torch.rand(starting_points_per_iter, state_dim,dtype=type) 
        #Add the statrting state
        states[0,:] = state_start
        #Simulate the firm path using the policy network
        with torch.no_grad():
            states, prod_states, fut_states  = simulate(states, policy_net, bounds_processor, simulation_steps) #This is the set of states we will use to train the value function. 
        #States here includes productivity state as the last column!
        #Now append future states to the states so that I can mini-batch them together.
        comb_states = torch.zeros(states.shape[0], state_dim, 3, dtype=type)
        comb_states[...,0] = states
        comb_states[...,1] = fut_states
        comb_states[...,2] = prod_states.unsqueeze(-1)
        # Mini-batch the simulated data
        minibatch_size = np.floor(states.shape[0]/minibatch_num).astype(int)
        minibatches = random_mini_batches(comb_states, minibatch_size)
        batch_index = 0
        for mb in minibatches:    
            optimizer_value.zero_grad()
            optimizer_policy.zero_grad()
            #Detach all the arrays again
            states= mb[..., 0]
            fut_states = mb[...,1]
            prod_states = mb[..., -1, 2]
            batch_index += 1
            i = torch.arange(states.shape[0])
            policies = policy_net(states.requires_grad_(True))
            #Gotta now do wages, hiring, and values separately
            rho_star = policies['wages'][i,prod_states.long(),:]
            values = policies['values'][i,prod_states.long(),:]
            hiring = policies['hiring'][i,prod_states.long()] 
            EJ_star, EW_star, future_grad, re, pc, n_1 = foc_optimizer.initiation(prod_states=prod_states, states = states, fut_states = fut_states, rho_star=rho_star, hiring=hiring, value_net=target_value_net)  #Note that I am using the target value here!!!          

            if ((batch_index) % 4)==0: #Policy Update
                #optimizer_policy.zero_grad() SHould not be here, since then we immediately forget the impact on future_grad, for example
                FOC_rho_resid,FOC_hire_resid = foc_optimizer.FOC_loss(states=states, rho_star=rho_star, hiring=hiring, pc=pc, re=re, EW_star=EW_star, future_grad=future_grad)
                FOC_wage_loss = nn.MSELoss()(FOC_rho_resid, torch.zeros_like(FOC_rho_resid))
                FOC_hire_loss =  nn.MSELoss()(FOC_hire_resid, torch.zeros_like(FOC_hire_resid))
                fut_value_loss = nn.MSELoss()(values*n_1.detach(),future_grad[:,K_n:].detach()) #Loss the predicted EW_star via values and the "actual" EW_star (keep in mind that, in calculating actual EW_star, we still keep in mind the size guess based on the fut_values guess, see Simulation())
                #Note that I detach EW_star so that other policies do not try to affect anything going on here.
                policy_loss = FOC_wage_loss + FOC_hire_loss + fut_value_loss #Get the total loss for the states in the minibatch
                if episode % 1000 == 0:
                    print(f"EW* norm: {EW_star.norm().item():.4f}")
                policy_loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer_policy.step()
                optimizer_policy.zero_grad()
            else: #Value function update
                #optimizer_value.zero_grad()
                target_values, target_grad = foc_optimizer.values(states=states, prod_states=prod_states, EJ_star=EJ_star, EW_star=EW_star, re=re, hiring=hiring, pc = pc, rho_star=rho_star, future_grad=future_grad) 
                pred_values = value_net(states)
                predicted_grad = get_batch_gradients(states, value_net,  num_y = pred_values.shape[1], range_tensor=bounds_processor.range)[:,:,:]
                pred_values = pred_values[i,prod_states.long()] #Get the values for the states in the minibatch
                predicted_grad = predicted_grad[i,prod_states.long(),:] #Get the values for the states in the minibatch
                value_loss = nn.MSELoss()(pred_values, target_values)
                value_grad_loss = nn.MSELoss()(predicted_grad, target_grad) #Get the value loss for the states in the minibatch
                tot_value_loss = 1e-1 * value_loss + λ * value_grad_loss #Combine the losses
                tot_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
                optimizer_value.step()
                optimizer_value.zero_grad()
                    # Collect your raw loss scalars
        losses = {
        'value':       value_loss.item(),
        'value_grad':  value_grad_loss.item(),
        'FOC_wage':    FOC_wage_loss.item(),
        'FOC_hire':    FOC_hire_loss.item(),
        'fut_value':   fut_value_loss.item(),
        }

        # One line hides all the plotting mess
        plotter.update(ep, losses)
        
        #Hard copy the target value at the end of every episode
        target_value_net.load_state_dict(value_net.state_dict(), strict=True)
        # Print progress
        if (episode + 1) % (num_episodes/20) == 0 or episode == 0:
            print(f"Iteration {episode + 1}, Value Loss: {value_loss.item():.6f}, Value Grad Loss:  {value_grad_loss.item():.6f},FOC_wage_loss: {FOC_wage_loss.item():.6f}, FOC_hire_loss: {FOC_hire_loss.item():.6f} ,fut_value_loss: {fut_value_loss.item():.6f}" )
    
    return value_net, policy_net

if __name__ == "__main__":
    # Define parameters
    K = 2 #Number of tenure steps
    #Number of states
    K_n = K #K size states
    K_v = K - 1 #K - 1 (ignoring bottom) value states
    K_q = K - 1 #K - 1 (ignoring bottom) quality states. Ignore them for now
    STATE_DIM = K_n + K_v # + K_q #Discrete prod-ty y as multiple outputs
    ACTION_DIM = K_v + 1 # + K_n  # rho_star + hiring + separations. BIG QUESTION: do I do my thing (firms internalizing finite K) first to check? And only then move on? That's longer, but def more safe
    HIDDEN_DIMS = [64,64]  # Basic architecture. Basically every paper has 2 inner layers, can make them wider though
    λ = 1.0  # Relative loss importance across networks

    #pref = Preferences(input_param=p_crs)
    cc=ContinuousContract(p_crs()) 
    cc_J,cc_W,cc_Wstar,rho_star = cc.J(0) 
    target_values = tensor(cc_J + cc.rho_grid[ax,:] * cc_W, dtype=type)
    target_W = tensor(cc_W, dtype=type)
    #assert torch.all(target_W[:,1:] >= target_W[:,:-1]), "target_W is not increasing"
    #NORMALIZE EVERYTHING!!!
    LOWER_BOUNDS = [0, 0 , cc.rho_grid[0]] # The state space is (y,n_0,n_1,ρ_1 * n_1). This is fine, since we don't actually care about the derivative wrt just ρ.
    UPPER_BOUNDS = [20, 50, cc.rho_grid[-1]] #Ideally this should come from fun_prod.max. Either way, these are bounds ONLY for wage. Not for size.
    minibatch_num = 4
    num_episodes=20000
    #Initialize
    bounds_processor_wage = StateBoundsProcessor(LOWER_BOUNDS,UPPER_BOUNDS)
    value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer = initialize(bounds_processor_wage, STATE_DIM, K_n, K_v, HIDDEN_DIMS,
                                                                                         learning_rate=[1e-5,1e-5], weight_decay = [1e-2, 1e-2] ,pre_training_steps=0, num_epochs=num_episodes, minibatch_num=1000)
    
    # Train value function
    print("Training value function...")
    beg=time()
    trained_value, trained_policy = train(
    STATE_DIM, value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer, bounds_processor_wage,
        num_episodes=num_episodes, minibatch_num=minibatch_num,
        starting_points_per_iter=5,
        simulation_steps=6,
        minibatch_size=300, λ=λ,
        target_values=target_values.t(), target_W=target_W.t(), use_saved_nets = False
    )
    print("Training time:", time()-beg)

    # Evaluate trained model
    #evaluate_value_function(trained_value, trained_policy, p, LOWER_BOUNDS, UPPER_BOUNDS,cc,target_values,cc_W,rho_star)

    # Save the model
    torch.save(trained_value.state_dict(), "trained_value_function.pt")
    torch.save(trained_policy.state_dict(), "trained_policy_function.pt")    
    print("Model saved")
