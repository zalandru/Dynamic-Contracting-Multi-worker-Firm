#Neural Network solver for the CRS model, with both sup and value networks
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
    def normalize_rho(self, states):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[ax, K_n:]) / self.range[ax, K_n:]    #Note: this will need to be adjusted at multiple steps    
    def denormalize_rho(self, normalized_states):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[ax,K_n:] + self.lower_bounds[ax,K_n:] #Check: 0.5 * 10 + 10= 15. 
    def denormalize(self, normalized_states):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[ax, :] + self.lower_bounds[ax, :] #Check: 0.5 * 10 + 10= 15.
    def normalize_dim(self, states,dim):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[dim]) / self.range[dim]    #Note: this will need to be adjusted at multiple steps      
    def denormalize_dim(self, normalized_states,dim):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[dim] + self.lower_bounds[dim] #Check: 0.5 * 10 + 10= 15. 
    def denormalize_size(self, normalized_states):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[ax,:K_n] + self.lower_bounds[ax,:K_n] #Check: 0.5 * 10 + 10= 15.

class StateBoundsProcessor_inf:
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
    def denormalize(self, normalized_states):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[ax, :] + self.lower_bounds[ax, :] #Check: 0.5 * 10 + 10= 15.
    def normalize_sup(self, states):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[ax, K_n+K_v:]) / self.range[ax, K_n+K_v:]
    def denormalize_sup(self, normalized_states):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[ax, K_n+K_v:] + self.lower_bounds[ax, K_n+K_v:] #Check: 0.5 * 10 + 10= 15.
    def normalize_dim(self, states,dim):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        return (states - self.lower_bounds[dim]) / self.range[dim]    #Note: this will need to be adjusted at multiple steps      
    def denormalize_dim(self, normalized_states,dim):
        """Convert normalized states back to original range"""
        return normalized_states * self.range[dim] + self.lower_bounds[dim] #Check: 0.5 * 10 + 10= 15. 
    
#Neural Nets. Note: alternatively, I could code give them the same trunk. Either way, this is called an Actor-Critic architecture.
class ValueFunctionNN(nn.Module):
    """Neural network to approximate the value function"""
    def __init__(self, state_dim, num_y, hidden_dims=[40, 30, 20, 10]):
        super(ValueFunctionNN, self).__init__()
        
        # Build layers
        layers = []
        input_dim = state_dim
        # shared trunk
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            # Consider adding layer normalization for stability
            #layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            input_dim = h
        self.trunk = nn.Sequential(*layers)

        # head for values: [B, num_y]
        self.value_head = nn.Linear(input_dim, num_y)

        self._init_weights()
        self.state_dim = state_dim
        self.num_y     = num_y
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
    def forward(self, x):
        B = x.size(0)
        features = self.trunk(x)                    # [B, hidden_dims[-1]]
        values   = self.value_head(features)        # [B, num_y]

        #grad_flat = self.grad_head(features)        # [B, num_y * state_dim]
        #grads = grad_flat.view(B, self.num_y, self.state_dim)  # [B, num_y, state_dim]
        return {
            'values': values
        }

class PolicyNN(nn.Module):
    """Neural network to approximate a multi-dimensional sup:
       - values: multiple values per productivity state y across a predefined set of K_v
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
            # Consider adding layer normalization for stability
            #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # ✨ Extra “adapter” for the value head
        self.value_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
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
        self.hiring_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )        
        
        self.hiring_head = nn.Sequential(
            nn.Linear(input_dim, num_y),
            nn.ReLU()
        )
        # Apply activation‐specific initialization
        self._init_weights()
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        for seq in (self.trunk, self.value_adapter, self.hiring_adapter, self.hiring_head, self.value_head):
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    def forward(self, x):
        # x: [B, state_dim]
        B = x.size(0)
        features = self.trunk(x)  # [B, hidden_dims[-1]]



        # values: reshape to [B, num_y, num_Kv]
        hv = self.value_adapter(features)  # [B, hidden_dims[-1]]
        values_flat = self.value_head(hv)            # [B, num_y * num_Kv]
        values = values_flat.view(B, self.num_y, self.K_v)  # [B, num_y, num_Kv]        
        # hire probabilities: [B, num_y]
        hh = self.hiring_adapter(features)  # [B, hidden_dims[-1]]
        hiring = self.hiring_head(hh)          # [B, num_y]

        return {
            'values': values,
            'hiring': hiring
        }

class infNN(nn.Module):
    """Neural network to approximate a multi-dimensional sup:
       - omega: multiple values per productivity state y across a predefined set of K_v
       - hiring decision: probability per productivity state y
    """
    def __init__(self, state_dim, num_y, K_v, hidden_dims=[40, 30, 20, 10], cc=None):
        super(infNN, self).__init__()
        self.K_v = K_v #Number of Value-related policies
        self.num_y = num_y
        #state_dim = state_dim + K_v + 1 #add policies as extra states
        #States are: n'_0,n'_1,v'_1. That's the only thing we need
        # shared trunk
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Consider adding layer normalization for stability
            #layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # wage head: output num_y * num_Kv wage values, then reshape
        self.wage_head = nn.Sequential(
            nn.Linear(input_dim, num_y * self.K_v),
            nn.ReLU()
        )
        # Apply activation‐specific initialization
        self._init_weights()
        # ---- add this block ----
        # He‐init the Linear, small positive bias to “turn on” the ReLU
       # lin = self.wage_head[0]
        #nn.init.kaiming_uniform_(lin.weight, nonlinearity='relu')
        #lin.bias.data.fill_(0.1)
        # ------------------------
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        for seq in (self.trunk,self.wage_head):
         for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
    def forward(self, x):
        # x: [B, state_dim]
        B = x.size(0)
        features = self.trunk(x)  # [B, hidden_dims[-1]]

        # omega: reshape to [B, num_y, num_Kv]
        omega_flat = self.wage_head(features)            # [B, num_y * num_Kv]
        omega = omega_flat.view(B, self.num_y, self.K_v)  # [B, num_y, num_Kv]     
        return {
            'omega': omega
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

def get_batch_gradients_standard(states, value_model, num_y, range_tensor=None):
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
    V_plus = value_model(flat_plus)['values']             # [B*D, num_y]
    V_plus = V_plus.view(B, D, num_y).permute(0, 2, 1)     # → [B, num_y, D]

    V_minus = value_model(states)['values'].unsqueeze(-1) # [B, num_y, 1]
    grads      = (V_plus - V_minus) / eps                          # [B, D, num_y]

    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = grads / range_tensor[ax,ax,:]  # broadcast over D

    return expectation_grads  # [B, y, D] or [B, D] if current_y is not None

def get_expectation_gradients_standard(states, value_model, P_mat,  i, range_tensor=None, current_y=None):
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
    V_plus = value_model(flat_plus)['values']             # [B*D, num_y]
    V_plus = V_plus.view(B, D, num_y).permute(0, 2, 1)     # → [B, num_y, D]

    V_minus = value_model(states)['values'].unsqueeze(-1) # [B, num_y, 1]
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

def get_batch_gradients(states, value_model, num_y, range_tensor=None):
    """
    Compute per-sample gradient estimates via Richardson extrapolation of central differences.
    Args:
        model: a network mapping states -> scalar values (V)
        states: tensor of shape [B, D]
        eps: finite-difference step size
    Returns:
        grads: tensor of shape [B, D]
    """
    B, D = states.shape
    grads = torch.zeros(B,num_y, D)
    eps=1e-3
    # Precompute unit vectors
    eye = torch.eye(D, device=states.device, dtype=states.dtype)

    for i in range(D):
        e = eye[i:i+1]  # [1, D]
        # Big step
        vp = value_model(states + eps * e)['values']
        vm = value_model(states)['values']
        G_big = (vp - vm) / ( eps)  # [B]

        # Half step
        eps2 = eps * 0.5
        vp2 = value_model(states + eps2 * e)['values']
        vm2 = vm
        G_small = (vp2 - vm2) / eps2  # [B]

        # Richardson combination
        grads[..., i] = (4 * G_small - G_big) / 3
    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        grads = grads / range_tensor[ax,ax,:]  # broadcast over D
    return grads

def get_expectation_gradients(states, value_model, P_mat,  i, range_tensor=None, current_y=None):
    """
    Compute per-sample gradient estimates via Richardson extrapolation of central differences.
    Args:
        model: a network mapping states -> scalar values (V)
        states: tensor of shape [B, D]
        eps: finite-difference step size
    Returns:
        grads: tensor of shape [B, D]
    """
    B, D = states.shape
    grads = torch.zeros(B,P_mat.shape[0], D)
    eps = 1e-3
    # Precompute unit vectors
    eye = torch.eye(D, device=states.device, dtype=states.dtype)

    for i in range(D):
        e = eye[i:i+1]  # [1, D]
        # Big step
        vp = value_model(states + eps * e)['values']
        vm = value_model(states)['values']
        G_big = (vp - vm) / ( eps)  # [B]

        # Half step
        eps2 = eps * 0.5
        vp2 = value_model(states + eps2 * e)['values']
        vm2 = vm
        G_small = (vp2 - vm2) / eps2  # [B]

        # Richardson combination
        grads[..., i] = (4 * G_small - G_big) / 3
        expectation_grads = torch.einsum("byd,zy->bzd", grads, P_mat)  # [B, y, D]
    
    #Optional: pick current y for the expectation
    if current_y is not None:
        #i=torch.arange(expectation_grads.shape[0])
        expectation_grads = expectation_grads[i,current_y.long(),:]
    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = expectation_grads / range_tensor[ax,:]  # broadcast over D
    return expectation_grads
#Class to compute the FOC residuals
class FOCresidual:
    """Class to compute the FOC residual for the CRS model"""
    def __init__(self, bounds_processor, bounds_processor_inf, K, p, cc):
        self.bounds_processor = bounds_processor  # Store bounds_processor
        self.bounds_processor_inf = bounds_processor_inf
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
        self.rho_normalized = self.bounds_processor.normalize_dim(self.rho_grid,K_n).unsqueeze(1).requires_grad_(True)

        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        self.v_0 = tensor(self.v_grid[0],dtype=type)

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
    def take_expectation(self, x, prod_states):    
        if x.ndim == 2:  # If x is [B, D]
            x = torch.einsum("by,zy->bz", x, self.Z_trans_tensor)
            x = x[self.i,prod_states.long()]
        else:
            x = torch.einsum("byd,zy->bzd", x, self.Z_trans_tensor)
            x = x[self.i,prod_states.long(),:]            
        return x
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
    def future_values(self, fut_size, prod_states, fut_states, omega, v_prime, hiring, value_net):
        """Compute the expected value of the value function for the given states and sup"""
        #policies_norm[:,K_v] = self.bounds_processor.normalize(fut_states) #No need to normalize, these are already correct.
        #Need to attach policies to EW_star here. Use fut_states as a way to get future size here in the derivative. Alternatively, I can loop here a couple of times, updating size based on pc(EW_star) and then calling it again
        self.i = torch.arange(fut_states.shape[0])
        fut_states_adj = torch.cat((fut_size, omega),dim=-1)
        fut_states_adj = self.bounds_processor.normalize(fut_states_adj)
        value_output = value_net(fut_states_adj) #This is the value function output
        future_grad = get_expectation_gradients(fut_states_adj, value_net, self.Z_trans_tensor, self.i, range_tensor=self.bounds_processor.range,current_y = prod_states)
        ERho_star = self.take_expectation(value_output['values'],prod_states) #This is the value function output
        EJ_star = ERho_star
        for k in range(K_n-1):
            EJ_star += - omega[:,k] * fut_size[:,k+1] * v_prime[:,k] #This should all be sup shape
        #EW_star = EW_star / fut_size #Normalizing, because ∂ 𝒫(y,n_k,ρ_k,z_k)/∂ ρ_k = n_k W_k. So to get true W_k gotta divide by the size
        return  EJ_star, future_grad
    def values(self, states, prod_states, EJ_star, v_prime, pc, re, hiring, omega, future_grad):
        """Compute the value function for the given states and sup"""
        states_d = self.bounds_processor.denormalize(states)
        size = states_d[:, :K_n]
        #s_pos = states_d[:,1] > 0 #For wages, look only at the cases where we have a positive number of seniors 
        rho = states_d[:,K_n:]
        wages = torch.zeros(rho.shape[0],K_n) 
        wages[:,:1] = self.pref.inv_utility(self.v_0 - self.p.beta*((v_prime[:,:1]+re[:,:1])))
        wages[:,1:] = tensor(np.interp(rho.detach().numpy(),self.rho_grid,self.w_grid),dtype = type)
        worker_values = torch.zeros((states.shape[0],K_v), dtype=type) #This is the value function for the worker, given the wages and the promised values
        if K_n > 2:
            worker_values[:,:K_v-1] = self.pref.utility(wages[:,1:K_n-1]) + self.p.beta * (v_prime[:,:K_v-1] + re[:,:K_v-1])
        worker_values[:,K_v-1:] = self.pref.utility(wages[:,K_n-1:]) + self.p.beta * (v_prime[:,K_v-1:] + re[:,K_v-1:])
        
        tot_size = 0
        tot_wage = 0
        tot_rho_value = 0
        for k in range(K_n):
            tot_size += size[:,k]
            tot_wage += size[:,k] * wages[:,k]
            if k > 0:
                tot_rho_value += size[:,k] * rho[:,k-1] * worker_values[:,k-1]
        values = self.fun_prod[prod_states.detach().long()] * self.production(tot_size) - self.p.hire_c * hiring - \
            tot_wage + tot_rho_value + self.p.beta * EJ_star
        grad = torch.zeros((states.shape[0],K_n+K_v), dtype=type)
        #grad[:,:K_n] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,:] + self.p.beta * pc_star * (future_grad[:,1:K_n] - omega * v_prime)# notice that future_grad is the same for juniors and seniors, because both end up becoming seniors
        grad[:,:1] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,:1] + self.p.beta * pc[:,:1] * (future_grad[:,1:2] - omega[:,:1] * v_prime[:,:1])
        if K_n > 2:
            grad[:,1:K_n-1] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,1:K_n-1] + self.p.beta * pc[:,1:] * (future_grad[:,2:K_n] - omega[:,1:] * v_prime[:,1:]) + worker_values[:,:K_v-1] * rho[:,:K_v-1]

        grad[:,K_n-1:K_n] = (self.fun_prod[prod_states.detach().long()] * self.production_1d_n(tot_size)).unsqueeze(1) - wages[:,K_n-1:] + self.p.beta * pc[:,K_v-1:] * (future_grad[:,K_n-1:K_n] - omega[:,K_v-1:] * v_prime[:,K_v-1:]) + worker_values[:,K_v-1:K_v] * rho[:,K_v-1:K_v]    
        
        grad[:,K_n:] = worker_values * size[:,1:] #The gradient is multiplied by the size
        return values, grad
    def FOC_loss(self, states, omega, hiring, v_prime, future_grad):
        """Given the Value network and the sup network,
        Compute the FOC residuals for given set of states
        Requires: EW_star for each point, derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star ? Right?
        Note: for hirings and layoffs, will need to ensure the loss is zero when the sup is consistent: sep=0 => FOC<0, sep=1 => FOC>0 etc
        """
        # Derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star
        EJ_deriv = future_grad[:,1:K_n] - omega * v_prime #Should it be like this??? Because, if I write 𝒫(y,n_k,̃ρ_k) = max_{v_k} J(y,n_k,v_k) + ̃ρ_k v_k, then v_k has no role to play in the derivative wrt n_k!
        size=self.bounds_processor.denormalize_size(states[:,:K_n])
        #Get worker's search decisions and associated return
        re, pc = self.getWorkerDecisions(v_prime)
        _, pc_d = self.getWorkerDecisions(v_prime + self.deriv_eps) 
        # After computing pc and pc_d:
        assert not torch.isnan(pc).any(), "NaN in pc"
        assert not torch.isnan(pc_d).any(), "NaN in pc_d"
        log_diff = torch.zeros_like(v_prime)
        #log_diff[:] = torch.nan
        log_diff[pc > 0] = torch.log(pc_d[pc > 0]) - torch.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
        #print("min pc", pc.min().item()) 0.51 because search ain't perfect
        assert torch.all(log_diff >= 0)
        assert torch.all(~torch.isnan(log_diff))
        # After computing log_diff:
        assert not torch.isnan(log_diff).any(), "NaN in log_diff"
        #Neeed an updated foc that includes size, same as in my 2-tenure step model!!! Check how I write this theoretically in order to adapt. Gotta make sure
        inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*((v_prime[:,0]+re[:,0])))
        assert torch.all(inv_utility_1d > 0)
        # After computing inv_utility_1d:
        assert not torch.isnan(inv_utility_1d).any(), "NaN in inv_utility_1d"
        rhos = torch.cat([1/inv_utility_1d.unsqueeze(1), self.bounds_processor.denormalize_rho(states[:, K_n:])], dim=1) #This is the rho vector, with ρ_1 * n_1 in the first position and the rest of ρ_k * n_k in the rest of the positions
        #2 cases: the usual case and the senior case
        focs_rho = torch.zeros((states.shape[0],K_v), dtype=type) 
        #Usual case first:
        if K_n > 2:

            focs_rho[:,:K_v-1] = rhos[:,:K_n-2] - omega[:,:K_v-1] + EJ_deriv[:,:K_v-1] * (log_diff[:,:K_v-1] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
        #Senior case:
        focs_rho_2ndpart =  rhos[:,K_n-2:K_n-1] * size[:,K_n-2:K_n-1] + rhos[:,K_n-1:] * size[:,K_n-1:]          
        focs_fut = - omega[:,K_v-1:] + EJ_deriv[:,K_v-1:] * (log_diff[:,K_v-1:] / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
        focs_rho[:, K_v-1:] = focs_fut * (size[ :,K_n-2:K_n-1] + size[ :,K_n-1:]) + focs_rho_2ndpart    
        assert not torch.isnan(focs_rho).any(), "NaN in focs_rho"
        #Essentially, if we have no juniors, we wanna disable the bottom foc. if have no seniors and below seniors, we wanna disable the top foc.
        #To be fair though, the rest of the focs do not have size explicily, so maybe I just let them train regardless? 
        focs_rho[(size[:,-1]+size[:,-2]) <= 0, K_v-1:] = 0
        #Now hiring FOC
        focs_hire = self.p.beta * future_grad[:, 0] - self.p.hire_c
        focs_hire_sp = torch.relu(focs_hire)
        # but we only “apply” it when hiring ≤ 0, otherwise zero out
        FOC_hire_resid = torch.where(
            hiring <= 0,
            focs_hire_sp,
            focs_hire,)
        assert not torch.isnan(focs_hire).any(), "NaN in inv_utility_1d"
        return focs_rho, FOC_hire_resid
    def get_fut_size(self, states, hiring, v_prime):
        re, pc = self.getWorkerDecisions(v_prime) #Now v_prime will be multidimensional, giving values for each k∈K_v
        size = self.bounds_processor.denormalize_size(states[:,:K_n])
        fut_size = torch.zeros((states.shape[0],K_n), dtype=type)
        fut_size[:,0] = hiring
        if K_n > 2:
            fut_size[:,1:K_n - 1] = size[:,0:K_n - 2] * pc[:,0:K_v-1] #Future size, based on n'= n * pc(v').
        fut_size[:,K_n-1:K_n] = (size[:,K_n-2:K_n-1] + size[:,K_n-1:K_n]) * pc[:,K_v-1:] #Senior case: n'_K=(n_{K-1}+n_K) * pc
        #n_1 = ((size[:,0]+size[:,1]) * pc.squeeze(1)).unsqueeze(1)
        
        return fut_size, re, pc

def simulate(starting_states, sup_net, inf_net, bounds_processor, bounds_processor_inf, simulation_steps):
    """Simulate the firm path using the sup network
    Track the reached states and the corresponding values (values may be too hard to track)
    Output: set of (reached) states and corresponding values
    Args:
        starting_states: [B, D] — input states (normalized), requires_grad NOT required
        prod_states:     [B]   — production states (discrete), requires_grad NOT required
        value_net:       neural net mapping [B, D] → [B, num_y]
        sup_net:      neural net mapping [B, D] → [B, num_y]
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
        #    S[his_start:his_end,:] = states.repeat(p.num_z, 1) #We repeat the sup from the previous step   
        S[his_start:his_end,:] = states.repeat(p.num_z, 1)
        sup = sup_net(states) 
        hiring = sup['hiring'] #Shape [B,num_y]
        v_prime = sup['values']

        y_idx      = torch.arange(p.num_z,device=states.device).repeat(states.shape[0])

        _, pc = foc_optimizer.getWorkerDecisions(v_prime.view(his_size,K_v))
        sizes = bounds_processor.denormalize_size(states[:,:K_n])
        #Set up new states
        states = torch.zeros(his_size, D, dtype= type)
        states[:,0] = hiring.view(-1) #Future jun size, based on hiring
        if K_n > 2:
            states[:,1:K_n - 1] = sizes[:,0:K_n - 2].repeat_interleave(p.num_z,dim=0) * pc[:,0:K_v-1]
        states[:,K_n-1:K_n] = (((sizes[:,K_n-2:K_n-1]+sizes[:,K_n-1:]).squeeze(1).repeat(p.num_z)) * pc[:,K_v-1:K_v].squeeze(1)).unsqueeze(1)        
        states_inf = bounds_processor_inf.normalize((torch.cat([states[:,:K_n],v_prime.view(his_size,K_v)],dim=1)))
        omega = inf_net(states_inf)['omega'][torch.arange(states.shape[0]),y_idx,:]
        states[:,K_n:] =  omega # state ω_1 = ρ_1*n_1
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

def initialize(bounds_processor, bounds_processor_inf, state_dim, K_n, K_v, hidden_dims, learning_rate, weight_decay, pre_training_steps, num_epochs, minibatch_num):
    #Initializations:
    
    # Initialize value function neural network
    value_net = ValueFunctionNN(state_dim, p.num_z, hidden_dims)
    sup_net = PolicyNN(state_dim, p.num_z, K_v, hidden_dims, cc)
    inf_net = infNN(state_dim, p.num_z, K_v, hidden_dims, cc)

    # Initialize neural network optimizer
    # Use Ranger21 for all three networks:
    optimizer_value = RangerOptimizer(
        params=value_net.parameters(),
        lr=learning_rate[0],
        weight_decay=weight_decay[0],
        num_epochs=num_epochs,       # for built‑in warmup + scheduler
        num_batches_per_epoch=minibatch_num - 3,
        num_warmup_iterations=int(0.05 * num_epochs),  # 5% of training
        logging_active=False  # no logging for now
    )
    optimizer_sup = RangerOptimizer(
        params=sup_net.parameters(),
        lr=learning_rate[1],
        weight_decay=weight_decay[2],
        num_epochs=num_epochs,
        num_batches_per_epoch= (minibatch_num / 3) - 1,
        num_warmup_iterations=int(0.05 * num_epochs),  # 5% of training
        logging_active=False  # no logging for now
    )
    optimizer_inf = RangerOptimizer(
        params=inf_net.parameters(),
        lr=learning_rate[2],
        weight_decay=weight_decay[2],
        num_epochs=num_epochs,
        num_batches_per_epoch= 1,
        num_warmup_iterations=int(0.05 * num_epochs),  # 5% of training
        logging_active=False  # no logging for now
    )

    # Initialize FOC computer
    foc_optimizer = FOCresidual(bounds_processor, bounds_processor_inf, K=K_n, p=p, cc=None)    

    #Step 0: basic guess
    value_net, sup_net, inf_net, optimizer_value, optimizer_sup, optimizer_inf = pre_training(optimizer_value,optimizer_sup,optimizer_inf,value_net,sup_net,inf_net,foc_optimizer,bounds_processor, bounds_processor_inf, K_n, K_v, pre_training_steps)   

    return value_net, sup_net, inf_net, optimizer_value, optimizer_sup, optimizer_inf, foc_optimizer
def pre_training(optimizer_value,optimizer_sup, optimizer_inf, value_net,sup_net,inf_net, foc_optimizer,bounds_processor, bounds_processor_inf, K_n, K_v, pre_training_steps):
    rho_states = bounds_processor.normalize_rho(tensor(cc.rho_grid[:,ax],dtype=type))
    assert torch.all(rho_states[1:] > rho_states[:-1]), "States are not increasing"
    crs_Rho = foc_optimizer.simple_Rho.t()
    #Train also on the gradient
    target_W = torch.zeros_like(crs_Rho) + tensor(foc_optimizer.v_grid[ax,:], dtype=type).t()
    #Let the sup just be today's rho.
    target_omega =  torch.zeros_like(crs_Rho) + tensor(foc_optimizer.rho_grid[ax,:], dtype=type).t()
    assert not torch.isnan(crs_Rho).any(), "NaN in crs_Rho"
    assert not torch.isnan(target_omega).any(), "NaN in target_omega"
    assert not torch.isnan(target_W).any(), "NaN in target_W"
    grad_size = torch.zeros((crs_Rho.shape[0], crs_Rho.shape[1], K_n), dtype=type) #Gradient of the target W wrt size. Shape [B,num_y, K_n]
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
            if k == 0:
                target_values += crs_Rho[ax, 0, :] * size[:, ax, k] #Since the junior workers are at the bottom value
                grad_size[:,:, k] = crs_Rho[ax, 0, :]
            else:
                grad_size[:,:, k] = crs_Rho
                target_values += crs_Rho * size[:,ax,k] #final shape [B, num_y] 
        predicted_values = value_net(states)['values']
        grads = get_batch_gradients(states, value_net, p.num_z, bounds_processor.range) #This is the gradient of the value function wrt states
        predicted_W = grads[:,:,K_n:] #These are gradients wrt rho_k forall k besides the bottom one
        predicted_grads_size = grads[:,:,:K_n] #These are gradients wrt size, i.e. ∂W/∂n_k
        #Add gradient loss and monotonicity loss
        #violation = torch.relu(predicted_grads[:-1,:] - predicted_grads[1:,:])
        #mon_loss = (violation ** 2).mean() #This is the loss function that forces the gradient to be increasing
        value_loss = nn.MSELoss()(predicted_values, target_values) + nn.MSELoss()(predicted_W, target_W[...,ax]) +  nn.MSELoss()(predicted_grads_size, grad_size)
        value_loss.backward() #Backpropagation
        optimizer_value.step() #Update the weights
        optimizer_value.zero_grad()
        #Policy loss: very specific here bcs its not a FOC loss. EVEN THOUGH I COULD MAKE IT A FOC LOSS.
        sup = sup_net(states)
        predicted_values = sup['values']
        
        future_value_loss = nn.MSELoss()(predicted_values, target_W[...,ax])
        violation = torch.relu(sup['hiring'][:,:-1] - sup['hiring'][:,1:])#Should hire more in better states. Tbf the ρ' and EW* should also be increasing, no?
        mon_loss = (violation ** 2).mean()
        sup_loss = future_value_loss + mon_loss
        sup_loss.backward()
        optimizer_sup.step()
        optimizer_sup.zero_grad()
        predicted_omega = inf_net(states)['omega'] #I am not training hiring here, only omega
        inf_loss = nn.MSELoss()(predicted_omega, target_omega[...,ax] * size[:,ax,1:K_n])
        inf_loss.backward()
        optimizer_inf.step()
        optimizer_inf.zero_grad()
        optimizer_value.zero_grad()
    return value_net, sup_net, inf_net, optimizer_value, optimizer_sup, optimizer_inf
def train(state_dim, value_net, sup_net, inf_net, optimizer_value, optimizer_sup, optimizer_inf, foc_optimizer, bounds_processor, bounds_processor_inf, num_episodes=20, starting_points_per_iter=100, simulation_steps=5, 
    minibatch_num=8, λ=1.0, target_values=None, target_W=None, use_saved_nets = False):
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
        sup_net.load_state_dict(torch.load("trained_sup_function.pt"))
    #Initialize a target network
    target_value_net = copy.deepcopy(value_net)
    # Ensure it's not updated by the optimizer
    for param in target_value_net.parameters():
        param.requires_grad = False
    target_sup_net = copy.deepcopy(sup_net)
    for param in target_sup_net.parameters():
        param.requires_grad = False
    target_inf_net = copy.deepcopy(inf_net)
    for param in target_inf_net.parameters():
        param.requires_grad = False
    print("Training...")
    # Training loop
    for episode in tqdm(range(num_episodes)):
        #torch.autograd.set_detect_anomaly(True)
        #Starting states. First with a single junior and nothing else. State = (y, {1,0},{rho_min,rho_min})
        state_start = torch.zeros(state_dim,dtype=type)
        state_start[0] = bounds_processor.normalize_dim(1,0) # 1 junior worker
        #Or if randomized. 
        states= torch.rand(starting_points_per_iter, state_dim,dtype=type) 
        #Add the starting state
        states[0,:] = state_start
        #Simulate the firm path using the sup network
        with torch.no_grad():
            states, prod_states, fut_states  = simulate(states, target_sup_net, target_inf_net, bounds_processor, bounds_processor_inf, simulation_steps) #This is the set of states we will use to train the value function. 
        #Now append future states to the states so that I can mini-batch them together.
        states.clamp_(0.0, 1.0)
        #OR, restrict attention to states that have positive size. Otherwise, can't rly learn anything:
        sum_size = states.sum(dim=1) #This is the sum of all sizes in the batch
        pos_size = sum_size > 0 #This is the case where we have positive size. Otherwise, can't rly learn anything.
        comb_states = torch.zeros(states[pos_size,:].shape[0], state_dim, 3, dtype=type)
        comb_states[...,0] = states[pos_size,:]
        comb_states[...,1] = fut_states[pos_size,:]
        comb_states[...,2] = prod_states[pos_size].unsqueeze(-1)
        # Mini-batch the simulated data
        minibatch_size = np.floor(comb_states.shape[0]/minibatch_num).astype(int)
        minibatches = random_mini_batches(comb_states, minibatch_size)
        batch_index = 0
        print_check = 0
        for mb in minibatches:    
            #Detach all the arrays again
            states= mb[..., 0]
            fut_states = mb[...,1]
            prod_states = mb[..., -1, 2]
            
            batch_index += 1
            i = torch.arange(states.shape[0])
            if ((batch_index) % minibatch_num)==0: #Inf Update, happens only once per episode
                #print("Inf update")
                print_check += 1
                #Inf Loss: detach everything *except* omega
                with torch.no_grad(): # compute all sup‐outputs
                    pol = target_sup_net(states)
                    v_prime  = pol['values'][i,prod_states.long(),:]  
                    hiring  = pol['hiring'][i,prod_states.long()]
                    fut_size,_,_= foc_optimizer.get_fut_size(states, hiring, v_prime)
                
                optimizer_inf.zero_grad()
                states_inf = torch.cat((fut_size, v_prime), dim=1) # [B, D]
                states_inf = bounds_processor_inf.normalize(states_inf) # Normalize the states
                omega = inf_net(states_inf)['omega'][i,prod_states.long(),:]
                _, future_grad = foc_optimizer.future_values(fut_size = fut_size, prod_states=prod_states, fut_states = fut_states, omega=omega, v_prime = v_prime, hiring=hiring, value_net=target_value_net)  #Note that I am using the target value here!!!          
                inf_loss = nn.MSELoss()(future_grad[:,K_n:], v_prime * fut_size[:,1:])
                assert omega.requires_grad
                assert future_grad.requires_grad
                inf_loss.backward()
                optimizer_inf.step()
            else:
             if ((batch_index) % (minibatch_num/3))==0: #Sup update
                #print("Sup update")
                #Sup loss: FOC residuals
                optimizer_sup.zero_grad()   
                policies = sup_net(states)
                #Gotta now do wages, hiring, and values separately
                v_prime = policies['values'][i,prod_states.long(),:]
                hiring = policies['hiring'][i,prod_states.long()] 
                fut_size,_,_= foc_optimizer.get_fut_size(states, hiring, v_prime)
                #with torch.no_grad():
                states_inf = torch.cat((fut_size, v_prime), dim=1) # [B, D]
                states_inf = bounds_processor_inf.normalize(states_inf) # Normalize the states
                omega = target_inf_net(states_inf)['omega'][i,prod_states.long(),:]                
                assert (~torch.isnan(omega)).all() and (~torch.isnan(v_prime)).all() and (~torch.isnan(hiring)).all(), "sup returns NaN"
                _, future_grad = foc_optimizer.future_values(fut_size = fut_size, prod_states=prod_states, fut_states = fut_states, omega=omega, v_prime = v_prime, hiring=hiring, value_net=target_value_net)  #Note that I am using the target value here!!!          
                FOC_rho_resid,FOC_hire_resid = foc_optimizer.FOC_loss(states=states, omega=omega, hiring=hiring, v_prime=v_prime, future_grad=future_grad)
                FOC_wage_loss = nn.MSELoss()(FOC_rho_resid, torch.zeros_like(FOC_rho_resid))
                FOC_hire_loss =  nn.MSELoss()(FOC_hire_resid, torch.zeros_like(FOC_hire_resid))
                assert not torch.isnan(FOC_rho_resid).any(), "NaN in FOC_wage_loss"
                assert not torch.isinf(FOC_rho_resid).any(), "inf in FOC_wage_loss"                                
                assert not torch.isnan(FOC_hire_resid).any(), "NaN in FOC_hire_loss"
                assert not torch.isinf(FOC_hire_resid).any(), "inf in FOC_hire_loss"                
                sup_loss = FOC_wage_loss + FOC_hire_loss
                sup_loss.backward()
                optimizer_sup.step()
                if ((episode + 1) % (num_episodes/20) == 0) & (print_check <= 1):
                    #print(f"EW* norm: {EW_star.norm().item():.4f}")
                    print(f"EW* mean: {v_prime.mean().item():.4f}")      
             else: #Value function update
                #print("Val update")
                optimizer_value.zero_grad()
                with torch.no_grad():
                    policies = target_sup_net(states)
                    #Gotta now do wages, hiring, and values separately
                    v_prime = policies['values'][i,prod_states.long(),:]
                    hiring = policies['hiring'][i,prod_states.long()] 
                    fut_size,re,pc= foc_optimizer.get_fut_size(states, hiring, v_prime)
                    #with torch.no_grad():
                    states_inf = torch.cat((fut_size, v_prime), dim=1) # [B, D]                    
                    states_inf = bounds_processor_inf.normalize(states_inf) # Normalize the states
                    omega = target_inf_net(states_inf)['omega'][i,prod_states.long(),:]
                assert (~torch.isnan(omega)).all() and (~torch.isnan(v_prime)).all() and (~torch.isnan(hiring)).all(), "sup returns NaN"
                EJ_star, future_grad = foc_optimizer.future_values(fut_size = fut_size, prod_states=prod_states, fut_states = fut_states, omega=omega, v_prime = v_prime, hiring=hiring, value_net=target_value_net)  #Note that I am using the target value here!!!          
                target_values, target_grad = foc_optimizer.values(states=states, prod_states=prod_states, EJ_star=EJ_star.detach(), v_prime=v_prime.detach(), re=re.detach(), hiring=hiring.detach(), pc = pc.detach(), future_grad=future_grad.detach(), omega=omega.detach()) #Get the target values and gradients
                value_output = value_net(states)
                assert (~torch.isnan(value_output['values'])).all(), "value returns NaN"
                pred_values = value_output['values']
                pred_values = pred_values[i,prod_states.long()] #Get the values for the states in the minibatch
                #predicted_grad = get_batch_gradients(states, value_net,  range_tensor=bounds_processor.range)[:,:,:]
                predicted_grad = get_batch_gradients(states, value_net, policies['hiring'].shape[1], range_tensor=bounds_processor.range)
                predicted_grad = predicted_grad[i,prod_states.long(),:] #Get the values for the states in the minibatch
                value_loss = nn.MSELoss()(pred_values, target_values)
                value_grad_loss = nn.MSELoss()(predicted_grad, target_grad) #Get the value loss for the states in the minibatch
                tot_value_loss = (value_loss + λ * value_grad_loss)/(1+λ) #Combine the losses
                tot_value_loss.backward()
                optimizer_value.step()
        #Hard copy the target value at the end of every episode
        target_value_net.load_state_dict(value_net.state_dict(), strict=True)
        target_sup_net.load_state_dict(sup_net.state_dict(), strict=True)
        target_inf_net.load_state_dict(inf_net.state_dict(), strict=True)        
        # Print progress
        if (episode + 1) % (num_episodes/20) == 0 or episode == 0:
            print(f"Iteration {episode + 1}, Value Loss: {value_loss.item():.6f}, Value Grad Loss:  {value_grad_loss.item():.6f},FOC_wage_loss: {FOC_wage_loss.item():.6f}, FOC_hire_loss: {FOC_hire_loss.item():.6f} ,inf_loss: {inf_loss.item():.6f}" )
        if (episode + 1) % (num_episodes/5) == 0:            
            evaluate_plot_sup(value_net, sup_net, inf_net, bounds_processor, bounds_processor_inf, num_samples=1000)                
    return value_net, sup_net, inf_net

def evaluate_plot_sup(value_net, sup_net, inf_net, bounds_processor, bounds_processor_inf, num_samples=1000):
    """Evaluate the sup by sampling random states and plotting the results"""

    #states = bounds_processor.normalize(states)
    
    # Get sup outputs
    with torch.no_grad():
        # Sample random states
        states = torch.rand(num_samples, bounds_processor.lower_bounds.shape[0], dtype=type)
        policies = sup_net(states)
        v_prime = policies['values'][:,1,:]
        hiring = policies['hiring'][:,1]
        fut_size,_,_= foc_optimizer.get_fut_size(states, hiring, v_prime)        
        states_inf = torch.cat((fut_size, v_prime), dim=1) # [B, D]
        states_inf = bounds_processor_inf.normalize(states_inf) # Normalize the states
        omega = inf_net(states_inf)['omega'][:,1,:]
        values = value_net(states)['values']
        grads = get_batch_gradients(states, value_net, policies['hiring'].shape[1], range_tensor=bounds_processor.range)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, -1].detach().numpy(), v_prime[:,-1].detach().numpy(), alpha=0.5)
    plt.title("Policy Evaluation: Promised Values vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Promised Values")
    plt.grid()
    plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, -1].detach().numpy(), omega[:,-1].detach().numpy(), alpha=0.5)
    plt.title("Policy Evaluation: Wages vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Wages")
    plt.grid()
    plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, -1].detach().numpy(), values[:,1].detach().numpy(), alpha=0.5)
    plt.title("Value Evaluation: Values vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Values")
    plt.grid()
    plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, -1].detach().numpy(), grads[:,1,-1].detach().numpy(), alpha=0.5)
    plt.title("Grads Evaluation: Grads vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Rho Grads")
    plt.grid()
    plt.show()

def make_nan_hook(param_name):
    def hook(grad):
        if torch.isnan(grad).any():
            grad = torch.clamp(grad, -0.1, 0.1)
            print(f"Clamped NaNs in gradient of {param_name}")
    return hook

if __name__ == "__main__":
    # Define parameters
    K = 2 #Number of tenure steps
    #Number of states
    K_n = K #K size states
    K_v = K - 1 #K - 1 (ignoring bottom) value states
    K_q = K - 1 #K - 1 (ignoring bottom) quality states. Ignore them for now
    STATE_DIM = K_n + K_v # + K_q #Discrete prod-ty y as multiple outputs
    ACTION_DIM = K_v + 1 # + K_n  # omega + hiring + separations. BIG QUESTION: do I do my thing (firms internalizing finite K) first to check? And only then move on? That's longer, but def more safe
    HIDDEN_DIMS = [64,64]  # Basic architecture. Basically every paper has 2 inner layers, can make them wider though

    #pref = Preferences(input_param=p_crs)
    cc=ContinuousContract(p_crs()) 
    cc_J,cc_W,cc_Wstar,omega = cc.J(0) 
    target_values = tensor(cc_J + cc.rho_grid[ax,:] * cc_W, dtype=type)
    target_W = tensor(cc_W, dtype=type)
    #NORMALIZE EVERYTHING!!!
    #Gotta establish bounds flexible with K
    LOWER_BOUNDS = [0] * K_n + [cc.rho_grid[0]] * K_v
    UPPER_BOUNDS = [20] * (K_n - 1) + [40] + [cc.rho_grid[-1]] * K_v
    # now the “inf” versions use v_grid instead of rho_grid:
    LOWER_BOUNDS_inf = [0] * K_n + [cc.v_grid[0]] * K_v
    UPPER_BOUNDS_inf = [20] * (K_n - 1) + [40] + [cc.v_grid[-1]] * K_v

    num_episodes= 20000
    minibatch_num = 9
    #Initialize
    bounds_processor_sup = StateBoundsProcessor(LOWER_BOUNDS,UPPER_BOUNDS)
    bounds_processor_inf = StateBoundsProcessor_inf(LOWER_BOUNDS_inf,UPPER_BOUNDS_inf)

    value_net, sup_net, inf_net, optimizer_value, optimizer_sup, optimizer_inf, foc_optimizer = initialize(bounds_processor_sup, bounds_processor_inf, STATE_DIM, 
    K_n, K_v, HIDDEN_DIMS, learning_rate=[5e-4,3e-4,1e-4], weight_decay = [5e-5, 3e-5, 1e-5], pre_training_steps=0, num_epochs=num_episodes, minibatch_num=minibatch_num)
    
    # Train value function
    print("Training value function...")
    beg=time()
    trained_value, trained_sup, trained_inf = train(
    STATE_DIM, value_net, sup_net, inf_net, optimizer_value, optimizer_sup, optimizer_inf, foc_optimizer, bounds_processor_sup,bounds_processor_inf, 
        num_episodes=num_episodes,
        starting_points_per_iter=20,
        simulation_steps=5,
        minibatch_num=minibatch_num, λ=5.0,
        target_values=target_values.t(), target_W=target_W.t(), use_saved_nets = False
    )
    print("Training time:", time()-beg)

    # Evaluate trained model
    evaluate_plot_sup(trained_value, trained_sup, trained_inf, bounds_processor_sup, bounds_processor_inf, num_samples=1000)
    #evaluate_value_function(trained_value, trained_sup, p, LOWER_BOUNDS, UPPER_BOUNDS,cc,target_values,cc_W,omega)

    # Save the model
    torch.save(trained_value.state_dict(), "trained_value_function.pt")
    torch.save(trained_sup.state_dict(), "trained_sup_function.pt")
    torch.save(trained_inf.state_dict(), "trained_inf_function.pt")
    print("Model saved")
