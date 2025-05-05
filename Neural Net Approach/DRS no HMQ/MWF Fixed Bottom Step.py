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
        return (states - self.lower_bounds) / self.range
        #Example: lower-upper is [0,1]. So normalize(0.5) = 2 * (0.5 - 0) /1 -1 = 0. Ok correct
        #Another example. lower-upper is [0,30]. Sn normalize 15= 2 * 15 / 30 -1 = 0 Ok good.
        # And normalize (20) = 40/30 - 1 = 1/3 yup
        # Now denormalize(1/3) = 0.5 ( 1/3 +1 ) * 30 + 0 = 2/3*30 = 20

        #Normalizing to (0,1) now. Checking: 15 in (0,30) is 15/30=0.5. 15 in (10,20) is 5/10=0.5
        
    def denormalize(self, normalized_states):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range + self.lower_bounds #Check: 0.5 * 10 + 10= 15.
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
            layers.append(nn.Softplus())  # Softplus activation function: allows slightly negative values
            input_dim = hidden_dim
        
        # Final layer: one output per discrete state y'
        layers.append(nn.Linear(input_dim, num_y)) #was input_dim instead of 16
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
class PolicyNN(nn.Module):
    """Neural network to approximate a multi-dimensional policy:
       - wages: multiple values per productivity state y across a predefined set of K_v
       - hiring decision: probability per productivity state y
    """
    def __init__(self, state_dim, num_y, K_v, hidden_dims=[40, 30, 20, 10]):
        super(PolicyNN, self).__init__()
        self.K_v = K_v #Number of Value-related policies
        self.num_y = num_y

        # shared trunk
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Softplus())
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
    def forward(self, x):
        # x: [B, state_dim]
        B = x.size(0)
        features = self.trunk(x)  # [B, hidden_dims[-1]]

        # wages: reshape to [B, num_y, num_Kv]
        wages_flat = self.wage_head(features)            # [B, num_y * num_Kv]
        wages = wages_flat.view(B, self.num_y, self.K_v)  # [B, num_y, num_Kv]

        # values: reshape to [B, num_y, num_Kv]
        values_flat = self.wage_head(features)            # [B, num_y * num_Kv]
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
def get_batch_gradients(states, value_model,  range_tensor=None):
    """
    Computes per-sample gradient of V(y, s) for all y ∈ {0, 1, ..., num_y-1}

    Args:
        states:        [B, D] — input states (normalized), requires_grad NOT required. B is batch size, D is state dimension
        value_model:   neural net mapping [B, D] → [B, num_y]
        P_mat:         [num_y, num_y] — transition matrix with rows P(y → y′)
        range_tensor:  [D] or scalar, optional — rescale gradients (e.g., if states were normalized)

    Returns:
        expectation_grads: [B, num_y, D] — ∇_s V(y, s) for each y or fixed y
    """

    B, D = states.shape
    #num_y = P_mat.shape[0]

    # Detach any prior graph, ensure float precision
    states = states.detach().requires_grad_(True)  # [B, D]

    # Wrap the model to handle single input vector s: [D]
    def model_single_input(s_vec):
        s_in = s_vec.unsqueeze(0)        # [1, D]
        return value_model(s_in).squeeze(0)  # [num_y]

    # Compute full Jacobian: [B, num_y, D]
    jac_fn = vmap(jacrev(model_single_input))
    jacobian = jac_fn(states)  # ∂V(y', s_b)/∂s_b  — shape: [B, num_y, D]

    # Multiply by P^T to get: ∇_s E_{y'|y} V(y', s)
    # jacobian: [B, y', D];     P^T: [y', y]
    #expectation_grads = torch.einsum("byd,zy->bzd", jacobian, P_mat.T)  # [B, y, D]

    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        jacobian = jacobian / range_tensor  # broadcast over D

    return jacobian  # [B, y, D] or [B, D] if current_y is not None

def get_expectation_gradients(states, value_model, P_mat,  range_tensor=None, current_y=None):
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

    # Detach any prior graph, ensure float precision
    states = states.detach().requires_grad_(True)  # [B, D]

    # Wrap the model to handle single input vector s: [D]
    def model_single_input(s_vec):
        s_in = s_vec.unsqueeze(0)        # [1, D]
        return value_model(s_in).squeeze(0)  # [num_y]

    # Compute full Jacobian: [B, num_y, D]
    jac_fn = vmap(jacrev(model_single_input))
    jacobian = jac_fn(states)  # ∂V(y', s_b)/∂s_b  — shape: [B, num_y, D]

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
        self.deriv_eps = 1e-4 # step size for derivative
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
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)
        # Normalize rho_grid to tensor for model input
        self.rho_normalized = self.bounds_processor.normalize(tensor(self.rho_grid, dtype=torch.float32)).unsqueeze(1).requires_grad_(True)

        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        self.v_0 = self.v_grid[0]

        self.simple_J = torch.zeros((self.p.num_z,self.p.num_v))
        self.simple_J= (self.fun_prod[:,ax] -self.pref.inv_utility(tensor(self.v_grid[ax,:]*(1-self.p.beta),dtype = type)))/ (1-self.p.beta)
        self.simple_Rho = self.simple_J + self.rho_grid[ax,:] * self.v_grid [ax,:]#We do indeed need to work with Rho here since we're taking W via its derivatives
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
        return np.power(sum_n, self.p.prod_alpha)
    def getWorkerDecisions(self, EW1, employed=True): #Andrei: Solves for the entire matrices of EW1 and EU
        """
        :param EW1: Expected value of employment
        :param EU:  Expected value of unemployment
        :param employed: whether the worker is employed (in which case we multiply by efficiency)
        :return: pe,re,qi search decision and associated return, as well as quit decision.
        """
        pe, re = self.js.solve_search_choice(EW1) #Uses the job search array to solve for the search choice
        assert (~torch.isnan(pe)).all(), "pe is not NaN"
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
    def future_values(self, prod_states, fut_states, rho_star, value_net):
        """Compute the expected value of the value function for the given states and policy"""
        #policies_norm[:,K_v] = self.bounds_processor.normalize(fut_states) #No need to normalize, these are already correct.
        #Need to attach policies to EW_star here. Use fut_states as a way to get future size here in the derivative. Alternatively, I can loop here a couple of times, updating size based on pc(EW_star) and then calling it again
        fut_size = fut_states[:,:K_n]
        future_grad = get_expectation_gradients(torch.cat((fut_size,self.bounds_processor.normalize(rho_star)),dim=-1), value_net, self.Z_trans_tensor, range_tensor=self.bounds_processor.range,current_y = prod_states)
        EW_star = future_grad[:,K_n:]
        
        # Above I am computng the gradient, that is EW_star * n1_star!!! ALTERNATIVELY, I could make 𝒫(y,n_k,ρ_k*n_k,z_k) to not have to deal with this. Should I? 
        # YES!!! BECAUSE ONLY THAT WAY I CAN ACTUALLY TRACK EW_STAR AT CLOSE TO ZERO VALUES. How to do this then? Ah
        #EW_star = get_expectation_gradients(fut_states, value_net, self.Z_trans_tensor, range_tensor=self.bounds_processor.range,current_y = prod_states)[:,0]
        V_all = value_net(fut_states)   
        ERho_star= torch.einsum("by,zy->bz", V_all, self.Z_trans_tensor) #Should this be transposed? Doesn't matter now but will later.
        i=torch.arange(fut_states.shape[0])
        ERho_star = ERho_star[i,prod_states.long()]
        EJ_star = ERho_star - rho_star.squeeze(1) * fut_size[:,1:] * EW_star #This should all be policy shape
        return ERho_star, EJ_star, EW_star, future_grad     
    def values(self, states, prod_states, EJ_star, EW_star, pc_star, re_star):
        """Compute the value function for the given states and policy"""
        states = self.bounds_processor.denormalize(states)
        wages = np.interp(states[:,0].detach().numpy(),self.rho_grid,self.w_grid)
        worker_values = tensor(self.pref.utility(wages),dtype=type) + self.p.beta * (EW_star + re_star)
        
        values = self.fun_prod[prod_states.detach().long()] - tensor(wages,dtype=type) + states.squeeze(1) * worker_values
        #values = tensor(values, dtype=type) #Convert to tensor
        values+= self.p.beta * pc_star * EJ_star 
        #print("Diff btw two versions", torch.abs(pc_star*EJ_star - (pc_star*ERho_star - policies.unsqueeze(1) * pc_star * EW_star) ).mean().item())
        return values, worker_values    
    def FOC_loss(self, states, rho_star, pc, re, EW_star, future_grad):
        """Given the Value network and the policy network,
        Compute the FOC residuals for given set of states
        Requires: EW_star for each point, derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star ? Right?
        Note: for hirings and layoffs, will need to ensure the loss is zero when the policy is consistent: sep=0 => FOC<0, sep=1 => FOC>0 etc
        """
        # Derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star
        EJ_deriv = future_grad[:,1:K_n] - rho_star * EW_star
        size=states[:,:K_n]
        #Get worker's search decisions and associated return
        _, pc_d = self.getWorkerDecisions(EW_star + self.deriv_eps) 
        log_diff = torch.zeros_like(EW_star)
        log_diff[:] = torch.nan
        log_diff[pc > 0] = torch.log(pc_d[pc > 0]) - torch.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
        #Neeed an updated foc that includes size, same as in my 2-tenure step model!!! Check how I write this theoretically in order to adapt. Gotta make sure
        inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*((EW_star+re)))
        foc_2ndpart = - size[:, 1:] *states[:, K_n:] -   size[:,:1]  / inv_utility_1d                         
        focs = rho_star - EJ_deriv * (log_diff / self.deriv_eps) #first dim is productivity, second is future marg utility, third is today's margial utility
        focs = focs*(size[ :,:1] + size[ :,1:]) + foc_2ndpart        
        #focs = rho_star[:,0] - self.bounds_processor.denormalize(states[:,0]) - EJ_star * log_diff / self.deriv_eps  #In the optimum this is zero
        return focs
    def wrapper(self,states, prod_states, policies, EJ_star, EW_star, re, pc):

        #FOC loss for each state
        focs = self.FOC_loss(states, policies, pc, EJ_star, EW_star)

        #Calculate value for each state
        values, worker_values = self.values(states, prod_states, EJ_star, EW_star, pc, re)
        
        return focs, values, worker_values
    def initiation(self, prod_states, fut_states, rho_star, hiring, value_net):
        _, EJ_star, EW_star, future_grad = self.future_values(prod_states, fut_states, rho_star, value_net)
        #Although we do have a guess of fut size, still best to update
        re, pc = self.getWorkerDecisions(EW_star)

        return EJ_star, EW_star, future_grad, re, pc        


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
    B = starting_states.shape[0] #aka initial batch size
    D = starting_states.shape[1]
    his_size_total=0
    for t in range(simulation_steps):
        his_size_total+= B * (p.num_z**(t+1))
    #his_size_total = p.num_z ** (1+simulation_steps) #How many states we can reach in simulation_steps steps
    #discount = 1.0
    S = torch.zeros(his_size_total, D, dtype=type) #Vector all reached states
    Fut_states = torch.zeros(his_size_total, D, dtype=type) #Vector off all reached policies. It's just policies for each state (productivity alrdy included via P)
    #V = torch.zeros(his_size_total, dtype=type) #Vector off all reached values. It's just values for each state (productivity alrdy included via P)
    P = torch.arange(S.shape[0]) % p.num_z #Vector off all production states

    his_start = 0
    his_end = his_start + p.num_z * B
    his_size = his_end - his_start
    for t in range(simulation_steps):
        if t==0:
            states = starting_states
        else:
            S[his_start:his_end,:] = states.repeat(p.num_z, 1) #We repeat the policy from the previous step   
        S[his_start:his_end,:] = states.repeat(p.num_z, 1)
        policy = policy_net(states) 
        hiring = policy['hiring'] #Shape [B,num_y]
        values = policy['values']
        wages = policy['wages']
        #Fut_val[his_start:his_end,:, :] = values.view(B*p.num_z,D) #Given today's y and state S, we get ρ*(S,y)
        #V[his_start:his_end] = value_net(states).view(-1)
        #Get probability of staying... wait. In that case, if I wanna do it later, I can compile the entire W = u(w_state_today) + beta * (EW_star=future_value + re_star=re from getWorkerDecisions)
        _, pc = foc_optimizer.getWorkerDecisions(values.view(his_size,K_v))
        #Set up new states
        states = torch.zeros(his_size, D, dtype= type)
        states[:,1:K_n] = ((states[:,0]+states[:,1]) * pc.squeeze(1)).unsqueeze(1)#Future size, based on n'= n * pc(v') * (1 - s). Extreme case here as n'_1=(n'_0+n'_1) * pc
        states[:,0] = hiring.view(-1) #Future jun size, based on hiring
        states[:,K_n:] = bounds_processor.normalize(wages.view(his_size,K_v))
        Fut_states[his_start:his_end,:] = states
        his_start = his_end
        his_end = his_start + his_size * p.num_z
        his_size = his_end - his_start
    assert (his_start == S.shape[0])
    #Append P to S. That way, when I sample minibatches, I can just sample S and P together.
    #S=torch.cat((S, P.unsqueeze(1)), dim=1)

    return S, P, Fut_states #Doing values here may be not as efficient since some of them may not even be sampled.

def initialize(bounds_processor, state_dim, K_n, K_v, hidden_dims, learning_rate, pre_training_steps):
    #Initializations:
    
    # Initialize value function neural network
    value_net = ValueFunctionNN(state_dim, p.num_z, hidden_dims)
    policy_net = PolicyNN(state_dim, p.num_z, K_v, hidden_dims)
    # Initialize neural network optimizer
    optimizer_value = optim.Adam(value_net.parameters(), lr=learning_rate[0])
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=learning_rate[1])

    # Initialize FOC computer
    foc_optimizer = FOCresidual(bounds_processor, K=K_n, p=p, cc=None)    

    #Step 0: basic guess
    value_net, policy_net, optimizer_value, optimizer_policy = pre_training(optimizer_value,optimizer_policy,value_net,policy_net,foc_optimizer,bounds_processor, K_n, K_v, pre_training_steps)   

    return value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer
def pre_training(optimizer_value,optimizer_policy,value_net,policy_net,foc_optimizer,bounds_processor, K_n, K_v, pre_training_steps):
    rho_states = bounds_processor.normalize(tensor(cc.rho_grid,dtype=type))
    assert torch.all(rho_states[1:] > rho_states[:-1]), "States are not increasing"
    crs_Rho=tensor(foc_optimizer.simple_Rho, dtype=type).t()
    #Train also on the gradient
    target_W = torch.zeros_like(crs_Rho) + tensor(foc_optimizer.v_grid[ax,:], dtype=type).t()
    #Let the policy just be today's rho.
    target_wages =  torch.zeros_like(crs_Rho) + tensor(foc_optimizer.rho_grid[ax,:], dtype=type).t()

    #Now randomize size... wait HOW!!! I don't have an upper bound for size!
    size = 20 * torch.rand(rho_states.shape[0], K_n,dtype=type) #Random sizes for each size state. Shape [B,K_n]
    states = torch.zeros(rho_states.shape[0],K_n+K_v)
    states[:,:K_n] = size #First size states, then rho, then quality
    states[:,K_n:] = rho_states.unsqueeze(1) 
    target_values = torch.zeros_like(crs_Rho,dtype=type)
    for k in range(K_n):
        target_values += crs_Rho * size[:,ax,k] #final shape [B, num_y] 
    #Multiply the target gradient by size
    target_W_grad = target_W[...,ax] * size[:,ax,1:] #Careful about the dimensions here. At the end this should be [B,num_y, D], where D=K_n
    print("Pre-training...")
    for _ in range(pre_training_steps):
        predicted_values = value_net(states)
        predicted_grads = get_batch_gradients(states, value_net,  range_tensor=bounds_processor.range)[:,:,:K_v] #These are gradients wrt rho_k forall k besides the bottom one
        policy = policy_net(states)
        predicted_wages = policy['wages'] #I am not training hiring here, only rho_star
        #Add gradient loss and monotonicity loss
        #violation = torch.relu(predicted_grads[:-1,:] - predicted_grads[1:,:])
        #mon_loss = (violation ** 2).mean() #This is the loss function that forces the gradient to be increasing
        value_loss = nn.MSELoss()(predicted_values, target_values) + nn.MSELoss()(predicted_grads, target_W_grad) 
        #Policy loss: very specific here bcs its not a FOC loss. EVEN THOUGH I COULD MAKE IT A FOC LOSS.
        predicted_values = policy['values']
        wage_loss = nn.MSELoss()(predicted_wages, target_wages[...,ax])
        future_value_loss = nn.MSELoss()(predicted_values, target_W[...,ax])
        policy_loss = wage_loss + future_value_loss
        optimizer_value.zero_grad()
        optimizer_policy.zero_grad()
        value_loss.backward() #Backpropagation
        optimizer_value.step() #Update the weights
        policy_loss.backward()
        optimizer_policy.step()
    return value_net, policy_net, optimizer_value, optimizer_policy
def train(state_dim, value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer, bounds_processor, num_episodes=20, starting_points_per_iter=100, simulation_steps=5, 
    minibatch_size=512, λ=1.0, target_values=None, target_W=None):
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
    #Initialize a target network
    target_value_net = copy.deepcopy(value_net)
    # Ensure it's not updated by the optimizer
    for param in target_value_net.parameters():
        param.requires_grad = False

    print("Training...")
    # Training loop
    for episode in tqdm(range(num_episodes)):
        #Starting states... tf do I do here now? Start with completely random ones?
        #AT LEAST I should have firms starting out. Aka with a single junior and nothing else. State = (y, {1,0},{rho_min,rho_min})
        state_start = torch.zeros(state_dim,dtype=type)#.requires_grad_(True)
        state_start[0] = 1 # 1 junior worker
        #Or if randomized. 
        states= torch.rand(starting_points_per_iter, state_dim,dtype=type)
        #Multiply starting size by 20
        states[:,:K_n] = states[:,:K_n] * 20
        #Combine these two: always a junior state + some random states here and there
        states[0,:] = state_start
        #Simulate the firm path using the policy network
        with torch.no_grad():
            states, prod_states, fut_states  = simulate(states, policy_net, bounds_processor, simulation_steps) #This is the set of states we will use to train the value function. 
        #States here includes productivity state as the last column!
        #Now append future states to the states so that I can mini-batch them together.
        comb_states = torch.zeros(states.shape[0], state_dim, 3, dtype=type)
        comb_states[...,0] = states
        comb_states[...,1] = fut_states
        comb_states[...,2] = prod_states[:,ax]
        # Mini-batch the simulated data
        minibatches = random_mini_batches(comb_states, minibatch_size)
        batch_index = 0
        for mb in minibatches:    
            #Detach all the arrays again
            states= mb[..., 0]
            fut_states = mb[...,1]
            prod_states = mb[:, -1, 2]
            batch_index += 1
            i = torch.arange(states.shape[0])
            policies = policy_net(states.requires_grad_(True))
            #Gotta now do wages, hiring, and values separately
            rho_star = policies['wages'][i,prod_states.long(),:]
            values = policies['values'][i,prod_states.long(),:]
            hiring = policies['hiring'][i,prod_states.long()] 
            EJ_star, EW_star, future_grad, re, pc = foc_optimizer.initiation(prod_states=prod_states, fut_states = fut_states, rho_star=rho_star, hiring=hiring, value_net=target_value_net)  #Note that I am using the target value here!!!          

            if ((episode+batch_index) % 4)==1: #Policy Update
                FOC_resid = foc_optimizer.FOC_loss(states=states, rho_star=rho_star, pc=pc, re=re, EW_star=EW_star, future_grad=future_grad)
                FOC_loss = nn.MSELoss()(FOC_resid, torch.zeros_like(FOC_resid))
                value_loss = nn.MSELoss()(values,EW_star) #Loss the predicted EW_star via values and the "actual" EW_star (keep in mind that, in calculating actual EW_star, we still keep in mind the size guess based on the fut_values guess, see Simulation())
                loss = FOC_loss + value_loss #Get the total loss for the states in the minibatch
                loss.backward()
                optimizer_policy.step()
                optimizer_policy.zero_grad()
            else: #Value function update
                target_values, target_W = foc_optimizer.values(states=states, prod_states=prod_states, EJ_star=EJ_star, EW_star=EW_star, pc_star=pc, re_star=re)
                pred_values = value_net(states)
                pred_values = pred_values[i,prod_states.long()] #Get the values for the states in the minibatch
                predicted_W = get_batch_gradients(states, value_net,  range_tensor=bounds_processor.range)[:,:,0]
                predicted_W = predicted_W[i,prod_states.long()] #Get the values for the states in the minibatch
                value_loss = nn.MSELoss()(pred_values, target_values) + nn.MSELoss()(predicted_W, target_W) #Get the value loss for the states in the minibatch
                loss = value_loss
                loss.backward()
                optimizer_value.step()
                optimizer_value.zero_grad()

        #Hard copy the target value at the end of every episode
        target_value_net.load_state_dict(value_net.state_dict(), strict=True)
        # Print progress
        if (episode + 1) % (num_episodes/10) == 0 or episode == 0:
            print(f"Iteration {episode + 1}, Loss: {loss.item():.6f} , Value Loss: {value_loss.item():.6f}, FOC_loss: {FOC_loss.item():.6f}" )
    
    return value_net, policy_net

def evaluate_value_function(model, policy, p, lower_bounds,upper_bounds,cc,cc_Rho,cc_W,rho_star):
    """
    Evaluate the trained value function on test points
    
    Args:
        model: Trained value function model
        num_test_points: Number of test points
        state_dim: Dimension of state space
    """

    bounds_processor = StateBoundsProcessor(lower_bounds,upper_bounds)
    # Generate random test states
    #test_states = torch.randn(num_test_points, state_dim)
    test_states = bounds_processor.normalize(tensor(cc.rho_grid, dtype=torch.float32)).unsqueeze(1).requires_grad_(True)
    # Evaluate model
    print(test_states[0,0],test_states[-1,0])
    values = model(test_states)[:,p.z_0-1]
    Z_trans_mat = createPoissonTransitionMatrix(p.num_z, p.z_corr)
    Z_trans_tensor = tensor(Z_trans_mat, dtype=torch.float32)
    EW=get_batch_gradients(test_states, model,  range_tensor=bounds_processor.range)[:,p.z_0-1,0].detach().numpy()
    policies = policy(test_states).detach().numpy()
    #EW_star=get_expectation_gradients(policies, model, Z_trans_tensor ,range_tensor=bounds_processor.range)[:,p.z_0-1,0].detach().numpy()    
    #EW_star_NN=get_expectation_gradients(test_states, model, Z_trans_tensor, bounds_processor.range)[:,p.z_0-1,0].detach().numpy() #This is just W, not EW_star!!! Bcs this is just a derivative of Rho wrt the rho_grid, which gives us today's W
    
    # Print results
    #print("\nValue function evaluation on test states:")
    #for i in range(min(5, num_test_points)):
    #    print(f"State {i+1}: Value = {values[i].item():.4f}")
    #Plot results
    plt.figure(figsize=(16, 6))
    plt.subplot(1,2,1)
    plt.plot(cc.rho_grid, cc_Rho[p.z_0-1,:], label = "VFI")
    plt.plot(cc.rho_grid, values.detach(), label = "NN")    
    #Plot the policy functions
    plt.subplot(1,2,2)
    plt.plot(cc.rho_grid, cc_W[p.z_0-1,:], label = "VFI")
    plt.plot(cc.rho_grid, EW, label = "NN")    
    plt.show()

    #plt.subplot(1,2,3)
    plt.plot(cc.rho_grid, rho_star[p.z_0-1,:], label = "VFI")
    plt.plot(cc.rho_grid, policies[:,p.z_0-1], label = "NN")    
    plt.show()    

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

    LOWER_BOUNDS = [cc.rho_grid[0]]
    UPPER_BOUNDS = [cc.rho_grid[-1]] #Ideally this should come from fun_prod.max. Either way, these are bounds ONLY for wage. Not for size.
    
    #Initialize
    bounds_processor_wage = StateBoundsProcessor(LOWER_BOUNDS,UPPER_BOUNDS)
    value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer = initialize(bounds_processor_wage,STATE_DIM, K_n, K_v, HIDDEN_DIMS,
                                                                                         learning_rate=[0.01,0.01],pre_training_steps=1)
    
    # Train value function
    print("Training value function...")
    beg=time()
    trained_value, trained_policy = train(
    STATE_DIM, value_net, policy_net, optimizer_value, optimizer_policy, foc_optimizer, bounds_processor_wage,
        num_episodes=10000,
        starting_points_per_iter=10,
        simulation_steps=4,
        minibatch_size=300, λ=λ,
        target_values=target_values.t(), target_W=target_W.t()
    )
    print("Training time:", time()-beg)

    # Evaluate trained model
    #evaluate_value_function(trained_value, trained_policy, p, LOWER_BOUNDS, UPPER_BOUNDS,cc,target_values,cc_W,rho_star)

    # Save the model
    torch.save(trained_value.state_dict(), "trained_value_function.pt")
    torch.save(trained_policy.state_dict(), "trained_policy_function.pt")    
    print("Model saved")
