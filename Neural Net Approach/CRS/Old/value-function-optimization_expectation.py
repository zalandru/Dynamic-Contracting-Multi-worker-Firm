import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap
from tqdm import tqdm
import numpy as np
from scipy.stats import lognorm as lnorm
from ContinuousContract import ContinuousContract
from primitives import Parameters
import opt_einsum as oe
from primitives import Preferences
from probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix
from search import JobSearchArray
import matplotlib.pyplot as plt
from time import time
tensor = torch.tensor
p = Parameters()
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
class StateBoundsProcessor:
    def __init__(self, lower_bounds, upper_bounds):
        """
        Initialize with lower and upper bounds for each state dimension
        
        Args:
            lower_bounds: List or tensor of lower bounds [x_1, x_2, ..., x_20]
            upper_bounds: List or tensor of upper bounds [y_1, y_2, ..., y_20]
        """
        self.lower_bounds = tensor(lower_bounds, dtype=torch.float32)
        self.upper_bounds = tensor(upper_bounds, dtype=torch.float32)
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
            layers.append(nn.Softplus())  # SiLU activation function
            input_dim = hidden_dim
        
        # Final layer: one output per discrete state y'
        layers.append(nn.Linear(input_dim, num_y)) #was input_dim instead of 16
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
# Learning rate scheduler example
def get_lr_scheduler(optimizer, total_iterations):
    """Create a learning rate scheduler that decays over time"""
    lambda_fn = lambda epoch: max(0.001, 1.0 - 2 * epoch / total_iterations)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
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
    expectation_grads = torch.einsum("byd,zy->bzd", jacobian, P_mat.T)  # [B, y, D]

    #Optional: pick current y for the expectation
    if current_y is not None:
        gr = torch.zeros(B, D, dtype=torch.float32)  # [B, D]
        # Select the gradients for the current y
        for iz in range(num_y):
            gr[current_y == iz,:] = expectation_grads[current_y == iz, iz, :]
            #expectation_grads[current_y == iz] = expectation_grads[:, iz, :]
        expectation_grads = gr
    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = expectation_grads / range_tensor  # broadcast over D

    return expectation_grads  # [B, y, D] or [B, D] if current_y is not None


class FOCOptimizer:
    """
    Class to solve first-order conditions given a state and value function
    This is a placeholder - you'll need to implement actual FOC logic
    """
    def __init__(self, state_dim, action_dim, value_function_model, bounds_processor, parameters=None, js=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bounds_processor = bounds_processor  # Store bounds_processor

        self.p = parameters
        self.deriv_eps = 1e-4 # step size for derivative
        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid

        # Production Function in the Model
        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = self.p.u_bf_m

        # Transition matrices
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)
        self.Z_trans_tensor = tensor(self.Z_trans_mat, dtype=torch.float32)
        # Value Function Setup
        self.J_grid   = -10 * np.ones((self.p.num_z,self.p.num_v)) #grid of job values, first productivity, then starting value, then tenure level
        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max(), self.p.num_v )
        self.rho_grid=1/self.pref.utility_1d(self.w_grid)
        # Normalize rho_grid to tensor for model input
        self.rho_normalized = self.bounds_processor.normalize(tensor(self.rho_grid, dtype=torch.float32)).unsqueeze(1).requires_grad_(True)
        test_states = bounds_processor.normalize(tensor(cc.rho_grid, dtype=torch.float32)).unsqueeze(1).requires_grad_(True)
        print("Rho grid diff", np.max(np.abs(self.rho_normalized.detach().numpy()-test_states.detach().numpy())))
        #print(self.rho_normalized.min(),self.rho_normalized.max())
        #Gotta fix the tightness+re functions somehow. Ultra simple J maybe?
        self.v_grid=np.linspace(np.divide(self.pref.utility(self.unemp_bf),1-self.p.beta), np.divide(self.pref.utility(self.fun_prod.max()),1-self.p.beta), self.p.num_v ) #grid of submarkets the worker could theoretically search in. only used here for simplicity!!!
        #print(self.rho_normalized[0,0],self.rho_normalized[-1,0])
        self.simple_J = np.zeros_like(self.J_grid)
        self.simple_J=np.divide(self.fun_prod[:,ax] -self.pref.inv_utility(self.v_grid[ax,:]*(1-self.p.beta)),1-self.p.beta)
        self.simple_Rho = self.simple_J + self.rho_grid[ax,:] * self.v_grid [ax,:]#We do indeed need to work with Rho here since we're taking W via its derivatives
        #Apply the matching function: take the simple function and consider its different values across v.
        self.prob_find_vx = self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0-1, :], 1.0)), self.p.sigma), 1/self.p.sigma)
        #Now get workers' probability to find a job while at some current value, as well as their return probabilities.
        if cc is None:
            self.js = JobSearchArray() #Andrei: note that for us this array will have only one element
            self.js.update(self.v_grid[:], self.prob_find_vx) #Andrei: two inputs: worker's value at the match quality of entrance (z_0-1), and the job-finding probability for the whole market
        else:
            self.js = cc.js
        #Note: I think??? js takes the values over the uniform grid only. so if I use NNs, gotta adapt it. But for now forget about updating it, keep it as is

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
        ve = self.js.ve(EW1)
        if employed:
            pe = pe * self.p.s_job
            re = re * self.p.s_job
        #print("Shape of pe:", pe.shape)
        # construct the continuation probability. #Andrei: probability the worker doesn't get fired and also doesn't leave
        pc = (1 - pe)

        return ve, re, pc #ve is vhat, the value the worker gets upon finding a job
    def matching_function(self,J1): #Andrei: the formula of their matching function, applied to each particula job value J1
        return self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(J1, self.p.kappa)), self.p.sigma),
                                1 / self.p.sigma)
    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)    
 
    def solve_foc(self, states, prod_states, value_function_model):
        """
        Solves first-order conditions to find optimal action and next state
        
        Args:
            state: Current state tensor
            value_function_model: Neural network model for value function
        
        Returns:
            Dictionary with optimal action, next state, and immediate reward
        """
        # This is a placeholder - replace with your actual FOC solver
        # In a real implementation, you would:
        # 1. Set up an optimization problem to find action that maximizes reward + discounted future value
        # 2. Use value_function_model to evaluate future values
        # 3. Return optimal action and resulting next state
        # Compute gradient with gradients enabled
        EW=self.EW
        rho_star = np.zeros(states.shape[0])
        reward = np.zeros_like(rho_star)
        # Placeholder implementation (just random actions and states)
        with torch.no_grad():
            states_denorm = self.bounds_processor.denormalize(states).numpy()
            prod_states = prod_states.numpy()
            # get worker decisions
            _, _, pc = self.getWorkerDecisions(EW) #This EW1i is computed by taking the derivative of Rho, which is our core value function, wrt rho, which is the value-related state-variable
            # get worker decisions at EW1i + epsilon
            _, _, pc_d = self.getWorkerDecisions(EW + self.deriv_eps)
            log_diff = np.zeros_like(self.EW)
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
            foc = self.rho_grid[:,ax] - self.vf_output * log_diff / self.deriv_eps #So the FOC wrt promised value is: pay shadow cost lambda today (rho_grid), but more likely that the worker stays tomorrow
            assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"

            #Fix a prod state, solve for all the states where that is the right state.
            for iz in range(self.p.num_z):
                states_z = states_denorm[prod_states == iz,0] #Note the 0 here signifying that we're working with the first dimension of the states
                rho_star[prod_states == iz] = np.interp(states_z,
                                        impose_increasing(foc[:,iz]),
                                        self.rho_grid)
            rho_star_tensor = tensor(rho_star, dtype=torch.float32)

            action = rho_star_tensor
            next_state = self.bounds_processor.normalize(rho_star_tensor)
            wage = np.interp(states_denorm[:,0],self.rho_grid,self.w_grid)
            utility = self.pref.utility(wage)
            for iz in range(self.p.num_z):        
                reward[prod_states == iz] = self.fun_prod[iz] - wage[prod_states == iz] + states_denorm[prod_states == iz,0] * utility[prod_states == iz]  # The entire Rho here. Big note though: this should be today's W, not EW
            reward = tensor(reward, dtype=torch.float32)
             


        return {
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "utility": tensor(utility, dtype=torch.float32)
        }

    def simulate_trajectory(self,state, prod_states, value_function_model, foc_optimizer, steps=5):
        """
        Simulate a trajectory starting from a state and using the current value function
    
        Args:
            state: Starting state tensor
            value_function_model: Current value function model
            foc_optimizer: Optimizer to solve FOCs
            steps: Number of steps to simulate
        
        Returns:
            Total discounted reward and final state value
        """
        # prepare expectation call
        total_reward = 0
        discount = 1.0
        w_value = 0.0
    
        current_state = state.clone().requires_grad_(True)
        #EW_TENSOR IS SHAPE (1,200,1) !!! IS THAT RIGHT? THINK MARK
        EW_tensor = get_expectation_gradients(self.rho_normalized, value_function_model, self.Z_trans_tensor, self.bounds_processor.range)[:,:,0]
        self.EW = EW_tensor.detach().numpy()
        #assert np.all(self.EW[ 1:] >= self.EW[ :-1])
        with torch.no_grad():
            # evaluate V(y', s) for all y' on your s-grid:
            V_all = value_function_model(self.rho_normalized)   # [num_z, num_y]
            EJ= torch.einsum("by,zy->bz", V_all, self.Z_trans_tensor).numpy()
            # now subtract the derivative as before:
            self.vf_output = EJ - self.rho_grid[:,ax] * self.EW 
            #self.vf_output = value_function_model(self.rho_normalized).squeeze(1).numpy() - self.rho_grid * self.EW #This EJpi from the CC FOC. I just precompute it for every point here

        for t in range(steps):
            
            # Solve FOC to get optimal action and next state
            result = foc_optimizer.solve_foc(current_state, prod_states, value_function_model)
            #Probability that the worker stays
            EW_star = get_expectation_gradients(result["next_state"].unsqueeze(1), value_function_model, self.Z_trans_tensor, self.bounds_processor.range, prod_states)[:,0]
            ve_star, re_star, pc_star = self.getWorkerDecisions(EW_star.detach().numpy())
            re_star = torch.from_numpy(re_star)
            pc_star = torch.from_numpy(pc_star)            
            # Accumulate discounted reward
            #TOTAL REWARD IS (200,200) SHAPE. WHY?
            total_reward += discount *( result["reward"] + self.p.beta * self.bounds_processor.denormalize(current_state.squeeze(1)) * (EW_star + re_star) - self.p.beta * result["action"] * pc_star * EW_star)  #This should be: reward + rho*beta*ve_star*pe_star (so the value worker gets from leaving) + discount*next_reward
            w_value += discount * (result["utility"] + self.p.beta * ( 1- pc_star) * ve_star) #Each period: utility from wage + value from J2J. Then, if worker stays (with prob pc_star), keep adding future utilities
            #else:
            #    total_reward += discount * result["reward"] #for the last period we don't enymore do ve_star as that's included... right? confirm later
            # Update state and discount factor
            current_state = result["next_state"]
            discount *= self.p.beta * pc_star
    
        # Add final state value
        with torch.no_grad():
            #Note that I take the expectation using the transition matrix
            final_value = torch.einsum("by,zy->bz", value_function_model(current_state.unsqueeze(1)), self.Z_trans_tensor) #Should the tensor be as is or transposed????
        fv=torch.zeros_like(prod_states)
        for iz in range(self.p.num_z):
            fv[prod_states==iz] = final_value[prod_states==iz,iz]
        final_value = fv
        total_value = total_reward + discount * final_value
        w_value = w_value + discount * EW_star #This is the value of the job the worker has today, not the value of the job he will have tomorrow. So we don't need to add it to the reward.
        
        #Seeing whether the gradients converge (later to be used for the loss function)
        #EW = get_batch_gradients(self.bounds_processor.range,state, value_function_model).squeeze(1)
        #print("Difference in W", torch.abs(EW-w_value).mean().item())
        return total_value, current_state, w_value



def train_value_function(
    state_dim,
    lower_bounds,
    upper_bounds,
    action_dim=5,
    hidden_dims=[40, 30, 20, 10],
    num_iterations=20, 
    starting_points_per_iter=100,
    simulation_steps=5,
    learning_rate=0.001,
    p=None,
    cc=None, target_values=None, target_W=None
):
    """
    Main training loop for value function approximation
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        num_iterations: Number of training iterations
        starting_points_per_iter: Number of starting points per iteration
        simulation_steps: Steps to simulate for each starting point
        learning_rate: Learning rate for neural network optimizer
        discount_factor: Discount factor for future rewards
    
    Returns:
        Trained value function model
    """
    bounds_processor = StateBoundsProcessor(lower_bounds,upper_bounds)
    # Initialize value function neural network
    value_function_model = ValueFunctionNN(state_dim, p.num_z, hidden_dims)

    # Initialize FOC optimizer
    foc_optimizer = FOCOptimizer(state_dim, action_dim, value_function_model, bounds_processor, p, cc)
    
    # Initialize neural network optimizer
    optimizer = optim.Adam(value_function_model.parameters(), lr=learning_rate)
    
    #Learning rate scheduler
    #scheduler = get_lr_scheduler(optimizer, num_iterations)
    #Step 0: basic guess    
    states = foc_optimizer.rho_normalized #This should be renormalized... right?
    assert torch.all(states[1:] > states[:-1]), "States are not increasing"
    #print(np.max(np.abs(cc.rho_grid-foc_optimizer.rho_grid)))
    target_values=tensor(foc_optimizer.simple_Rho, dtype=torch.float32).t()
    target_W = torch.zeros_like(target_values) + tensor(foc_optimizer.v_grid[ax,:], dtype=torch.float32).t()
    #Train also on the gradient
    print("Pre-training...")
    for _ in range(50):
        optimizer.zero_grad()
        predicted_values = value_function_model(states)
        predicted_grads = get_batch_gradients(states, value_function_model,  range_tensor=bounds_processor.range)[:,:,0]
        #Add gradient loss and monotonicity loss
        violation = torch.relu(predicted_grads[:-1,:] - predicted_grads[1:,:])
        mon_loss = (violation ** 2).mean() #This is the loss function that forces the gradient to be increasing
        loss = nn.MSELoss()(predicted_values, target_values) + mon_loss + nn.MSELoss()(predicted_grads, target_W) 
        #loss = nn.MSELoss()(predicted_values[:,p.z_0-1], target_values[:,p.z_0-1]) + 10 * mon_loss + nn.MSELoss()(predicted_grads[:,p.z_0-1], target_W[:,p.z_0-1]) 
        #loss = mon_loss
        loss.backward() 
        optimizer.step()
    # After pre-training:
    with torch.no_grad():
        test_pred = value_function_model(foc_optimizer.rho_normalized)
        print("Pre-train MAE:", torch.abs(test_pred - target_values).mean().item())
        print("Monotonicity MAE:", mon_loss.item())
        print("Gradient MAE:", torch.abs(predicted_grads - target_W).mean().item())
        # During training, after pre-training:
        #predicted_grads = get_batch_gradients(states, value_function_model, bounds_processor.range)[:,p.z_0-1,0]
        #plt.plot(states.detach().numpy(), predicted_grads.numpy())
        #plt.title("Training Gradients")
        #plt.show()


# If these are unstable, your simulation logic is flawed
    print("Training...")
    # Training loop
    for iteration in tqdm(range(num_iterations)):
        # Generate uniform random starting states
        states= torch.rand(starting_points_per_iter, state_dim,dtype=torch.float32).requires_grad_(True)
        pred_values = torch.zeros((starting_points_per_iter))
        EW = torch.zeros((starting_points_per_iter))
        states, _ = torch.sort(states, dim=0)
        prod_states = np.random.choice(range(p.num_z), size=(starting_points_per_iter))
        prod_states = tensor(prod_states, dtype=torch.float32)

        # Calculate target values through simulation
        target_values = []
        # Simulate trajectory and get total discounted reward
        target_values, _, W = foc_optimizer.simulate_trajectory(
            states, prod_states, value_function_model, foc_optimizer, simulation_steps)
        #target_values  = target_values.unsqueeze(1)
        
        #target_values = tensor(target_values, dtype=torch.float32)
        
        # Update neural network based on simulated values
        optimizer.zero_grad()
        predicted_values = value_function_model(states)
        EW_NN = get_batch_gradients(states, value_function_model,  range_tensor=bounds_processor.range)[:,:,0]
        for iz in range(p.num_z):
            pred_values[prod_states==iz] = predicted_values[prod_states==iz,iz]
            EW[prod_states==iz] = EW_NN[prod_states==iz,iz]
        predicted_values = pred_values
        EW_NN=EW
        #Also compare the gradient wrt rho to the worker's value W
        
        #AND add monotonicity loss on the gradient
        violation = torch.relu(EW_NN[:-1] - EW_NN[1:])
        mon_loss = (violation ** 2).mean() #This is the loss function that forces the gradient to be increasing
        #print(np.mean(np.abs(predicted_values.detach().numpy()-target_values.detach().numpy())/np.abs(target_values.detach().numpy())))
        loss = nn.MSELoss()(predicted_values, target_values) + nn.MSELoss()(EW_NN, W) + 50.0 * mon_loss #+ nn.MSELoss()(predicted_values, target_values)
        loss.backward() #
        #torch.nn.utils.clip_grad_norm_(value_function_model.parameters(), max_norm=2.0)  # Critical!
        optimizer.step()
        
        # Update learning rate
        #scheduler.step()
        # Print progress
        if (iteration + 1) % (num_iterations/10) == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss.item():.6f} ,Monotonicity Loss: {mon_loss.item():.6f}")
    
    return value_function_model

def evaluate_value_function(model, p, lower_bounds,upper_bounds,cc,cc_Rho,cc_Wstar):
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
    EW_star_NN=get_batch_gradients(test_states, model,  range_tensor=bounds_processor.range)[:,p.z_0-1,0].detach().numpy()
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
    plt.plot(cc.rho_grid, cc_Wstar[p.z_0-1,:], label = "VFI")
    plt.plot(cc.rho_grid, EW_star_NN, label = "NN")    
    plt.show()

if __name__ == "__main__":
    # Define parameters
    STATE_DIM = 1  # Just one, continuous state, the promised value. Next step will be adding (discrete) y
    ACTION_DIM = 1  # Adjust based on your problem
    HIDDEN_DIMS = [64,64]  # Decreasing width architecture
    pref = Preferences(input_param=p)
    cc=ContinuousContract(p) 
    _,cc_W,cc_Wstar,cc_J = cc.J(0) 
    target_values = tensor(cc_J + cc.rho_grid[ax,:] * cc_W, dtype=torch.float32)
    target_W = tensor(cc_W, dtype=torch.float32)
    assert torch.all(target_W[:,1:] >= target_W[:,:-1]), "target_W is not increasing"
    LOWER_BOUNDS = [cc.rho_grid[0]]
    UPPER_BOUNDS = [cc.rho_grid[-1]] #Ideally this should come from fun_prod.max
    # Train value function
    print("Training value function...")
    beg=time()
    trained_model = train_value_function(
        state_dim=STATE_DIM,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        action_dim=ACTION_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_iterations=5000,
        starting_points_per_iter=200,
        simulation_steps=1,
        learning_rate=0.01,
        p=p,
        cc=cc, target_values=target_values.t(), target_W=target_W.t()
    )
    print("Training time:", time()-beg)
    # Evaluate trained model
    evaluate_value_function(trained_model, p, LOWER_BOUNDS, UPPER_BOUNDS,cc,target_values,cc_W)
    
    # Save the model
    torch.save(trained_model.state_dict(), "trained_value_function.pt")
    print("Model saved to trained_value_function.pt")
