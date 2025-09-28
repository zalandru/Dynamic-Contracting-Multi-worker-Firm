#Neural Network solver for the CRS model, with both sup and value networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.func import jacrev, vmap
import math
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
from time import time
import math
import copy
from plotter import LossPlotter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from time import time
import torch.optim.lr_scheduler as lr_scheduler
# ---- set once, at top of your entrypoint (BEFORE importing torch.compile paths) ----
import os, pathlib, matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
OUTDIR = pathlib.Path(os.environ.get("OUT_DIR", "runs/local/plots"))
OUTDIR.mkdir(parents=True, exist_ok=True)
print(f"[mpl backend] {matplotlib.get_backend()}")
print(f"[plots dir]   {OUTDIR.resolve()}")
import matplotlib.pyplot as plt
#For the cluster saving
from pathlib import Path
from datetime import datetime
# make caches persistent across runs, not random Temp dirs
#os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", r"C:\Users\andre\.cache\torch\inductor")
#os.environ.setdefault("TRITON_CACHE_DIR",        r"C:\Users\andre\.cache\triton")
def savefig_now(basename: str, fig=None, ext="png", dpi=150):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = OUTDIR / f"{basename}_{ts}_{os.getpid()}.{ext}"
    (fig or plt).tight_layout()
    (fig or plt).savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig) if fig else plt.close()
    print(f"[saved] {path}", flush=True)
    return path

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
# add a helper
def to_(x): 
    return x.to(device) if torch.is_tensor(x) else x
# --- AMP config (new) ---
def init_amp(device):
    # device_type is "cuda" on GPU, else "cpu"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    if device_type == "cuda":
        # Prefer BF16 on Ampere+ (more stable than FP16)
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        amp_dtype = torch.bfloat16  # CPU autocast uses BF16

    # GradScaler only needed for FP16 on CUDA
    use_scaler = (device_type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    return device_type, amp_dtype, scaler, use_scaler
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
        x =  (states - self.lower_bounds[ax, :]) / self.range[ax, :]
        return x#.clamp(-5.0, 5.0)  # gentle, in normalized coords

        #Example: lower-upper is [0,1]. So normalize(0.5) = 2 * (0.5 - 0) /1 -1 = 0. Ok correct
        #Another example. lower-upper is [0,30]. Sn normalize 15= 2 * 15 / 30 -1 = 0 Ok good.
        # And normalize (20) = 40/30 - 1 = 1/3 yup
        # Now denormalize(1/3) = 0.5 ( 1/3 +1 ) * 30 + 0 = 2/3*30 = 20

        #Normalizing to (0,1) now. Checking: 15 in (0,30) is 15/30=0.5. 15 in (10,20) is 5/10=0.5       
    def normalize_rho(self, states):
        """Scale states from [lower_bound, upper_bound] to [0, 1]"""
        x = (states - self.lower_bounds[ax, K_n:]) / self.range[ax, K_n:]    #Note: this will need to be adjusted at multiple steps    
        return x#.clamp(-5.0, 5.0)  # gentle, in normalized coords
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
        x = (states - self.lower_bounds[dim]) / self.range[dim]    #Note: this will need to be adjusted at multiple steps      
        return x#.clamp(-5.0, 5.0)  # gentle, in normalized coords
    def denormalize_dim(self, normalized_states,dim):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range[dim] + self.lower_bounds[dim] #Check: 0.5 * 10 + 10= 15. 
    def denormalize_size(self, normalized_states):
        """Convert normalized states back to original range"""
        #return 0.5 * (normalized_states + 1) * self.range + self.lower_bounds
        return normalized_states * self.range[ax,:K_n] + self.lower_bounds[ax,:K_n] #Check: 0.5 * 10 + 10= 15.
    def to(self, device):
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)
        self.range        = self.range.to(device)
        return self
#Neural Nets. Note: alternatively, I could code give them the same trunk. Either way, this is called an Actor-Critic architecture.
# These are forced to be increasing in y, kinda riskily tho, due to the cumsum.
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
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            input_dim = h
        self.trunk = nn.Sequential(*layers)

        # head for values: [B, num_y]
        self.value_head = nn.Sequential(
        nn.Linear(input_dim, num_y),
        #nn.Softplus() #So that these are increasing in y
        )
        self._init_weights()
        self.state_dim = state_dim
        self.num_y     = num_y
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        #for layer in self.trunk:
        #    if isinstance(layer, nn.Linear):
        #        torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('tanh'))
        #        torch.nn.init.zeros_(layer.bias)
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)            
    def forward(self, x):
        B = x.size(0)
        #x_tanh = 2*x - 1
        features = self.trunk(x)                    # [B, hidden_dims[-1]]
        values   = self.value_head(features)        # [B, num_y]
        #values = torch.cumsum(values, dim = 1) #This is the cumulative sum of the values across num_y

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
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # future value v' head: output num_y * num_Kv values, then reshape
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, num_y * self.K_v * num_y), #Policy v'_{y',k}(y), for each [y,k,y']
            nn.Softplus()
        )


        self.hiring_head = nn.Sequential(
            nn.Linear(input_dim, num_y),
            nn.Softplus()
        )
        # Apply activation‐specific initialization
        self._init_weights()
    def _init_weights(self):
        # 1) Trunk: He/Kaiming
        for layer in self.trunk:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.0)
        # 2) Hiring_head: Xavier/Glorot
        for seq in (self.hiring_head, self.value_head):
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.01)
    def squash_frac(self,x):      # = x/(1+x)
        return x / (1 + x)
    def forward(self, x):
        # x: [B, state_dim]
        B = x.size(0)
        #x_tanh = 2*x - 1 #Renormalizing to [-1,1]
        features = self.trunk(x)  # [B, hidden_dims[-1]]



        # values: reshape to [B, num_y, num_Kv]
        #hv = self.value_adapter(features)  # [B, hidden_dims[-1]]
        values_flat = self.value_head(features)            # [B, num_y * num_Kv]
        values = values_flat.view(B, self.num_y, self.K_v, self.num_y)  # [B, num_y, num_Kv]
        values = torch.cumsum(values, dim = 1) / self.num_y #this is the cumulative sum of the values across num_y       
        values = self.squash_frac(values)
        # hire probabilities: [B, num_y]
        #hh = self.hiring_adapter(features)  # [B, hidden_dims[-1]]
        hiring = self.hiring_head(features)          # [B, num_y]
        hiring = torch.cumsum(hiring, dim = 1) / self.num_y
        hiring = self.squash_frac(hiring)

        return {
            'values': values,
            'hiring': hiring
        }

def grad_l2_norm(loss: torch.Tensor, params):
    """Return L2 norm of grads of `loss` w.r.t. `params` (list of tensors)."""
    grads = torch.autograd.grad(
        loss, [p for p in params if p.requires_grad],
        retain_graph=True, create_graph=False, allow_unused=True
    )
    total = torch.zeros([], device=loss.device)
    for g in grads:
        if g is None: 
            continue
        total = total + g.pow(2).sum()
    return total.sqrt()

class LambdaBalancer:
    def __init__(self, init_lambda=1.0, ema=0.9, minv=0.1, maxv=10.0, eps=1e-12):
        self.value = init_lambda
        self.ema   = ema
        self.minv  = minv
        self.maxv  = maxv
        self.eps   = eps
    def update(self, g_val, g_grad):
        # target so that g_val ≈ λ * g_grad  =>  λ_target = g_val / (g_grad + eps)
        target = (g_val / (g_grad + self.eps)).clamp(self.minv, self.maxv).item()
        self.value = self.ema * self.value + (1 - self.ema) * target
        return self.value

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

#Fin Diff gradients
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
    #states = states.requires_grad_(True)
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
    grads      = (V_plus - V_minus) / eps                          # [B, num_y, D]

    # Optional: rescale gradients if states were normalized
    if range_tensor is not None:
        expectation_grads = grads.cpu() / range_tensor[ax,ax,:]  # broadcast over D

    return expectation_grads  # [B, y, D] or [B, D] if current_y is not None

def compile_or_fallback(module, device, mode="reduce-overhead"):
    try:
        if device.type == "cuda":
            import triton  # checks availability
            return torch.compile(module, backend="inductor", mode=mode)
        else:
            # CPU Inductor doesn't need Triton
            return torch.compile(module, backend="inductor", mode=mode)
    except Exception as e:
        print(f"[torch.compile fallback: {e.__class__.__name__}] using backend='aot_eager'")
        return torch.compile(module, backend="aot_eager", mode=mode)

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
        #self.rho_normalized = self.bounds_processor.normalize_rho(self.rho_grid).unsqueeze(1).requires_grad_(True)

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
    def take_expectation(self, x, i,prod_states,v_prime=None):    
        if v_prime is not None:
            x = torch.einsum("bkZ,YZ->bkY", x, self.Z_trans_tensor) #Shouldn't this be for all of them though? Since I'm looking for E_{y'|y}? Or am I wrong here?
            x = x[i,:,prod_states.long()] 
        else:
         if x.ndim == 2:  # If x is [B, D]
            x = torch.einsum("by,zy->bz", x, self.Z_trans_tensor)
            x = x[i,prod_states.long()]
         else:
            x = torch.einsum("byd,zy->bzd", x, self.Z_trans_tensor)
            x = x[i,prod_states.long(),:]            
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
        # construct the continuation probability.
        pc = (1 - pe)

        return re, pc
    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:],
                        (1))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)    

        """Given the Value network and the sup network,
        Compute the FOC residuals for given set of states
        Requires: EW_star for each point, derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star ? Right?
        Note: for hirings and layoffs, will need to ensure the loss is zero when the sup is consistent: sep=0 => FOC<0, sep=1 => FOC>0 etc
        """
        # Derivative E_{y'|y}∂J / ∂n'_{k+1}=E_{y'|y}∂𝒫 / ∂n'_{k+1} - ρ*_{k+1} EW_star
        EJ_deriv = future_grad_exp[:,1:K_n]
        states_d = self.bounds_processor.denormalize(states)
        size=states_d[:,:K_n]
        v = states_d[:,K_n:]
        #Get worker's search decisions and associated return
        re, pc = self.getWorkerDecisions(v_prime_exp)
        re_d, pc_d = self.getWorkerDecisions(v_prime_exp + self.deriv_eps) 
        assert torch.all(re_d + v_prime_exp + self.deriv_eps>= re + v_prime_exp), "continuation value is not increasing"
        # After computing pc and pc_d:
        assert not torch.isnan(pc).any(), "NaN in pc"
        assert not torch.isnan(pc_d).any(), "NaN in pc_d"
        log_diff = torch.zeros_like(v_prime_exp)
        #log_diff[:] = torch.nan
        log_diff[pc > 0] = torch.log(pc_d[pc > 0]) - torch.log(pc[pc > 0]) #This is log derivative of pc wrt the promised value
        #print("min pc", pc.min().item()) 0.51 because search ain't perfect
        assert torch.all(log_diff >= 0)
        assert torch.all(~torch.isnan(log_diff))
        # After computing log_diff:
        assert not torch.isnan(log_diff).any(), "NaN in log_diff"
        wage = foc_optimizer.pref.inv_utility(v - p.beta * (v_prime_exp + re)) #Senior wage that we actually got to calculate now
        rho = wage
        inv_utility_1d = self.pref.inv_utility_1d(self.v_0-self.p.beta*(v_prime_exp+re))
        assert torch.all(inv_utility_1d > 0)
        # After computing inv_utility_1d:
        assert not torch.isnan(inv_utility_1d).any(), "NaN in inv_utility_1d"


        focs_rho_2ndpart =  rho * size[:,1:] + size[:,:1] / inv_utility_1d               
        focs_rho = EJ_deriv * (log_diff / self.deriv_eps)
        focs_rho = focs_rho * (size[ :,:1] + size[ :,1:]) + focs_rho_2ndpart     #Shape [B,K_n-1]
        assert not torch.isnan(focs_rho).any(), "NaN in focs_rho"
        focs_v_prime = focs_rho[:,ax,:] + + future_grad[:,:,K_n:] / pc[:,ax,:] #We just do all of them like this in order not to have any weird asymmetry. Shape [B,num_y,K_n-1]
        focs_v_prime[size[:,0]+size[:,1] <= 0, :, :] = 0 #If size is zero, then the FOC is zero. This is because we do not have any production and no hiring, so the FOC is zero.

        #FOCs across v'_{y'}... ahhh gota figure this one out! Because this shouldn't be an expected gradient, but the actual future ones:
        # Idea though is that the should be equal across y' : \frac{\partial J(y',...)}{\partia v'_{y'}} = \frac{\partial J(y'',...)}{\partial v'_{y''}}, wher y' and y'' are both future prod states.
        #focs_v_prime = (future_grad[:,:-1,K_n:] -  future_grad[:,1:,K_n:]) #Differences between the future grads wrt v'_{y'}, should be equal these guys.

        #Now hiring FOC
        #Set up using an Fischer-Burmeiser function. For KKT problems of the type
        # A>=0, H>=0, A*H=0
        # For my case, A=hiring, H = - foc_hire
        #Ψ(a,h) = a + h - √( a^2 + h^2) = 0
        a = hiring
        h = 1 - self.p.beta * future_grad_exp[:, 0] / self.p.hire_c
        Ψ = a + h - torch.sqrt(a**2 + h**2)
        #fraction_near_zero = (hiring < 1e-3).float().mean().item()

        #print(f"Fraction of firms with hiring ≈ 0: {fraction_near_zero:.2%}")
        #print(f"Lowest hiring: {hiring.min().item():.4f}, Highest hiring: {hiring.max().item():.4f}")
        return focs_v_prime, Ψ
    def get_fut_size(self, states, v_prime_exp):
        re, pc = self.getWorkerDecisions(v_prime_exp)
        size = self.bounds_processor.denormalize_size(states[:,:K_n])
        n_1 = ((size[:,0]+size[:,1]) * pc.squeeze(1)).unsqueeze(1)
        
        return n_1, re, pc
    def simulate(self, starting_states, sup_net, value_net, bounds_processor, Z_trans_tensor, simulation_steps, random_paths=5):
        """
        Random-path simulation.
        Returns:
            S          : [T*N, D]  current (input) states at each step
            P          : [T*N]     current productivity y for each row of S
            Fut_states : [T*N, D]  realized next states aligned with S
        """
        device = starting_states.device
        dtype  = starting_states.dtype
        Z      = p.num_z

        # replicate starting points to open multiple Monte Carlo paths
        states = starting_states.repeat(random_paths, 1)  # [N, D]
        N      = states.shape[0]
        iN     = torch.arange(N, device=device)
        rewards      = torch.zeros((N, simulation_steps), device=device, dtype=dtype)

        S_time = torch.empty(simulation_steps+1, N, states.shape[1], device=device, dtype=dtype)
        P_time = torch.empty(simulation_steps,    N, device=device, dtype=torch.long)         # y_t
        # initialize current y for each path
        y_idx = torch.randint(0, Z, (N,), device=device)
        #Initialize current sizes and values
        states_d = bounds_processor.denormalize(states)
        size = states_d[:, :K_n]
        wages = torch.zeros(states_d.shape[0],K_n, device=device) 
        tot_size = size.sum(dim=1)
        S_time[0] = states
        for t in range(simulation_steps):
            # record the CURRENT y for these input states
            P_time[t] = y_idx



            # policies at the current y
            pol      = sup_net(states)
            hiring   = pol['hiring'][iN, y_idx] * self.bounds_processor.upper_bounds[0]                          # [N]
            v_prime  = pol['values'][iN, y_idx, :, :] * self.bounds_processor.upper_bounds[K_n]                   # [N, K_v, Z]

            # E_{y'|y} v′ for worker decisions (same convention as elsewhere)
            #v_prime_exp_all = torch.einsum("bkz,yz->bky", v_prime, Z_trans_tensor)  # [N, K_v, Z]
            v_prime_exp_all = v_prime @ Z_trans_tensor.T
            v_prime_exp     = v_prime_exp_all[iN, :, y_idx]                            # [N, K_v]
            re, pc = foc_optimizer.getWorkerDecisions(v_prime_exp)
            size = states_d[:,:K_n]
            tot_size = size.sum(dim=1) #Now I'm summing up over the actual current sizes. Before I was using tot_size from below, which was not correct since it didn't incorporate hiring on pc effects
            v = states_d[:,K_n:]
            wages = torch.zeros(states_d.shape[0],K_n, device=device) 
            wages[:,1:] = foc_optimizer.pref.inv_utility(v - p.beta * (v_prime_exp + re))
            wages[:,:1] = self.pref.inv_utility(self.v_0 - self.p.beta*((v_prime_exp+re)))
            tot_wage = (wages * size).sum(dim=1) #LOL I DIDN'T SUM THEM UP BY SIZE
            #Getting reward for this state
            r_t = self.fun_prod[y_idx.detach().long()] * self.production(tot_size) - self.p.hire_c * hiring - tot_wage
            rewards[:, t] = r_t
            # build realized next state using the DRAWN y′
            sizes    = bounds_processor.denormalize_size(states[:, :K_n]) # [N, K_n]
            tot_size = sizes[:, 0] + sizes[:, 1]                          # [N]
            n1       = (tot_size * pc.squeeze(1)).unsqueeze(1)            # [N,1]

            # draw next productivity y′ ~ P(y→·)
            next_prod_probs = Z_trans_tensor[y_idx, :]                    # [N, Z]
            y_next = torch.multinomial(next_prod_probs, num_samples=1).squeeze(1)  # [N]

            next_states = torch.empty_like(states)                        # [N, D]
            next_states[:, 0]     = hiring
            next_states[:, 1:K_n] = n1
            # each child gets v′(·, y′)
            next_states[:, K_n:]  = v_prime[iN, :, y_next]                # [N, K_v]
  
            # safety & normalization
            assert not torch.isnan(next_states).any(), "NaN in simulated states"
            assert torch.all(next_states[:, K_n:] >= 0), "Negative v′ components in states"
            states_d = next_states
            next_states = bounds_processor.normalize(next_states)
            S_time[t+1] = next_states
            #all_states.append(next_states)
            states = next_states
            y_idx  = y_next
        #Conclude values
        values_last = value_net(states)['values']
        # Clamp the expectation using observed values
        r_max = rewards.max().detach()
        V_cap = (r_max / (1.0 - self.p.beta)).clamp_min(1.0)  # simple, safe cap
        r_min = rewards.min().detach()
        V_bot = (r_min) / (1.0 - self.p.beta)
        values_last = values_last.clamp(min = V_bot, max = V_cap)
        values_last_exp = self.take_expectation(values_last, iN, y_idx)

        # ----- Discounted backward cumulative sum to get G_t for every visited S_t -----
        G = torch.empty_like(rewards)                 # [N, T]
        running = values_last_exp                     # [N]
        for t in range(simulation_steps - 1, -1, -1):
            running = rewards[:, t] + self.p.beta * running
            G[:, t] = running
        # stitch trajectories: (S_t, y_t) → S, P; and S_{t+1} → Fut_states
        S          = S_time[:-1]                    # [T*N, D]
        #Fut_states = torch.cat(all_states[1:],  dim=0)                    # [T*N, D]
        P          = P_time                    # [T*N]
        # Flatten G in the same time-major order
        G_flat = torch.cat([G[:, t] for t in range(simulation_steps)], dim=0)      # [T*N]
        # get G only for the starting states in order to train policy
        G_starting = G[:,0]

        # 1) Pack states/time explicitly
        #S_time = torch.stack(all_states[:-1], dim=0)   # [T, N, D]  = S_t
        #P_time = torch.stack(all_P,          dim=0)    # [T, N]     = y_t

        # 2) You already built rewards: rewards is [N, T] in your code — transpose to [T, N]
        #    And G was computed by backward-sum producing [N, T]; make it [T, N] too.
        G_time = G.transpose(0, 1).contiguous()        # [T, N]
        R_time = rewards.transpose(0, 1).contiguous()   # [T, N]
        # 3) Now flatten *all* with the same reshape (time-major)
        T, N, D = S.shape
        S_flat  = S_time[:-1].reshape(simulation_steps * N, -1)             # [T*N, D]
        P_flat  = P_time.reshape(simulation_steps * N)                # [T*N]
        G_flat  = G_time.reshape(T * N)                # [T*N] 
        R_flat = R_time.reshape(T * N)                  # [T*N]
        #assert torch.all(S == S_flat), torch.all(P == P_flat)       
        #assert (torch.abs(P_flat[N:2*N] - all_P[1]).max() < 1e-8), (torch.abs(G_flat[N:2*N] - G[:,1]).max() < 1e-8)
        #assert ( torch.abs(P_flat[:N] - all_P[0]).max() < 1e-8) #Checking that the indices match
        return S_flat, P_flat, G_flat, G_starting, R_flat, values_last_exp
    def to(self, device):
        self.fun_prod        = self.fun_prod.to(device)
        self.Z_trans_tensor  = self.Z_trans_tensor.to(device)
        self.rho_grid        = self.rho_grid.to(device)
        #self.rho_normalized  = self.rho_normalized.to(device)
        return self

def soft_update(target_net, source_net, tau=0.005):
    """
    θ_tgt ← τ·θ_src + (1–τ)·θ_tgt
    Args:
        target_net: torch.nn.Module whose params will be updated in-place
        source_net: torch.nn.Module, the “online” network
        tau (float): interpolation factor (small, e.g. 0.005)
    """
    for tgt_param, src_param in zip(target_net.parameters(), source_net.parameters()):
        tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)
def initialize(bounds_processor, state_dim, K_n, K_v, HIDDEN_DIMS_CRIT, HIDDEN_DIMS_POL, learning_rate, weight_decay, pre_training_steps, num_epochs, minibatch_num):
    #Initializations:
    # Initialize the neural networks
    value_net = ValueFunctionNN(state_dim, p.num_z, HIDDEN_DIMS_CRIT)
    sup_net = PolicyNN(state_dim, p.num_z, K_v, HIDDEN_DIMS_POL, cc)  
    #Compile the nets
    value_net = compile_or_fallback(value_net, device)
    sup_net   = compile_or_fallback(sup_net,   device)
    # --- Optimizers --- fused AdamW is faster on CUDA
    use_fused = (device.type == "cuda")
    optimizer_value  = AdamW(value_net.parameters(),  lr=learning_rate[0], weight_decay=weight_decay[0], betas=(0.9, 0.999), fused=use_fused)
    optimizer_sup = AdamW(sup_net.parameters(),    lr=learning_rate[1], weight_decay=weight_decay[1], betas=(0.9, 0.999), fused=use_fused)

    #optimizer_value  = AdamW(value_net.parameters(),  lr=learning_rate[0], weight_decay=weight_decay[0], betas=(0.9, 0.999))
    #optimizer_sup = AdamW(sup_net.parameters(),    lr=learning_rate[1], weight_decay=weight_decay[1], betas=(0.9, 0.999))
    # Initialize FOC computer
    foc_optimizer = FOCresidual(bounds_processor, K=K_n, p=p, cc=None)    

    # --- Schedules ---
    # Fill these based on your loop:
    # total_steps = num_epochs * (num_batches_per_epoch)
    # warmup_steps = ~1-2k steps (or ~2-5% of total)
    warmup_steps = num_epochs // 20
    #total_steps  = num_epochs            # example; set this to your real total
    cosine_steps = max(1, num_epochs - warmup_steps)

    def make_sched(opt, base_lr):
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=base_lr * 0.1)
        return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scheduler_value  = make_sched(optimizer_value, learning_rate[0])
    scheduler_sup = make_sched(optimizer_sup, learning_rate[1])
 

    return value_net, sup_net, optimizer_value, optimizer_sup, scheduler_value, scheduler_sup, foc_optimizer

@torch.no_grad()
def td_lambda_targets_from_flat(
    R_flat,                      # [T*N]
    S_flat, P_flat,              # [T*N, D], [T*N]
    target_value_net,            # slow/Polyak critic
    T, N,                        # simulation_steps, num_paths
    V_last_bootstrap,            # [N] bootstrap for last step
    gamma: float,
    lam: float,
):
    """
    Compute TD(λ) / GAE-style targets for the value function.
    Shapes follow your simulate() flattening order (time-major).
    Returns:
        Rlam_flat: [T*N] TD(λ) targets = V_t + Adv_t(λ)
    """
    device = S_flat.device
    D = S_flat.shape[1]

    # 1) reshape to time-major
    R = R_flat.view(T, N)                      # [T, N]
    S = S_flat.view(T, N, D).contiguous()      # [T, N, D]
    P = P_flat.view(T, N).contiguous()         # [T, N]

    # 2) V_t from the target critic
    V_all = target_value_net(S.view(T*N, D))['values']            # [T*N, Z]
    V_t   = V_all.gather(1, P.view(T*N, 1)).squeeze(1).view(T, N) # [T, N]

    # 3) Next values with bootstrap on the last step
    V_next = torch.empty_like(V_t)             # [T, N]
    V_next[:-1] = V_t[1:]
    V_next[-1]  = V_last_bootstrap             # [N]

    # 4) TD deltas and backward GAE recursion
    deltas = R + gamma * V_next - V_t          # [T, N]
    adv = torch.zeros_like(V_t)                # [T, N]
    gae = torch.zeros(N, device=device, dtype=V_t.dtype)
    for t in range(T-1, -1, -1):
        gae = deltas[t] + gamma * lam * gae
        adv[t] = gae

    # 5) TD(λ) targets for the value net
    Rlam = V_t + adv                           # [T, N]
    return Rlam.view(T*N)                      # [T*N]


def train(state_dim, value_net, sup_net, optimizer_value, optimizer_sup, scheduler_value, scheduler_sup, foc_optimizer, bounds_processor, num_episodes=20, starting_points_per_iter=100, simulation_steps=5, 
    minibatch_num=8, λ=1.0, use_saved_nets = False):
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
    # Define which losses you’ll track
    loss_names = ['value','sup_loss']
    
    # AMP setup
    amp_device_type, amp_dtype, scaler, use_scaler = init_amp(device)
    # Initialize plot before training
    plotter = LossPlotter(loss_names, pause=0.001, update_interval=50, ma_window=10,show_raw=False,show_ma=True)

    delta_v =tensor([0,0,1e-3],device=device)

    state_start = torch.zeros(state_dim,dtype=type, device=device)#.requires_grad_(True)
    state_start[0] = bounds_processor.normalize_dim(1,0) # 1 junior worker
    state_start[1] = bounds_processor.normalize_dim(1e-3,1) # Tiny positive seniors
    state_start[2] = bounds_processor.normalize_dim(cc.v_grid[10],1)
    horizon = 128
    random_paths = 2000 
    print("Training...")
    # Training loop
    for episode in tqdm(range(num_episodes)):
        ep = episode + 1
        torch.compiler.cudagraph_mark_step_begin()  # helps CUDA Graphs know a new iteration begins
        optimizer_value.zero_grad(set_to_none=True)
        optimizer_sup.zero_grad(set_to_none=True)
        #torch.autograd.set_detect_anomaly(True)
        #Starting states. First with a single junior and nothing else. State = (y, {1,0},{rho_min,rho_min}). Plus a bunch of random starting states

        #state_start[2] = torch.rand(1, K_v)
        #Or if randomized. 
        starting_states = torch.rand(starting_points_per_iter, state_dim,dtype=type, device=device) 
        #Add the starting state
        starting_states[0,:] = state_start

        #Simulate the firm path using the sup network
        #Let the simulation steps increase over time
        #horizon = np.floor(128 + 0 * (ep/num_episodes)).astype(int)
        #if horizon <= 8:
        #    random_paths = np.minimum(p.num_z ** horizon, 1000).astype(int)
        #else:
        #Do both simulations: start with a deterministic one just a few periods ahead, then a random one from the last states
        #Now use those states to simulate further, but with random paths
        
        #beg= time()
        with torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype):
            with torch.no_grad():
                states, prod_states, G_flat, G_starting, R_flat, V_last_bootstrap = foc_optimizer.simulate(starting_states, sup_net, target_value_net, bounds_processor, foc_optimizer.Z_trans_tensor, horizon, random_paths) #This is the set of states we will use to train the value function.
        #end = time()
            #print("Simul time", end - beg)
            if (ep) % (num_episodes/20) == 0:
             with torch.no_grad():
                S_det = states.detach()
                for d in range(S_det.shape[1]):
                    col = S_det[:, d]
                    print(f"dim {d}: mean={col.mean():.4g}, std={col.std():.4g}, min={col.min():.4g}, max={col.max():.4g}")
            if ep == 1:
                i = torch.arange(states.shape[0],device=device)


            #Train the value net
            # value branch: fully detached copies
            S_v, P_v, G_v, R_v, Vv = states.detach(), prod_states.detach(), G_flat.detach(), R_flat.detach(), V_last_bootstrap.detach()

            #optimizer_value.zero_grad()
            # Build TD(λ) targets (time-major reconstruction uses sim_steps_ep and N)
            N = starting_points_per_iter * random_paths
            T = horizon
            tdlam_flat = td_lambda_targets_from_flat(
            R_flat=R_v,
            S_flat=S_v,
            P_flat=P_v,
            target_value_net=target_value_net,
            T=T, N=N,
            V_last_bootstrap=Vv,
            gamma=p.beta, lam=0.95
            )
            value_output = value_net(states.detach())
            assert (~torch.isnan(value_output['values'])).all(), "value returns NaN"
            pred_values = value_output['values']  
            pred_values = pred_values[i,P_v.long()]
            tgt= tdlam_flat.detach()
            μ  = tgt.mean()
            σ = tgt.std(unbiased=False).clamp_min(1e-6)
            # Standardize BOTH using the target’s μ, σ
            tgt_n  = (tgt  - μ) / σ
            pred_n = (pred_values - μ) / σ
            #Add a quick monotonicity loss
            states_v = S_v + delta_v
            values_v = value_net(states_v)['values'][i,P_v.long()]
            mon_loss = torch.relu( values_v - pred_values).mean() #value function should be decreasing in v, so zero out cases where pred_values > values_v
            value_loss =  nn.HuberLoss()(pred_n, tgt_n) + 1e+2 * mon_loss

            #Sup net update
            #I first train the sup net on this
            adv_flat = (G_flat - pred_values.detach()) #+ 0.5 * (R_flat - pred_values.detach()) #Advantage is TD error + immediate reward - baseline
            scale = adv_flat.std(unbiased=False).clamp_min(1e-6).detach()
            policy = sup_net(S_v) #maybe I just track it in the simulation?
            policy_v = sup_net(S_v + delta_v)
            mon_loss_sup = torch.relu(-( policy_v['values'][i,P_v.long(),:,:] - policy['values'][i,P_v.long(),:,:])).mean() #v'(v) should be increasing in v
            mon_loss_hiring = torch.relu( policy_v['hiring'][i,P_v.long()] - policy['hiring'][i,P_v.long()]).mean() #hiring(v) should be decreasing in v
            sup_loss = - (adv_flat/scale).mean() + 1e+2 * mon_loss_sup + 1e+2 * mon_loss_hiring

            #This ain't an advantage, but tbf my policy ain't random here, so do I even need it?
        
        if use_scaler:
            # VALUE loss
            scaler.scale(value_loss).backward()  # keep graph for sup_loss backward if they share parts
            scaler.unscale_(optimizer_value)                      # so clipping sees true grads
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            scaler.step(optimizer_value)

            # POLICY loss
            scaler.scale(sup_loss).backward()
            scaler.unscale_(optimizer_sup)
            torch.nn.utils.clip_grad_norm_(sup_net.parameters(), 1.0)
            scaler.step(optimizer_sup)

            scaler.update()
        else:
            # BF16 (or CPU) path — no scaler
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            optimizer_value.step()

            sup_loss.backward()
            torch.nn.utils.clip_grad_norm_(sup_net.parameters(), 1.0)
            optimizer_sup.step()
        #sup_loss.backward()
        #torch.nn.utils.clip_grad_norm_(sup_net.parameters(), max_norm = 1.0) #Clip the gradients to avoid exploding gradients
        #optimizer_sup.step()
        scheduler_sup.step()
        #value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm = 1.0) #Clip the gradients to avoid exploding gradients
        #optimizer_value.step()
        #Later I can consider adding my usual stuff but let's try these basics for now
        scheduler_value.step()  # or just .step(episode)

        #Collect your raw loss scalars
        losses = {
        'value':       value_loss.item(),
        'sup_loss':  sup_loss.item(),
        }
        tail = (p.beta**horizon) * V_last_bootstrap.abs().mean()
        if (episode + 1) % (num_episodes/20) == 0:
            print( tail/G_flat.abs().mean().item() )        

        # One line hides all the plotting mess
        plotter.update(ep, losses)
        if (ep == 2000):
            plotter.update_interval = 500 #Slow down the plotting
        #Soft update target value at the end of every episode
        soft_update(target_value_net, value_net, tau=0.05)
        #soft_update(target_sup_net, sup_net, tau=0.05)     
        # Print progress
        if (episode + 1) % (num_episodes/20) == 0:
            print(f"Iteration {episode + 1}, Value Loss: {value_loss.item():.6f}, Sup Loss:  {sup_loss.item():.6f}" )
        if (episode + 1) % (num_episodes/20) == 0 or episode == 1:            
            evaluate_plot_precise(value_net, sup_net, [1,1], bounds_processor, foc_optimizer)    
        if (episode + 1) % (num_episodes/2) == 0:         
            evaluate_plot_sup(value_net, sup_net,bounds_processor, num_samples=1000) 
        #if (episode + 1) % (num_episodes // 20) == 0:
            #eval_losses = evaluate_loss(value_net, sup_net, bounds_processor, foc_optimizer, horizon=horizon, random_paths=random_paths)
            #print(f"[Eval] Value Eval Loss: {eval_losses['value_eval']:.6f}, "
            #  f"Sup Eval Loss: {eval_losses['sup_eval']:.6f}")
            # optionally feed into LossPlotter too
            #plotter.update(ep, {**losses, **eval_losses})           
    return value_net, sup_net

def evaluate_plot_sup(value_net, sup_net, bounds_processor, num_samples=1000):
    """Evaluate the sup by sampling random states and plotting the results"""

    #states = bounds_processor.normalize(states)
    
    # Get sup outputs
    with torch.no_grad():
        # Sample random states
        states = torch.rand(num_samples, bounds_processor.lower_bounds.shape[0], dtype=type, device=device)
        policies = sup_net(states)
        prom_values = policies['values'][:,1,:,1] * bounds_processor.upper_bounds[K_n]
        hiring = policies['hiring'][:,1] * bounds_processor.upper_bounds[0]

        values = value_net(states)['values']
        grads = get_batch_gradients(states, value_net, policies['hiring'].shape[1], range_tensor=bounds_processor.range.cpu())

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, 2].detach().numpy(), prom_values[:,0].detach().numpy(), alpha=0.5)
    plt.title("Policy Evaluation: Promised Values vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Promised Values")
    plt.grid()
    savefig_now("policy_eval_promised_values"); plt.close()
    #plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, 2].detach().numpy(), hiring.detach().numpy(), alpha=0.5)
    plt.title("Policy Evaluation: Hiring vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Hiring")
    plt.grid()
    savefig_now("policy_eval_hiring"); plt.close()
    #plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, 2].detach().numpy(), values[:,1].detach().numpy(), alpha=0.5)
    plt.title("Value Evaluation: Values vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Values")
    plt.grid()
    savefig_now("value_eval"); plt.close()
    #plt.show()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, 2].detach().numpy(), grads[:,1,-1].detach().numpy(), alpha=0.5)
    plt.title("Grads Evaluation: Grads vs State")
    plt.xlabel("State (normalized)")
    plt.ylabel("Rho Grads")
    plt.grid()
    savefig_now("grad_eval"); plt.close()
    #plt.show()

def evaluate_plot_precise(value_net, sup_net, init_size, bounds_processor, foc_optimizer):
    """
    Evaluate the trained value function on test points
    
    Args:
        model: Trained value function model
        num_test_points: Number of test points
        state_dim: Dimension of state space
    """
    test_states = bounds_processor.normalize_dim(tensor(foc_optimizer.v_grid, dtype = type, device=device),-1).unsqueeze(1)
    size = torch.zeros(test_states.shape[0], K_n, dtype = type, device=device)
    size[:, 0] = bounds_processor.normalize_dim(init_size[0],0) # 1 junior worker
    size[:, 1] = bounds_processor.normalize_dim(init_size[1],1) # 1 senior worker
    test_states = torch.cat((size,test_states),dim=1)
    # Evaluate model
    values = value_net(test_states)['values'][:,p.z_0-1]
    #Evaluate policies
    policy = sup_net(test_states)
    v_prime = policy['values'][:,p.z_0-1,:,p.z_0-1] * bounds_processor.upper_bounds[K_n]
    hiring = policy['hiring'][:,p.z_0-1] * bounds_processor.upper_bounds[0]

    W=get_batch_gradients(test_states, value_net,  num_y = foc_optimizer.p.num_z, range_tensor=bounds_processor.range.cpu())[:,p.z_0-1,-1].detach().numpy()


    # Print results
    #print("\nValue function evaluation on test states:")
    #for i in range(min(5, num_test_points)):
    #    print(f"State {i+1}: Value = {values[i].item():.4f}")
    #Plot results
    plt.figure(figsize=(14, 4))
    plt.subplot(1,3,1)
    #plt.plot(cc.rho_grid, cc_Rho[p.z_0-1,:], label = "VFI")
    plt.plot(foc_optimizer.v_grid, values.detach().cpu(), label = "NN")   
    plt.title("Value")  # Add a title to this plot
    plt.legend()  # To show the label in the legend 
    #Plot the gradient
    plt.subplot(1,3,2)
    #plt.plot(cc.rho_grid, cc_W[p.z_0-1,:], label = "VFI")
    plt.plot(foc_optimizer.v_grid, W, label = "NN")    
    plt.title("Value Gradient (=n_1 v_1)")  # Add a title to this plot
    plt.legend()  # To show the label in the legend

    plt.subplot(1,3,3)
    plt.plot(foc_optimizer.v_grid, v_prime[:,0].detach().cpu().numpy(), label = "NN v_prime")    
    plt.plot(foc_optimizer.v_grid, hiring.detach().cpu().numpy(), label = "NN hiring")
    plt.title("Sup policies")  # Add a title to this plot
    plt.legend()  # To show the label in the legend

    plt.tight_layout()  # Adjust spacing for better visualization
    #plt.show()
    savefig_now("policy_eval_precise"); plt.close()
def plot_reached_states(foc_optimizer, starting_states, sup_net, value_net, 
                        bounds_processor, sim_steps=10, random_paths=5,
                        dim_x=0, dim_y=1):
    """
    Runs foc_optimizer.simulate() and plots visited (reached) states in two chosen dimensions.

    Args:
        foc_optimizer:   FOCresidual instance
        starting_states: [N, D] tensor of normalized starting states
        sup_net:         trained policy network
        value_net:       trained value network
        bounds_processor:StateBoundsProcessor instance
        sim_steps:       number of simulation steps
        random_paths:    number of random paths per starting state
        dim_x:           index of state dimension for x-axis
        dim_y:           index of state dimension for y-axis
    """
    # Simulate to collect visited states
    S, P, _, _ = foc_optimizer.simulate(
        starting_states, sup_net, value_net,
        bounds_processor, foc_optimizer.Z_trans_tensor,
        sim_steps, random_paths
    )

    # Optionally denormalize for more interpretable plots
    S_denorm = bounds_processor.denormalize(S.detach())

    # Scatter plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(S_denorm[:, dim_x].cpu().numpy(),
                     S_denorm[:, dim_y].cpu().numpy(),
                     c=P.cpu().numpy(), cmap='viridis', alpha=0.6)
    plt.xlabel(f"State dim {dim_x}")
    plt.ylabel(f"State dim {dim_y}")
    plt.title("Reached States in Simulation")
    cbar = plt.colorbar(sc)
    cbar.set_label("Productivity state (y index)")
    plt.grid(True)
    #plt.show()
    savefig_now("reached_states"); plt.close()

@torch.no_grad()
def evaluate_loss(value_net, sup_net, bounds_processor, foc_optimizer,
                  starting_points=200, horizon=10, random_paths=100):
    """
    Run evaluation episode without gradient updates.
    Returns dict of losses comparable to training losses.
    """
    # Sample random starting states
    state_start = torch.zeros(3,dtype=type)#.requires_grad_(True)
    state_start[0] = bounds_processor.normalize_dim(1,0) # 1 junior worker
    state_start[1] = bounds_processor.normalize_dim(1e-3,1) # Tiny positive seniors
    state_start[2] = bounds_processor.normalize_dim(cc.v_grid[10],1)
    #state_start[2] = torch.rand(1, K_v)
    #Or if randomized. 
    starting_states = torch.rand(1, 3,dtype=type) 
    #Add the starting state
    starting_states[0,:] = state_start

    # Simulate environment
    S, P, G_flat, G_starting, R_flat, V_last_bootstrap = foc_optimizer.simulate(
        starting_states, sup_net, value_net, bounds_processor, foc_optimizer.Z_trans_tensor,
        simulation_steps=horizon, random_paths=random_paths
    )

    N = starting_points * random_paths
    T = horizon

    # TD(λ) targets
    tdlam_flat = td_lambda_targets_from_flat(
        R_flat=R_flat,
        S_flat=S,
        P_flat=P,
        target_value_net=value_net,  # use same net here
        T=T, N=N,
        V_last_bootstrap=V_last_bootstrap,
        gamma=p.beta, lam=0.95
    )

    i = torch.arange(S.shape[0])
    pred_values = value_net(S)['values'][i, P]
    μ, σ = tdlam_flat.mean(), tdlam_flat.std(unbiased=False).clamp_min(1e-6)
    tgt_n  = (tdlam_flat - μ) / σ
    pred_n = (pred_values - μ) / σ
    value_eval_loss = nn.HuberLoss()(pred_n, tgt_n)

    # simple sup evaluation: negative mean of returns
    adv_flat = (G_flat - pred_values.detach())
    scale = adv_flat.std(unbiased=False).clamp_min(1e-6).detach()
    sup_eval_loss = - (adv_flat/scale).mean() 
    #sup_eval_loss = -(G_starting).mean()

    return {
        "value_eval": value_eval_loss.item(),
        "sup_eval": sup_eval_loss.item(),
    }

if __name__ == "__main__":
    # Define parameters
    K = 2 #Number of tenure steps
    #Number of states
    K_n = K #K size states
    K_v = K - 1 #K - 1 (ignoring bottom) value states
    K_q = K - 1 #K - 1 (ignoring bottom) quality states. Ignore them for now
    STATE_DIM = K_n + K_v # + K_q #Discrete prod-ty y as multiple outputs
    ACTION_DIM = K_v + 1 # + K_n  # omega + hiring + separations. 
    HIDDEN_DIMS_CRIT = [64,64]
    HIDDEN_DIMS_POL = [64,64]  # Basic architecture. Basically every paper has 2 inner layers, can make them wider though

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.0+
    except Exception:
        pass

    #pref = Preferences(input_param=p_crs)
    cc=ContinuousContract(p_crs()) 
    #cc_J,cc_W,cc_Wstar,omega = cc.J(0) 
    #target_values = tensor(cc_J + cc.rho_grid[ax,:] * cc_W, dtype=type)
    #target_W = tensor(cc_W, dtype=type)
    #NORMALIZE EVERYTHING!!!
    LOWER_BOUNDS = [0, 0 , cc.v_grid[0]] # The state space is (y,n_0,n_1,v_1).
    UPPER_BOUNDS = [20, 100, 1.1 * cc.v_grid[-1]]

    num_episodes= 40000
    minibatch_num = 8
    #Initialize
    bounds_processor = StateBoundsProcessor(LOWER_BOUNDS,UPPER_BOUNDS).to(device)

    
    learning_rate=[1e-3,1e-4]
    value_net, sup_net, optimizer_value, optimizer_sup, scheduler_value, scheduler_sup, foc_optimizer = initialize(bounds_processor,  STATE_DIM, 
    K_n, K_v, HIDDEN_DIMS_CRIT, HIDDEN_DIMS_POL, learning_rate=learning_rate, weight_decay = [1e-4, 3e-4], pre_training_steps=0, num_epochs=num_episodes, minibatch_num=minibatch_num)
    value_net = value_net.to(device)
    sup_net = sup_net.to(device)
    foc_optimizer = foc_optimizer.to(device)
    # Train value function
    print("Training value function...")
    beg=time()
    trained_value, trained_sup = train(
    STATE_DIM, value_net, sup_net, optimizer_value, optimizer_sup, scheduler_value, scheduler_sup,foc_optimizer, bounds_processor,
        num_episodes=num_episodes,
        starting_points_per_iter=1,
        simulation_steps=5,
        minibatch_num=minibatch_num, λ=5.0,use_saved_nets = False    )
    print("Training time:", time()-beg)

    # Evaluate trained model
    #evaluate_plot_sup(trained_value, trained_sup, trained_inf, bounds_processor, num_samples=1000)
    #evaluate_value_function(trained_value, trained_sup, p, LOWER_BOUNDS, UPPER_BOUNDS,cc,target_values,cc_W,omega)

    # Save the model
    torch.save(trained_value.state_dict(), "trained_value_function.pt")
    torch.save(trained_sup.state_dict(), "trained_sup_function.pt")
    print("Model saved")