import math
import time
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim

# ---- TorchRL bits
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
from torchrl.data import Unbounded, Bounded, Composite

from tensordict import TensorDict  
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import Compose, ObservationNorm, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector

from torchrl.modules import (
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from primitives import Parameters
from primitives import Preferences
from ContinuousContract import ContinuousContract
from primitives_CRS import Parameters as p_crs
from probabilities import createPoissonTransitionMatrix
from search_tensor import JobSearchArray
from scipy.stats import lognorm as lnorm
ax = np.newaxis
tensor = torch.tensor
type = torch.float32
# ============ 1) Your simulator adapter (plug your econ code here) ============

@dataclass
class EconBounds:
    # Physical ranges of *raw* state and action components.
    # Use your model’s known safe ranges (or leave None to skip scaling).
    # For actions: we’ll use env.action_spec in [-1, 1] and map to these.
    state_low: torch.Tensor      # [obs_dim]
    state_high: torch.Tensor     # [obs_dim]
    #state_range = state_high - state_low
    vprime_low: torch.Tensor     # [K_v]  (if you output v')
    vprime_high: torch.Tensor    # [K_v]


class EconDetails:
    """Class to compute extra details for the model"""
    def __init__(self, K, p, cc):
        #self.bounds_processor = bounds_processor  # Store bounds_processor
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


class EconModel:
    """
    Thin adapter around your existing model.

    You only need to implement:
      - init_state(batch): Tensor[batch, obs_dim]
      - step(state, action): (next_state, reward, done, info)

    ACTION LAYOUT (example):
      action = [ hiring, v_prime[0], v_prime[1], ... ]
      hiring \in [0,1], v_prime[j] \in [vprime_low[j], vprime_high[j]]

    Map this to your own semantics:
      - If your policy acts on 'hiring' and a vector of promised values v', keep that.
      - If not, redefine action layout to whatever your simulator expects and
        adjust the mapping in EconEnv._act_physical().
    """
    def __init__(self, device: torch.device, dtype: torch.dtype, beta: float, K_n: int, K_v: int, num_y: int, details: EconDetails,bounds: EconBounds):
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.K_n = K_n
        self.K_v = K_v
        self.bounds = bounds
        self.num_y = num_y
        self.details = details
        # ---- Example: import your code here if needed
        # import importlib.util, pathlib
        # spec = importlib.util.spec_from_file_location("econ", "/path/to/mwf_primal_advantage.py")
        # econ = importlib.util.module_from_spec(spec); spec.loader.exec_module(econ)
        # self.m = econ.FOCresidual(...)

    @torch.no_grad()
    def init_state(self, batch_size: int) -> torch.Tensor:
        """
        TODO: sample or construct initial states for 'batch_size' envs.
        Return shape: [batch, obs_dim]
        """
        obs_dim = self.bounds.state_low.numel()
        ## Example: start near the middle of bounds
        #mid = 0.5 * (self.bounds.state_low + self.bounds.state_high)
        #mid = mid.to(self.device, self.dtype)
        starting_states = torch.zeros(batch_size, obs_dim, device=self.device, dtype=self.dtype)

        starting_states[:,0] = 1 #Start with a single junior
        starting_states[:,1:self.K_n] = 0
        starting_states[:,self.K_n:] = 0
        return starting_states
        #return mid.expand(batch_size, obs_dim).clone()

    @torch.no_grad()
    def step(self, state: torch.Tensor, action_phys: torch.Tensor, y_idx: torch.Tensor, y_next_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        TODO: advance your simulator by one period.
        Args:
            state:  [B, obs_dim]  (raw, physical units)
            action_phys: [B, act_dim] with your physical ranges applied
                         action_phys[:, 0]    -> hiring ∈ [0,1]
                         action_phys[:, 1: ]  -> v' components in [vprime_low, vprime_high]
        Returns:
            next_state: [B, obs_dim]  (raw)
            reward:     [B]           (float)
            done:       [B] in {0,1}  (episode termination if you use finite-horizon)
            info:       dict (you can add diagnostics)
        """
        B = state.shape[0]
        sizes = state[:,:self.K_n]
        v = state[:,self.K_n:]
        # ---- Replace everything below with your one-step economics.
        # The snippet just acts as a placeholder with stable numbers.
        # Example dynamics (dummy): n_{t+1} is partial adjustment via hiring, v' becomes part of state.
        hiring = action_phys[:, 0]                            # [B, 1]
        vprime_flat = action_phys[:, 1: 1 + self.K_v * self.num_y]               # [B, K_v]
        vprime_sched = vprime_flat.view(B, self.K_v, self.num_y)             # [B, K_v, num_y]
        #Take expected v_prime
        v_prime_exp_all = torch.einsum("bky,yz->bkz", vprime_sched, self.details.Z_trans_tensor)  # [N, K_v, Z]
        iN = torch.arange(B)
        v_prime_exp     = v_prime_exp_all[iN, :, y_idx]                           # [N, K_v]
        re, pc = self.details.getWorkerDecisions(v_prime_exp) 
        wages = torch.zeros(B,self.K_n)
        wages[:,1:] = self.details.pref.inv_utility(v - self.details.p.beta * (v_prime_exp + re))
        wages[:,:1] = self.details.pref.inv_utility(self.details.v_0 - self.details.p.beta*((v_prime_exp+re)))       
        # Suppose your state = [n_0, n_1, v_component]
        next_state = state.clone()
        # toy: total employment: n_total = n0 + n1 ; next n1 = hiring * n_total

        next_state[:,0] = hiring
        next_state[:,1] = sizes.sum(dim=1) * pc.squeeze(1)
        next_state[:,self.K_n:] = vprime_sched[iN,:,y_next_idx] #Need to decide future y before this

        # toy reward: concave output of employment minus a penalty on large v'
        reward = self.details.fun_prod[y_idx.detach().long()] * self.details.production(next_state[:,:self.K_n].sum(dim=1)) - self.details.p.hire_c * hiring - (wages*sizes).sum(dim=1)
        reward = reward.to(self.dtype)

        # finite-horizon episodes? set a horizon in EconEnv; we set done=False here
        done = torch.zeros(B, device=state.device, dtype=torch.bool)
        info = {}
        return next_state, reward, done, info


# ============ 2) TorchRL Environment wrapper ============

class EconEnv(EnvBase):
    """
    TorchRL Env that wraps your EconModel. Actions come in as [-1, 1]^A,
    and we map them to physical ranges before calling model.step(...).
    """
    def __init__(
        self,
        model: EconModel,
        horizon: int,
        P_y: torch.Tensor,                # [Y, Y] row-stochastic Markov matrix P(y'|y)
        pi0: Optional[torch.Tensor],      # [Y] initial distribution (optional; else uniform)
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        sample_yprime: bool = True,       # sample y' (True) or use expectation (False)
    ):
        super().__init__(device=device, batch_size=torch.Size([1]))
        self.dtype = dtype
        self.model = model
        self.horizon = horizon
        self.sample_yprime = sample_yprime

        self.num_y = model.num_y
        self.obs_cont_dim = model.bounds.state_low.numel()
        obs_dim = self.obs_cont_dim + self.num_y
        self.obs_dim = obs_dim
        self._action_dim = action_dim

        # Store transition matrix & initial distribution
        self.P_y = P_y.to(device, dtype=torch.float32)  # sampling uses float32 probs
        if pi0 is None:
            self.pi0 = torch.full((self.num_y,), 1.0 / self.num_y, device=device, dtype=torch.float32)
        else:
            self.pi0 = pi0.to(device, dtype=torch.float32)
        
        # --- specs
        obs_spec = Unbounded(shape=(obs_dim,), device=device, dtype=dtype)
        act_spec = Bounded(
            low=-torch.ones(action_dim, device=device, dtype=dtype),
            high= torch.ones(action_dim, device=device, dtype=dtype),
            shape=(self.batch_size[0],action_dim),
            device=device,
            dtype=dtype,
        )
        rew_spec = Unbounded(shape=(1,), device=device, dtype=dtype)
        done_spec = Bounded(0, 1, shape=(1,), dtype=torch.bool, device=device)

        self.observation_spec = Composite(observation=obs_spec, shape=self.batch_size).to(device)
        self.action_spec = act_spec
        self.reward_spec = rew_spec
        self.done_spec = done_spec

        #self.state = None
        self.state_cont = None   # [B, obs_cont_dim]        
        self.y_idx = None        # [B] int64
        self._t = None

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
        return seed

    def _obs_from_raw(self, state_cont: torch.Tensor, y_idx: torch.Tensor) -> torch.Tensor:
        """Concatenate continuous raw state with one-hot(y)."""
        B = state_cont.shape[0]
        one_hot = torch.nn.functional.one_hot(y_idx, num_classes=self.num_y).to(state_cont.dtype)
        obs = torch.cat([state_cont, one_hot], dim=-1)
        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        batch = 1 if tensordict is None or tensordict.batch_size == torch.Size([]) else tensordict.batch_size[0]
        self.state_cont = self.model.init_state(batch)
        # sample initial y ~ pi0
        self.y_idx = torch.multinomial(self.pi0.expand(batch, -1), num_samples=1).squeeze(1)
        self._t = torch.zeros(batch, device=self.device, dtype=torch.long)
        obs = self._obs_from_raw(self.state_cont, self.y_idx)  # [B, obs_dim]
        # in _reset
        assert obs.shape[0] == batch and obs.ndim == 2, f"obs shape {obs.shape}"
        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td.set("observation", obs)
        td.set("done", torch.zeros(batch, 1, device=self.device, dtype=torch.bool))
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get("action")  # [-1, 1] scale from policy
        # Map to physical ranges your simulator expects
        action_phys = self._act_physical(action)

        # Sample next y' or use expectation
        B = action.shape[0]
        cur_y = self.y_idx  # [B]
        P_rows = self.P_y[cur_y]  # [B, Y]

        if self.sample_yprime:
            y_next = torch.multinomial(P_rows, num_samples=1).squeeze(1)  # [B]
        else:
            # pick most likely y' as a deterministic representative (or keep expectation inside model)
            y_next = torch.argmax(P_rows, dim=1)

        next_state_cont, r, done, info = self.model.step(self.state_cont, action_phys, cur_y, y_next)

        self._t += 1
        # finite horizon done
        horizon_done = (self._t >= self.horizon).unsqueeze(-1)

        # Update raw internal state
        self.state_cont = next_state_cont
        self.y_idx = y_next
        # Build next observation (raw -> transformed env will normalize)
        next_obs = self._obs_from_raw(self.state_cont, self.y_idx)
        # in _step
        assert r.ndim == 1 and r.shape[0] == B, f"reward shape {r.shape}"
        assert done.ndim == 1 and done.shape[0] == B, f"done shape {done.shape}"

        out = TensorDict({}, batch_size=[B], device=self.device)
        out.set("next", TensorDict({
            "observation": next_obs,
            "reward": r.unsqueeze(-1),
            "done": torch.logical_or(done.unsqueeze(-1), horizon_done),
        }, batch_size= self.batch_size, device=self.device))
        return out

    def _act_physical(self, a: torch.Tensor) -> torch.Tensor:
        """
        Map action a in [-1,1]^A to physical:
          a0 -> hiring in [0,1]
          a1: -> v'[0] in [vlow, vhigh]
          ...
        Adjust if your action layout differs.
        """
        B = a.shape[0]
        K_v, num_y = self.model.K_v, self.model.num_y
        # hiring: [-1,1] -> [stateLow[0],state_high[0]]
        hiring = 0.5 * (a[:, :1] + 1.0) * (self.model.bounds.state_high[0] - self.model.bounds.state_low[0]) + self.model.bounds.state_low[0]

        # vprime: [-1,1] -> [vlow, vhigh] (elementwise)
        vflat   = a[:, 1: 1 + K_v * num_y]
        vsched  = vflat.view(B, K_v, num_y)

        vlow = self.model.bounds.vprime_low.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)
        vhigh = self.model.bounds.vprime_high.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)
        # affine map per k, broadcast over y′
        vsched_phys = 0.5 * (vsched + 1.0) * (vhigh - vlow)[None, :, None] + vlow[None, :, None]

        return torch.cat([hiring, vsched_phys.reshape(B, K_v * num_y)], dim=-1)


# ============ 3) Build actor–critic and PPO training ============

def make_actor(obs_dim: int, act_dim: int, hidden=(64, 64)) -> ProbabilisticActor:
    # Produce loc, scale for a TanhNormal (stochastic at train; mean at eval)
    net = MLP(in_features=obs_dim, out_features=2 * act_dim, depth=len(hidden), num_cells=hidden, activation_class=nn.SiLU)
    extractor = NormalParamExtractor()  # splits into loc & scale (pre-softplus)
    td_module = TensorDictModule(nn.Sequential(net, extractor), in_keys=["observation"], out_keys=["loc", "scale"])

    actor = ProbabilisticActor(
        module=td_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"min": -1.0, "max": 1.0},  # action_spec already in [-1,1]
        return_log_prob=True,  # adds "sample_log_prob" to tensordict
    )
    return actor


def make_critic(obs_dim: int, hidden=(64, 64)) -> ValueOperator:
    net = MLP(in_features=obs_dim, out_features=1, depth=len(hidden), num_cells=hidden, activation_class=nn.SiLU)
    critic = ValueOperator(module=net, in_keys=["observation"])
    return critic


def flatten_batch_time(td: TensorDict) -> TensorDict:
    # Collector returns [T, B, ...]; PPO trains on flat [T*B, ...]
    return td.reshape(-1)

def make_env_fn(model, horizon, P_y, pi0, action_dim, device, dtype):
    def _make():
        base_env = EconEnv(model=model, horizon=horizon, P_y=P_y, pi0=pi0, action_dim=action_dim, device=device, dtype=dtype, sample_yprime=True)
        obs_dim = base_env.obs_dim  # = cont_state_dim + num_y (one-hot)

        transforms = Compose(
            ObservationNorm(
                in_keys=[("next", "observation")],
                loc=torch.zeros(obs_dim, device=device, dtype=dtype),
                scale=torch.ones(obs_dim, device=device, dtype=dtype),
                standard_normal=True,   # keep running update of stats
            ),
            #StepCounter(max_steps=horizon),
        )
        return TransformedEnv(base_env, transforms).to(device)
    return _make

def ppo_train(
    model: EconModel,
    P_y: torch.Tensor,                 # [Y, Y] Markov matrix
    pi0: Optional[torch.Tensor] = None,
    horizon: int = 64,
    frames_per_batch: int = 4096,
    total_frames: int = 500_000,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    ent_coef: float = 0.0,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 10,
    minibatch_size: int = 1024,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32, p = None
):
    torch.manual_seed(seed)
    device = torch.device(device)
    obs_dim = model.bounds.state_low.numel() + model.num_y
    act_dim = 1 + model.K_v * model.num_y  # hiring + v' components (adjust if you change action layout)

    # ---- Env factory with transforms
    env_maker = make_env_fn(model, horizon, P_y, pi0, act_dim, device, dtype)
    env = env_maker()
    check_env_specs(env)
    # ---- Env with transforms
    #base_env = EconEnv(model=model, horizon=horizon, action_dim=act_dim, device=device, dtype=dtype)
    #transforms = Compose(
    #    ObservationNorm(in_keys=["observation"], standard_normal=True),  # running z-score
    #    StepCounter(max_steps=horizon),
    #)
    #env = TransformedEnv(base_env, transforms).to(device)

    # ---- Policy & value
    actor = make_actor(obs_dim, act_dim).to(device)
    critic = make_critic(obs_dim).to(device)

    # ---- Collector (single process; increase num_workers with Parallel collectors if needed)
    policy_for_collection = actor
    collector = SyncDataCollector(
        create_env_fn=env_maker,
        policy=policy_for_collection,
        frames_per_batch=frames_per_batch,
        #max_frames_per_traj=total_frames,
        device=device,
        storing_device=device,
        init_random_frames=0,
        split_trajs=True,  # better for GAE on variable-length episodes
    )

    # ---- Advantage and PPO loss
    advantage = GAE(gamma=gamma, lmbda=gae_lambda, value_network=critic)
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=clip_eps,
        entropy_coef=ent_coef,
        critic_coef=value_coef,
        # normalize_advantage=True  # TorchRL >= 0.5 has internal norm; else do it manually
    )

    optim_all = optim.Adam([
        {"params": actor.parameters(), "lr": lr},
        {"params": critic.parameters(), "lr": lr},
    ])

    # ---- Training loop
    pbar = tqdm(total=total_frames, desc="PPO")
    frames_seen = 0
    best_eval = -float("inf")

    for batch in collector:
        # batch: TensorDict with leading dims [T, B]
        frames = batch.numel()
        frames_seen += frames
        pbar.update(frames)

        # Compute advantages & targets
        with torch.no_grad():
            batch = advantage(batch)  # adds "advantage" and "value_target"
            # optional: normalize advantages
            adv = batch.get("advantage")
            batch.set_("advantage", (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8))

        # Store old log-prob for the ratio
        old_log_prob = batch.get("sample_log_prob").detach()
        batch.set("old_log_prob", old_log_prob)

        # Flatten time/batch for minibatching
        flat = flatten_batch_time(batch)

        # ---- PPO epochs / minibatches
        for _ in range(ppo_epochs):
            # random minibatch indices
            perm = torch.randperm(flat.batch_size[0], device=device)
            for i in range(0, flat.batch_size[0], minibatch_size):
                idx = perm[i : i + minibatch_size]
                sub = flat[idx]

                # PPO loss returns a dict with total loss and components
                loss_dict = loss_module(sub)
                loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]


                loss.backward()
                clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
                optim_all.step()
                optim_all.zero_grad(set_to_none=True)
        # ---- quick deterministic eval on the same env (mean action)
        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            eval_td = env.reset()
            ep_return = torch.zeros(1, device=device)
            for _ in range(horizon):
                eval_td = actor(eval_td)     # fills "action"
                eval_td = env.step(eval_td)  # applies step; "next" populated
                ep_return += eval_td.get("next")["reward"].squeeze(-1)
                eval_td = eval_td.get("next").rename(None)
            mean_return = ep_return.mean().item()
            best_eval = max(best_eval, mean_return)
        pbar.set_postfix({"R_eval": f"{mean_return:.2f}", "R_best": f"{best_eval:.2f}"})

        if frames_seen >= total_frames:
            break

    pbar.close()
    # Return trained modules and the transformed env (it holds obs-normalization stats)
    return actor, critic, env


# ============ 4) Script entry point with sane defaults ============

def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dtype = torch.float32
    p = Parameters()
    # ---- Fill these with your real problem sizes & ranges ----
    K = 2
    K_n = K
    K_v = K - 1             # you used this layout in your current code
    obs_dim = K_n + K_v     # e.g., [n0, n1, v_level]
    act_dim = 1 + K_v * p.num_z       # hiring + v' components
    cc=ContinuousContract(p_crs()) 
    # Example physical bounds (replace with yours):
    # (These are *raw* units, not normalized.)
    state_low = torch.tensor([0, 0 , cc.v_grid[0]])
    state_high = torch.tensor([10, 20, cc.v_grid[-1]])
    vprime_low = torch.full((K_v,), float(cc.v_grid[0]),  dtype=dtype, device=device)     # length K_v
    vprime_high = torch.full((K_v,), float(cc.v_grid[-1]),  dtype=dtype, device=device)   # length K_v

    beta = p.beta  # set to your discount factor
    bounds = EconBounds(state_low=state_low, state_high=state_high,
                        vprime_low=vprime_low, vprime_high=vprime_high)

    details = EconDetails(K,p,cc)
    model = EconModel(device=device, dtype=dtype, beta=beta, K_n=K_n, K_v=K_v, num_y=p.num_z, details=details, bounds=bounds)
    P_y = model.details.Z_trans_tensor
    # Optional initial distribution (uniform here)
    pi0 = torch.full((p.num_z,), 1.0 / p.num_z)

    actor, critic, env = ppo_train(
        model=model,
        P_y=P_y,                    # y transition probability
        pi0=pi0,                    # y initial distribution
        horizon=64,                 # episode length
        frames_per_batch=4096,      # ~64 steps x 64 envs (collector runs env in batch)
        total_frames=200_000,       # raise this for better performance
        gamma=beta,
        gae_lambda=0.95,
        clip_eps=0.2,
        lr=3e-4,
        ent_coef=0.01,              # small entropy helps exploration, can decay to 0
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        minibatch_size=1024,
        seed=0,
        device=str(device),
        dtype=dtype, p = p
    )

    # Save trained weights + obs normalizer stats
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, "ppo_agent.pt")
    print("Saved to ppo_agent.pt")


if __name__ == "__main__":
    main()
