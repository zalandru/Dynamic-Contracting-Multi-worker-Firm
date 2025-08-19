import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict

import torch
import torch.nn as nn

# === External econ modules (from your project) ===
from primitives import Parameters, Preferences
from ContinuousContract import ContinuousContract
from primitives_CRS import Parameters as p_crs
from probabilities import createPoissonTransitionMatrix
from search_tensor import JobSearchArray
from scipy.stats import lognorm as lnorm

# === Gymnasium / SB3 ===
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
import matplotlib.pyplot as plt

ax = np.newaxis
DTYPE = torch.float32

def tensor(x, device, dtype=DTYPE):
    return torch.as_tensor(x, device=device, dtype=dtype)

# ========================= 1) Econ helpers (same logic, cleaned) =========================

@dataclass
class EconBounds:
    state_low: torch.Tensor      # [obs_cont_dim]
    state_high: torch.Tensor     # [obs_cont_dim]
    vprime_low: torch.Tensor     # [K_v]
    vprime_high: torch.Tensor    # [K_v]


class EconDetails:
    def __init__(self, device: torch.device, K, p, cc):
        self.device = device
        self.K = K
        self.p = p
        self.pref = Preferences(input_param=self.p)
        self.Z_grid = self._construct_z_grid()

        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = torch.tensor(self.fun_prod, dtype=DTYPE, device=self.device)
        self.unemp_bf = self.p.u_bf_m

        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)
        self.Z_trans_tensor = torch.tensor(self.Z_trans_mat, dtype=torch.float32, device=self.device)

        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max().item(), self.p.num_v)
        self.rho_grid = torch.tensor(1 / self.pref.utility_1d(self.w_grid), dtype=DTYPE, device=self.device)

        self.v_grid = np.linspace(
            np.divide(self.pref.utility(self.unemp_bf), 1 - self.p.beta),
            np.divide(self.pref.utility(self.fun_prod.max().item()), 1 - self.p.beta),
            self.p.num_v,
        )
        self.v_0 = torch.tensor(self.v_grid[0], dtype=DTYPE, device=self.device)
        self.simple_J = torch.zeros((self.p.num_z,self.p.num_v),device=device)
        self.simple_J = (
            (self.fun_prod[:, ax] - self.pref.inv_utility(torch.tensor(self.v_grid[ax, :] * (1 - self.p.beta), dtype=DTYPE, device=self.device)))
            / (1 - self.p.beta)
        )

        self.prob_find_vx = self.p.alpha * np.power(
            1 - np.power(np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0 - 1, :].cpu().numpy(), 1.0)), self.p.sigma),
            1 / self.p.sigma,
        )
        if cc is None:
            self.js = JobSearchArray(device=self.device)
            self.js.update(self.v_grid[:], self.prob_find_vx)
        else:
            self.js = cc.js
    @torch.no_grad()
    def _construct_z_grid(self):
        exp_z = np.linspace(0, 1, self.p.num_z + 2)[1:-1]
        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)
    @torch.no_grad()
    def production(self, sum_n):
        return sum_n ** self.p.prod_alpha
    @torch.no_grad()
    def getWorkerDecisions(self, EW1, employed=True):
        pe, re = self.js.solve_search_choice(EW1)
        assert (~torch.isnan(pe)).all(), "pe is NaN"
        assert (pe <= 1).all(), "pe <= 1 violated"
        assert (pe >= -1e-10).all(), "pe < 0"
        if employed:
            pe = pe * self.p.s_job
            re = re * self.p.s_job
        pc = (1 - pe)
        return re, pc
    @torch.no_grad()
    def get_U(self):
        #Iterate on U
        U = self.pref.utility(self.unemp_bf) / ( 1 - self.p.beta)
        critU = 1
        while critU >= 1e-3:
            U2 = U
            ru, _ = self.getWorkerDecisions(U, employed=False)
            U = self.pref.utility(self.unemp_bf) + self.p.beta * ( U + ru)
            #Or, direct formulation
            #U = (self.pref.utility(self.unemp_bf) + self.p.beta * ru) / ( 1 - self.p.beta)
            U = 0.4 * U + 0.6 * U2
            critU = torch.abs(U - U2)
        self.U=U
        return self

class EconModel:
    def __init__(self, device: torch.device, dtype: torch.dtype, beta: float, K_n: int, K_v: int, num_y: int, details: EconDetails, bounds: EconBounds):
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.K_n = K_n
        self.K_v = K_v
        self.num_y = num_y
        self.details = details
        self.bounds = bounds

        #Get U
        self.details.get_U()
    @torch.no_grad()
    def init_state(self, batch_size: int) -> torch.Tensor:
        obs_cont_dim = self.bounds.state_low.numel()
        s = torch.zeros(batch_size, obs_cont_dim, device=self.device, dtype=self.dtype)
        s[:, 0] = 1.0
        s[:, 1:self.K_n] = 1e-2 #Just a few seniors so that the value has some real impact
        s[:, self.K_n : self.K_n + self.K_v] = (self.details.v_grid[-1] + self.details.v_grid[0]) / 2#Since my vprime is delta parametrized, I want to incentivize it to be at least a bit positive
        s[:, self.K_n + self.K_v:] = self.details.p.q_0
        return s

    @torch.no_grad()
    def step(self, state: torch.Tensor, action_phys: torch.Tensor, y_idx: torch.Tensor, y_next_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        B = state.shape[0]
        sizes = state[:, :self.K_n]
        v = state[:, self.K_n:self.K_n + self.K_v]
        q = torch.zeros(state.shape[0], state[:,self.K_n+self.K_v:].shape[1] + 1)
        q[:,0] = self.details.p.q_0
        q[:,1:] = state[:,self.K_n+self.K_v:]
        hiring = action_phys[:, 0]
        sep = action_phys[:, 1: 1 + self.K_n]
        vprime_flat = action_phys[:, 1 + self.K_n: 1 + self.K_n + self.K_v * self.num_y]
        vprime_sched = vprime_flat.view(B, self.K_v, self.num_y)

        v_prime_exp_all = torch.einsum("bkz,yz->bky", vprime_sched, self.details.Z_trans_tensor)
        iN = torch.arange(B, device=state.device)
        v_prime_exp = v_prime_exp_all[iN, :, y_idx]
        re, pc = self.details.getWorkerDecisions(v_prime_exp)

        # Compute argument to inv_utility (which is exp for log utility)
        #all_v = torch.zeros(B, self.K_n, device=state.device, dtype=self.dtype)
        #all_v[:, 0] = self.details.v_0
        #all_v[:, 1:] = v
        #u_emp = v -                  self.details.p.beta * (sep[:,0] * self.details.U + (1-sep[:,0]) * (v_prime_exp + re))
        #u_unemp = self.details.v_0 - self.details.p.beta * (sep[:,1] * self.details.U + (1-sep[:,1]) * (v_prime_exp + re))
        wages = torch.zeros(B, self.K_n, device=state.device, dtype=self.dtype)
        wages[:, 1:] = self.details.pref.inv_utility(v - self.details.p.beta * (sep[:,1] * self.details.U + (1-sep[:,1]) * (v_prime_exp + re)))
        wages[:, :1] = self.details.pref.inv_utility(self.details.v_0 - self.details.p.beta * (sep[:,0] * self.details.U + (1-sep[:,0]) * (v_prime_exp + re)))
        #if not torch.isfinite(wages).all():
        #    print("u_emp range:", float(u_emp.min()), float(u_emp.max()))
        #    print("u_unemp range:", float(u_unemp.min()), float(u_unemp.max()))

        next_state = state.clone()
        next_state[:, 0] = hiring
        next_state[:, 1] = sizes.sum(dim=1) * pc.squeeze(1)
        next_state[:, self.K_n:self.K_n + self.K_v] = vprime_sched[iN, :, y_next_idx]
        #Future q
        good_juns = sizes[:,0] * pc.squeeze(1) * torch.min( 1 - sep[:,0], torch.tensor(self.details.p.q_0, device=state.device, dtype=self.dtype) )
        good_sens = sizes[:,1] * pc.squeeze(1) * torch.min( 1 - sep[:,1], q[:,1] )
        q_1 = (good_sens + good_juns).unsqueeze(1) / next_state[:,1:self.K_n]
        q_1[q_1 > 1] = 1 #Should never be the case bah, just in case lol.
        next_state[:, self.K_n + self.K_v:] = q_1
        
        tot_size_adj = (sizes * ( q + ( 1 - q ) * self.details.p.prod_q)).sum(dim=1)
        reward = (
            self.details.fun_prod[y_idx.detach().long()] * self.details.production(tot_size_adj)
            - self.details.p.hire_c * hiring
            - (wages * sizes).sum(dim=1)
        ).to(self.dtype)
        #with torch.no_grad():
            # how much room the policy had to keep wages small?
        #    gap_needed = (v / self.details.p.beta) - v_prime_exp    # if >0, couldn't reach v'/≈v/β
        #    print("max(v)", float(v.max()), "max(v')", float(v_prime_exp.max()),
        #      "max(gap_needed)", float(gap_needed.max()))
        done = torch.zeros(B, device=state.device, dtype=torch.bool)
        return next_state, reward, done, {}


# ========================= 2) Gymnasium environment =========================

class EconGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model: EconModel, P_y: torch.Tensor, pi0: Optional[torch.Tensor], horizon: int = 64, device: str = "cpu", dtype: torch.dtype = DTYPE, sample_yprime: bool = True):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model
        self.P_y = P_y.to(self.device, dtype=torch.float32)
        self.horizon = int(horizon)
        self.sample_yprime = sample_yprime
        self.num_y = model.num_y
        self.obs_cont_dim = model.bounds.state_low.numel()
        self.obs_dim = self.obs_cont_dim + self.num_y
        self.K_v = model.K_v
        self.K_n = model.K_n

        if pi0 is None:
            self.pi0 = torch.full((self.num_y,), 1.0 / self.num_y, device=self.device, dtype=torch.float32)
        else:
            self.pi0 = pi0.to(self.device, dtype=torch.float32)

        # observation: concatenation of continuous state and one-hot y
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # action: [hiring, v'(K_v x num_y)] in [-1,1]
        self.act_dim = 1 + self.K_n + self.K_v * self.num_y #hiring + sep(K_n) + v'(K_v x num_y)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # internal state
        self.t = 0
        self.state_cont = None  # torch [1, obs_cont_dim]
        self.y_idx = None       # int

    # ---- helpers ----
    def _act_physical(self, a: torch.Tensor) -> torch.Tensor:
        B = a.shape[0]
        #Map from [-1,1] to real values
        # Hiring to [0,state_high[0]]
        hiring = 0.5 * (a[:, :1] + 1.0) * (self.model.bounds.state_high[0] - self.model.bounds.state_low[0]) + self.model.bounds.state_low[0]
        #Sep to [0,1]. btw, maybe I'll want to "average" it out to zero?
        sep = 0.5 * (a[:,1: 1 + self.K_n] + 1.0)
        #Future value
        vflat = a[:, 1 + self.K_n: 1 + self.K_n + self.K_v * self.num_y]
        vsched = vflat.view(B, self.K_v, self.num_y)

        # current v (state) for delta parameterization
        v_curr = self.model_state_for_env()  # see helper below

        vlow  = self.model.bounds.vprime_low.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)   # [B, K_v]
        vhigh = self.model.bounds.vprime_high.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)  # [B, K_v]

        # allowable symmetric delta around current v while staying in [vlow, vhigh]
        dmax = torch.minimum(vhigh - v_curr, v_curr - vlow).clamp_min(1e-6)  # [B, K_v]

        # map raw [-1,1] → [-dmax,+dmax]
        dv = vsched * dmax[:, :, None]    # broadcast over num_y

        vsched_phys = (v_curr[:, :, None] + dv).clamp(vlow[:, :, None], vhigh[:, :, None])
        return torch.cat([hiring, sep, vsched_phys.reshape(B, self.K_v * self.num_y)], dim=-1)

    def model_state_for_env(self):
        # pull the current v from self.state_cont: shape [1, obs_cont_dim]
        # here K_v = 1 so we take [:, self.model.K_n:]
        v_curr = self.state_cont[:, self.model.K_n:self.model.K_n + self.model.K_v]  # [B, K_v]
        return v_curr
    def _obs_from_raw(self, state_cont: torch.Tensor, y_idx: int) -> np.ndarray:
        one_hot = torch.nn.functional.one_hot(torch.tensor([y_idx], device=self.device), num_classes=self.num_y).to(state_cont.dtype)
        obs = torch.cat([state_cont, one_hot], dim=-1)[0]  # [obs_dim]
        return obs.detach().cpu().numpy().astype(np.float32)

    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.state_cont = self.model.init_state(1).to(self.device)
        self.y_idx = torch.multinomial(self.pi0, num_samples=1).item()
        obs = self._obs_from_raw(self.state_cont, self.y_idx)
        return obs, {}

    def step(self, action: np.ndarray):
        a = torch.as_tensor(action, device=self.device, dtype=self.dtype).unsqueeze(0)  # [1, act_dim]
        action_phys = self._act_physical(a)

        # sample next y
        row = self.P_y[self.y_idx]  # [num_y]
        if self.sample_yprime:
            y_next = torch.multinomial(row, num_samples=1).item()
        else:
            y_next = torch.argmax(row).item()

        next_state_cont, reward_t, done, _ = self.model.step(self.state_cont, action_phys, torch.tensor([self.y_idx], device=self.device), torch.tensor([y_next], device=self.device))

        self.state_cont = next_state_cont
        self.y_idx = y_next
        obs = self._obs_from_raw(self.state_cont, self.y_idx)

        self.t += 1
        terminated = False
        truncated = (self.t >= self.horizon)
        reward = float(reward_t.item())
        info: Dict = {}
        return obs, reward, terminated, truncated, info




#========================= 3) Helpers for SB3 inference and plotting =========================

class PlotCallback(BaseCallback):
    def __init__(self, plot_env, make_env, dim_index, grid_min, grid_max,
                 grid_points=200, plot_every_ts=200_000, y_idx=None):
        super().__init__()
        self.plot_env = plot_env
        self.make_env = make_env
        self.dim_index = dim_index
        self.grid = np.linspace(grid_min, grid_max, grid_points)
        self.grid_points = grid_points
        self.y_idx = y_idx
        self.plot_every_ts = int(plot_every_ts)
        self._last_plot_ts = -10**18  # force an initial plot

    # ---------- helpers (unchanged logic) ----------
    def _vn(self) -> VecNormalize | None:
        # Do NOT toggle training flags here; we just read stats
        return self.model.get_vec_normalize_env()

    def _normalize_obs(self, obs_np: np.ndarray) -> np.ndarray:
        vn = self._vn()
        if vn is None:
            return obs_np
        mean, var, eps = vn.obs_rms.mean, vn.obs_rms.var, vn.epsilon
        clip = getattr(vn, "clip_obs", np.inf)
        return np.clip((obs_np - mean) / np.sqrt(var + eps), -clip, clip)

    def _denorm_value(self, v_norm: np.ndarray) -> np.ndarray:
        vn = self._vn()
        if vn is None or not getattr(vn, "norm_reward", True):
            return v_norm
        ret_std = float(np.sqrt(vn.ret_rms.var + vn.epsilon))
        return v_norm * ret_std

    def _value_and_action(self, obs_raw_np: np.ndarray):
        obs_norm_np = self._normalize_obs(obs_raw_np)
        pol = self.model.policy
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_norm_np, device=pol.device, dtype=torch.float32)
            if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
            v_norm = pol.predict_values(obs_t).squeeze(-1).cpu().numpy()
        acts_model, _ = self.model.predict(obs_norm_np, deterministic=True)
        return self._denorm_value(v_norm), acts_model

    def _actions_to_physical_from_obs(self, actions_np: np.ndarray, obs_np: np.ndarray) -> np.ndarray:
        env = self.plot_env
        a = torch.as_tensor(actions_np, dtype=env.dtype, device=env.device)
        B = a.shape[0]
        hiring = 0.5 * (a[:, :1] + 1.0) * (env.model.bounds.state_high[0] - env.model.bounds.state_low[0]) + env.model.bounds.state_low[0]
        sep = 0.5 * (a[:, 1: 1 + env.K_n] + 1.0)
        vflat = a[:, 1 + env.K_n: 1 + env.K_n + env.K_v * env.num_y]
        vsched = vflat.view(B, env.K_v, env.num_y)
        obs_cont = torch.as_tensor(obs_np[:, :env.obs_cont_dim], dtype=env.dtype, device=env.device)
        v_curr = obs_cont[:, env.model.K_n : env.model.K_n + env.K_v]
        vlow  = env.model.bounds.vprime_low.to(env.device, env.dtype).unsqueeze(0).expand(B, -1)
        vhigh = env.model.bounds.vprime_high.to(env.device, env.dtype).unsqueeze(0).expand(B, -1)
        dmax = torch.minimum(vhigh - v_curr, v_curr - vlow).clamp_min(1e-6)
        dv = vsched * dmax[:, :, None]
        vsched_phys = (v_curr[:, :, None] + dv).clamp(vlow[:, :, None], vhigh[:, :, None])
        return torch.cat([hiring, sep, vsched_phys.reshape(B, env.K_v * env.num_y)], dim=-1).cpu().numpy()

    def _make_obs_grid(self):
        env = self.plot_env
        base = env.model.init_state(1)[0].cpu().numpy()
        y_idx = self.y_idx
        if y_idx is None:
            try: y_idx = int(env.model.details.p.z_0 - 1)
            except Exception: y_idx = 0
        oh = np.zeros(env.num_y, dtype=np.float32); oh[y_idx] = 1.0
        obs_mat = np.repeat(np.concatenate([base, oh]).reshape(1, -1), len(self.grid), axis=0)
        obs_mat[:, self.dim_index] = self.grid
        return obs_mat, y_idx

    def _make_fig(self, obs_mat, vals_true, acts_phys, y_idx):
        y_vec = np.full(obs_mat.shape[0], y_idx, dtype=int)
        hire, sep, vprime0 = extract_policy_components(acts_phys, self.plot_env.num_y, self.plot_env.K_n, self.plot_env.K_v, y_vec)
        fig = plt.figure(figsize=(12, 3.6))
        ax = fig.add_subplot(1, 4, 1); ax.plot(self.grid, vals_true); ax.set_title(f"Value (true) vs state[{self.dim_index}]"); ax.set_xlabel(f"state[{self.dim_index}]"); ax.set_ylabel("V")
        ax = fig.add_subplot(1, 4, 2); ax.plot(self.grid, hire);     ax.set_title("Hiring (physical)"); ax.set_xlabel(f"state[{self.dim_index}]")
        ax = fig.add_subplot(1, 4, 3); ax.plot(self.grid, sep);  ax.set_title("v' (k=0) (physical)"); ax.set_xlabel(f"state[{self.dim_index}]")        
        ax = fig.add_subplot(1, 4, 4); ax.plot(self.grid, vprime0);  ax.set_title("v' (k=0) (physical)"); ax.set_xlabel(f"state[{self.dim_index}]")
        fig.tight_layout()
        return fig

    def _plot_once(self):
        obs_mat, y_idx = self._make_obs_grid()
        vals_true, acts_model = self._value_and_action(obs_mat)
        acts_phys = self._actions_to_physical_from_obs(acts_model, obs_mat)
        fig = self._make_fig(obs_mat, vals_true, acts_phys, y_idx)
        self.logger.record("plots/value_policy_grid", Figure(fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))

    # --------- schedule on timesteps, not on calls ---------
    def _maybe_plot(self, force=False):
        if force or (self.num_timesteps - self._last_plot_ts) >= self.plot_every_ts:
            self._plot_once()
            self._last_plot_ts = self.num_timesteps

    def _on_training_start(self) -> None:
        self._maybe_plot(force=True)
        return True

    def _on_step(self) -> bool:
        self._maybe_plot(force=False)
        return True


def make_gym_env(model, P_y, pi0, horizon, device):
    def _make():
        env = EconGymEnv(model=model, P_y=P_y, pi0=pi0, horizon=horizon, device=device)
        #return Monitor(env)
        return env
    return _make


def extract_policy_components(actions_np: np.ndarray, num_y: int, K_n: int, K_v, y_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """From normalized actions in [-1,1], extract hiring, sep and v_prime for current y (k=0).
    actions_np: (N, 1 + K_n + K_v * num_y)
    y_idx: (N,) ints inferred from obs one-hot when rolling out.
    Returns (hiring_norm, vprime0_norm).
    """
    hiring = actions_np[:, 0]
    sep = actions_np[:, 1: 1 + K_n]
    idx = 1 + K_n + K_v * (0 * num_y + y_idx)
    vprime0 = actions_np[np.arange(actions_np.shape[0]), idx]
    return hiring, sep, vprime0



class SB3InferenceAdapter:
    """
    Handles:
      - loading VecNormalize stats
      - normalizing obs for the model
      - querying value and action
      - denormalizing the value back to true units (if reward was normalized)
    """
    def __init__(
        self,
        model: PPO,
        make_env: Callable,            # same factory you used for training
        vecnorm_path: str,             # path saved via vec_env.save("...")
        reward_was_normalized: bool,   # set True if training used norm_reward=True
    ):
        self.model = model
        self.policy = model.policy
        # load + freeze stats (no reward norm during inference)
        dummy = DummyVecEnv([make_env])
        self.vn: VecNormalize = VecNormalize.load(vecnorm_path, dummy)
        self.vn.training = False
        self.vn.norm_reward = False
        # precompute return std for denorm
        self.ret_std = float(np.sqrt(self.vn.ret_rms.var + self.vn.epsilon)) if reward_was_normalized else 1.0

    def normalize_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        mean, var, eps = self.vn.obs_rms.mean, self.vn.obs_rms.var, self.vn.epsilon
        clip = getattr(self.vn, "clip_obs", np.inf)
        obs = (obs_raw - mean) / np.sqrt(var + eps)
        return np.clip(obs, -clip, clip)

    def value_and_action(self, obs_raw: np.ndarray, deterministic: bool = True):
        """Returns (V_true, action_model_space)."""
        obs_norm = self.normalize_obs(obs_raw)
        # critic on normalized obs
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_norm, device=self.policy.device, dtype=torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            v_norm = self.policy.predict_values(obs_t).squeeze(-1).cpu().numpy()
        # denorm value back to true reward units
        v_true = v_norm * self.ret_std
        # actor on normalized obs
        actions, _ = self.model.predict(obs_norm, deterministic=deterministic)
        return np.asarray(v_true), np.asarray(actions)

def plot_along_grid_clean(
    model: PPO,
    env,                                   # your EconGymEnv (plain, not VecNormalize)
    make_env: Callable,                     # same as training
    vecnorm_path: str,                      # saved stats from training
    reward_was_normalized: bool,            # True if norm_reward=True during training
    dim_index: int,
    grid_min: float,
    grid_max: float,
    grid_points: int = 200,
    base_state: Optional[np.ndarray] = None,
    y_idx: Optional[int] = None,
    actions_to_physical: Optional[Callable] = None,  # your existing mapper
    extract_policy_components: Optional[Callable] = None,  # your existing helper
):
    """
    Plots:
      - Value in TRUE reward units along a 1D state grid
      - Policy components (physical units) along the same grid
    Assumes your observation = [continuous_state, one_hot(y)] like in your env.
    """
    adapter = SB3InferenceAdapter(
        model=model,
        make_env=make_env,
        vecnorm_path=vecnorm_path,
        reward_was_normalized=reward_was_normalized,
    )

    obs_cont_dim = env.obs_cont_dim
    assert 0 <= dim_index < obs_cont_dim

    # baseline state (continuous part)
    if base_state is None:
        base_state = env.model.init_state(1)[0].cpu().numpy()  # (obs_cont_dim,)
    # discrete y to hold fixed (one-hot)
    if y_idx is None:
        try:
            y_idx = int(env.model.details.p.z_0 - 1)
        except Exception:
            y_idx = 0

    grid = np.linspace(grid_min, grid_max, grid_points)
    oh = np.zeros(env.num_y, dtype=np.float32); oh[y_idx] = 1.0
    obs_mat = np.repeat(np.concatenate([base_state, oh]).reshape(1, -1), grid_points, axis=0)
    obs_mat[:, dim_index] = grid

    # model-space value (denormalized) + action (model space)
    v_true, actions_model = adapter.value_and_action(obs_mat, deterministic=True)
    
    # obs_mat = [continuous_state, one_hot(y)]  shape: (B, obs_cont_dim + num_y)
    obs_cont_mat = obs_mat[:, :env.obs_cont_dim]

    # make v_curr available to _act_physical's Δ-mapping
    env.state_cont = torch.as_tensor(obs_cont_mat, dtype=env.dtype, device=env.device)
    # to physical units (your functions)
    if actions_to_physical is not None:
        acts_phys = actions_to_physical(env, actions_model)
    else:
        acts_phys = actions_model  # fallback

    if extract_policy_components is not None:
        y_vec = np.full(grid_points, y_idx, dtype=int)
        hiring, sep, vprime = extract_policy_components(acts_phys, env.num_y, env.K_n, env.K_v, y_vec)
    else:
        # fallback: plot first two dims if present
        comp1 = acts_phys[:, 0] if acts_phys.ndim == 2 and acts_phys.shape[1] > 0 else acts_phys
        comp2 = acts_phys[:, 1] if acts_phys.ndim == 2 and acts_phys.shape[1] > 1 else np.zeros_like(comp1)
    # Take expected vprime
    #vprime_sched = vprime_flat.view(B, self.K_v, self.num_y)

    #v_prime_exp_all = torch.einsum("bkz,yz->bky", vprime_sched, self.details.Z_trans_tensor)
    #iN = torch.arange(B, device=state.device)
    #v_prime_exp = v_prime_exp_all[iN, :, y_idx]
    # plots
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 4, 1)
    plt.plot(grid, v_true)
    plt.title(f"Value (true units) vs state[{dim_index}]")
    plt.xlabel(f"state[{dim_index}]"); plt.ylabel("V (true)")

    plt.subplot(1, 4, 2)
    plt.plot(grid, hiring)
    plt.title("Hiring (physical)"); plt.xlabel(f"state[{dim_index}]")

    plt.subplot(1, 4, 3)
    plt.plot(grid, sep)
    plt.title(f"Layoffs (physical)"); plt.xlabel(f"state[{dim_index}]")

    plt.subplot(1, 4, 4)
    plt.plot(grid, vprime)
    plt.title(f"v' (physical)"); plt.xlabel(f"state[{dim_index}]")
    plt.tight_layout()

def actions_to_physical(env: EconGymEnv, actions_np: np.ndarray) -> np.ndarray:
    a_t = torch.as_tensor(actions_np, dtype=env.dtype, device=env.device)
    return env._act_physical(a_t).cpu().numpy()

# ========================= 4) Train with Stable-Baselines3 PPO =========================
def main():
    #device = "cpu"
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pick devices
    policy_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_device = torch.device('cpu')  # keep env/model on CPU to avoid per-process GPU usage
    dtype = torch.float32
    p = Parameters()

    # --- grids / bounds
    K = 2
    K_n = K
    K_v = K - 1
    cc = ContinuousContract(p_crs())

    state_low = torch.tensor([0, 0, cc.v_grid[0], p.q_0], dtype=dtype)
    state_high = torch.tensor([10, 20, 1.0 * cc.v_grid[-1], 1], dtype=dtype)
    vprime_low = torch.full((K_v,), float(cc.v_grid[0]), dtype=dtype)
    vprime_high = torch.full((K_v,), float(1.01 * cc.v_grid[-1]), dtype=dtype)

    beta = p.beta
    bounds = EconBounds(state_low=state_low, state_high=state_high, vprime_low=vprime_low, vprime_high=vprime_high)

    details = EconDetails(env_device, K, p, cc)

    model = EconModel(device=env_device, dtype=dtype, beta=beta, K_n=K_n, K_v=K_v, num_y=p.num_z, details=details, bounds=bounds)
    P_y = model.details.Z_trans_tensor
    pi0 = torch.full((p.num_z,), 1.0 / p.num_z)

    # --- Gym envs
    # parallel envs (see section below), or keep DummyVecEnv for 1 env:
    n_envs = 8  # try 4 or 8 if you have cores; else set to 1 and use DummyVecEnv
    make_env = make_gym_env(model, P_y, pi0, horizon=32, device=env_device)
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])  # or DummyVecEnv([make_env]) if n_envs=1
    # normalize observations and rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=float(beta), clip_obs=10.0, clip_reward=10.0)
    # after you build vec_env (SubprocVecEnv + VecNormalize)
    vec_env = VecMonitor(vec_env)  # logs ep_len_mean/ep_rew_mean to TB too

    # --- SB3 PPO
    policy_kwargs =  dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.SiLU)
    #model_sb3 = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=beta, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=1, device=device)
    model_sb3 = PPO(
    "MlpPolicy", vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,        # smaller step
    n_steps=2048,              # per env ⇒ total batch = n_steps * n_envs
    batch_size=2048,            # must divide total batch
    n_epochs=10,                
    gamma=float(beta),         # your beta
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=0.2,         # <— value function clipping
    ent_coef=0.0,
    vf_coef=0.25,              # reduce value loss weight
    max_grad_norm=0.5,
    target_kl=0.01,            # optional safety brake
    verbose=1,
    device=env_device,  # policy on CPU
    tensorboard_log="tb/MWF",
)
    
    #tensorboard --logdir tb/MWF --bind_all
    #vec_env.training = True
    #obs = vec_env.reset()
    #for _ in range(2000):  # ~2k vector steps = 2k*n_envs transitions
    #    actions = [vec_env.action_space.sample() for _ in range(n_envs)]
    #    obs, r, done, info = vec_env.step(actions)
    
    grid_min = float(bounds.state_low[K_n].item())
    grid_max = float(bounds.state_high[K_n].item())
    
    plot_env = EconGymEnv(model, P_y, pi0, horizon=32, device=env_device)  # plain env for shapes
    plot_cb = PlotCallback(
        plot_env=plot_env,
        make_env=make_env,             # not strictly used here, but handy if you extend
        dim_index=2,                   # pick the state dimension to sweep
        grid_min=grid_min, grid_max=grid_max, grid_points=200,
        plot_every_ts=200_000,            # log every 200k steps
       y_idx=None,
    )

    # EVAL (mirror the wrapper order)
    eval_env = SubprocVecEnv([make_env])          # or more, but 1 is fine for EvalCallback
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, gamma=float(beta))
    eval_env = VecMonitor(eval_env)               # top wrapper

    # Let EvalCallback sync stats; then make sure eval reports TRUE rewards:
    eval_cb = EvalCallback(
    eval_env=eval_env,
    eval_freq=100_000,
    n_eval_episodes=10,
    deterministic=True,
    )

    # After you create the model and before/at training start you can (optionally) freeze eval env:
    vn_eval = eval_cb.eval_env.venv  # unwrap VecMonitor one level
    if isinstance(vn_eval, VecNormalize):
        vn_eval.training = False
        vn_eval.norm_reward = False
    # Optional: periodic checkpoints too
    from stable_baselines3.common.callbacks import CheckpointCallback
    ckpt_cb = CheckpointCallback(save_freq=200_000, save_path="checkpoints/", name_prefix="ppo")

    total_timesteps = 10_000_000
    model_sb3.learn(total_timesteps=total_timesteps, callback=[eval_cb, plot_cb, ckpt_cb], tb_log_name="run1")

    # After training
    #model_sb3.save("ppo_econ_agent.zip")
    vec_env.save("vecnorm_stats.pkl")
    
    eval_vec_env = SubprocVecEnv([make_env])   # or DummyVecEnv if single env
    vecnorm = VecNormalize.load("vecnorm_stats.pkl", eval_vec_env)
    vecnorm.training = False
    vecnorm.norm_reward = False   # report real rewards
    # quick eval

    n_eval_eps = 20
    ep_returns = []
    for _ in range(n_eval_eps):
        obs = vecnorm.reset()
        done = [False]  # vec env API
        ep_r = 0.0
        while not done[0]:
            action, _ = model_sb3.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vecnorm.step(action)      # ← use vecnorm
            ep_r += float(rewards[0])                              # real reward
            done = dones
        ep_returns.append(ep_r)
    print({"eval_mean_reward": np.mean(ep_returns), "n_eval_eps": len(ep_returns)})

    model_sb3.save("ppo_econ_agent.zip")
    print("Saved SB3 PPO model to ppo_econ_agent.zip")

    # ----- Plots -----
    #eval_env = EconGymEnv(model=model, P_y=P_y, pi0=pi0, horizon=64, device=env_device)
    # 1) Scatter on visited states, along last continuous dim (e.g., v)
    #plot_scatter_reached(model_sb3, eval_env, dim_index=eval_env.obs_cont_dim - 1, episodes=3, max_steps=64)
    # 2) Along a fixed grid for the same dim
    #grid_min = float(bounds.state_low[eval_env.obs_cont_dim - 1].item())
    #grid_max = float(bounds.state_high[eval_env.obs_cont_dim - 1].item())
    #base_state = eval_env.model.init_state(1)[0].cpu().numpy()
    #plot_along_grid(model_sb3, eval_env, dim_index=eval_env.obs_cont_dim - 1, grid_min=grid_min, grid_max=grid_max, grid_points=200, base_state=base_state)
    #plt.show()

    # for plotting:
    make_env = make_gym_env(model, P_y, pi0, horizon=32, device=env_device)

    plot_along_grid_clean(
    model=model_sb3,
    env=EconGymEnv(model, P_y, pi0, horizon=32, device=env_device),  # plain env for shapes
    make_env=make_env,
    vecnorm_path="vecnorm_stats.pkl",
    reward_was_normalized=True,    # set False if you trained with norm_reward=False
    dim_index=2,
    grid_min=grid_min, grid_max=grid_max, grid_points=200,
    base_state=None,               # optional
    y_idx=None,                    # optional
    actions_to_physical=actions_to_physical,
    extract_policy_components=extract_policy_components,
    )
    plt.show()

if __name__ == "__main__":
    main()
