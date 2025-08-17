import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
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
    def __init__(self, K, p, cc):
        self.K = K
        self.p = p
        self.pref = Preferences(input_param=self.p)
        self.Z_grid = self._construct_z_grid()

        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = torch.tensor(self.fun_prod, dtype=DTYPE)
        self.unemp_bf = self.p.u_bf_m

        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)
        self.Z_trans_tensor = torch.tensor(self.Z_trans_mat, dtype=torch.float32)

        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max().item(), self.p.num_v)
        self.rho_grid = torch.tensor(1 / self.pref.utility_1d(self.w_grid), dtype=DTYPE)

        self.v_grid = np.linspace(
            np.divide(self.pref.utility(self.unemp_bf), 1 - self.p.beta),
            np.divide(self.pref.utility(self.fun_prod.max().item()), 1 - self.p.beta),
            self.p.num_v,
        )
        self.v_0 = torch.tensor(self.v_grid[0], dtype=DTYPE)

        self.simple_J = (
            (self.fun_prod[:, ax] - self.pref.inv_utility(torch.tensor(self.v_grid[ax, :] * (1 - self.p.beta), dtype=DTYPE)))
            / (1 - self.p.beta)
        )

        self.prob_find_vx = self.p.alpha * np.power(
            1 - np.power(np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0 - 1, :].numpy(), 1.0)), self.p.sigma),
            1 / self.p.sigma,
        )
        if cc is None:
            self.js = JobSearchArray()
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

    @torch.no_grad()
    def init_state(self, batch_size: int) -> torch.Tensor:
        obs_cont_dim = self.bounds.state_low.numel()
        s = torch.zeros(batch_size, obs_cont_dim, device=self.device, dtype=self.dtype)
        s[:, 0] = 1.0
        return s

    @torch.no_grad()
    def step(self, state: torch.Tensor, action_phys: torch.Tensor, y_idx: torch.Tensor, y_next_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        B = state.shape[0]
        sizes = state[:, :self.K_n]
        v = state[:, self.K_n:]

        hiring = action_phys[:, 0]
        vprime_flat = action_phys[:, 1 : 1 + self.K_v * self.num_y]
        vprime_sched = vprime_flat.view(B, self.K_v, self.num_y)

        v_prime_exp_all = torch.einsum("bkz,yz->bky", vprime_sched, self.details.Z_trans_tensor)
        iN = torch.arange(B, device=state.device)
        v_prime_exp = v_prime_exp_all[iN, :, y_idx]
        re, pc = self.details.getWorkerDecisions(v_prime_exp)

        # Compute argument to inv_utility (which is exp for log utility)
        u_emp = v - self.details.p.beta * (v_prime_exp + re)
        u_unemp = self.details.v_0 - self.details.p.beta * (v_prime_exp + re)
        wages = torch.zeros(B, self.K_n, device=state.device, dtype=self.dtype)
        wages[:, 1:] = self.details.pref.inv_utility(v - self.details.p.beta * (v_prime_exp + re))
        wages[:, :1] = self.details.pref.inv_utility(self.details.v_0 - self.details.p.beta * (v_prime_exp + re))
        if not torch.isfinite(wages).all():
            print("u_emp range:", float(u_emp.min()), float(u_emp.max()))
            print("u_unemp range:", float(u_unemp.min()), float(u_unemp.max()))

        next_state = state.clone()
        next_state[:, 0] = hiring
        next_state[:, 1] = sizes.sum(dim=1) * pc.squeeze(1)
        next_state[:, self.K_n:] = vprime_sched[iN, :, y_next_idx]

        reward = (
            self.details.fun_prod[y_idx.detach().long()] * self.details.production(state[:, :self.K_n].sum(dim=1))
            - self.details.p.hire_c * hiring
            - (wages * sizes).sum(dim=1)
        ).to(self.dtype)

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

        if pi0 is None:
            self.pi0 = torch.full((self.num_y,), 1.0 / self.num_y, device=self.device, dtype=torch.float32)
        else:
            self.pi0 = pi0.to(self.device, dtype=torch.float32)

        # observation: concatenation of continuous state and one-hot y
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # action: [hiring, v'(K_v x num_y)] in [-1,1]
        self.act_dim = 1 + self.K_v * self.num_y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # internal state
        self.t = 0
        self.state_cont = None  # torch [1, obs_cont_dim]
        self.y_idx = None       # int

    # ---- helpers ----
    def _act_physical(self, a: torch.Tensor) -> torch.Tensor:
        """Map action from [-1,1] to physical domain using bounds."""
        B = a.shape[0]
        hiring = 0.5 * (a[:, :1] + 1.0) * (self.model.bounds.state_high[0] - self.model.bounds.state_low[0]) + self.model.bounds.state_low[0]
        vflat = a[:, 1 : 1 + self.K_v * self.num_y]
        vsched = vflat.view(B, self.K_v, self.num_y)
        vlow = self.model.bounds.vprime_low.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)
        vhigh = self.model.bounds.vprime_high.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)
        vsched_phys = 0.5 * (vsched + 1.0) * (vhigh - vlow)[None, :, None] + vlow[None, :, None]
        return torch.cat([hiring, vsched_phys.reshape(B, self.K_v * self.num_y)], dim=-1)

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


# ========================= 3) Train with Stable-Baselines3 PPO =========================

def make_gym_env(model, P_y, pi0, horizon, device):
    def _make():
        env = EconGymEnv(model=model, P_y=P_y, pi0=pi0, horizon=horizon, device=device)
        return Monitor(env)
    return _make

def sb3_value_and_action(model_sb3: PPO, obs_np: np.ndarray, deterministic: bool = True):
    """Return (values_np, actions_np) for observations (N, obs_dim).
    Actions come from model.predict(..., deterministic=True) so they respect env.action_space."""
    policy = model_sb3.policy
    obs_t = torch.as_tensor(obs_np, device=policy.device, dtype=torch.float32)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    with torch.no_grad():
        features = policy.extract_features(obs_t)
        _, latent_vf = policy.mlp_extractor(features)
        values = policy.value_net(latent_vf).squeeze(-1)
    actions_np, _ = model_sb3.predict(obs_np, deterministic=deterministic)
    return values.cpu().numpy(), np.asarray(actions_np)


def extract_policy_components(actions_np: np.ndarray, num_y: int, K_v: int, y_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """From normalized actions in [-1,1], extract hiring and v_prime for current y (k=0).
    actions_np: (N, 1 + K_v * num_y)
    y_idx: (N,) ints inferred from obs one-hot when rolling out.
    Returns (hiring_norm, vprime0_norm).
    """
    hiring = actions_np[:, 0]
    if K_v <= 0:
        vprime0 = np.zeros_like(hiring)
    else:
        idx = 1 + (0 * num_y + y_idx)
        vprime0 = actions_np[np.arange(actions_np.shape[0]), idx]
    return hiring, vprime0


def rollout_collect_reached(model_sb3: PPO, env: EconGymEnv, episodes: int = 5, max_steps: Optional[int] = None):
    """Roll out deterministic policy in a fresh (non-vec) env and collect obs, values, actions."""
    obs_list, val_list, act_list = [], [], []
    y_list = []
    for _ in range(episodes):
        obs, _ = env.reset()
        t = 0
        done = False
        while not done:
            vals, acts = sb3_value_and_action(model_sb3, obs)
            obs_list.append(obs)
            val_list.append(vals[0])
            act_list.append(acts)
            y_idx = int(np.argmax(obs[-env.num_y:]))
            y_list.append(y_idx)
            action, _ = model_sb3.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            t += 1
            if max_steps is not None and t >= max_steps:
                break
    return np.vstack(obs_list), np.array(val_list), np.vstack(act_list), np.array(y_list)


def plot_scatter_reached(model_sb3: PPO, env: EconGymEnv, dim_index: int, episodes: int = 5, max_steps: Optional[int] = None):
    """Scatter: value and policies vs a chosen *continuous* state dim along visited states."""
    obs, vals, acts, y_idx = rollout_collect_reached(model_sb3, env, episodes, max_steps)
    x = obs[:, dim_index]
    hire, vprime0 = extract_policy_components(actions_to_physical(env,acts), env.num_y, env.K_v, y_idx)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(x, vals, s=8)
    plt.title("Value vs state[{}] (visited)".format(dim_index))
    plt.xlabel(f"state[{dim_index}]")
    plt.ylabel("V")

    plt.subplot(1, 3, 2)
    plt.scatter(x, hire, s=8)
    plt.title("Hiring (norm) vs state[{}]".format(dim_index))
    plt.xlabel(f"state[{dim_index}]")
    plt.ylabel("hiring in [-1,1]")

    plt.subplot(1, 3, 3)
    plt.scatter(x, vprime0, s=8)
    plt.title("v' (k=0, current y) (norm)")
    plt.xlabel(f"state[{dim_index}]")
    plt.ylabel("v' in [-1,1]")

    plt.tight_layout()
    #plt.show()

def actions_to_physical(env: EconGymEnv, actions_np: np.ndarray) -> np.ndarray:
    a_t = torch.as_tensor(actions_np, dtype=env.dtype, device=env.device)
    return env._act_physical(a_t).cpu().numpy()
def plot_along_grid(model_sb3: PPO, env: EconGymEnv, dim_index: int, grid_min: float, grid_max: float, grid_points: int = 200, base_state: Optional[np.ndarray] = None, y_idx: Optional[int] = None):
    """Plot value and policy (deterministic action) along a 1D grid of a chosen continuous state dim.
    - base_state: (obs_cont_dim,) continuous state used as baseline for other dims; default = env.model.init_state(1)[0]
    - y_idx: discrete state fixed for the grid; default = middle (or 0 if undefined)
    """
    obs_cont_dim = env.obs_cont_dim
    assert 0 <= dim_index < obs_cont_dim, "dim_index must target a continuous state dim"

    if base_state is None:
        base_state = env.model.init_state(1)[0].cpu().numpy()
    if y_idx is None:
        try:
            y_idx = int(env.model.details.p.z_0 - 1)
        except Exception:
            y_idx = 0

    grid = np.linspace(grid_min, grid_max, grid_points)

    oh = np.zeros(env.num_y, dtype=np.float32)
    oh[y_idx] = 1.0
    obs_mat = np.repeat(np.concatenate([base_state, oh]).reshape(1, -1), grid_points, axis=0)
    obs_mat[:, dim_index] = grid

    vals, acts = sb3_value_and_action(model_sb3, obs_mat)
    y_vec = np.full(grid_points, y_idx, dtype=int)
    acts = actions_to_physical(env,acts)
    hire, vprime0 = extract_policy_components(acts, env.num_y, env.K_v, y_vec)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(grid, vals)
    plt.title("Value along grid (state[{}])".format(dim_index))
    plt.xlabel(f"state[{dim_index}]")
    plt.ylabel("V")

    plt.subplot(1, 3, 2)
    plt.plot(grid, hire)
    plt.title("Hiring (norm) vs grid")
    plt.xlabel(f"state[{dim_index}]")
    plt.ylabel("hiring in [-1,1]")

    plt.subplot(1, 3, 3)
    plt.plot(grid, vprime0)
    plt.title("v' (k=0, y={}) (norm)".format(y_idx))
    plt.xlabel(f"state[{dim_index}]")
    plt.ylabel("v' in [-1,1]")

    plt.tight_layout()
    #plt.show()
def main():
    device = "cpu"
    dtype = torch.float32
    p = Parameters()

    # --- grids / bounds
    K = 2
    K_n = K
    K_v = K - 1
    cc = ContinuousContract(p_crs())

    state_low = torch.tensor([0, 0, cc.v_grid[0]], dtype=dtype)
    state_high = torch.tensor([10, 20, 1.5 * cc.v_grid[-1]], dtype=dtype)
    vprime_low = torch.full((K_v,), float(cc.v_grid[0]), dtype=dtype)
    vprime_high = torch.full((K_v,), float(1.5 * cc.v_grid[-1]), dtype=dtype)

    beta = p.beta
    bounds = EconBounds(state_low=state_low, state_high=state_high, vprime_low=vprime_low, vprime_high=vprime_high)

    details = EconDetails(K, p, cc)
    model = EconModel(device=torch.device(device), dtype=dtype, beta=beta, K_n=K_n, K_v=K_v, num_y=p.num_z, details=details, bounds=bounds)
    P_y = model.details.Z_trans_tensor
    pi0 = torch.full((p.num_z,), 1.0 / p.num_z)

    # --- Gym envs
    # parallel envs (see section below), or keep DummyVecEnv for 1 env:
    n_envs = 8  # try 4 or 8 if you have cores; else set to 1 and use DummyVecEnv
    make_env = make_gym_env(model, P_y, pi0, horizon=32, device=device)
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])  # or DummyVecEnv([make_env]) if n_envs=1
    # normalize observations and rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=float(beta), clip_obs=5.0, clip_reward=5.0)


    # --- SB3 PPO
    policy_kwargs =  dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.SiLU)
    #model_sb3 = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=beta, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=1, device=device)
    model_sb3 = PPO(
    "MlpPolicy", vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,        # smaller step
    n_steps=1024,              # per env ⇒ total batch = n_steps * n_envs
    batch_size=2048,            # must divide total batch
    n_epochs=5,                # fewer passes per update
    gamma=float(beta),         # your beta
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=0.2,         # <— value function clipping
    ent_coef=0.0,
    vf_coef=0.25,              # reduce value loss weight
    max_grad_norm=0.5,
    target_kl=0.01,            # optional safety brake
    verbose=1,
    device=device,
)
    
    vec_env.training = True
    obs = vec_env.reset()
    for _ in range(2000):  # ~2k vector steps = 2k*n_envs transitions
        actions = [vec_env.action_space.sample() for _ in range(n_envs)]
        obs, r, done, info = vec_env.step(actions)

    total_timesteps = 4000_000
    model_sb3.learn(total_timesteps=total_timesteps)

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
    eval_env = EconGymEnv(model=model, P_y=P_y, pi0=pi0, horizon=64, device=device)
    # 1) Scatter on visited states, along last continuous dim (e.g., v)
    plot_scatter_reached(model_sb3, eval_env, dim_index=eval_env.obs_cont_dim - 1, episodes=3, max_steps=64)
    # 2) Along a fixed grid for the same dim
    grid_min = float(bounds.state_low[eval_env.obs_cont_dim - 1].item())
    grid_max = float(bounds.state_high[eval_env.obs_cont_dim - 1].item())
    base_state = eval_env.model.init_state(1)[0].cpu().numpy()
    plot_along_grid(model_sb3, eval_env, dim_index=eval_env.obs_cont_dim - 1, grid_min=grid_min, grid_max=grid_max, grid_points=200, base_state=base_state)
    plt.show()
if __name__ == "__main__":
    main()
