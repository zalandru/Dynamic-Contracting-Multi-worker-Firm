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
from torchrl.data import Unbounded, Bounded, Composite

from tensordict import TensorDict  
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import Compose, ObservationNorm, Transform
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

# ============ 1) Econ helpers ============

@dataclass
class EconBounds:
    state_low: torch.Tensor      # [obs_dim]
    state_high: torch.Tensor     # [obs_dim]
    vprime_low: torch.Tensor     # [K_v]
    vprime_high: torch.Tensor    # [K_v]


class EconDetails:
    def __init__(self, K, p, cc):
        self.K = K
        self.p = p
        self.deriv_eps = 1e-3
        self.pref = Preferences(input_param=self.p)
        self.Z_grid = self.construct_z_grid()

        self.fun_prod = self.p.prod_a * np.power(self.Z_grid, self.p.prod_rho)
        self.fun_prod = tensor(self.fun_prod, dtype=type)
        self.unemp_bf = self.p.u_bf_m

        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)
        self.Z_trans_tensor = tensor(self.Z_trans_mat, dtype=torch.float32)

        self.w_grid = np.linspace(self.unemp_bf, self.fun_prod.max(), self.p.num_v)
        self.rho_grid = tensor(1 / self.pref.utility_1d(self.w_grid), dtype=type)

        self.v_grid = np.linspace(
            np.divide(self.pref.utility(self.unemp_bf), 1 - self.p.beta),
            np.divide(self.pref.utility(self.fun_prod.max()), 1 - self.p.beta),
            self.p.num_v,
        )
        self.v_0 = tensor(self.v_grid[0], dtype=type)

        self.simple_J = torch.zeros((self.p.num_z, self.p.num_v), dtype=type)
        self.simple_J = (
            (self.fun_prod[:, ax] - self.pref.inv_utility(tensor(self.v_grid[ax, :] * (1 - self.p.beta), dtype=type)))
            / (1 - self.p.beta)
        )
        self.simple_Rho = self.simple_J + self.rho_grid[ax, :] * tensor(self.v_grid[ax, :], dtype=type)

        self.prob_find_vx = self.p.alpha * np.power(
            1 - np.power(np.divide(self.p.kappa, np.maximum(self.simple_J[self.p.z_0 - 1, :], 1.0)), self.p.sigma),
            1 / self.p.sigma,
        )
        if cc is None:
            self.js = JobSearchArray()
            self.js.update(self.v_grid[:], self.prob_find_vx.numpy())
        else:
            self.js = cc.js

    def take_expectation(self, x, i, prod_states, v_prime=None):
        if v_prime is not None:
            x = torch.einsum("bkZ,YZ->bkY", x, self.Z_trans_tensor)
            x = x[i, :, prod_states.long()]
        else:
            if x.ndim == 2:
                x = torch.einsum("by,zy->bz", x, self.Z_trans_tensor)
                x = x[i, prod_states.long()]
            else:
                x = torch.einsum("byd,zy->bzd", x, self.Z_trans_tensor)
                x = x[i, prod_states.long(), :]
        return x

    def production(self, sum_n):
        return sum_n ** self.p.prod_alpha

    def production_1d_n(self, sum_n):
        return self.p.prod_alpha * (sum_n ** (self.p.prod_alpha - 1))

    def getWorkerDecisions(self, EW1, employed=True):
        pe, re = self.js.solve_search_choice(EW1)
        assert (~torch.isnan(pe)).all(), "pe is NaN"
        assert (pe <= 1).all(), "pe is not less than 1"
        assert (pe >= -1e-10).all(), "pe is not larger than 0"
        if employed:
            pe = pe * self.p.s_job
            re = re * self.p.s_job
        pc = (1 - pe)
        return re, pc

    def construct_z_grid(self):
        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:], (1))
        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)


class EconModel:
    def __init__(self, device: torch.device, dtype: torch.dtype, beta: float, K_n: int, K_v: int, num_y: int, details: EconDetails, bounds: EconBounds):
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.K_n = K_n
        self.K_v = K_v
        self.bounds = bounds
        self.num_y = num_y
        self.details = details

    @torch.no_grad()
    def init_state(self, batch_size: int) -> torch.Tensor:
        obs_dim = self.bounds.state_low.numel()
        starting_states = torch.zeros(batch_size, obs_dim, device=self.device, dtype=self.dtype)
        starting_states[:, 0] = 1
        starting_states[:, 1:self.K_n] = 0
        starting_states[:, self.K_n:] = 0
        return starting_states

    @torch.no_grad()
    def step(self, state: torch.Tensor, action_phys: torch.Tensor, y_idx: torch.Tensor, y_next_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        B = state.shape[0]
        sizes = state[:, :self.K_n]
        v = state[:, self.K_n:]

        hiring = action_phys[:, 0]
        vprime_flat = action_phys[:, 1: 1 + self.K_v * self.num_y]
        vprime_sched = vprime_flat.view(B, self.K_v, self.num_y)

        v_prime_exp_all = torch.einsum("bky,yz->bkz", vprime_sched, self.details.Z_trans_tensor)
        iN = torch.arange(B)
        v_prime_exp = v_prime_exp_all[iN, :, y_idx]
        re, pc = self.details.getWorkerDecisions(v_prime_exp)

        wages = torch.zeros(B, self.K_n, device=state.device, dtype=self.dtype)
        wages[:, 1:] = self.details.pref.inv_utility(v - self.details.p.beta * (v_prime_exp + re))
        wages[:, :1] = self.details.pref.inv_utility(self.details.v_0 - self.details.p.beta * (v_prime_exp + re))

        next_state = state.clone()
        next_state[:, 0] = hiring
        next_state[:, 1] = sizes.sum(dim=1) * pc.squeeze(1)
        next_state[:, self.K_n:] = vprime_sched[iN, :, y_next_idx]

        reward = (
            self.details.fun_prod[y_idx.detach().long()] * self.details.production(next_state[:, :self.K_n].sum(dim=1))
            - self.details.p.hire_c * hiring
            - (wages * sizes).sum(dim=1)
        ).to(self.dtype)

        done = torch.zeros(B, device=state.device, dtype=torch.bool)
        info = {}
        return next_state, reward, done, info


# ============ 2) TorchRL Environment wrapper ============

class EconEnv(EnvBase):
    def __init__(
        self,
        model: EconModel,
        horizon: int,
        P_y: torch.Tensor,
        pi0: Optional[torch.Tensor],
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        sample_yprime: bool = True,
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

        self.P_y = P_y.to(device, dtype=torch.float32)
        if pi0 is None:
            self.pi0 = torch.full((self.num_y,), 1.0 / self.num_y, device=device, dtype=torch.float32)
        else:
            self.pi0 = pi0.to(device, dtype=torch.float32)

        # ---- Specs: leaves include env batch leading dim to match Composite(shape=[1])
        obs_spec = Unbounded(shape=(1, obs_dim), device=device, dtype=dtype)
        act_spec = Bounded(
            low=-torch.ones(1, action_dim, device=device, dtype=dtype),
            high=torch.ones(1, action_dim, device=device, dtype=dtype),
            shape=(1, action_dim),
            device=device,
            dtype=dtype,
        )
        rew_spec = Unbounded(shape=(1, 1), device=device, dtype=dtype)
        done_spec = Bounded(0, 1, shape=(1, 1), dtype=torch.bool, device=device)

        self.observation_spec = Composite(observation=obs_spec, shape=self.batch_size).to(device)
        self.action_spec = act_spec
        self.reward_spec = rew_spec
        self.done_spec = done_spec

        self.state_cont = None
        self.y_idx = None
        self._t = None

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
        return seed

    def _obs_from_raw(self, state_cont: torch.Tensor, y_idx: torch.Tensor) -> torch.Tensor:
        B = state_cont.shape[0]
        one_hot = torch.nn.functional.one_hot(y_idx, num_classes=self.num_y).to(state_cont.dtype)
        obs = torch.cat([state_cont, one_hot], dim=-1)
        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        batch = 1 if tensordict is None or tensordict.batch_size == torch.Size([]) else tensordict.batch_size[0]
        self.state_cont = self.model.init_state(batch)
        self.y_idx = torch.multinomial(self.pi0.expand(batch, -1), num_samples=1).squeeze(1)
        self._t = torch.zeros(batch, device=self.device, dtype=torch.long)
        obs = self._obs_from_raw(self.state_cont, self.y_idx)
        assert obs.shape[0] == batch and obs.ndim == 2, f"obs shape {obs.shape}"
        td = TensorDict({}, batch_size=self.batch_size, device=self.device)
        td.set("observation", obs)
        td.set("done", torch.zeros(batch, 1, device=self.device, dtype=torch.bool))
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get("action")
        action_phys = self._act_physical(action)

        B = action.shape[0]
        cur_y = self.y_idx
        P_rows = self.P_y[cur_y]

        if self.sample_yprime:
            y_next = torch.multinomial(P_rows, num_samples=1).squeeze(1)
        else:
            y_next = torch.argmax(P_rows, dim=1)

        next_state_cont, r, done, info = self.model.step(self.state_cont, action_phys, cur_y, y_next)

        self._t += 1
        horizon_done = (self._t >= self.horizon).unsqueeze(-1)

        self.state_cont = next_state_cont
        self.y_idx = y_next
        next_obs = self._obs_from_raw(self.state_cont, self.y_idx)

        assert r.ndim == 1 and r.shape[0] == B, f"reward shape {r.shape}"
        assert done.ndim == 1 and done.shape[0] == B, f"done shape {done.shape}"

        out = TensorDict({}, batch_size=self.batch_size, device=self.device)
        out.set(
            "next",
            TensorDict(
                {
                    "observation": next_obs,
                    "reward": r.unsqueeze(-1),
                    "done": torch.logical_or(done.unsqueeze(-1), horizon_done),
                },
                batch_size=self.batch_size,
                device=self.device,
            ),
        )
        return out

    def _act_physical(self, a: torch.Tensor) -> torch.Tensor:
        B = a.shape[0]
        K_v, num_y = self.model.K_v, self.model.num_y

        # hiring: map [-1,1] -> [state_low[0], state_high[0]]
        hiring = 0.5 * (a[:, :1] + 1.0) * (self.model.bounds.state_high[0] - self.model.bounds.state_low[0]) + self.model.bounds.state_low[0]

        # v': map [-1,1] -> [vlow, vhigh]
        vflat = a[:, 1: 1 + K_v * num_y]
        vsched = vflat.view(B, K_v, num_y)

        vlow = self.model.bounds.vprime_low.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)
        vhigh = self.model.bounds.vprime_high.to(self.device, self.dtype).unsqueeze(0).expand(B, -1)
        vsched_phys = 0.5 * (vsched + 1.0) * (vhigh - vlow)[None, :, None] + vlow[None, :, None]

        return torch.cat([hiring, vsched_phys.reshape(B, K_v * num_y)], dim=-1)


# ============ 3) Actor–critic and PPO ============

def make_actor(obs_dim: int, act_dim: int, hidden=(64, 64)) -> ProbabilisticActor:
    net = MLP(in_features=obs_dim, out_features=2 * act_dim, depth=len(hidden), num_cells=hidden, activation_class=nn.SiLU)
    extractor = NormalParamExtractor()
    td_module = TensorDictModule(nn.Sequential(net, extractor), in_keys=["observation"], out_keys=["loc", "scale"])

    actor = ProbabilisticActor(
        module=td_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"min": -1.0, "max": 1.0},
        return_log_prob=True,
    )
    return actor


def make_critic(obs_dim: int, hidden=(64, 64)) -> ValueOperator:
    net = MLP(in_features=obs_dim, out_features=1, depth=len(hidden), num_cells=hidden, activation_class=nn.SiLU)
    critic = ValueOperator(module=net, in_keys=["observation"])
    return critic


def flatten_batch_time(td: TensorDict) -> TensorDict:
    return td.reshape(-1)


def make_env_fn(model, horizon, P_y, pi0, action_dim, device, dtype):
    class MirrorObservation(Transform):
        """Ensure both 'observation' and ('next','observation') exist by mirroring whichever is present."""
        def _call(self, td):
            try:
                top_obs = td.get("observation")
                has_top = True
            except KeyError:
                top_obs = None
                has_top = False
            try:
                next_obs = td.get(("next", "observation"))
                has_next = True
            except KeyError:
                next_obs = None
                has_next = False
            if has_next and not has_top:
                td.set("observation", next_obs)
            elif has_top and not has_next:
                td.set(("next", "observation"), top_obs)
            return td

    def _make():
        base_env = EconEnv(
            model=model,
            horizon=horizon,
            P_y=P_y,
            pi0=pi0,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
            sample_yprime=True,
        )
        obs_dim = base_env.obs_dim
        transforms = Compose(
            MirrorObservation(),
            ObservationNorm(
                in_keys=["observation"],
                loc=torch.zeros(obs_dim, device=device, dtype=dtype),
                scale=torch.ones(obs_dim, device=device, dtype=dtype),
                standard_normal=True,
            ),
        )
        return TransformedEnv(base_env, transforms).to(device)
    return _make


def ppo_train(
    model: EconModel,
    P_y: torch.Tensor,
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
    dtype: torch.dtype = torch.float32,
    p = None,
):
    torch.manual_seed(seed)
    device = torch.device(device)
    obs_dim = model.bounds.state_low.numel() + model.num_y
    act_dim = 1 + model.K_v * model.num_y

    env_maker = make_env_fn(model, horizon, P_y, pi0, act_dim, device, dtype)
    env = env_maker()
    check_env_specs(env)

    actor = make_actor(obs_dim, act_dim).to(device)
    critic = make_critic(obs_dim).to(device)

    policy_for_collection = actor
    collector = SyncDataCollector(
        create_env_fn=env_maker,
        policy=policy_for_collection,
        frames_per_batch=frames_per_batch,
        device=device,
        storing_device=device,
        init_random_frames=0,
        split_trajs=True,
    )

    advantage = GAE(gamma=gamma, lmbda=gae_lambda, value_network=critic)
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=clip_eps,
        entropy_coef=ent_coef,
        critic_coef=value_coef,
    )

    optim_all = optim.Adam([
        {"params": actor.parameters(), "lr": lr},
        {"params": critic.parameters(), "lr": lr},
    ])

    pbar = tqdm(total=total_frames, desc="PPO")
    frames_seen = 0
    best_eval = -float("inf")

    for batch in collector:
        frames = batch.numel()
        frames_seen += frames
        pbar.update(frames)

        with torch.no_grad():
            batch = advantage(batch)
            adv = batch.get("advantage")
            batch.set_("advantage", (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8))

        old_log_prob = batch.get("sample_log_prob").detach()
        batch.set("old_log_prob", old_log_prob)

        flat = flatten_batch_time(batch)

        for _ in range(ppo_epochs):
            perm = torch.randperm(flat.batch_size[0], device=device)
            for i in range(0, flat.batch_size[0], minibatch_size):
                idx = perm[i : i + minibatch_size]
                sub = flat[idx]

                loss_dict = loss_module(sub)
                loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]

                loss.backward()
                clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
                optim_all.step()
                optim_all.zero_grad(set_to_none=True)

        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            eval_td = env.reset()
            ep_return = torch.zeros(1, device=device)
            for _ in range(horizon):
                eval_td = actor(eval_td)
                eval_td = env.step(eval_td)
                ep_return += eval_td.get("next")["reward"].squeeze(-1)
                eval_td = eval_td.get("next").rename(None)
            mean_return = ep_return.mean().item()
            best_eval = max(best_eval, mean_return)
        pbar.set_postfix({"R_eval": f"{mean_return:.2f}", "R_best": f"{best_eval:.2f}"})

        if frames_seen >= total_frames:
            break

    pbar.close()
    return actor, critic, env


# ============ 4) Script entry ============

def main():
    device = torch.device("cpu")
    dtype = torch.float32
    p = Parameters()

    K = 2
    K_n = K
    K_v = K - 1
    act_dim = 1 + K_v * p.num_z
    cc = ContinuousContract(p_crs())

    state_low = torch.tensor([0, 0, cc.v_grid[0]], device=device, dtype=dtype)
    state_high = torch.tensor([10, 20, cc.v_grid[-1]], device=device, dtype=dtype)
    vprime_low = torch.full((K_v,), float(cc.v_grid[0]), dtype=dtype, device=device)
    vprime_high = torch.full((K_v,), float(cc.v_grid[-1]), dtype=dtype, device=device)

    beta = p.beta
    bounds = EconBounds(state_low=state_low, state_high=state_high, vprime_low=vprime_low, vprime_high=vprime_high)

    details = EconDetails(K, p, cc)
    model = EconModel(device=device, dtype=dtype, beta=beta, K_n=K_n, K_v=K_v, num_y=p.num_z, details=details, bounds=bounds)
    P_y = model.details.Z_trans_tensor
    pi0 = torch.full((p.num_z,), 1.0 / p.num_z)

    actor, critic, env = ppo_train(
        model=model,
        P_y=P_y,
        pi0=pi0,
        horizon=64,
        frames_per_batch=4096,
        total_frames=200_000,
        gamma=beta,
        gae_lambda=0.95,
        clip_eps=0.2,
        lr=3e-4,
        ent_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        minibatch_size=1024,
        seed=0,
        device=str(device),
        dtype=dtype,
        p=p,
    )

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, "ppo_agent.pt")
    print("Saved to ppo_agent.pt")


if __name__ == "__main__":
    main()
