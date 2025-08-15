# PPO-style trainer for your MWF model
# -------------------------------------------------------------
# This file drops into your project and reuses your environment
# pieces (Parameters p, StateBoundsProcessor, FOCresidual, etc.).
# It trains a stochastic policy with the PPO clipped objective
# over the same state/action semantics you already use:
#   state  = (hiring_prev, n0, n1, v1, ...)[normalized to 0..1]
#   action = (hiring,  v_prime[ k, y'] for k=1..K_v, y'=1..Z)
# where next state's v components take the column v_prime[:, y_next].
#
# How to use (minimal):
#   from MWF_PPO import PPOTrainer
#   trainer = PPOTrainer(foc_optimizer, bounds_processor, K_n, K_v, p)
#   trainer.train(num_updates=1000)
#
# Notes:
# - We do **not** backprop through environment dynamics.
# - We sample actions with a tanh-squashed Normal and map to your
#   physical bounds (LOWER_BOUNDS/UPPER_BOUNDS).
# - We learn a value net V(s)[y] and pick the entry of the current y.
# -------------------------------------------------------------
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Tuple


# -------------------------
# Small utilities
# -------------------------
@dataclass
class RolloutBatch:
    # Time-major tensors: [T, N, ...]
    obs: torch.Tensor          # [T, N, D]
    y: torch.Tensor            # [T, N]
    actions: torch.Tensor      # [T, N, A]  (on the **physical** scale)
    logp: torch.Tensor         # [T, N]
    rewards: torch.Tensor      # [T, N]
    values: torch.Tensor       # [T, N]
    y_next: torch.Tensor       # [T, N]
    obs_next: torch.Tensor     # [T, N, D]
    value_next: torch.Tensor   # [T, N]


def _tanh_squash_and_logp(mu, log_std, raw_eps, low, high):
    """Given Normal(mu, exp(log_std)) and standard-normal `raw_eps`,
    produce: u = mu + sigma*raw_eps; a = low + (tanh(u)+1)/2 * (high-low);
    and log_prob(a) with exact change-of-variables.

    Args (shapes broadcastable per-sample):
      mu, log_std, raw_eps: [..., A]
      low, high: [A]
    Returns:
      a_phys: [..., A]
      logp:   [...]
    """
    sigma = torch.exp(log_std)
    u = mu + sigma * raw_eps
    t = torch.tanh(u)
    # scale from [-1,1] → [low, high]
    scale = (high - low) * 0.5
    a_phys = low + (t + 1.0) * scale

    # base normal logp of u
    base = -0.5 * (((u - mu) / (sigma + 1e-8))**2 + 2.0*log_std + math.log(2.0*math.pi))
    base = base.sum(dim=-1)

    # log|det J| = Σ log( scale * (1 - tanh(u)^2) )
    log_det = torch.log(scale + 1e-8).sum(dim=-1) + torch.log1p(-t*t + 1e-12).sum(dim=-1)
    logp = base - log_det
    return a_phys, logp


# -------------------------
# Networks
# -------------------------
class Critic(nn.Module):
    """V(s)[y] — mirrors your ValueFunctionNN head shape.
    Expect normalized obs in [0,1]."""
    def __init__(self, state_dim: int, num_y: int, hidden=(64,64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_y)

    def forward(self, x):
        z = self.trunk(x)
        return self.head(z)  # [B, num_y]


class Actor(nn.Module):
    """State-dependent mean; y-dependent log_std (stable and simple).
    Outputs means for:
      - hiring: [B, num_y]
      - v_prime: [B, num_y, K_v, Z]
    and maintains a learnable log_std per y and action dim.
    """
    def __init__(self, state_dim: int, num_y: int, K_v: int, Z: int, hidden=(64,64)):
        super().__init__()
        self.num_y = num_y
        self.K_v = K_v
        self.Z = Z
        self.A = 1 + K_v*Z  # hiring + flattened v′
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.hiring_mu = nn.Linear(in_dim, num_y)
        self.v_mu = nn.Linear(in_dim, num_y * K_v * Z)
        # y- and dim-specific log-std (state-independent): [num_y, A]
        self.log_std = nn.Parameter(torch.full((num_y, self.A), -0.5))

    def forward(self, x):
        B = x.shape[0]
        z = self.trunk(x)
        hire_mu = self.hiring_mu(z)                    # [B, Y]
        v_mu = self.v_mu(z).view(B, self.num_y, self.K_v, self.Z)
        return hire_mu, v_mu, self.log_std             # log_std: [Y, A]


# -------------------------
# PPO trainer
# -------------------------
class PPOTrainer:
    def __init__(self, foc_optimizer, bounds_processor, K_n: int, K_v: int, p,
                 hidden_actor=(64,64), hidden_critic=(64,64), device=None):
        """
        foc_optimizer: your FOCresidual instance (for rewards/dynamics)
        bounds_processor: your StateBoundsProcessor instance
        p: your global Parameters (needs .num_z, .beta etc.)
        """
        self.foc = foc_optimizer
        self.bp = bounds_processor
        self.p = p
        self.K_n = K_n
        self.K_v = K_v
        self.Z = p.num_z
        self.D = K_n + K_v
        self.A = 1 + K_v*self.Z
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(self.D, self.Z, K_v, self.Z, hidden_actor).to(self.device)
        self.critic = Critic(self.D, self.Z, hidden_critic).to(self.device)
        self.opt_actor = AdamW(self.actor.parameters(), lr=3e-4, betas=(0.9,0.999))
        self.opt_critic = AdamW(self.critic.parameters(), lr=3e-4, betas=(0.9,0.999))

        # convenience bounds (per-action dim)
        low = []
        high = []
        # hiring dim bound = state dim 0
        low.append(self.bp.lower_bounds[0].item())
        high.append(self.bp.upper_bounds[0].item())
        # v′ dims share per-k bounds, replicated across Z
        for k in range(K_v):
            lb = self.bp.lower_bounds[K_n + k].item()
            ub = self.bp.upper_bounds[K_n + k].item()
            for _ in range(self.Z):
                low.append(lb); high.append(ub)
        self.act_low = torch.tensor(low, device=self.device)
        self.act_high = torch.tensor(high, device=self.device)

    # ---------- rollout collection ----------
    @torch.no_grad()
    def collect(self, start_states: torch.Tensor, T: int, paths: int = 1) -> RolloutBatch:
        """Generate on-policy rollouts of length T from given start states.
        `start_states` must be normalized in [0,1] and shape [N, D].
        We replicate them over `paths` Monte-Carlo branches.
        """
        p, Z, K_n, K_v = self.p, self.Z, self.K_n, self.K_v
        device = self.device
        # replicate starting points across random paths
        states = start_states.repeat(paths, 1).to(device)
        N = states.shape[0]
        # init current y for each path
        y = torch.randint(0, Z, (N,), device=device)

        obs = torch.empty(T, N, self.D, device=device)
        ys = torch.empty(T, N, dtype=torch.long, device=device)
        actions = torch.empty(T, N, self.A, device=device)
        logps = torch.empty(T, N, device=device)
        rewards = torch.empty(T, N, device=device)
        values_t = torch.empty(T, N, device=device)
        y_nexts = torch.empty(T, N, dtype=torch.long, device=device)
        obs_next = torch.empty(T, N, self.D, device=device)
        values_tp1 = torch.empty(T, N, device=device)

        # unnormalized (physical) denorm states for building rewards
        states_phys = self.bp.denormalize(states)
        wages = torch.zeros(N, K_n, device=device)

        for t in range(T):
            obs[t] = states
            ys[t] = y
            # actor outputs (means and log_std table)
            hire_mu, v_mu, log_std_table = self.actor(states)
            # slice the row corresponding to each current y
            # shapes: [N], [N, K_v, Z], [N, A]
            idx = (torch.arange(N, device=device), y)
            mu_h = hire_mu[idx]
            mu_v = v_mu[torch.arange(N, device=device), y]         # [N, K_v, Z]
            mu_flat = torch.cat([mu_h.unsqueeze(1), mu_v.reshape(N, -1)], dim=1)  # [N, A]
            log_std = log_std_table[y]  # [N, A] by fancy indexing

            # sample via tanh-Normal and map to physical bounds
            raw_eps = torch.randn_like(mu_flat)
            a_phys, logp = _tanh_squash_and_logp(mu_flat, log_std, raw_eps,
                                                 self.act_low, self.act_high)
            actions[t] = a_phys
            logps[t] = logp

            # critic value at current (s,y)
            v_all = self.critic(states)                        # [N, Z]
            v_t = v_all[idx]                                   # [N]
            values_t[t] = v_t

            # ----- environment step (vectorized) -----
            # build E_{y'|y} v′ for worker decisions
            v_prime = a_phys[:, 1:].reshape(N, K_v, Z)
            v_prime_exp_all = torch.einsum('bkz,yz->bky', v_prime, self.foc.Z_trans_tensor)  # [N,K_v,Y]
            v_prime_exp = v_prime_exp_all[torch.arange(N, device=device), :, y]              # [N,K_v]
            re, pc = self.foc.getWorkerDecisions(v_prime_exp)

            size_phys = states_phys[:, :K_n]
            tot_size = size_phys.sum(dim=1)
            v_phys = states_phys[:, K_n:]
            # wages for two groups, following your code
            wages[:, 1:] = self.foc.pref.inv_utility(v_phys - p.beta * (v_prime_exp + re))
            wages[:, :1] = self.foc.pref.inv_utility(self.foc.v_0 - p.beta * (v_prime_exp + re))
            tot_wage = wages.sum(dim=1)

            hiring = a_phys[:, 0]
            r_t = self.foc.fun_prod[y] * self.foc.production(tot_size) - p.hire_c * hiring - tot_wage
            rewards[t] = r_t

            # draw next productivity and build next state (physical), then renormalize
            next_prod_probs = self.foc.Z_trans_tensor[y, :]                 # [N, Z]
            y_next = torch.multinomial(next_prod_probs, 1).squeeze(1)
            y_nexts[t] = y_next

            n1 = ((size_phys[:, 0] + size_phys[:, 1] + hiring) * pc.squeeze(1)).unsqueeze(1)
            next_phys = torch.empty_like(states_phys)
            next_phys[:, 0] = hiring
            next_phys[:, 1:K_n] = n1
            next_phys[:, K_n:] = v_prime[torch.arange(N, device=device), :, y_next]
            # safety
            assert not torch.isnan(next_phys).any()
            assert torch.all(next_phys[:, K_n:] >= 0)

            states_phys = next_phys
            nxt = self.bp.normalize(next_phys)
            obs_next[t] = nxt
            # V at next (s', y')
            v_all_tp1 = self.critic(nxt)
            values_tp1[t] = v_all_tp1[torch.arange(N, device=device), y_next]
            # advance
            states = nxt
            y = y_next

        return RolloutBatch(obs=obs, y=ys, actions=actions, logp=logps,
                             rewards=rewards, values=values_t,
                             y_next=y_nexts, obs_next=obs_next, value_next=values_tp1)

    # ---------- GAE ----------
    @staticmethod
    def compute_gae(rew, val, val_next, gamma: float, lam: float):
        """All are [T, N]. Returns advantages [T,N] and returns [T,N]."""
        T, N = rew.shape
        adv = torch.zeros_like(rew)
        gae = torch.zeros(N, device=rew.device)
        for t in reversed(range(T)):
            delta = rew[t] + gamma * val_next[t] - val[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = adv + val
        return adv, ret

    # ---------- PPO update ----------
    def ppo_update(self, batch: RolloutBatch, epochs=10, minibatch=4,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=1e-3):
        T, N, D = batch.obs.shape
        A = batch.actions.shape[-1]
        # flatten time and batch
        def flat(x):
            return x.reshape(T*N, *x.shape[2:]) if x.dim() > 2 else x.reshape(T*N)
        obs = flat(batch.obs)
        y = flat(batch.y)
        acts = flat(batch.actions)
        old_logp = flat(batch.logp)
        rew = batch.rewards
        val = batch.values
        val_next = batch.value_next
        with torch.no_grad():
            adv, ret = self.compute_gae(rew, val, val_next, self.p.beta, 0.95)
            adv_f = flat(adv)
            ret_f = flat(ret)
            # normalize advantages
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        idx = torch.randperm(T*N, device=self.device)
        mb = T*N // minibatch
        for _ in range(epochs):
            for k in range(minibatch):
                sel = idx[k*mb : (k+1)*mb]
                o = obs[sel]
                yy = y[sel]
                a = acts[sel]
                old_lp = old_logp[sel]
                targ_v = ret_f[sel]
                adv_k = adv_f[sel]

                # actor forward
                hire_mu, v_mu, log_std_tab = self.actor(o)
                # slice by current y
                mu_h = hire_mu[torch.arange(o.shape[0], device=self.device), yy]
                mu_v = v_mu[torch.arange(o.shape[0], device=self.device), yy]
                mu_flat = torch.cat([mu_h.unsqueeze(1), mu_v.reshape(o.shape[0], -1)], dim=1)
                log_std = log_std_tab[yy]

                # reconstruct the **pre-squash** variable by solving for u is not needed;
                # we compute new logp under current params for the **given** a
                # by sampling an equivalent u via inverse tanh is messy; instead,
                # recompute logp with fresh eps by matching distribution — PPO uses ratio of densities,
                # so we must evaluate new π(a). We'll map a → t in [-1,1], then u = atanh(t).
                #
                # a = low + (t+1)/2 * (high-low)  ⇒  t = 2*(a-low)/(high-low) - 1
                t = 2.0*(a - self.act_low)/ (self.act_high - self.act_low + 1e-8) - 1.0
                t = torch.clamp(t, -0.999999, 0.999999)
                u = 0.5*torch.log((1+t)/(1-t))     # atanh(t)
                sigma = torch.exp(log_std)
                base = -0.5 * (((u - mu_flat)/(sigma+1e-8))**2 + 2.0*log_std + math.log(2*math.pi))
                base = base.sum(dim=-1)
                log_det = torch.log((self.act_high - self.act_low)*0.5 + 1e-8).sum(dim=-1) \
                          + torch.log1p(-t*t + 1e-12).sum(dim=-1)
                new_logp = base - log_det

                ratio = torch.exp(new_logp - old_lp)
                surr1 = ratio * adv_k
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_k
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy (use base normal entropy, cheap and effective)
                ent = (0.5 * (1.0 + math.log(2*math.pi)) + log_std).sum(dim=-1).mean()

                # critic
                v_pred_all = self.critic(o)
                v_pred = v_pred_all[torch.arange(o.shape[0], device=self.device), yy]
                v_loss = 0.5 * (v_pred - targ_v).pow(2).mean()

                loss = policy_loss + vf_coef * v_loss - ent_coef * ent

                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_actor.step()
                self.opt_critic.step()

    # ---------- high-level train ----------
    def train(self, num_updates=1000, start_points=256, T=8, paths=1):
        self.actor.train(); self.critic.train()
        for upd in range(1, num_updates+1):
            # build randomized starting states similar to your current loop
            # one junior worker + random v, etc.
            s0 = torch.zeros(self.D, dtype=torch.float32, device=self.device)
            s0[0] = self.bp.normalize_dim(1.0, 0)  # 1 junior worker
            start = s0.unsqueeze(0).repeat(start_points, 1)
            # jitter around bounds
            start[:, 1:self.K_n] = torch.rand_like(start[:, 1:self.K_n])
            start[:, self.K_n:]  = torch.rand_like(start[:, self.K_n:])

            batch = self.collect(start, T=T, paths=paths)
            self.ppo_update(batch, epochs=10, minibatch=4)
            if upd % 50 == 0:
                with torch.no_grad():
                    J = batch.rewards.mean().item() / (1 - float(self.p.beta) + 1e-8)
                print(f"update {upd:4d}  avg episodic return≈{J:.3f}  std(log_std)={self.actor.log_std.std().item():.3f}")

    # ---------- evaluation (greedy) ----------
    @torch.no_grad()
    def act_greedy(self, obs_norm: torch.Tensor, y_idx: torch.Tensor) -> torch.Tensor:
        """Deterministic policy for evaluation (use means)."""
        hire_mu, v_mu, _ = self.actor(obs_norm)
        mu_h = hire_mu[torch.arange(obs_norm.shape[0], device=self.device), y_idx]
        mu_v = v_mu[torch.arange(obs_norm.shape[0], device=self.device), y_idx]
        mu_flat = torch.cat([mu_h.unsqueeze(1), mu_v.reshape(obs_norm.shape[0], -1)], dim=1)
        # map tanh(mu) to bounds (no noise)
        t = torch.tanh(mu_flat)
        a = self.act_low + (t + 1.0) * 0.5 * (self.act_high - self.act_low)
        return a  # [B, A] on physical scale
