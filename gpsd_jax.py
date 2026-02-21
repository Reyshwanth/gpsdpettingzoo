"""
JAX-accelerated GPSD (GPS-Denied Coverage) Environment.

A pure-JAX reimplementation of the PettingZoo GPSD environment, compatible
with JaxMARL's MultiAgentEnv interface.  All operations are JIT-compilable
and can be vmap'd across hundreds of parallel environments on GPU.

Key features ported from the PettingZoo version:
  - Unicycle kinematics (constant speed, discrete turn-rate control)
  - GPS-denied zone with EKF covariance growth / GPS reset
  - Cooperative coverage of POIs with covariance-dependent rewards
  - Inter-agent range measurements with EKF update
  - Body-frame relative observations (+y forward, +x right)

Usage:
    from gpsd_jax import GPSDJAX, GPSDState
    env = GPSDJAX(num_agents=5)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    actions = {a: 0 for a in env.agents}
    obs, state, rewards, dones, info = env.step(key, state, actions)
"""

import jax
import jax.numpy as jnp
import chex
import numpy as np
from functools import partial
from typing import Dict, Tuple, Optional
from flax import struct

# ---------------------------------------------------------------------------
# We define a lightweight MultiAgentEnv base + spaces inline so that this file
# is completely self-contained (no JaxMARL dependency required at runtime).
# If you have JaxMARL installed you can subclass its MultiAgentEnv instead.
# ---------------------------------------------------------------------------

class Space:
    """Minimal space descriptor."""
    pass

class Discrete(Space):
    def __init__(self, n: int):
        self.n = n
        self.shape = ()
        self.dtype = jnp.int32

    def sample(self, key: chex.PRNGKey):
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n)

class Box(Space):
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = jnp.float32


# ---------------------------------------------------------------------------
# State dataclass  (all arrays → JIT-friendly)
# ---------------------------------------------------------------------------
@struct.dataclass
class GPSDState:
    """Full JAX-compatible state for the GPSD environment."""
    p_pos: chex.Array        # [num_agents, 2] true positions
    p_belief: chex.Array     # [num_agents, 2] believed positions (dead-reckoning)
    heading: chex.Array      # [num_agents,]   heading angles (rad)
    p_cov: chex.Array        # [num_agents, 2, 2] position covariance matrices
    poi_pos: chex.Array      # [num_pois, 2] POI positions (fixed)
    covered: chex.Array      # [num_pois,] bool mask of covered POIs
    done: chex.Array         # [num_agents,]
    step: int                # current timestep


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class GPSDJAX:
    """Pure-JAX GPSD environment following the JaxMARL API.

    All methods are decorated with ``@partial(jax.jit, ...)`` so they can be
    compiled once and executed on GPU.  Use ``jax.vmap(env.step)`` to run
    thousands of environments in parallel.
    """

    def __init__(
        self,
        num_agents: int = 5,
        cell_width: float = 0.25,
        max_steps: int = 100,
        min_w: float = -1.0,
        max_w: float = 1.0,
        speed: float = 0.2,
        r_c: float = 0.5,
        cov_c: float = 0.15,
        p_noise: float = 0.1,
        r_cov: float = 0.01,
        local_ratio: float = 0.1,
        dt: float = 0.1,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.dt = dt
        self.speed = speed
        self.r_c = r_c
        self.cov_c = cov_c
        self.p_noise = p_noise
        self.r_cov = r_cov
        self.local_ratio = local_ratio

        # 5 discrete turn rates
        self.omega_values = jnp.linspace(min_w, max_w, 5)
        self.num_actions = 5

        # POI grid (computed once, stored as JAX arrays)
        zone_min, zone_max = -0.5, 0.5
        zone_size = zone_max - zone_min
        n_cells = int(np.round(zone_size / cell_width))
        actual_cw = zone_size / n_cells
        centers_1d = zone_min + actual_cw * (np.arange(n_cells) + 0.5)
        cx, cy = np.meshgrid(centers_1d, centers_1d)
        self.poi_positions = jnp.array(
            np.stack([cx.ravel(), cy.ravel()], axis=-1)
        )
        self.num_pois = len(self.poi_positions)

        # Agent / observation / action spaces
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_range = jnp.arange(num_agents)

        # Obs dim: 1 (heading) + 2 (pos) + 1 (cov) + num_pois*2 + (N-1)*2 + (N-1)*2
        self.obs_dim = 4 + self.num_pois * 2 + (num_agents - 1) * 2 + (num_agents - 1) * 2
        self.observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.num_actions) for a in self.agents}

    # ----- space helpers (JaxMARL compat) -----
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    @property
    def name(self):
        return "GPSD"

    # ===================================================================
    # RESET
    # ===================================================================
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], GPSDState]:
        key, k_pos, k_angle, k_heading = jax.random.split(key, 4)

        # Random swarm center outside GPS-denied zone
        # We sample until outside; for JIT-friendliness just reject-sample
        # up to a fixed number of tries and fall back to a corner.
        swarm_center = self._sample_outside_zone(k_pos)

        # Agents clustered within 0.3 of swarm center
        k_angles, k_radii = jax.random.split(k_angle)
        angles = jax.random.uniform(k_angles, shape=(self.num_agents,), minval=0.0, maxval=2.0 * jnp.pi)
        radii = jax.random.uniform(k_radii, shape=(self.num_agents,), minval=0.0, maxval=0.3)
        offsets = jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=-1)
        p_pos = swarm_center[None, :] + offsets
        p_belief = p_pos.copy()

        # Headings: point toward center (0,0)
        heading = jnp.arctan2(-p_pos[:, 1], -p_pos[:, 0])

        p_cov = jnp.zeros((self.num_agents, 2, 2))
        covered = jnp.zeros(self.num_pois, dtype=jnp.bool_)
        done = jnp.zeros(self.num_agents, dtype=jnp.bool_)

        state = GPSDState(
            p_pos=p_pos,
            p_belief=p_belief,
            heading=heading,
            p_cov=p_cov,
            poi_pos=self.poi_positions,
            covered=covered,
            done=done,
            step=0,
        )

        obs = self.get_obs(state)
        return obs, state

    # ===================================================================
    # STEP
    # ===================================================================
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: GPSDState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], GPSDState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)

        obs_st, state_st, rewards, dones, info = self.step_env(key, state, actions)

        # Auto-reset on episode end
        obs_re, state_re = self.reset(key_reset)
        all_done = dones["__all__"]

        state_out = jax.tree.map(
            lambda x, y: jax.lax.select(all_done, x, y), state_re, state_st
        )
        obs_out = jax.tree.map(
            lambda x, y: jax.lax.select(all_done, x, y), obs_re, obs_st
        )
        return obs_out, state_out, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: GPSDState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], GPSDState, Dict[str, float], Dict[str, bool], Dict]:
        key, k_range = jax.random.split(key)

        # Decode actions → omega values  [num_agents,]
        action_arr = jnp.array([actions[a] for a in self.agents])
        omega = self.omega_values[action_arr]

        # ---- Unicycle kinematics ----
        new_pos, new_belief, new_heading = self._integrate(
            state.p_pos, state.p_belief, state.heading, omega
        )

        # ---- EKF covariance prediction ----
        new_cov = self._update_covariance_predict(new_pos, state.p_cov)

        # ---- EKF range measurement update ----
        new_belief, new_cov = self._update_ekf_range(
            k_range, new_pos, new_belief, new_cov
        )

        # ---- Coverage check ----
        new_covered, coverage_reward = self._check_coverage(
            new_pos, new_cov, state.covered
        )

        # ---- Rewards ----
        local_rewards = self._local_reward(new_pos, new_cov)
        global_rew = coverage_reward

        rewards = {
            a: local_rewards[i] * self.local_ratio + global_rew * (1.0 - self.local_ratio)
            for i, a in enumerate(self.agents)
        }

        new_step = state.step + 1
        done_flag = new_step >= self.max_steps
        done_arr = jnp.full(self.num_agents, done_flag)

        state_out = GPSDState(
            p_pos=new_pos,
            p_belief=new_belief,
            heading=new_heading,
            p_cov=new_cov,
            poi_pos=state.poi_pos,
            covered=new_covered,
            done=done_arr,
            step=new_step,
        )

        obs = self.get_obs(state_out)
        dones = {a: done_arr[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = jnp.all(done_arr)

        # ---- Environment metrics ----
        cov_traces = jax.vmap(jnp.trace)(new_cov)  # [N,]
        in_zone = self._is_in_gpsd_zone(new_pos)   # [N,]
        num_covered = jnp.sum(new_covered.astype(jnp.float32))
        info = {
            # Coverage
            "coverage_ratio": num_covered / self.num_pois,
            "num_pois_covered": num_covered,
            "all_pois_covered": jnp.all(new_covered).astype(jnp.float32),
            # Covariance
            "mean_cov_trace": jnp.mean(cov_traces),
            "max_cov_trace": jnp.max(cov_traces),
            "num_agents_above_cov_thresh": jnp.sum(
                (cov_traces > self.cov_c).astype(jnp.float32)
            ),
            # Zone
            "num_agents_in_zone": jnp.sum(in_zone.astype(jnp.float32)),
            # Rewards breakdown
            "mean_local_reward": jnp.mean(local_rewards),
            "coverage_reward": coverage_reward,
            # Step
            "env_step": new_step.astype(jnp.float32),
        }
        # Broadcast scalars to per-agent arrays so reshaping in training works
        info = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.num_agents,)), info
        )

        return obs, state_out, rewards, dones, info

    # ===================================================================
    # OBSERVATIONS  (body-frame: +y forward, +x right)
    # ===================================================================
    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: GPSDState) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0,))
        def _obs(aidx):
            heading_val = state.heading[aidx]
            belief = state.p_belief[aidx]
            cov_trace = jnp.trace(state.p_cov[aidx])

            # POI relative positions in body frame
            poi_rel_world = state.poi_pos - belief[None, :]           # [N_r, 2]
            poi_rel_body = self._world_to_body_vmap(poi_rel_world, heading_val) # [N_r, 2]

            # Other agents' relative positions in body frame
            other_beliefs = jnp.concatenate([
                state.p_belief[:aidx],
                state.p_belief[aidx + 1:]
            ], axis=0)  # [N-1, 2]
            other_rel_world = other_beliefs - belief[None, :]
            other_rel_body = self._world_to_body_vmap(other_rel_world, heading_val) # [N-1, 2]

            # Other agents' comm info: [range_from_belief, cov_trace]
            other_cov_traces = jnp.concatenate([
                jnp.array([jnp.trace(state.p_cov[j]) for j in range(aidx)]),
                jnp.array([jnp.trace(state.p_cov[j]) for j in range(aidx + 1, self.num_agents)]),
            ])
            other_dists = jnp.sqrt(jnp.sum(other_rel_world ** 2, axis=-1))
            # -1 signals out of range
            range_meas = jnp.where(other_dists <= self.r_c, other_dists, -1.0)
            other_comm = jnp.stack([range_meas, other_cov_traces], axis=-1)  # [N-1, 2]

            return jnp.concatenate([
                jnp.array([heading_val]),
                belief,
                jnp.array([cov_trace]),
                poi_rel_body.flatten(),
                other_rel_body.flatten(),
                other_comm.flatten(),
            ])

        # vmap doesn't work cleanly with dynamic indexing of other agents,
        # so we build per-agent
        obs_all = jnp.stack([_obs_single(i, state, self) for i in range(self.num_agents)])
        return {a: obs_all[i] for i, a in enumerate(self.agents)}

    # ===================================================================
    # PHYSICS HELPERS
    # ===================================================================
    def _integrate(self, p_pos, p_belief, heading, omega):
        """Unicycle kinematics for all agents (vectorised)."""
        speed = self.speed
        dt = self.dt

        # dp when omega is non-negligible
        new_heading = heading + omega * dt
        v = speed  # scalar speed

        # Compute displacement using unicycle model
        # When |omega| > eps:  dp = (v/omega) * [sin(h+w*dt) - sin(h), -cos(h+w*dt) + cos(h)]
        # When |omega| ~ 0:   dp = v*dt * [cos(h), sin(h)]
        eps = 1e-8
        large_omega = jnp.abs(omega) > eps

        dp_turn = jnp.stack([
            (v / (omega + eps)) * (jnp.sin(new_heading) - jnp.sin(heading)),
            (v / (omega + eps)) * (-jnp.cos(new_heading) + jnp.cos(heading)),
        ], axis=-1)

        dp_straight = jnp.stack([
            v * dt * jnp.cos(heading),
            v * dt * jnp.sin(heading),
        ], axis=-1)

        dp = jnp.where(large_omega[:, None], dp_turn, dp_straight)
        new_pos = p_pos + dp
        new_belief = p_belief + dp  # dead reckoning propagation

        return new_pos, new_belief, new_heading

    def _is_in_gpsd_zone(self, pos):
        """Check if position(s) are inside the GPS-denied zone [-0.5, 0.5]^2.
        Works for single pos (2,) or batched (N, 2)."""
        return (jnp.abs(pos[..., 0]) <= 0.5) & (jnp.abs(pos[..., 1]) <= 0.5)

    def _update_covariance_predict(self, p_pos, p_cov):
        """EKF prediction: grow covariance inside GPSD zone, reset outside."""
        in_zone = self._is_in_gpsd_zone(p_pos)  # [N,]
        q = (self.p_noise ** 2) * jnp.eye(2)
        # F = I for constant-velocity model
        new_cov = p_cov + q[None, :, :]  # EKF predict: P = F P F^T + Q (F=I)
        # Outside zone: reset to 0
        new_cov = jnp.where(in_zone[:, None, None], new_cov, jnp.zeros_like(new_cov))
        return new_cov

    def _update_ekf_range(self, key, p_pos, p_belief, p_cov):
        """EKF measurement update using pairwise range measurements.

        For each ordered pair (i,j) within r_c distance, agent i performs
        a Kalman update using the noisy range measurement.  This is
        implemented via a sequential scan over pairs for JIT compatibility.
        """
        n = self.num_agents

        def _single_update(carry, pair_idx):
            belief, cov, key = carry
            i = pair_idx // n
            j = pair_idx % n

            # Skip self-pairs
            is_self = (i == j)

            # True range
            true_range = jnp.sqrt(jnp.sum((p_pos[i] - p_pos[j]) ** 2))
            in_range = true_range <= self.r_c

            # Noisy measurement
            key, k_noise = jax.random.split(key)
            z = true_range + jax.random.normal(k_noise) * jnp.sqrt(self.r_cov)
            z = jnp.maximum(z, 1e-8)

            # Predicted range from beliefs
            delta = belief[i] - belief[j]
            pred_range = jnp.sqrt(jnp.sum(delta ** 2))
            safe_pred = jnp.maximum(pred_range, 1e-8)

            # Jacobian H_i = delta^T / ||delta||  shape (1, 2)
            H_i = (delta / safe_pred).reshape(1, 2)
            H_j = -H_i

            # Innovation covariance S
            S = H_i @ cov[i] @ H_i.T + H_j @ cov[j] @ H_j.T + self.r_cov
            S_inv = 1.0 / jnp.maximum(S[0, 0], 1e-12)

            # Innovation
            y = z - pred_range

            # Kalman gain
            K = cov[i] @ H_i.T * S_inv  # (2, 1)

            # Updated belief and covariance
            new_belief_i = belief[i] + (K * y).flatten()
            I2 = jnp.eye(2)
            new_cov_i = (I2 - K @ H_i) @ cov[i]

            # Only apply if in range and not self
            apply = in_range & (~is_self)
            updated_belief = jnp.where(apply, new_belief_i, belief[i])
            updated_cov = jnp.where(apply, new_cov_i, cov[i])

            belief = belief.at[i].set(updated_belief)
            cov = cov.at[i].set(updated_cov)

            return (belief, cov, key), None

        # Also reset belief to true pos outside zone
        in_zone = self._is_in_gpsd_zone(p_pos)
        p_belief = jnp.where(in_zone[:, None], p_belief, p_pos)

        pair_indices = jnp.arange(n * n)
        (p_belief, p_cov, _), _ = jax.lax.scan(
            _single_update, (p_belief, p_cov, key), pair_indices
        )

        return p_belief, p_cov

    # ===================================================================
    # REWARD
    # ===================================================================
    def _local_reward(self, p_pos, p_cov):
        """Per-agent local reward (vectorised)."""
        cov_traces = jax.vmap(jnp.trace)(p_cov)  # [N,]
        in_zone = self._is_in_gpsd_zone(p_pos)   # [N,]

        # Step penalty
        rew = jnp.full(self.num_agents, -0.1)

        # Comm reward: for each agent, sum 1/(1+cov_j) for neighbors j within r_c with cov < threshold
        def _comm_reward(i):
            dists = jnp.sqrt(jnp.sum((p_pos - p_pos[i][None, :]) ** 2, axis=-1))
            in_comm = (dists <= self.r_c) & (cov_traces < self.cov_c) & (jnp.arange(self.num_agents) != i)
            return jnp.sum(jnp.where(in_comm, 1.0 / (1.0 + cov_traces), 0.0))

        comm_rew = jax.vmap(_comm_reward)(jnp.arange(self.num_agents))
        rew = rew + comm_rew

        # In-zone high-cov penalty
        excess_cov = jnp.maximum(cov_traces - self.cov_c, 0.0)
        zone_penalty = jnp.where(in_zone, jnp.exp(excess_cov) - 1.0, 0.0)
        # Out-of-zone: penalize distance from center
        dist_to_center = jnp.sqrt(jnp.sum(p_pos ** 2, axis=-1))
        outside_penalty = jnp.where(~in_zone, jnp.exp(dist_to_center - 0.5), 0.0)

        rew = rew - zone_penalty - outside_penalty
        return rew

    def _check_coverage(self, p_pos, p_cov, covered):
        """Check which POIs are newly covered and compute global reward."""
        cov_traces = jax.vmap(jnp.trace)(p_cov)  # [N,]

        def _check_poi(carry, poi_idx):
            covered, rew = carry
            poi = self.poi_positions[poi_idx]  # use self ref (static)

            dists = jnp.sqrt(jnp.sum((p_pos - poi[None, :]) ** 2, axis=-1))  # [N,]
            # Best agent: closest with smallest cov, within r_c
            in_range = dists < self.r_c
            # Among agents in range, find best cov_trace
            best_cov = jnp.where(in_range, cov_traces, jnp.inf).min()
            best_dist = jnp.where(in_range, dists, jnp.inf).min()

            can_cover = (best_dist < self.r_c) & (best_cov < self.cov_c) & (~covered[poi_idx])
            coverage_rew = jnp.where(can_cover, 10.0 / (1.0 + best_cov), 0.0)

            # Penalty for being close but cov too high
            close_but_high = (best_dist < self.r_c) & (best_cov >= self.cov_c) & (~covered[poi_idx])
            penalty = jnp.where(close_but_high, jnp.exp(best_cov - self.cov_c), 0.0)

            new_covered = covered.at[poi_idx].set(covered[poi_idx] | can_cover)
            new_rew = rew + coverage_rew - penalty
            return (new_covered, new_rew), None

        poi_indices = jnp.arange(self.num_pois)
        (new_covered, total_rew), _ = jax.lax.scan(
            _check_poi, (covered, 0.0), poi_indices
        )

        # Bonus for all covered
        all_covered_bonus = jnp.where(jnp.all(new_covered), 50.0, 0.0)
        total_rew = total_rew + all_covered_bonus

        return new_covered, total_rew

    # ===================================================================
    # COORDINATE TRANSFORM
    # ===================================================================
    def _world_to_body(self, rel_world, heading):
        """Transform world-frame relative pos to body frame.
           +x = right, +y = forward (along heading)."""
        cos_h = jnp.cos(heading)
        sin_h = jnp.sin(heading)
        x_body =  sin_h * rel_world[0] - cos_h * rel_world[1]
        y_body =  cos_h * rel_world[0] + sin_h * rel_world[1]
        return jnp.array([x_body, y_body])

    def _world_to_body_vmap(self, rel_batch, heading):
        """Vectorised body-frame transform for a batch of relative positions."""
        cos_h = jnp.cos(heading)
        sin_h = jnp.sin(heading)
        x_body =  sin_h * rel_batch[:, 0] - cos_h * rel_batch[:, 1]
        y_body =  cos_h * rel_batch[:, 0] + sin_h * rel_batch[:, 1]
        return jnp.stack([x_body, y_body], axis=-1)

    # ===================================================================
    # SAMPLING HELPER
    # ===================================================================
    def _sample_outside_zone(self, key):
        """Sample a point in [-1, 1]^2 that is outside the GPSD zone [-0.5, 0.5]^2.
        Uses rejection sampling with a fixed budget; falls back to a corner."""
        def _body(carry):
            key, pos, found = carry
            key, k = jax.random.split(key)
            candidate = jax.random.uniform(k, shape=(2,), minval=-1.0, maxval=1.0)
            outside = ~self._is_in_gpsd_zone(candidate)
            pos = jnp.where(outside & (~found), candidate, pos)
            found = found | outside
            return (key, pos, found)

        def _cond(carry):
            _, _, found = carry
            return ~found

        init = (key, jnp.array([0.75, 0.75]), jnp.bool_(False))
        # Use bounded while loop via lax.while_loop
        _, pos, _ = jax.lax.while_loop(_cond, _body, init)
        return pos


# -----------------------------------------------------------------------
# Module-level observation builder (avoids vmap issues with dynamic slicing)
# -----------------------------------------------------------------------
def _obs_single(i, state, env):
    """Build observation for agent i."""
    heading_val = state.heading[i]
    belief = state.p_belief[i]
    cov_trace = jnp.trace(state.p_cov[i])

    # POI relative positions in body frame
    poi_rel = state.poi_pos - belief[None, :]
    poi_body = env._world_to_body_vmap(poi_rel, heading_val)

    # Other agents: build [N-1, 2] arrays using integer indexing (not boolean)
    other_idx = jnp.array([j for j in range(env.num_agents) if j != i])
    other_beliefs = state.p_belief[other_idx]

    other_rel = other_beliefs - belief[None, :]
    other_body = env._world_to_body_vmap(other_rel, heading_val)

    # Comm info
    other_cov_tr = jax.vmap(jnp.trace)(state.p_cov[other_idx])
    other_dists = jnp.sqrt(jnp.sum(other_rel ** 2, axis=-1))
    range_meas = jnp.where(other_dists <= env.r_c, other_dists, -1.0)
    other_comm = jnp.stack([range_meas, other_cov_tr], axis=-1)

    return jnp.concatenate([
        jnp.array([heading_val]),
        belief,
        jnp.array([cov_trace]),
        poi_body.flatten(),
        other_body.flatten(),
        other_comm.flatten(),
    ])


# -----------------------------------------------------------------------
# Convenience: Log wrapper (same as JaxMARL's LogWrapper but self-contained)
# -----------------------------------------------------------------------
@struct.dataclass
class LogEnvState:
    env_state: GPSDState
    episode_returns: chex.Array
    episode_lengths: chex.Array
    returned_episode_returns: chex.Array
    returned_episode_lengths: chex.Array


class GPSDLogWrapper:
    """Wraps GPSDJAX to track episode returns/lengths (JIT-compatible)."""

    def __init__(self, env: GPSDJAX):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros(self._env.num_agents),
            episode_lengths=jnp.zeros(self._env.num_agents),
            returned_episode_returns=jnp.zeros(self._env.num_agents),
            returned_episode_lengths=jnp.zeros(self._env.num_agents),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, actions
        )
        ep_done = done["__all__"]
        reward_arr = jnp.stack([reward[a] for a in self._env.agents])
        new_ret = state.episode_returns + reward_arr * self._env.num_agents
        new_len = state.episode_lengths + 1

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_ret * (1 - ep_done),
            episode_lengths=new_len * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
                + new_ret * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
                + new_len * ep_done,
        )
        # Episode-level metrics (only valid when ep_done)
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full(self._env.num_agents, ep_done)
        # Pass through env metrics (already per-agent broadcast from step_env)
        return obs, state, reward, done, info


# -----------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    env = GPSDJAX(num_agents=3, cell_width=0.5)
    print(f"GPSD JAX environment: {env.num_agents} agents, {env.num_pois} POIs")
    print(f"Observation dim: {env.obs_dim}")

    obs, state = env.reset(key)
    print(f"Initial positions:\n{state.p_pos}")
    print(f"Obs shape: {obs['agent_0'].shape}")

    for t in range(5):
        key, k = jax.random.split(key)
        actions = {a: jax.random.randint(k, (), 0, 5) for a in env.agents}
        obs, state, rewards, dones, info = env.step(k, state, actions)
        r_str = ", ".join(f"{a}: {rewards[a]:.3f}" for a in env.agents)
        print(f"  step {t+1}: rewards = [{r_str}]  done={dones['__all__']}")

    print("\nSmoke test passed!")
