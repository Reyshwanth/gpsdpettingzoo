"""MAPPO (Multi-Agent PPO) training script for the GPSD environment.

Implements Centralized Training with Decentralized Execution (CTDE):
  - Actor: uses only the agent's LOCAL observation to select actions.
  - Critic: uses the GLOBAL state (concatenation of ALL agents' observations
    from the same game instance) to estimate value.

At test time only the actor is needed, so execution is fully decentralized.

The environment, rollout, GAE, and PPO update are otherwise identical to the
DTDE version in train_gpsd_ppo.py.

Usage:
    python train_gpsd_mappo.py                        # defaults
    python train_gpsd_mappo.py --total-timesteps 500000 --num-envs 20
    python train_gpsd_mappo.py --track --wandb-project-name gpsd  # W&B logging
"""

# flake8: noqa

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.mpe.gpsd.gpsd import parallel_env as make_gpsd_parallel_env


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="MAPPO (CTDE) for GPSD environment")

    # --- General ---
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="gpsd-ppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture a video of the trained agent after training")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save the final trained model")

    # --- GPSD environment parameters ---
    parser.add_argument("--num-agents", type=int, default=5,
        help="number of agents (N_a) in the GPSD environment")
    parser.add_argument("--cell-width", type=float, default=0.25,
        help="cell width for the POI grid in the GPS denied zone")
    parser.add_argument("--max-cycles", type=int, default=250,
        help="max steps per episode in the GPSD environment")
    parser.add_argument("--speed", type=float, default=0.1,
        help="constant forward speed of agents")
    parser.add_argument("--r-c", type=float, default=0.3,
        help="communication/coverage radius")
    parser.add_argument("--cov-c", type=float, default=0.1,
        help="coverage covariance trace threshold")
    parser.add_argument("--local-ratio", type=float, default=0.9995,
        help="weight for local vs global reward (0=fully global, 1=fully local)")

    # --- PPO hyperparameters ---
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=40,
        help="the number of parallel environment slots (must be a multiple of --num-agents)")
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5,
        help="the K epochs to update the policy")
    parser.add_argument("--critic-epochs", type=int, default=5,
        help="extra critic-only epochs per update (on top of update-epochs)")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.05,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="toggles whether to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.00,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=10.0,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.02,
        help="the target KL divergence threshold")
    parser.add_argument("--norm-reward", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles reward normalization using running mean/std")

    args = parser.parse_args()

    assert args.num_envs % args.num_agents == 0, (
        f"--num-envs ({args.num_envs}) must be a multiple of --num-agents ({args.num_agents})"
    )
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# ---------------------------------------------------------------------------
# Reward normalisation (Welford online mean/variance)
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm."""
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + np.square(delta) * self.count * batch_count / tot_count) / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ---------------------------------------------------------------------------
# Value normalisation (PopArt-lite) — normalise value targets
# ---------------------------------------------------------------------------
class ValueNorm:
    """Normalise value targets by running mean/std of returns.

    The critic learns in normalised space; outputs are de-normalised for GAE.
    This decouples the critic from the absolute scale of rewards, which is
    highly non-stationary in coverage tasks.
    """
    def __init__(self, clip=10.0, epsilon=1e-8):
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = epsilon
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x):
        """Update with a flat numpy array of returns."""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        self.running_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        self.running_var = (
            m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        ) / tot_count
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.running_var) + self.epsilon

    def normalize(self, x):
        """Normalise a torch tensor of returns → targets for the critic."""
        return torch.clamp(
            (x - self.running_mean) / self.std, -self.clip, self.clip
        )

    def denormalize(self, x):
        """De-normalise critic output → original-scale values."""
        return x * self.std + self.running_mean


# ---------------------------------------------------------------------------
# Observation normalisation (per-feature running mean/std)
# ---------------------------------------------------------------------------
class RunningMeanStdVec:
    """Tracks per-feature running mean and variance (Welford's algorithm)."""
    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        """Update statistics with a batch of observations. x: (batch, *shape)"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + np.square(delta) * self.count * batch_count / tot_count) / tot_count
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        """Normalize a torch tensor in-place using running stats. Returns tensor."""
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std_t = torch.sqrt(torch.tensor(self.var, dtype=torch.float32, device=x.device)) + 1e-8
        return torch.clamp((x - mean_t) / std_t, -clip, clip)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Helper: build global state from vectorised observations
# ---------------------------------------------------------------------------
def extract_world_obs(info, num_envs, world_obs_dim, device):
    """Extract world_observation from the info list returned by the vectorised env.

    Each element of info is a dict with key 'world_obs' containing a numpy
    array of shape (world_obs_dim,).  All agents in the same game share the
    same world_obs, but supersuit duplicates it per-agent.

    Returns:
        Tensor of shape (num_envs, world_obs_dim).
    """
    if isinstance(info, (list, tuple)):
        wo_list = []
        for item in info:
            if isinstance(item, dict) and 'world_obs' in item:
                wo_list.append(item['world_obs'])
            else:
                wo_list.append(np.zeros(world_obs_dim, dtype=np.float32))
        return torch.tensor(np.stack(wo_list), dtype=torch.float32).to(device)
    else:
        return torch.zeros(num_envs, world_obs_dim, device=device)


def build_global_state(local_obs, num_agents, world_obs=None):
    """Construct the centralised critic input from per-agent observations
    and (optionally) the world observation.

    The vectorised env lays out agents as:
        [game0_agent0, game0_agent1, ..., game0_agentN,
         game1_agent0, ...]

    We reshape to (num_games, num_agents, obs_dim), concatenate along the
    agent axis → (num_games, num_agents*obs_dim), then append the
    world_observation (shared per game) and repeat so every agent in the
    same game sees the *same* global state.

    Args:
        local_obs:  Tensor  (num_envs, obs_dim)
        num_agents: int
        world_obs:  Tensor  (num_envs, world_obs_dim) or None

    Returns:
        global_state: Tensor (num_envs, global_dim)
            where global_dim = num_agents*obs_dim + world_obs_dim
    """
    total = local_obs.shape[0]
    obs_dim = local_obs.shape[1]
    num_games = total // num_agents

    # (num_games, num_agents, obs_dim)
    reshaped = local_obs.view(num_games, num_agents, obs_dim)
    # (num_games, num_agents * obs_dim)
    concat = reshaped.reshape(num_games, num_agents * obs_dim)

    if world_obs is not None:
        # world_obs is (total, world_obs_dim); identical for agents in the
        # same game, so just take one per game.
        wo_per_game = world_obs.view(num_games, num_agents, -1)[:, 0, :]
        concat = torch.cat([concat, wo_per_game], dim=-1)

    global_dim = concat.shape[-1]
    # Broadcast back: each agent in a game gets the same global state
    global_state = concat.unsqueeze(1).expand(num_games, num_agents, global_dim)
    # Flatten back to (total, global_dim)
    return global_state.reshape(total, global_dim)


# ---------------------------------------------------------------------------
# MAPPO Agent: separate actor (local) + critic (centralised)
# ---------------------------------------------------------------------------
class MAPPOAgent(nn.Module):
    """Actor uses local obs; critic uses global state (all agents' obs)."""

    def __init__(self, obs_dim: int, act_dim: int, num_agents: int,
                 global_dim: int = None):
        super().__init__()
        self.num_agents = num_agents
        if global_dim is None:
            global_dim = obs_dim * num_agents

        # --- Decentralised Actor (local observations only) ---
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(128, act_dim), std=0.01)

        # --- Centralised Critic (global state) ---
        # ReLU + LayerNorm: avoids tanh saturation, allows unbounded gradients
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(global_dim, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.critic_head = layer_init(nn.Linear(128, 1), std=0.01)

    def get_value(self, global_state):
        """Centralised value: V(s_global)."""
        return self.critic_head(self.critic_net(global_state))

    def get_action(self, local_obs, action=None):
        """Decentralised policy: pi(a | o_local)."""
        hidden = self.actor_net(local_obs)
        logits = self.actor_head(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_action_and_value(self, local_obs, global_state, action=None):
        """Combined forward for training: actor uses local, critic uses global."""
        act, logprob, entropy = self.get_action(local_obs, action)
        value = self.get_value(global_state)
        return act, logprob, entropy, value


# ---------------------------------------------------------------------------
# Environment factory (same as DTDE version)
# ---------------------------------------------------------------------------
def make_env(args):
    env = make_gpsd_parallel_env(
        N_a=args.num_agents,
        cell_width=args.cell_width,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        speed=args.speed,
        r_c=args.r_c,
        cov_c=args.cov_c,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    num_copies = args.num_envs // args.num_agents
    envs = ss.concat_vec_envs_v1(
        env, num_copies, num_cpus=0, base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    return envs


# ---------------------------------------------------------------------------
# Video recording helper
# ---------------------------------------------------------------------------
def record_video(args, agent_model, run_name, num_episodes=1):
    """Record a video — uses only the decentralised actor for execution."""
    try:
        import imageio
    except ImportError:
        print("imageio not installed – skipping video recording.")
        return

    print("Recording video of trained policy (decentralised execution)...")
    device = next(agent_model.parameters()).device

    env = make_gpsd_parallel_env(
        N_a=args.num_agents,
        cell_width=args.cell_width,
        max_cycles=args.max_cycles,
        speed=args.speed,
        r_c=args.r_c,
        cov_c=args.cov_c,
        render_mode="rgb_array",
    )

    frames = []
    for ep in range(num_episodes):
        obs, infos = env.reset()
        for _ in range(args.max_cycles):
            with torch.no_grad():
                obs_array = np.stack([obs[a] for a in env.agents])
                obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)
                # Decentralised execution: only the actor is used
                actions, _, _ = agent_model.get_action(obs_tensor)

            action_dict = {
                agent_name: actions[i].item()
                for i, agent_name in enumerate(env.agents)
            }
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if all(terminations.values()) or all(truncations.values()):
                break

    env.close()

    if frames:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        video_path = f"runs/{run_name}/gpsd_trained.mp4"
        imageio.mimwrite(video_path, frames, fps=30, codec='libx264')
        print(f"Saved video → {video_path}")
    else:
        print("No frames captured – skipping video save.")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    args = parse_args()
    run_name = f"mappo_{args.seed}_{int(time.time())}"
    print(f"\n{'='*70}")
    print(f"  GPSD MAPPO (CTDE) Training  —  {run_name}")
    print(f"{'='*70}")
    print(args)
    print()

    # --- Weights & Biases ---
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # --- TensorBoard ---
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    envs = make_env(args)
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = envs.single_action_space.n
    # World observation dim: N_a * 6 + N_r * 3
    n_cells = int(np.round(1.0 / args.cell_width))
    n_pois = n_cells * n_cells
    world_obs_dim = args.num_agents * 6 + n_pois * 3
    global_dim = obs_dim * args.num_agents + world_obs_dim
    print(f"Observation dim (local)  : {obs_dim}")
    print(f"World obs dim            : {world_obs_dim}")
    print(f"Global state dim (critic): {global_dim}")
    print(f"Action dim               : {act_dim}")
    print(f"Num vec envs             : {args.num_envs}  "
          f"({args.num_envs // args.num_agents} games × {args.num_agents} agents)")
    print(f"Batch size               : {args.batch_size}")
    print(f"Minibatch size           : {args.minibatch_size}")
    print()

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # ------------------------------------------------------------------
    # MAPPO Agent & optimisers (fully decoupled actor/critic optimizers)
    # ------------------------------------------------------------------
    agent = MAPPOAgent(obs_dim, act_dim, args.num_agents,
                       global_dim=global_dim).to(device)
    actor_params_list = list(agent.actor_net.parameters()) + list(agent.actor_head.parameters())
    critic_params_list = list(agent.critic_net.parameters()) + list(agent.critic_head.parameters())
    actor_optimizer  = optim.Adam(actor_params_list,  lr=args.learning_rate,       eps=1e-5)
    critic_optimizer = optim.Adam(critic_params_list, lr=args.learning_rate * 3.0, eps=1e-5)
    actor_params = sum(p.numel() for p in agent.actor_net.parameters()) + \
                   sum(p.numel() for p in agent.actor_head.parameters())
    critic_params = sum(p.numel() for p in agent.critic_net.parameters()) + \
                    sum(p.numel() for p in agent.critic_head.parameters())
    print(f"Actor parameters : {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print(f"Total parameters : {actor_params + critic_params:,}")
    print()

    # ------------------------------------------------------------------
    # Rollout storage
    # ------------------------------------------------------------------
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    global_states = torch.zeros(
        (args.num_steps, args.num_envs, global_dim)
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_world_obs = extract_world_obs(info, args.num_envs, world_obs_dim, device)
    next_termination = torch.zeros(args.num_envs).to(device)
    next_truncation = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(f"Number of PPO updates: {num_updates}\n")

    # --- Reward normalization ---
    reward_rms = RunningMeanStd() if args.norm_reward else None

    # --- Value normalization (PopArt-lite) ---
    value_norm = ValueNorm()

    # --- Observation normalization ---
    obs_rms = RunningMeanStdVec((obs_dim,))
    global_obs_rms = RunningMeanStdVec((global_dim,))

    # --- Manual episode tracking ---
    episode_rewards = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_coverage = np.zeros(args.num_envs, dtype=np.float64)
    episode_alg_conn = np.zeros(args.num_envs, dtype=np.float64)
    episode_in_zone_steps = np.zeros(args.num_envs, dtype=np.float64)
    recent_returns = []
    recent_coverage = []
    recent_alg_conn = []
    recent_zone_ratios = []

    # ------------------------------------------------------------------
    # Graceful Ctrl+C
    # ------------------------------------------------------------------
    import signal as _signal
    _interrupted = [False]
    def _sigint_handler(sig, frame):
        print("\n\nCtrl+C received – finishing current update before saving…")
        _interrupted[0] = True
    _signal.signal(_signal.SIGINT, _sigint_handler)

    for update in range(1, num_updates + 1):
        # --- Learning rate annealing ---
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"]  = lrnow
            critic_optimizer.param_groups[0]["lr"] = lrnow * 3.0

        # ===============================================================
        # Rollout phase – collect experience
        # ===============================================================
        coverage_ratios = []
        algebraic_conns = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation

            # Build global state for the centralised critic
            gs = build_global_state(next_obs, args.num_agents, next_world_obs)
            global_states[step] = gs

            # Action selection (normalize obs for the network)
            with torch.no_grad():
                norm_obs = obs_rms.normalize(next_obs)
                norm_gs = global_obs_rms.normalize(gs)
                action, logprob, _, value = agent.get_action_and_value(norm_obs, norm_gs)
                # Critic outputs normalised values → denormalise for GAE
                values[step] = value_norm.denormalize(value).flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Environment step
            next_obs, reward, termination, truncation, info = envs.step(
                action.cpu().numpy()
            )
            reward_array = np.array(reward, dtype=np.float64)
            if reward_rms is not None:
                reward_rms.update(reward_array)
                reward_array = reward_rms.normalize(reward_array)
            rewards[step] = torch.tensor(reward_array, dtype=torch.float32).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_world_obs = extract_world_obs(info, args.num_envs, world_obs_dim, device)
            # Update observation normalizers with raw obs (actor + critic)
            obs_rms.update(next_obs.cpu().numpy())
            next_gs_unnorm = build_global_state(next_obs, args.num_agents, next_world_obs)
            global_obs_rms.update(next_gs_unnorm.cpu().numpy())
            next_termination = torch.Tensor(termination).to(device)
            next_truncation = torch.Tensor(truncation).to(device)

            # --- Track coverage ratios ---
            if isinstance(info, (list, tuple)):
                step_coverage_ratios = []
                for idx, item in enumerate(info):
                    if isinstance(item, dict) and 'coverage_ratio' in item:
                        step_coverage_ratios.append(item['coverage_ratio'])
                if step_coverage_ratios:
                    coverage_ratios.append(np.mean(step_coverage_ratios))
            
            # --- Track algebraic connectivity (per-step, smooth) ---
            if isinstance(info, (list, tuple)):
                step_alg_conns = []
                for idx, item in enumerate(info):
                    if isinstance(item, dict) and 'algebraic_connectivity' in item:
                        step_alg_conns.append(item['algebraic_connectivity'])
                if step_alg_conns:
                    avg_step_alg_conn = np.mean(step_alg_conns)
                    writer.add_scalar("charts/algebraic_connectivity", avg_step_alg_conn, global_step)
                    if args.track:
                        wandb.log({"charts/algebraic_connectivity": avg_step_alg_conn}, step=global_step)

            # --- Track per-agent GPSD zone presence ---
            if isinstance(info, (list, tuple)):
                for idx, item in enumerate(info):
                    if isinstance(item, dict) and 'in_gpsd_zone' in item:
                        episode_in_zone_steps[idx] += item['in_gpsd_zone']

            # --- Manual episode tracking ---
            episode_rewards += reward
            episode_lengths += 1

            if isinstance(info, (list, tuple)):
                for idx, item in enumerate(info):
                    if isinstance(item, dict) and "coverage_ratio" in item:
                        episode_coverage[idx] = item["coverage_ratio"]
            elif isinstance(info, dict) and "coverage_ratio" in info:
                episode_coverage[:] = info["coverage_ratio"]

            # Detect episode ends and log
            done_flags = np.maximum(termination, truncation)
            for idx in range(args.num_envs):
                if done_flags[idx]:
                    agent_idx = idx % args.num_agents
                    ep_ret = episode_rewards[idx]
                    ep_len = episode_lengths[idx]
                    ep_cov = episode_coverage[idx]

                    writer.add_scalar(
                        f"charts/episodic_return_agent{agent_idx}",
                        ep_ret, global_step,
                    )
                    writer.add_scalar(
                        f"charts/episodic_length_agent{agent_idx}",
                        ep_len, global_step,
                    )
                    writer.add_scalar(
                        f"charts/coverage_ratio_agent{agent_idx}",
                        ep_cov, global_step,
                    )
                    

                    # Log GPSD zone time ratio for this agent
                    zone_ratio = episode_in_zone_steps[idx] / max(ep_len, 1)
                    writer.add_scalar(
                        f"charts/gpsd_zone_ratio_agent{agent_idx}",
                        zone_ratio, global_step,
                    )
                    recent_zone_ratios.append(zone_ratio)

                    recent_returns.append(ep_ret)
                    recent_coverage.append(ep_cov)
                    
                    # Reset episodic buffers for this env
                    episode_rewards[idx] = 0.0
                    episode_lengths[idx] = 0
                    episode_coverage[idx] = 0.0
                    episode_in_zone_steps[idx] = 0.0
                    
                    # Clear step-wise buffers if at end of full environment step
                    # (only clear once for all agents in the same env)
                    if idx % args.num_agents == args.num_agents - 1:
                        algebraic_conns = []
                        coverage_ratios = []

            # Log rolling averages every 20 episodes
            if len(recent_returns) >= 20:
                avg_ret = np.mean(recent_returns[-100:])
                avg_cov = np.mean(recent_coverage[-100:])
                writer.add_scalar("charts/avg_episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/avg_coverage_ratio", avg_cov, global_step)
                
                avg_alg_conn = 0.0
                if recent_alg_conn:
                    avg_alg_conn = np.mean(recent_alg_conn[-100:])
                    writer.add_scalar("charts/avg_algebraic_connectivity", avg_alg_conn, global_step)

                avg_zone_ratio = 0.0
                if recent_zone_ratios:
                    avg_zone_ratio = np.mean(recent_zone_ratios[-100:])
                    writer.add_scalar("charts/avg_gpsd_zone_ratio", avg_zone_ratio, global_step)

                if args.track:
                    wandb.log({
                        "charts/avg_episodic_return": avg_ret,
                        "charts/avg_coverage_ratio": avg_cov,
                        "charts/avg_algebraic_connectivity": avg_alg_conn,
                        "charts/avg_gpsd_zone_ratio": avg_zone_ratio,
                    }, step=global_step)

        # ===============================================================
        # Advantage estimation (GAE) — critic uses global state
        # ===============================================================
        with torch.no_grad():
            next_gs = build_global_state(next_obs, args.num_agents, next_world_obs)
            norm_next_gs = global_obs_rms.normalize(next_gs)
            # Denormalise bootstrap value
            next_value = value_norm.denormalize(
                agent.get_value(norm_next_gs)
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            next_done = torch.maximum(next_termination, next_truncation)
            dones = torch.maximum(terminations, truncations)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # ===============================================================
        # Update value normaliser with the freshly-computed returns
        # ===============================================================
        value_norm.update(returns.cpu().numpy().flatten())

        # ===============================================================
        # Flatten the rollout batch
        # ===============================================================
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_global_states = global_states.reshape(-1, global_dim)
        # Pre-normalize the entire batch for the optimization phase
        b_obs_norm = obs_rms.normalize(b_obs)
        b_gs_norm = global_obs_rms.normalize(b_global_states)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Normalised return targets for the value function
        b_returns_norm = value_norm.normalize(b_returns)
        b_values_norm = value_norm.normalize(b_values)

        # ===============================================================
        # Optimisation phase – MAPPO update
        # ===============================================================
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Actor: local obs → new logprobs, entropy
                # Critic: global state → new value
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_norm[mb_inds],
                    b_gs_norm[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (in normalised space + Huber for robustness)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = nn.functional.huber_loss(
                        newvalue, b_returns_norm[mb_inds], reduction="none", delta=10.0
                    )
                    v_clipped = b_values_norm[mb_inds] + torch.clamp(
                        newvalue - b_values_norm[mb_inds],
                        -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = nn.functional.huber_loss(
                        v_clipped, b_returns_norm[mb_inds], reduction="none", delta=10.0
                    )
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = nn.functional.huber_loss(
                        newvalue, b_returns_norm[mb_inds], reduction="mean", delta=10.0
                    )

                entropy_loss = entropy.mean()
                # Actor loss (policy + entropy bonus)
                actor_loss = pg_loss - args.ent_coef * entropy_loss
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                # Compute actor and critic gradients together (one backward)
                total_loss = actor_loss + v_loss * args.vf_coef
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor_params_list,  args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic_params_list, args.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # ---------------------------------------------------------------
        # Extra critic-only epochs (critic trains independently of actor
        # KL budget, closing the gap that causes EV to plateau)
        # ---------------------------------------------------------------
        for _ce in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                newvalue = agent.get_value(b_gs_norm[mb_inds]).view(-1)
                # No clip_vloss in critic-only epochs — let it learn freely
                v_loss = nn.functional.huber_loss(
                    newvalue, b_returns_norm[mb_inds], reduction="mean", delta=10.0
                )
                critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params_list, args.max_grad_norm)
                critic_optimizer.step()

        # ===============================================================
        # Logging
        # ===============================================================
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if reward_rms is not None:
            raw_reward_mean = reward_rms.mean
            raw_reward_std = np.sqrt(reward_rms.var)
            norm_reward_mean = rewards.mean().item()
            norm_reward_std = rewards.std().item()
            writer.add_scalar("rewards/raw_mean", raw_reward_mean, global_step)
            writer.add_scalar("rewards/raw_std", raw_reward_std, global_step)
            writer.add_scalar("rewards/normalized_mean", norm_reward_mean, global_step)
            writer.add_scalar("rewards/normalized_std", norm_reward_std, global_step)
            if args.track:
                wandb.log({
                    "rewards/raw_mean": raw_reward_mean,
                    "rewards/raw_std": raw_reward_std,
                    "rewards/normalized_mean": norm_reward_mean,
                    "rewards/normalized_std": norm_reward_std,
                }, step=global_step)

        if coverage_ratios:
            avg_coverage_ratio = np.mean(coverage_ratios)
            writer.add_scalar("charts/avg_coverage_ratio", avg_coverage_ratio, global_step)

        batch_mean_reward = rewards.mean().item()
        writer.add_scalar("charts/batch_mean_reward", batch_mean_reward, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if args.track:
            log_dict = {
                "charts/learning_rate": actor_optimizer.param_groups[0]["lr"],
                "charts/SPS": sps,
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/batch_mean_reward": batch_mean_reward,
            }
            if recent_returns:
                log_dict["charts/avg_episodic_return"] = np.mean(recent_returns[-100:])
            if recent_coverage:
                log_dict["charts/avg_coverage_ratio"] = np.mean(recent_coverage[-100:])
            wandb.log(log_dict, step=global_step)

        if update % 10 == 0 or update == num_updates:
            avg_r = np.mean(recent_returns[-100:]) if recent_returns else float('nan')
            avg_c = np.mean(recent_coverage[-100:]) if recent_coverage else float('nan')
            print(
                f"  [Update {update:>4}/{num_updates}]  "
                f"step={global_step:>8}  "
                f"SPS={sps:>5}  "
                f"pg_loss={pg_loss.item():+.4f}  "
                f"v_loss={v_loss.item():.4f}  "
                f"entropy={entropy_loss.item():.4f}  "
                f"avg_return={avg_r:.2f}  "
                f"avg_coverage={avg_c:.3f}  "
                f"explained_var={explained_var:.3f}"
            )

        if _interrupted[0]:
            break

    # ====================================================================
    # Post-training (runs on normal completion AND after Ctrl+C)
    # ====================================================================
    _signal.signal(_signal.SIGINT, _signal.SIG_DFL)
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    if _interrupted[0]:
        print(f"  Training interrupted at step {global_step}.  Time: {elapsed:.1f}s")
    else:
        print(f"  Training complete!  Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Final SPS: {int(global_step / elapsed)}")
    print(f"{'='*70}")

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/gpsd_mappo_agent.pt"
        torch.save({
            "model_state_dict": agent.state_dict(),
            "actor_optimizer_state_dict":  actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "obs_rms": obs_rms,
            "args": vars(args),
            "global_step": global_step,
        }, model_path)
        print(f"  Model saved → {model_path}")

    if args.capture_video:
        record_video(args, agent, run_name)

    envs.close()
    writer.close()
    print("Done.")
