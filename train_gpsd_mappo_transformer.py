"""MAPPO-Transformer training script for the GPSD environment.

Implements Centralized Training with Decentralized Execution (CTDE)
with MLP feature extractor + Gated Causal Transformer:
  - obs → MLP(256,256) → stack last 16 frames → Gated Causal Transformer
    (1-layer, 4-head) → actor / critic heads.
  - Actor: uses only the agent's LOCAL observation history.
  - Critic: uses the GLOBAL state history.

Key design:
  - MLP extracts a 256-dim feature each step.
  - A rolling buffer of the last 16 features forms the sequence.
  - A 1-layer, 4-head causal Transformer with gated feed-forward (SwiGLU)
    processes the sequence.
  - The final position's output feeds the actor/critic heads.

Usage:
    python train_gpsd_mappo_transformer.py
    python train_gpsd_mappo_transformer.py --total-timesteps 500000
    python train_gpsd_mappo_transformer.py --track --wandb-project-name gpsd
"""

# flake8: noqa

import argparse
import math
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.mpe.gpsd.gpsd import parallel_env as make_gpsd_parallel_env


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="MAPPO-Transformer (CTDE) for GPSD environment")

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
    parser.add_argument("--local-ratio", type=float, default=0.999,
        help="weight for local vs global reward (0=fully global, 1=fully local)")

    # --- Transformer parameters ---
    parser.add_argument("--embed-dim", type=int, default=256,
        help="MLP feature / Transformer embedding dimension")
    parser.add_argument("--context-length", type=int, default=16,
        help="number of past frames in the Transformer context window")
    parser.add_argument("--n-heads", type=int, default=4,
        help="number of attention heads in the Transformer")

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
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=15,
        help="the K epochs to update the policy")
    parser.add_argument("--critic-epochs", type=int, default=5,
        help="extra critic-only epochs per update (on top of update-epochs)")
    parser.add_argument("--critic-lr-multiplier", type=float, default=5.0,
        help="critic LR = learning_rate * this multiplier (default: 5.0)")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.05,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="toggles whether to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.0,
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
# Value normalisation (PopArt-lite)
# ---------------------------------------------------------------------------
class ValueNorm:
    def __init__(self, clip=10.0, epsilon=1e-8):
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = epsilon
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x):
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
        return torch.clamp(
            (x - self.running_mean) / self.std, -self.clip, self.clip
        )

    def denormalize(self, x):
        return x * self.std + self.running_mean


# ---------------------------------------------------------------------------
# Observation normalisation (per-feature running mean/std)
# ---------------------------------------------------------------------------
class RunningMeanStdVec:
    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
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
# Gated Causal Transformer components
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU gating: split input in half, gate one half with SiLU of the other."""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        # Project to 2× hidden for gating
        self.w1 = layer_init(nn.Linear(in_features, hidden_features))
        self.w2 = layer_init(nn.Linear(in_features, hidden_features))
        self.w3 = layer_init(nn.Linear(hidden_features, in_features))

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GatedTransformerBlock(nn.Module):
    """Single Transformer block with causal attention and SwiGLU FFN.

    Uses gated residual connections: output = x + gate * sublayer(x)
    where gate is a learned scalar initialised near zero.
    """
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, embed_dim * 2)

        # Gated residual: learnable scalars initialised near 0
        self.gate_attn = nn.Parameter(torch.tensor(0.1))
        self.gate_ffn = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: (seq_len, seq_len) causal mask
            key_padding_mask: (batch, seq_len) True = ignore
        """
        # Self-attention with gated residual
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.gate_attn * attn_out

        # FFN with gated residual
        h = self.ln2(x)
        x = x + self.gate_ffn * self.ffn(h)

        return x


class TransformerMemory(nn.Module):
    """1-layer gated causal Transformer with learned positional embeddings."""

    def __init__(self, embed_dim, n_heads, context_length, dropout=0.0):
        super().__init__()
        self.context_length = context_length
        self.pos_embed = nn.Parameter(
            torch.randn(1, context_length, embed_dim) * 0.02
        )
        self.block = GatedTransformerBlock(embed_dim, n_heads, dropout)
        self.ln_out = nn.LayerNorm(embed_dim)

        # Pre-compute causal mask (upper-triangle = -inf)
        causal = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal)  # True = blocked

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim)  seq_len <= context_length
            key_padding_mask: (batch, seq_len)  True = padding (ignore)

        Returns:
            (batch, embed_dim) — output at the last valid position
        """
        B, T, D = x.shape
        x = x + self.pos_embed[:, :T, :]

        # Causal mask for this sequence length
        # nn.MultiheadAttention with batch_first=True expects (T,T) float mask
        # where -inf blocks attention
        causal = self.causal_mask[:T, :T].float() * (-1e9)

        x = self.block(x, attn_mask=causal, key_padding_mask=key_padding_mask)
        x = self.ln_out(x)

        # Take the last position output for each batch element
        if key_padding_mask is not None:
            # Find last non-padded position per batch
            # ~key_padding_mask: True = valid
            valid = ~key_padding_mask  # (B, T)
            # Index of last valid token
            last_idx = valid.long().sum(dim=1) - 1  # (B,)
            last_idx = last_idx.clamp(min=0)
            x = x[torch.arange(B, device=x.device), last_idx]
        else:
            x = x[:, -1]  # (B, D)

        return x


# ---------------------------------------------------------------------------
# MAPPO-Transformer Agent
# ---------------------------------------------------------------------------
class MAPPOTransformerAgent(nn.Module):
    """Actor and critic with MLP extractor → frame buffer → Gated Transformer."""

    def __init__(self, obs_dim, act_dim, num_agents, embed_dim=256,
                 context_length=16, n_heads=4, global_dim=None):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.context_length = context_length
        if global_dim is None:
            global_dim = obs_dim * num_agents

        # --- Actor: MLP feature extractor + Transformer ---
        self.actor_extractor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, embed_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.Tanh(),
        )
        self.actor_transformer = TransformerMemory(
            embed_dim, n_heads, context_length
        )
        self.actor_head = layer_init(nn.Linear(embed_dim, act_dim), std=0.01)

        # --- Critic: MLP feature extractor + Transformer ---
        self.critic_extractor = nn.Sequential(
            layer_init(nn.Linear(global_dim, embed_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.Tanh(),
        )
        self.critic_transformer = TransformerMemory(
            embed_dim, n_heads, context_length
        )
        self.critic_head = layer_init(nn.Linear(embed_dim, 1), std=0.01)

    def extract_actor_features(self, local_obs):
        """MLP feature extraction for actor. (batch, obs_dim) → (batch, embed_dim)"""
        return self.actor_extractor(local_obs)

    def extract_critic_features(self, global_state):
        """MLP feature extraction for critic. (batch, global_dim) → (batch, embed_dim)"""
        return self.critic_extractor(global_state)

    def get_action_from_seq(self, actor_seq, key_padding_mask=None, action=None):
        """
        Args:
            actor_seq: (batch, seq_len, embed_dim) — sequence of actor features
            key_padding_mask: (batch, seq_len) — True = padding
            action: optional action to evaluate

        Returns:
            action, log_prob, entropy
        """
        out = self.actor_transformer(actor_seq, key_padding_mask)  # (batch, embed_dim)
        logits = self.actor_head(out)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value_from_seq(self, critic_seq, key_padding_mask=None):
        """
        Args:
            critic_seq: (batch, seq_len, embed_dim) — sequence of critic features
            key_padding_mask: (batch, seq_len) — True = padding

        Returns:
            value: (batch, 1)
        """
        out = self.critic_transformer(critic_seq, key_padding_mask)  # (batch, embed_dim)
        return self.critic_head(out)


# ---------------------------------------------------------------------------
# Environment factory
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
# Helper: build global state from vectorised observations
# ---------------------------------------------------------------------------
def extract_world_obs(info, num_envs, world_obs_dim, device):
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
    total = local_obs.shape[0]
    obs_dim = local_obs.shape[1]
    num_games = total // num_agents
    reshaped = local_obs.view(num_games, num_agents, obs_dim)
    concat = reshaped.reshape(num_games, num_agents * obs_dim)
    if world_obs is not None:
        wo_per_game = world_obs.view(num_games, num_agents, -1)[:, 0, :]
        concat = torch.cat([concat, wo_per_game], dim=-1)
    global_dim = concat.shape[-1]
    global_state = concat.unsqueeze(1).expand(num_games, num_agents, global_dim)
    return global_state.reshape(total, global_dim)


# ---------------------------------------------------------------------------
# Video recording helper
# ---------------------------------------------------------------------------
def record_video(args, agent_model, run_name, obs_rms, num_episodes=1):
    try:
        import imageio
    except ImportError:
        print("imageio not installed – skipping video recording.")
        return

    print("Recording video of trained policy (Transformer execution)...")
    device = next(agent_model.parameters()).device
    C = args.context_length
    D = args.embed_dim

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
        obs_dict, infos = env.reset()
        # Rolling actor feature buffer: (num_agents, context_length, embed_dim)
        actor_buffer = torch.zeros(args.num_agents, C, D).to(device)
        buf_len = torch.zeros(args.num_agents, dtype=torch.long).to(device)

        for _ in range(args.max_cycles):
            with torch.no_grad():
                obs_array = np.stack([obs_dict[a] for a in env.agents])
                obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)
                norm_obs = obs_rms.normalize(obs_tensor)

                # Extract features
                feats = agent_model.extract_actor_features(norm_obs)  # (N, D)

                # Shift buffer and append
                actor_buffer = torch.roll(actor_buffer, -1, dims=1)
                actor_buffer[:, -1, :] = feats
                buf_len = (buf_len + 1).clamp(max=C)

                # Build padding mask: True = padding
                pad_mask = torch.zeros(args.num_agents, C, dtype=torch.bool, device=device)
                for i in range(args.num_agents):
                    if buf_len[i] < C:
                        pad_mask[i, :C - buf_len[i]] = True

                actions, _, _ = agent_model.get_action_from_seq(
                    actor_buffer, key_padding_mask=pad_mask
                )

            action_dict = {
                agent_name: actions[i].item()
                for i, agent_name in enumerate(env.agents)
            }
            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)

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
    run_name = f"mappo_tfm_{args.seed}_{int(time.time())}"
    print(f"\n{'='*70}")
    print(f"  GPSD MAPPO-Transformer (CTDE) Training  —  {run_name}")
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
    n_cells = int(np.round(1.0 / args.cell_width))
    n_pois = n_cells * n_cells
    world_obs_dim = args.num_agents * 6 + n_pois * 3
    global_dim = obs_dim * args.num_agents + world_obs_dim
    C = args.context_length
    D = args.embed_dim

    print(f"Observation dim (local)  : {obs_dim}")
    print(f"World obs dim            : {world_obs_dim}")
    print(f"Global state dim (critic): {global_dim}")
    print(f"Action dim               : {act_dim}")
    print(f"Transformer embed dim    : {D}")
    print(f"Context length           : {C}")
    print(f"Attention heads          : {args.n_heads}")
    print(f"Num vec envs             : {args.num_envs}  "
          f"({args.num_envs // args.num_agents} games × {args.num_agents} agents)")
    print(f"Batch size               : {args.batch_size}")
    print()

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # ------------------------------------------------------------------
    # Agent & optimisers
    # ------------------------------------------------------------------
    agent = MAPPOTransformerAgent(
        obs_dim, act_dim, args.num_agents,
        embed_dim=D, context_length=C, n_heads=args.n_heads,
        global_dim=global_dim,
    ).to(device)

    actor_params_list = (
        list(agent.actor_extractor.parameters()) +
        list(agent.actor_transformer.parameters()) +
        list(agent.actor_head.parameters())
    )
    critic_params_list = (
        list(agent.critic_extractor.parameters()) +
        list(agent.critic_transformer.parameters()) +
        list(agent.critic_head.parameters())
    )
    actor_optimizer  = optim.Adam(actor_params_list,  lr=args.learning_rate,       eps=1e-5)
    critic_optimizer = optim.Adam(critic_params_list, lr=args.learning_rate * args.critic_lr_multiplier, eps=1e-5)

    actor_params = sum(p.numel() for p in actor_params_list)
    critic_params = sum(p.numel() for p in critic_params_list)
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

    # --- Rolling feature buffers for rollout (actor & critic) ---
    actor_buffer = torch.zeros(args.num_envs, C, D).to(device)
    critic_buffer = torch.zeros(args.num_envs, C, D).to(device)
    buf_len = torch.zeros(args.num_envs, dtype=torch.long).to(device)

    # --- Normalization ---
    reward_rms = RunningMeanStd() if args.norm_reward else None
    value_norm = ValueNorm()
    obs_rms = RunningMeanStdVec((obs_dim,))
    global_obs_rms = RunningMeanStdVec((global_dim,))

    # --- Manual episode tracking ---
    episode_rewards = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_coverage = np.zeros(args.num_envs, dtype=np.float64)
    episode_in_zone_steps = np.zeros(args.num_envs, dtype=np.float64)
    recent_returns = []
    recent_coverage = []
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
            critic_optimizer.param_groups[0]["lr"] = lrnow * args.critic_lr_multiplier

        # ===============================================================
        # Rollout phase – collect experience
        # ===============================================================
        coverage_ratios = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation

            # Reset buffers at episode boundaries
            next_done = torch.maximum(next_termination, next_truncation)
            done_mask = next_done.bool()
            if done_mask.any():
                actor_buffer[done_mask] = 0.0
                critic_buffer[done_mask] = 0.0
                buf_len[done_mask] = 0

            # Build global state
            gs = build_global_state(next_obs, args.num_agents, next_world_obs)
            global_states[step] = gs

            with torch.no_grad():
                norm_obs = obs_rms.normalize(next_obs)
                norm_gs = global_obs_rms.normalize(gs)

                # Extract features
                actor_feats = agent.extract_actor_features(norm_obs)   # (E, D)
                critic_feats = agent.extract_critic_features(norm_gs)  # (E, D)

                # Shift buffers and append new features
                actor_buffer = torch.roll(actor_buffer, -1, dims=1)
                actor_buffer[:, -1, :] = actor_feats
                critic_buffer = torch.roll(critic_buffer, -1, dims=1)
                critic_buffer[:, -1, :] = critic_feats
                buf_len = (buf_len + 1).clamp(max=C)

                # Build padding masks: True = padding (should be ignored)
                positions = torch.arange(C, device=device).unsqueeze(0)  # (1, C)
                pad_start = C - buf_len.unsqueeze(1)  # (E, 1)
                pad_mask = positions < pad_start  # (E, C)

                # Get action and value
                action, logprob, _ = agent.get_action_from_seq(
                    actor_buffer, key_padding_mask=pad_mask
                )
                value = agent.get_value_from_seq(
                    critic_buffer, key_padding_mask=pad_mask
                )
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

            # --- Track algebraic connectivity (per-step) ---
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

                    writer.add_scalar(f"charts/episodic_return_agent{agent_idx}", ep_ret, global_step)
                    writer.add_scalar(f"charts/episodic_length_agent{agent_idx}", ep_len, global_step)
                    writer.add_scalar(f"charts/coverage_ratio_agent{agent_idx}", ep_cov, global_step)

                    zone_ratio = episode_in_zone_steps[idx] / max(ep_len, 1)
                    writer.add_scalar(f"charts/gpsd_zone_ratio_agent{agent_idx}", zone_ratio, global_step)
                    recent_zone_ratios.append(zone_ratio)

                    recent_returns.append(ep_ret)
                    recent_coverage.append(ep_cov)

                    episode_rewards[idx] = 0.0
                    episode_lengths[idx] = 0
                    episode_coverage[idx] = 0.0
                    episode_in_zone_steps[idx] = 0.0

            # Log rolling averages every 20 episodes
            if len(recent_returns) >= 20:
                avg_ret = np.mean(recent_returns[-100:])
                avg_cov = np.mean(recent_coverage[-100:])
                writer.add_scalar("charts/avg_episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/avg_coverage_ratio", avg_cov, global_step)

                avg_zone_ratio = 0.0
                if recent_zone_ratios:
                    avg_zone_ratio = np.mean(recent_zone_ratios[-100:])
                    writer.add_scalar("charts/avg_gpsd_zone_ratio", avg_zone_ratio, global_step)

                if args.track:
                    wandb.log({
                        "charts/avg_episodic_return": avg_ret,
                        "charts/avg_coverage_ratio": avg_cov,
                        "charts/avg_gpsd_zone_ratio": avg_zone_ratio,
                    }, step=global_step)

        # ===============================================================
        # Advantage estimation (GAE) — Transformer critic
        # ===============================================================
        with torch.no_grad():
            # Bootstrap value from current state + buffer
            next_gs = build_global_state(next_obs, args.num_agents, next_world_obs)
            norm_next_gs = global_obs_rms.normalize(next_gs)
            critic_feats_boot = agent.extract_critic_features(norm_next_gs)
            # Shift buffer for bootstrap
            boot_buffer = torch.roll(critic_buffer.clone(), -1, dims=1)
            boot_buffer[:, -1, :] = critic_feats_boot
            boot_len = (buf_len + 1).clamp(max=C)
            positions = torch.arange(C, device=device).unsqueeze(0)
            pad_start = C - boot_len.unsqueeze(1)
            boot_pad_mask = positions < pad_start
            next_value = value_norm.denormalize(
                agent.get_value_from_seq(boot_buffer, key_padding_mask=boot_pad_mask)
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
        # Update value normaliser
        # ===============================================================
        value_norm.update(returns.cpu().numpy().flatten())

        # ===============================================================
        # Optimisation phase — reconstruct sequences from stored obs
        # ===============================================================
        # Compute all MLP features for the full rollout in one batch
        T = args.num_steps
        E = args.num_envs

        all_obs_norm = obs_rms.normalize(obs.reshape(-1, obs_dim)).reshape(T, E, obs_dim)
        all_gs_norm = global_obs_rms.normalize(global_states.reshape(-1, global_dim)).reshape(T, E, global_dim)

        with torch.no_grad():
            all_actor_feats = agent.extract_actor_features(
                all_obs_norm.reshape(-1, obs_dim)
            ).reshape(T, E, D)
            all_critic_feats = agent.extract_critic_features(
                all_gs_norm.reshape(-1, global_dim)
            ).reshape(T, E, D)

        # Build episode masks from dones to know where to reset context
        all_dones = torch.maximum(terminations, truncations)  # (T, E)

        # For each timestep t, we need to build a context window of up to C
        # features, respecting episode boundaries (no looking past a done).
        # Pre-compute context windows: (T, E, C, D) and padding masks: (T, E, C)
        # Pad beginning of rollout with zeros
        padded_actor = torch.zeros(C - 1 + T, E, D, device=device)
        padded_actor[C - 1:] = all_actor_feats
        padded_critic = torch.zeros(C - 1 + T, E, D, device=device)
        padded_critic[C - 1:] = all_critic_feats
        padded_dones = torch.ones(C - 1 + T, E, device=device)  # 1 = done (padding)
        padded_dones[C - 1:] = all_dones

        # Extract sliding windows
        actor_windows = torch.stack(
            [padded_actor[t:t + C] for t in range(T)], dim=0
        )  # (T, C, E, D)
        critic_windows = torch.stack(
            [padded_critic[t:t + C] for t in range(T)], dim=0
        )  # (T, C, E, D)
        done_windows = torch.stack(
            [padded_dones[t:t + C] for t in range(T)], dim=0
        )  # (T, C, E)

        # Transpose to (T, E, C, D) for batch processing
        actor_windows = actor_windows.permute(0, 2, 1, 3)   # (T, E, C, D)
        critic_windows = critic_windows.permute(0, 2, 1, 3) # (T, E, C, D)
        done_windows = done_windows.permute(0, 2, 1)         # (T, E, C)

        # Create padding masks from done_windows:
        # For each (t, e), find the latest done in the context before the
        # current position. Everything before that done is "padding".
        # A simple approach: scan backwards from position C-2 and mark
        # everything at or before the last done as padding.
        pad_masks = torch.zeros(T, E, C, dtype=torch.bool, device=device)
        for c_idx in range(C - 2, -1, -1):
            # If there was a done at position c_idx, everything at
            # positions <= c_idx should be masked (it's from a prev episode)
            # done_windows[:, :, c_idx] is 1 where done occurred
            is_done = done_windows[:, :, c_idx].bool()  # (T, E)
            pad_masks[:, :, :c_idx + 1] |= is_done.unsqueeze(-1)

        # Flatten for minibatching: (T*E, C, D) and (T*E, C)
        flat_actor_windows = actor_windows.reshape(T * E, C, D)
        flat_critic_windows = critic_windows.reshape(T * E, C, D)
        flat_pad_masks = pad_masks.reshape(T * E, C)
        flat_logprobs = logprobs.reshape(-1)
        flat_actions = actions.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = values.reshape(-1)
        flat_returns_norm = value_norm.normalize(flat_returns)
        flat_values_norm = value_norm.normalize(flat_values)

        b_inds = np.arange(T * E)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, T * E, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Re-extract features (with gradients this time)
                mb_obs = all_obs_norm.reshape(-1, obs_dim)[mb_inds]
                mb_gs = all_gs_norm.reshape(-1, global_dim)[mb_inds]
                mb_actor_feats_new = agent.extract_actor_features(mb_obs)   # (mb, D)
                mb_critic_feats_new = agent.extract_critic_features(mb_gs)  # (mb, D)

                # Replace the last position in the context window with
                # the freshly-computed (gradient-carrying) features
                mb_actor_ctx = flat_actor_windows[mb_inds].clone()
                mb_actor_ctx[:, -1, :] = mb_actor_feats_new
                mb_critic_ctx = flat_critic_windows[mb_inds].clone()
                mb_critic_ctx[:, -1, :] = mb_critic_feats_new

                mb_pad = flat_pad_masks[mb_inds]
                mb_acts = flat_actions[mb_inds].long()

                # Actor forward
                _, newlogprob, entropy = agent.get_action_from_seq(
                    mb_actor_ctx, key_padding_mask=mb_pad, action=mb_acts
                )
                # Critic forward
                newvalue = agent.get_value_from_seq(
                    mb_critic_ctx, key_padding_mask=mb_pad
                ).view(-1)

                logratio = newlogprob - flat_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = flat_advantages[mb_inds]
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

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = F.huber_loss(
                        newvalue, flat_returns_norm[mb_inds], reduction="none", delta=10.0
                    )
                    v_clipped = flat_values_norm[mb_inds] + torch.clamp(
                        newvalue - flat_values_norm[mb_inds],
                        -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = F.huber_loss(
                        v_clipped, flat_returns_norm[mb_inds], reduction="none", delta=10.0
                    )
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = F.huber_loss(
                        newvalue, flat_returns_norm[mb_inds], reduction="mean", delta=10.0
                    )

                entropy_loss = entropy.mean()
                actor_loss = pg_loss - args.ent_coef * entropy_loss
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
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
        # Extra critic-only epochs
        # ---------------------------------------------------------------
        for _ce in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, T * E, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_gs = all_gs_norm.reshape(-1, global_dim)[mb_inds]
                mb_critic_feats_new = agent.extract_critic_features(mb_gs)
                mb_critic_ctx = flat_critic_windows[mb_inds].clone()
                mb_critic_ctx[:, -1, :] = mb_critic_feats_new
                mb_pad = flat_pad_masks[mb_inds]

                newvalue = agent.get_value_from_seq(
                    mb_critic_ctx, key_padding_mask=mb_pad
                ).view(-1)
                v_loss = F.huber_loss(
                    newvalue, flat_returns_norm[mb_inds], reduction="mean", delta=10.0
                )
                critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params_list, args.max_grad_norm)
                critic_optimizer.step()

        # ===============================================================
        # Logging
        # ===============================================================
        y_pred = flat_values.cpu().numpy()
        y_true = flat_returns.cpu().numpy()
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
    # Post-training
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
        model_path = f"runs/{run_name}/gpsd_mappo_transformer_agent.pt"
        torch.save({
            "model_state_dict": agent.state_dict(),
            "actor_optimizer_state_dict":  actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        }, model_path)
        print(f"  Model saved → {model_path}")

    if args.capture_video:
        record_video(args, agent, run_name, obs_rms)

    envs.close()
    writer.close()
    print("Done.")
