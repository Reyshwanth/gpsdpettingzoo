"""MAPPO + GAT Actor + Transformer Critic for GPSD-Conn environment.

Architecture overview:
  Actor  —  Graph Attention Network (GAT) over local neighbourhood
            + MLP head with tanh squashing for smooth actions.
  Critic —  Transformer self-attention over all agents' states
            + Fiedler value / adjacency features (centralised).

Additional training signals:
  - Contrastive auxiliary loss on Actor hidden states (distance prediction).
  - Connectivity reward shaping from gpsd_conn.py.
  - Curriculum learning: connectivity penalty weight ramps up over training.

No external GNN library required — GAT is implemented in pure PyTorch.

Usage:
    conda run -n gpsd python train_gpsd_gat.py
    conda run -n gpsd python train_gpsd_gat.py --total-timesteps 5000000
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

from pettingzoo.mpe.gpsd.gpsd_conn import parallel_env as make_gpsd_parallel_env


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(
        description="MAPPO + GAT Actor + Transformer Critic for GPSD"
    )

    # --- General ---
    parser.add_argument("--exp-name", type=str,
        default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="gpsd-gat")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)

    # --- GPSD environment ---
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--cell-width", type=float, default=0.25)
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--r-c", type=float, default=0.3)
    parser.add_argument("--cov-c", type=float, default=0.5)

    # --- PPO hyperparameters ---
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--critic-epochs", type=int, default=5)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)),
        default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)),
        default=False, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)

    # --- Connectivity / curriculum ---
    parser.add_argument("--conn-penalty-start", type=float, default=5.0,
        help="initial connectivity penalty weight (curriculum start)")
    parser.add_argument("--conn-penalty-end", type=float, default=1.0,
        help="final connectivity penalty weight (curriculum end)")
    parser.add_argument("--buffer-eps", type=float, default=0.05,
        help="buffer zone ε for R_spatial")
    parser.add_argument("--contrastive-coef", type=float, default=0.1,
        help="weight of the contrastive distance-prediction auxiliary loss")

    # --- Architecture ---
    parser.add_argument("--gat-heads", type=int, default=4,
        help="number of attention heads in the GAT actor")
    parser.add_argument("--gat-hidden", type=int, default=64,
        help="hidden dim per head in GAT")
    parser.add_argument("--transformer-heads", type=int, default=4,
        help="number of attention heads in transformer critic")
    parser.add_argument("--transformer-layers", type=int, default=2,
        help="number of transformer encoder layers in critic")
    parser.add_argument("--embed-dim", type=int, default=128,
        help="embedding dimension for both actor and critic")

    args = parser.parse_args()

    assert args.num_envs % args.num_agents == 0
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# ======================================================================
# Utility: observation parsing
# ======================================================================
# Obs layout (57-dim for 5 agents, 16 POIs):
#   heading(1) | belief_pos(2) | cov_trace(1) | in_gpsd(1)
#   | poi_rel(N_r*2) | other_pos((N_a-1)*2) | other_comm((N_a-1)*2)
#   | other_heading((N_a-1)*1)

def parse_obs_layout(obs_dim, num_agents, num_pois):
    """Return slice indices into a flat observation vector."""
    idx = 0
    slices = {}
    slices["heading"] = (idx, idx + 1); idx += 1
    slices["pos"] = (idx, idx + 2); idx += 2
    slices["cov"] = (idx, idx + 1); idx += 1
    slices["in_gpsd"] = (idx, idx + 1); idx += 1
    slices["poi_rel"] = (idx, idx + num_pois * 2); idx += num_pois * 2
    n_other = num_agents - 1
    slices["other_pos"] = (idx, idx + n_other * 2); idx += n_other * 2
    slices["other_comm"] = (idx, idx + n_other * 2); idx += n_other * 2
    slices["other_heading"] = (idx, idx + n_other); idx += n_other
    assert idx == obs_dim, f"obs_dim mismatch: parsed {idx}, expected {obs_dim}"
    return slices


# ======================================================================
# Running statistics utilities
# ======================================================================

class RunningMeanStdVec:
    """Per-feature running mean/var (Welford)."""
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
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + np.square(delta) * self.count * batch_count / tot_count) / tot_count
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std_t = torch.sqrt(torch.tensor(self.var, dtype=torch.float32, device=x.device)) + 1e-8
        return torch.clamp((x - mean_t) / std_t, -clip, clip)


class ValueNorm:
    """PopArt-lite value normalisation."""
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
        self.running_mean += delta * batch_count / tot_count
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
        return torch.clamp((x - self.running_mean) / self.std, -self.clip, self.clip)

    def denormalize(self, x):
        return x * self.std + self.running_mean


# ======================================================================
# Network helpers
# ======================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ======================================================================
# GAT layer  (pure PyTorch, no PyG)
# ======================================================================

class GATLayer(nn.Module):
    """Single multi-head Graph Attention layer.

    Operates on a *batch* of graphs that all have the same number of nodes
    (num_agents), which is the case for our vectorised environment.

    Input:  node features  (batch, N, F_in)
            adjacency mask (batch, N, N)  — 1 = edge, 0 = no edge
    Output: node features  (batch, N, heads * F_out)
    """

    def __init__(self, in_features, out_features, heads=4, dropout=0.0,
                 concat=True):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(heads, in_features, out_features))
        self.a_src = nn.Parameter(torch.empty(heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(heads, out_features, 1))
        self.bias = nn.Parameter(torch.zeros(heads, out_features))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        """
        Args:
            x:   (B, N, F_in)
            adj: (B, N, N) — binary adjacency, 1 = connected

        Returns:
            out: (B, N, heads*out_features) if concat else (B, N, out_features)
        """
        B, N, F_in = x.shape
        H = self.heads
        F_out = self.out_features

        # Linear transform per head: (B, N, F_in) @ (H, F_in, F_out) → (B, H, N, F_out)
        h = torch.einsum("bnf,hfo->bhno", x, self.W)  # (B, H, N, F_out)

        # Attention coefficients
        attn_src = torch.einsum("bhno,hoi->bhni", h, self.a_src).squeeze(-1)  # (B,H,N)
        attn_dst = torch.einsum("bhno,hoi->bhni", h, self.a_dst).squeeze(-1)  # (B,H,N)
        # (B,H,N,1) + (B,H,1,N) → (B,H,N,N)
        attn = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2)
        attn = self.leaky_relu(attn)

        # Mask: only attend to connected neighbours (+ self)
        # adj: (B, N, N) → (B, 1, N, N)
        self_loop = torch.eye(N, device=adj.device).unsqueeze(0)  # (1, N, N)
        mask = (adj + self_loop).clamp(max=1.0).unsqueeze(1)  # (B, 1, N, N)
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # Handle NaN from all-masked rows (isolated nodes)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        # Aggregate: (B,H,N,N) @ (B,H,N,F_out) → (B,H,N,F_out)
        out = torch.matmul(attn, h) + self.bias.unsqueeze(0).unsqueeze(2)

        if self.concat:
            # (B, N, H*F_out)
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * F_out)
        else:
            out = out.mean(dim=1)  # (B, N, F_out)
        return out


# ======================================================================
# GAT Actor
# ======================================================================

class GATActor(nn.Module):
    """Decentralised actor using Graph Attention over local neighbours.

    1. Parse observation → self-features + neighbour features.
    2. Build local adjacency from neighbour distances.
    3. Run 2-layer GAT over the local subgraph.
    4. MLP policy head with tanh squashing on logits for smooth actions.
    5. Auxiliary distance-prediction head for contrastive learning.
    """

    def __init__(self, obs_dim, act_dim, num_agents, num_pois,
                 embed_dim=128, gat_heads=4, gat_hidden=64, r_c=0.3):
        super().__init__()
        self.num_agents = num_agents
        self.num_pois = num_pois
        self.r_c = r_c
        self.obs_dim = obs_dim
        self.slices = parse_obs_layout(obs_dim, num_agents, num_pois)
        n_other = num_agents - 1

        # --- Feature encoder for self-node ---
        # self features: heading(1) + pos(2) + cov(1) + in_gpsd(1) = 5
        self_feat_dim = 5
        self.self_encoder = nn.Sequential(
            layer_init(nn.Linear(self_feat_dim, embed_dim)),
            nn.ReLU(),
        )

        # --- Feature encoder for neighbour nodes ---
        # Per-neighbour: rel_pos(2) + range(1) + cov_trace(1) + rel_heading(1) = 5
        nbr_feat_dim = 5
        self.nbr_encoder = nn.Sequential(
            layer_init(nn.Linear(nbr_feat_dim, embed_dim)),
            nn.ReLU(),
        )

        # --- GAT layers ---
        # Layer 1: embed_dim → gat_hidden * gat_heads  (concat)
        self.gat1 = GATLayer(embed_dim, gat_hidden, heads=gat_heads, concat=True)
        self.gat1_norm = nn.LayerNorm(gat_heads * gat_hidden)
        # Layer 2: gat_heads*gat_hidden → embed_dim  (mean over heads)
        self.gat2 = GATLayer(gat_heads * gat_hidden, embed_dim,
                             heads=gat_heads, concat=False)
        self.gat2_norm = nn.LayerNorm(embed_dim)

        # --- POI feature encoder ---
        # Encode POI relative positions into a fixed-size summary
        self.poi_encoder = nn.Sequential(
            layer_init(nn.Linear(num_pois * 2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, embed_dim)),
            nn.ReLU(),
        )

        # --- Policy head ---
        combined_dim = embed_dim + embed_dim  # GAT self-node + POI summary
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
            nn.Tanh(),  # smooth squashing
        )
        # Learnable temperature for logits
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

        # --- Contrastive distance-prediction head ---
        # Predicts distance to each neighbour from the self-node's GAT embedding
        self.dist_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, n_other), std=0.1),
        )

    def _parse_and_build_graph(self, obs):
        """Parse vectorised obs → node features + adjacency.

        Args:
            obs: (B*N_a, obs_dim) flat observations from the vectorised env.
                 B*N_a must be divisible by num_agents so we can group agents
                 into games.

        Returns:
            self_feat:  (B_games, N_a, embed_dim)
            adj:        (B_games, N_a, embed_dim)  — for the GAT
            poi_feat:   (B_games * N_a, embed_dim)
            true_dists: (B_games * N_a, N_a-1) — ground-truth neighbour distances
        """
        s = self.slices
        total = obs.shape[0]  # B_total = num_games * num_agents
        N_a = self.num_agents
        assert total % N_a == 0
        B = total // N_a

        # --- Extract per-agent features ---
        heading   = obs[:, s["heading"][0]:s["heading"][1]]       # (T, 1)
        pos       = obs[:, s["pos"][0]:s["pos"][1]]               # (T, 2)
        cov       = obs[:, s["cov"][0]:s["cov"][1]]               # (T, 1)
        in_gpsd   = obs[:, s["in_gpsd"][0]:s["in_gpsd"][1]]       # (T, 1)
        poi_rel   = obs[:, s["poi_rel"][0]:s["poi_rel"][1]]       # (T, N_r*2)
        other_pos = obs[:, s["other_pos"][0]:s["other_pos"][1]]   # (T, (N_a-1)*2)
        other_comm = obs[:, s["other_comm"][0]:s["other_comm"][1]] # (T, (N_a-1)*2)
        other_hdg = obs[:, s["other_heading"][0]:s["other_heading"][1]]  # (T, N_a-1)

        # Self node features: [heading, pos_x, pos_y, cov, in_gpsd]
        self_raw = torch.cat([heading, pos, cov, in_gpsd], dim=-1)  # (T, 5)

        # Neighbour features: for each of (N_a-1) neighbours,
        # [rel_x, rel_y, range, cov_trace, rel_heading]
        n_other = N_a - 1
        oth_pos_2d = other_pos.view(total, n_other, 2)
        oth_comm_2d = other_comm.view(total, n_other, 2)  # [range, cov]
        oth_hdg_2d = other_hdg.view(total, n_other, 1)

        # Zero out sentinel values for out-of-range neighbours
        # (sentinel = -1.0 for all fields when out of obs range 10*r_c)
        range_vals = oth_comm_2d[:, :, 0:1]  # (T, n_other, 1)
        out_of_obs = (range_vals < 0)  # -1 sentinel
        oth_pos_clean = oth_pos_2d.clone()
        oth_pos_clean[out_of_obs.expand_as(oth_pos_clean)] = 0.0
        oth_comm_clean = oth_comm_2d.clone()
        oth_comm_clean[out_of_obs.expand_as(oth_comm_clean)] = 0.0
        oth_hdg_clean = oth_hdg_2d.clone()
        oth_hdg_clean[out_of_obs.squeeze(-1)] = 0.0

        nbr_raw = torch.cat([
            oth_pos_clean,                       # (T, n_other, 2)
            oth_comm_clean[:, :, 0:1],           # range (T, n_other, 1)
            oth_comm_clean[:, :, 1:2],           # cov   (T, n_other, 1)
            oth_hdg_clean,                       # hdg   (T, n_other, 1)
        ], dim=-1)  # (T, n_other, 5)

        # Ground-truth distances (from range measurements; used for contrastive)
        true_dists = oth_comm_2d[:, :, 0]  # (T, n_other), -1 = out of range

        # --- Encode ---
        self_emb = self.self_encoder(self_raw)  # (T, embed_dim)
        nbr_flat = nbr_raw.view(total * n_other, -1)
        nbr_emb = self.nbr_encoder(nbr_flat).view(total, n_other, -1)  # (T, n_other, E)

        # --- Build per-game graph ---
        # Reshape to games: (B, N_a, embed_dim)
        self_emb_g = self_emb.view(B, N_a, -1)

        # Build adjacency from true_dists across agents in the same game
        # We need the *actual* positions to compute adjacency, but we can
        # approximate from the obs: agent i sees neighbours j's range.
        # For the GAT we use the inter-agent data available per-game.
        #
        # Since each agent's obs contains ranges to all other agents,
        # we reconstruct the adjacency per game.
        dists_g = true_dists.view(B, N_a, n_other)  # (B, N_a, N_a-1)

        # Build (B, N_a, N_a) adjacency
        # IMPORTANT: connectivity = within communication radius r_c,
        # NOT observation range (10*r_c).  range_meas > 0 only means
        # the agent is visible (within 10*r_c).  True connectivity
        # requires range_meas <= r_c.
        adj = torch.zeros(B, N_a, N_a, device=obs.device)
        for i in range(N_a):
            k = 0
            for j in range(N_a):
                if j == i:
                    continue
                range_ij = dists_g[:, i, k]
                connected = ((range_ij > 0) & (range_ij <= self.r_c)).float()
                adj[:, i, j] = connected
                k += 1

        # POI embedding
        poi_emb = self.poi_encoder(poi_rel)  # (T, embed_dim)

        return self_emb_g, adj, poi_emb, true_dists

    def forward(self, obs, action=None):
        """Full actor forward.

        Args:
            obs:    (T, obs_dim)   — T = num_games * num_agents
            action: (T,) optional  — if given, compute log_prob of this action

        Returns:
            action:      (T,)
            log_prob:    (T,)
            entropy:     (T,)
            actor_embed: (T, embed_dim) — self-node GAT embedding, for contrastive loss
            true_dists:  (T, N_a-1)     — ground truth distances
        """
        N_a = self.num_agents
        T = obs.shape[0]
        B = T // N_a

        self_emb_g, adj, poi_emb, true_dists = self._parse_and_build_graph(obs)
        # self_emb_g: (B, N_a, E),  adj: (B, N_a, N_a)

        # --- GAT message passing ---
        h = F.relu(self.gat1_norm(self.gat1(self_emb_g, adj)))  # (B, N_a, H*F)
        h = F.relu(self.gat2_norm(self.gat2(h, adj)))           # (B, N_a, E)

        # Flatten back to (T, E)
        actor_embed = h.view(T, -1)

        # Combine with POI summary
        combined = torch.cat([actor_embed, poi_emb], dim=-1)  # (T, 2E)

        # Policy head: tanh squashes logits, scale controls sharpness
        raw_logits = self.policy_head(combined)  # (T, act_dim)  post-tanh
        logits = raw_logits * self.logit_scale

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, actor_embed, true_dists

    def contrastive_loss(self, actor_embed, true_dists):
        """Auxiliary contrastive loss: predict neighbour distances from GAT embedding.

        Args:
            actor_embed: (T, embed_dim)
            true_dists:  (T, N_a-1)  — observed range measurements, -1 = out of range

        Returns:
            loss: scalar
        """
        pred_dists = self.dist_head(actor_embed.detach())  # (T, N_a-1)
        # Only supervise on valid measurements (range > 0)
        valid = (true_dists > 0).float()
        if valid.sum() < 1:
            return torch.tensor(0.0, device=actor_embed.device)
        sq_err = (pred_dists - true_dists) ** 2 * valid
        return sq_err.sum() / (valid.sum() + 1e-8)


# ======================================================================
# Transformer Critic
# ======================================================================

class TransformerCritic(nn.Module):
    """Centralised critic using Transformer self-attention over all agents.

    Input: per-agent state features + adjacency + Fiedler value.
    Processes all agents' features via self-attention, then pools to a
    single scalar value per game (broadcast to all agents).
    """

    def __init__(self, obs_dim, num_agents, num_pois,
                 embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.slices = parse_obs_layout(obs_dim, num_agents, num_pois)

        # Per-agent feature encoder
        # Each agent: heading(1) + pos(2) + cov(1) + in_gpsd(1) = 5
        self.agent_encoder = nn.Sequential(
            layer_init(nn.Linear(5, embed_dim)),
            nn.ReLU(),
        )

        # POI encoder per agent
        self.poi_encoder = nn.Sequential(
            layer_init(nn.Linear(num_pois * 2, embed_dim)),
            nn.ReLU(),
        )

        # Adjacency row encoder (N_a values → embed)
        self.adj_encoder = nn.Sequential(
            layer_init(nn.Linear(num_agents, embed_dim)),
            nn.ReLU(),
        )

        # Fiedler value encoder (scalar → embed)
        self.fiedler_encoder = nn.Sequential(
            layer_init(nn.Linear(1, embed_dim)),
            nn.ReLU(),
        )

        # Combine agent + poi + adj + fiedler per token
        self.token_proj = nn.Sequential(
            layer_init(nn.Linear(embed_dim * 4, embed_dim)),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Value head: pool → scalar
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1), std=0.01),
        )

    def forward(self, obs, adjacency, fiedler):
        """Compute value for each agent.

        Args:
            obs:       (T, obs_dim) — T = num_games * num_agents
            adjacency: (T, N_a)     — row of adjacency matrix for this agent
            fiedler:   (T, 1)       — Fiedler value (same for all agents in a game)

        Returns:
            value: (T, 1)
        """
        s = self.slices
        T = obs.shape[0]
        N_a = self.num_agents
        B = T // N_a

        # --- Extract per-agent self features (for ALL agents in each game) ---
        # Reshape obs to (B, N_a, obs_dim) to get all agents' data
        obs_g = obs.view(B, N_a, -1)

        heading_all = obs_g[:, :, s["heading"][0]:s["heading"][1]]
        pos_all = obs_g[:, :, s["pos"][0]:s["pos"][1]]
        cov_all = obs_g[:, :, s["cov"][0]:s["cov"][1]]
        gpsd_all = obs_g[:, :, s["in_gpsd"][0]:s["in_gpsd"][1]]
        poi_all = obs_g[:, :, s["poi_rel"][0]:s["poi_rel"][1]]

        self_feat = torch.cat([heading_all, pos_all, cov_all, gpsd_all], dim=-1)
        # (B, N_a, 5)

        agent_emb = self.agent_encoder(self_feat)       # (B, N_a, E)
        poi_emb = self.poi_encoder(poi_all)              # (B, N_a, E)

        # Adjacency: reshape to (B, N_a, N_a)
        adj_g = adjacency.view(B, N_a, N_a)
        adj_emb = self.adj_encoder(adj_g)                # (B, N_a, E)

        # Fiedler: (T, 1) → (B, N_a, 1) → (B, N_a, E)
        fiedler_g = fiedler.view(B, N_a, 1)
        fiedler_emb = self.fiedler_encoder(fiedler_g)    # (B, N_a, E)

        # Combine into tokens
        tokens = torch.cat([agent_emb, poi_emb, adj_emb, fiedler_emb], dim=-1)
        tokens = self.token_proj(tokens)                  # (B, N_a, E)

        # Transformer self-attention
        tokens = self.transformer(tokens)                 # (B, N_a, E)

        # Pool to per-game value (mean over agents), broadcast back
        game_emb = tokens.mean(dim=1, keepdim=True)       # (B, 1, E)
        game_emb = game_emb.expand(B, N_a, -1)            # (B, N_a, E)
        game_emb = game_emb.reshape(T, -1)                # (T, E)

        value = self.value_head(game_emb)                 # (T, 1)
        return value


# ======================================================================
# Combined Agent
# ======================================================================

class GATMAPPOAgent(nn.Module):
    """Wraps GATActor + TransformerCritic into one module."""

    def __init__(self, obs_dim, act_dim, num_agents, num_pois, args):
        super().__init__()
        self.num_agents = num_agents
        self.actor = GATActor(
            obs_dim, act_dim, num_agents, num_pois,
            embed_dim=args.embed_dim,
            gat_heads=args.gat_heads,
            gat_hidden=args.gat_hidden,
            r_c=args.r_c,
        )
        self.critic = TransformerCritic(
            obs_dim, num_agents, num_pois,
            embed_dim=args.embed_dim,
            n_heads=args.transformer_heads,
            n_layers=args.transformer_layers,
        )

    def get_value(self, obs, adjacency, fiedler):
        return self.critic(obs, adjacency, fiedler)

    def get_action(self, obs, action=None):
        act, logp, ent, embed, dists = self.actor(obs, action)
        return act, logp, ent, embed, dists

    def get_action_and_value(self, obs, adjacency, fiedler, action=None):
        act, logp, ent, embed, dists = self.actor(obs, action)
        value = self.critic(obs, adjacency, fiedler)
        return act, logp, ent, value, embed, dists


# ======================================================================
# Environment factory
# ======================================================================

def make_env(args, conn_penalty_weight=1.0):
    env = make_gpsd_parallel_env(
        N_a=args.num_agents,
        cell_width=args.cell_width,
        max_cycles=args.max_cycles,
        speed=args.speed,
        r_c=args.r_c,
        cov_c=args.cov_c,
        conn_penalty_weight=conn_penalty_weight,
        buffer_eps=args.buffer_eps,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    num_copies = args.num_envs // args.num_agents
    envs = ss.concat_vec_envs_v1(
        env, num_copies, num_cpus=0, base_class="gymnasium",
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    return envs


# ======================================================================
# Topology extraction from info dicts
# ======================================================================

def extract_topology_from_obs(obs, num_agents, r_c, device):
    """Build adjacency + Fiedler value from observations.

    Since the vectorised env flattens info dicts, we reconstruct adjacency
    from the inter-agent range measurements embedded in each agent's obs.

    Args:
        obs: (T, obs_dim) tensor
        num_agents: int
        r_c: communication radius (used as threshold)
        device: torch device

    Returns:
        adj_rows: (T, N_a) — each agent's row of the adjacency matrix
        fiedler:  (T, 1)   — Fiedler value per agent (same for all in a game)
    """
    T = obs.shape[0]
    N = num_agents
    B = T // N
    obs_dim = obs.shape[1]

    # Parse observation layout to find other_comm section
    num_pois = (obs_dim - 5 - (N - 1) * 5) // 2
    slices = parse_obs_layout(obs_dim, N, num_pois)

    other_comm = obs[:, slices["other_comm"][0]:slices["other_comm"][1]]
    # (T, (N-1)*2) → (T, N-1, 2)
    other_comm = other_comm.view(T, N - 1, 2)
    ranges = other_comm[:, :, 0]  # (T, N-1)

    # Build full adjacency per game
    # ranges_g: (B, N, N-1)
    ranges_g = ranges.view(B, N, N - 1)

    # IMPORTANT: connectivity = within communication radius r_c,
    # NOT observation range (10*r_c).  range > 0 only means visible.
    adj_full = torch.zeros(B, N, N, device=device)
    for i in range(N):
        k = 0
        for j in range(N):
            if j == i:
                continue
            range_ij = ranges_g[:, i, k]
            connected = ((range_ij > 0) & (range_ij <= r_c)).float()
            adj_full[:, i, j] = connected
            k += 1

    # Symmetrise (in case noisy ranges cause asymmetry)
    adj_full = ((adj_full + adj_full.transpose(1, 2)) > 0).float()

    # Compute Fiedler value per game
    fiedler_vals = torch.zeros(B, device=device)
    for g in range(B):
        A = adj_full[g].cpu().numpy()
        D = np.diag(A.sum(axis=1))
        L = D - A
        eigvals = np.linalg.eigvalsh(L)
        fv = float(max(eigvals[1], 0.0)) if len(eigvals) >= 2 else 0.0
        fiedler_vals[g] = fv

    # Expand to per-agent: (B, N, N) and (B, 1) → (T, N) and (T, 1)
    adj_rows = adj_full.view(T, N)  # each agent gets its own row
    # But wait — adj_full is (B, N, N) and we need row i for agent i.
    # After .view(T, N) this maps correctly: agent_i in game_g maps to
    # row index g*N+i, which is adj_full[g, i, :].
    adj_rows = adj_full.reshape(T, N)

    fiedler_per_agent = fiedler_vals.unsqueeze(1).expand(B, N).reshape(T, 1)

    return adj_rows, fiedler_per_agent


# ======================================================================
# Video recording
# ======================================================================

def record_video(args, agent_model, run_name, num_episodes=1):
    try:
        import imageio
    except ImportError:
        print("imageio not installed – skipping video.")
        return

    print("Recording video...")
    device = next(agent_model.parameters()).device

    from pettingzoo.mpe.gpsd.gpsd_conn import parallel_env as make_env_raw
    env = make_env_raw(
        N_a=args.num_agents, cell_width=args.cell_width,
        max_cycles=args.max_cycles, speed=args.speed,
        r_c=args.r_c, cov_c=args.cov_c,
        render_mode="rgb_array",
    )

    frames = []
    for ep in range(num_episodes):
        obs_dict, infos = env.reset()
        for _ in range(args.max_cycles):
            with torch.no_grad():
                obs_arr = np.stack([obs_dict[a] for a in env.agents])
                obs_t = torch.tensor(obs_arr, dtype=torch.float32).to(device)
                actions, _, _, _, _ = agent_model.get_action(obs_t)

            action_dict = {
                name: actions[i].item()
                for i, name in enumerate(env.agents)
            }
            obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if all(terms.values()) or all(truncs.values()):
                break

    env.close()
    if frames:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        path = f"runs/{run_name}/gpsd_trained.mp4"
        imageio.mimwrite(path, frames, fps=30, codec="libx264")
        print(f"Saved video → {path}")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    args = parse_args()
    run_name = f"gat_{args.seed}_{int(time.time())}"
    print(f"\n{'='*70}")
    print(f"  GPSD GAT-MAPPO Training  —  {run_name}")
    print(f"{'='*70}")
    print(args)
    print()

    # --- W&B ---
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=False, config=vars(args),
            name=run_name, save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    print(f"Device: {device}")

    # --- Curriculum: start with high connectivity penalty ---
    conn_penalty = args.conn_penalty_start
    envs = make_env(args, conn_penalty_weight=conn_penalty)

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = envs.single_action_space.n
    num_pois = int((1.0 / args.cell_width) ** 2)
    print(f"Obs dim: {obs_dim}   Act dim: {act_dim}   POIs: {num_pois}")
    print(f"Num envs: {args.num_envs}  "
          f"({args.num_envs // args.num_agents} games × {args.num_agents} agents)")
    print(f"Batch: {args.batch_size}   Minibatch: {args.minibatch_size}")
    print()

    # --- Agent ---
    agent = GATMAPPOAgent(
        obs_dim, act_dim, args.num_agents, num_pois, args
    ).to(device)

    actor_params = list(agent.actor.parameters())
    critic_params = list(agent.critic.parameters())
    actor_optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(critic_params, lr=args.learning_rate * 3.0, eps=1e-5)

    n_actor = sum(p.numel() for p in actor_params)
    n_critic = sum(p.numel() for p in critic_params)
    print(f"Actor params:  {n_actor:,}")
    print(f"Critic params: {n_critic:,}")
    print(f"Total params:  {n_actor + n_critic:,}")
    print()

    # --- Rollout storage ---
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    rewards_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    terms_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    truncs_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    values_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    # Topology storage
    adj_buf = torch.zeros(args.num_steps, args.num_envs, args.num_agents).to(device)
    fiedler_buf = torch.zeros(args.num_steps, args.num_envs, 1).to(device)

    # --- Init ---
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_term = torch.zeros(args.num_envs).to(device)
    next_trunc = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Normalisers
    obs_rms = RunningMeanStdVec((obs_dim,))
    value_norm = ValueNorm()

    # Episode tracking
    episode_rewards = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_coverage = np.zeros(args.num_envs, dtype=np.float64)
    recent_returns = []
    recent_coverage = []
    recent_fiedler = []

    print(f"PPO updates: {num_updates}\n")

    # --- Ctrl+C handler ---
    import signal as _signal
    _interrupted = [False]
    def _sigint(sig, frame):
        print("\n\nCtrl+C – saving…")
        _interrupted[0] = True
    _signal.signal(_signal.SIGINT, _sigint)

    for update in range(1, num_updates + 1):
        # --- Curriculum: linearly decay connectivity penalty ---
        progress = (update - 1) / max(num_updates - 1, 1)
        conn_penalty = (args.conn_penalty_start
                        + (args.conn_penalty_end - args.conn_penalty_start) * progress)

        # --- LR annealing ---
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"] = lrnow
            critic_optimizer.param_groups[0]["lr"] = lrnow * 3.0

        # ========== Rollout ==========
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            terms_buf[step] = next_term
            truncs_buf[step] = next_trunc

            # Topology from obs
            adj_rows, fiedler = extract_topology_from_obs(
                next_obs, args.num_agents, args.r_c, device
            )
            adj_buf[step] = adj_rows
            fiedler_buf[step] = fiedler

            with torch.no_grad():
                norm_obs = obs_rms.normalize(next_obs)
                (action, logprob, _, value,
                 _, _) = agent.get_action_and_value(
                    norm_obs, adj_rows, fiedler
                )
                values_buf[step] = value_norm.denormalize(value).flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, term, trunc, info = envs.step(
                action.cpu().numpy()
            )
            rewards_buf[step] = torch.tensor(
                np.array(reward, dtype=np.float64), dtype=torch.float32
            ).to(device).view(-1)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            obs_rms.update(next_obs.cpu().numpy())
            next_term = torch.tensor(term, dtype=torch.float32).to(device)
            next_trunc = torch.tensor(trunc, dtype=torch.float32).to(device)

            # Episode tracking
            episode_rewards += reward
            episode_lengths += 1

            if isinstance(info, (list, tuple)):
                for idx, item in enumerate(info):
                    if isinstance(item, dict):
                        if "coverage_ratio" in item:
                            episode_coverage[idx] = item["coverage_ratio"]
                        if "fiedler_value" in item:
                            recent_fiedler.append(item["fiedler_value"])

            done_flags = np.maximum(term, trunc)
            for idx in range(args.num_envs):
                if done_flags[idx]:
                    agent_idx = idx % args.num_agents
                    writer.add_scalar(
                        f"charts/episodic_return_agent{agent_idx}",
                        episode_rewards[idx], global_step,
                    )
                    recent_returns.append(episode_rewards[idx])
                    recent_coverage.append(episode_coverage[idx])
                    episode_rewards[idx] = 0.0
                    episode_lengths[idx] = 0
                    episode_coverage[idx] = 0.0

            if len(recent_returns) >= 20:
                avg_ret = np.mean(recent_returns[-100:])
                avg_cov = np.mean(recent_coverage[-100:])
                writer.add_scalar("charts/avg_episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/avg_coverage_ratio", avg_cov, global_step)

        # ========== GAE ==========
        with torch.no_grad():
            adj_next, fiedler_next = extract_topology_from_obs(
                next_obs, args.num_agents, args.r_c, device
            )
            norm_next = obs_rms.normalize(next_obs)
            next_value = value_norm.denormalize(
                agent.get_value(norm_next, adj_next, fiedler_next)
            ).reshape(1, -1)

            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            dones = torch.maximum(terms_buf, truncs_buf)
            next_done = torch.maximum(next_term, next_trunc)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = (rewards_buf[t]
                         + args.gamma * nextvalues * nextnonterminal
                         - values_buf[t])
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        value_norm.update(returns.cpu().numpy().flatten())

        # ========== Flatten ==========
        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_obs_norm = obs_rms.normalize(b_obs)
        b_adj = adj_buf.reshape(-1, args.num_agents)
        b_fiedler = fiedler_buf.reshape(-1, 1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_returns_norm = value_norm.normalize(b_returns)
        b_values_norm = value_norm.normalize(b_values)

        # ========== PPO Update ==========
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb = b_inds[start:end]

                (_, newlogprob, entropy, newvalue,
                 actor_embed, true_dists) = agent.get_action_and_value(
                    b_obs_norm[mb], b_adj[mb], b_fiedler[mb],
                    b_actions.long()[mb],
                )

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss (Huber, normalised targets)
                newvalue = newvalue.view(-1)
                v_loss = F.huber_loss(
                    newvalue, b_returns_norm[mb], reduction="mean", delta=10.0
                )

                # Contrastive auxiliary loss
                contrastive = agent.actor.contrastive_loss(actor_embed, true_dists)

                entropy_loss = entropy.mean()
                actor_loss = (pg_loss
                              - args.ent_coef * entropy_loss
                              + args.contrastive_coef * contrastive)

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss = actor_loss + v_loss * args.vf_coef
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Extra critic epochs
        for _ in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb = b_inds[start:end]
                nv = agent.get_value(
                    b_obs_norm[mb], b_adj[mb], b_fiedler[mb]
                ).view(-1)
                vl = F.huber_loss(nv, b_returns_norm[mb], reduction="mean", delta=10.0)
                critic_optimizer.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                critic_optimizer.step()

        # ========== Logging ==========
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate",
                          actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/contrastive", contrastive.item(), global_step)
        writer.add_scalar("charts/conn_penalty_weight", conn_penalty, global_step)

        # Adjacency density (fraction of edges in GAT graph)
        adj_density = b_adj.mean().item()  # proportion of 1s in adjacency rows
        writer.add_scalar("charts/adj_density", adj_density, global_step)

        avg_fiedler = np.mean(recent_fiedler[-200:]) if recent_fiedler else 0.0
        writer.add_scalar("charts/avg_fiedler_value", avg_fiedler, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if args.track:
            log_dict = {
                "charts/learning_rate": actor_optimizer.param_groups[0]["lr"],
                "charts/SPS": sps,
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "losses/contrastive": contrastive.item(),
                "charts/conn_penalty_weight": conn_penalty,
                "charts/avg_fiedler_value": avg_fiedler,
                "charts/adj_density": adj_density,
            }
            if recent_returns:
                log_dict["charts/avg_episodic_return"] = np.mean(recent_returns[-100:])
            if recent_coverage:
                log_dict["charts/avg_coverage_ratio"] = np.mean(recent_coverage[-100:])
            wandb.log(log_dict, step=global_step)

        if update % 10 == 0 or update == num_updates:
            avg_r = np.mean(recent_returns[-100:]) if recent_returns else float("nan")
            avg_c = np.mean(recent_coverage[-100:]) if recent_coverage else float("nan")
            print(
                f"  [{update:>4}/{num_updates}]  "
                f"step={global_step:>8}  SPS={sps:>5}  "
                f"pg={pg_loss.item():+.4f}  v={v_loss.item():.4f}  "
                f"ent={entropy_loss.item():.3f}  ctr={contrastive.item():.4f}  "
                f"ret={avg_r:.2f}  cov={avg_c:.3f}  "
                f"fiedler={avg_fiedler:.3f}  adj={adj_density:.3f}  EV={explained_var:.3f}  "
                f"conn_w={conn_penalty:.2f}"
            )

        if _interrupted[0]:
            break

    # ========== Post-training ==========
    _signal.signal(_signal.SIGINT, _signal.SIG_DFL)
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    if _interrupted[0]:
        print(f"  Interrupted at step {global_step}. Time: {elapsed:.1f}s")
    else:
        print(f"  Complete! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        path = f"runs/{run_name}/gpsd_gat_agent.pt"
        torch.save({
            "model_state_dict": agent.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        }, path)
        print(f"  Model saved → {path}")

    if args.capture_video:
        record_video(args, agent, run_name)

    envs.close()
    writer.close()
    print("Done.")
