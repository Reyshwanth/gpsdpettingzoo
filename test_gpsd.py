#!/usr/bin/env python3
"""Test script for GPSD environment with visualization of trained policies."""
import sys
import argparse
import os
import glob
import pickle
sys.path.insert(0, "PettingZoo")
from pettingzoo.mpe.gpsd.gpsd import env, raw_env, parallel_env
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# ---------------------------------------------------------------------------
# Network helpers (must match train_gpsd_ppo.py)
# ---------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialisation (CleanRL convention)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """MLP policy + value network for vector observations."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class CNNAgent(nn.Module):
    """CNN + MLP policy/value network (must match train_gpsd_cnn.py)."""

    POI_VEC_START = 5
    POI_CHANNELS  = 3
    CNN_FEAT_DIM  = 128

    def __init__(self, num_agents: int, n_cells: int, act_dim: int):
        super().__init__()
        self.num_agents = num_agents
        self.n_cells = n_cells
        self.poi_end = self.POI_VEC_START + n_cells * n_cells * self.POI_CHANNELS
        self.vec_dim = self.POI_VEC_START + (num_agents - 1) * 4

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(self.POI_CHANNELS, 16, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        combined_dim = self.CNN_FEAT_DIM + self.vec_dim
        self.network = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def _split_obs(self, x):
        vec_part1 = x[:, :self.POI_VEC_START]
        poi_flat  = x[:, self.POI_VEC_START:self.poi_end]
        vec_part2 = x[:, self.poi_end:]
        vec = torch.cat([vec_part1, vec_part2], dim=1)
        poi_img = poi_flat.reshape(-1, self.n_cells, self.n_cells, self.POI_CHANNELS)
        poi_img = poi_img.permute(0, 3, 1, 2)
        return vec, poi_img

    def _encode(self, x):
        vec, poi_img = self._split_obs(x)
        cnn_features = self.cnn(poi_img)
        combined = torch.cat([vec, cnn_features], dim=1)
        return self.network(combined)

    def get_value(self, x):
        return self.critic(self._encode(x))

    def get_action_and_value(self, x, action=None):
        hidden = self._encode(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


# ---------------------------------------------------------------------------
# MAPPO agent (architecture used by train_gpsd_mappo.py)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Observation normalisation (must match train_gpsd_mappo.py)
# ---------------------------------------------------------------------------
class RunningMeanStdVec:
    """Tracks per-feature running mean and variance (Welford's algorithm)."""
    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def normalize(self, x, clip=10.0):
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std_t = torch.sqrt(torch.tensor(self.var, dtype=torch.float32, device=x.device)) + 1e-8
        return torch.clamp((x - mean_t) / std_t, -clip, clip)


class MAPPOAgent(nn.Module):
    """Actor uses local obs; critic uses global state (all agents' obs).

    IMPORTANT: hidden dim is 128, matching train_gpsd_mappo.py.
    """

    def __init__(self, obs_dim: int, act_dim: int, num_agents: int,
                 global_dim: int = None):
        super().__init__()
        self.num_agents = num_agents
        if global_dim is None:
            global_dim = obs_dim * num_agents

        # --- Decentralised Actor (local observations only) ---
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(512, act_dim), std=0.01)

        # --- Centralised Critic (global state) ---
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(global_dim, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.critic_head = layer_init(nn.Linear(512, 1), std=0.01)

    def get_value(self, global_state):
        return self.critic_head(self.critic_net(global_state))

    def get_action(self, local_obs, action=None):
        hidden = self.actor_net(local_obs)
        logits = self.actor_head(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_action_and_value(self, local_obs, global_state, action=None):
        act, logprob, entropy = self.get_action(local_obs, action)
        value = self.get_value(global_state)
        return act, logprob, entropy, value


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------
def list_available_policies():
    """Find all saved policy models in the runs directory."""
    # Include any saved .pt files inside run folders (ppo, mappo, etc.)
    model_paths = glob.glob("runs/*/*.pt")
    policies = []
    for path in sorted(model_paths):
        run_name = path.split('/')[1]
        policies.append({
            'name': run_name,
            'path': path,
            'timestamp': run_name.split('__')[-1] if '__' in run_name else 'unknown'
        })
    return policies


def load_checkpoint_args(model_path):
    """Load the training args saved in a checkpoint (if available)."""
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        return checkpoint['args']
    return {}


def load_policy(model_path, obs_dim, act_dim, device, num_agents=5, n_cells=4):
    """Load a trained policy from disk (auto-detects MLP vs CNN vs MAPPO architecture)."""
    checkpoint = torch.load(model_path, map_location=device)

    # Detect architecture from checkpoint metadata
    arch = None
    ckpt_n_cells = n_cells
    ckpt_num_agents = num_agents
    if isinstance(checkpoint, dict):
        arch = checkpoint.get('arch', None)
        ckpt_n_cells = checkpoint.get('n_cells', n_cells)
        ckpt_args = checkpoint.get('args', {})
        ckpt_num_agents = ckpt_args.get('num_agents', num_agents)
        if 'cell_width' in ckpt_args:
            ckpt_n_cells = int(np.round(1.0 / ckpt_args['cell_width']))

    # Determine which state_dict to inspect/load
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else None

    # If checkpoint appears to be a MAPPO-trained model (keys like 'actor_net'),
    # instantiate the MAPPOAgent and wrap it so it exposes get_action_and_value(x).
    is_mappo = False
    if isinstance(state_dict, dict):
        for k in state_dict.keys():
            if k.startswith('actor_net') or k.startswith('actor_head') or k.startswith('critic_head'):
                is_mappo = True
                break

    if is_mappo:
        # Determine hidden dim from state dict (default to 128 if not found)
        hidden_dim = 128
        if isinstance(state_dict, dict) and 'actor_net.0.weight' in state_dict:
            hidden_dim = state_dict['actor_net.0.weight'].shape[0]
            
        # Build MAPPO agent with matching dims
        # The critic global_dim is obs_dim * num_agents + world_obs_dim
        # world_obs_dim = N_a * 6 + N_pois * 3 (where N_pois = n_cells^2)
        n_pois = ckpt_n_cells * ckpt_n_cells
        world_obs_dim = ckpt_num_agents * 6 + n_pois * 3
        global_dim = obs_dim * ckpt_num_agents + world_obs_dim

        mappo_agent = MAPPOAgent(
            obs_dim, act_dim, ckpt_num_agents, 
            global_dim=global_dim, hidden_dim=hidden_dim
        ).to(device)
        print(f"  Architecture: MAPPO (detected from checkpoint, hidden={hidden_dim})")
        mappo_agent.load_state_dict(state_dict)

        # --- Load obs normaliser if saved alongside the checkpoint ---
        obs_rms = None
        if isinstance(checkpoint, dict) and 'obs_rms' in checkpoint:
            obs_rms = checkpoint['obs_rms']
            print(f"  Loaded obs normaliser (obs_rms) from checkpoint")
        else:
            # Try loading from a sibling pickle file
            rms_path = os.path.join(os.path.dirname(model_path), "obs_rms.pkl")
            if os.path.exists(rms_path):
                with open(rms_path, "rb") as f:
                    obs_rms = pickle.load(f)
                print(f"  Loaded obs normaliser from {rms_path}")
            else:
                print("  WARNING: No obs normaliser found — feeding raw observations!")

        # Wrapper that normalises observations and properly reconstructs
        # the global state from all agents' local observations.
        class MAPPOWrapper(nn.Module):
            def __init__(self, agent, num_agents, global_dim, obs_rms=None):
                super().__init__()
                self.agent = agent
                self.num_agents = num_agents
                self.global_dim = global_dim
                self.obs_rms = obs_rms

            def eval(self):
                self.agent.eval()

            def get_action_and_value(self, local_obs, action=None):
                # Normalise observations the same way training does
                if self.obs_rms is not None:
                    local_obs = self.obs_rms.normalize(local_obs)

                # Decentralised execution: only the actor is needed
                act, logprob, entropy = self.agent.get_action(local_obs, action)
                return act, logprob, entropy, torch.zeros((local_obs.shape[0], 1), device=local_obs.device)

        wrapped = MAPPOWrapper(mappo_agent, ckpt_num_agents, global_dim, obs_rms=obs_rms)
        wrapped.eval()
        return wrapped

    # Fallback: standard (PPO-style) architectures
    if arch == 'cnn':
        agent = CNNAgent(ckpt_num_agents, ckpt_n_cells, act_dim).to(device)
        print(f"  Architecture: CNN (n_cells={ckpt_n_cells})")
    else:
        agent = Agent(obs_dim, act_dim).to(device)
        print(f"  Architecture: MLP")

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)

    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Test and visualize GPSD policies")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy .pt file or 'random' or 'list' to show all")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Maximum cycles per episode")
    parser.add_argument("--cell-width", type=float, default=None,
                        help="Cell width override (auto-detected from checkpoint if not set)")
    parser.add_argument("--num-agents", type=int, default=None,
                        help="Number of agents override (auto-detected from checkpoint if not set)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (faster)")
    parser.add_argument("--plot-rewards", action="store_true",
                        help="Save a plot of rewards per agent over time")
    parser.add_argument("--plot-ratios", action="store_true",
                        help="Save a plot of global/local reward magnitude ratios")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # List available policies if requested
    if args.policy == "list":
        policies = list_available_policies()
        print("=" * 70)
        print("AVAILABLE TRAINED POLICIES")
        print("=" * 70)
        for i, pol in enumerate(policies, 1):
            print(f"{i}. {pol['name']}")
            print(f"   Path: {pol['path']}")
            print()
        if not policies:
            print("No trained policies found in runs/ directory")
        print("Usage: python test_gpsd.py --policy <path_to_policy.pt>")
        print("   or: python test_gpsd.py --policy random")
        sys.exit(0)
    
    # Determine policy type
    use_trained_policy = args.policy is not None and args.policy != "random"
    
    print("=" * 70)
    if use_trained_policy:
        print(f"GPSD ENVIRONMENT - TRAINED POLICY: {args.policy}")
    else:
        print("GPSD ENVIRONMENT - RANDOM POLICY VISUALIZATION")
    print("=" * 70)

    # --- Auto-detect env params from checkpoint ---
    ckpt_args = {}
    policy_path = None
    if use_trained_policy:
        policy_path = args.policy
        if os.path.isdir(policy_path):
            # If a directory is provided, pick the first .pt file inside (supports ppo/mappo)
            pt = None
            for candidate in glob.glob(os.path.join(policy_path, "*.pt")):
                pt = candidate
                break
            if pt is None:
                print(f"\n✗ Error: No .pt file found in directory: {policy_path}")
                sys.exit(1)
            policy_path = pt
            print(f"Directory provided, using: {policy_path}")
        if not os.path.exists(policy_path):
            print(f"\n✗ Error: Policy file not found: {policy_path}")
            print("Run with --policy list to see available policies")
            sys.exit(1)
        ckpt_args = load_checkpoint_args(policy_path)
        if ckpt_args:
            print(f"  Loaded training config from checkpoint")

    # Resolve env parameters: CLI override > checkpoint > defaults
    N_a = args.num_agents or ckpt_args.get('num_agents', 5)
    cell_width = args.cell_width or ckpt_args.get('cell_width', 0.25)
    speed = ckpt_args.get('speed', 0.1)
    r_c = ckpt_args.get('r_c', 0.3)
    cov_c = ckpt_args.get('cov_c', 0.1)
    max_cycles = args.max_cycles

    # Show computed grid info
    zone_size = 1.0
    n_cells = int(np.round(zone_size / cell_width))
    n_pois = n_cells * n_cells
    print(f"  cell_width={cell_width}, grid={n_cells}x{n_cells}, POIs={n_pois}, N_a={N_a}")

    # Create the environment with rendering enabled
    render_mode = None if args.no_render else "human"
    e = raw_env(
        N_a=N_a,
        cell_width=cell_width,
        max_cycles=max_cycles,
        speed=speed,
        r_c=r_c,
        cov_c=cov_c,
        render_mode=render_mode
    )

    print(f"\nEnvironment: {e.metadata.get('name', 'unknown')}")
    print(f"Agents: {e.possible_agents}")
    print(f"Number of agents: {len(e.possible_agents)}")
    print(f"Action space (agent_0): {e.action_space(e.possible_agents[0])}")
    
    # Load trained policy if specified
    policy_agent = None
    device = torch.device("cpu")
    if use_trained_policy:
        obs_dim = e.observation_space(e.possible_agents[0]).shape[0]
        act_dim = e.action_space(e.possible_agents[0]).n
        print(f"\nLoading trained policy from: {policy_path}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {act_dim}")
        policy_agent = load_policy(policy_path, obs_dim, act_dim, device,
                                    num_agents=N_a, n_cells=n_cells)
        print("✓ Policy loaded successfully!")

    # Run episodes
    for episode in range(args.num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{args.num_episodes}")
        print(f"{'='*70}")
        
        # Reset environment
        e.reset()
        print(f"✓ Environment reset successfully!")
        print(f"✓ All agents initialized OUTSIDE GPS denied zone (center 10x10)")
        print(f"✓ All POIs placed INSIDE GPS denied zone")

        # Check initial positions
        print("\nInitial positions:")
        for agent in e.world.agents:
            in_zone = e.world.is_in_gpsd_zone(agent.state.p_pos)
            print(f"  {agent.name}: pos=({agent.state.p_pos[0]:.2f}, {agent.state.p_pos[1]:.2f}), "
                  f"heading={agent.state.heading:.2f}, in_GPSD={in_zone}")

        print(f"\n{'='*70}")
        if use_trained_policy:
            print(f"RUNNING TRAINED POLICY for {args.max_cycles} timesteps")
        else:
            print(f"RUNNING RANDOM POLICY for {args.max_cycles} timesteps")
        if render_mode:
            print("Close the window or wait for completion...")
        print(f"{'='*70}")

        try:
            step_count = 0
            total_reward = 0
            episode_rewards = []
            coverage_progress = []
            
            # Track rewards for each agent over time
            agent_history = {agent: [] for agent in e.possible_agents}
            ratio_history = {agent: [] for agent in e.possible_agents}
            
            # Run episode
            for cycle in range(args.max_cycles):
                cycle_rewards = []
                
                for _ in range(len(e.possible_agents)):
                    agent_name = e.agent_selection
                    obs, rew, term, trunc, info = e.last()
                    
                    # Record reward for this agent
                    agent_history[agent_name].append(rew)
                    
                    # Record local and global reward for ratio plotting
                    if args.plot_ratios:
                        local_r = info.get("local_reward", 0.0)
                        global_r = info.get("global_reward", 0.0)
                        # Avoid division by zero, use epsilon
                        ratio = abs(global_r) / (abs(local_r) + 1e-6)
                        ratio_history[agent_name].append(ratio)
                    
                    cycle_rewards.append(rew)
                    total_reward += rew
                    
                    if term or trunc:
                        action = None
                    else:
                        if use_trained_policy and policy_agent is not None:
                            # Use trained policy
                            with torch.no_grad():
                                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                                action_tensor, _, _, _ = policy_agent.get_action_and_value(obs_tensor)
                                action = action_tensor.item()
                        else:
                            # Random action
                            action = e.action_space(agent_name).sample()
                    
                    e.step(action)
                    step_count += 1
                    
                    # Break if episode ends
                    if not e.agents:
                        break
                
                if not e.agents:
                    print(f"\nEpisode terminated at cycle {cycle + 1}")
                    break
                
                # Track metrics every 10 cycles
                if (cycle + 1) % 10 == 0 or cycle == 0:
                    avg_reward = np.mean(cycle_rewards) if cycle_rewards else 0
                    episode_rewards.append(avg_reward)
                    
                    # Count covered POIs
                    covered = sum(e.scenario.covered) if hasattr(e.scenario, 'covered') else 0
                    total_pois = len(e.world.landmarks)
                    coverage_progress.append((covered, total_pois))
                    
                    print(f"\n--- Cycle {cycle + 1} ---")
                    print(f"  Average reward: {avg_reward:.3f}")
                    print(f"  POIs covered: {covered}/{total_pois}")
                    
                    # Print agent states
                    for agent in e.world.agents:
                        in_zone = e.world.is_in_gpsd_zone(agent.state.p_pos)
                        cov_trace = np.trace(agent.state.p_covariance) if isinstance(agent.state.p_covariance, np.ndarray) else 0
                        belief = agent.state.p_belief if agent.state.p_belief is not None else agent.state.p_pos
                        belief_err = np.sqrt(np.sum(np.square(agent.state.p_pos - belief)))
                        print(f"    {agent.name}: pos=({agent.state.p_pos[0]:.2f}, {agent.state.p_pos[1]:.2f}), "
                              f"belief_err={belief_err:.4f}, in_GPSD={in_zone}, cov={cov_trace:.4f}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        # Plot rewards if requested
        if args.plot_rewards:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                for agent_name, rewards in agent_history.items():
                    plt.plot(rewards, label=agent_name, alpha=0.8)
                
                plt.title(f"Rewards per Agent - Episode {episode + 1}")
                plt.xlabel("Cycle")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Create a plots directory if it doesn't exist
                os.makedirs("plots", exist_ok=True)
                plot_filename = f"plots/episode_{episode + 1}_rewards.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"✓ Reward plot saved to: {plot_filename}")
            except ImportError:
                print("! Warning: matplotlib not found, skipping plot generation.")
            except Exception as e:
                print(f"! Warning: Failed to generate plot: {e}")

        # Plot reward ratios if requested
        if args.plot_ratios:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                for agent_name, ratios in ratio_history.items():
                    plt.plot(ratios, label=agent_name, alpha=0.8)
                
                plt.title(f"Global/Local Reward Magnitude Ratio - Episode {episode + 1}")
                plt.xlabel("Cycle")
                plt.ylabel("|Global| / (|Local| + eps)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log') # Use log scale as ratios can vary wildly
                
                os.makedirs("plots", exist_ok=True)
                plot_filename = f"plots/episode_{episode + 1}_ratios.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"✓ Reward ratio plot saved to: {plot_filename}")
            except Exception as e:
                print(f"! Warning: Failed to generate ratio plot: {e}")

        # Episode summary
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward: {total_reward/step_count if step_count > 0 else 0:.3f}")
        if coverage_progress:
            final_covered, final_total = coverage_progress[-1]
            print(f"Final coverage: {final_covered}/{final_total} POIs ({100*final_covered/final_total:.1f}%)")
        print()

    e.close()

    print(f"{'='*70}")
    print("✓ Testing completed!")
    print(f"{'='*70}\n")
