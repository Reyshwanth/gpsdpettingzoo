#!/usr/bin/env python3
"""Test script for GPSD environment with visualization of trained policies."""
import sys
import argparse
import os
import glob
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


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------
def list_available_policies():
    """Find all saved policy models in the runs directory."""
    model_paths = glob.glob("runs/*/gpsd_ppo_agent.pt")
    policies = []
    for path in sorted(model_paths):
        run_name = path.split('/')[1]
        policies.append({
            'name': run_name,
            'path': path,
            'timestamp': run_name.split('__')[-1] if '__' in run_name else 'unknown'
        })
    return policies


def load_policy(model_path, obs_dim, act_dim, device):
    """Load a trained policy from disk."""
    agent = Agent(obs_dim, act_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint format and raw state dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format (includes optimizer state, args, etc.)
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Raw state dict format
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
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (faster)")
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

    # Create the environment with rendering enabled
    render_mode = None if args.no_render else "human"
    e = raw_env(
        N_a=5,
        cell_width=0.25,  # GPS denied zone (1.0 wide) / 0.25 = 4x4 = 16 POIs
        max_cycles=args.max_cycles,
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
        # Auto-complete path if directory is provided
        policy_path = args.policy
        if os.path.isdir(policy_path):
            # User provided directory, append the model filename
            policy_path = os.path.join(policy_path, "gpsd_ppo_agent.pt")
            print(f"Directory provided, using: {policy_path}")
        
        if not os.path.exists(policy_path):
            print(f"\n✗ Error: Policy file not found: {policy_path}")
            print("Run with --policy list to see available policies")
            sys.exit(1)
        
        obs_dim = e.observation_space(e.possible_agents[0]).shape[0]
        act_dim = e.action_space(e.possible_agents[0]).n
        print(f"\nLoading trained policy from: {policy_path}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {act_dim}")
        policy_agent = load_policy(policy_path, obs_dim, act_dim, device)
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
            
            # Run episode
            for cycle in range(args.max_cycles):
                cycle_rewards = []
                
                for _ in range(len(e.possible_agents)):
                    agent_name = e.agent_selection
                    obs, rew, term, trunc, info = e.last()
                    
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
