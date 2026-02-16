#!/usr/bin/env python3
"""Compare multiple trained policies on the GPSD environment."""
import sys
import argparse
import os
import glob
sys.path.insert(0, "PettingZoo")
from pettingzoo.mpe.gpsd.gpsd import raw_env
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt


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
# Policy evaluation
# ---------------------------------------------------------------------------
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


def evaluate_policy(policy_agent, device, num_episodes=5, max_cycles=100, seed=None):
    """Evaluate a policy and return performance metrics."""
    results = {
        'rewards': [],
        'coverage': [],
        'steps': [],
        'final_coverage_pct': [],
    }
    
    for episode in range(num_episodes):
        e = raw_env(
            N_a=5,
            cell_width=0.25,
            max_cycles=max_cycles,
            render_mode=None
        )
        
        if seed is not None:
            e.reset(seed=seed + episode)
        else:
            e.reset()
        
        episode_reward = 0
        step_count = 0
        
        for cycle in range(max_cycles):
            for _ in range(len(e.possible_agents)):
                agent_name = e.agent_selection
                obs, rew, term, trunc, info = e.last()
                
                episode_reward += rew
                
                if term or trunc:
                    action = None
                else:
                    if policy_agent is not None:
                        # Use trained policy
                        with torch.no_grad():
                            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                            action_tensor, _, _, _ = policy_agent.get_action_and_value(obs_tensor)
                            action = action_tensor.item()
                    else:
                        # Random policy
                        action = e.action_space(agent_name).sample()
                
                e.step(action)
                step_count += 1
                
                if not e.agents:
                    break
            
            if not e.agents:
                break
        
        # Final metrics
        covered = sum(e.scenario.covered) if hasattr(e.scenario, 'covered') else 0
        total_pois = len(e.world.landmarks)
        coverage_pct = 100 * covered / total_pois if total_pois > 0 else 0
        
        results['rewards'].append(episode_reward)
        results['coverage'].append(covered)
        results['steps'].append(step_count)
        results['final_coverage_pct'].append(coverage_pct)
        
        e.close()
    
    return results


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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GPSD policies")
    parser.add_argument("--policies", type=str, nargs='+', default=None,
                        help="Paths to policy .pt files to compare (or 'all' for all policies)")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate each policy")
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Maximum cycles per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--include-random", action="store_true",
                        help="Include random baseline policy")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save comparison plot to this file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cpu")
    
    # Determine which policies to compare
    if args.policies is None or (len(args.policies) == 1 and args.policies[0] == 'all'):
        available_policies = list_available_policies()
        if not available_policies:
            print("No trained policies found in runs/ directory")
            sys.exit(1)
        policy_paths = [p['path'] for p in available_policies]
        policy_names = [p['name'] for p in available_policies]
    else:
        policy_paths = args.policies
        policy_names = [os.path.basename(os.path.dirname(p)) for p in policy_paths]
    
    print("=" * 70)
    print("GPSD POLICY COMPARISON")
    print("=" * 70)
    print(f"Evaluating {len(policy_paths)} policies with {args.num_episodes} episodes each")
    print()
    
    # Get observation and action dimensions
    temp_env = raw_env(N_a=5, cell_width=0.25, max_cycles=100)
    temp_env.reset()
    obs_dim = temp_env.observation_space(temp_env.possible_agents[0]).shape[0]
    act_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    temp_env.close()
    
    # Evaluate each policy
    all_results = {}
    
    for policy_path, policy_name in zip(policy_paths, policy_names):
        print(f"\nEvaluating: {policy_name}")
        print(f"  Path: {policy_path}")
        
        if not os.path.exists(policy_path):
            print(f"  ✗ File not found, skipping")
            continue
        
        policy_agent = load_policy(policy_path, obs_dim, act_dim, device)
        results = evaluate_policy(policy_agent, device, args.num_episodes, args.max_cycles, args.seed)
        all_results[policy_name] = results
        
        print(f"  Average reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
        print(f"  Average coverage: {np.mean(results['final_coverage_pct']):.1f}% ± {np.std(results['final_coverage_pct']):.1f}%")
        print(f"  Average steps: {np.mean(results['steps']):.1f}")
    
    # Random baseline
    if args.include_random:
        print(f"\nEvaluating: Random Policy (baseline)")
        results = evaluate_policy(None, device, args.num_episodes, args.max_cycles, args.seed)
        all_results['Random'] = results
        print(f"  Average reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
        print(f"  Average coverage: {np.mean(results['final_coverage_pct']):.1f}% ± {np.std(results['final_coverage_pct']):.1f}%")
        print(f"  Average steps: {np.mean(results['steps']):.1f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<50} {'Avg Reward':>12} {'Avg Coverage':>15}")
    print("-" * 70)
    
    sorted_policies = sorted(all_results.items(), 
                            key=lambda x: np.mean(x[1]['rewards']), 
                            reverse=True)
    
    for policy_name, results in sorted_policies:
        display_name = policy_name if len(policy_name) <= 47 else policy_name[:44] + "..."
        avg_reward = np.mean(results['rewards'])
        avg_coverage = np.mean(results['final_coverage_pct'])
        print(f"{display_name:<50} {avg_reward:>12.2f} {avg_coverage:>14.1f}%")
    
    # Create visualization
    if len(all_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('GPSD Policy Comparison', fontsize=16)
        
        # Prepare data
        policy_labels = []
        rewards_data = []
        coverage_data = []
        steps_data = []
        
        for policy_name, results in sorted_policies:
            short_name = policy_name.split('__')[-1] if '__' in policy_name else policy_name
            policy_labels.append(short_name)
            rewards_data.append(results['rewards'])
            coverage_data.append(results['final_coverage_pct'])
            steps_data.append(results['steps'])
        
        # Plot 1: Rewards
        axes[0, 0].boxplot(rewards_data, labels=policy_labels)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Coverage
        axes[0, 1].boxplot(coverage_data, labels=policy_labels)
        axes[0, 1].set_title('Coverage Performance')
        axes[0, 1].set_ylabel('POIs Covered (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Steps
        axes[1, 0].boxplot(steps_data, labels=policy_labels)
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_ylabel('Steps Taken')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary bar chart
        avg_rewards = [np.mean(r) for r in rewards_data]
        std_rewards = [np.std(r) for r in rewards_data]
        x_pos = np.arange(len(policy_labels))
        axes[1, 1].bar(x_pos, avg_rewards, yerr=std_rewards, capsize=5)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(policy_labels, rotation=45, ha='right')
        axes[1, 1].set_title('Average Rewards (with std)')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {args.save_plot}")
        else:
            plt.show()
    
    print("\n" + "=" * 70)
    print("✓ Comparison completed!")
    print("=" * 70)
