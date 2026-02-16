"""CleanRL PPO training script for the GPSD (GPS-Denied Coverage) environment.

Adapted from the CleanRL PPO implementation for PettingZoo multi-agent environments.
Reference: https://pettingzoo.farama.org/tutorials/cleanrl/advanced_PPO/

The GPSD environment is a cooperative multi-agent coverage problem where agents
navigate into a GPS-denied zone to cover points of interest while managing
position uncertainty (EKF covariance). All agents share a parameter-shared
policy trained with PPO.

Key differences from the Atari CleanRL tutorial:
  - Vector observations (1D) instead of images → MLP network, not CNN
  - 5 agents by default (not 2) → num_envs must be a multiple of num_agents
  - No Atari-specific preprocessing (frame stacking, colour reduction, etc.)

Usage:
    python train_gpsd_ppo.py                        # defaults
    python train_gpsd_ppo.py --total-timesteps 500000 --num-envs 20
    python train_gpsd_ppo.py --track --wandb-project-name gpsd  # W&B logging

Authors: Adapted from Costa (https://github.com/vwxyzjn) and Elliot (https://github.com/elliottower)
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

# ---------------------------------------------------------------------------
# The local PettingZoo (with GPSD) should be installed in editable mode:
#   pip install -e PettingZoo/
# ---------------------------------------------------------------------------
from pettingzoo.mpe.gpsd.gpsd import parallel_env as make_gpsd_parallel_env


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="CleanRL PPO for GPSD environment")

    # --- General ---
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="gpsd-ppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture a video of the trained agent after training")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save the final trained model")

    # --- GPSD environment parameters ---
    parser.add_argument("--num-agents", type=int, default=5,
        help="number of agents (N_a) in the GPSD environment")
    parser.add_argument("--cell-width", type=float, default=0.25,
        help="cell width for the POI grid in the GPS denied zone")
    parser.add_argument("--max-cycles", type=int, default=100,
        help="max steps per episode in the GPSD environment")
    parser.add_argument("--speed", type=float, default=0.1,
        help="constant forward speed of agents")
    parser.add_argument("--r-c", type=float, default=0.3,
        help="communication/coverage radius")
    parser.add_argument("--cov-c", type=float, default=0.015,
        help="coverage covariance trace threshold")

    # --- PPO hyperparameters ---
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
        help="total timesteps of the experiment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=40,
        help="the number of parallel environment slots (must be a multiple of --num-agents)")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggles whether to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args = parser.parse_args()

    # Derived values
    assert args.num_envs % args.num_agents == 0, (
        f"--num-envs ({args.num_envs}) must be a multiple of --num-agents ({args.num_agents})"
    )
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialisation (CleanRL convention)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Agent (parameter-shared across all GPSD agents)
# ---------------------------------------------------------------------------
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
# Environment factory
# ---------------------------------------------------------------------------
def make_env(args):
    """Create a vectorised GPSD environment compatible with CleanRL's PPO loop.

    Pipeline:
        parallel_env  →  pettingzoo_env_to_vec_env  →  concat_vec_envs
    Each parallel env contributes `num_agents` slots to the vector env, so we
    create  num_envs // num_agents  independent copies.
    """
    env = make_gpsd_parallel_env(
        N_a=args.num_agents,
        cell_width=args.cell_width,
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
    """Record a video of the trained policy playing the GPSD environment."""
    try:
        import imageio
    except ImportError:
        print("imageio not installed – skipping video recording.")
        return

    print("Recording video of trained policy...")
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
                actions, _, _, _ = agent_model.get_action_and_value(obs_tensor)

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
        imageio.mimwrite(video_path, frames, fps=30)
        print(f"Saved video → {video_path}")
    else:
        print("No frames captured – skipping video save.")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    args = parse_args()
    run_name = f"gpsd__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(f"\n{'='*70}")
    print(f"  GPSD PPO Training  —  {run_name}")
    print(f"{'='*70}")
    print(args)
    print()

    # --- Weights & Biases ---
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
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
    print(f"Observation dim : {obs_dim}")
    print(f"Action dim      : {act_dim}")
    print(f"Num vec envs    : {args.num_envs}  "
          f"({args.num_envs // args.num_agents} games × {args.num_agents} agents)")
    print(f"Batch size      : {args.batch_size}")
    print(f"Minibatch size  : {args.minibatch_size}")
    print()

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # ------------------------------------------------------------------
    # Agent & optimiser
    # ------------------------------------------------------------------
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # ------------------------------------------------------------------
    # Rollout storage
    # ------------------------------------------------------------------
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
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
    next_termination = torch.zeros(args.num_envs).to(device)
    next_truncation = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(f"Number of PPO updates: {num_updates}\n")

    for update in range(1, num_updates + 1):
        # --- Learning rate annealing ---
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===============================================================
        # Rollout phase – collect experience
        # ===============================================================
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation

            # Action selection
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Environment step
            next_obs, reward, termination, truncation, info = envs.step(
                action.cpu().numpy()
            )
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_termination = torch.Tensor(termination).to(device)
            next_truncation = torch.Tensor(truncation).to(device)

            # --- Log episodic returns ---
            # SuperSuit's vectorised envs return info as a list of dicts.
            # When an episode ends the wrapper inserts an "episode" key.
            if isinstance(info, (list, tuple)):
                for idx, item in enumerate(info):
                    if isinstance(item, dict) and "episode" in item:
                        agent_idx = idx % args.num_agents
                        writer.add_scalar(
                            f"charts/episodic_return_agent{agent_idx}",
                            item["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar(
                            f"charts/episodic_length_agent{agent_idx}",
                            item["episode"]["l"],
                            global_step,
                        )
                        if agent_idx == 0:
                            print(
                                f"  update={update}/{num_updates}  "
                                f"global_step={global_step}  "
                                f"agent_{agent_idx}_return={item['episode']['r']:.2f}"
                            )

        # ===============================================================
        # Advantage estimation (GAE)
        # ===============================================================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        # Flatten the rollout batch
        # ===============================================================
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ===============================================================
        # Optimisation phase – PPO update
        # ===============================================================
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Approx KL: http://joschu.net/blog/kl-approx.html
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

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # ===============================================================
        # Logging
        # ===============================================================
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if update % 10 == 0 or update == num_updates:
            print(
                f"  [Update {update:>4}/{num_updates}]  "
                f"step={global_step:>8}  "
                f"SPS={sps:>5}  "
                f"pg_loss={pg_loss.item():+.4f}  "
                f"v_loss={v_loss.item():.4f}  "
                f"entropy={entropy_loss.item():.4f}  "
                f"explained_var={explained_var:.3f}"
            )

    # ==================================================================
    # Post-training
    # ==================================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Training complete!  Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Final SPS: {int(global_step / elapsed)}")
    print(f"{'='*70}")

    # --- Save model ---
    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/gpsd_ppo_agent.pt"
        torch.save({
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        }, model_path)
        print(f"  Model saved → {model_path}")

    # --- Record video ---
    if args.capture_video:
        record_video(args, agent, run_name)

    envs.close()
    writer.close()
    print("Done.")
