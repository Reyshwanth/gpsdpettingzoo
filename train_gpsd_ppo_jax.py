"""GPU-accelerated IPPO training for the GPSD environment using JAX.

Based on JaxMARL's IPPO baseline (PureJaxRL-style end-to-end JAX training).
The entire training loop—environment stepping, rollout collection, GAE,
and PPO updates—runs inside a single ``jax.jit``-compiled function,
enabling massive speedups on GPU / TPU.

Key features:
  - Parameter-shared IPPO (all agents share one network)
  - Fully JIT-compiled training loop (no Python overhead per step)
  - ``jax.vmap`` over parallel environments and random seeds
  - WandB logging via ``jax.experimental.io_callback``
  - Saves trained parameters to disk after training

Usage:
    python train_gpsd_ppo_jax.py                       # defaults
    python train_gpsd_ppo_jax.py --total-timesteps 2000000 --num-envs 64
    python train_gpsd_ppo_jax.py --track               # enable W&B

Requirements:
    pip install jax jaxlib flax optax distrax wandb matplotlib
"""

import argparse
import os
import time
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import matplotlib.pyplot as plt

from gpsd_jax import GPSDJAX, GPSDLogWrapper


# ===================================================================
# CLI Arguments
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="JAX-accelerated IPPO for GPSD")

    # General
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-seeds", type=int, default=1,
                   help="Number of independent seeds to vmap over")
    p.add_argument("--track", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="gpsd-jax")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--save-model", action="store_true", default=True)

    # Environment
    p.add_argument("--num-agents", type=int, default=5)
    p.add_argument("--cell-width", type=float, default=0.25)
    p.add_argument("--max-cycles", type=int, default=100)
    p.add_argument("--speed", type=float, default=0.2)
    p.add_argument("--r-c", type=float, default=0.5)
    p.add_argument("--cov-c", type=float, default=0.15)

    # PPO
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=64,
                   help="Number of parallel environments (per seed)")
    p.add_argument("--num-steps", type=int, default=128,
                   help="Rollout length per environment per update")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--activation", type=str, default="tanh",
                   choices=["tanh", "relu"])

    return p.parse_args()


# ===================================================================
# Network
# ===================================================================
class ActorCritic(nn.Module):
    """MLP actor-critic with parameter sharing (Flax / JAX)."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Shared trunk (larger than JaxMARL default to match torch version)
        actor = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0))(x)
        actor = act_fn(actor)
        actor = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0))(actor)
        actor = act_fn(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(actor)
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(x)
        critic = act_fn(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(critic)
        critic = act_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(value, axis=-1)


# ===================================================================
# Transition tuple (stored in rollout buffer)
# ===================================================================
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict  # pytree of arrays


# ===================================================================
# Batchify / unbatchify helpers
# ===================================================================
def batchify(x: dict, agent_list, num_actors):
    """Stack agent observations into (num_actors, obs_dim)."""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    """Reshape flat action vector back to per-agent dict."""
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# ===================================================================
# Training factory
# ===================================================================
def make_train(config):
    """Build the fully JIT-compiled training function."""

    # Create environment
    env = GPSDJAX(
        num_agents=config["NUM_AGENTS"],
        cell_width=config["CELL_WIDTH"],
        max_steps=config["MAX_CYCLES"],
        speed=config["SPEED"],
        r_c=config["R_C"],
        cov_c=config["COV_C"],
    )
    env = GPSDLogWrapper(env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (
            count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # ------------------------------------------------------------------
    def train(rng):
        # ---- Init network ----
        network = ActorCritic(
            env.action_space(env.agents[0]).n,
            activation=config["ACTIVATION"],
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # ---- Init environments (vmap over NUM_ENVS) ----
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # ---- Training loop ----
        def _update_step(runner_state, unused):
            # ---- Collect trajectories ----
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Batchify observations
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # Select action
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Unbatchify actions → per-agent dict for env.step
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                # Convert to int for discrete actions
                env_act = jax.tree.map(lambda x: x.squeeze().astype(jnp.int32), env_act)

                # Step environments
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                info = jax.tree.map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # ---- Compute GAE ----
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Explained variance: how well the value function explains the returns
            var_y = jnp.var(targets)
            explained_var = 1.0 - jnp.var(targets - traj_batch.value) / (var_y + 1e-8)

            # ---- PPO update ----
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Policy loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                                               1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "approx_kl": ((total_loss[1][3] - 1) - jnp.log(total_loss[1][3])).mean(),
                        "clip_frac": jnp.mean(
                            jnp.abs(total_loss[1][3] - 1.0) > config["CLIP_EPS"]
                        ),
                    }
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # ---- Build metric dict ----
            # Loss info: mean over epochs and minibatches
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # Episode metrics: only count completed episodes
            ep_done_mask = traj_batch.info["returned_episode"]  # (NUM_STEPS, NUM_ACTORS)
            ep_done_any = ep_done_mask.sum() > 0

            # Mean episode return/length (only from completed episodes)
            safe_count = jnp.maximum(ep_done_mask.sum(), 1.0)
            mean_ep_return = (
                (traj_batch.info["returned_episode_returns"] * ep_done_mask).sum()
                / safe_count
            )
            mean_ep_length = (
                (traj_batch.info["returned_episode_lengths"] * ep_done_mask).sum()
                / safe_count
            )

            # Environment metrics: mean over all steps and actors
            metric = {
                # Episode stats (filtered by completed episodes)
                "episode/return_mean": mean_ep_return,
                "episode/length_mean": mean_ep_length,
                "episode/num_completed": ep_done_mask.sum(),
                # Coverage metrics (averaged over rollout)
                "env/coverage_ratio": traj_batch.info["coverage_ratio"].mean(),
                "env/num_pois_covered": traj_batch.info["num_pois_covered"].mean(),
                "env/all_pois_covered": traj_batch.info["all_pois_covered"].mean(),
                # Covariance metrics
                "env/mean_cov_trace": traj_batch.info["mean_cov_trace"].mean(),
                "env/max_cov_trace": traj_batch.info["max_cov_trace"].mean(),
                "env/agents_above_cov_thresh": traj_batch.info["num_agents_above_cov_thresh"].mean(),
                # Zone metrics
                "env/num_agents_in_zone": traj_batch.info["num_agents_in_zone"].mean(),
                # Reward breakdown
                "reward/mean_local": traj_batch.info["mean_local_reward"].mean(),
                "reward/coverage": traj_batch.info["coverage_reward"].mean(),
                "reward/mean_total": traj_batch.reward.mean(),
                # PPO losses
                "loss/total": loss_info["total_loss"],
                "loss/actor": loss_info["actor_loss"],
                "loss/critic": loss_info["critic_loss"],
                "loss/entropy": loss_info["entropy"],
                "loss/approx_kl": loss_info["approx_kl"],
                "loss/clip_frac": loss_info["clip_frac"],
                # Explained variance
                "loss/explained_var": explained_var,
                # Learning rate (from schedule)
                "lr": linear_schedule(train_state.step)
                    if config["ANNEAL_LR"] else config["LR"],
            }

            # ---- W&B callback ----
            if config.get("TRACK", False):
                def callback(metric):
                    import wandb
                    wandb.log({
                        k: float(v) for k, v in metric.items()
                    })
            else:
                def callback(metric):
                    return

            jax.experimental.io_callback(callback, None, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ===================================================================
# Main
# ===================================================================
def main():
    args = parse_args()

    config = {
        "NUM_AGENTS": args.num_agents,
        "CELL_WIDTH": args.cell_width,
        "MAX_CYCLES": args.max_cycles,
        "SPEED": args.speed,
        "R_C": args.r_c,
        "COV_C": args.cov_c,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "LR": args.lr,
        "ANNEAL_LR": args.anneal_lr,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "NUM_MINIBATCHES": args.num_minibatches,
        "UPDATE_EPOCHS": args.update_epochs,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ACTIVATION": args.activation,
        "SEED": args.seed,
        "NUM_SEEDS": args.num_seeds,
        "TRACK": args.track,
    }

    print("=" * 70)
    print("  GPSD JAX-Accelerated IPPO Training")
    print("=" * 70)
    print(f"  Device       : {jax.devices()}")
    print(f"  Num agents   : {config['NUM_AGENTS']}")
    print(f"  Num envs     : {config['NUM_ENVS']}")
    print(f"  Num seeds    : {config['NUM_SEEDS']}")
    print(f"  Total steps  : {config['TOTAL_TIMESTEPS']:,}")

    # Compute derived values for display
    _env = GPSDJAX(num_agents=config["NUM_AGENTS"], cell_width=config["CELL_WIDTH"])
    num_actors = _env.num_agents * config["NUM_ENVS"]
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    print(f"  Num actors   : {num_actors}")
    print(f"  Num updates  : {num_updates}")
    print(f"  Obs dim      : {_env.obs_dim}")
    print(f"  Num POIs     : {_env.num_pois}")
    print("=" * 70)

    # ---- W&B ----
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=f"gpsd_jax_{args.seed}_{int(time.time())}",
        )

    # ---- Build & JIT-compile training function ----
    t0 = time.time()
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    print(f"JIT compilation starting...")

    # ---- Run training (vmap over seeds) ----
    rng = jax.random.PRNGKey(args.seed)
    if args.num_seeds > 1:
        rngs = jax.random.split(rng, args.num_seeds)
        out = jax.vmap(train_jit)(rngs)
    else:
        out = train_jit(rng)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Training complete!  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    total_steps = config["TOTAL_TIMESTEPS"] * config["NUM_SEEDS"]
    print(f"  Effective SPS: {total_steps / elapsed:,.0f}")
    print(f"{'=' * 70}")

    # ---- Save model ----
    if args.save_model:
        os.makedirs("runs_jax", exist_ok=True)
        run_name = f"gpsd_jax_{args.seed}_{int(time.time())}"

        if args.num_seeds > 1:
            # Save first seed's params
            params = jax.tree.map(lambda x: x[0], out["runner_state"][0].params)
        else:
            params = out["runner_state"][0].params

        import pickle
        model_path = f"runs_jax/{run_name}_params.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(jax.device_get(params), f)
        print(f"  Model saved → {model_path}")

    # ---- Plot metrics ----
    try:
        metrics = out["metrics"]
        if args.num_seeds > 1:
            metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle("GPSD JAX IPPO Training", fontsize=14)

        # Row 0 — environment metrics
        axes[0, 0].plot(np.array(metrics["episode/return_mean"]))
        axes[0, 0].set_title("Episode Return (mean)")
        axes[0, 0].set_xlabel("Updates")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(np.array(metrics["env/coverage_ratio"]))
        axes[0, 1].set_title("Coverage Ratio")
        axes[0, 1].set_xlabel("Updates")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(np.array(metrics["env/mean_cov_trace"]), label="mean")
        axes[0, 2].plot(np.array(metrics["env/max_cov_trace"]), label="max", linestyle="--")
        axes[0, 2].axhline(y=0.15, color="r", linestyle=":", alpha=0.6, label="threshold")
        axes[0, 2].set_title("Cov Trace (mean / max)")
        axes[0, 2].set_xlabel("Updates")
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].grid(True, alpha=0.3)

        # Row 1 — reward & losses
        axes[1, 0].plot(np.array(metrics["reward/mean_total"]), label="total")
        axes[1, 0].plot(np.array(metrics["reward/coverage"]), label="coverage")
        axes[1, 0].plot(np.array(metrics["reward/mean_local"]), label="local")
        axes[1, 0].set_title("Reward Breakdown")
        axes[1, 0].set_xlabel("Updates")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(np.array(metrics["loss/actor"]), label="actor")
        axes[1, 1].plot(np.array(metrics["loss/critic"]), label="critic")
        axes[1, 1].set_title("Actor & Critic Loss")
        axes[1, 1].set_xlabel("Updates")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(np.array(metrics["loss/entropy"]))
        axes[1, 2].set_title("Entropy")
        axes[1, 2].set_xlabel("Updates")
        axes[1, 2].grid(True, alpha=0.3)

        # Row 2 — PPO diagnostics
        ev = np.array(metrics["loss/explained_var"])
        axes[2, 0].plot(ev)
        axes[2, 0].axhline(y=0.0, color="r", linestyle=":", alpha=0.6)
        axes[2, 0].axhline(y=1.0, color="g", linestyle=":", alpha=0.6)
        axes[2, 0].set_title("Explained Variance")
        axes[2, 0].set_xlabel("Updates")
        axes[2, 0].set_ylim(-1.05, 1.05)
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(np.array(metrics["loss/approx_kl"]))
        axes[2, 1].set_title("Approx KL Divergence")
        axes[2, 1].set_xlabel("Updates")
        axes[2, 1].grid(True, alpha=0.3)

        axes[2, 2].plot(np.array(metrics["loss/clip_frac"]))
        axes[2, 2].set_title("Clip Fraction")
        axes[2, 2].set_xlabel("Updates")
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"runs_jax/gpsd_jax_training.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Plot saved  → {plot_path}")
    except Exception as e:
        print(f"  Warning: Could not save plot: {e}")

    if args.track:
        import wandb
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
