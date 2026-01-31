import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork


class Trajectory(struct.PyTreeNode):
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class PPO(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)
    tau: chex.Scalar = struct.field(pytree_node=True, default=0.6)
    def make_act(self, ts):
        def act(obs, rng):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs, rng, method="act")
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
        else:
            actor = GaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                **agent_kwargs,
            )

        critic = VNetwork(**agent_kwargs)
        return {"actor": actor, "critic": critic}

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])

        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        critic_params = self.critic.init(rng_critic, obs_ph)

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        return {"actor_ts": actor_ts, "critic_ts": critic_ts}

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_expectile_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample action
            unclipped_action, log_prob = self.actor.apply(
                ts.actor_ts.params, ts.last_obs, rng_action, method="action_log_prob"
            )
            value = self.critic.apply(ts.critic_ts.params, ts.last_obs)

            # Clip action
            if self.discrete:
                action = unclipped_action
            else:
                low = self.env.action_space(self.env_params).low
                high = self.env.action_space(self.env_params).high
                action = jnp.clip(unclipped_action, low, high)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                ts = ts.replace(obs_rms_state=obs_rms_state)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(
                    ts.rew_rms_state, reward, done
                )
                ts = ts.replace(rew_rms_state=rew_rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, unclipped_action, log_prob, reward, value, done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + self.gamma * next_value * (1 - transition.done)
                - transition.value
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def calculate_expectile_gae(self, trajectories, last_val):
        v_all = trajectories.value
        v_old = v_all # V_t     [T-1,B]
        v_tp1 = jnp.concatenate([v_all[1:], last_val[None, :]], axis=0)  # V_{t+1} [T-1,B]
        last_value = last_val  # V_T     [B]

        r = trajectories.reward  # r_t [T,B]

        d = (1.0 - trajectories.done) # [T,B]

        def step(G_next, xs):
            r_t, d_t, v_tp1_t = xs  # each [B]
            u = G_next - v_tp1_t
            pos = jnp.maximum(u, 0.0)  # (u)_+
            neg = jnp.maximum(-u, 0.0)  # (u)_-
            lam = self.gae_lambda / self.tau
            soft = v_tp1_t + lam * (self.tau * pos - (1.0 - self.tau) * neg)
            G_t = r_t + self.gamma * d_t * soft
            return G_t, G_t

        _, G_rev = jax.lax.scan(
            step,
            last_value,
            (r[::-1], d[::-1], v_tp1[::-1]),
        )

        returns = G_rev[::-1]  # [T,B]
        adv = jax.lax.stop_gradient(returns - v_old)
        return adv, returns

    def expectile_actor_loss_helper(self, adv):
        tau = self.tau
        weight = jnp.where(adv > 0, 2.0 * tau, 2.0 * (1.0 - tau))
        return weight

    def expectile_value_loss_helper(self, target, value, value_clipped):
        tau = self.tau
        delta1 = target - value
        delta2 = target - value_clipped

        w1 = jnp.where(delta1 > 0, 2.0 * tau, 2.0 * (1.0 - tau))
        w2 = jnp.where(delta2 > 0, 2.0 * tau, 2.0 * (1.0 - tau))

        return w1, w2
    def update_actor(self, ts, batch):
        def actor_loss_fn(params):
            log_prob, entropy = self.actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )

            # without norm
            # advantages = batch.advantages

            # correct tau
            weight = self.expectile_actor_loss_helper(advantages)
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(weight * pi_loss1, weight * pi_loss2).mean()
            return pi_loss - self.ent_coef * entropy

        grads = jax.grad(actor_loss_fn)(ts.actor_ts.params)
        return ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))

    def update_critic(self, ts, batch):
        def critic_loss_fn(params):
            value = self.critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)

            # expectile tau
            w1, w2 = self.expectile_value_loss_helper(batch.targets, value, value_pred_clipped)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(w1 * value_losses, w2 * value_losses_clipped).mean()
            return self.vf_coef * value_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))

    def update(self, ts, batch):
        ts = self.update_actor(ts, batch)
        ts = self.update_critic(ts, batch)
        return ts
