# source https://github.com/sfujim/TD3_BC
# https://arxiv.org/abs/2106.06860
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple
import pickle

import distrax
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from tqdm.auto import trange
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import custom_envs_gym
from custom_envs_gym import SCAwareObservation
from utils import get_dataset
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import d4rl


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    project: str = "OfflineRSRL"
    """the W&B project to log to"""
    group: str = "TD3BC"
    """the W&B group to log to"""
    name: str = "TD3BC_SRM"
    """the name of the run, used for W&B tracking"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    dir: str = "runs"
    """the root directory of all experiments"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    eval_episodes_final: int = 0
    """the number of episodes to evaluate the agent at the end of training"""
    eval_interval: int = 10000
    """the interval of evaluation"""
    log_interval: int = 1000
    """the interval of logging"""
    data_dir: str = ""
    """the directory of the dataset if not using d4rl"""

    # Algorithm specific arguments
    env_id: str = "halfcheetah-medium-expert-v2"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    policy_learning_rate: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_learning_rate: float = 3e-4
    """the learning rate of the Q network optimizer"""
    n_quantiles: int = 50
    """the number of quantiles"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    reward_normalizer: float = 1.0
    """the reward normalizing factor of the extended state space"""
    extended_critic: bool = False
    """whether to use the extended state space for the critic"""
    extended_actor: bool = False
    """whether to use the extended state space for the actor"""
    h_frequency: int = 500
    """the timesteps it takes to update the function h"""
    tau_h: float = 1.0
    """target smoothing coefficient for h function"""
    h_update_start: int = 0
    """the timestep to start updating the h function"""
    alpha: float = 0.9
    """the weight of the BC loss"""
    normalize_state: bool = False
    """whether to normalize the state"""
    n_jitted_updates: int = 10
    """the number of jitted updates"""
    hidden_dims: Sequence[int] = (256, 256)
    """the hidden dimension of the networks"""
    risk_alpha: float = 1.0
    """the alpha parameter of cvar, or the nu parameter of dual power, or the lambda parameter of exponential risk measure"""
    risk_alphas: str = "1.0"
    """the alpha parameter of srm"""
    risk_weights: str = "1.0"
    """the weight parameter of srm"""
    risk_measure: str = "CVaR"
    """the risk measure to be used"""

    def __hash__(
        self,
    ):  # make args hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    add_layer_norm: bool = False
    layer_norm_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if self.add_layer_norm:  # Add layer norm after activation
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:  # Add activation after layer norm
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_quantiles: int

    @nn.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, self.n_quantiles), add_layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, self.n_quantiles), add_layer_norm=True)(x)
        return q1, q2


class TD3Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float = 1.0  # In D4RL, action is scaled to [-1, 1]

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = MLP((*self.hidden_dims, self.action_dim))(observation)
        action = self.max_action * jnp.tanh(action)  # scale to [-max_action, max_action]
        action = action + 1.001  # shift to (0.001, 2.001)
        action = action / jnp.sum(action, axis=-1, keepdims=True)  # normalize
        return action


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


# huber loss function
def huber(x, k=1.0):
    return jnp.where(jnp.abs(x) < k, 0.5 * jnp.power(x, 2), k * (jnp.abs(x) - 0.5 * k))


# note that this function is not calculating the true q value, it is like the advantage function
def function_h(quantiles, s, c, theta_0, mu, args):
    # Calculate Q-values
    batch_size = args.batch_size
    n_quantiles = args.n_quantiles
    # tau_hat = (2 * jnp.arange(args.n_quantiles, dtype=jnp.float32) + 1.0) / (2.0 * args.n_quantiles)
    s_batch = jnp.tile(jnp.reshape(s, (-1, 1)), (1, n_quantiles))  # (batch_size, n_quantiles)
    c_batch = jnp.tile(jnp.reshape(c, (-1, 1)), (1, n_quantiles))  # (batch_size, n_quantiles)
    spc_quantiles = s_batch + c_batch * quantiles  # (batch_size, n_quantiles)
    theta_0_batch = jnp.tile(jnp.reshape(theta_0, (1, -1)), (batch_size, 1))  # (batch_size, n_quantiles)
    # theta_tau_batch = jnp.tile(jnp.reshape(tau_hat * theta_0, (1, -1)), (batch_size, 1))  # (batch_size, n_quantiles)
    diff = jnp.expand_dims(spc_quantiles, -1).transpose(0, 2, 1) - jnp.expand_dims(theta_0_batch, -1)  # (batch_size, n_quantiles, n_quantiles)
    mindiff = jnp.minimum(0, diff)  # (batch_size, n_quantiles, n_quantiles)
    # theta_tau_p_mindiff = jnp.expand_dims(theta_tau_batch, -1) + mindiff
    mu_batch = jnp.tile(jnp.reshape(mu[:-1], (1, -1, 1)), (batch_size, 1, n_quantiles))  # (batch_size, n_quantiles, n_quantiles)
    mu_spc_quantiles = jnp.multiply(mu_batch, mindiff)  # (batch_size, n_quantiles, n_quantiles)
    q_values = mu_spc_quantiles.sum((1, 2)) / n_quantiles  # (batch_size,)
    q_values += mu[-1] * (quantiles.mean(1))
    return q_values


# Dual Power Risk Measures
def dual_power(x, alpha=None):
    return alpha * ((1 - x) ** (alpha - 1))


# CVaR Risk Measures
def CVaR(x, alpha=None):
    return (1 / alpha) * (1 - jnp.heaviside(x - alpha, 0))


# Weighted Sum of CVaR Risk Measures
def weighted_sum_of_cvar(x, alphas=None, weights=None):
    return jnp.sum(jnp.array([weights[i] * CVaR(x, alphas[i]) for i in range(len(alphas))]), axis=0)


# Exponential Risk Measures
def exponential_risk_measures(x, alpha=None):
    return (alpha * jnp.exp(-alpha * x)) / (1 - jnp.exp(-alpha))


# calculate mu function
def calculate_mu_function(phi_values):
    mu = jnp.zeros_like(phi_values)
    mu = mu.at[:-1].set(phi_values[:-1] - phi_values[1:])
    mu = mu.at[-1].set(phi_values[-1])
    return mu


def target_update(model: TrainState, target_model: TrainState, tau: float) -> TrainState:
    new_target_params = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params)
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(train_state: TrainState, loss_fn: Callable) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class TD3BCTrainer(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_actor: TrainState
    target_critic: TrainState
    max_action: float = 1.0

    def update_actor(
        agent,
        batch: Transition,
        rng: jax.random.PRNGKey,
        args: Args,
        theta_0: jnp.ndarray,
        mu: jnp.ndarray,
        obs_mean: jnp.ndarray,
        obs_std: jnp.ndarray,
    ) -> Tuple["TD3BCTrainer", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            predicted_action = agent.actor.apply_fn(actor_params, batch.observations if args.extended_actor else batch.observations[:, :-2])
            critic_params = jax.lax.stop_gradient(agent.critic.params)
            q_value, _ = agent.critic.apply_fn(
                critic_params, batch.observations if args.extended_critic else batch.observations[:, :-2], predicted_action
            )
            s = args.reward_normalizer * (batch.observations[:, -2] * (obs_std[-2]) + obs_mean[-2])  #  + 1e-5
            c = batch.observations[:, -1] * (obs_std[-1]) + obs_mean[-1]  #  + 1e-5
            q_value = function_h(q_value, s, c, theta_0, mu, args)  # (batch_size,)

            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean()) + 1e-5
            loss_lambda = args.alpha / mean_abs_q

            bc_loss = jnp.sqrt(jnp.square(predicted_action - batch.actions).mean())
            loss_actor = (-1.0 * q_value.mean() / mean_abs_q) * args.alpha + bc_loss * (1.0 - args.alpha)
            return loss_actor

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(
        agent, batch: Transition, rng: jax.random.PRNGKey, args: Args, tau_hat_reshaped: jnp.ndarray
    ) -> Tuple["TD3BCTrainer", jnp.ndarray]:
        def critic_loss_fn(critic_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            q_pred_1, q_pred_2 = agent.critic.apply_fn(
                critic_params, batch.observations if args.extended_critic else batch.observations[:, :-2], batch.actions
            )
            target_next_action = agent.target_actor.apply_fn(
                agent.target_actor.params, batch.next_observations if args.extended_actor else batch.next_observations[:, :-2]
            )
            policy_noise = args.policy_noise * agent.max_action * jax.random.normal(rng, batch.actions.shape)
            target_next_action = target_next_action + policy_noise.clip(-args.noise_clip, args.noise_clip)
            target_next_action = target_next_action.clip(-agent.max_action, agent.max_action)
            q_next_1, q_next_2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.next_observations if args.extended_critic else batch.next_observations[:, :-2], target_next_action
            )
            next_q = jnp.minimum(q_next_1, q_next_2)
            q_target = batch.rewards[..., None] + args.gamma * next_q * (1 - batch.dones[..., None])
            q_target = jax.lax.stop_gradient(q_target)
            diff1 = jnp.expand_dims(q_target, -1).transpose(0, 2, 1) - jnp.expand_dims(q_pred_1, -1)
            diff2 = jnp.expand_dims(q_target, -1).transpose(0, 2, 1) - jnp.expand_dims(q_pred_2, -1)
            value_loss_1 = (huber(diff1) * jnp.abs(tau_hat_reshaped - jnp.less(diff1, 0).astype(jnp.float32))).mean(2).sum(1).mean()
            value_loss_2 = (huber(diff2) * jnp.abs(tau_hat_reshaped - jnp.less(diff2, 0).astype(jnp.float32))).mean(2).sum(1).mean()
            value_loss = (value_loss_1 + value_loss_2).mean()
            return value_loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, critic_loss_fn)
        return agent._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(3,))
    def update_n_times(
        agent,
        data: Transition,
        rng: jax.random.PRNGKey,
        args: Args,
        theta_0: jnp.ndarray,
        mu: jnp.ndarray,
        obs_mean: jnp.ndarray,
        obs_std: jnp.ndarray,
        tau_hat_reshaped: jnp.ndarray,
    ) -> Tuple["TD3BCTrainer", Dict]:
        for _ in range(args.n_jitted_updates):  # we can jit for roop for static unroll
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch_idx = jax.random.randint(batch_rng, (args.batch_size,), 0, len(data.observations))
            batch: Transition = jax.tree.map(lambda x: x[batch_idx], data)
            agent, critic_loss = agent.update_critic(batch, critic_rng, args, tau_hat_reshaped)
            if _ % args.policy_frequency == 0:
                agent, actor_loss = agent.update_actor(batch, actor_rng, args, theta_0, mu, obs_mean, obs_std)
                new_target_critic = target_update(agent.critic, agent.target_critic, args.tau)
                new_target_actor = target_update(agent.actor, agent.target_actor, args.tau)
                agent = agent._replace(
                    target_critic=new_target_critic,
                    target_actor=new_target_actor,
                )
        return agent, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @jax.jit
    def get_actions(
        agent,
        params: flax.core.FrozenDict[str, Any],
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        action = agent.actor.apply_fn(params, obs)
        action = action.clip(-agent.max_action, agent.max_action)
        return action

    @jax.jit
    def update_h(agent, initial_state: np.ndarray, theta_0: jnp.ndarray):
        initial_action = agent.actor.apply_fn(agent.actor.params, initial_state if args.extended_actor else initial_state[:-2])
        theta_0_new = agent.critic.apply_fn(agent.critic.params, initial_state if args.extended_critic else initial_state[:-2], initial_action)[0]
        theta_0 = optax.incremental_update(theta_0_new, theta_0, args.tau_h)
        return theta_0


def create_trainer(observations: jnp.ndarray, actions: jnp.ndarray, args: Args) -> TD3BCTrainer:
    rng = jax.random.PRNGKey(args.seed)
    rng, critic_rng, actor_rng = jax.random.split(rng, 3)

    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = TD3Actor(
        hidden_dims=args.hidden_dims,
        action_dim=action_dim,
    )
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations if args.extended_actor else observations[:-2]),
        tx=optax.adam(learning_rate=args.policy_learning_rate),
    )
    # initialize target actor
    target_actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations if args.extended_actor else observations[:-2]),
        tx=optax.adam(learning_rate=args.policy_learning_rate),
    )
    # initialize critic
    critic_model = DoubleCritic(
        hidden_dims=args.hidden_dims,
        n_quantiles=args.n_quantiles,
    )
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations if args.extended_critic else observations[:-2], actions),
        tx=optax.adam(learning_rate=args.q_learning_rate),
    )
    # initialize target critic
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations if args.extended_critic else observations[:-2], actions),
        tx=optax.adam(learning_rate=args.q_learning_rate),
    )
    return TD3BCTrainer(
        actor=actor_train_state,
        critic=critic_train_state,
        target_actor=target_actor_train_state,
        target_critic=target_critic_train_state,
    )


def scale_action(action_space: gym.spaces.Box, action: np.ndarray) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action: Action to scale
    :return: Scaled action
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space: gym.spaces.Box, scaled_action: np.ndarray) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param scaled_action: Action to un-scale
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


def evaluate_policy(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    params: flax.core.FrozenDict[str, Any],
    env: gym.Env,
    eval_episodes: int,
    obs_mean: np.array,
    obs_std: np.array,
    seed: int,
    print_cvars: bool = False,
) -> float:  # D4RL specific
    observation, done = env.reset(seed=seed), False
    episode_returns = []
    for _ in trange(eval_episodes, desc="Evaluating Model"):
        episode_return = 0
        while not done:
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(params, observation if args.extended_actor else observation[:-2])
            # action = unscale_action(env.action_space, action)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        observation, done = env.reset(), False
        episode_returns.append(episode_return)
    episode_returns = env.get_normalized_score(np.array(episode_returns)) * 100.0
    if print_cvars:
        print("CVaR 0.02:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.02)]))
        print("CVaR 0.05:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.05)]))
        print("CVaR 0.10:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.10)]))
        print("CVaR 0.20:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.20)]))
        print("CVaR 0.25:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.25)]))
        print("CVaR 0.40:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.40)]))
        print("CVaR 0.50:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.50)]))
        print("CVaR 0.60:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.60)]))
        print("CVaR 0.80:", np.mean(episode_returns[episode_returns <= np.quantile(episode_returns, 0.80)]))
        print("Expectation:", np.mean(episode_returns))
    return episode_returns


def evaluate(path: str, eval_episodes: int, seed=1, print_cvars: bool = False, normalize: bool = False):
    # read config file
    args_path = f"{path}/args.pkl"
    with open(args_path, "rb") as file:
        args = pickle.load(file)
    env = SCAwareObservation(gym.make(args.env_id), args.gamma, args.reward_normalizer)
    _, obs_mean, obs_std, _, _ = get_dataset(env, args=args, extended_obs=True)
    obs, done = env.reset(seed=seed), False
    actor = TD3Actor(hidden_dims=args.hidden_dims, action_dim=np.prod(env.action_space.shape))
    qf = DoubleCritic(args.hidden_dims, args.n_quantiles)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    actor_params = actor.init(actor_key, obs if args.extended_actor else obs[:-2])
    qf_params = qf.init(qf_key, obs if args.extended_critic else obs[:-2], env.action_space.sample())
    # note: qf_params is not used in this script
    model_path = f"{path}/flax.model"
    with open(model_path, "rb") as f:
        (actor_params, qf_params) = flax.serialization.from_bytes((actor_params, qf_params), f.read())
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    episodic_returns = []
    episodic_discounted_returns = []
    for _ in trange(eval_episodes, desc="Evaluating Final Model", dynamic_ncols=True):
        episode_return, discounted_return, time_step = 0, 0, 0
        while not done:
            obs = (obs - obs_mean) / obs_std
            action = jax.device_get(actor.apply(actor_params, obs if args.extended_actor else obs[:-2]))
            # action = unscale_action(env.action_space, action)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            discounted_return += reward * (args.gamma**time_step)
            time_step += 1
        obs, done = env.reset(), False
        episodic_returns += [episode_return]
        episodic_discounted_returns += [discounted_return]
    episodic_returns = np.array(episodic_returns)
    episodic_discounted_returns = np.array(episodic_discounted_returns)
    if print_cvars:
        print("CVaR 0.02:", np.mean(episodic_discounted_returns[episodic_discounted_returns <= np.quantile(episodic_discounted_returns, 0.02)]))
        print("CVaR 0.05:", np.mean(episodic_discounted_returns[episodic_discounted_returns <= np.quantile(episodic_discounted_returns, 0.05)]))
        print("CVaR 0.10:", np.mean(episodic_discounted_returns[episodic_discounted_returns <= np.quantile(episodic_discounted_returns, 0.10)]))
        print("CVaR 0.25:", np.mean(episodic_discounted_returns[episodic_discounted_returns <= np.quantile(episodic_discounted_returns, 0.25)]))
        print("CVaR 0.50:", np.mean(episodic_discounted_returns[episodic_discounted_returns <= np.quantile(episodic_discounted_returns, 0.50)]))
        print("Expectation:", np.mean(episodic_discounted_returns))
    if normalize:
        return 0, env.get_normalized_score(episodic_returns) * 100.0
    return episodic_returns, episodic_discounted_returns


def evaluate_policy_sequential(path, seed=1):
    """Runs a loaded policy through the entire environment data sequentially."""
    from datetime import timedelta

    # read config file
    args_path = f"{path}/args.pkl"
    with open(args_path, "rb") as file:
        args = pickle.load(file)
    env = SCAwareObservation(gym.make(args.env_id, random_start=False), args.gamma, args.reward_normalizer)
    _, obs_mean, obs_std, _, _ = get_dataset(env, args=args, extended_obs=True)
    obs, done = env.reset(seed=seed), False
    actor = TD3Actor(hidden_dims=args.hidden_dims, action_dim=np.prod(env.action_space.shape))
    qf = DoubleCritic(args.hidden_dims, args.n_quantiles)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    actor_params = actor.init(actor_key, obs if args.extended_actor else obs[:-2])
    qf_params = qf.init(qf_key, obs if args.extended_critic else obs[:-2], env.action_space.sample())
    # note: qf_params is not used in this script
    model_path = f"{path}/flax.model"
    with open(model_path, "rb") as f:
        (actor_params, qf_params) = flax.serialization.from_bytes((actor_params, qf_params), f.read())
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    portfolio_values = []
    portfolio_weights = []

    obs = env.reset()  # Reset to start of test data
    portfolio_values.append(env.initial_portfolio_value)

    # Loop until the environment signals the end of data or true termination
    # env._current_tick starts near window_size, ends at env._data_len

    # We loop step-by-step based on internal tick counter rather than fixed episode length
    for i in trange(env.get_env_info()["start_tick"], env.get_env_info()["data_len"]):
        obs = (obs - obs_mean) / obs_std
        action = jax.device_get(actor.apply(actor_params, obs if args.extended_actor else obs[:-2]))
        # action = unscale_action(env.action_space, action)

        obs, reward, terminated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        portfolio_weights.append(info["portfolio_weights"])

    return (
        np.array(portfolio_values),
        env.get_env_info()["return_dates"][env.get_env_info()["start_tick"] :].insert(0, env.get_env_info()["return_dates"][0] - timedelta(days=1)),
        np.array(portfolio_weights),
        env.asset_symbols,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.name}__{args.seed}__{int(time.time())}"
    run_name += f"__{args.n_quantiles}__{args.risk_alpha}__{args.risk_measure}"
    if args.risk_measure == "WSCVaR":
        run_name += f"__{args.risk_alphas}__{args.risk_weights}"
    if args.track:
        import wandb

        wandb.init(
            config=vars(args),
            project=args.project,
            group=args.group,
            name=args.name,
            id=int(time.time()),
            sync_tensorboard=True,
            # monitor_gym=True,
            # save_code=True,
        )
    writer = SummaryWriter(f"{args.dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    env = SCAwareObservation(gym.make(args.env_id), args.gamma, args.reward_normalizer)
    rng = jax.random.PRNGKey(args.seed)
    dataset, obs_mean, obs_std, theta_0, initial_state = get_dataset(env, args=args, extended_obs=True)
    print(f"Dataset size: {len(dataset.observations)}")
    initial_state = env.reset()
    initial_state = (initial_state - obs_mean) / (obs_std)  # + 1e-5)
    # create agent
    example_batch: Transition = jax.tree.map(lambda x: x[0], dataset)
    agent = create_trainer(example_batch.observations, example_batch.actions, args)

    tau_hat = (2 * jnp.arange(args.n_quantiles, dtype=jnp.float32) + 1.0) / (2.0 * args.n_quantiles)
    tau_hat_reshaped = jnp.tile(tau_hat[None, :, None], (args.batch_size, 1, args.n_quantiles))

    taus = jnp.linspace(0.0, 1.0, args.n_quantiles + 1)  # (n_quantiles+1,)
    if args.risk_measure == "CVaR":
        phi_values = CVaR(taus, alpha=args.risk_alpha)
    elif args.risk_measure == "Dual":
        phi_values = dual_power(taus, alpha=args.risk_alpha)
    elif args.risk_measure == "WSCVaR":
        alphas = jnp.array(args.risk_alphas.split(","), dtype=jnp.float32)
        weights = jnp.array(args.risk_weights.split(","), dtype=jnp.float32)
        assert len(alphas) == len(weights), "The number of alphas and weights should be the same."
        phi_values = weighted_sum_of_cvar(taus, alphas=alphas, weights=weights)
    elif args.risk_measure == "Exp":
        phi_values = exponential_risk_measures(taus, alpha=args.risk_alpha)
    else:
        raise ValueError("The risk measure is not defined.")
    mu = calculate_mu_function(phi_values)

    theta_0 = jax.lax.stop_gradient(theta_0)
    mu = jax.lax.stop_gradient(mu)
    tau_hat_reshaped = jax.lax.stop_gradient(tau_hat_reshaped)
    initial_state = jax.lax.stop_gradient(initial_state)

    num_steps = args.total_timesteps // args.n_jitted_updates
    for i in trange(num_steps + 1, smoothing=0.1, dynamic_ncols=True, desc="Training"):
        rng, update_rng = jax.random.split(rng)
        agent, update_info = agent.update_n_times(
            dataset,
            update_rng,
            args,
            theta_0,
            mu,
            obs_mean,
            obs_std,
            tau_hat_reshaped,
        )  # update parameters
        for k, v in update_info.items():
            writer.add_scalar(f"charts/{k}", v, i)
        if (i >= args.h_update_start) and (i % args.h_frequency == 0):
            theta_0_new = agent.update_h(initial_state, theta_0)
            writer.add_scalar("charts/srm", jnp.abs(theta_0_new).mean(), i)
            writer.add_scalar("charts/h_diff", jnp.abs(theta_0_new - theta_0).mean(), i)
            h_diff = jnp.abs(theta_0_new - theta_0).mean()
            if args.track:
                h_diff_metrics = {f"{args.env_id}/h_abs_diff": h_diff}
                wandb.log(h_diff_metrics, step=i)
            theta_0 = theta_0_new
        if args.track and (i % args.log_interval == 0):
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if (args.eval_episodes > 0) and (i % args.eval_interval == 0):
            policy_fn = agent.get_actions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cpu_policy_fn = jax.jit(agent.get_actions, backend="cpu")
            cpu_params = jax.device_put(agent.actor.params, jax.devices("cpu")[0])
            cpu_obs_mean = jax.device_put(obs_mean, jax.devices("cpu")[0])
            cpu_obs_std = jax.device_put(obs_std, jax.devices("cpu")[0])
            with jax.default_device(jax.devices("cpu")[0]):
                normalized_score = evaluate_policy(
                    cpu_policy_fn,
                    cpu_params,
                    env,
                    eval_episodes=args.eval_episodes,
                    obs_mean=cpu_obs_mean,
                    obs_std=cpu_obs_std,
                    seed=args.seed,
                    print_cvars=False,
                )
            if args.track:
                eval_metrics = {f"{args.env_id}/normalized_score": normalized_score}
                wandb.log(eval_metrics, step=i)
            writer.add_scalar("charts/CVaR 0.1", np.mean(normalized_score[normalized_score <= np.quantile(normalized_score, 0.1)]), i)
            writer.add_scalar("charts/CVaR 0.2", np.mean(normalized_score[normalized_score <= np.quantile(normalized_score, 0.2)]), i)
            writer.add_scalar("charts/Mean", np.mean(normalized_score), i)

    path = f"{args.dir}/{run_name}"
    if args.save_model:
        os.makedirs(path, exist_ok=True)
        model_path = f"{args.dir}/{run_name}/flax.model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        agent.actor.params,
                        agent.critic.params,
                    ]
                )
            )
        config_path = f"{path}/args.pkl"
        with open(config_path, "wb") as file:
            pickle.dump(args, file)
        print(f"model saved to {model_path}")
    # final evaluation
    if args.eval_episodes_final > 0:
        with jax.default_device(jax.devices("cpu")[0]):
            episodic_returns, episodic_discounted_returns = evaluate(
                path,
                eval_episodes=args.eval_episodes_final,
                seed=args.seed,
            )
        print("Final Evaluation Score:", np.mean(episodic_discounted_returns))
    if args.track:
        wandb.log({f"{args.env_id}/final_normalized_score": np.mean(episodic_discounted_returns)})
        wandb.finish()
