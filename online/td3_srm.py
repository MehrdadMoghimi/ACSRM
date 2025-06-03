# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb.analytics.sentry")

import os
import sys
import random
import time
from dataclasses import dataclass
import pickle
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
import chex
import tqdm
import flashbax as fbx
from flashbax.vault import Vault
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# jax.config.update("jax_platform_name", "cpu")
import custom_envs_gym
from custom_envs_gym import SCAwareObservation


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    project: str = "OnlineRSRL"
    """the W&B project to log to"""
    group: str = "TD3"
    """the W&B group to log to"""
    name: str = "TD3_SRM"
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

    # Algorithm specific arguments
    env_id: str = "Hopper-v3"
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
    """the scale of policy noise"""
    exploration_noise: float = 0.1
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
    risk_alpha: float = 1.0
    """the alpha parameter of cvar, or the nu parameter of dual power, or the lambda parameter of exponential risk measure"""
    risk_alphas: str = "1.0"
    """the alpha parameter of srm"""
    risk_weights: str = "1.0"
    """the weight parameter of srm"""
    risk_measure: str = "CVaR"
    """the risk measure to be used"""


def make_env_sc_aware(env_id, seed, idx, capture_video, run_name, gamma, normalizer):
    def thunk():
        if capture_video and idx == 0:
            env = SCAwareObservation(gym.make(env_id, render_mode="rgb_array"), gamma, normalizer)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = SCAwareObservation(gym.make(env_id), gamma, normalizer)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    n_quantiles: int
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_quantiles)(x)
        return x


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


def default_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_in", "uniform")


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    n_units: int = 256
    init_scale: float = 1e-1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units, kernel_init=default_init(self.init_scale))(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units, kernel_init=default_init(self.init_scale))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=default_init(self.init_scale))(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


def evaluate(path: str, eval_episodes: int, capture_video: bool = False, seed=1, print_cvars: bool = False, normalize: bool = False):
    # read config file
    args_path = f"{path}/args.pkl"
    with open(args_path, "rb") as file:
        args = pickle.load(file)

    envs = gym.vector.SyncVectorEnv([make_env_sc_aware(args.env_id, seed, 0, capture_video, "", args.gamma, args.reward_normalizer)])
    obs = envs.reset(seed=seed)
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    qf = QNetwork(n_quantiles=args.n_quantiles)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    actor_params = actor.init(actor_key, obs if args.extended_actor else obs[:, :-2])
    qf_params = qf.init(qf_key, obs if args.extended_critic else obs[:, :-2], envs.action_space.sample())
    # note: qf_params is not used in this script
    model_path = f"{path}/flax.model"
    with open(model_path, "rb") as f:
        (actor_params, qf_params) = flax.serialization.from_bytes((actor_params, qf_params), f.read())
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    episodic_returns = []
    episodic_discounted_returns = []
    discounted_return, time_step = 0, 0
    for _ in tqdm.tqdm(range(eval_episodes), desc="Evaluating Final Model", dynamic_ncols=True):
        discounted_return, time_step = 0, 0
        done = False
        while not done:
            actions = actor.apply(actor_params, obs if args.extended_actor else obs[:, :-2])
            actions = np.array([jax.device_get(actions)[0]])
            obs, rewards, dones, infos = envs.step(actions)
            discounted_return += rewards[0] * (args.gamma**time_step)
            time_step += 1
            done = dones[0]
        episodic_returns += [infos[0]["episode"]["r"]]
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
        return 0, envs.call("get_normalized_score", episodic_returns)[0] * 100.0
    return episodic_returns, episodic_discounted_returns


def create_expert_buffer(path, buffer_size, batch_size, seed=0, extended=True):
    args_path = f"{path}/args.pkl"
    with open(args_path, "rb") as file:
        args = pickle.load(file)

    envs = gym.vector.SyncVectorEnv([make_env_sc_aware(args.env_id, 0, 0, False, "", args.gamma, args.reward_normalizer)])
    obs = envs.reset(seed=seed)
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    qf = QNetwork(args.n_quantiles)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    actor_params = actor.init(actor_key, obs)
    qf_params = qf.init(qf_key, obs, envs.action_space.sample())
    # note: qf_params is not used in this script
    model_path = f"{path}/flax.model"
    with open(model_path, "rb") as f:
        (actor_params, qf_params) = flax.serialization.from_bytes((actor_params, qf_params), f.read())
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)
    buffer = fbx.make_flat_buffer(
        max_length=buffer_size,
        min_length=batch_size,
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batch_size=None,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    obs = envs.reset(seed=0)
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    obs, rewards, dones, _ = envs.step(actions)
    timestep = TimeStep(obs=obs[0] if extended else obs[0, :-2], action=actions[0], reward=rewards[0], done=dones[0])
    buffer_state = buffer.init(timestep)
    v = Vault(vault_name="expert_buffer", experience_structure=buffer_state.experience, rel_dir=path)
    for _ in tqdm.tqdm(range(buffer_size), desc="Filling Expert Replay Buffer"):
        actions = actor.apply(actor_params, obs if args.extended_actor else obs[:, :-2])
        actions = np.array([jax.device_get(actions)[0]])
        next_obs, rewards, dones, _ = envs.step(actions)
        timestep = TimeStep(obs=obs[0] if extended else obs[0, :-2], action=actions[0], reward=rewards[0], done=dones[0])
        buffer_state = buffer.add(buffer_state, timestep)
        obs = next_obs
    return v.write(buffer_state, (0, buffer_size))


def create_random_buffer(path, buffer_size, batch_size, extended=True):
    args_path = f"{path}/args.pkl"
    with open(args_path, "rb") as file:
        args = pickle.load(file)

    envs = gym.vector.SyncVectorEnv([make_env_sc_aware(args.env_id, 0, 0, False, "", args.gamma, args.reward_normalizer)])
    buffer = fbx.make_flat_buffer(
        max_length=buffer_size,
        min_length=batch_size,
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batch_size=None,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    obs = envs.reset(seed=0)
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    obs, rewards, dones, _ = envs.step(actions)
    timestep = TimeStep(obs=obs[0] if extended else obs[0, :-2], action=actions[0], reward=rewards[0], done=dones[0])
    buffer_state = buffer.init(timestep)
    v = Vault(vault_name="random_buffer", experience_structure=buffer_state.experience, rel_dir=path)
    for _ in tqdm.tqdm(range(buffer_size), desc="Filling Random Replay Buffer"):
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        next_obs, rewards, dones, _ = envs.step(actions)
        timestep = TimeStep(obs=obs[0] if extended else obs[0, :-2], action=actions[0], reward=rewards[0], done=dones[0])
        buffer_state = buffer.add(buffer_state, timestep)
        obs = next_obs
    return v.write(buffer_state, (0, buffer_size))


@jax.jit
def soft_update(tau: float, qf1_state: TrainState) -> TrainState:
    qf1_state = qf1_state.replace(target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, tau))
    return qf1_state


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env_sc_aware(args.env_id, args.seed, 0, args.capture_video, run_name, args.gamma, args.reward_normalizer)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs.single_observation_space.dtype = np.float32
    # INIT BUFFER
    buffer = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_sequences=False,
        add_batch_size=None,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    obs = envs.reset(seed=args.seed)
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    obs, rewards, dones, _ = envs.step(actions)
    timestep = TimeStep(obs=obs[0], action=actions[0], reward=rewards[0], done=dones[0])
    buffer_state = buffer.init(timestep)
    if args.save_model:
        v = Vault(vault_name="buffer", experience_structure=buffer_state.experience, rel_dir=f"{args.dir}/{run_name}")

    tau_hat = (2 * jnp.arange(args.n_quantiles) + 1) / (2.0 * args.n_quantiles)
    tau_hat_reshaped = jnp.tile(tau_hat[None, :, None], (args.batch_size, 1, args.n_quantiles))
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs if args.extended_actor else obs[:, :-2]),
        target_params=actor.init(actor_key, obs if args.extended_actor else obs[:, :-2]),
        tx=optax.adam(learning_rate=args.policy_learning_rate),
    )
    qf = QNetwork(n_quantiles=args.n_quantiles)
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs if args.extended_critic else obs[:, :-2], envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs if args.extended_critic else obs[:, :-2], envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.q_learning_rate),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs if args.extended_critic else obs[:, :-2], envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs if args.extended_critic else obs[:, :-2], envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.q_learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

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

    initial_state = envs.reset(seed=args.seed)
    initial_action = actor.apply(actor_state.params, initial_state if args.extended_actor else initial_state[:, :-2])
    theta_0 = qf.apply(qf1_state.params, initial_state if args.extended_critic else initial_state[:, :-2], initial_action)[0]

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key = jax.random.split(key, 2)
        clipped_noise = (
            jnp.clip((jax.random.normal(noise_key, actions.shape) * args.policy_noise), -args.noise_clip, args.noise_clip) * actor.action_scale
        )
        next_state_actions = jnp.clip(
            actor.apply(actor_state.target_params, next_observations if args.extended_actor else next_observations[:, :-2]) + clipped_noise,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )
        qf1_next_target = qf.apply(
            qf1_state.target_params, next_observations if args.extended_critic else next_observations[:, :-2], next_state_actions
        )
        qf2_next_target = qf.apply(
            qf2_state.target_params, next_observations if args.extended_critic else next_observations[:, :-2], next_state_actions
        )
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = rewards[..., None] + (1 - terminations[..., None]) * args.gamma * (min_qf_next_target)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations if args.extended_critic else observations[:, :-2], actions)
            diff = jnp.expand_dims(next_q_value, -1).transpose(0, 2, 1) - jnp.expand_dims(qf_a_values, -1)
            loss = (huber(diff) * jnp.abs(tau_hat_reshaped - jnp.less(diff, 0).astype(jnp.float32))).mean(2).sum(1)
            return loss.mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
        theta_0: jnp.ndarray,
    ):
        def actor_loss(params):
            actions = actor.apply(params, observations if args.extended_actor else observations[:, :-2])
            qf1_a_values = qf.apply(qf1_state.params, observations if args.extended_critic else observations[:, :-2], actions)
            actor_loss = -function_h(
                quantiles=qf1_a_values,
                s=args.reward_normalizer * observations[:, -2],
                c=observations[:, -1],
                mu=mu,
                theta_0=theta_0,
                args=args,
            ).mean()
            return actor_loss

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, (qf1_state, qf2_state), actor_loss_value, grads

    @jax.jit
    def update_h(actor_state: TrainState, qf1_state: TrainState, initial_state: np.ndarray, theta_0: jnp.ndarray):
        initial_action = actor.apply(actor_state.params, initial_state if args.extended_actor else initial_state[:, :-2])
        theta_0_new = qf.apply(qf1_state.params, initial_state if args.extended_critic else initial_state[:, :-2], initial_action)[0]
        theta_0 = optax.incremental_update(theta_0_new, theta_0, args.tau_h)
        return theta_0

    start_time = time.time()
    discounted_return, time_step = 0, 0
    for global_step in tqdm.tqdm(range(args.total_timesteps), smoothing=0.1, dynamic_ncols=True):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.apply(actor_state.params, obs if args.extended_actor else obs[:, :-2])
            actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(0, float(envs.single_action_space.high[0]) * args.exploration_noise, size=envs.single_action_space.shape)
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        discounted_return += rewards[0] * args.gamma**time_step
        time_step += 1
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if dones[0]:
            # print(f"global_step={global_step}, episodic_return={infos[0]['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos[0]["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos[0]["episode"]["l"], global_step)
            writer.add_scalar("charts/episodic_discounted_return", discounted_return, global_step)
            discounted_return, time_step = 0, 0

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        timestep = TimeStep(obs=obs[0], action=actions[0], reward=rewards[0], done=dones[0])
        buffer_state = buffer.add(buffer_state, timestep)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            learn_batch = buffer.sample(buffer_state, key).experience
            # jax.debug.print("learn_batch{}", learn_batch.first.obs)

            (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                learn_batch.first.obs,
                learn_batch.first.action,
                learn_batch.second.obs,
                learn_batch.first.reward.flatten(),
                learn_batch.first.done.flatten(),
                key,
            )
            if global_step % args.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value, grads = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    learn_batch.first.obs,
                    theta_0,
                )
            # update the target networks
            if global_step % args.target_network_frequency == 0:
                qf1_state = soft_update(args.tau, qf1_state)
                qf2_state = soft_update(args.tau, qf2_state)
                actor_state = soft_update(args.tau, actor_state)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                writer.add_scalar("grads/actor_grads", grads["params"]["Dense_2"]["kernel"].mean(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update the h function
            if global_step % args.h_frequency == 0:
                theta_0 = update_h(actor_state, qf1_state, initial_state, theta_0)

    path = f"{args.dir}/{run_name}"
    if args.save_model:
        model_path = f"{args.dir}/{run_name}/flax.model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                    ]
                )
            )
        config_path = f"{path}/args.pkl"
        with open(config_path, "wb") as file:
            pickle.dump(args, file)
        v.write(buffer_state, (0, args.buffer_size))
        print(f"model saved to {model_path}")
    if args.eval_episodes > 0:
        episodic_returns, episodic_discounted_returns = evaluate(
            path,
            eval_episodes=args.eval_episodes,
        )
        print(f"eval_return={np.mean(episodic_returns):.3f}, eval_discounted_return={np.mean(episodic_discounted_returns):.3f}")

    envs.close()
    writer.close()
