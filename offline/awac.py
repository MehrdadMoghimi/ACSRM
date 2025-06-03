# source https://github.com/ikostrikov/jaxrl
# https://arxiv.org/abs/2006.09359
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
    group: str = "AWAC"
    """the W&B group to log to"""
    name: str = "AWAC"
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
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2
    """the frequency of updates for the target nerworks"""
    beta: float = 1.0
    """the lagrange multiplier for the policy constraint"""
    normalize_state: bool = False
    """whether to normalize the state"""
    n_jitted_updates: int = 10
    """the number of jitted updates"""
    critic_hidden_dims: Sequence[int] = (256, 256)
    """the hidden dimension of the critic networks"""
    actor_hidden_dims: Sequence[int] = (256, 256)
    """the hidden dimension of the actor network"""

    def __hash__(
        self,
    ):  # make args hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


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

    @nn.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        return q1, q2


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
    final_fc_init_scale: float = 1e-3

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)
        means = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(outputs)
        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return distribution


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def target_update(model: TrainState, target_model: TrainState, tau: float) -> Tuple[TrainState, jnp.ndarray]:
    new_target_params = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params)
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(train_state: TrainState, loss_fn: Callable) -> Tuple[float, Any]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class AWACTrainer(NamedTuple):
    rng: jax.random.PRNGKey
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    max_action: float = 1.0 - 1e-5
    exp_adv_max: float = 100.0

    def update_actor(
        agent,
        batch: Transition,
        rng: jax.random.PRNGKey,
        args: Args,
    ) -> Tuple["AWACTrainer", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = agent.actor.apply_fn(actor_params, batch.observations)
            pi_actions = dist.sample(seed=rng)
            q_1, q_2 = agent.critic.apply_fn(agent.critic.params, batch.observations, pi_actions)
            v = jnp.minimum(q_1, q_2)

            actions = jnp.clip(batch.actions, -agent.max_action, agent.max_action)
            q_1, q_2 = agent.critic.apply_fn(agent.critic.params, batch.observations, actions)
            q = jnp.minimum(q_1, q_2)
            adv = q - v
            weights = jnp.minimum(jnp.exp(adv / args.beta), agent.exp_adv_max)

            weights = jax.lax.stop_gradient(weights)

            log_prob = dist.log_prob(batch.actions)
            loss = -jnp.mean(log_prob[..., None] * weights).mean()
            return loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(agent, batch: Transition, rng: jax.random.PRNGKey, args: Args) -> Tuple["AWACTrainer", jnp.ndarray]:
        def critic_loss_fn(critic_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            q_pred_1, q_pred_2 = agent.critic.apply_fn(critic_params, batch.observations, batch.actions)
            target_next_action = agent.actor.apply_fn(agent.actor.params, batch.next_observations).sample(seed=rng)
            target_next_action = target_next_action.clip(-agent.max_action, agent.max_action)
            q_next_1, q_next_2 = agent.target_critic.apply_fn(agent.target_critic.params, batch.next_observations, target_next_action)
            next_q = jnp.minimum(q_next_1, q_next_2)
            q_target = batch.rewards[..., None] + args.gamma * next_q * (1 - batch.dones[..., None])
            q_target = jax.lax.stop_gradient(q_target)
            value_loss_1 = jnp.square(q_pred_1 - q_target)
            value_loss_2 = jnp.square(q_pred_2 - q_target)
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
    ) -> Tuple["AWACTrainer", Dict]:
        for _ in range(args.n_jitted_updates):  # we can jit for roop for static unroll
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch_idx = jax.random.randint(batch_rng, (args.batch_size,), 0, len(data.observations))
            batch: Transition = jax.tree.map(lambda x: x[batch_idx], data)
            agent, critic_loss = agent.update_critic(batch, critic_rng, args)
            if _ % args.policy_frequency == 0:
                agent, actor_loss = agent.update_actor(batch, actor_rng, args)
                new_target_critic = target_update(agent.critic, agent.target_critic, args.tau)
                agent = agent._replace(target_critic=new_target_critic)
        return agent, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @jax.jit
    def sample_actions(
        agent,
        params: flax.core.FrozenDict[str, Any],
        obs: jnp.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        action = agent.actor.apply_fn(params, obs, temperature=temperature).sample(seed=seed)
        action = action.clip(-agent.max_action, agent.max_action)
        return action


def create_trainer(observations: jnp.ndarray, actions: jnp.ndarray, args: Args) -> AWACTrainer:
    rng = jax.random.PRNGKey(args.seed)
    rng, critic_rng, actor_rng = jax.random.split(rng, 3)

    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        hidden_dims=args.actor_hidden_dims,
        action_dim=action_dim,
    )
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(learning_rate=args.policy_learning_rate),
    )
    # initialize critic
    critic_model = DoubleCritic(
        hidden_dims=args.critic_hidden_dims,
    )
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=args.q_learning_rate),
    )
    # initialize target critic
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=args.q_learning_rate),
    )
    return AWACTrainer(
        rng,
        actor=actor_train_state,
        critic=critic_train_state,
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
            action = policy_fn(params, observation)
            action = unscale_action(env.action_space, action)
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
    env = gym.make(args.env_id)
    _, obs_mean, obs_std = get_dataset(env, args=args)
    obs, done = env.reset(seed=seed), False
    agent = create_trainer(obs, env.action_space.sample(), args)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    # note: qf_params is not used in this script
    model_path = f"{path}/flax.model"
    with open(model_path, "rb") as f:
        (actor_params, _) = flax.serialization.from_bytes((agent.actor.params, agent.critic.params), f.read())
    agent = agent._replace(actor=agent.actor.replace(params=actor_params))
    policy_fn = partial(agent.sample_actions, temperature=0.0, seed=actor_key)
    episodic_returns = []
    episodic_discounted_returns = []
    for _ in trange(eval_episodes, desc="Evaluating Final Model", dynamic_ncols=True):
        episode_return, discounted_return, time_step = 0, 0, 0
        while not done:
            obs = (obs - obs_mean) / obs_std
            action = jax.device_get(policy_fn(actor_params, obs))
            action = unscale_action(env.action_space, action)
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.name}__{args.seed}__{int(time.time())}"
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
    env = gym.make(args.env_id)
    rng = jax.random.PRNGKey(args.seed)
    dataset, obs_mean, obs_std = get_dataset(env, args=args)
    print(f"Dataset size: {len(dataset.observations)}")
    # create agent
    example_batch: Transition = jax.tree.map(lambda x: x[0], dataset)
    agent = create_trainer(example_batch.observations, example_batch.actions, args)

    num_steps = args.total_timesteps // args.n_jitted_updates
    for i in trange(num_steps, smoothing=0.1, dynamic_ncols=True, desc="Training"):
        rng, update_rng = jax.random.split(rng)
        agent, update_info = agent.update_n_times(
            dataset,
            update_rng,
            args,
        )  # update parameters
        for k, v in update_info.items():
            writer.add_scalar(f"charts/{k}", v, i)
        if args.track and (i % args.log_interval == 0):
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if (args.eval_episodes > 0) and (i % args.eval_interval == 0):
            policy_fn = partial(agent.sample_actions, temperature=0.0, seed=jax.random.PRNGKey(0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cpu_policy_fn = jax.jit(policy_fn, backend="cpu")
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
