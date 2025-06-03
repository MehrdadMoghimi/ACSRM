# source https://github.com/young-geng/JaxCQL
# https://arxiv.org/abs/2006.04779
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import time
from copy import deepcopy
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
import tqdm
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
    group: str = "CQL"
    """the W&B group to log to"""
    name: str = "CQL"
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
    normalize_state: bool = False
    """whether to normalize the state"""
    n_jitted_updates: int = 10
    """the number of jitted updates"""
    hidden_dims: Sequence[int] = (256, 256)
    """the hidden dimension of the network"""
    orthogonal_init: bool = False
    """whether to use orthogonal initialization"""
    policy_log_std_multiplier: float = 1.0
    """the multiplier of the policy log std"""
    policy_log_std_offset: float = -1.0
    """the offset of the policy log std"""
    alpha_multiplier: float = 1.0
    use_automatic_entropy_tuning: bool = True
    backup_entropy: bool = False
    target_entropy: float = 0.0
    optimizer_type: str = "adam"
    use_cql: bool = True
    cql_n_actions: int = 10
    cql_importance_sample: bool = True
    cql_lagrange: bool = False
    cql_target_action_gap: float = 1.0
    cql_temp: float = 1.0
    cql_min_q_weight: float = 5.0
    cql_max_target_backup: bool = False
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf

    def __hash__(
        self,
    ):  # make args hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


def extend_and_repeat(tensor: jnp.ndarray, axis: int, repeat: int) -> jnp.ndarray:
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(val - target))


def value_and_multi_grad(fun: Callable, n_outputs: int, argnums=0, has_aux=False) -> Callable:
    def select_output(index: int) -> Callable:
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]

        return wrapped

    grad_fns = tuple(jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux) for i in range(n_outputs))

    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)

    return multi_grad_fn


def update_target_network(main_params: Any, target_params: Any, tau: float) -> Any:
    return jax.tree_util.tree_map(lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params)


def multiple_action_q_function(forward: Callable) -> Callable:
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs) -> jnp.ndarray:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


class Scalar(nn.Module):
    init_value: float

    def setup(self) -> None:
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self) -> jnp.ndarray:
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, input_tensor: jnp.ndarray) -> jnp.ndarray:
        x = input_tensor
        for h in self.hidden_dims:
            if self.orthogonal_init:
                x = nn.Dense(
                    h,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = nn.relu(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(1e-2, "fan_in", "uniform"),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        return output


class FullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = FullyConnectedNetwork(
            output_dim=1,
            hidden_dims=self.hidden_dims,
            orthogonal_init=self.orthogonal_init,
        )(x)
        return jnp.squeeze(x, -1)


class TanhGaussianPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0

    def setup(self) -> None:
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim,
            hidden_dims=self.hidden_dims,
            orthogonal_init=self.orthogonal_init,
        )
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)

    def log_prob(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        return action_distribution.log_prob(actions)

    def __call__(
        self,
        observations: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic=False,
        repeat=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

        return samples, log_prob


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {"{}/{}".format(prefix, key): value for key, value in collected.items()}
    return collected


class CQLTrainer(object):

    def __init__(self, args, policy, qf, rng):
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "sgd": optax.sgd,
        }[args.optimizer_type]

        rng, policy_rng, q1_rng, q2_rng = jax.random.split(rng, 4)

        policy_params = self.policy.init(policy_rng, jnp.zeros((10, self.observation_dim)), policy_rng)
        self._train_states["policy"] = TrainState.create(params=policy_params, tx=optimizer_class(args.policy_learning_rate), apply_fn=None)

        qf1_params = self.qf.init(
            q1_rng,
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )
        self._train_states["qf1"] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(args.q_learning_rate),
            apply_fn=None,
        )
        qf2_params = self.qf.init(
            q2_rng,
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )
        self._train_states["qf2"] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(args.q_learning_rate),
            apply_fn=None,
        )
        self._target_qf_params = deepcopy({"qf1": qf1_params, "qf2": qf2_params})

        model_keys = ["policy", "qf1", "qf2"]

        if args.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            rng, log_alpha_rng = jax.random.split(rng)
            self._train_states["log_alpha"] = TrainState.create(
                params=self.log_alpha.init(log_alpha_rng),
                tx=optimizer_class(args.policy_learning_rate),
                apply_fn=None,
            )
            model_keys.append("log_alpha")

        if args.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            rng, log_alpha_prime_rng = jax.random.split(rng)
            self._train_states["log_alpha_prime"] = TrainState.create(
                params=self.log_alpha_prime.init(log_alpha_prime_rng),
                tx=optimizer_class(args.q_learning_rate),
                apply_fn=None,
            )
            model_keys.append("log_alpha_prime")

        self._model_keys = tuple(model_keys)

    @partial(jax.jit, static_argnames=("self", "args", "bc"))
    def train(self, train_states, target_q_params, dataset, rng, args, bc=False):
        for _ in range(args.n_jitted_updates):
            rng, batch_rng, update_rng = jax.random.split(rng, 3)
            batch_indices = jax.random.randint(batch_rng, (args.batch_size,), 0, len(dataset.observations))
            batch = jax.tree.map(lambda x: x[batch_indices], dataset)
            train_states, target_q_params, metrics = self._train_step(train_states, target_q_params, update_rng, batch, args, bc)
        return train_states, target_q_params, metrics

    def _train_step(self, train_states, target_qf_params, _rng, batch, args, bc=False):

        def loss_fn(train_params):
            observations = batch.observations
            actions = batch.actions
            rewards = batch.rewards
            next_observations = batch.next_observations
            dones = batch.dones

            loss_collection = {}

            rng, new_actions_rng = jax.random.split(_rng)
            new_actions, log_pi = self.policy.apply(train_params["policy"], observations, new_actions_rng)

            if args.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params["log_alpha"]) * (log_pi + args.target_entropy).mean()
                loss_collection["log_alpha"] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params["log_alpha"])) * args.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = args.alpha_multiplier

            """ Policy loss """
            if bc:
                rng, bc_rng = jax.random.split(rng)
                log_probs = self.policy.apply(
                    train_params["policy"],
                    observations,
                    actions,
                    bc_rng,
                    method=self.policy.log_prob,
                )
                policy_loss = (alpha * log_pi - log_probs).mean()
            else:
                q_new_actions = jnp.minimum(
                    self.qf.apply(train_params["qf1"], observations, new_actions),
                    self.qf.apply(train_params["qf2"], observations, new_actions),
                )
                policy_loss = (alpha * log_pi - q_new_actions).mean()

            loss_collection["policy"] = policy_loss

            """ Q function loss """
            q1_pred = self.qf.apply(train_params["qf1"], observations, actions)
            q2_pred = self.qf.apply(train_params["qf2"], observations, actions)

            if args.cql_max_target_backup:
                rng, cql_rng = jax.random.split(rng)
                new_next_actions, next_log_pi = self.policy.apply(
                    train_params["policy"],
                    next_observations,
                    cql_rng,
                    repeat=args.cql_n_actions,
                )
                target_q_values = jnp.minimum(
                    self.qf.apply(target_qf_params["qf1"], next_observations, new_next_actions),
                    self.qf.apply(target_qf_params["qf2"], next_observations, new_next_actions),
                )
                max_target_indices = jnp.expand_dims(jnp.argmax(target_q_values, axis=-1), axis=-1)
                target_q_values = jnp.take_along_axis(target_q_values, max_target_indices, axis=-1).squeeze(-1)
                next_log_pi = jnp.take_along_axis(next_log_pi, max_target_indices, axis=-1).squeeze(-1)
            else:
                rng, cql_rng = jax.random.split(rng)
                new_next_actions, next_log_pi = self.policy.apply(train_params["policy"], next_observations, cql_rng)
                target_q_values = jnp.minimum(
                    self.qf.apply(target_qf_params["qf1"], next_observations, new_next_actions),
                    self.qf.apply(target_qf_params["qf2"], next_observations, new_next_actions),
                )

            if args.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = jax.lax.stop_gradient(rewards + (1.0 - dones) * args.gamma * target_q_values)
            qf1_loss = mse_loss(q1_pred, td_target)
            qf2_loss = mse_loss(q2_pred, td_target)

            ### CQL
            if args.use_cql:
                batch_size = actions.shape[0]
                rng, random_rng = jax.random.split(rng)
                cql_random_actions = jax.random.uniform(
                    random_rng,
                    shape=(batch_size, args.cql_n_actions, self.action_dim),
                    minval=-1.0,
                    maxval=1.0,
                )
                rng, current_rng = jax.random.split(rng)
                cql_current_actions, cql_current_log_pis = self.policy.apply(
                    train_params["policy"],
                    observations,
                    current_rng,
                    repeat=args.cql_n_actions,
                )
                rng, next_rng = jax.random.split(rng)
                cql_next_actions, cql_next_log_pis = self.policy.apply(
                    train_params["policy"],
                    next_observations,
                    next_rng,
                    repeat=args.cql_n_actions,
                )

                cql_q1_rand = self.qf.apply(train_params["qf1"], observations, cql_random_actions)
                cql_q2_rand = self.qf.apply(train_params["qf2"], observations, cql_random_actions)
                cql_q1_current_actions = self.qf.apply(train_params["qf1"], observations, cql_current_actions)
                cql_q2_current_actions = self.qf.apply(train_params["qf2"], observations, cql_current_actions)
                cql_q1_next_actions = self.qf.apply(train_params["qf1"], observations, cql_next_actions)
                cql_q2_next_actions = self.qf.apply(train_params["qf2"], observations, cql_next_actions)

                cql_cat_q1 = jnp.concatenate(
                    [
                        cql_q1_rand,
                        jnp.expand_dims(q1_pred, 1),
                        cql_q1_next_actions,
                        cql_q1_current_actions,
                    ],
                    axis=1,
                )
                cql_cat_q2 = jnp.concatenate(
                    [
                        cql_q2_rand,
                        jnp.expand_dims(q2_pred, 1),
                        cql_q2_next_actions,
                        cql_q2_current_actions,
                    ],
                    axis=1,
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if args.cql_importance_sample:
                    random_density = jnp.log(0.5**self.action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [
                            cql_q1_rand - random_density,
                            cql_q1_next_actions - cql_next_log_pis,
                            cql_q1_current_actions - cql_current_log_pis,
                        ],
                        axis=1,
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [
                            cql_q2_rand - random_density,
                            cql_q2_next_actions - cql_next_log_pis,
                            cql_q2_current_actions - cql_current_log_pis,
                        ],
                        axis=1,
                    )

                cql_qf1_ood = jax.scipy.special.logsumexp(cql_cat_q1 / args.cql_temp, axis=1) * args.cql_temp
                cql_qf2_ood = jax.scipy.special.logsumexp(cql_cat_q2 / args.cql_temp, axis=1) * args.cql_temp

                """Subtract the log likelihood of data"""
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred,
                    args.cql_clip_diff_min,
                    args.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred,
                    args.cql_clip_diff_min,
                    args.cql_clip_diff_max,
                ).mean()

                if args.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(self.log_alpha_prime.apply(train_params["log_alpha_prime"])),
                        a_min=0.0,
                        a_max=1000000.0,
                    )
                    cql_min_qf1_loss = alpha_prime * args.cql_min_q_weight * (cql_qf1_diff - args.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * args.cql_min_q_weight * (cql_qf2_diff - args.cql_target_action_gap)

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5

                    loss_collection["log_alpha_prime"] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * args.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * args.cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_loss + cql_min_qf1_loss
                qf2_loss = qf2_loss + cql_min_qf2_loss

            loss_collection["qf1"] = qf1_loss
            loss_collection["qf2"] = qf2_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params)

        new_train_states = {key: train_states[key].apply_gradients(grads=grads[i][key]) for i, key in enumerate(self.model_keys)}
        new_target_qf_params = {}
        new_target_qf_params["qf1"] = update_target_network(
            new_train_states["qf1"].params,
            target_qf_params["qf1"],
            args.tau,
        )
        new_target_qf_params["qf2"] = update_target_network(
            new_train_states["qf2"].params,
            target_qf_params["qf2"],
            args.tau,
        )

        metrics = collect_metrics(
            aux_values,
            [
                "log_pi",
                "policy_loss",
                "qf1_loss",
                "qf2_loss",
                "alpha_loss",
                "alpha",
                "q1_pred",
                "q2_pred",
                "target_q_values",
            ],
        )

        if args.use_cql:
            metrics.update(
                collect_metrics(
                    aux_values,
                    [
                        "cql_std_q1",
                        "cql_std_q2",
                        "cql_q1_rand",
                        "cql_q2_rand" "cql_qf1_diff",
                        "cql_qf2_diff",
                        "cql_min_qf1_loss",
                        "cql_min_qf2_loss",
                        "cql_q1_current_actions",
                        "cql_q2_current_actions" "cql_q1_next_actions",
                        "cql_q2_next_actions",
                        "alpha_prime",
                        "alpha_prime_loss",
                    ],
                    "cql",
                )
            )

        return new_train_states, new_target_qf_params, metrics

    @partial(jax.jit, static_argnames=("self",))
    def get_actions(self, params, obs):
        action, _ = self.policy.apply(params, obs.reshape(1, -1), jax.random.PRNGKey(0), deterministic=True)
        return action.squeeze(0)

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}


def create_trainer(observations: jnp.ndarray, actions: jnp.ndarray, args: Args) -> CQLTrainer:
    policy = TanhGaussianPolicy(
        observation_dim=observations.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=args.hidden_dims,
        orthogonal_init=args.orthogonal_init,
        log_std_multiplier=args.policy_log_std_multiplier,
        log_std_offset=args.policy_log_std_offset,
    )
    qf = FullyConnectedQFunction(
        observation_dim=observations.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=args.hidden_dims,
        orthogonal_init=args.orthogonal_init,
    )
    rng = jax.random.PRNGKey(args.seed)
    return CQLTrainer(args, policy, qf, rng)


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
    for _ in tqdm.tqdm(range(eval_episodes), desc="Evaluating Model", dynamic_ncols=True):
        episode_return = 0
        while not done:
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(params, observation)
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
    train_states = agent._train_states
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    # note: qf_params is not used in this script
    model_path = f"{path}/flax.model"
    with open(model_path, "rb") as f:
        (actor_params, _) = flax.serialization.from_bytes((train_states["policy"].params, train_states["qf1"].params), f.read())
    train_states["policy"] = train_states["policy"].replace(params=actor_params)
    actor_params = train_states["policy"].params
    episodic_returns = []
    episodic_discounted_returns = []
    for _ in tqdm.tqdm(range(eval_episodes), desc="Evaluating Final Model", dynamic_ncols=True):
        episode_return, discounted_return, time_step = 0, 0, 0
        while not done:
            obs = (obs - obs_mean) / obs_std
            action = jax.device_get(agent.get_actions(actor_params, obs))
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

    if args.target_entropy >= 0.0:
        args.target_entropy = -np.prod(env.action_space.shape).item()

    sac = create_trainer(dataset.observations, dataset.actions, args)
    train_states, target_qf_params = sac._train_states, sac._target_qf_params

    num_steps = args.total_timesteps // args.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Training"):
        rng, update_rng = jax.random.split(rng)
        train_states, target_qf_params, update_info = sac.train(
            train_states,
            target_qf_params,
            dataset,
            update_rng,
            args,
            bc=False,
        )  # update parameters
        for k, v in update_info.items():
            writer.add_scalar(f"charts/{k}", v, i)
        if args.track and (i % args.log_interval == 0):
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if (args.eval_episodes > 0) and (i % args.eval_interval == 0):
            policy_fn = sac.get_actions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cpu_policy_fn = jax.jit(policy_fn, backend="cpu")
            cpu_params = jax.device_put(train_states["policy"].params, jax.devices("cpu")[0])
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
            writer.add_scalar("charts/Mean", np.mean(normalized_score), i)

    path = f"{args.dir}/{run_name}"
    if args.save_model:
        os.makedirs(path, exist_ok=True)
        model_path = f"{args.dir}/{run_name}/flax.model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        train_states["policy"].params,
                        train_states["qf1"].params,
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
