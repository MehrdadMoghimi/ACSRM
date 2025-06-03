import os
from flashbax.vault import Vault
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
from typing import Any, NamedTuple
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import d4rl
import gym
import os
import pickle
import h5py


def get_replay_buffer_vault(path: str, vault_name: str = "buffer", vault_uid=None) -> Vault:
    if vault_name is None:
        vault_name = "buffer"
    if os.path.isdir(path):
        if os.path.isdir(os.path.join(path, vault_name)):
            new_path = os.path.join(path, vault_name)
            if vault_uid is None:
                if len(os.listdir(new_path)) == 0:
                    raise ValueError("Vault is empty")
                if len(os.listdir(new_path)) > 1:
                    raise ValueError("Vault has more than one buffers")
                else:
                    vault_uid = os.listdir(new_path)[0]
            else:
                new_path = os.path.join(path, vault_name, vault_uid)
                if not os.path.isdir(new_path):
                    print("Vault UID: ", vault_uid)
                    raise ValueError("Vault UID not valid")
        else:
            print("Vault name: ", vault_name)
            raise ValueError("Vault name not valid")
    else:
        print("Path: ", path)
        raise ValueError("Path not valid")
    print(f"Vault Name: {vault_name}, Vault UID: {vault_uid}")
    v = Vault(vault_name=vault_name, rel_dir=path, vault_uid=vault_uid).read()
    return v


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def get_h5py_dataset(env):
    data_dict = {}
    with h5py.File(env.h5py_path, "r") as dataset_file:
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ["observations", "actions", "rewards", "terminals"]:
        assert key in data_dict, "Dataset is missing key %s" % key
    N_samples = data_dict["observations"].shape[0]
    # if env.observation_space.shape is not None:
    #    assert data_dict["observations"].shape[1:] == env.observation_space.shape, "Observation shape does not match env: %s vs %s" % (
    #        str(data_dict["observations"].shape[1:]),
    #        str(env.observation_space.shape),
    #    )
    assert data_dict["actions"].shape[1:] == env.action_space.shape, "Action shape does not match env: %s vs %s" % (
        str(data_dict["actions"].shape[1:]),
        str(env.action_space.shape),
    )
    if data_dict["rewards"].shape == (N_samples, 1):
        data_dict["rewards"] = data_dict["rewards"][:, 0]
    assert data_dict["rewards"].shape == (N_samples,), "Reward has wrong shape: %s" % (str(data_dict["rewards"].shape))
    if data_dict["terminals"].shape == (N_samples, 1):
        data_dict["terminals"] = data_dict["terminals"][:, 0]
    assert data_dict["terminals"].shape == (N_samples,), "Terminals has wrong shape: %s" % (str(data_dict["rewards"].shape))
    return data_dict


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


env_id_list = [
    "hopper-medium-v0-s",
    "hopper-expert-v0-s",
    "walker2d-medium-v0-s",
    "walker2d-expert-v0-s",
    "halfcheetah-medium-v0-s",
    "halfcheetah-expert-v0-s",
]


def get_dataset(
    env: gym.Env,
    vault_name=None,
    vault_uid=None,
    args: Any = None,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    extended_obs: bool = False,
    num_of_copies: int = 0,
    noise_scale: float = 10.0,
) -> Transition:

    # if env registered in d4rl, use d4rl dataset, else use path to load vault
    if env.spec.id in d4rl.infos.DATASET_URLS.keys():
        if env.spec.id in env_id_list:
            dataset = get_h5py_dataset(env)
            dataset = d4rl.qlearning_dataset(env, dataset=dataset)
        else:
            dataset = d4rl.qlearning_dataset(env)
        obs = np.array(dataset["observations"])
        rewards = np.array(dataset["rewards"])
        actions = np.array(dataset["actions"])
        next_obs = np.roll(obs, -1, axis=0)
        imputed_next_observations = np.roll(dataset["observations"], -1, axis=0)
        same_obs = np.all(
            np.isclose(imputed_next_observations, dataset["next_observations"], atol=1e-5),
            axis=-1,
        )
        dones = 1.0 - same_obs.astype(np.float32)
        dones[-1] = 1.0
        if clip_to_eps:
            lim = 1 - eps
            actions = np.clip(dataset["actions"], -lim, lim)
    elif "1R2R" in args.data_dir:
        with open(args.data_dir, "rb") as handle:
            dataset = pickle.load(handle)
        obs = np.array(dataset["observations"])
        rewards = np.array(dataset["rewards"].squeeze())
        actions = np.array(dataset["actions"])
        # print("actions: ", [actions.min(), actions.max()])
        dones = np.array(dataset["terminals"].squeeze())
        next_obs = np.array(dataset["next_observations"])
    elif args.data_dir != "":
        buffer_state = get_replay_buffer_vault(args.data_dir, vault_name=vault_name, vault_uid=vault_uid)
        obs = np.array(buffer_state.experience["obs"][0])
        rewards = np.array(buffer_state.experience["reward"][0])
        actions = np.array(buffer_state.experience["action"][0])
        dones = np.array(buffer_state.experience["done"][0])
        next_obs = np.roll(obs, -1, axis=0)
        dones[-1] = 1
    else:
        raise ValueError("Please provide either env or path to load dataset")

    if extended_obs:
        N = rewards.shape[0]
        s_obs = np.zeros((N,))
        s_prime_obs = np.zeros((N,))
        c_obs = np.zeros((N,))
        c_prime_obs = np.zeros((N,))
        initial_states_ = []
        returns_ = []
        episode_step = 0
        s = 0.0
        c = 1.0
        for i in range(N):
            if episode_step == 0:
                initial_states_.append(np.append(obs[i], [s, c]))
            c_obs[i] = c
            s_obs[i] = s
            c *= args.gamma
            s += c * rewards[i] / args.reward_normalizer
            c_prime_obs[i] = c
            s_prime_obs[i] = s
            episode_step += 1
            if dones[i]:
                returns_.append(s)
                c = 1.0
                s = 0.0
                episode_step = 0
        episode_returns = np.array(returns_)
        initial_states = np.array(initial_states_)
        obs = np.concatenate([obs, s_obs[:, None], c_obs[:, None]], axis=-1)
        next_obs = np.concatenate([next_obs, s_prime_obs[:, None], c_prime_obs[:, None]], axis=-1)
        if args.n_quantiles is None:
            raise ValueError("Extended obs requires n_quantiles to be set")
        tau = (2 * np.arange(args.n_quantiles, dtype=np.float32) + 1.0) / (2.0 * args.n_quantiles)
        theta_0 = np.quantile(episode_returns, tau)
        initial_state = initial_states.mean(0)

        if num_of_copies > 0:
            M = obs.shape[1]
            k = num_of_copies
            random_offsets = np.zeros((k * N, M))
            random_offsets[:, -2] = np.random.normal(loc=0, scale=noise_scale, size=(k * N))
            obs_copies = np.tile(obs, (k, 1)) + random_offsets
            next_obs_copies = np.tile(next_obs, (k, 1)) + random_offsets
            obs = np.concatenate([obs, obs_copies], axis=0)
            next_obs = np.concatenate([next_obs, next_obs_copies], axis=0)
            rewards = np.tile(rewards, (k + 1,))
            actions = np.tile(actions, (k + 1, 1))
            dones = np.tile(dones, (k + 1,))
            print("Extended obs: ", obs.shape, next_obs.shape, rewards.shape, actions.shape, dones.shape)

    dataset = Transition(
        observations=jnp.array(obs, dtype=jnp.float32),
        actions=jnp.array(actions, dtype=jnp.float32),
        rewards=jnp.array(rewards, dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
        next_observations=jnp.array(next_obs, dtype=jnp.float32),
    )
    rng = jax.random.PRNGKey(0)
    rng, rng_permute = jax.random.split(rng, 2)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree.map(lambda x: x[perm], dataset)
    # normalize states
    obs_mean, obs_std = jnp.zeros((obs.shape[1],)), jnp.ones((obs.shape[1],))
    if args.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std),  # + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std),  # + 1e-5),
        )
        if extended_obs:
            initial_state = (initial_state - obs_mean) / (obs_std)  # + 1e-5)
    if extended_obs:
        return dataset, obs_mean, obs_std, jnp.array(theta_0, dtype=jnp.float32), jnp.array(initial_state, dtype=jnp.float32)
    return dataset, obs_mean, obs_std
