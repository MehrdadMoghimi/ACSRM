import os

import d4rl.gym_mujoco
import d4rl
import gym
import h5py

# from experience_replay import D4RL_Dataset
from reward_wrappers import RewardHighVelocity, RewardUnhealthyPose

# from create_hdf5 import HDF5_Creator
# from utilities import get_keys

path_to_datasets = os.environ.get("D4RL_DATASET_DIR", os.path.expanduser("~/.d4rl/datasets"))
env_id_list = [
    "hopper-medium-v0-s",
    "hopper-expert-v0-s",
    "walker2d-medium-v0-s",
    "walker2d-expert-v0-s",
    "halfcheetah-medium-v0-s",
    "halfcheetah-expert-v0-s",
]

info = {
    "hopper-medium-v0-s": {"name": "hopper-medium-v0", "prob_pose_penal": 0.1, "cost_pose": -50},
    "hopper-expert-v0-s": {"name": "hopper-expert-v0", "prob_pose_penal": 0.1, "cost_pose": -50},
    "walker2d-medium-v0-s": {"name": "walker2d-medium-v0", "prob_pose_penal": 0.1, "cost_pose": -30, "terminate_when_unhealthy": 1},
    "walker2d-expert-v0-s": {"name": "walker2d-expert-v0", "prob_pose_penal": 0.1, "cost_pose": -30, "terminate_when_unhealthy": 1},
    "halfcheetah-medium-v0-s": {"name": "halfcheetah-medium-v0", "prob_vel_penal": 0.05, "max_vel": 4, "cost_vel": -70},
    "halfcheetah-expert-v0-s": {"name": "halfcheetah-expert-v0", "prob_vel_penal": 0.05, "max_vel": 10, "cost_vel": -60},
}


def get_gym_name(dataset_name):
    # Get v3 version of environments for extra information to be available
    if "cheetah" in dataset_name:
        return "HalfCheetah-v3"
    elif "hopper" in dataset_name:
        return "Hopper-v3"
    elif "walker" in dataset_name:
        return "Walker2d-v3"
    else:
        raise ValueError("{dataset_name} is not in D4RL")


def get_env(env_name):
    reset_noise_scale = None
    eval_terminate_when_unhealthy = True
    terminate_when_unhealthy = False if not eval_terminate_when_unhealthy else True

    env_d4rl = gym.make(info[env_name]["name"])
    dataset_name = env_d4rl.dataset_filepath[:-5]
    # Use v3 version of environments for extra information to be available
    kwargs = {"terminate_when_unhealthy": terminate_when_unhealthy} if "cheetah" not in dataset_name else {}
    if reset_noise_scale is not None:
        kwargs["reset_noise_scale"] = reset_noise_scale
    env = gym.make(get_gym_name(dataset_name), **kwargs).unwrapped

    if "prob_vel_penal" in info[env_name].keys() is not None and info[env_name]["prob_vel_penal"] > 0:
        dict_env = {"prob_vel_penal": info[env_name]["prob_vel_penal"], "cost_vel": info[env_name]["cost_vel"], "max_vel": info[env_name]["max_vel"]}

        fname = f"{dataset_name}_" f'prob{dict_env["prob_vel_penal"]}_' f'penal{dict_env["cost_vel"]}_' f'maxvel{dict_env["max_vel"]}.hdf5'

        env = RewardHighVelocity(env, **dict_env)

    elif "prob_pose_penal" in info[env_name].keys() and info[env_name]["prob_pose_penal"] > 0:
        dict_env = {
            "prob_pose_penal": info[env_name]["prob_pose_penal"],
            "cost_pose": info[env_name]["cost_pose"],
        }

        fname = f"{dataset_name}_" f'prob{dict_env["prob_pose_penal"]}_' f'penal{dict_env["cost_pose"]}_' "pose.hdf5"
        env = RewardUnhealthyPose(env, **dict_env)

    else:
        fname = env_d4rl.dataset_filepath

    env.h5py_path = os.path.join(path_to_datasets, fname)

    if "cheetah" in dataset_name:
        env = CustomEnvironmentWrapper(env, -57.95, 659.49)
    elif "hopper" in dataset_name:
        env = CustomEnvironmentWrapper(env, -13.37, 1602.04)
    elif "walker" in dataset_name:
        env = CustomEnvironmentWrapper(env, -17.32, 1790.02)
    else:
        raise ValueError("{dataset_name} is not in D4RL")

    return env


class CustomEnvironmentWrapper(gym.Wrapper):
    """Wrapper that adds a custom get_normalized_score method."""

    def __init__(self, env, min_score, max_score):
        super().__init__(env)
        self.min_score = min_score
        self.max_score = max_score

    def get_normalized_score(self, raw_score):
        """Normalizes the raw score using pre-defined min and max scores."""
        return (raw_score - self.min_score) / (self.max_score - self.min_score)

    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        return getattr(self.env, name)

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        return super().reset(**kwargs)
