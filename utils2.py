import os
import pandas as pd
import numpy as np
import jax

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

jax.config.update("jax_platform_name", "cpu")

from online.sac_isrm import weighted_sum_of_cvar, exponential_risk_measures

from online.sac import evaluate as sac_evaluate
from online.sac_isrm import evaluate as sac_isrm_evaluate
from online.ac import evaluate as ac_evaluate
from online.ac_srm import evaluate as ac_srm_evaluate
from online.ac_isrm import evaluate as ac_isrm_evaluate
from online.td3 import evaluate as td3_evaluate
from online.td3_srm import evaluate as td3_srm_evaluate
from online.td3_isrm import evaluate as td3_isrm_evaluate

from offline.awac import evaluate as awac_evaluate
from offline.oac_srm import evaluate as awac_srm_evaluate
from offline.td3bc import evaluate as td3bc_evaluate
from offline.td3bc_srm import evaluate as td3bc_srm_evaluate
from offline.td3bc_isrm import evaluate as td3bc_isrm_evaluate
from offline.iql import evaluate as iql_evaluate
from offline.cql import evaluate as cql_evaluate

from dataclasses import dataclass


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""


def run_simulation_from_dir(path, Nsimulations=None, eval_sim_seed=None, normalize=False):
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for i, directory in enumerate(directories):
        environment_name = directory.split("__")[0]
        agent_name = directory.split("__")[1].lower()
        agent_seed = int(directory.split("__")[2])
        sim_seed = agent_seed if eval_sim_seed is None else eval_sim_seed
        print("Processing model {} --> {} with seed {}".format(i + 1, directory, sim_seed))
        new_path = os.path.join(path, directory)
        # write switch case for each agent to use the correct evaluation function
        if agent_name == "sac":
            risk_measure = "Mean"
            alpha = 1.0
            weight = 1.0
            _, rewards = sac_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "sac_isrm":
            risk_measure = directory.split("__")[6]
            if risk_measure == "WSCVaR":
                alpha = directory.split("__")[7]
                weight = directory.split("__")[8]
            else:
                alpha = directory.split("__")[5]
                weight = 1.0
            _, rewards = sac_isrm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "ac":
            risk_measure = "Mean"
            alpha = 1.0
            weight = 1.0
            _, rewards = ac_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "ac_srm":
            risk_measure = directory.split("__")[6]
            if risk_measure == "WSCVaR":
                alpha = directory.split("__")[7]
                weight = directory.split("__")[8]
            else:
                alpha = directory.split("__")[5]
                weight = 1.0
            _, rewards = ac_srm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "ac_isrm":
            risk_measure = directory.split("__")[6]
            if risk_measure == "WSCVaR":
                alpha = directory.split("__")[7]
                weight = directory.split("__")[8]
            else:
                alpha = directory.split("__")[5]
                weight = 1.0
            _, rewards = ac_isrm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "td3":
            risk_measure = "Mean"
            alpha = 1.0
            weight = 1.0
            _, rewards = td3_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "td3_srm":
            risk_measure = directory.split("__")[6]
            if risk_measure == "WSCVaR":
                alpha = directory.split("__")[7]
                weight = directory.split("__")[8]
            else:
                alpha = directory.split("__")[5]
                weight = 1.0
            _, rewards = td3_srm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        elif agent_name == "td3_isrm":
            risk_measure = directory.split("__")[6]
            if risk_measure == "WSCVaR":
                alpha = directory.split("__")[7]
                weight = directory.split("__")[8]
            else:
                alpha = directory.split("__")[5]
                weight = 1.0
            _, rewards = td3_isrm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
        else:
            raise ValueError("Unknown model name: {}".format(agent_name))
        df = pd.DataFrame(rewards, columns=["rewards"])
        df = df.assign(
            environment_name=environment_name,
            agent=agent_name,
            risk_measure=risk_measure,
            alpha=str(alpha),
            weight=str(weight),
            # n_quantile=args["n_quantiles"],
            sim_seed=sim_seed,
            agent_seed=agent_seed,
        )
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


def run_simulation_from_dir_offline(path, Nsimulations=None, eval_sim_seed=None, normalize=False):
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for i, directory in enumerate(directories):
        environment_name = directory.split("__")[0]
        agent_name = directory.split("__")[1].lower()
        agent_seed = int(directory.split("__")[2])
        sim_seed = agent_seed if eval_sim_seed is None else eval_sim_seed
        print("Processing model {} --> {} with seed {}".format(i + 1, directory, sim_seed))
        new_path = os.path.join(path, directory)
        # write switch case for each agent to use the correct evaluation function
        try:
            if agent_name == "awac":
                risk_measure = "Mean"
                alpha = 1.0
                weight = 1.0
                _, rewards = awac_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            elif agent_name == "awac_srm":
                risk_measure = directory.split("__")[6]
                if risk_measure == "WSCVaR":
                    alpha = directory.split("__")[7]
                    weight = directory.split("__")[8]
                else:
                    alpha = directory.split("__")[5]
                    weight = 1.0
                _, rewards = awac_srm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            elif agent_name == "td3bc":
                risk_measure = "Mean"
                alpha = 1.0
                weight = 1.0
                _, rewards = td3bc_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            elif agent_name == "td3bc_srm":
                risk_measure = directory.split("__")[6]
                if risk_measure == "WSCVaR":
                    alpha = directory.split("__")[7]
                    weight = directory.split("__")[8]
                else:
                    alpha = directory.split("__")[5]
                    weight = 1.0
                _, rewards = td3bc_srm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            elif agent_name == "td3bc_isrm":
                risk_measure = directory.split("__")[6]
                if risk_measure == "WSCVaR":
                    alpha = directory.split("__")[7]
                    weight = directory.split("__")[8]
                else:
                    alpha = directory.split("__")[5]
                    weight = 1.0
                _, rewards = td3bc_isrm_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            elif agent_name == "iql":
                risk_measure = "Mean"
                alpha = 1.0
                weight = 1.0
                _, rewards = iql_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            elif agent_name == "cql":
                risk_measure = "Mean"
                alpha = 1.0
                weight = 1.0
                _, rewards = cql_evaluate(new_path, eval_episodes=Nsimulations, seed=sim_seed, normalize=normalize)
            else:
                raise ValueError("Unknown model name: {}".format(agent_name))
        except:
            print(f"Error in {directory}")
            continue
        df = pd.DataFrame(rewards, columns=["rewards"])
        df = df.assign(
            environment_name=environment_name,
            agent=agent_name,
            risk_measure=risk_measure,
            alpha=str(alpha),
            weight=str(weight),
            n_quantile=int(directory.split("__")[4]) if len(directory.split("__")) > 4 else 0,
            sim_seed=sim_seed,
            agent_seed=agent_seed,
        )
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


def load_data_from_dir(path):
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, directory in enumerate(directories):
        print("Processing model {} --> {}".format(ii + 1, directory))
        env_name = directory.split("__")[0]
        agent_name = directory.split("__")[1]
        agent_seed = int(directory.split("__")[2])
        risk_measure = "Mean"
        alpha = "1.0"
        if len(directory.split("__")) > 4:
            risk_measure = directory.split("__")[6]
            alpha = directory.split("__")[-2] + "_" + directory.split("__")[-1]
        new_path = os.path.join(path, directory)
        scalar_accumulator = EventAccumulator(str(new_path)).Reload().scalars
        keys = scalar_accumulator.Keys()
        assert "charts/episodic_return" in keys
        idx = keys.index("charts/episodic_return")
        df = pd.DataFrame(scalar_accumulator.Items(keys[idx]))
        df["wall_time"] = df["wall_time"] - df["wall_time"][0]
        df = df.assign(
            environment_name=env_name,
            agent=agent_name,
            agent_seed=agent_seed,
            risk_measure=risk_measure,
            alpha=alpha,
        )
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


env_id_list = [
    "hopper-medium-v0-s",
    "hopper-expert-v0-s",
    "walker2d-medium-v0-s",
    "walker2d-expert-v0-s",
    "halfcheetah-medium-v0-s",
    "halfcheetah-expert-v0-s",
]

GAME_NAMES = [
    ("NChain-v0", "NChain"),
    ("Trading-v0", "Trading"),
    ("Trading-v1", "Trading"),
    ("HIVTreatment-v0", "HIV Treatment"),
    ("HIVTreatment-v1", "HIV Treatment"),
    ("HalfCheetah-v2", "HalfCheetah"),
    ("halfcheetah-random-v2", "HalfCheetah Random"),
    ("halfcheetah-medium-expert-v2", "HalfCheetah Medium Expert"),
    ("halfcheetah-medium-replay-v2", "HalfCheetah Medium Replay"),
    ("halfcheetah-medium-v2", "HalfCheetah Medium"),
    ("Hopper-v2", "Hopper"),
    ("HopperModNoise-v0", "Hopper Mod Noise"),
    ("HopperHighNoise-v0", "Hopper High Noise"),
    ("hopper-random-v2", "Hopper Random"),
    ("hopper-medium-expert-v2", "Hopper Medium Expert"),
    ("hopper-medium-replay-v2", "Hopper Medium Replay"),
    ("hopper-medium-v2", "Hopper Medium"),
    ("Walker2d-v2", "Walker2d"),
    ("Walker2dModNoise-v0", "Walker2d Mod Noise"),
    ("Walker2dHighNoise-v0", "Walker2d High Noise"),
    ("walker2d-random-v2", "Walker2d Random"),
    ("walker2d-medium-expert-v2", "Walker2d Medium Expert"),
    ("walker2d-medium-replay-v2", "Walker2d Medium Replay"),
    ("walker2d-medium-v2", "Walker2d Medium"),
    ("hopper-high-noise-medium-v0", "Hopper High Noise Medium"),
    ("hopper-high-noise-medium-expert-v0", "Hopper High Noise Medium Expert"),
    ("hopper-high-noise-medium-replay-v0", "Hopper High Noise Medium Replay"),
    ("hopper-mod-noise-medium-v0", "Hopper Mod Noise Medium"),
    ("hopper-mod-noise-medium-expert-v0", "Hopper Mod Noise Medium Expert"),
    ("hopper-mod-noise-medium-replay-v0", "Hopper Mod Noise Medium Replay"),
    ("walker2d-high-noise-medium-v0", "Walker2d High Noise Medium"),
    ("walker2d-high-noise-medium-expert-v0", "Walker2d High Noise Medium Expert"),
    ("walker2d-high-noise-medium-replay-v0", "Walker2d High Noise Medium Replay"),
    ("walker2d-mod-noise-medium-v0", "Walker2d Mod Noise Medium"),
    ("walker2d-mod-noise-medium-expert-v0", "Walker2d Mod Noise Medium Expert"),
    ("walker2d-mod-noise-medium-replay-v0", "Walker2d Mod Noise Medium Replay"),
    ("hopper-medium-v0-s", "Hopper Medium Stochastic"),
    ("hopper-expert-v0-s", "Hopper Expert Stochastic"),
    ("walker2d-medium-v0-s", "Walker2d Medium Stochastic"),
    ("walker2d-expert-v0-s", "Walker2d Expert Stochastic"),
    ("halfcheetah-medium-v0-s", "HalfCheetah Medium Stochastic"),
    ("halfcheetah-expert-v0-s", "HalfCheetah Expert Stochastic"),
    ("hopper-medium-v0", "Hopper Medium Stochastic"),
    ("hopper-expert-v0", "Hopper Expert Stochastic"),
    ("walker2d-medium-v0", "Walker2d Medium Stochastic"),
    ("walker2d-expert-v0", "Walker2d Expert Stochastic"),
    ("halfcheetah-medium-v0", "HalfCheetah Medium Stochastic"),
    ("halfcheetah-expert-v0", "HalfCheetah Expert Stochastic"),
]


GAME_NAME_MAP = dict(GAME_NAMES)

AGENT_NAMES = [
    ("oraac", "ORAAC"),
    ("codac", "CODAC"),
    ("sac", "SAC"),
    ("sac_qrdqn", "DSAC"),
    ("sac_srm", "SAC-SRM"),
    ("sac_isrm", "SAC-iSRM"),
    ("sac_qricvar", "SAC-iCVaR"),
    ("ac", "AC"),
    ("ac_qrdqn", "DAC"),
    ("ac_qrdqn_d", "DAC-D"),
    ("ac_srm", "AC-SRM"),
    ("ac_isrm", "AC-iSRM"),
    ("ac_srm_d", "AC-SRM-D"),
    ("ac_isrm_d", "AC-iSRM-D"),
    ("ac_qricvar", "AC-iCVaR"),
    ("ddpg", "DDPG"),
    ("ddpg_qrdqn", "D3PG"),
    ("ddpg_srm", "DDPG-SRM"),
    ("ddpg_qricvar", "D3PG-iCVaR"),
    ("td3", "TD3"),
    ("td3_qrdqn", "DTD3"),
    ("td3_srm", "TD3-SRM"),
    ("td3_isrm", "TD3-iSRM"),
    ("td3_qricvar", "DTD3-iCVaR"),
    ("awac", "AWAC"),
    ("awac_qrdqn", "DAWAC"),
    ("awac_srm", "AWAC-SRM"),
    ("awac_qricvar", "DAWAC-iCVaR"),
    ("td3bc", "TD3BC"),
    ("td3bc_qrdqn", "DTD3BC"),
    ("td3bc_srm", "TD3BC-SRM"),
    ("td3bc_isrm", "TD3BC-iSRM"),
    ("td3bc_qricvar", "DTD3BC-iCVaR"),
    ("iql", "IQL"),
    ("cql", "CQL"),
]
AGENT_NAME_MAP = dict(AGENT_NAMES)


def environment_pretty(row):
    if row["environment_name"] not in GAME_NAME_MAP:
        return row["environment_name"]
    return GAME_NAME_MAP[row["environment_name"]]


def agent_pretty(row):
    row["agent"] = row["agent"].lower()
    if (
        row["agent"] == "sac"
        or row["agent"] == "ac"
        or row["agent"] == "ddpg"
        or row["agent"] == "td3"
        or row["agent"] == "sac_qrdqn"
        or row["agent"] == "ac_qrdqn"
        or row["agent"] == "ac_qrdqn_d"
        or row["agent"] == "ddpg_qrdqn"
        or row["agent"] == "td3_qrdqn"
        or row["agent"] == "awac"
        or row["agent"] == "awac_qrdqn"
        or row["agent"] == "td3bc"
        or row["agent"] == "td3bc_qrdqn"
        or row["agent"] == "iql"
        or row["agent"] == "cql"
    ):
        return f"{AGENT_NAME_MAP[row['agent']]}"
    elif (
        row["agent"] == "sac_srm"
        or row["agent"] == "sac_isrm"
        or row["agent"] == "ac_srm"
        or row["agent"] == "ac_isrm"
        or row["agent"] == "ac_srm_d"
        or row["agent"] == "ac_isrm_d"
        or row["agent"] == "ddpg_srm"
        or row["agent"] == "td3_srm"
        or row["agent"] == "td3_isrm"
        or row["agent"] == "awac_srm"
        or row["agent"] == "td3bc_srm"
        or row["agent"] == "td3bc_isrm"
        or row["agent"] == "oraac"
        or row["agent"] == "codac"
    ):
        if row["risk_measure"] == "CVaR":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "Dual":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\nu$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "WSCVaR":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "MC" or row["risk_measure"] == "mc":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "Exp":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\lambda$=" + f"{row['alpha']})"
        else:
            raise ValueError("Unknown risk measure: {}".format(row["risk_measure"]))
    elif (
        row["agent"] == "sac_qricvar"
        or row["agent"] == "ac_qricvar"
        or row["agent"] == "ddpg_qricvar"
        or row["agent"] == "td3_qricvar"
        or row["agent"] == "awac_qricvar"
        or row["agent"] == "td3bc_qricvar"
    ):
        return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
    else:
        raise ValueError("Unknown agent name: {}".format(row["agent"]))


def add_columns(df):
    df["environment_pretty"] = df.apply(environment_pretty, axis=1)
    df["Model"] = df.apply(agent_pretty, axis=1)
    # add df["weight"]=1.0 if column does not exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["agent_id"] = (
        df["agent"] + "_" + df["alpha"].astype(str) + "_" + df["weight"].astype(str)
    )  # srm with SRM as risk measure is identified by alphas not weights
    return df


def add_columns2(df):
    df["environment_pretty"] = df.apply(environment_pretty, axis=1)
    df["Model"] = df.apply(agent_pretty, axis=1)
    return df


def mean_value(group):
    # Perform some operation on the group
    result = group.mean()  # Replace this with your actual operation
    return result


def cvar9(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.9)])
    return result


def cvar8(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.8)])
    return result


def cvar7(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.7)])
    return result


def cvar6(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.6)])
    return result


def cvar5(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.5)])
    return result


def cvar4(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.4)])
    return result


def cvar3(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.3)])
    return result


def cvar2(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.2)])
    return result


def cvar1(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.1)])
    return result


def cvar05(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.05)])
    return result


def cvar02(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values <= np.quantile(r_values, 0.02)])
    return result


def srm5(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.5, 1.0], weights=[0.5, 0.5])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def exp1(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values = exponential_risk_measures(taus_middle, alpha=1.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values) / nq
    return result


def exp2(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values = exponential_risk_measures(taus_middle, alpha=2.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values) / nq
    return result


def exp4(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values = exponential_risk_measures(taus_middle, alpha=4.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values) / nq
    return result


def exp6(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values = exponential_risk_measures(taus_middle, alpha=6.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values) / nq
    return result


def exp8(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values = exponential_risk_measures(taus_middle, alpha=8.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values) / nq
    return result


def srm2(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.5, 0.5])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm08(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.2, 0.8])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm06(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.4, 0.6])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm04(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.6, 0.4])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm02(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.8, 0.2])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


AGG_RISK_VALUES = [
    (r"$\mathbb{E}$", mean_value),
    (r"$\operatorname{CVaR}_{0.5}$", cvar5),
    (r"$\operatorname{CVaR}_{0.2}$", cvar2),
    (r"$\operatorname{WSCVaR}_{0.5,1.0}_{0.5,0.5}$", srm5),
    (r"$\operatorname{WSCVaR}_{0.2,1.0}_{0.5,0.5}$", srm2),
]

AGG_RISK_VALUES2 = [
    (r"$\mathbb{E}$", mean_value),
    (r"$\operatorname{CVaR}_{0.2}$", cvar2),
    (r"$\operatorname{CVaR}_{0.1}$", cvar1),
    (r"$\operatorname{CVaR}_{0.02}$", cvar02),
]

AGG_RISK_VALUES3 = [
    (r"$\mathbb{E}$", mean_value),
    (r"$\operatorname{CVaR}_{0.2}$", cvar2),
    # (r"$\operatorname{CVaR}_{0.1}$", cvar1),
]

AGG_RISK_VALUES4 = [
    (r"$\mathbb{E}$", mean_value),
    # (r"$\operatorname{CVaR}_{0.8}$", cvar8),
    # (r"$\operatorname{CVaR}_{0.6}$", cvar6),
    # (r"$\operatorname{CVaR}_{0.4}$", cvar4),
    (r"$\operatorname{CVaR}_{0.2}$", cvar2),
    (r"$\operatorname{CVaR}_{0.1}$", cvar1),
    (r"$\operatorname{CVaR}_{0.05}$", cvar05),
    (r"$\operatorname{CVaR}_{0.02}$", cvar02),
]

AGG_RISK_VALUES5 = [
    (r"$\operatorname{CVaR}_{1.0}$", mean_value),
    (r"$\operatorname{CVaR}_{0.9}$", cvar9),
    (r"$\operatorname{CVaR}_{0.8}$", cvar8),
    (r"$\operatorname{CVaR}_{0.7}$", cvar7),
    (r"$\operatorname{CVaR}_{0.6}$", cvar6),
    (r"$\operatorname{CVaR}_{0.5}$", cvar5),
    (r"$\operatorname{CVaR}_{0.4}$", cvar4),
    (r"$\operatorname{CVaR}_{0.3}$", cvar3),
    (r"$\operatorname{CVaR}_{0.2}$", cvar2),
    (r"$\operatorname{CVaR}_{0.1}$", cvar1),
]

AGG_RISK_VALUES6 = [
    (r"$\mathbb{E}$", mean_value),
    # (r"$\operatorname{Exp}_{1.0}$", exp1),
    (r"$\operatorname{Exp}_{2.0}$", exp2),
    (r"$\operatorname{Exp}_{4.0}$", exp4),
    (r"$\operatorname{Exp}_{6.0}$", exp6),
    (r"$\operatorname{Exp}_{8.0}$", exp8),
]

AGG_RISK_VALUES7 = [
    (r"$\mathbb{E}$", mean_value),
    (r"$\operatorname{MC}_{0.2,0.8}$", srm08),
    (r"$\operatorname{MC}_{0.4,0.6}$", srm06),
    (r"$\operatorname{MC}_{0.6,0.4}$", srm04),
    (r"$\operatorname{MC}_{0.8,0.2}$", srm02),
    (r"$\operatorname{CVaR}_{0.2}$", cvar2),
]


def sharpe_and_drawdown(portfolio_series, risk_free_rate=0.0):
    """
    Assumes portfolio_series is sorted in time and contains portfolio values.
    Returns a Series with Sharpe ratio and Max drawdown.
    """
    returns = portfolio_series.pct_change().dropna()
    excess_returns = returns - risk_free_rate

    # Sharpe Ratio (using daily returns; scale by sqrt(252) for annual)
    sharpe = excess_returns.mean() / excess_returns.std(ddof=0) * np.sqrt(252) if excess_returns.std(ddof=0) > 0 else np.nan

    # Max Drawdown
    cumulative = portfolio_series.cummax()
    drawdowns = (portfolio_series - cumulative) / cumulative
    max_drawdown = drawdowns.min()

    return pd.Series({"sharpe": sharpe, "max_drawdown": max_drawdown})


def make_agent_hue_kws(experiments):
    pairs = [(exp["agent_name"], exp["color"]) for exp in experiments]
    agent_names, colors = zip(*pairs)
    hue_kws = dict(color=colors)
    return list(agent_names), hue_kws


def moving_average(values, window_size):
    # numpy.convolve uses zero for initial missing values, so is not suitable.
    numerator = np.nancumsum(values)
    # The sum of the last window_size values.
    numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
    # numerator[:window_size] = np.nan
    denominator = np.ones(len(values)) * window_size
    denominator[:window_size] = np.arange(1, window_size + 1)
    smoothed = numerator / denominator
    assert values.shape == smoothed.shape
    return smoothed


def smooth(df, smoothing_window, index_columns, columns):
    dfg = df.groupby(index_columns)
    for col in columns:
        df[col] = dfg[col].transform(lambda s: moving_average(s.values, smoothing_window))
    return df


def smooth_dataframe(df):
    return smooth(
        df,
        smoothing_window=10,
        index_columns=["agent", "environment_name", "agent_seed"],
        columns=[
            "value",
        ],
    )
