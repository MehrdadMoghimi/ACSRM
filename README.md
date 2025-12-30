# Actor-Critic with Static Spectral Risk Measure (AC-SRM)

This repository provides the official implementation for the paper: **"Risk-sensitive Actor-Critic with Static Spectral Risk Measure for Online and Offline Reinforcement Learning"**.

## 📝 Overview

This project introduces a novel actor-critic framework for optimizing static Spectral Risk Measures (SRMs) in both online and offline reinforcement learning settings. SRMs are a flexible family of risk measures that include well-known metrics like Conditional Value-at-Risk (CVaR) and Mean-CVaR.

The key innovation of this work is a bi-level optimization approach that decomposes the problem into an inner loop for policy optimization and an outer loop for updating a risk function based on the return distribution. This framework is designed to overcome the limitations of traditional methods that often lead to suboptimal policies, especially in scenarios where worst-case outcomes have severe consequences.

### Core Contributions

  * **A Unified Framework:** We propose a versatile actor-critic framework for optimizing static SRMs that is adaptable to both online and offline learning environments.
  * **Theoretical Guarantees:** The paper establishes convergence proofs for the proposed algorithm in a finite state-action setting.
  * **Stochastic and Deterministic Policies:** The framework supports both stochastic and deterministic policies, with specific implementations tailored for each.
  * **State-of-the-Art Performance:** Extensive empirical evaluations demonstrate that the proposed algorithms outperform existing risk-sensitive methods across a variety of domains, including finance, healthcare, and robotics.

## 🤖 Implemented Algorithms

This repository includes the following algorithms developed in the paper:

  * **AC-SRM:** An actor-critic with Spectral Risk Measure for online reinforcement learning.
  * **OAC-SRM:** An offline actor-critic with Spectral Risk Measure, which incorporates policy constraints to handle distributional shift.
  * **TD3-SRM:** A deterministic policy gradient-based approach for online learning, built upon the high-performing TD3 algorithm.
  * **TD3BC-SRM:** A deterministic policy gradient-based approach for offline learning, extending TD3BC with the SRM framework.

## 🔧 Getting Started

1.  Clone the repository:
    ```bash
    git clone https://github.com/MehrdadMoghimi/ACSRM.git
    cd ACSRM
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The implementations for the online and offline algorithms are provided. The hyperparameters for each model can be found in the appendix of the paper.


**Online Setting:**
The code for online algorithms like AC-SRM and TD3-SRM can be run in the provided environments. For detailed implementation, refer to Algorithm 2 in the paper.

```bash
python online/td3_srm.py --env-id HIVTreatment-v1 --save-model --dir runs/hiv --risk-measure CVaR --risk-alpha 0.2 --n-quantiles 100
```

**Offline Setting:**
For offline algorithms such as OAC-SRM and TD3BC-SRM, you will need to use the pre-collected datasets. The paper details the datasets used for each experiment, which are sourced from established benchmarks. The implementation details can be found in Algorithm 3.

```bash
python offline/td3bc_srm.py --env-id HIVTreatment-v1 --save-model  --dir runs/hiv_offline --data-dir 1R2R/hivtreatment-medium-replay-v0.pkl --n-quantiles 100 --risk-measure CVaR --risk-alpha 0.2
```

## Project Structure

The project is organized as follows:

```
.
├── custom_envs_gym.py            # Custom Gym environments
├── get_env.py                    # Utility for getting environments
├── LICENSE                       # Project License
├── README.md                     # This README file
├── requirements.txt              # Python package dependencies
├── reward_wrappers.py            # Wrappers for reward functions
├── utils.py                      # General utility functions
├── utils2.py                     # Additional utility functions
├── offline/                      # Implementations of offline RL algorithms
│   ├── oac_isrm.py
│   ├── oac_srm.py
│   ├── awac.py
│   ├── cql.py
│   ├── iql.py
│   ├── td3bc_isrm.py
│   ├── td3bc_srm.py
│   └── td3bc.py
└── online/                       # Implementations of online RL algorithms
    ├── ac_isrm.py
    ├── ac_srm.py
    ├── ac.py
    ├── sac_isrm.py
    ├── sac.py
    ├── td3_isrm.py
    ├── td3_srm.py
    └── td3.py
```


## 🔬 Experiments and Results

The algorithms were benchmarked against state-of-the-art risk-neutral and risk-sensitive methods in a variety of environments:

  * **Finance:** Mean-reverting trading and portfolio allocation tasks.
  * **Healthcare:** HIV treatment simulation.
  * **Robotics:** Stochastic MuJoCo continuous control tasks.

The results consistently show that optimizing for static SRMs leads to policies that are not only risk-averse but also achieve strong performance, often outperforming methods based on iterative risk measures.

### Key Findings

  * **State augmentation** is crucial for both the actor and the critic to ensure the final policy aligns with the static risk objective.
  * The framework allows for the generation of a **diverse set of risk-sensitive policies** from a single offline dataset, catering to different risk preferences.
  * **Deterministic policy variants (TD3-SRM and TD3BC-SRM)** showed particularly strong performance, suggesting that the added randomness of stochastic policies can sometimes hinder risk-sensitive objectives.

## 🙏 Acknowledgments

This project builds upon the excellent work of:

- [JAX-CORL](https://github.com/nissymori/JAX-CORL) - for providing inspiration and reference implementations for JAX-based RL algorithms
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - for high-quality, clean implementations of RL algorithms


## Citing this Work

If you use this code or the ideas presented in the paper for your research, please cite:

```bibtex
@article{moghimi2025risk,
  title={Risk-sensitive Actor-Critic with Static Spectral Risk Measures for Online and Offline Reinforcement Learning},
  author={Moghimi, Mehrdad and Ku, Hyejin},
  journal={arXiv preprint arXiv:2507.03900},
  year={2025}
}
```
