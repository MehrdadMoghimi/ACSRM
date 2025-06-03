import gym
from gym.spaces import Discrete, Box, Dict
import numpy as np
from gym.envs.registration import register
from scipy.integrate import solve_ivp
from gym.spaces.utils import flatten_space, flatten

from scipy.integrate import ODEintWarning
from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc

import warnings

warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import d4rl
    from d4rl import offline_env
    from d4rl.utils.wrappers import NormalizedBoxEnv

import pandas as pd


import yfinance as yf
import warnings

from pathlib import Path  # For easier path handling

from collections import deque
import scipy.linalg  # For Cholesky decomposition


class SyntheticPortfolioEnv(gym.Env):
    """
    A Gymnasium environment for portfolio allocation using synthetic data
    generated via Multivariate Geometric Brownian Motion (MV-GBM),
    calibrated from historical data.

    Args:
        asset_symbols (list[str]): List of ALL ticker symbols (e.g., ['AAPL', 'MSFT', 'CASH']).
        data_start_date (str): Start date for HISTORICAL data used for calibration.
        data_end_date (str): End date for HISTORICAL data used for calibration.
        # --- Simulation Parameters ---
        num_steps (int): Number of simulation steps per episode.
        time_step_size (float): Time increment for simulation step in years (e.g., 1/252 for daily).
        initial_prices (np.ndarray | None): Initial prices for non-cash assets. If None, use last historical price.
        override_mu (np.ndarray | float | None): Optionally override calibrated drift (mu). Can be scalar or vector for non-cash assets. None uses historical.
        # --- Env Parameters ---
        window_size (int): Number of past simulated returns in observation.
        initial_portfolio_value (float): Starting portfolio value.
        transaction_cost_pct (float): Proportional transaction cost.
        # --- Data Handling ---
        cache_dir (str): Directory for caching downloaded calibration data.
        force_download (bool): Force redownload ignoring cache.
        log_returns (bool): Use log returns internally (recommended for GBM).

    """

    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        asset_symbols,
        data_start_date,
        data_end_date,
        num_steps=21 * 3,  # Default to 3 months simulation
        time_step_size=1 / 252.0,
        initial_prices=None,
        override_mu=None,  # e.g., set to 0 for zero drift simulation
        window_size=21,
        initial_portfolio_value=1000.0,
        transaction_cost_pct=0.001,
        cache_dir="./synthetic_calib_cache",
        force_download=False,
        log_returns=True,
    ):  # Log returns are standard for GBM

        super().__init__()

        # --- Store Parameters ---
        self.all_asset_symbols = asset_symbols  # Includes CASH if present
        self.num_all_assets = len(self.all_asset_symbols)
        self.non_cash_asset_symbols = [s for s in asset_symbols if s.upper() != "CASH"]
        self.num_non_cash_assets = len(self.non_cash_asset_symbols)
        self.cash_present = "CASH" in self.all_asset_symbols
        self.cash_index = self.all_asset_symbols.index("CASH") if self.cash_present else -1

        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.cache_dir = Path(cache_dir)
        self.force_download = force_download
        self.num_steps = num_steps  # Simulation steps per episode
        self.time_step_size = time_step_size
        self.window_size = window_size
        self.initial_portfolio_value = initial_portfolio_value
        self.transaction_cost_pct = transaction_cost_pct
        self.override_mu = override_mu
        self.log_returns = log_returns  # Using log returns for GBM math

        # --- Calibrate GBM Parameters from Historical Data ---
        self._calibrate_gbm_params()  # Sets self.mu, self.sigma, self.corr, self.chol_L, self.last_historical_prices

        # --- Determine Initial Prices ---
        if initial_prices is None:
            # Use last historical price as starting point for simulation
            self._initial_prices_non_cash = self.last_historical_prices
        elif len(initial_prices) == self.num_non_cash_assets:
            self._initial_prices_non_cash = np.array(initial_prices, dtype=np.float64)
        else:
            raise ValueError(f"initial_prices must have length {self.num_non_cash_assets} or be None")

        # --- Action Space ---
        # Target weights for ALL assets (including cash)
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_all_assets,), dtype=np.float32)

        # --- Observation Space ---
        # 1. Past `window_size` simulated log returns for NON-CASH assets.
        # 2. Current portfolio weights for ALL assets.
        obs_shape = (self.num_non_cash_assets * self.window_size + self.num_all_assets,)
        # Bounds: Returns can be volatile, Weights [0, 1]
        low_bounds = np.concatenate(
            [
                np.full(self.num_non_cash_assets * self.window_size, -0.5),
                np.zeros(self.num_all_assets),
            ]  # Log returns bounds  # Weights bounds
        )
        high_bounds = np.concatenate(
            [
                np.full(self.num_non_cash_assets * self.window_size, 0.5),
                np.ones(self.num_all_assets),
            ]  # Log returns bounds  # Weights bounds
        )
        self.observation_space = Box(low=low_bounds, high=high_bounds, shape=obs_shape, dtype=np.float32)

        # --- Internal State Variables ---
        self._time_step = None
        self._portfolio_value = None
        self._portfolio_weights = None  # Current weights (w_t) for ALL assets
        self._current_prices_non_cash = None  # Current prices for NON-CASH assets
        # Store recent log returns for observation (non-cash assets only)
        self._log_return_history = None

    def _load_historical_data(self):
        """Loads historical data for NON-CASH assets for calibration."""
        all_data = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading historical data for calibration ({self.non_cash_asset_symbols})...")
        for symbol in self.non_cash_asset_symbols:
            cache_filename = f"{symbol}_{self.data_start_date.replace('-', '')}_{self.data_end_date.replace('-', '')}.csv"
            cache_filepath = self.cache_dir / cache_filename
            data = None
            if not self.force_download and cache_filepath.exists():
                try:
                    data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True, header=[0, 1])
                except Exception:
                    data = None
            if data is None:
                try:
                    data = yf.download(
                        symbol,
                        start=self.data_start_date,
                        end=self.data_end_date,
                        progress=False,
                    )
                    if data.empty:
                        raise ValueError(f"No data for {symbol}")
                    data.to_csv(cache_filepath)
                except Exception as e:
                    raise ConnectionError(f"Failed download/cache for {symbol}: {e}")
            if "Close" not in data.columns:
                raise ValueError(f"'Close' not found for {symbol}")
            all_data[symbol] = data["Close"]

        df_prices = pd.concat(all_data, axis=1).ffill().bfill()  # Combine and fill NaNs
        return df_prices[self.non_cash_asset_symbols]  # Ensure correct column order

    def _calibrate_gbm_params(self):
        """Estimates MV-GBM parameters (mu, sigma, corr, L) from historical data."""
        df_prices = self._load_historical_data()
        if df_prices.empty or df_prices.isnull().values.any():
            raise ValueError("Historical price data for calibration is empty or contains NaNs after processing.")
        self.last_historical_prices = df_prices.iloc[-1].to_numpy()

        # Calculate log returns for non-cash assets
        log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

        if log_returns.empty:
            raise ValueError("Not enough historical data to calculate returns for calibration.")

        # Estimate annualized drift (mu) - Use with caution!
        mean_log_ret = log_returns.mean().to_numpy()
        # Estimate annualized covariance matrix
        cov_matrix_log_ret = log_returns.cov().to_numpy() * (1 / self.time_step_size)  # Annualize covariance

        # Estimate annualized volatility (sigma) vector
        self.sigma = np.sqrt(np.diag(cov_matrix_log_ret))  # Annualized std dev

        # Estimate drift (mu)
        if self.override_mu is not None:
            if np.isscalar(self.override_mu):
                self.mu = np.full(self.num_non_cash_assets, self.override_mu)
                print(f"Using overridden drift (mu): {self.mu}")
            elif len(self.override_mu) == self.num_non_cash_assets:
                self.mu = np.array(self.override_mu)
                print(f"Using overridden drift (mu): {self.mu}")
            else:
                raise ValueError("override_mu must be scalar or vector of length num_non_cash_assets")
        else:
            # mu = E[log(S_t/S_0)]/T + 0.5*sigma^2
            self.mu = (mean_log_ret / self.time_step_size) + 0.5 * self.sigma**2
            # self.mu += 0.8
            print(f"Using calibrated historical drift (mu): {self.mu}")
            warnings.warn("Historical drift is often unstable; consider overriding `mu`.")

        # Estimate correlation matrix
        # Avoid recomputing from covariance for numerical stability if vols are tiny
        if np.any(self.sigma < 1e-8):
            warnings.warn("Near-zero volatility found for some assets. Correlation matrix might be unreliable.")
            # Fallback: Use correlation from original log returns if needed
            self.corr = log_returns.corr().to_numpy()
        else:
            inv_sigma_diag = np.diag(1.0 / self.sigma)
            # Recalculate cov matrix to ensure consistency before calculating corr
            cov_matrix_adj = cov_matrix_log_ret  # Already annualized
            self.corr = inv_sigma_diag @ cov_matrix_adj @ inv_sigma_diag

        # Ensure correlation matrix is valid (symmetric, diag=1, PSD)
        self.corr = (self.corr + self.corr.T) / 2  # Enforce symmetry
        np.fill_diagonal(self.corr, 1.0)
        # Add small jitter for positive semi-definiteness if needed before Cholesky
        min_eig = np.min(np.real(scipy.linalg.eigvals(self.corr)))
        if min_eig < 1e-10:  # Allow small tolerance
            print(f"Warning: Correlation matrix not positive semi-definite (min eig={min_eig}). Adding jitter.")
            jitter = max(0, -min_eig) + 1e-10
            self.corr += np.eye(self.num_non_cash_assets) * jitter
            self.corr /= np.sqrt(np.outer(np.diag(self.corr), np.diag(self.corr)))  # Renormalize diag to 1

        # Calculate Cholesky decomposition of the COVARIANCE matrix scaled by dt
        # cov_dt = diag(sigma) @ corr @ diag(sigma) * dt
        scaled_cov = np.diag(self.sigma) @ self.corr @ np.diag(self.sigma) * self.time_step_size
        try:
            self.chol_L = scipy.linalg.cholesky(scaled_cov, lower=True)
        except scipy.linalg.LinAlgError:
            print("ERROR: Scaled covariance matrix not positive definite for Cholesky.")
            # Fallback or error handling needed - e.g., use nearest PSD matrix
            # For now, raise error
            raise ValueError("Could not compute Cholesky decomposition. Check data or parameters.")

        print("Calibration complete.")
        print(f"  Sigma (Annualized Vol): {self.sigma}")
        print(f"  Correlation Matrix:\n{self.corr}")  # Optional: Print if needed

    def _simulate_gbm_step(self):
        """Simulates one step of MV-GBM for non-cash assets."""
        # Generate independent standard normal variables
        Z = self.np_random.standard_normal(self.num_non_cash_assets)
        # Generate correlated Wiener increments using Cholesky
        dW = self.chol_L @ Z  # Correlated increments for the time step

        # Calculate log returns for this step: log(S_t / S_{t-1}) = (mu - 0.5*sigma^2)*dt + dW
        step_log_returns = (self.mu - 0.5 * self.sigma**2) * self.time_step_size + dW

        # Calculate new prices: S_t = S_{t-1} * exp(log return)
        new_prices = self._current_prices_non_cash * np.exp(step_log_returns)

        return new_prices, step_log_returns

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._time_step = 0
        self._portfolio_value = self.initial_portfolio_value
        # Start with initial prices for non-cash assets
        self._current_prices_non_cash = self._initial_prices_non_cash.copy()

        # Initial weights (e.g., all in cash if available, else equal weight)
        self._portfolio_weights = np.zeros(self.num_all_assets)
        if self.cash_present:
            self._portfolio_weights[self.cash_index] = 1.0
        else:
            self._portfolio_weights[:] = 1.0 / self.num_all_assets

        # Initialize log return history deque with zeros (or simulate initial steps)
        self._log_return_history = deque(
            np.zeros((self.window_size, self.num_non_cash_assets)),
            maxlen=self.window_size,
        )
        # Optional: Could simulate initial steps to populate history more realistically

        observation = self._get_obs()
        info = self._get_info()

        return observation  # , info

    def step(self, action):
        """Executes one time step within the environment."""
        # 0. Store previous state
        previous_value = self._portfolio_value
        previous_weights = self._portfolio_weights.copy()  # Weights for ALL assets

        # 1. Process Action (Normalize target weights)
        if np.sum(action) <= 1e-8:
            target_weights = np.zeros(self.num_all_assets)
            if self.cash_present:
                target_weights[self.cash_index] = 1.0
            else:
                target_weights[:] = 1.0 / self.num_all_assets
        else:
            target_weights = action / np.sum(action)  # Normalize

        # 2. Calculate Transaction Costs (Based on ALL assets including cash)
        weight_change = target_weights - previous_weights
        traded_value = np.sum(np.abs(weight_change)) * previous_value
        transaction_costs = traded_value * self.transaction_cost_pct

        # 3. Value after costs, before market move
        value_after_costs = previous_value - transaction_costs

        # 4. Simulate Market Returns for the current step (for non-cash assets)
        new_prices_non_cash, step_log_returns_non_cash = self._simulate_gbm_step()

        # --- 5. Calculate new portfolio value after market move ---
        # Need simple returns (exp(log_return) - 1) for value calculation
        step_simple_returns_non_cash = np.exp(step_log_returns_non_cash) - 1.0

        # Create full simple return vector (including 0 for cash)
        step_simple_returns_full = np.zeros(self.num_all_assets)
        non_cash_indices = [i for i, s in enumerate(self.all_asset_symbols) if s.upper() != "CASH"]
        step_simple_returns_full[non_cash_indices] = step_simple_returns_non_cash
        # Cash return is implicitly 0

        # Value evolves based on target weights applied to value_after_costs
        portfolio_return_factor = np.sum(target_weights * (1 + step_simple_returns_full))
        self._portfolio_value = value_after_costs * portfolio_return_factor

        # Floor value
        if self._portfolio_value < 1e-6:
            self._portfolio_value = 1e-6

        # --- 6. Calculate new actual portfolio weights (ALL assets) ---
        asset_values_end = (value_after_costs * target_weights) * (1 + step_simple_returns_full)
        self._portfolio_weights = asset_values_end / self._portfolio_value
        self._portfolio_weights /= np.sum(self._portfolio_weights)  # Renormalize

        # --- 7. Calculate Reward ---
        # Simple return (or log return) of the portfolio value
        if self.log_returns:
            reward = np.log(self._portfolio_value / previous_value) if previous_value > 1e-9 else 0.0
        else:
            reward = (self._portfolio_value / previous_value) - 1.0
        # reward = self._portfolio_value - previous_value

        # --- 8. Update Internal State ---
        self._time_step += 1
        self._current_prices_non_cash = new_prices_non_cash  # Update non-cash prices
        # Add the new simulated log returns to history
        self._log_return_history.append(step_log_returns_non_cash)

        # --- 9. Check Termination Conditions ---
        terminated = self._portfolio_value <= 1e-3 * self.initial_portfolio_value  # Ruin
        truncated = self._time_step >= self.num_steps

        # --- 10. Get New Observation & Info ---
        observation = self._get_obs()
        info = self._get_info()
        info["step_log_returns_non_cash"] = step_log_returns_non_cash

        return observation, reward, terminated or truncated, info

    def _get_obs(self):
        """Constructs the observation array."""
        # Get window_size log returns history for non-cash assets
        # Deque automatically handles the window
        returns_history = np.array(self._log_return_history)

        # Flatten returns history
        flat_returns_history = returns_history.flatten()

        # Concatenate non-cash returns history and ALL current weights
        obs = np.concatenate((flat_returns_history, self._portfolio_weights)).astype(np.float32)

        # Ensure observation fits bounds (clipping)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _get_info(self):
        """Returns supplementary information about the state."""
        # Combine non-cash prices with cash value (implicitly 1.0) if needed
        current_prices_full = {}
        nc_idx = 0
        for i, symbol in enumerate(self.all_asset_symbols):
            if symbol.upper() == "CASH":
                current_prices_full[symbol] = 1.0
            else:
                if nc_idx < len(self._current_prices_non_cash):
                    current_prices_full[symbol] = self._current_prices_non_cash[nc_idx]
                else:  # Should not happen if indices align
                    current_prices_full[symbol] = np.nan
                nc_idx += 1

        return {
            "time_step": self._time_step,
            "portfolio_value": self._portfolio_value,
            "portfolio_weights": self._portfolio_weights.copy(),
            "current_prices": current_prices_full,  # Dictionary format might be easier
        }

    def render(self, mode="human"):
        print(f"Step: {self._time_step}, Portfolio Value: {self._portfolio_value:.2f}")

    def close(self):
        pass

    def get_normalized_score(env_name, score):
        ref_min_score = 0.0  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 100.0  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


class PortfolioAllocationEnv(gym.Env):
    """
    A Gymnasium environment for portfolio allocation across multiple assets.

    The agent observes historical returns and current portfolio weights,
    and chooses target weights for the next period. The environment simulates
    rebalancing, transaction costs, and portfolio value evolution.

    Args:
        asset_symbols (list[str]): List of ticker symbols for assets (e.g., ['AAPL', 'MSFT', 'AGG']). Include 'CASH' if desired (returns treated as 0).
        data_start_date (str): Start date for historical data.
        data_end_date (str): End date for historical data.
        is_training (bool): Flag to select train/test data split.
        train_test_split (float): Proportion of data for training.
        cache_dir (str): Directory for caching downloaded data.
        force_download (bool): Force redownload ignoring cache.
        window_size (int): Number of past time steps of returns to include in the observation.
        initial_portfolio_value (float): Starting value of the portfolio.
        transaction_cost_pct (float): Proportional cost for buying/selling (applied to traded value).
        episode_length (int): Maximum number of steps per episode.
        random_start (bool): If True, each episode starts at a random point in the data slice.
        log_returns (bool): If True, use log returns for reward calculation and observation.
    """

    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        asset_symbols,
        is_training,
        data_start_date="1980-01-01",
        data_end_date="2024-12-31",
        train_test_split=0.8,
        cache_dir="./portfolio_data_cache",
        force_download=False,
        window_size=5,
        initial_portfolio_value=100.0,
        transaction_cost_pct=0.001,  # e.g., 0.1%
        episode_length=21,  # e.g., one trading year
        random_start=True,
        log_returns=False,
    ):

        super().__init__()

        # --- Keep track of non-cash assets ---
        self.non_cash_asset_symbols = [s for s in asset_symbols if s.upper() != "CASH"]
        self.num_non_cash_assets = len(self.non_cash_asset_symbols)
        self.cash_present = "CASH" in asset_symbols
        if self.cash_present:
            self.cash_index = asset_symbols.index("CASH")  # Store index if cash exists
        else:
            self.cash_index = -1

        self.asset_symbols = asset_symbols
        self.num_assets = len(asset_symbols)
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.is_training = is_training
        self.train_test_split = train_test_split
        self.cache_dir = Path(cache_dir)
        self.force_download = force_download
        self.window_size = window_size
        self.initial_portfolio_value = initial_portfolio_value
        self.transaction_cost_pct = transaction_cost_pct
        self.episode_length = episode_length
        self.random_start = random_start
        self.log_returns = log_returns  # Use log returns?

        # --- Data Loading & Processing ---
        self._load_data()  # Loads returns for all assets into self._returns (shape: [time, num_assets])
        self._data_len = len(self._returns)

        if self._data_len < self.window_size + self.episode_length:
            warnings.warn("Data length is possibly too short for window_size + episode_length.")

        # --- Action Space ---
        # Agent outputs target weights for each asset (including cash if specified).
        # Weights should ideally sum to 1. We handle normalization in step().
        # Using [0, 1] bounds assumes long-only. Use [-1, 1] or larger if shorting is allowed.
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)

        # --- Observation Space ---
        # Consists of:
        # 1. Past `window_size` returns for each asset.
        # 2. Current portfolio weights for each asset.
        obs_shape = (self.num_non_cash_assets * self.window_size + self.num_assets,)
        # Define reasonable bounds for returns (e.g., -0.5 to 0.5 daily return)
        # Weights bounds are [0, 1].
        low_bounds = np.concatenate(
            [
                np.full(self.num_non_cash_assets * self.window_size, -0.5),  # Returns part (non-cash only)
                np.zeros(self.num_assets),  # Weights part (all assets)
            ]
        )
        high_bounds = np.concatenate(
            [
                np.full(self.num_non_cash_assets * self.window_size, 0.5),  # Returns part (non-cash only)
                np.ones(self.num_assets),  # Weights part (all assets)
            ]
        )
        self.observation_space = Box(low=low_bounds, high=high_bounds, shape=obs_shape, dtype=np.float32)

        # --- Internal State Variables ---
        self._current_tick = None
        self._start_tick = None
        self._time_step = None  # Steps within the current episode
        self._portfolio_value = None
        self._portfolio_weights = None  # Current weights (w_t)

    def _load_data(self):
        """Loads and prepares data for all assets, handling caching and alignment."""
        all_data = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for symbol in self.asset_symbols:
            if symbol.upper() == "CASH":
                continue  # Skip download for cash

            cache_filename = f"{symbol}_{self.data_start_date.replace('-', '')}_{self.data_end_date.replace('-', '')}.csv"
            cache_filepath = self.cache_dir / cache_filename
            data = None
            if not self.force_download and cache_filepath.exists():
                try:
                    data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True, header=[0, 1])
                    if data.empty or "Close" not in data.columns:
                        data = None
                except Exception:
                    data = None

            if data is None:
                print(f"Downloading data for {symbol}...")
                try:
                    data = yf.download(
                        symbol,
                        start=self.data_start_date,
                        end=self.data_end_date,
                        progress=False,
                    )
                    if data.empty:
                        raise ValueError(f"No data for {symbol}")
                    data.to_csv(cache_filepath)
                except Exception as e:
                    raise ConnectionError(f"Failed download/cache for {symbol}: {e}")

            all_data[symbol] = data["Close"]  # Store Close prices

        # Combine all series into a single DataFrame, forward-fill missing values
        df_prices = pd.concat(all_data, axis=1).ffill()  # .bfill() # Forward and back fill to handle initial NaNs

        # Add cash column if specified
        if "CASH" in self.asset_symbols:
            df_prices["CASH"] = 1.0  # Cash price is constant

        # Calculate returns
        if self.log_returns:
            df_returns = np.log(df_prices / df_prices.shift(1))
        else:
            df_returns = df_prices.pct_change()

        # df_returns[self.non_cash_asset_symbols] = df_returns[self.non_cash_asset_symbols] + 1.0 / 252.0

        # Drop initial NaN row and align symbols according to self.asset_symbols
        df_returns = df_returns.dropna()
        all_returns_np = df_returns.loc[:, self.asset_symbols].to_numpy()

        if len(all_returns_np) < 2:
            raise ValueError("Not enough data points after processing.")

        # Split data
        split_index = int(len(all_returns_np) * self.train_test_split)
        if self.is_training:
            self._returns = all_returns_np[:split_index]
            self._return_dates = df_returns.index[:split_index]
        else:
            self._returns = all_returns_np[split_index:]
            self._return_dates = df_returns.index[split_index:]

        print(f"Loaded {'Train' if self.is_training else 'Test'} data: {self._returns.shape[0]} steps, {self._returns.shape[1]} assets.")

    def step(self, action):
        """Executes one time step within the environment."""
        # 0. Store previous state for reward calculation
        previous_value = self._portfolio_value
        previous_weights = self._portfolio_weights

        # 1. Process Action (Normalize target weights to sum to 1)
        if np.sum(action) <= 1e-8:  # Handle case where agent outputs all zeros
            # Default to equal weight or all cash? Let's do all cash if available.
            if "CASH" in self.asset_symbols:
                target_weights = np.zeros(self.num_assets)
                target_weights[self.asset_symbols.index("CASH")] = 1.0
            else:  # Fallback to equal weight if no cash
                target_weights = np.ones(self.num_assets) / self.num_assets
        else:
            target_weights = action / np.sum(action)  # Normalize weights

        # 2. Calculate Transaction Costs based on weight changes
        weight_change = target_weights - previous_weights
        traded_value = np.sum(np.abs(weight_change)) * previous_value
        transaction_costs = traded_value * self.transaction_cost_pct

        # 3. Value after costs, before market move
        value_after_costs = previous_value - transaction_costs

        # 4. Get Market Returns for the current step
        # Ensure current_tick is valid
        if self._current_tick >= self._data_len:
            # This should ideally be caught by truncation logic, but handle defensively
            warnings.warn("Accessed data beyond available length.")
            step_returns = np.zeros(self.num_assets)  # Assume zero return if out of data
        else:
            step_returns = self._returns[self._current_tick]

        # 5. Calculate new portfolio value after market move
        # Value evolves based on target weights applied to value_after_costs
        portfolio_return_factor = np.sum(target_weights * (1 + step_returns))  # - 1.0 / 252.0))
        self._portfolio_value = value_after_costs * portfolio_return_factor

        # Handle potential division by zero if portfolio value becomes negligible
        if self._portfolio_value < 1e-6:
            self._portfolio_value = 1e-6  # Floor value to avoid errors

        # 6. Calculate new actual portfolio weights
        asset_values_end = (value_after_costs * target_weights) * (1 + step_returns)
        self._portfolio_weights = asset_values_end / self._portfolio_value
        # Renormalize weights due to potential floating point inaccuracies
        self._portfolio_weights /= np.sum(self._portfolio_weights)

        # 7. Calculate Reward
        # Simple log return of the portfolio value
        reward = np.log(self._portfolio_value / previous_value)
        # Or, reward = portfolio return net of costs
        # reward = self._portfolio_value - previous_value

        # --- For Risk-Sensitive RL (Option 1: Reward Shaping) ---
        # You would calculate risk here and subtract a penalty from 'reward'
        # E.g., estimate covariance from recent history (using self._returns)
        # recent_returns = self._returns[self._current_tick-self.window_size+1 : self._current_tick+1]
        # if recent_returns.shape[0] >= 2: # Need at least 2 points for covariance
        #    cov_matrix = np.cov(recent_returns, rowvar=False)
        #    portfolio_variance = target_weights @ cov_matrix @ target_weights.T
        #    risk_penalty = some_lambda * portfolio_variance
        #    reward -= risk_penalty
        # ---------------------------------------------------------

        # 8. Update Time Pointers
        self._time_step += 1
        self._current_tick += 1

        # 9. Check Termination Conditions
        terminated = self._portfolio_value <= 1e-3 * self.initial_portfolio_value  # Ruin condition
        truncated = (self._time_step >= self.episode_length) or (self._current_tick >= self._data_len)

        # 10. Get New Observation
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated or truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self._time_step = 0
        self._portfolio_value = self.initial_portfolio_value

        # Initial weights (e.g., all in cash if available, else equal weight)
        self._portfolio_weights = np.zeros(self.num_assets)
        if "CASH" in self.asset_symbols:
            self._portfolio_weights[self.asset_symbols.index("CASH")] = 1.0
        else:
            self._portfolio_weights[:] = 1.0 / self.num_assets

        # Determine starting tick
        if self.random_start:
            # Need at least window_size history AND episode_length future steps
            max_start_tick = self._data_len - self.episode_length - 1
            min_start_tick = self.window_size - 1
            if max_start_tick <= min_start_tick:
                warnings.warn("Data too short for random start with specified window/episode length. Starting at window_size.")
                self._start_tick = min_start_tick
            else:
                self._start_tick = self.np_random.integers(min_start_tick, max_start_tick + 1)
        else:
            self._start_tick = self.window_size - 1  # Need window_size history

        self._current_tick = self._start_tick

        observation = self._get_obs()
        # info = self._get_info()

        return observation  # , info

    def _get_obs(self):
        """Constructs the observation array, excluding cash return history."""
        # Get window_size returns history ending at the current tick for ALL assets
        # Note: self._current_tick points to the return for the step *about* to be used
        # Observation should reflect data available *before* the current step's return is known.
        # So we use data up to self._current_tick - 1
        end_idx = self._current_tick
        start_idx = max(0, end_idx - self.window_size)
        returns_history_full = self._returns[start_idx:end_idx]  # Shape: [actual_len, num_assets]

        # Pad if history is shorter than window_size (at the beginning of data)
        actual_len = returns_history_full.shape[0]
        if actual_len < self.window_size:
            padding = np.zeros((self.window_size - actual_len, self.num_assets))
            returns_history_full = np.concatenate((padding, returns_history_full), axis=0)

        # --- Select ONLY non-cash asset columns ---
        if self.cash_present:
            # Create a boolean mask to select non-cash columns
            non_cash_mask = np.ones(self.num_assets, dtype=bool)
            non_cash_mask[self.cash_index] = False
            returns_history_non_cash = returns_history_full[:, non_cash_mask]
        else:
            # If no cash, use all columns
            returns_history_non_cash = returns_history_full

        # Flatten returns history (non_cash_assets x time) -> (non_cash_assets * time)
        flat_returns_history = returns_history_non_cash.flatten()

        # Concatenate non-cash returns history and ALL current weights
        obs = np.concatenate((flat_returns_history, self._portfolio_weights)).astype(np.float32)

        # Ensure observation fits bounds (clipping) - bounds already adjusted in __init__
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _get_info(self):
        """Returns supplementary information about the state."""
        return {
            "current_tick": self._current_tick,
            "time_step": self._time_step,
            "portfolio_value": self._portfolio_value,
            "portfolio_weights": self._portfolio_weights.copy(),  # Return copy
            "transaction_costs": 0,  # Placeholder - could calculate and return costs in step() if needed
        }

    def get_env_info(self):
        return {
            "returns": self._returns,
            "return_dates": self._return_dates,
            "data_len": self._data_len,
            "start_tick": self._start_tick,
        }

    def render(self, mode="human"):
        # Implement basic rendering if desired (e.g., print portfolio value)
        print(f"Step: {self._time_step}, Tick: {self._current_tick}, Portfolio Value: {self._portfolio_value:.2f}")

    def close(self):
        pass  # Cleanup resources if needed

    def get_normalized_score(env_name, score):
        ref_min_score = 0.0  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 100.0  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)

    def get_normalized_score2(env_name, score):
        ref_min_score = -0.0776  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 0.024863  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)

    def get_normalized_score3(env_name, score):
        ref_min_score = -0.08150  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 0.028109  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


class RealTradingEnv(gym.Env):
    """
    A stock trading environment using historical data from Yahoo Finance with local caching.

    Args:
        stock_symbol (str): The ticker symbol for the stock (e.g., 'AAPL').
        is_training (bool): If True, uses the training data split; otherwise uses the test data split.
        data_start_date (str): Start date for data download (e.g., '2010-01-01').
        data_end_date (str): End date for data download (e.g., '2023-12-31').
        cache_dir (str): Directory to store/load downloaded data cache (default: './data_cache').
        force_download (bool): If True, ignore cache and force download (default: False).
        train_test_split (float): Proportion of data to use for training (default: 0.7).
        params (dict, optional): Dictionary of environment parameters. Defaults include:
            - phi (float): Transaction cost coefficient (quadratic cost).
            - psi (float): Terminal inventory penalty coefficient (quadratic cost).
            - episode_length (int): Number of steps in one episode.
            - max_q (float): Maximum absolute inventory allowed.
            - max_u (float): Maximum absolute trade size allowed per step.
            - initial_price (float): Price to start simulation at in reset().
            - price_obs_scale (float): Scaling factor for price observation bounds.
            - random_start (bool): If True, pick a random start point in the data for each episode.
    """

    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        stock_symbol,
        is_training,
        data_start_date="1970-01-01",
        data_end_date="2025-12-31",
        train_test_split=0.8,
        cache_dir="./data_cache",
        force_download=False,
        params=None,
    ):  # Added cache_dir, force_download
        super(RealTradingEnv, self).__init__()

        # --- Data Loading and Processing ---
        self.stock_symbol = stock_symbol
        self.is_training = is_training
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.train_test_split = train_test_split
        self.cache_dir = Path(cache_dir)  # Use Path object
        self.force_download = force_download
        self._load_data()  # Load and process data (now uses caching)

        # --- Default and User Parameters ---
        default_params = {
            "phi": 0.001,
            "psi": 0.1,
            "episode_length": 21,
            "max_q": 5.0,
            "max_u": 2.0,
            "initial_price": 1.0,
            "price_obs_scale": 3.0,
            "random_start": True,
        }
        self.params = default_params if params is None else {**default_params, **params}

        # Ensure episode length isn't longer than available data
        if self.params["episode_length"] > len(self._returns):
            warnings.warn(
                f"episode_length ({self.params['episode_length']}) > available data steps ({len(self._returns)}). Setting episode_length to {len(self._returns)}."
            )
            self.params["episode_length"] = len(self._returns)

        # --- Action Space ---
        self.action_space = Box(
            low=np.array([-self.params["max_u"]]),
            high=np.array([self.params["max_u"]]),
            shape=(1,),
            dtype=np.float32,
        )

        # --- Observation Space ---
        low_price_bound = self.params["initial_price"] / self.params["price_obs_scale"]
        high_price_bound = self.params["initial_price"] * self.params["price_obs_scale"]
        self.observation_space = Box(
            low=np.array([low_price_bound, -self.params["max_q"], 0]),
            high=np.array([high_price_bound, self.params["max_q"], self.params["episode_length"]]),
            shape=(3,),
            dtype=np.float32,
        )

        # --- Environment State Variables ---
        self._stock_price = None
        self._agent_inventory = None
        self._time_step = None
        self._current_tick = None
        self._start_tick = None

    def _construct_cache_filename(self):
        """Creates a unique filename for caching based on parameters."""
        # Sanitize dates for filename
        start_str = self.data_start_date.replace("-", "")
        end_str = self.data_end_date.replace("-", "")
        return f"{self.stock_symbol}_{start_str}_{end_str}.csv"

    def _load_data(self):
        """
        Downloads data from Yahoo Finance or loads from local cache,
        calculates returns, and splits into train/test sets.
        """
        cache_filename = self._construct_cache_filename()
        cache_filepath = self.cache_dir / cache_filename  # Use / operator from Pathlib

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        data = None  # Initialize data DataFrame

        # --- Try loading from cache first ---
        if not self.force_download and cache_filepath.exists():
            print(f"Attempting to load data from cache: {cache_filepath}")
            try:
                data = pd.read_csv(cache_filepath, header=[0, 1], index_col=0, parse_dates=True)
                if data.empty:
                    print("Cache file is empty. Will attempt download.")
                    data = None  # Treat as cache miss
                else:
                    # Quick check if data looks reasonable (e.g., has Close)
                    if "Close" not in data.columns:
                        print("Cached data missing 'Close'. Will attempt download.")
                        data = None
                    else:
                        print(f"Successfully loaded data for {self.stock_symbol} from cache.")
            except Exception as e:
                print(f"Failed to load data from cache ({cache_filepath}): {e}. Will attempt download.")
                data = None  # Ensure data is None if loading fails

        # --- Download if cache missed or forced ---
        if data is None:
            if self.force_download:
                print(f"Forcing download for {self.stock_symbol}...")
            else:
                print(f"Cache not found or invalid. Downloading {self.stock_symbol} data from {self.data_start_date} to {self.data_end_date}...")

            try:
                data = yf.download(
                    self.stock_symbol,
                    start=self.data_start_date,
                    end=self.data_end_date,
                    progress=False,
                )
                if data.empty:
                    raise ValueError(f"No data found for {self.stock_symbol} in the specified date range.")
                print("Download complete.")

                # --- Save to cache after successful download ---
                try:
                    data.to_csv(cache_filepath, index=True)  # Save with Date index
                    print(f"Data saved to cache: {cache_filepath}")
                except Exception as e:
                    print(f"Warning: Failed to save data to cache ({cache_filepath}): {e}")

            except Exception as e:
                # If download fails, re-raise the error after cleaning up potential partial cache file
                if cache_filepath.exists():
                    try:
                        cache_filepath.unlink()  # Remove potentially corrupted cache file
                    except OSError as rm_err:
                        print(f"Warning: Could not remove incomplete cache file {cache_filepath}: {rm_err}")
                raise ConnectionError(f"Failed to download data for {self.stock_symbol}: {e}")

        # --- Process Data (Calculate Returns) ---
        if "Close" not in data.columns:
            raise ValueError("Downloaded data does not contain 'Close' column.")

        # Use Close for returns to account for dividends and splits
        data["returns"] = data["Close"].pct_change()
        # Drop the first row with NaN return
        all_returns = data["returns"].dropna().to_numpy()

        if len(all_returns) < 2:
            raise ValueError("Not enough data points to calculate returns after dropping NaNs.")

        # --- Split Data ---
        split_index = int(len(all_returns) * self.train_test_split)

        if self.is_training:
            self._returns = all_returns[:split_index]
            print(f"Using training data: {len(self._returns)} steps.")
        else:
            self._returns = all_returns[split_index:]
            print(f"Using test data: {len(self._returns)} steps.")

        if len(self._returns) == 0:
            raise ValueError("Selected data slice (train or test) is empty.")

        self._data_len = len(self._returns)

    def step(self, action):
        # Clip action to ensure it's within bounds (redundant if using compliant RL libraries, but safe)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        trade_size = action[0]

        # Store current price before update for reward calculation
        current_price = self._stock_price

        # Calculate reward *before* updating state for next step
        reward = -current_price * trade_size - self.params["phi"] * np.power(trade_size, 2)

        # --- Update State ---
        # 1. Update Price based on historical return
        current_return = self._returns[self._current_tick]
        self._stock_price = current_price * (1 + current_return)
        # self._stock_price = np.clip(self._stock_price, self.observation_space.low[0], self.observation_space.high[0])

        # 2. Update Inventory
        self._agent_inventory += trade_size
        # self._agent_inventory = np.clip(self._agent_inventory, -self.params["max_q"], self.params["max_q"])

        # 3. Update Time Step counters
        self._time_step += 1
        self._current_tick += 1  # Move to the next data point

        # --- Check Termination Conditions ---
        terminated = False

        if self._time_step >= self.params["episode_length"]:
            terminated = True
        if self._current_tick >= self._data_len:
            terminated = True

        # --- Add Terminal Reward if Episode Ended ---
        if terminated:
            terminal_pnl = self._agent_inventory * self._stock_price
            terminal_penalty = self.params["psi"] * np.power(self._agent_inventory, 2)
            reward += terminal_pnl - terminal_penalty

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._time_step = 0
        self._agent_inventory = 0.0
        self._stock_price = self.params["initial_price"]

        if self.params["random_start"]:
            max_start_tick = self._data_len - self.params["episode_length"] - 1
            if max_start_tick <= 0:
                self._start_tick = 0
                if self._data_len < self.params["episode_length"]:
                    warnings.warn(
                        f"Data length ({self._data_len}) is less than episode_length ({self.params['episode_length']}). Episode will be shorter."
                    )
            else:
                self._start_tick = self.np_random.integers(0, max_start_tick + 1)
        else:
            self._start_tick = 0

        self._current_tick = self._start_tick

        observation = self._get_obs()
        # info = self._get_info()

        return observation  # , info

    def _get_obs(self):
        obs = np.array(
            [
                np.clip(
                    self._stock_price,
                    self.observation_space.low[0],
                    self.observation_space.high[0],
                ),
                np.clip(
                    self._agent_inventory,
                    self.observation_space.low[1],
                    self.observation_space.high[1],
                ),
                self._time_step,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self):
        return {
            "current_tick": self._current_tick,
            "start_tick": self._start_tick,
            "time_step": self._time_step,
            "stock_price": self._stock_price,
            "inventory": self._agent_inventory,
        }

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def get_normalized_score(env_name, score):
        ref_min_score = 0.0  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 100.0  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


class TradingEnv(gym.Env):
    def __init__(self, params=None):
        super(TradingEnv, self).__init__()

        # default parameters for the model
        default_params = {
            "kappa": 2,  # kappa of the OU process
            "sigma": 1,  # 0.2,  # standard deviation of the OU process
            "theta": 1,  # mean-reversion level of the OU process
            "phi": 0.005,  # transaction costs
            "psi": 0.5,  # terminal penalty on the inventory
            "T": 1,  # trading horizon
            "Ndt": 10,  # number of periods
            "max_q": 5,  # maximum value for the inventory
            "max_u": 2,  # maximum value for the trades
            "random_reset": False,  # reset the inventory to a random value between -max_q and max_q if True, otherwise reset to 0
        }

        self.params = default_params if params is None else {**default_params, **params}

        self.action_space = Box(
            low=np.array([-self.params["max_u"]]),
            high=np.array([self.params["max_u"]]),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space: state representing the stock price, the agent's inventory and the current time step
        self.observation_space = Box(
            low=np.array(
                [
                    self.params["theta"] - 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
                    -self.params["max_q"],
                    0,
                ]
            ),
            high=np.array(
                [
                    self.params["theta"] + 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
                    self.params["max_q"],
                    self.params["Ndt"],
                ]
            ),
            shape=(3,),
            dtype=np.float32,
        )

    def step(self, action):
        # if not self.action_space.contains(action):
        # assert self.action_space.contains(action), "Invalid action"

        # reward is calculated with the current stock price and the current action
        # Only is time = T-1, the reward also includes the terminal penalty on the inventory
        reward = -self._stock_price * action[0] - self.params["phi"] * np.power(action[0], 2)

        # price of the stock at next time step - OU process
        dt = self.params["T"] / self.params["Ndt"]
        eta = self.params["sigma"] * np.sqrt((1 - np.exp(-2 * self.params["kappa"] * dt)) / (2 * self.params["kappa"]))
        self._stock_price = (
            self.params["theta"] + (self._stock_price - self.params["theta"]) * np.exp(-self.params["kappa"] * dt) + eta * np.random.normal()
        )

        self._time_step += 1

        # inventory at next time step - add the trade to current inventory
        self._agent_inventory += action[0]

        # Check if the next state is the last state
        if self._time_step == self.params["Ndt"]:
            # reward - profit with terminal penalty calculated with the new price of the stock and the new inventory
            reward += self._agent_inventory * self._stock_price - self.params["psi"] * np.power(self._agent_inventory, 2)
            terminated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        # Return the expected five values: observation, reward, done, truncated, info
        return observation, reward, terminated, info  # False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # the agent's inventory is initialized to a random value between -max_q and max_q if random_reset is True
        if self.params["random_reset"]:
            self._agent_inventory = np.random.uniform(-self.params["max_q"], self.params["max_q"])
            # the stock price is initialized to a random value
            self._stock_price = np.random.normal(
                self.params["theta"],
                4 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
            )
            self._stock_price = np.min(
                [
                    np.max([self._stock_price, self.observation_space.low[0]]),
                    self.observation_space.high[0],
                ]
            )
        else:
            self._stock_price = self.params["theta"]
            self._agent_inventory = 0

        # the current time step is set to 0
        self._time_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation  # , info

    def _get_obs(self):
        return np.array(
            [self._stock_price, self._agent_inventory, self._time_step],
            dtype=np.float32,
        )

    def _get_info(self):
        return {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def get_normalized_score(env_name, score):
        ref_min_score = -6.17  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 1.72  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


class TradingEnv2(gym.Env):
    def __init__(self, params=None):
        super(TradingEnv2, self).__init__()

        # default parameters for the model
        default_params = {
            "kappa": 2,  # kappa of the OU process
            "sigma": 0.2,  # standard deviation of the OU process
            "theta": 1,  # mean-reversion level of the OU process
            "phi": 0.005,  # transaction costs
            "psi": 0.5,  # terminal penalty on the inventory
            "T": 1,  # trading horizon
            "Ndt": 10,  # number of periods
            "max_q": 5,  # maximum value for the inventory
            "max_u": 2,  # maximum value for the trades
            "random_reset": False,  # reset the inventory to a random value between -max_q and max_q if True, otherwise reset to 0
        }

        self.params = default_params if params is None else {**default_params, **params}

        self.action_space = Box(
            low=np.array([-self.params["max_u"]]),
            high=np.array([self.params["max_u"]]),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space: state representing the stock price, the agent's inventory and the current time step
        self.observation_space = Box(
            low=np.array(
                [
                    self.params["theta"] - 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
                    -self.params["max_q"],
                    0,
                ]
            ),
            high=np.array(
                [
                    self.params["theta"] + 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
                    self.params["max_q"],
                    self.params["Ndt"],
                ]
            ),
            shape=(3,),
            dtype=np.float32,
        )

    def step(self, action):
        # if not self.action_space.contains(action):
        # assert self.action_space.contains(action), "Invalid action"

        # reward is calculated with the current stock price and the current action
        # Only is time = T-1, the reward also includes the terminal penalty on the inventory
        reward = -self._stock_price * action[0] - self.params["phi"] * np.power(action[0], 2)

        # price of the stock at next time step - OU process
        dt = self.params["T"] / self.params["Ndt"]
        eta = self.params["sigma"] * np.sqrt((1 - np.exp(-2 * self.params["kappa"] * dt)) / (2 * self.params["kappa"]))
        self._stock_price = (
            self.params["theta"] + (self._stock_price - self.params["theta"]) * np.exp(-self.params["kappa"] * dt) + eta * np.random.normal()
        )

        self._time_step += 1

        # inventory at next time step - add the trade to current inventory
        self._agent_inventory += action[0]

        # Check if the next state is the last state
        if self._time_step == self.params["Ndt"]:
            # reward - profit with terminal penalty calculated with the new price of the stock and the new inventory
            reward += self._agent_inventory * self._stock_price - self.params["psi"] * np.power(self._agent_inventory, 2)
            terminated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        # Return the expected five values: observation, reward, done, truncated, info
        return observation, reward, terminated, info  # False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # the agent's inventory is initialized to a random value between -max_q and max_q if random_reset is True
        if self.params["random_reset"]:
            self._agent_inventory = np.random.uniform(-self.params["max_q"], self.params["max_q"])
            # the stock price is initialized to a random value
            self._stock_price = np.random.normal(
                self.params["theta"],
                4 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
            )
            self._stock_price = np.min(
                [
                    np.max([self._stock_price, self.observation_space.low[0]]),
                    self.observation_space.high[0],
                ]
            )
        else:
            self._stock_price = self.params["theta"]
            self._agent_inventory = 0

        # the current time step is set to 0
        self._time_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation  # , info

    def _get_obs(self):
        return np.array(
            [self._stock_price, self._agent_inventory, self._time_step],
            dtype=np.float32,
        )

    def _get_info(self):
        return {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def get_normalized_score(env_name, score):
        ref_min_score = -6.73  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 0.4  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


class CurrencyExchange(gym.Env):
    """
    Currency exchange domain.

    There are three state features:
    s[0]: t, the time step in {0, 1, 2, ..., 50}
    s[1]: m, the amount of money remaining to exchange, [0, 100]
    s[2]: p, the exchange rate, [0, 3]

    The action represents
    """

    def __init__(self):
        self.obs_low = np.array(np.zeros(3))
        self.obs_high = np.array([50, 100, 5])
        act_low = np.array(np.ones(1) * -1)
        act_high = np.array(np.ones(1))
        self.observation_space = Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        # self.active_observation_shape = obs_space.shape

        # parameters for price model
        self.price_mu = 1.5
        self.price_sigma = 0.2
        self.price_theta = 0.05

        # initial price
        self.init_price_mu = 1.0
        self.init_price_sigma = 0.05

        self.num_steps = 20
        self.dt = 1
        self.state = self.reset()

    def step(self, a):
        t = self.state[0]
        m = self.state[1]
        p = self.state[2]

        t_next = t + 1
        m_next = m * (1 - np.clip(a, 0, 1))
        reward = ((m - m_next) * p).item()
        p_next = p + self.price_theta * (self.price_mu - p) + self.price_sigma * np.random.normal() * np.sqrt(self.dt)
        p_next = np.clip(p_next, 0, 5)

        if int(np.round(t_next)) == self.num_steps or (m_next < 0.1):
            terminal = True
        else:
            terminal = False

        s_next = np.array([t_next, m_next.item(), p_next.item()])
        self.state = s_next.copy()
        return s_next, reward, terminal, {}

    def reset(self, seed=0):
        np.random.seed(seed)
        t = 0
        m = 100
        p = np.random.normal(loc=self.init_price_mu, scale=self.init_price_sigma)
        s = np.array([t, m, p])
        self.state = s.copy()
        return s

    def get_normalized_score(env_name, score):
        ref_min_score = 0.0  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 1.0  # d4rl.infos.REF_MAX_SCORE[env_name]
        return score  # (score - ref_min_score) / (ref_max_score - ref_min_score)


class HIVTreatment2(gym.Env):
    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """

    def __init__(self):
        super(HIVTreatment2, self).__init__()
        self.obs_low = np.array(np.ones(6) * -5)
        self.obs_high = np.array(np.ones(6) * 8)
        act_low = np.array(np.ones(2) * -1)
        act_high = np.array(np.ones(2))
        self.observation_space = Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        # self.active_observation_shape = obs_space.shape

        self.num_steps = 50
        self.dt = 20  #: measurement every 20 days
        self.logspace = True  #: whether observed states are in log10 space or not

        self.dosage_noise = 0.15
        self.max_noise = 0.3
        self.max_eps1 = 0.7
        self.max_eps2 = 0.3

    def step(self, a):
        self.t += 1
        # if self.logspace:
        #    s = np.power(10, s)

        eps1, eps2 = a[0], a[1]

        # rescale to action space
        eps1 = (eps1 + 1) / 2 * self.max_eps1
        eps2 = (eps2 + 1) / 2 * self.max_eps1

        # scale by noise level
        eps1 = eps1 * (1 + np.random.normal(scale=self.dosage_noise))
        eps2 = eps2 * (1 + np.random.normal(scale=self.dosage_noise))

        # clip
        eps1 = np.clip(eps1, 0.0, (1 + self.max_noise) * self.max_eps1)
        eps2 = np.clip(eps2, 0.0, (1 + self.max_noise) * self.max_eps2)

        # integrate with lsoda method
        ns = lsoda(
            funcptr=rhs.address,
            u0=self.state,
            t_eval=np.linspace(0.0, self.dt, 2001),
            data=np.array([eps1, eps2]),
            rtol=1.49012e-8,
            atol=1.49012e-8,
        )[0][-1, :]
        V, E = ns[4], ns[5]
        # the reward function penalizes treatment because of side-effects
        reward = -0.1 * V - 2e4 * eps1**2 - 2e3 * eps2**2 + 1e3 * E
        reward = reward / 1e6 - 1.0
        self.state = ns.copy()
        if self.logspace:
            ns = np.log10(ns)

        terminal = False
        if self.t == self.num_steps:
            terminal = True

        return ns, reward, terminal, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        # random.seed(seed)
        # np.random.seed(seed)
        self.t = 0
        # non-healthy stable state of the system
        s = np.array([163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0])
        self.state = s.copy()

        if self.logspace:
            return np.log10(s)

        return s

    def get_normalized_score(env_name, score):
        ref_min_score = -48.7  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 5557.9  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


@cfunc(lsoda_sig)
def rhs(t, s, ds, p):
    """
    system derivate per time. The unit of time are days.
    """
    eps1, eps2 = p[0], p[1]
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = 0.34
    k1 = 8e-7
    k2 = 1e-4
    delta = 0.7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.0
    c = 13.0
    rho1 = 1.0
    rho2 = 1.0
    lambdaE = 1
    bE = 0.3
    Kb = 100
    d_E = 0.25
    Kd = 500
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s[0], s[1], s[2], s[3], s[4], s[5]

    # compute derivatives
    tmp1 = (1.0 - eps1) * k1 * V * T1
    tmp2 = (1.0 - f * eps1) * k2 * V * T2
    dT1 = lambda1 - d1 * T1 - tmp1
    dT2 = lambda2 - d2 * T2 - tmp2
    dT1s = tmp1 - delta * T1s - m1 * E * T1s
    dT2s = tmp2 - delta * T2s - m2 * E * T2s
    dV = (1.0 - eps2) * NT * delta * (T1s + T2s) - c * V - ((1.0 - eps1) * rho1 * k1 * T1 + (1.0 - f * eps1) * rho2 * k2 * T2) * V
    dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

    ds[0] = dT1
    ds[1] = dT2
    ds[2] = dT1s
    ds[3] = dT2s
    ds[4] = dV
    ds[5] = dE


class HIVTreatment(gym.Env):
    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """

    def __init__(self):
        super(HIVTreatment, self).__init__()
        self.obs_low = np.array(np.ones(6) * -5)
        self.obs_high = np.array(np.ones(6) * 8)
        act_low = np.array(np.ones(2) * -1)
        act_high = np.array(np.ones(2))
        self.observation_space = Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        # self.active_observation_shape = obs_space.shape

        self.num_steps = 50
        self.dt = 20  #: measurement every 20 days
        self.logspace = True  #: whether observed states are in log10 space or not

        self.dosage_noise = 0.15
        self.max_noise = 0.3
        self.max_eps1 = 0.7
        self.max_eps2 = 0.3

    def step(self, a):
        self.t += 1
        # if self.logspace:
        #    s = np.power(10, s)

        eps1, eps2 = a[0], a[1]

        # rescale to action space
        eps1 = (eps1 + 1) / 2 * self.max_eps1
        eps2 = (eps2 + 1) / 2 * self.max_eps1

        # scale by noise level
        eps1 = eps1 * (1 + np.random.normal(scale=self.dosage_noise))
        eps2 = eps2 * (1 + np.random.normal(scale=self.dosage_noise))

        # clip
        eps1 = np.clip(eps1, 0.0, (1 + self.max_noise) * self.max_eps1)
        eps2 = np.clip(eps2, 0.0, (1 + self.max_noise) * self.max_eps2)

        ns = solve_ivp(
            dsdt,
            [0, self.dt],
            self.state,
            args=(eps1, eps2),
            method="LSODA",
            t_eval=np.linspace(0.0, self.dt, 2001),
        ).y[:, -1]
        T1, T2, T1s, T2s, V, E = ns
        # the reward function penalizes treatment because of side-effects
        reward = -0.1 * V - 2e4 * eps1**2 - 2e3 * eps2**2 + 1e3 * E
        reward = reward / 1e6 - 1.0
        self.state = ns.copy()
        if self.logspace:
            ns = np.log10(ns)

        terminal = False
        if self.t == self.num_steps:
            terminal = True

        return ns, reward, terminal, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        # random.seed(seed)
        # np.random.seed(seed)
        self.t = 0
        # non-healthy stable state of the system
        s = np.array([163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0])
        self.state = s.copy()

        if self.logspace:
            return np.log10(s)

        return s

    def get_normalized_score(env_name, score):
        ref_min_score = -48.7  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 5557.9  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


def dsdt(t, s, eps1, eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = 0.34
    k1 = 8e-7
    k2 = 1e-4
    delta = 0.7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.0
    c = 13.0
    rho1 = 1.0
    rho2 = 1.0
    lambdaE = 1
    bE = 0.3
    Kb = 100
    d_E = 0.25
    Kd = 500
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1.0 - eps1) * k1 * V * T1
    tmp2 = (1.0 - f * eps1) * k2 * V * T2
    dT1 = lambda1 - d1 * T1 - tmp1
    dT2 = lambda2 - d2 * T2 - tmp2
    dT1s = tmp1 - delta * T1s - m1 * E * T1s
    dT2s = tmp2 - delta * T2s - m2 * E * T2s
    dV = (1.0 - eps2) * NT * delta * (T1s + T2s) - c * V - ((1.0 - eps1) * rho1 * k1 * T1 + (1.0 - f * eps1) * rho2 * k2 * T2) * V
    dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

    return np.array([dT1, dT2, dT1s, dT2s, dV, dE])


class NChainEnv(gym.Env):

    def __init__(self, n=9, slip=0.1, left=2, right=1, pit=-10):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.left = left  # payout for state=1
        self.right = right  # payout for state=n-2
        self.pit = pit  # payout for state=0
        self.state = int((self.n - 1) / 2)  # Start at the middle
        self.action_space = Discrete(3)  # 0: left , 1: right, 2: stay
        self.observation_space = flatten_space(Discrete(self.n))

    def step(self, action):
        done = False
        reward = 0
        assert self.action_space.contains(action)
        rand = np.random.rand()
        # print("rand", rand)
        if rand < self.slip:
            if action == 0:
                if rand < self.slip / 2 + 0.01:
                    action = 2
                else:
                    action = 1
            elif action == 1:
                if rand < self.slip / 2 + 0.01:
                    action = 2
                else:
                    action = 0
            else:
                if rand < self.slip / 2:
                    action = 0
                else:
                    action = 1

        if action == 0:
            self.state -= 1
            if self.state == 0:
                reward = self.pit
                done = True
            elif self.state == 1:
                reward = self.left
            elif self.state == self.n - 2:
                reward = self.right
            else:
                reward = 0

        elif action == 1:
            self.state += 1
            if self.state == self.n - 2:
                reward = self.right
            elif self.state == self.n:
                self.state -= 1
            else:
                reward = 0
        else:  # no-op
            reward = 0
        return self.get_obs(), reward, done, {}

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)
        self.state = int((self.n - 1) / 2)
        return self.get_obs()

    def get_obs(self):
        self.observation = flatten(Discrete(self.n), self.state)
        return self.observation

    def get_normalized_score(env_name, score):
        ref_min_score = 0.0  # d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max_score = 1.0  # d4rl.infos.REF_MAX_SCORE[env_name]
        return (score - ref_min_score) / (ref_max_score - ref_min_score)


class SCAwareObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, gamma: float = 1.0, normalizer: float = 1.0):
        """Initialize :class:`SCAwareObservation` that requires an environment with a flat :class:`Box` observation space.

        Args:
            env: The environment to apply the wrapper
            gamma: The discount factor
        """
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        # assert env.observation_space.dtype == np.float32
        self.env = env
        self.gamma = gamma
        self.normalizer = normalizer
        low = np.append(self.observation_space.low, [-np.inf, 0.0])
        high = np.append(self.observation_space.high, [np.inf, 1])
        self.observation_space = Box(low, high, dtype=np.float32)
        if hasattr(env, "_max_episode_steps"):
            self._max_episode_steps = env._max_episode_steps

    def observation(self, observation):
        """Adds to the observation with the current s and c values.

        Args:
            observation: The observation to add the s and c values to

        Returns:
            The observation with the s and c values appended
        """
        return np.append(observation, [self.s, self.c])

    def reset(self, **kwargs):
        """Reset the environment setting the s to zero and c to 1.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        obs = super().reset(**kwargs)
        self.s = 0
        self.c = 1

        return self.observation(obs)

    def step(self, action):
        """Steps through the environment, incrementing the s and c values.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        obs, reward, done, info = self.env.step(action)
        self.s += self.c * reward / self.normalizer
        self.c *= self.gamma
        return self.observation(obs), reward, done, info

    def __getattr__(self, attr):
        # Delegate attribute lookup to the inner environment.
        return getattr(self.env, attr)


class BAwareObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, gamma: float = 1.0, b_0=None, normalizer: float = 1.0):
        """Initialize :class:`BAwareObservation` that requires an environment with a flat :class:`Box` observation space.

        Args:
            env: The environment to apply the wrapper
            gamma: The discount factor
        """
        gym.Wrapper.__init__(self, env)
        assert isinstance(env.observation_space, Box)
        # assert env.observation_space.dtype == np.float32
        assert b_0 is not None
        self.env = env
        self.gamma = gamma
        self.normalizer = normalizer
        self.b_0 = b_0
        low = np.append(self.observation_space.low, [-np.inf])
        high = np.append(self.observation_space.high, [np.inf])
        self.observation_space = Box(low, high, dtype=np.float32)
        if hasattr(env, "_max_episode_steps"):
            self._max_episode_steps = env._max_episode_steps

    def observation(self, observation):
        """Adds to the observation with the current b value.

        Args:
            observation: The observation to add the b value to

        Returns:
            The observation with the b value appended
        """
        return np.append(observation, [self.b])

    def reset(self, **kwargs):
        """Reset the environment setting the b to kwargs['b']

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        obs = super().reset(**kwargs)
        self.b = self.b_0

        return self.observation(obs)

    def step(self, action):
        """Steps through the environment, incrementing the b value.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        obs, reward, done, info = self.env.step(action)
        self.b = (self.b - (reward / self.normalizer)) / self.gamma
        return self.observation(obs), reward, done, info


# register the environments
register(id="Trading-v0", entry_point="custom_envs_gym:TradingEnv", max_episode_steps=10)
register(id="Trading-v1", entry_point="custom_envs_gym:TradingEnv2", max_episode_steps=100)
register(
    id="CurrencyExchange-v0",
    entry_point="custom_envs_gym:CurrencyExchange",
    max_episode_steps=20,
)
# register(id="HIVTreatment-v0", entry_point="custom_envs_gym:HIVTreatment", max_episode_steps=50)
register(
    id="HIVTreatment-v1",
    entry_point="custom_envs_gym:HIVTreatment2",
    max_episode_steps=50,
)


register(
    id="PortfolioAllocationEnv-ETF-Train-v2",
    entry_point="custom_envs_gym:PortfolioAllocationEnv",
    max_episode_steps=21 * 3,
    kwargs={
        "transaction_cost_pct": 0.0025,
        "window_size": 1,
        "episode_length": 21 * 3,
        "data_start_date": "2005-01-01",
        "data_end_date": "2024-12-31",
        "asset_symbols": ["SPY", "GLD", "CASH"],
        "is_training": True,
    },
)
register(
    id="PortfolioAllocationEnv-ETF-Test-v2",
    entry_point="custom_envs_gym:PortfolioAllocationEnv",
    max_episode_steps=21 * 3,
    kwargs={
        "transaction_cost_pct": 0.0025,
        "window_size": 1,
        "episode_length": 21 * 3,
        "data_start_date": "2005-01-01",
        "data_end_date": "2024-12-31",
        "asset_symbols": ["SPY", "GLD", "CASH"],
        "is_training": False,
    },
)


register(
    id="PortfolioAllocationEnv-ETF-Train-v3",
    entry_point="custom_envs_gym:PortfolioAllocationEnv",
    max_episode_steps=21 * 3,
    kwargs={
        "transaction_cost_pct": 0.0025,
        "window_size": 5,
        "episode_length": 21 * 3,
        "data_start_date": "2005-01-01",
        "data_end_date": "2024-12-31",
        "asset_symbols": ["SPY", "GLD", "CASH"],
        "is_training": True,
    },
)
register(
    id="PortfolioAllocationEnv-ETF-Test-v3",
    entry_point="custom_envs_gym:PortfolioAllocationEnv",
    max_episode_steps=21 * 3,
    kwargs={
        "transaction_cost_pct": 0.0025,
        "window_size": 5,
        "episode_length": 21 * 3,
        "data_start_date": "2005-01-01",
        "data_end_date": "2024-12-31",
        "asset_symbols": ["SPY", "GLD", "CASH"],
        "is_training": False,
    },
)


class NormalizeLogReturns(gym.ObservationWrapper):
    """
    Normalizes only the log return history part of the observation space
    using provided pre-calculated means and standard deviations.
    Portfolio weights remain unnormalized.

    Args:
        env: The SyntheticPortfolioEnv instance to wrap.
        log_return_means: Numpy array of mean single-step log returns for each non-cash asset.
        log_return_stds: Numpy array of std dev of single-step log returns for each non-cash asset.
        epsilon: Small value added to std dev to prevent division by zero.
    """

    def __init__(
        self,
        env: SyntheticPortfolioEnv,
        log_return_means: np.ndarray,
        log_return_stds: np.ndarray,
        epsilon: float = 1e-8,
    ):
        super().__init__(env)

        # --- Basic Type/Shape Checks ---
        if not isinstance(env.unwrapped, SyntheticPortfolioEnv):
            warnings.warn(
                "NormalizeLogReturns wrapper is designed for SyntheticPortfolioEnv.",
                UserWarning,
            )

        if not isinstance(log_return_means, np.ndarray) or not isinstance(log_return_stds, np.ndarray):
            raise TypeError("log_return_means and log_return_stds must be numpy arrays.")

        self.num_non_cash_assets = env.unwrapped.num_non_cash_assets
        expected_shape = (self.num_non_cash_assets,)
        if log_return_means.shape != expected_shape or log_return_stds.shape != expected_shape:
            raise ValueError(
                f"log_return_means and log_return_stds must have shape {expected_shape}, " f"got {log_return_means.shape} and {log_return_stds.shape}"
            )
        # --- Store normalization parameters ---
        self.log_return_means = log_return_means.astype(np.float32)
        # Ensure std deviations are non-zero before storing
        self.log_return_stds = np.maximum(log_return_stds.astype(np.float32), epsilon)
        self.epsilon = epsilon  # Though applied above, store if needed elsewhere

        # --- Store env dimensions for convenience ---
        self.window_size = env.unwrapped.window_size
        self.num_all_assets = env.unwrapped.num_all_assets
        self.log_returns_flat_size = self.num_non_cash_assets * self.window_size

        # --- Define the new observation space ---
        # Normalized returns can theoretically go beyond typical bounds, but practically are often clipped.
        # Use wider bounds for the normalized part. Weights remain [0, 1].
        # We can use -inf/inf if the agent/framework handles it, or wide finite bounds.
        # Let's use reasonably wide finite bounds for broader compatibility.
        norm_low = -20.0
        norm_high = 20.0

        low_bounds = np.concatenate(
            [
                np.full(self.log_returns_flat_size, norm_low, dtype=np.float32),
                np.zeros(self.num_all_assets, dtype=np.float32),
            ]  # Weights lower bound
        )
        high_bounds = np.concatenate(
            [
                np.full(self.log_returns_flat_size, norm_high, dtype=np.float32),
                np.ones(self.num_all_assets, dtype=np.float32),
            ]  # Weights upper bound
        )

        original_shape = env.observation_space.shape
        self.observation_space = Box(low=low_bounds, high=high_bounds, shape=original_shape, dtype=np.float32)
        print("[NormalizeLogReturns] Wrapper initialized.")
        print(f"[NormalizeLogReturns] New observation space: {self.observation_space}")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies normalization to the log returns part of the observation."""
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)  # Ensure numpy array

        # Split observation into log returns and weights
        log_returns_flat = obs[: self.log_returns_flat_size]
        weights = obs[self.log_returns_flat_size :]

        # Reshape log returns for easier broadcasting
        # Shape becomes (window_size, num_non_cash_assets)
        log_returns_history = log_returns_flat.reshape(self.window_size, self.num_non_cash_assets)

        # Normalize using stored means/stds (broadcasting applies them per asset across the window)
        normalized_returns_history = (log_returns_history - self.log_return_means) / self.log_return_stds

        # Flatten normalized returns back
        normalized_returns_flat = normalized_returns_history.flatten()

        # Combine normalized returns with ORIGINAL weights
        new_obs = np.concatenate((normalized_returns_flat, weights)).astype(np.float32)

        # Clip observation to the defined space bounds (optional but recommended)
        new_obs = np.clip(new_obs, self.observation_space.low, self.observation_space.high)

        return new_obs

    # Optional: Add reset method if the wrapper needs internal state reset
    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs)
    #     return self.observation(obs), info

    # Optional: Add step method if needed (usually not for ObservationWrapper)
    # def step(self, action):
    #     observation, reward, terminated, truncated, info = self.env.step(action)
    #     return self.observation(observation), reward, terminated, truncated, info


def create_normalized_portfolio_env(**kwargs):
    """
    Factory function to create SyntheticPortfolioEnv, calculate step log return
    stats from its calibration data, and wrap with NormalizeLogReturns.
    """
    print(f"[Factory Specific] Creating base SyntheticPortfolioEnv with kwargs: {kwargs}")
    # Create the base environment first to access its calibration data/methods
    base_env = SyntheticPortfolioEnv(**kwargs)

    # --- Calculate step log return stats from the *same* calibration data ---
    # Need to access the data used internally or recalculate it here.
    # Easiest if we modify _calibrate_gbm_params to store log_returns_df or
    # recalculate here based on config. Let's recalculate for clarity:
    print("[Factory Specific] Recalculating log returns for normalization stats...")
    df_prices = base_env._load_historical_data()  # Use internal method
    log_returns_df = np.log(df_prices / df_prices.shift(1)).dropna()

    if log_returns_df.empty:
        raise ValueError("[Factory Specific] Cannot calculate normalization stats: No log returns.")

    log_return_means_step = log_returns_df.mean().to_numpy(dtype=np.float32)
    log_return_stds_step = log_returns_df.std().to_numpy(dtype=np.float32)
    print(f"[Factory Specific] Step Means: {log_return_means_step}")
    print(f"[Factory Specific] Step Stds: {log_return_stds_step}")

    # --- Apply the specific wrapper ---
    print("[Factory Specific] Applying NormalizeLogReturns wrapper.")
    wrapped_env = NormalizeLogReturns(base_env, log_return_means_step, log_return_stds_step)

    print(f"[Factory Specific] Returning wrapped environment: {wrapped_env}")
    return wrapped_env


gym.register(
    id="NormalizedSyntheticPortfolio-v1",
    entry_point="custom_envs_gym:create_normalized_portfolio_env",
    max_episode_steps=21 * 3,
    kwargs={
        # "override_mu": 2 * np.array([0.09289327, 0.10917936, 0.06584878, 0.08015655]),
        "window_size": 1,
        "num_steps": 21 * 3,
        "data_start_date": "2000-01-01",
        "data_end_date": "2024-12-31",
        "asset_symbols": ["SPY", "EWA", "EWG", "GLD", "CASH"],  #
    },
)


for agent in ["hopper", "walker2d", "halfcheetah"]:
    for dataset in ["medium", "expert"]:
        for version in ["v0-s"]:
            env_name = "%s-%s-%s" % (agent, dataset, version)
            register(
                id=env_name,
                entry_point="get_env:get_env",
                max_episode_steps=200 if agent == "halfcheetah" else 500,
                kwargs={
                    "env_name": env_name,
                    # "ref_min_score": REF_MIN_SCORE[env_name],
                    # "ref_max_score": REF_MAX_SCORE[env_name],
                },
            )
            d4rl.infos.DATASET_URLS[env_name] = ""
