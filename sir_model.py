"""
SIR Stochastic Model using the Gillespie Algorithm.

This module provides the SIRStochasticModel class which simulates epidemic
dynamics using exact stochastic simulation (Gillespie algorithm) and supports
saving/loading trained model parameters.
"""

import json
import pickle
from pathlib import Path

import numpy as np


class SIRStochasticModel:
    """
    SIR epidemic model using the Gillespie (exact stochastic) algorithm.

    Parameters
    ----------
    N : int
        Total population size.
    beta : float
        Transmission rate (infections per unit time per infected individual).
    gamma : float
        Recovery rate (recoveries per unit time per infected individual).
    """

    def __init__(self, N: int = 1000, beta: float = 0.3, gamma: float = 0.1):
        self.N = N
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        S0: int,
        I0: int,
        R0: int,
        T_max: float = 160.0,
        seed: int | None = None,
    ) -> dict:
        """
        Run a single Gillespie simulation.

        Parameters
        ----------
        S0, I0, R0 : int
            Initial counts for Susceptible, Infected, Recovered.
        T_max : float
            Maximum simulation time.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        dict with keys 'time', 'S', 'I', 'R' – each a list of values.
        """
        rng = np.random.default_rng(seed)

        S, I, R = int(S0), int(I0), int(R0)
        t = 0.0

        time_arr = [t]
        S_arr = [S]
        I_arr = [I]
        R_arr = [R]

        while t < T_max and I > 0:
            rate_infection = self.beta * S * I / self.N
            rate_recovery = self.gamma * I
            rate_total = rate_infection + rate_recovery

            if rate_total == 0:
                break

            # Time to next event (exponential)
            tau = -np.log(rng.random()) / rate_total
            t += tau

            # Which event occurs?
            if rng.random() < rate_infection / rate_total:
                S -= 1
                I += 1
            else:
                I -= 1
                R += 1

            time_arr.append(t)
            S_arr.append(S)
            I_arr.append(I)
            R_arr.append(R)

        return {"time": time_arr, "S": S_arr, "I": I_arr, "R": R_arr}

    def simulate_mean(
        self,
        S0: int,
        I0: int,
        R0: int,
        T_max: float = 160.0,
        n_runs: int = 50,
        n_timepoints: int = 200,
        seed: int | None = None,
    ) -> dict:
        """
        Run *n_runs* simulations and return the mean trajectory on a regular
        time grid (useful for training / evaluation).

        Returns
        -------
        dict with keys 'time', 'S', 'I', 'R' – each a 1-D numpy array of
        length *n_timepoints*.
        """
        rng = np.random.default_rng(seed)
        t_grid = np.linspace(0, T_max, n_timepoints)

        S_all = np.zeros((n_runs, n_timepoints))
        I_all = np.zeros((n_runs, n_timepoints))
        R_all = np.zeros((n_runs, n_timepoints))

        for k in range(n_runs):
            run_seed = int(rng.integers(0, 2**31))
            result = self.simulate(S0, I0, R0, T_max=T_max, seed=run_seed)

            t = np.array(result["time"])
            S_all[k] = np.interp(t_grid, t, result["S"])
            I_all[k] = np.interp(t_grid, t, result["I"])
            R_all[k] = np.interp(t_grid, t, result["R"])

        return {
            "time": t_grid,
            "S": S_all.mean(axis=0),
            "I": I_all.mean(axis=0),
            "R": R_all.mean(axis=0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "sir_model.pkl") -> None:
        """Persist model parameters to *path* (pickle)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._params(), fh)
        print(f"Model saved to {path}")

    def save_json(self, path: str = "sir_model.json") -> None:
        """Persist model parameters to *path* (human-readable JSON)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self._params(), fh, indent=2)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = "sir_model.pkl") -> "SIRStochasticModel":
        """Load a previously saved model from *path*."""
        with open(path, "rb") as fh:
            params = pickle.load(fh)
        model = cls(**params)
        print(f"Model loaded from {path}  (N={model.N}, β={model.beta}, γ={model.gamma})")
        return model

    @classmethod
    def load_json(cls, path: str = "sir_model.json") -> "SIRStochasticModel":
        """Load a previously saved model from a JSON *path*."""
        with open(path) as fh:
            params = json.load(fh)
        model = cls(**params)
        print(f"Model loaded from {path}  (N={model.N}, β={model.beta}, γ={model.gamma})")
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _params(self) -> dict:
        return {"N": self.N, "beta": self.beta, "gamma": self.gamma}

    def __repr__(self) -> str:
        return (
            f"SIRStochasticModel(N={self.N}, beta={self.beta:.4f}, gamma={self.gamma:.4f})"
        )
