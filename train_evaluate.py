"""
Training and Evaluation for the SIR Stochastic Model.

Workflow
--------
1. Generate synthetic "observed" data using a reference SIRStochasticModel.
2. Train  – fit beta and gamma parameters to the observed data with
            scipy.optimize.minimize (Nelder-Mead).
3. Evaluate – compute MSE / RMSE / MAE for each compartment and plot
              the fitted model against the observations.
4. Save the trained model to disk.

Run
---
    python train_evaluate.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sir_model import SIRStochasticModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 1000
S0, I0, R0 = 999, 1, 0
T_MAX = 160.0
N_TIMEPOINTS = 200
N_RUNS_OBS = 100   # runs used to build the "observed" dataset
N_RUNS_FIT = 50    # runs used inside the optimisation (faster)
SEED_OBS = 42
MODEL_SAVE_PATH = "sir_model.pkl"
MODEL_SAVE_JSON = "sir_model.json"

# True (reference) parameters used to generate synthetic observations
BETA_TRUE = 0.3
GAMMA_TRUE = 0.1


# ---------------------------------------------------------------------------
# Helper – interpolate a stochastic trajectory onto a regular grid
# ---------------------------------------------------------------------------

def _interp_mean(model: SIRStochasticModel, n_runs: int, seed: int) -> dict:
    """Return mean trajectory on a regular time grid."""
    return model.simulate_mean(
        S0, I0, R0,
        T_max=T_MAX,
        n_runs=n_runs,
        n_timepoints=N_TIMEPOINTS,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Step 1 – Generate observed data
# ---------------------------------------------------------------------------

def generate_observed_data() -> dict:
    """Simulate the 'ground truth' epidemic trajectory."""
    print("=" * 60)
    print("Step 1 – Generating observed data")
    print(f"  β_true={BETA_TRUE}, γ_true={GAMMA_TRUE}, N={N}")
    true_model = SIRStochasticModel(N=N, beta=BETA_TRUE, gamma=GAMMA_TRUE)
    obs = _interp_mean(true_model, n_runs=N_RUNS_OBS, seed=SEED_OBS)
    print(f"  Generated {N_RUNS_OBS} trajectories – peak I ≈ {obs['I'].max():.1f}")
    return obs


# ---------------------------------------------------------------------------
# Step 2 – Training (parameter fitting)
# ---------------------------------------------------------------------------

def _objective(params: np.ndarray, obs: dict, fit_seed: int) -> float:
    """Mean-squared error between model prediction and observations."""
    beta, gamma = params
    if beta <= 0 or gamma <= 0 or beta > 2 or gamma > 2:
        return 1e10  # penalty for invalid parameters

    model = SIRStochasticModel(N=N, beta=float(beta), gamma=float(gamma))
    pred = _interp_mean(model, n_runs=N_RUNS_FIT, seed=fit_seed)

    mse = (
        np.mean((pred["S"] - obs["S"]) ** 2)
        + np.mean((pred["I"] - obs["I"]) ** 2)
        + np.mean((pred["R"] - obs["R"]) ** 2)
    ) / (3 * N**2)  # normalised
    return float(mse)


def train(obs: dict) -> SIRStochasticModel:
    """
    Fit beta and gamma to observed data using Nelder-Mead optimisation.

    Returns a fitted SIRStochasticModel.
    """
    print("\n" + "=" * 60)
    print("Step 2 – Training (parameter fitting via Nelder-Mead)")

    # Initial guess (deliberately perturbed from truth)
    beta0, gamma0 = 0.4, 0.15
    print(f"  Initial guess: β={beta0}, γ={gamma0}")

    result = minimize(
        _objective,
        x0=[beta0, gamma0],
        args=(obs, SEED_OBS + 1),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 200, "disp": True},
    )

    beta_fit, gamma_fit = result.x
    r0_fit = beta_fit / gamma_fit
    r0_true = BETA_TRUE / GAMMA_TRUE
    print(f"\n  Optimisation {'converged' if result.success else 'stopped'}:")
    print(f"  β_fit={beta_fit:.4f}  (true {BETA_TRUE})")
    print(f"  γ_fit={gamma_fit:.4f}  (true {GAMMA_TRUE})")
    print(f"  R0_fit={r0_fit:.3f}  (true {r0_true:.1f})")

    return SIRStochasticModel(N=N, beta=float(beta_fit), gamma=float(gamma_fit))


# ---------------------------------------------------------------------------
# Step 3 – Evaluation
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print(f"  {label:12s}  MSE={mse:10.2f}  RMSE={rmse:8.2f}  MAE={mae:8.2f}")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def evaluate(
    fitted_model: SIRStochasticModel,
    obs: dict,
    plot: bool = True,
) -> dict:
    """
    Evaluate the fitted model against observed data and (optionally) plot.

    Returns a dict with per-compartment metrics.
    """
    print("\n" + "=" * 60)
    print("Step 3 – Evaluation")

    pred = _interp_mean(fitted_model, n_runs=N_RUNS_OBS, seed=SEED_OBS + 2)
    t_grid = obs["time"]

    metrics = {}
    print(f"  {'Compartment':12s}  {'MSE':>14}  {'RMSE':>10}  {'MAE':>10}")
    print("  " + "-" * 52)
    for comp in ("S", "I", "R"):
        metrics[comp] = _compute_metrics(obs[comp], pred[comp], comp)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, comp, color in zip(axes, ("S", "I", "R"), ("steelblue", "crimson", "forestgreen")):
            ax.plot(t_grid, obs[comp], label="Observed", linestyle="--", color="gray")
            ax.plot(t_grid, pred[comp], label="Fitted model", color=color)
            ax.set_title(f"Compartment: {comp}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            ax.legend()
        fig.suptitle(
            f"SIR Stochastic Model – Evaluation\n"
            f"β_fit={fitted_model.beta:.4f}, γ_fit={fitted_model.gamma:.4f}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig("evaluation_plot.png", dpi=120, bbox_inches="tight")
        print("\n  Evaluation plot saved to evaluation_plot.png")
        plt.close()

    return metrics


# ---------------------------------------------------------------------------
# Step 4 – Save trained model
# ---------------------------------------------------------------------------

def save_model(model: SIRStochasticModel) -> None:
    print("\n" + "=" * 60)
    print("Step 4 – Saving trained model")
    model.save(MODEL_SAVE_PATH)
    model.save_json(MODEL_SAVE_JSON)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore")

    obs = generate_observed_data()
    fitted_model = train(obs)
    metrics = evaluate(fitted_model, obs)
    save_model(fitted_model)

    # Persist metrics for reference
    metrics_path = "evaluation_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Trained model: {fitted_model}")


if __name__ == "__main__":
    main()
