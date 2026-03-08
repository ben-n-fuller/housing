import numpy as np
import pandas as pd

from .params import DeterministicParams, StochasticParams, RatePath, sample_rate_paths
from .buy import BuyState, initial_buy_state, step_buy, buy_snapshot, net_worth_buy
from .rent import RentState, initial_rent_state, step_rent, rent_snapshot, net_worth_rent


DEFAULT_HORIZONS = [1, 2, 3, 5, 7]


def delta(buy_state: BuyState, rent_state: RentState, det: DeterministicParams, months_held: int) -> float:
    return net_worth_buy(buy_state, det, months_held) - net_worth_rent(rent_state, det, months_held)


def simulate_trial(det: DeterministicParams, path: RatePath, horizons: list[int] = DEFAULT_HORIZONS) -> list[dict]:
    n_years = len(path)
    n_months = n_years * 12

    buy = initial_buy_state(det)
    rent = initial_rent_state(det)
    horizon_months = {h * 12 for h in horizons if h <= n_years}

    snapshots = []
    for month in range(1, n_months + 1):
        rates = path[(month - 1) // 12]
        buy = step_buy(buy, det, rates, is_year_end=(month % 12 == 0))
        rent = step_rent(rent, det, rates, is_year_boundary=(month % 12 == 1 and month > 1))

        if month in horizon_months:
            snapshots.append({
                "horizon_years": month // 12,
                **buy_snapshot(buy, det, month),
                **rent_snapshot(rent, det, month),
                "delta": delta(buy, rent, det, month),
            })

    return snapshots


def run_monte_carlo(
    det: DeterministicParams,
    stochastic: StochasticParams,
    horizons: list[int] = DEFAULT_HORIZONS,
    n_trials: int = 10_000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    n_years = max(horizons)
    paths = sample_rate_paths(stochastic, n_years, n_trials, rng)

    rows = []
    for trial, path in enumerate(paths):
        for snapshot in simulate_trial(det, path, horizons):
            snapshot["trial"] = trial
            rows.append(snapshot)

    return pd.DataFrame(rows)
