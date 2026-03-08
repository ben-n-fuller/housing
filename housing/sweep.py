from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .params import (
    DeterministicParams, StochasticParams, MarginalDist, DEFAULT_CORRELATION,
)
from .simulation import run_monte_carlo
from .plotting import (
    delta_summary, plot_delta_violin, plot_prob_buy_wins,
    plot_delta_fan, plot_net_worth_comparison,
)


BASELINE_PRICE = 700_000
DOWN_RATIO = 100_000 / BASELINE_PRICE
RENT_RATIO = 2_500 / BASELINE_PRICE


def build_sweep_grid(
    base_det: DeterministicParams,
    base_stochastic: StochasticParams,
    appreciation_means: list[float],
    interest_rates: list[float],
    sell_cost_rates: list[float],
    home_prices: list[float],
) -> list[tuple[DeterministicParams, StochasticParams, dict]]:
    """Build the Cartesian product of parameter combinations.

    Down payment and rent scale proportionally with home price.
    Initial savings is held fixed at the baseline value.
    """
    grid = []
    for app_mean, rate, sell_cost, price in product(
        appreciation_means, interest_rates, sell_cost_rates, home_prices,
    ):
        sp = replace(
            base_stochastic,
            appreciation=MarginalDist(mean=app_mean, std=base_stochastic.appreciation.std),
        )
        dp = replace(
            base_det,
            home_price=price,
            down_payment=round(price * DOWN_RATIO),
            interest_rate=rate,
            sell_transaction_cost_rate=sell_cost,
            rent_monthly_initial=round(price * RENT_RATIO),
        )
        param_dict = {
            "appreciation_mean": app_mean,
            "interest_rate": rate,
            "sell_cost_rate": sell_cost,
            "home_price": price,
        }
        grid.append((dp, sp, param_dict))
    return grid


def _combo_dirname(params: dict) -> str:
    app = f"app{params['appreciation_mean']:.0%}".replace("%", "")
    rate = f"rate{params['interest_rate']:.1%}".replace(".", "").replace("%", "")
    sell = f"sell{params['sell_cost_rate']:.0%}".replace("%", "")
    price = f"price{params['home_price']/1e3:.0f}k"
    return f"{app}_{rate}_{sell}_{price}"


def save_combo_artifacts(
    results: pd.DataFrame,
    det: DeterministicParams,
    params: dict,
    output_dir: Path,
) -> None:
    """Save per-combo plots and summary to *output_dir*."""
    combo_dir = output_dir / _combo_dirname(params)
    combo_dir.mkdir(parents=True, exist_ok=True)

    # Summary text
    summary_styler = delta_summary(results)
    lines = [
        "Parameter Combination",
        "=" * 40,
        f"  Appreciation mean : {params['appreciation_mean']:.1%}",
        f"  Mortgage rate     : {params['interest_rate']:.1%}",
        f"  Sell cost rate    : {params['sell_cost_rate']:.0%}",
        f"  Home price        : ${params['home_price']:,.0f}",
        f"  Down payment      : ${det.down_payment:,.0f}",
        f"  Initial rent      : ${det.rent_monthly_initial:,.0f}/mo",
        f"  Initial savings   : ${det.initial_savings:,.0f}",
        "",
        "Delta Summary (Buy - Rent)",
        "=" * 40,
        summary_styler.data.to_string(),
    ]
    (combo_dir / "summary.txt").write_text("\n".join(lines))

    # Plots
    for name, plot_fn in [
        ("delta_violin", plot_delta_violin),
        ("prob_buy_wins", plot_prob_buy_wins),
        ("delta_fan", plot_delta_fan),
        ("net_worth_comparison", plot_net_worth_comparison),
    ]:
        fig = plot_fn(results)
        fig.savefig(combo_dir / f"{name}.png", dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)


def run_sweep(
    grid: list[tuple[DeterministicParams, StochasticParams, dict]],
    horizons: list[int],
    n_trials: int = 2_000,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Run Monte Carlo for every combo in *grid* and return aggregated stats.

    If *output_dir* is provided, saves per-combo plots and summaries there.
    """
    if rng is None:
        rng = np.random.default_rng()
    if output_dir is not None:
        output_dir = Path(output_dir)

    records: list[dict] = []
    n_combos = len(grid)

    for i, (det, stochastic, params) in enumerate(grid):
        if i % 24 == 0:
            print(f"  Combo {i + 1}/{n_combos} ...")

        res = run_monte_carlo(
            det, stochastic, horizons=horizons, n_trials=n_trials, rng=rng,
        )

        if output_dir is not None:
            save_combo_artifacts(res, det, params, output_dir)

        for h in horizons:
            deltas = res.loc[res["horizon_years"] == h, "delta"]
            records.append({
                **params,
                "horizon": h,
                "mean_delta": deltas.mean(),
                "median_delta": deltas.median(),
                "std_delta": deltas.std(),
                "p10": deltas.quantile(0.10),
                "p90": deltas.quantile(0.90),
                "prob_buy_wins": (deltas > 0).mean(),
            })

    print(f"  Done: {n_combos} combos × {len(horizons)} horizons = {len(records)} rows")
    return pd.DataFrame(records)
