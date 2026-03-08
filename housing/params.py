from dataclasses import dataclass
import numpy as np


@dataclass
class MarginalDist:
    mean: float
    std: float


@dataclass
class StochasticParams:
    appreciation: MarginalDist
    investment_return: MarginalDist
    rent_growth: MarginalDist
    correlation: np.ndarray | None = None

    def means(self) -> np.ndarray:
        return np.array([self.appreciation.mean, self.investment_return.mean, self.rent_growth.mean])

    def covariance_matrix(self) -> np.ndarray:
        stds = np.array([self.appreciation.std, self.investment_return.std, self.rent_growth.std])
        if self.correlation is not None:
            return np.outer(stds, stds) * self.correlation
        return np.diag(stds ** 2)


@dataclass
class DeterministicParams:
    home_price: float
    down_payment: float
    interest_rate: float
    loan_term_years: int
    property_tax_rate: float
    hoa_monthly: float
    insurance_monthly: float
    maintenance_rate: float
    buy_closing_cost_rate: float
    sell_transaction_cost_rate: float
    rent_monthly_initial: float
    initial_savings: float
    monthly_income: float

    # Tax
    ltcg_rate: float
    stcg_rate: float
    home_sale_exclusion: float
    marginal_tax_rate: float = 0.24
    standard_deduction: float = 14_600.0
    salt_deduction: float = 10_000.0


@dataclass
class AnnualRates:
    appreciation: float
    investment_return: float
    rent_growth: float


@dataclass
class RatePath:
    years: list[AnnualRates]

    def __len__(self) -> int:
        return len(self.years)

    def __getitem__(self, year_index: int) -> AnnualRates:
        return self.years[year_index]


def sample_rate_paths(
    stochastic: StochasticParams,
    n_years: int,
    n_trials: int,
    rng: np.random.Generator,
) -> list[RatePath]:
    mean = stochastic.means()
    cov = stochastic.covariance_matrix()
    draws = rng.multivariate_normal(mean, cov, size=(n_trials, n_years))

    return [
        RatePath(years=[
            AnnualRates(appreciation=draws[i, y, 0], investment_return=draws[i, y, 1], rent_growth=draws[i, y, 2])
            for y in range(n_years)
        ])
        for i in range(n_trials)
    ]


def annual_to_monthly(rate_annual: float) -> float:
    return (1.0 + rate_annual) ** (1.0 / 12.0) - 1.0


def investment_tax(balance: float, cost_basis: float, det: DeterministicParams, months_held: int) -> float:
    gain = max(0.0, balance - cost_basis)
    rate = det.ltcg_rate if months_held > 12 else det.stcg_rate
    return gain * rate


DEFAULT_CORRELATION = np.array([
    # appreciation  investment  rent_growth
    [1.0,           0.3,        0.5],
    [0.3,           1.0,        0.2],
    [0.5,           0.2,        1.0],
])
