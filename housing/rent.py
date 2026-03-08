from dataclasses import dataclass

from .params import AnnualRates, DeterministicParams, annual_to_monthly, investment_tax


@dataclass
class RentState:
    rent: float
    investment_balance: float
    investment_cost_basis: float


def initial_rent_state(det: DeterministicParams) -> RentState:
    return RentState(
        rent=det.rent_monthly_initial,
        investment_balance=det.initial_savings,
        investment_cost_basis=det.initial_savings,
    )


def renter_monthly_cost(state: RentState) -> float:
    return state.rent


def net_worth_rent(state: RentState, det: DeterministicParams, months_held: int) -> float:
    return state.investment_balance - investment_tax(state.investment_balance, state.investment_cost_basis, det, months_held)


def step_rent(state: RentState, det: DeterministicParams, rates: AnnualRates, is_year_boundary: bool) -> RentState:
    monthly_inv_return = annual_to_monthly(rates.investment_return)

    # Rent increases annually at year boundaries
    new_rent = state.rent * (1 + rates.rent_growth) if is_year_boundary else state.rent

    # Investments grow, add income, subtract rent
    net_contribution = det.monthly_income - new_rent
    new_investments = state.investment_balance * (1 + monthly_inv_return) + net_contribution

    return RentState(
        rent=new_rent,
        investment_balance=new_investments,
        investment_cost_basis=state.investment_cost_basis + net_contribution,
    )


def rent_snapshot(state: RentState, det: DeterministicParams, t: int) -> dict:
    return {
        "rent": state.rent,
        "renter_monthly_cost": renter_monthly_cost(state),
        "investment_balance_rent": state.investment_balance,
        "investment_tax_rent": investment_tax(state.investment_balance, state.investment_cost_basis, det, t),
        "net_worth_rent": net_worth_rent(state, det, t),
    }
