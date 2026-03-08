from dataclasses import dataclass
from .params import AnnualRates, DeterministicParams, annual_to_monthly, investment_tax


@dataclass
class BuyState:
    home_value: float
    principal_balance: float
    investment_balance: float
    investment_cost_basis: float
    interest_paid_ytd: float = 0.0


def initial_buy_state(det: DeterministicParams) -> BuyState:
    buy_closing_cost = det.home_price * det.buy_closing_cost_rate
    initial_investments = det.initial_savings - det.down_payment - buy_closing_cost

    return BuyState(
        home_value=det.home_price,
        principal_balance=det.home_price - det.down_payment,
        investment_balance=initial_investments,
        investment_cost_basis=initial_investments,
    )


def mortgage_payment(det: DeterministicParams) -> float:
    principal = det.home_price - det.down_payment
    monthly_rate = det.interest_rate / 12
    n_payments = det.loan_term_years * 12

    if monthly_rate == 0:
        return principal / n_payments

    return principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)


def interest_payment(state: BuyState, det: DeterministicParams) -> float:
    monthly_rate = det.interest_rate / 12
    return state.principal_balance * monthly_rate


def principal_payment(state: BuyState, det: DeterministicParams) -> float:
    return mortgage_payment(det) - interest_payment(state, det)


def property_tax(state: BuyState, det: DeterministicParams) -> float:
    return state.home_value * det.property_tax_rate / 12


def maintenance_cost(state: BuyState, det: DeterministicParams) -> float:
    return state.home_value * det.maintenance_rate / 12


def equity(state: BuyState) -> float:
    return state.home_value - state.principal_balance


def sell_transaction_cost(state: BuyState, det: DeterministicParams) -> float:
    return state.home_value * det.sell_transaction_cost_rate


def net_sale_proceeds(state: BuyState, det: DeterministicParams) -> float:
    return state.home_value - state.principal_balance - sell_transaction_cost(state, det)


def owner_monthly_cost(state: BuyState, det: DeterministicParams) -> float:
    return (
        mortgage_payment(det)
        + property_tax(state, det)
        + det.hoa_monthly
        + det.insurance_monthly
        + maintenance_cost(state, det)
    )


def home_sale_tax(state: BuyState, det: DeterministicParams, months_held: int) -> float:
    gain = state.home_value - det.home_price
    if months_held >= 24:
        gain = max(0.0, gain - det.home_sale_exclusion)
    else:
        gain = max(0.0, gain)
    rate = det.ltcg_rate if months_held > 12 else det.stcg_rate
    return gain * rate


def net_worth_buy(state: BuyState, det: DeterministicParams, months_held: int) -> float:
    proceeds = net_sale_proceeds(state, det) - home_sale_tax(state, det, months_held)
    inv_after_tax = state.investment_balance - investment_tax(state.investment_balance, state.investment_cost_basis, det, months_held)
    return proceeds + inv_after_tax


def mortgage_interest_tax_benefit(interest_paid_ytd: float, det: DeterministicParams) -> float:
    itemized = interest_paid_ytd + det.salt_deduction
    benefit = max(0.0, itemized - det.standard_deduction) * det.marginal_tax_rate
    return benefit


def step_buy(state: BuyState, det: DeterministicParams, rates: AnnualRates, is_year_end: bool = False) -> BuyState:
    monthly_inv_return = annual_to_monthly(rates.investment_return)
    monthly_appr = annual_to_monthly(rates.appreciation)

    # Amortization
    interest = state.principal_balance * (det.interest_rate / 12)
    principal = mortgage_payment(det) - interest
    new_balance = state.principal_balance - principal

    # Home appreciates each month
    new_home_value = state.home_value * (1 + monthly_appr)

    # Total monthly outflow
    cost = (
        mortgage_payment(det)
        + new_home_value * det.property_tax_rate / 12
        + det.hoa_monthly
        + det.insurance_monthly
        + new_home_value * det.maintenance_rate / 12
    )

    # Track interest for tax deduction
    new_interest_ytd = state.interest_paid_ytd + interest

    # Annual mortgage interest tax benefit, credited at year-end
    tax_benefit = 0.0
    if is_year_end:
        tax_benefit = mortgage_interest_tax_benefit(new_interest_ytd, det)
        new_interest_ytd = 0.0

    # Investments grow, add income, subtract costs, add tax benefit
    net_contribution = det.monthly_income - cost + tax_benefit
    new_investments = state.investment_balance * (1 + monthly_inv_return) + net_contribution

    return BuyState(
        home_value=new_home_value,
        principal_balance=new_balance,
        investment_balance=new_investments,
        investment_cost_basis=state.investment_cost_basis + net_contribution,
        interest_paid_ytd=new_interest_ytd,
    )


def buy_snapshot(state: BuyState, det: DeterministicParams, t: int) -> dict:
    return {
        "t": t,
        "home_value": state.home_value,
        "principal_balance": state.principal_balance,
        "mortgage_payment": mortgage_payment(det),
        "interest_payment": interest_payment(state, det),
        "principal_payment": principal_payment(state, det),
        "property_tax": property_tax(state, det),
        "maintenance_cost": maintenance_cost(state, det),
        "equity": equity(state),
        "sell_transaction_cost": sell_transaction_cost(state, det),
        "home_sale_tax": home_sale_tax(state, det, t),
        "net_sale_proceeds": net_sale_proceeds(state, det) - home_sale_tax(state, det, t),
        "owner_monthly_cost": owner_monthly_cost(state, det),
        "investment_balance_buy": state.investment_balance,
        "investment_tax_buy": investment_tax(state.investment_balance, state.investment_cost_basis, det, t),
        "net_worth_buy": net_worth_buy(state, det, t),
        "interest_paid_ytd": state.interest_paid_ytd,
    }
