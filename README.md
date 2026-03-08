# Housing

Monte Carlo simulation comparing rent vs. buy strategies. Estimates the distribution of net worth difference for variable time horizons.

## Setup

```
uv venv && uv pip install -e .
```

Then open `housing_calculations.ipynb` with the kernel pointed at `.venv/bin/python`.

## Project Structure

- `housing_calculations.ipynb` — main notebook for configuring parameters and running simulations
- `housing/params.py` — parameter dataclasses (`DeterministicParams`, `StochasticParams`), rate path sampling, and defaults
- `housing/buy.py` — buy-side state, derived quantities (mortgage, equity, etc.), and monthly step function
- `housing/rent.py` — rent-side state, derived quantities, and monthly step function
- `housing/simulation.py` — single-trial simulation and Monte Carlo runner
- `housing/plotting.py` — summary statistics table and visualization (violin, fan chart, P(buy wins), net worth comparison)
