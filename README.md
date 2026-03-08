# Housing

Monte Carlo simulation comparing rent vs. buy strategies over 1–7 year horizons. Estimates the distribution of net worth difference $\Delta(t) = \text{NW}_{\text{buy}}(t) - \text{NW}_{\text{rent}}(t)$.

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
