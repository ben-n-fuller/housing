import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


DOLLAR_K_FMT = FuncFormatter(lambda x, _: f"${x/1000:.0f}k")
PERCENT_FMT = FuncFormatter(lambda x, _: f"{x:.0%}")


def delta_summary(results: pd.DataFrame) -> pd.io.formats.style.Styler:
    summary = results.groupby("horizon_years")["delta"].agg(
        mean="mean",
        median="median",
        std="std",
        p10=lambda x: x.quantile(0.10),
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
        prob_buy_wins=lambda x: (x > 0).mean(),
    )
    return summary.style.format({
        "mean": "${:,.0f}", "median": "${:,.0f}", "std": "${:,.0f}",
        "p10": "${:,.0f}", "p25": "${:,.0f}", "p75": "${:,.0f}", "p90": "${:,.0f}",
        "prob_buy_wins": "{:.1%}",
    })


def plot_delta_violin(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=results, x="horizon_years", y="delta", inner="quartile", cut=0, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("$\\Delta$ (Buy − Rent)")
    ax.set_title("Distribution of $\\Delta(t)$ by Horizon")
    ax.yaxis.set_major_formatter(DOLLAR_K_FMT)
    fig.tight_layout()
    return fig


def plot_prob_buy_wins(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    prob = results.groupby("horizon_years")["delta"].apply(lambda x: (x > 0).mean())
    ax.bar(prob.index, prob.values, color="steelblue", edgecolor="white")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("P($\\Delta > 0$)")
    ax.set_title("Probability that Buying Wins")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PERCENT_FMT)
    for i, v in enumerate(prob.values):
        ax.text(prob.index[i], v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
    fig.tight_layout()
    return fig


def plot_delta_fan(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    g = results.groupby("horizon_years")["delta"]
    median = g.median()
    p10 = g.quantile(0.10)
    p90 = g.quantile(0.90)

    ax.plot(median.index, median.values, marker="o", color="steelblue", label="Median")
    ax.fill_between(median.index, p10.values, p90.values, alpha=0.2, color="steelblue", label="10th–90th percentile")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("$\\Delta$ (Buy − Rent)")
    ax.set_title("Median $\\Delta(t)$ with 80% Confidence Band")
    ax.yaxis.set_major_formatter(DOLLAR_K_FMT)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_net_worth_comparison(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    g = results.groupby("horizon_years")
    nw_buy = g["net_worth_buy"].median()
    nw_rent = g["net_worth_rent"].median()

    x = np.arange(len(nw_buy))
    width = 0.35
    ax.bar(x - width / 2, nw_buy.values, width, label="Buy", color="steelblue")
    ax.bar(x + width / 2, nw_rent.values, width, label="Rent", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(nw_buy.index)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Median Net Worth")
    ax.set_title("Median Net Worth: Buy vs. Rent")
    ax.yaxis.set_major_formatter(DOLLAR_K_FMT)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Aggregate sweep plots
# ---------------------------------------------------------------------------

SWEEP_FEATURES = [
    ("appreciation_mean", "Appreciation Rate"),
    ("interest_rate", "Mortgage Rate"),
    ("sell_cost_rate", "Sell Transaction Cost"),
    ("home_price", "Home Price"),
    ("horizon", "Horizon (years)"),
]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (coefficients, R²) for OLS with intercept appended to *X*."""
    Xa = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    y_hat = Xa @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def plot_ols_coefficients(sweep_df: pd.DataFrame) -> plt.Figure:
    """Standardised OLS coefficients for P(buy wins) and mean delta."""
    cols = [c for c, _ in SWEEP_FEATURES]
    labels = [l for _, l in SWEEP_FEATURES]

    X_raw = sweep_df[cols].values.astype(float)
    mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0)
    X_std = (X_raw - mu) / sigma

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (target, title) in zip(axes, [
        ("prob_buy_wins", "P(Buy Wins)"),
        ("mean_delta", "Mean Δ (Buy − Rent)"),
    ]):
        y = sweep_df[target].values
        beta, r2 = _ols(X_std, y)
        coefs = beta[:-1]
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coefs]
        ax.barh(labels, coefs, color=colors)
        ax.set_xlabel("Standardized Coefficient")
        ax.set_title(f"{title}  (R² = {r2:.3f})")
        ax.axvline(0, color="black", linewidth=0.5)

    fig.suptitle(
        "Linear Sensitivity Analysis (standardized coefficients)", fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_partial_dependence(sweep_df: pd.DataFrame) -> plt.Figure:
    """Marginal effect of each parameter on P(buy wins)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fmt_map = {
        "appreciation_mean": lambda x: f"{x:.0%}",
        "interest_rate": lambda x: f"{x:.1%}",
        "sell_cost_rate": lambda x: f"{x:.0%}",
        "home_price": lambda x: f"${x / 1e3:.0f}K",
        "horizon": lambda x: f"{x:.0f}",
    }

    for ax, (col, label) in zip(axes.flat, SWEEP_FEATURES):
        pd_data = sweep_df.groupby(col)["prob_buy_wins"].mean()
        ax.plot(
            pd_data.index, pd_data.values, "o-",
            linewidth=2, markersize=8, color="#3498db",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("P(Buy Wins)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_xticks(pd_data.index)
        ax.set_xticklabels([fmt_map[col](x) for x in pd_data.index])

    axes.flat[-1].set_visible(False)
    fig.suptitle(
        "Partial Dependence: P(Buy Wins) averaged over all other parameters",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_joint_heatmap(
    sweep_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> plt.Figure:
    """Appreciation × mortgage-rate heatmaps of P(buy wins), one per horizon."""
    if horizons is None:
        horizons = sorted(sweep_df["horizon"].unique())

    fig, axes = plt.subplots(
        1, len(horizons), figsize=(5 * len(horizons), 4.5), sharey=True,
    )
    if len(horizons) == 1:
        axes = [axes]

    im = None
    for ax, h in zip(axes, horizons):
        hm = (
            sweep_df[sweep_df["horizon"] == h]
            .groupby(["appreciation_mean", "interest_rate"])["prob_buy_wins"]
            .mean()
            .unstack()
        )
        im = ax.pcolormesh(
            range(hm.shape[1] + 1), range(hm.shape[0] + 1), hm.values,
            cmap="RdYlGn", vmin=0, vmax=1,
        )
        for ri in range(hm.shape[0]):
            for ci in range(hm.shape[1]):
                v = hm.values[ri, ci]
                color = "black" if 0.3 < v < 0.7 else "white"
                ax.text(
                    ci + 0.5, ri + 0.5, f"{v:.0%}",
                    ha="center", va="center", fontsize=9, color=color,
                )
        ax.set_xticks([i + 0.5 for i in range(hm.shape[1])])
        ax.set_xticklabels([f"{x:.1%}" for x in hm.columns], fontsize=8)
        ax.set_xlabel("Mortgage Rate")
        ax.set_title(f"Year {h}")
        if ax is axes[0]:
            ax.set_yticks([i + 0.5 for i in range(hm.shape[0])])
            ax.set_yticklabels([f"{x:.0%}" for x in hm.index])
            ax.set_ylabel("Appreciation Rate")

    fig.colorbar(im, ax=axes, label="P(Buy Wins)", shrink=0.85, pad=0.02)
    fig.suptitle(
        "P(Buy Wins): Appreciation × Mortgage Rate "
        "(marginalized over price & sell cost)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_breakeven_heatmap(sweep_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of interpolated breakeven horizon (years until P(buy wins) ≥ 50%)."""
    param_cols = ["appreciation_mean", "interest_rate", "sell_cost_rate", "home_price"]
    breakeven_records: list[dict] = []

    for keys, group in sweep_df.groupby(param_cols):
        gs = group.sort_values("horizon")
        h_vals = gs["horizon"].values.astype(float)
        p_vals = gs["prob_buy_wins"].values

        be = np.nan
        if p_vals[0] >= 0.5:
            be = h_vals[0]
        else:
            for j in range(len(p_vals) - 1):
                if p_vals[j] < 0.5 <= p_vals[j + 1]:
                    be = h_vals[j] + (
                        (0.5 - p_vals[j])
                        / (p_vals[j + 1] - p_vals[j])
                        * (h_vals[j + 1] - h_vals[j])
                    )
                    break
        breakeven_records.append(
            dict(zip(param_cols, keys)) | {"breakeven_years": be},
        )

    breakeven_df = pd.DataFrame(breakeven_records)
    be_hm = (
        breakeven_df
        .groupby(["appreciation_mean", "interest_rate"])["breakeven_years"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        range(be_hm.shape[1] + 1), range(be_hm.shape[0] + 1), be_hm.values,
        cmap="RdYlGn_r", vmin=1, vmax=12,
    )
    for ri in range(be_hm.shape[0]):
        for ci in range(be_hm.shape[1]):
            v = be_hm.values[ri, ci]
            txt = f"{v:.1f}" if not np.isnan(v) else ">10"
            ax.text(
                ci + 0.5, ri + 0.5, txt,
                ha="center", va="center", fontsize=11, fontweight="bold",
            )
    ax.set_xticks([i + 0.5 for i in range(be_hm.shape[1])])
    ax.set_xticklabels([f"{x:.1%}" for x in be_hm.columns])
    ax.set_yticks([i + 0.5 for i in range(be_hm.shape[0])])
    ax.set_yticklabels([f"{x:.0%}" for x in be_hm.index])
    ax.set_xlabel("Mortgage Rate")
    ax.set_ylabel("Appreciation Rate")
    ax.set_title("Breakeven Horizon (years until P(Buy Wins) ≥ 50%)")
    fig.colorbar(im, ax=ax, label="Years")
    fig.tight_layout()
    return fig
