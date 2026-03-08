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
