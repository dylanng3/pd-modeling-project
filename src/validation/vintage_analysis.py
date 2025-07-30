"""
Vintage Analysis Module
=======================

This module performs vintage-level stability and backtesting analysis for the Home Credit Default Risk dataset.

- Loads data from the original files (application_train, installments_payments) located in data/external/raw/.
- Computes key metrics by vintage (loan origination month): number of loans, number of customers, default rate, AUC by vintage (if scores are provided), and behavioral metrics (delay, paid ratio).
- Generates charts and exports results if needed.

This module is designed to support model monitoring, risk analytics, and temporal performance tracking in credit risk modeling projects. 
It enables users to analyze how risk and model performance evolve over time, identify trends, and detect potential data or model drift across different loan origination periods.

Quick test:
- Simply remove the triple quotes at the bottom;
- Run the file directly:
      python -m src.validation.vintage_analysis
to execute the main block.
"""

from __future__ import annotations
from src.validation.performance_metrics import auc_gini

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Project‑local utilities
from src.data_pipeline.loaders import load_application_train, load_installments_payments


__all__ = [
    "prepare_vintage",
    "compute_vintage_metrics",
    "plot_vintage_curve",
    "run",
]

# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

def _resolve_snapshot_date(df_days: pd.Series, snapshot_date: str | pd.Timestamp = "2018-01-01") -> pd.Series:
    """Convert a DAYS_* series (negative int) into actual datetime."""
    snap = pd.to_datetime(snapshot_date)
    return snap + pd.to_timedelta(df_days, unit="D")


def prepare_vintage(installments: pd.DataFrame, snapshot_date: str | pd.Timestamp = "2018-01-01") -> pd.DataFrame:
    """Return a DataFrame with one row per *loan* (SK_ID_PREV) and its vintage_month.

    We take the **earliest** planned instalment date (DAYS_INSTALMENT) as a proxy
    for loan origination.
    """
    # Earliest instalment per loan
    grp = installments.groupby(["SK_ID_PREV", "SK_ID_CURR"], as_index=False)["DAYS_INSTALMENT"].min()
    grp["origination_date"] = _resolve_snapshot_date(grp["DAYS_INSTALMENT"], snapshot_date)
    grp["vintage_month"] = grp["origination_date"].dt.to_period("M")
    return grp[["SK_ID_PREV", "SK_ID_CURR", "vintage_month"]]


def compute_payment_behaviour(installments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavioural metrics per loan:

    * mean_delay_days:  (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
    * paid_ratio      :  AMT_PAYMENT / AMT_INSTALMENT  (mean)
    """
    df = installments.copy()
    df["delay"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["paid_ratio"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)

    agg = df.groupby("SK_ID_PREV").agg(
        mean_delay_days=("delay", "mean"),
        mean_paid_ratio=("paid_ratio", "mean"),
    )
    return agg.reset_index()


def compute_vintage_metrics(
    installments: pd.DataFrame,
    application: pd.DataFrame,
    pd_scores: Optional[pd.Series] = None,
    snapshot_date: str | pd.Timestamp = "2018-01-01",
) -> pd.DataFrame:
    """Return a DataFrame with one row per vintage‑month & metrics.

    Parameters
    ----------
    installments : raw installments_payments table
    application  : application_train with TARGET
    pd_scores    : optional Series indexed by SK_ID_CURR with model PD score – to compute AUC/Gini
    """
    vint = prepare_vintage(installments, snapshot_date)
    behav = compute_payment_behaviour(installments)

    # Merge behaviour into vintage table (loan level)
    loan_level = vint.merge(behav, on="SK_ID_PREV", how="left")

    # Merge TARGET & optional score – at customer level
    loan_level = loan_level.merge(
        application[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="left"
    )
    if pd_scores is not None:
        loan_level = loan_level.merge(pd_scores.rename("pd_score"), left_on="SK_ID_CURR", right_index=True, how="left")

    # Aggregate to vintage‑month
    agg_funcs = {
        "SK_ID_PREV": "count",
        "SK_ID_CURR": pd.Series.nunique,
        "TARGET": "mean",
        "mean_delay_days": "mean",
        "mean_paid_ratio": "mean",
    }
    vintage_df = loan_level.groupby("vintage_month").agg(agg_funcs).rename(columns={
        "SK_ID_PREV": "num_loans",
        "SK_ID_CURR": "num_customers",
        "TARGET": "default_rate",
    })

    # AUC per vintage (if score provided)
    if pd_scores is not None and "pd_score" in loan_level:
        auc_list: list[Tuple[str, float]] = []
        for vm, sub in loan_level.groupby("vintage_month"):
            if sub["pd_score"].notna().sum() > 20 and sub["TARGET"].nunique() == 2:
                auc_val, _ = auc_gini(sub["TARGET"], sub["pd_score"])
                auc_list.append((vm, auc_val))
        auc_df = pd.DataFrame(auc_list, columns=["vintage_month", "AUC"]).set_index("vintage_month")
        vintage_df = vintage_df.join(auc_df, how="left")

    # Reset index for easy writing/export
    vintage_df = vintage_df.reset_index()
    return vintage_df


def plot_vint_default_rate(vintage_df: pd.DataFrame, metric: str = "default_rate", save_path: str = None, title: str = None):
    plt.figure(figsize=(12, 6))
    x = vintage_df["vintage_month"].astype(str)
    y = vintage_df[metric]
    plt.plot(x, y, marker="o")
    step = max(1, len(x) // 12)
    plt.xticks(ticks=range(0, len(x), step), labels=[x.iloc[i] for i in range(0, len(x), step)], rotation=30, ha="right")
    plt.ylabel(metric)
    if title:
        plt.title(title)
    else:
        if metric == "default_rate":
            plt.title("Default Rate by Vintage Month")
        else:
            plt.title(f"Vintage trend – {metric}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_vint_multi(vintage_df: pd.DataFrame, metrics: list[str], save_path: str = None, title: str = None):
    plt.figure(figsize=(10, 5))
    x = vintage_df["vintage_month"].astype(str)
    for metric in metrics:
        plt.plot(x, vintage_df[metric], marker="o", label=metric)
    # Tích hợp step cho nhãn trục X
    step = max(1, len(x) // 12)
    plt.xticks(ticks=range(0, len(x), step), labels=[x.iloc[i] for i in range(0, len(x), step)], rotation=30, ha="right")
    plt.ylabel("Value")
    if title:
        plt.title(title)
    else:
        plt.title("Vintage Analysis: Default Rate, Delay & Paid Ratio")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_vint_bar(vintage_df: pd.DataFrame, metric: str = "num_loans", save_path: str = None, title: str = None):
    plt.figure(figsize=(10, 5))
    x = vintage_df["vintage_month"].astype(str)
    plt.bar(x, vintage_df[metric], color="#1976d2", alpha=0.8)
    # Tích hợp step cho nhãn trục X
    step = max(1, len(x) // 12)
    plt.xticks(ticks=range(0, len(x), step), labels=[x.iloc[i] for i in range(0, len(x), step)], rotation=30, ha="right")
    plt.ylabel(metric)
    if title:
        plt.title(title)
    else:
        if metric == "num_loans":
            plt.title("Number of Loans by Vintage Month")
        elif metric == "num_customers":
            plt.title("Number of Customers by Vintage Month")
        else:
            plt.title(f"Vintage bar chart – {metric}")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_vint_behaviour(vintage_df: pd.DataFrame, save_path: str = None, title: str = None):
    x = vintage_df["vintage_month"].astype(str)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "#d32f2f"
    color2 = "#388e3c"
    ax1.set_xlabel("Vintage Month")
    ax1.set_ylabel("Mean Delay Days", color=color1)
    ax1.plot(x, vintage_df["mean_delay_days"], marker="o", color=color1, label="mean_delay_days")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Paid Ratio", color=color2)
    ax2.plot(x, vintage_df["mean_paid_ratio"], marker="s", color=color2, label="mean_paid_ratio")
    ax2.tick_params(axis="y", labelcolor=color2)
    # Tích hợp step cho nhãn trục X
    step = max(1, len(x) // 12)
    ax1.set_xticks(range(0, len(x), step))
    ax1.set_xticklabels([x.iloc[i] for i in range(0, len(x), step)], rotation=30, ha="right")
    if title:
        plt.title(title)
    else:
        plt.title("Vintage Behavioural Metrics: Delay & Paid Ratio")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()


# For quick test
if __name__ == "__main__":
    # End-to-end vintage analysis for Home Credit dataset
    snapshot_date = "2018-01-01"
    export_csv = "validation_results/vintage_results/vintage_metrics.csv"

    print("Loading data…")
    app = load_application_train()
    inst = load_installments_payments()

    print("Computing vintage metrics…")
    vintage_df = compute_vintage_metrics(inst, app, snapshot_date=snapshot_date)

    print(vintage_df.head())

    if export_csv:
        Path(export_csv).parent.mkdir(parents=True, exist_ok=True)
        vintage_df.to_csv(export_csv, index=False)
        print(f"Vintage metrics saved to {export_csv}")

    plot_vint_default_rate(vintage_df, metric="default_rate", save_path="validation_results/vintage_results/vintage_default_rate.png")

    plot_vint_multi(vintage_df, metrics=["default_rate", "mean_delay_days", "mean_paid_ratio"], save_path="validation_results/vintage_results/vintage_multi_metric.png")

    plot_vint_bar(vintage_df, metric="num_loans", save_path="validation_results/vintage_results/vintage_num_loans.png")

    plot_vint_bar(vintage_df, metric="num_customers", save_path="validation_results/vintage_results/vintage_num_customers.png")

    plot_vint_behaviour(vintage_df, save_path="validation_results/vintage_results/vintage_behaviour.png")

    print("Plot saved to validation_results/vintage_results")