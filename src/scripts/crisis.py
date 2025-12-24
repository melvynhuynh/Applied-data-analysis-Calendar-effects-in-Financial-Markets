import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def add_dotcom_period(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["dotcom_period"] = "post_bubble"
    df.loc[df["Date"] < "2000-01-01", "dotcom_period"] = "pre_bubble"
    df.loc[
        (df["Date"] >= "2000-01-01") & (df["Date"] <= "2002-12-31"),
        "dotcom_period"
    ] = "bubble"

    return df


def plot_january_dotcom(january_df):
    order = ["pre_bubble", "bubble", "post_bubble"]

    plt.figure(figsize=(8, 5))

    sns.boxplot(data=january_df, x="dotcom_period",
                y="Return", order=order, showfliers=True)

    plt.title("January Returns Across Dot-Com Market Periods")
    plt.xlabel("Market Periods")
    plt.ylabel("Daily Return")

    plt.tight_layout()
    plt.show()


def plot_jan_dotcom_average(january_df):
    summary = (january_df.groupby("dotcom_period")["Return"].agg(
        mean="mean", std="std", n="count").reset_index())

    summary["ci"] = 1.96 * summary["std"] / np.sqrt(summary["n"])
    order = ["pre_bubble", "bubble", "post_bubble"]

    means = summary.set_index("dotcom_period").loc[order, "mean"]
    cis = summary.set_index("dotcom_period").loc[order, "ci"]

    plt.figure(figsize=(8, 5))

    plt.bar(order, means, yerr=cis, capsize=6)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.title(
        "Average January Returns Across Dot Com Market (95% confidence intervals Error bars)")
    plt.xlabel("Market Regime")
    plt.ylabel("Average Daily Return")

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()


def add_global_financial_crisis_period(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["financial_crisis"] = "post_crisis"
    df.loc[df["Date"] < "2008-01-01", "financial_crisis"] = "pre_crisis"
    df.loc[
        (df["Date"] >= "2008-01-01") & (df["Date"] <= "2009-12-31"),
        "financial_crisis"
    ] = "crisis"

    return df


def plot_jan_crisis(january_df):
    order = ["pre_crisis", "crisis", "post_crisis"]
    plt.figure(figsize=(8, 5))

    sns.boxplot(data=january_df, x="financial_crisis",
                y="Return", order=order, showfliers=True)

    plt.title("January Returns Across Global Financial Crisis Regimes")
    plt.xlabel("Market Regime")
    plt.ylabel("Daily Return")

    plt.tight_layout()
    plt.show()


def plot_jan_crisis_average(january_df):

    summary = (january_df.groupby("financial_crisis")["Return"].agg(
        mean="mean", std="std", n="count").reset_index())

    summary["ci"] = 1.96 * summary["std"] / np.sqrt(summary["n"])
    order = ["pre_crisis", "crisis", "post_crisis"]

    means = summary.set_index("financial_crisis").loc[order, "mean"]
    cis = summary.set_index("financial_crisis").loc[order, "ci"]

    plt.figure(figsize=(8, 5))
    plt.bar(order, means, yerr=cis, capsize=6)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.title("Average January Returns Across Global Financial Crisis Regimes")
    plt.xlabel("Market Regime")
    plt.ylabel("Average Daily Return")

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()


def plot_scr_crisis(scr_df):
    order = ["pre_crisis", "crisis", "post_crisis"]

    plt.figure(figsize=(8, 5))

    sns.boxplot(data=scr_df, x="financial_crisis",
                y="Return", order=order, showfliers=True)

    plt.title("Santa Claus Rally Returns Across Global Financial Crisis Regimes")
    plt.xlabel("Market Regime")
    plt.ylabel("Daily Return")

    plt.tight_layout()
    plt.show()


def plot_scr_crisis_average(scr_df):
    summary_scr = (scr_df.groupby("financial_crisis")["Return"].agg(
        mean="mean", std="std", n="count").reset_index())

    summary_scr["ci"] = 1.96 * summary_scr["std"] / np.sqrt(summary_scr["n"])
    order = ["pre_crisis", "crisis", "post_crisis"]

    means = summary_scr.set_index("financial_crisis").loc[order, "mean"]
    cis = summary_scr.set_index("financial_crisis").loc[order, "ci"]

    plt.figure(figsize=(8, 5))

    plt.bar(order, means, yerr=cis, capsize=6)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.title(
        "Average Santa Claus Rally Returns Across Global Financial Crisis Regimes")
    plt.xlabel("Market Regime")
    plt.ylabel("Average Daily Return")

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()


def plot_scr_bubble(scr_df):
    order = ["pre_bubble", "bubble", "post_bubble"]

    plt.figure(figsize=(8, 5))

    sns.boxplot(data=scr_df, x="dotcom_period",
                y="Return", order=order, showfliers=True)

    plt.title("Santa Claus Rally Returns Across Dot-Com Bubble Regimes")
    plt.xlabel("Market Regime")
    plt.ylabel("Daily Return")

    plt.tight_layout()
    plt.show()


def plot_scr_bubble_average(scr_df):
    summary_scr = (scr_df.groupby("dotcom_period")["Return"].agg(
        mean="mean", std="std", n="count").reset_index())

    summary_scr["ci"] = 1.96 * summary_scr["std"] / np.sqrt(summary_scr["n"])
    order = ["pre_bubble", "bubble", "post_bubble"]

    means = summary_scr.set_index("dotcom_period").loc[order, "mean"]
    cis = summary_scr.set_index("dotcom_period").loc[order, "ci"]

    plt.figure(figsize=(8, 5))

    plt.bar(order, means, yerr=cis, capsize=6)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.title("Average Santa Claus Rally Returns Across Dot-Com Bubble Regimes")
    plt.xlabel("Market Regime")
    plt.ylabel("Average Daily Return")

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()
