from scipy import stats
import numpy as np


def check_normalitiy(x, effect_name=""):
    stat, p_value = stats.normaltest(x)
    label = f"[{effect_name}] " if effect_name else ""

    print(f"{label}Stat: {stat:.4f}")
    print(f"{label}P-value: {p_value:.4f}")


def check_day_of_the_week_normality(df):
    for i in range(5):
        filtered_dfnz = df[df['day'] == i]['Return']
        check_normalitiy(filtered_dfnz, f"day = {i} Returns")


def check_sell_in_may_normality(df):
    df = df.copy()
    df["Season"] = np.where(df["Month"].between(5, 10), "summer", "winter")
    df["SeasonYear"] = np.where(df["Month"].isin(
        [11, 12]), df["Year"]+1, df["Year"])
    g = df.groupby(["Ticker", "SeasonYear", "Season"])[
        "Return"].sum().reset_index()
    season_mean = g.groupby(["SeasonYear", "Season"])[
        "Return"].mean().reset_index(name="mean")
    pivot = season_mean.pivot(
        index="SeasonYear", columns="Season", values="mean").dropna()
    summer_returns = pivot["summer"]
    winter_returns = pivot["winter"]
    check_normalitiy(summer_returns, "Summer Returns")
    check_normalitiy(winter_returns, "Winter Returns")


def check_period_normality(df):
    for period in ["ToM", "Middle", "Rest"]:
        x = df.loc[df["period_month_general"] == period, "Return"]
        check_normalitiy(x, effect_name=period)


def calculate_welch_t_test(x, y, effect_name=""):
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
    label = f"[{effect_name}] " if effect_name else ""

    print(f"{label}T-test: {t_stat:.4f}, P-value: {p_value:.4f}")


def tom_ttest(df):
    tom = df.loc[df["period_month_general"] == "ToM", "Return"]
    rest = df.loc[df["period_month_general"] == "Rest", "Return"]
    mid = df.loc[df["period_month_general"] == "Middle", "Return"]

    calculate_welch_t_test(tom, rest, "Turn of the month effect")
    calculate_welch_t_test(mid, rest, "Middle of the month effect")


def moday_effect_ttest(df):
    for i in range(5):
        for j in range(5):
            filtered_dfnz = df.loc[df["day"] == i, "Return"]
            filtered_df = df.loc[df["day"] == j, "Return"]
            calculate_welch_t_test(
                filtered_dfnz, filtered_df, f"Day {i} vs day {j}")


def calculate_mannwhitneyu(x, y, effect_name="", alternative="greater"):
    stat, p_value = stats.mannwhitneyu(x, y, alternative=alternative)
    label = f"[{effect_name}] " if effect_name else ""

    print(f"{label}mannwhitneyu-test: {stat:.4f}, P-value: {p_value:.4f}")


def moday_effect_calculate_mannwhitneyu(df):
    for i in range(5):
        for j in range(i, 5):
            if i == j:
                continue
            filtered_dfnz = df.loc[df["day"] == i, "Return"]
            filtered_df = df.loc[df["day"] == j, "Return"]
            calculate_mannwhitneyu(
                filtered_dfnz, filtered_df, f"Day {i} vs day {j}", alternative="two-sided")
