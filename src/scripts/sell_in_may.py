import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import networkx as nx
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

PARQ_DIR = r"C:\Users\FURKAN\Desktop\stocks_parquet"
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def load_all_daily(parq_dir: str):
    """
    I loaded all daily stock data from parquet files.
    I merged them into one DataFrame and cleaned missing values.
    """
    paths = glob.glob(os.path.join(parq_dir, "*.parquet")) # I collected all parquet paths
    dfs = [pd.read_parquet(p, columns=["Date","Open","Close","Adj Close","Volume","Ticker"]) for p in paths]
    d = pd.concat(dfs, ignore_index=True)  # I merged all files into one dataset
    d["Date"] = pd.to_datetime(d["Date"]) # I converted dates to datetime objects
    return d.dropna(subset=["Date","Open","Close","Adj Close","Ticker"]) # I removed incomplete rows

def compute_log_returns(df: pd.DataFrame):
    """
    I calculated the daily returns for each stock.
    I also added year and month info for later seasonal analysis.
    """
    d = df.sort_values(["Ticker","Date"]).copy() # I sorted by ticker and date to align prices
    for c in ["Adj Close","Open","Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")  # I ensured numeric values only
    # I filtered out rows with invalid or zero prices
    d = d[(d["Adj Close"]>0) & (d["Open"]>0) & (d["Close"]>0)]
    # I found the previous day's adjusted close for each ticker
    prev = d.groupby("Ticker")["Adj Close"].shift(1)
    # I computed the continuous compounding  return
    d["Return"] = np.log(d["Adj Close"] / prev)
    # I extracted time-based features
    d["Year"]  = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    return d.dropna(subset=["Return"])

def sell_in_may_report_plot(d: pd.DataFrame):
    """
    I analyzed and visualized the 'Sell in May' effect.
    I compared average stock returns between summer (May–Oct) and winter (Nov–Apr) periods.
    """
    m = d.copy() # I made a copy to avoid modifying the original data
    # I labeled months 5–10 as summer and the rest as winter
    m["Season"] = np.where(m["Month"].between(5,10), "summer", "winter")
    # I shifted the year for Nov–Dec so that each winter spans one consistent season-year
    m["SeasonYear"] = np.where(m["Month"].isin([11,12]), m["Year"]+1, m["Year"])
    # I summed daily returns per ticker, season, and year
    g = m.groupby(["Ticker","SeasonYear","Season"])["Return"].sum().reset_index()
    # I averaged returns across all tickers for each season-year
    season_mean = g.groupby(["SeasonYear","Season"])["Return"].mean().reset_index(name="mean_logret")
    # I pivoted data to have separate columns for summer and winter returns
    pivot = season_mean.pivot(index="SeasonYear", columns="Season", values="mean_logret").dropna()
    diff_pct = (np.exp(pivot["winter"] - pivot["summer"]) - 1.0) * 100
    diff_log = pivot["winter"] - pivot["summer"]
    # I performed a one-sample t-test to check if mean(winter−summer) > 0
    summer_returns = pivot["summer"]
    winter_returns = pivot["winter"]
    # Mann-Whitney U test between summer and winter returns
    stat, p_val = mannwhitneyu(summer_returns, winter_returns)
    plt.figure(figsize=(10,5))
    plt.bar(pivot.index.astype(int), diff_pct.values)
    plt.axhline(0, ls="--", lw=1, c="black")
    plt.title(f"Sell in May: Winter − Summer (%)  mean={diff_pct.mean():.2f}%  p={p_val:.4f}  N={len(pivot)}")
    plt.xlabel("Season Year"); plt.ylabel("Winter − Summer (%)")
    plt.tight_layout()
    plt.show()

def label_season_with_pivot(d: pd.DataFrame, pivot_month: int) -> pd.DataFrame:
    df = d.copy()
    """
    I labeled each record as 'summer' or 'winter' based on a chosen pivot month.
    I also created a 'CycleYear' column to align seasons that cross over calendar years.
    if pivot_month = 5 (May), then 'summer' = May–Oct, 'winter' = Nov–Apr.
    """
    # I calculated how far each record's month is from the pivot (values from 0 to 11)
    k = (df["Month"] - pivot_month) % 12
    # I labeled months within 6 months after the pivot as "summer", others as "winter"
    df["Season"] = np.where(k <= 5, "summer", "winter")
    # I shifted the year for months before the pivot
    df["CycleYear"] = np.where(df["Month"] < pivot_month, df["Year"] + 1, df["Year"])
    return df

def seasonal_stats_for_pivot(d: pd.DataFrame, pivot_month: int):
    """
    I computed seasonal performance stats for a given pivot month.
    I labeled seasons and aligned them into a CycleYear.
    I summed per-ticker returns within each season, averaged across tickers per year,
    then tested the winter − summer return gap with a one-sample t-test.
    """
    df = label_season_with_pivot(d, pivot_month)  # I created Season and CycleYear fields
    # I summed seasonal returns per ticker, then averaged across tickers for each CycleYear
    season_mean = (
        df.groupby(["Ticker", "CycleYear", "Season"])["Return"].sum()
          .groupby(["CycleYear", "Season"]).mean()
          .unstack("Season")
    )
    # I enforced column order and dropped incomplete years
    season_mean = season_mean.reindex(columns=["winter", "summer"]).dropna()
    diff_log = season_mean["winter"] - season_mean["summer"]
    diff_pct_mean = (np.exp(diff_log) - 1.0).mean() * 100.0
    return {
        "pivot_month": pivot_month,
        "month_name": MONTH_NAMES[pivot_month-1],
        "N_years": int(len(season_mean)),
        "mean_diff_%": float(diff_pct_mean),
    }

def sell_in_every_month_plot(d: pd.DataFrame):
    """
    I compared 'winter − summer' performance for every possible pivot (1–12).
    I reused seasonal_stats_for_pivot to keep logic consistent across months.
    """
    # I computed seasonal stats for each pivot month
    rows = [seasonal_stats_for_pivot(d, m) for m in range(1, 13)]
    s = pd.DataFrame(rows).sort_values("pivot_month")
    # I kept only months with valid data
    # Normally, here I always get an error for each run of the code, and interestingly the
    # error stopped when I write the this line,
    s = s[(s["N_years"] > 0) & (s["mean_diff_%"].notna())]
    plt.figure(figsize=(10,5))
    plt.bar(s["month_name"], s["mean_diff_%"])
    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.title("Sell in {Month}?  Winter − Summer (%) by pivot month")
    plt.xlabel("Pivot month (summer starts)")
    plt.ylabel("Average difference (%)")
    plt.tight_layout()
    plt.show()

def make_equal_weight_monthly_index(d: pd.DataFrame, logret_col: str = "Return"):
    """
    I built an equal-weighted monthly index from daily returns.
    I first aggregated daily returns per stock per month,
    converted them to simple returns, and averaged equally across all tickers.
    """
    x = d.copy()
    # I extracted year and month from daily dates
    x["Year"]  = x["Date"].dt.year
    x["Month"] = x["Date"].dt.month
    # I summed daily returns for each stock within the same month
    m = x.groupby(["Ticker","Year","Month"], as_index=False)[logret_col].sum()
    # I converted returns to simple percentage returns
    m["ret_ticker"] = np.exp(m[logret_col]) - 1.0
    # I took the equal-weighted average of all tickers for each month
    mret = m.groupby(["Year","Month"], as_index=False)["ret_ticker"].mean().rename(columns={"ret_ticker":"ret"})
    # I set each month's date to its month-end for plotting consistency
    mret["Date"] = pd.to_datetime(dict(year=mret["Year"], month=mret["Month"], day=1)) + pd.offsets.MonthEnd(0)
    # I returned a clean monthly index with date and equal-weighted return
    return mret[["Date","ret"]].sort_values("Date").reset_index(drop=True)

def backtest_sell_in_may_plot(monthly: pd.DataFrame):
    """
    I backtested the 'Sell in May' strategy against a simple Buy & Hold.
    """
    df = monthly.set_index("Date").sort_index()
    # I marked winter months (Nov–Apr) as active periods
    is_winter = df.index.month.isin([11, 12, 1, 2, 3, 4])
    # I kept returns only for winter months, zero otherwise
    strat_gross = np.where(is_winter, df["ret"], 0.0)
    # I computed cumulative growth for Buy & Hold and Sell in May
    bh = (1 + df["ret"]).cumprod()
    sm = (1 + pd.Series(strat_gross, index=df.index)).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(bh.index, bh, label="Buy & Hold")
    plt.plot(sm.index, sm, label="Sell in May (no transaction cost)")
    plt.yscale("log")
    plt.title("Backtest: Sell in May vs Buy & Hold (No Transaction Cost)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def sell_in_may_report_with_fed(d: pd.DataFrame, fed_parquet="data/FEDFUNDS.parquet"):
    """
    I extended the sell_in_may_report_plot analysis by adding Fed Funds data.
    I used the same 'Sell in May' logic (winter vs summer returns),
    but here I compared it with Fed Funds rates for November and May.
    """
    m = d.copy()  # I kept the original intact
    m["Season"] = np.where(m["Month"].between(5, 10), "summer", "winter")
    m["SeasonYear"] = np.where(m["Month"].isin([11, 12]), m["Year"] + 1, m["Year"])
    season_mean = (
        m.groupby(["Ticker", "SeasonYear", "Season"])["Return"].sum()
         .groupby(["SeasonYear", "Season"]).mean()
         .unstack("Season")
         .dropna(subset=["winter", "summer"])
    )
    season_mean["diff_%"] = (np.exp(season_mean["winter"] - season_mean["summer"]) - 1) * 100
    mean_diff = season_mean["diff_%"].mean()
    fed = pd.read_parquet(fed_parquet)
    fed["observation_date"] = pd.to_datetime(fed["observation_date"], errors="coerce")
    fed["Year"] = fed["observation_date"].dt.year
    fed["Month"] = fed["observation_date"].dt.month
    fed_nov = fed.query("Month == 11").groupby("Year")["FEDFUNDS"].mean().reset_index(name="FEDFUNDS_Nov")
    fed_may = fed.query("Month == 5").groupby("Year")["FEDFUNDS"].mean().reset_index(name="FEDFUNDS_May")
    merged = (
        season_mean.reset_index()
        .merge(fed_nov, left_on="SeasonYear", right_on="Year", how="left")
        .merge(fed_may, on="Year", how="left")
    )
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(merged["SeasonYear"], merged["diff_%"], label="Winter−Summer diff (%)", color="lightblue")
    ax1.axhline(0, ls="--", lw=1, c="black")
    ax1.set_xlabel("Season Year")
    ax1.set_ylabel("Winter−Summer (%)")
    ax2 = ax1.twinx()
    ax2.plot(merged["SeasonYear"], merged["FEDFUNDS_Nov"], "--", label="Fed Rate (Nov)")
    ax2.plot(merged["SeasonYear"], merged["FEDFUNDS_May"], "-.", label="Fed Rate (May)")
    ax2.set_ylabel("Fed Funds Rate (%)")
    ax1.set_title(f"Sell in May vs Fed Funds (Nov & May) — mean diff = {mean_diff:.2f}%")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def regression_analysis_seasonality(d: pd.DataFrame):
    """
    I ran a simple seasonality regression using an equal-weight monthly index.
    I avoided taking a mean over daily observations because it biases months with more tickers.
    """
    x = d.copy()
    x["Year"]  = x["Date"].dt.year
    x["Month"] = x["Date"].dt.month

    m = x.groupby(["Ticker", "Year", "Month"], as_index=False)["Return"].sum()

    monthly_data = m.groupby(["Year", "Month"], as_index=False)["Return"].mean()

    monthly_data["is_winter"] = monthly_data["Month"].isin([11, 12, 1, 2, 3, 4]).astype(int)

    model = smf.ols("Return ~ is_winter", data=monthly_data).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )
    print(model.summary())

def risk_adjusted_analysis(d: pd.DataFrame):
    m = d.copy()
    m["Season"] = np.where(m["Month"].between(5,10), "summer", "winter")
    
    stats = m.groupby("Season")["Return"].agg(["mean", "std", "count"])
    stats["sharpe"] = (stats["mean"] / stats["std"]) * np.sqrt(252) 
    
    print("RISK-ADJUSTED STATISTICS")
    print(stats)

    plt.figure(figsize=(8,6))
    sns.boxplot(x="Season", y="Return", data=m)
    plt.title("Return Distribution and Outliers (Summer vs Winter)")
    plt.show()

def predict_market_direction(d: pd.DataFrame):
    """
    I predicted if next month will be positive using an equal-weight monthly index.
    I avoided daily-level means, and I used a proper monthly series.
    """
    x = d.copy()
    x["Year"]  = x["Date"].dt.year
    x["Month"] = x["Date"].dt.month

    m = x.groupby(["Ticker", "Year", "Month"], as_index=False)["Return"].sum()

    monthly = m.groupby(["Year", "Month"], as_index=False)["Return"].mean()
    monthly = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)

    monthly["target"] = (monthly["Return"].shift(-1) > 0).astype(int)
    monthly["prev_ret"] = monthly["Return"].shift(1)
    monthly = monthly.dropna().reset_index(drop=True)

    X = monthly[["Month", "prev_ret"]]
    y = monthly["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(
        n_estimators=500,    
        max_depth=3,           
        min_samples_leaf=10,     
        max_features=1.0,       
        class_weight="balanced", 
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("ML MODEL PERFORMANCE (Monthly Equal-Weight)")
    print(classification_report(y_test, preds))

def regression_with_fed(daily_df, fed_parquet):
    df = daily_df.copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    m = (
        df.groupby(["Ticker", "Year", "Month"], as_index=False)["Return"]
          .sum()
    )
    monthly_ret = (
        m.groupby(["Year", "Month"], as_index=False)["Return"]
         .mean()
    )
    fed = pd.read_parquet(fed_parquet)
    fed["observation_date"] = pd.to_datetime(fed["observation_date"])
    fed["Year"]  = fed["observation_date"].dt.year
    fed["Month"] = fed["observation_date"].dt.month
    df_merged = monthly_ret.merge(
        fed[["Year", "Month", "FEDFUNDS"]],
        on=["Year", "Month"],
        how="inner"
    )
    df_merged["is_winter"] = df_merged["Month"].isin([11,12,1,2,3,4]).astype(int)
    model = smf.ols(
        "Return ~ is_winter + FEDFUNDS",
        data=df_merged
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print("REGRESSION ANALYSIS WITH FED")
    print(model.summary())
    return df_merged, model
    
def prepare_monthly_with_fed(daily_df, fed_parquet):
    df = daily_df.copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    m = df.groupby(["Ticker", "Year", "Month"], as_index=False)["Return"].sum()

    monthly = m.groupby(["Year", "Month"], as_index=False)["Return"].mean()

    fed = pd.read_parquet(fed_parquet)
    fed["observation_date"] = pd.to_datetime(fed["observation_date"])
    fed["Year"]  = fed["observation_date"].dt.year
    fed["Month"] = fed["observation_date"].dt.month

    out = monthly.merge(
        fed[["Year", "Month", "FEDFUNDS"]],
        on=["Year", "Month"],
        how="inner"
    )

    out["is_winter"] = out["Month"].isin([11,12,1,2,3,4]).astype(int)

    return out

def sell_in_may_fed_tightening(daily_df, fed_parquet):
    df = prepare_monthly_with_fed(daily_df, fed_parquet)

    df["fed_change"] = df["FEDFUNDS"].diff()
    df["fed_tightening"] = (df["fed_change"] > 0).astype(int)

    df = df.dropna()

    model = smf.ols(
        "Return ~ is_winter + fed_tightening + is_winter:fed_tightening",
        data=df
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    print("SELL IN MAY, FED TIGHTENING")
    print(model.summary())

    return df, model

def sell_in_may_volatility_regime(daily_df, fed_parquet, window=12):
    df = prepare_monthly_with_fed(daily_df, fed_parquet)

    df = df.sort_values(["Year", "Month"])
    df["vol"] = df["Return"].rolling(window).std()

    median_vol = df["vol"].median()
    df["high_vol"] = (df["vol"] > median_vol).astype(int)

    df = df.dropna()

    model = smf.ols(
        "Return ~ is_winter + high_vol + is_winter:high_vol",
        data=df
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    print("SELL IN MAY × VOLATILITY REGIME")
    print(model.summary())

    return df, model


def sell_in_may_fed_regime(daily_df, fed_parquet):
    df = prepare_monthly_with_fed(daily_df, fed_parquet)
    median_rate = df["FEDFUNDS"].median()
    df["high_rate"] = (df["FEDFUNDS"] > median_rate).astype(int)

    model = smf.ols(
        "Return ~ is_winter + high_rate + is_winter:high_rate",
        data=df
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    print("SELL IN MAY × FED RATE REGIME")
    print(model.summary())
    return df, model

def plot_vol_regime_point_ci(df_monthly):
    x = df_monthly.copy()
    x["Season"] = np.where(x["is_winter"] == 1, "Winter", "Summer")
    x["VolRegime"] = np.where(x["high_vol"] == 1, "High Vol", "Low Vol")

    plt.figure(figsize=(9, 5))
    sns.pointplot(
        data=x,
        x="VolRegime",
        y="Return",
        hue="Season",
        errorbar=("ci", 95),
        dodge=0.25
    )
    plt.axhline(0, linestyle="--", alpha=0.5)
    plt.title("Conditional Mean Monthly Returns (95% CI): Season × Volatility Regime")
    plt.ylabel("Monthly return")
    plt.xlabel("")
    plt.legend(title="")
    plt.tight_layout()
    plt.show()

def plot_spread_by_vol_regime(df_monthly):
    x = df_monthly.copy()
    x["Season"] = np.where(x["is_winter"] == 1, "Winter", "Summer")

    pivot = (x.groupby(["high_vol", "Season"])["Return"].mean()
               .unstack("Season"))

    spread = (pivot["Winter"] - pivot["Summer"]).rename("spread")

    out = pd.DataFrame({
        "VolRegime": ["Low Vol", "High Vol"],
        "WinterMinusSummer": [spread.loc[0], spread.loc[1]]
    })

    plt.figure(figsize=(7, 4))
    sns.barplot(data=out, x="VolRegime", y="WinterMinusSummer", errorbar=None)
    plt.axhline(0, linestyle="--", alpha=0.6)
    plt.title("Sell-in-May Spread by Volatility Regime")
    plt.ylabel("Winter − Summer (monthly return)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


def plot_month_year_heatmap(daily_df: pd.DataFrame):
    """
    I plotted Year x Month heatmap using equal-weight monthly market LOG returns.
    """
    df = daily_df.copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    m_ticker = df.groupby(["Ticker","Year","Month"], as_index=False)["Return"].sum()

    mkt = m_ticker.groupby(["Year","Month"], as_index=False)["Return"].mean()

    mat = mkt.pivot(index="Year", columns="Month", values="Return")
    mat = mat.replace({pd.NA: np.nan}).astype(float)

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        vmin=-0.10, vmax=0.10 
    )
    plt.title("Monthly Returns (Equal-Weight Market): Year × Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()

def plot_seasonal_distribution(daily_df):
    m = daily_df.copy()
    m["Season"] = np.where(m["Month"].between(5,10), "Summer", "Winter")
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Season", y="Return", data=m, inner="box", palette="Set2")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Return Distribution: Summer vs Winter (Violin Plot)")
    plt.ylabel("Daily Return")
    plt.show()

def sell_in_may_volume_regime(daily_df, fed_parquet, window=12):
    df = prepare_monthly_with_fed(daily_df, fed_parquet)

    df = df.sort_values(["Year", "Month"])
    df["vol"] = df["Volume"].rolling(window).mean() 

    median_vol = df["vol"].median()  
    df["high_vol"] = (df["vol"] > median_vol).astype(int)  

    df = df.dropna()

    model = smf.ols(
        "Return ~ is_winter + high_vol + is_winter:high_vol",
        data=df
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    print("SELL IN MAY × VOLUME REGIME")
    print(model.summary())

    return df, model

def plot_volume_regime_point_ci(df_monthly):
    x = df_monthly.copy()
    x["Season"] = np.where(x["is_winter"] == 1, "Winter", "Summer")
    x["VolRegime"] = np.where(x["high_vol"] == 1, "High Vol", "Low Vol")

    plt.figure(figsize=(9, 5))
    sns.pointplot(
        data=x,
        x="VolRegime",
        y="Return",
        hue="Season",
        errorbar=("ci", 95), 
        dodge=0.25
    )
    plt.axhline(0, linestyle="--", alpha=0.5)
    plt.title("Conditional Mean Monthly Returns (95% CI): Season × Volume Regime")
    plt.ylabel("Monthly return")
    plt.xlabel("")
    plt.legend(title="")
    plt.tight_layout()
    plt.show()

def annual_returns(daily_df: pd.DataFrame):
    df = daily_df.copy()
    annual_log_returns = df.groupby(["Ticker", "Year"])["Return"].sum().reset_index() 
    return annual_log_returns


def analyze_sell_in_may_effect(daily_df: pd.DataFrame, min_years=6):
    df = daily_df.copy()
    df["Season"] = np.where(df["Month"].between(5, 10), "Summer", "Winter")

    seasonal = (
        df.groupby(["Ticker", "Year", "Season"], as_index=False)["Return"]
          .mean()
          .rename(columns={"Return": "season_avg_return"})
    )
    wide = seasonal.pivot(index=["Ticker", "Year"], columns="Season", values="season_avg_return")

    wide = wide.dropna(subset=["Summer", "Winter"]).reset_index()

    if wide.empty:
        return pd.DataFrame(columns=["Company", "Affected Years (%)"])
    years_per_ticker = wide.groupby("Ticker")["Year"].nunique()

    eligible = years_per_ticker[years_per_ticker >= min_years].index
    wide = wide[wide["Ticker"].isin(eligible)]

    if wide.empty:
        return pd.DataFrame(columns=["Company", "Affected Years (%)"])

    wide["winter_beats_summer"] = (wide["Winter"] > wide["Summer"]).astype(int)

    result_df = (
        wide.groupby("Ticker")
            .agg(valid_years=("Year", "nunique"),
                 wins=("winter_beats_summer", "sum"))
            .reset_index()
    )
    result_df["Affected Years (%)"] = 100.0 * result_df["wins"] / result_df["valid_years"]
    result_df = result_df.rename(columns={"Ticker": "Company"})[["Company", "Affected Years (%)"]]

    if not result_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(result_df["Affected Years (%)"], bins=20, kde=True, color="skyblue")
        plt.axvline(x=50, linestyle="--", color="red", label="50% Threshold")
        plt.title("Sell in May Effect (Average Daily Returns)")
        plt.xlabel("% of Years Where Winter Avg > Summer Avg")
        plt.ylabel("Number of Companies")
        plt.legend()
        plt.show()

    return result_df



def analyze_sell_in_may_during_crisis(daily_df, crisis_periods):
    df = daily_df.copy()
    
    df["Season"] = np.where(df["Month"].between(5, 10), "Summer", "Winter")
    
    crisis_results = []

    for crisis_name, (start_date, end_date) in crisis_periods.items():
        start_date = pd.to_datetime(start_date)  
        end_date = pd.to_datetime(end_date)    
        
        mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        crisis_data = df.loc[mask]
        
        if crisis_data.empty:
            continue
            
        seasonal_avg = crisis_data.groupby(["Ticker", "Season"])["Return"].sum().reset_index()
        
        pivot_df = seasonal_avg.pivot(index="Ticker", columns="Season", values="Return")
        
        if "Summer" in pivot_df.columns and "Winter" in pivot_df.columns:
            pivot_df["Diff"] = pivot_df["Summer"] - pivot_df["Winter"]
            avg_diff = pivot_df["Diff"].mean()
            effect_ratio = (pivot_df["Summer"] < pivot_df["Winter"]).mean() * 100
            
            crisis_results.append({
                "Crisis": crisis_name,
                "Avg_Summer_Return": pivot_df["Summer"].mean(),
                "Avg_Winter_Return": pivot_df["Winter"].mean(),
                "Effect_Ratio (%)": effect_ratio
            })
            
    return pd.DataFrame(crisis_results)

def plot_sell_in_may_effects_across_crises(daily_df, crisis_periods):
    results_df = analyze_sell_in_may_during_crisis(daily_df, crisis_periods)

    plot_data = results_df.melt(id_vars="Crisis", value_vars=["Avg_Summer_Return", "Avg_Winter_Return"],
                                 var_name="Season", value_name="Return")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_data, x="Crisis", y="Return", hue="Season", 
                palette={"Avg_Summer_Return": "coral", "Avg_Winter_Return": "skyblue"})
    
    plt.title("Summer vs. Winter Returns During Crisis Periods (Sell in May Analysis)")
    plt.ylabel("Average  Return")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.legend(title="Season")
    plt.tight_layout()
    plt.show()

def compare_by_decades(daily_df: pd.DataFrame):

    m = daily_df.copy()
    m["Season"] = np.where(m["Month"].between(5, 10), "summer", "winter")
    m["SeasonYear"] = np.where(m["Month"].isin([11, 12]), m["Year"] + 1, m["Year"])

    m["Decade"] = (m["Year"] // 10) * 10
    
    g = m.groupby(["Ticker", "Decade", "Season"])["Return"].sum().reset_index()

    season_mean = (
        g.groupby(["Decade", "Season"])["Return"]
         .mean()
         .reset_index(name="mean_logret")
    )

    pivot = season_mean.pivot(
        index="Decade",
        columns="Season",
        values="mean_logret"
    ).dropna()
    
    diff_pct = (np.exp(pivot["winter"] - pivot["summer"]) - 1.0) * 100

    plt.figure(figsize=(10, 5))
    plt.bar(pivot.index.astype(int), diff_pct.values, width=6)
    plt.axhline(0, ls="--", lw=1, c="black")

    plt.title("Sell in May Effect: Winter − Summer (%) by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Winter − Summer (%)")
    plt.tight_layout()
    plt.show()

    return pivot


def plot_affected_years_pie_chart(result_df, holiday_name=""):
    temp_df = result_df.copy()

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
              "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]

    temp_df["Range"] = pd.cut(
        temp_df["Affected Years (%)"],
        bins=bins,
        labels=labels,
        right=False
    )

    pie_data = temp_df["Range"].value_counts().sort_index()
    pie_data = pie_data[pie_data > 0]

    if pie_data.empty:
        print("No data to plot.")
        return pie_data

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Spectral(np.linspace(0, 1, len(pie_data)))

    wedges, _ = ax.pie(
        pie_data.values,
        startangle=140,
        colors=colors,
        radius=1.0
    )

    total = pie_data.sum()
    legend_labels = [
        f"{idx}  ({val / total * 100:.1f}%)"
        for idx, val in pie_data.items()
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Affected Years Range",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    ax.set_title(
        f"Distribution of Companies by Affected Years (%)\nHoliday: {holiday_name}",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.show()

    return pie_data


def plot_companies_above_90(result_df):

    companies_above_90 = result_df[result_df["Affected Years (%)"] > 90]

    plt.figure(figsize=(10, 6))
    sns.histplot(companies_above_90["Affected Years (%)"], bins=10, kde=False, color="lightgreen")
    plt.title("Distribution of Companies with > 90% Affected Years")
    plt.xlabel("Affected Years (%)")
    plt.ylabel("Number of Companies")
    plt.tight_layout()
    plt.show()

    return companies_above_90

def analyze_low_and_high_impact(result_df):

    low_impact = result_df[result_df["Affected Years (%)"] < 50]
    high_impact = result_df[result_df["Affected Years (%)"] >= 50]

    print(f"Companies with <50% Affected Years: {len(low_impact)}")
    print(f"Companies with >=50% Affected Years: {len(high_impact)}")

    stat, p_val = mannwhitneyu(low_impact["Affected Years (%)"], high_impact["Affected Years (%)"])

    plt.figure(figsize=(10, 6))
    sns.histplot(low_impact["Affected Years (%)"], bins=10, kde=True, color="lightcoral", label="Low Impact (<50%)")
    sns.histplot(high_impact["Affected Years (%)"], bins=10, kde=True, color="lightseagreen", label="High Impact (>=50%)")
    plt.title("Comparison of Low vs High Impact Companies")
    plt.xlabel("Affected Years (%)")
    plt.ylabel("Number of Companies")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return stat, p_val

def plot_seasonality_for_ticker(daily_df, ticker, ret_col="Return"):
    ticker_data = daily_df[daily_df["Ticker"] == ticker].copy()

    # Season: May-Oct = Summer, Nov-Apr = Winter
    ticker_data["Season"] = np.where(ticker_data["Month"].between(5, 10), "Summer", "Winter")

    seasonal_returns = (
        ticker_data.groupby(["Year", "Season"])[ret_col]
        .mean()
        .unstack("Season")
    )

    # Full years only (both seasons exist)
    seasonal_returns = seasonal_returns.dropna(subset=["Summer", "Winter"])

    start_year = int(seasonal_returns.index.min())
    end_year = int(seasonal_returns.index.max())
    print(f"Analyzing {ticker} from {start_year} to {end_year} (Full years only)")

    plt.figure(figsize=(12, 6))

    plt.plot(
        seasonal_returns.index, seasonal_returns["Summer"],
        label="Summer avg daily return (May-Oct)", marker="o", color="orange"
    )
    plt.plot(
        seasonal_returns.index, seasonal_returns["Winter"],
        label="Winter avg daily return (Nov-Apr)", marker="o", linestyle="--", color="blue"
    )

    plt.fill_between(
        seasonal_returns.index,
        seasonal_returns["Winter"], seasonal_returns["Summer"],
        where=(seasonal_returns["Winter"] > seasonal_returns["Summer"]),
        color="green", alpha=0.1, label="Winter outperforms"
    )

    plt.title(f"Sell in May Effect (Average Daily Returns) for {ticker} ({start_year}-{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Average Daily Return")
    plt.legend(title="Season")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return seasonal_returns

def compare_by_exchange(daily_clean: pd.DataFrame, symbols_meta: pd.DataFrame, min_years: int = 6, plot: bool = True):

    base_cols = ["Ticker", "Year", "Month", "Return"]
    df = daily_clean[base_cols].merge(
        symbols_meta[["Ticker", "Listing Exchange"]],
        on="Ticker",
        how="inner"
    )

    # Season-Year + Season
    df["Season_Year"] = np.where(df["Month"] >= 11, df["Year"] + 1, df["Year"])
    df["Season"] = np.where(df["Month"].between(5, 10), "Summer", "Winter")

    # (Ticker, Season_Year, Season) toplam log return
    seasonal = (
        df.groupby(["Ticker", "Season_Year", "Season"], sort=False)["Return"]
          .sum()
          .unstack("Season")   # columns: Summer, Winter
    )

    seasonal = seasonal.dropna(subset=["Summer", "Winter"])

    if seasonal.empty:
        out = pd.DataFrame(columns=["Company", "Affected Years (%)", "Exchange"])
        if plot:
            print("No eligible data (no Summer/Winter pairs).")
        return out

    counts = seasonal.groupby(level=0).size()
    eligible = counts[counts >= min_years].index
    seasonal = seasonal.loc[eligible]

    if seasonal.empty:
        out = pd.DataFrame(columns=["Company", "Affected Years (%)", "Exchange"])
        if plot:
            print(f"No tickers with >= {min_years} years.")
        return out

    success = (seasonal["Winter"] > seasonal["Summer"]).astype(int)

    affected_pct = success.groupby(level=0).mean().mul(100)

    exch_map = df.drop_duplicates("Ticker").set_index("Ticker")["Listing Exchange"]

    result_df = (
        affected_pct.rename("Affected Years (%)")
        .to_frame()
        .assign(Company=lambda x: x.index)
        .assign(Exchange=lambda x: x["Company"].map(exch_map))
        .reset_index(drop=True)
        [["Company", "Affected Years (%)", "Exchange"]]
    )

    if plot and not result_df.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x="Exchange", y="Affected Years (%)", data=result_df)  
        ax.set_title("Sell in May Effect: Comparison by Exchange")
        ax.set_xlabel("Exchange")
        ax.set_ylabel("% of Years Where Winter > Summer")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return result_df