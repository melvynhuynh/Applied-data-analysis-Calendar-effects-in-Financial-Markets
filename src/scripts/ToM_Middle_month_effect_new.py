
import calendar
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd

def assign_period_of_month(df, window_tom=4, window_mid=4):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["LastDay"] = df["Date"].dt.days_in_month

    # ToM
    k = window_tom // 2   # days before and after the month switch

    # Add ToM boolean column
    df["is_ToM"] = (
        (df["Day"] <= k) |
        (df["Day"] > df["LastDay"] - k)
    )

    # Middle month 
    mid_center = 15
    half = window_mid // 2
    mid_start = mid_center - (half - 1)
    mid_end = mid_center + half

    df["is_Middle"] = df["Day"].between(mid_start, mid_end)

    # Adding the period labels 

    df["period_of_month"] = "Rest"

    # Month names list for mapping
    month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}

    # Helper to convert month number to "JanFeb", "FebMar" etc
    def tom_label(year, month, day, lastday):

        if day > lastday - k: 
            m1 = month
            m2 = 1 if month == 12 else month + 1
        elif day <= k:         
            m1 = 12 if month == 1 else month - 1
            m2 = month
        else:
            return "Rest"

        return month_names[m1] + month_names[m2]


    df.loc[df["is_ToM"], "period_of_month"] = df.loc[df["is_ToM"]].apply(
        lambda row: tom_label(row["Year"], row["Month"], row["Day"], row["LastDay"]),
        axis=1
    )

    # Middle month overwrites Rest and ToM
    df.loc[df["is_Middle"], "period_of_month"] = "Middle"


    df = df.drop(columns=["Day", "LastDay"])

    return df

def add_general_period_column(df):
    df = df.copy()

    df["period_month_general"] = df["period_of_month"].apply(
        lambda x: "ToM" if (x not in ["Rest", "Middle"]) else x
    )

    return df

def descriptive_statistics_month_effect(df, period_col="period_month_general", return_col="Return"):




    desc_stats = df.groupby(period_col)[return_col].agg(
        Mean="mean",
        Std="std",
        Count="count"
    )
    
    print("\nDescriptive statistics by period:\n")
    print(desc_stats)

    order = ["ToM", "Middle", "Rest"]




    return desc_stats


def average_return_per_month(df):

    df = df.copy()


    #  Compute ToM average per month 


    month_abbr_to_num = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    tom_df = df[df["period_month_general"] == "ToM"].copy()


    tom_df["ToM_M2"] = tom_df["period_of_month"].str.slice(3, 6).map(month_abbr_to_num) # Map the month turn to numerics

    avg_tom = tom_df.groupby("ToM_M2")["Return"].mean()   # Compute the mean return

    #  Compute Middle month average per month 
    mid_df = df[df["period_month_general"] == "Middle"]
    avg_middle = mid_df.groupby("Month")["Return"].mean()

    #  Compute rest of the month average per month 
    rest_df = df[df["period_month_general"] == "Rest"]
    avg_rest = rest_df.groupby("Month")["Return"].mean()


    # Build final table with 12 months × 3 columns

    final = pd.DataFrame({
        "ToM": avg_tom,
        "Middle": avg_middle,
        "Rest": avg_rest,
    })


    final = final.reindex(range(1, 13))

    # Replace month numbers by names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    final.index = final.index.map(month_names)

    print("\nAverage Return per Month (ToM, Middle, Rest):\n")
    print(final)


    # Plot

    plt.figure(figsize=(12, 6))

    plt.plot(final.index, final["ToM"], label="ToM", marker="o")
    plt.plot(final.index, final["Middle"], label="Middle", marker="o")
    plt.plot(final.index, final["Rest"], label="Rest", marker="o")

    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Average Return")
    plt.title("Evolution of Monthly Mean Returns by Period Category")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return final




def month_effect_stat_test(
    df,
    return_col="Return",
    period_col="period_month_general",
    ticker_col="Ticker",
    alpha=0.05,
    cluster_by_ticker=True
):


    #  Create the dummy variables

    data = df.copy()

    X = data[["is_ToM", "is_Middle"]]
    X = sm.add_constant(X)
    y = data[return_col]

  
    #  Perform regression

    if cluster_by_ticker:
        ticker_codes = df[ticker_col].astype("category").cat.codes
        date_codes   = df["Date"].astype("category").cat.codes
        groups = np.column_stack((ticker_codes, date_codes))
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={'groups': groups}
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC3")


    # Print regression results and compute the p value 

    print("\n================ Regression Results ================\n")
    print(model.summary())


    beta_tom = model.params["is_ToM"]
    pval_tom = model.pvalues["is_ToM"]

    beta_mid = model.params["is_Middle"]
    pval_mid = model.pvalues["is_Middle"]

    print("\n================ Statistical Tests ================\n")

    print(f"Turn-of-the-Month (ToM):")
    print(f"  Coefficient (β_ToM) = {beta_tom:.6f}")
    print(f"  p-value = {pval_tom:.6f}")

    if (beta_tom > 0) and (pval_tom < alpha):
        print(f"  There IS a statistically significant higher {return_col} during Turn-of-the-Month.\n")
    else:
        print(f"  There is NOT a statistically significant higher {return_col} during Turn-of-the-Month.\n")

    print(f"Middle-of-the-Month:")
    print(f"  Coefficient (β_Middle) = {beta_mid:.6f}")
    print(f"  p-value = {pval_mid:.6f}")

    if (beta_mid > 0) and (pval_mid < alpha):
        print(f"  There IS a statistically significant higher {return_col} during Middle-of-the-Month.\n")
    else:
        print(f"  There is NOT a statistically significant higher {return_col} during Middle-of-the-Month.\n")


    return {
        "ToM_vs_Rest": {
            "beta": beta_tom,
            "p_value": pval_tom,
            "significant": (beta_tom > 0) and (pval_tom < alpha)
        },
        "Middle_vs_Rest": {
            "beta": beta_mid,
            "p_value": pval_mid,
            "significant": (beta_mid > 0) and (pval_mid < alpha)
        }
    }






def Month_effect_role_of_volume(
    df,
    return_col="Return",
    period_col="period_of_month_general",
    ticker_col="Ticker",
    date_col="Date"
):

    data = df.copy()



    y = data[return_col]


    ticker_codes = data[ticker_col].astype("category").cat.codes
    date_codes   = data[date_col].astype("category").cat.codes
    groups = np.column_stack((ticker_codes, date_codes))

 
    # TOTAL EFFECT REGRESSION (WITHOUT VOLUME)

    X_total = data[["is_ToM"]]
    X_total = sm.add_constant(X_total)

    model_total = sm.OLS(y, X_total).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups}
    )

    beta_total = model_total.params["is_ToM"]

    # DIRECT EFFECT REGRESSION (WITH VOLUME)

    X_direct = data[["is_ToM", "log_volume"]]
    X_direct = sm.add_constant(X_direct)

    model_direct = sm.OLS(y, X_direct).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups}
    )

    beta_direct = model_direct.params["is_ToM"]


    # Indirect Effect

    beta_indirect = beta_total - beta_direct
    mediation_ratio = beta_indirect / beta_total if beta_total != 0 else np.nan

    



    
    print("\n TOTAL EFFECT ( Return = f(Period) ) ")
    print(model_total.summary())
    print(f"\nβ_total = {beta_total:.6f}")

    print("\n  DIRECT EFFECT (Return = f(Period, Volume) ) ")
    print(model_direct.summary())
    print(f"\nβ_direct = {beta_direct:.6f}")

    print("\n INDIRECT EFFECT via Volume ")
    print(f"β_indirect = β_total - β_direct = {beta_indirect:.6f}")
    print(f"\nMediation Ratio = Indirect / Total = {mediation_ratio:.4f}")


    
    return {
        "beta_total": beta_total,
        "beta_direct": beta_direct,
        "beta_indirect": beta_indirect
    }




def Month_effect_full_regression(
    df,
    X_features,               
    y_col="Return",           
    ticker_col="Ticker",
    date_col="Date"
):

    data = df.copy()


    cols_needed = X_features + [y_col]
    data = data.dropna(subset=cols_needed)

    y = data[y_col]

    X = data[X_features]
    X = sm.add_constant(X)


    ticker_codes, _ = pd.factorize(data[ticker_col])
    date_codes, _   = pd.factorize(data[date_col])
    groups = np.column_stack((ticker_codes, date_codes))

    #Perform the regression

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups}
    )


    coef_dict = {}
    for name, value in model.params.items():
        coef_dict[name] = float(value)   

    return model, coef_dict



def Month_effect_per_ListExchange(df, X_features):




    
    exchanges = df["Listing Exchange"].dropna().unique()

    beta_results = {}

    for exch in exchanges:
        sub_df = df[df["Listing Exchange"] == exch]


        # Run regression
        model, coefs = Month_effect_full_regression(
            sub_df,
            X_features,
            y_col="Return",
            ticker_col="Ticker",
            date_col="Date"
        )

        # Store TOM coefficient
        beta_results[exch] = coefs.get("is_ToM", np.nan)

    
    # Plot histogram of beta_TOM
  
    plt.figure(figsize=(10, 5))
    plt.bar(beta_results.keys(), beta_results.values())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("β_ToM")
    plt.title("Turn-of-the-Month Effect per Listing Exchange")
    plt.tight_layout()
    plt.show()

    return beta_results






def Month_effect_per_TurnedMonth(df, X_features, y_col="Return"):

    data = df.copy()

    # Extract turned month label for TOM rows

    tom_labels = [
        lbl for lbl in data["period_of_month"].unique()
        if isinstance(lbl, str) and lbl not in ["Rest", "Middle"]
    ]


    calendar_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def sort_key(label):
        # Take the second part of each TOM (e.g takes "Feb" from "JanFeb")
        second = label[3:6]
        return calendar_order.index(second)

    tom_labels = sorted(tom_labels, key=sort_key)


    # Run regression for each turned month 

    beta_results = {}

    for lbl in tom_labels:

        sub_df = data[
            (data["period_of_month"] == lbl) | (data["period_month_general"] != "ToM")
        ]

        model, coefs = Month_effect_full_regression(
            sub_df,
            X_features=X_features,
            y_col=y_col,
            ticker_col="Ticker",
            date_col="Date"
        )

        beta_results[lbl] = coefs.get("is_ToM", np.nan)


    # Plot histogram 

    plt.figure(figsize=(12, 5))
    plt.bar(beta_results.keys(), beta_results.values())
    plt.xticks(rotation=45)
    plt.ylabel("β_TOM")
    plt.xlabel("Turn of the Month Transition")
    plt.title("Turn-of-the-Month Effect per Transition (JanFeb, FebMar, ...)")
    plt.tight_layout()
    plt.show()

    return beta_results



def Month_effect_per_year_window(df, X_features, year_window, y_col="Return"):

  

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year

    min_year = int(data["Year"].min())
    max_year = int(data["Year"].max())


    # Create the years windows 

    windows = [(y, y+year_window) for y in range(min_year, max_year, year_window)]

    beta_results = {}
    labels = []


    # Run regression for each window 

    for (y0, y1) in windows:
        sub_df = data[(data["Year"] >= y0) & (data["Year"] < y1)]

        if len(sub_df) < 50000:   # Don't perform regression if the data is too small
            continue


        model, coefs = Month_effect_full_regression(
            sub_df,
            X_features=X_features,
            y_col=y_col,
            ticker_col="Ticker",
            date_col="Date"
        )

        beta_tom = coefs.get("is_ToM", np.nan)

        label = f"{y0}-{y1}"
        labels.append(label)
        beta_results[label] = beta_tom

    # Plot histogram

    plt.figure(figsize=(12, 5))
    plt.bar(labels, [beta_results[l] for l in labels])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("β_TOM")
    plt.xlabel("Year window")
    plt.title("Turn-of-the-Month Effect over Years")
    plt.tight_layout()
    plt.show()

    return beta_results

