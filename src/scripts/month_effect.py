import pandas as pd
import matplotlib.pyplot as plt
import calendar
import random
import numpy as np


def random_volume_return_each_month(df):
    
    df["Date"] = pd.to_datetime(df["Date"])

    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    month_names = list(calendar.month_name)[1:]  # From January to december

    plt.figure(figsize=(16, 40))
    plot_index = 1

    for month_idx, month_name in enumerate(month_names, start=1):

        # Filter dataframe for this month
        month_df = df[df["Month"] == month_idx]

        if month_df.empty:
            continue

        # Random stock chosen 
        tickers_available = month_df["Ticker"].unique()
        chosen_ticker = random.choice(tickers_available)

        ticker_month_df = month_df[month_df["Ticker"] == chosen_ticker]

        # We group the data by year and month and we choose a random month 
        valid_groups = [
            group for _, group in ticker_month_df.groupby(["Year", "Month"])
            if len(group) >= 15
        ]

        if not valid_groups:
            continue  

        chosen_month_df = random.choice(valid_groups).sort_values("Date").copy()

        # Create day index within the month
        chosen_month_df["DayIndex"] = range(1, len(chosen_month_df) + 1)

        # Plot of the volume
        plt.subplot(12, 2, plot_index)
        plt.plot(chosen_month_df["DayIndex"], chosen_month_df["Volume"])
        plt.title(f"{month_name}: Volume ({chosen_ticker})")
        plt.xlabel("Day of Month")
        plt.ylabel("Volume")

        # Plot of the return 
        plt.subplot(12, 2, plot_index + 1)
        plt.plot(chosen_month_df["DayIndex"], chosen_month_df["Return"])
        plt.title(f"{month_name}: Return ({chosen_ticker})")
        plt.xlabel("Day of Month")
        plt.ylabel("Return")

        plot_index += 2

    plt.tight_layout()
    plt.show()


def avg_volume_within_months(df, y1, y2):

    
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter the data frame in order to keep the the data between year y1 and year y2 
    df = df[(df["Date"].dt.year >= y1) & (df["Date"].dt.year <= y2)]

    if df.empty:
        print(f"No data found between years {y1} and {y2}.")
        return None

    # Normalize the volume by dividing by the maximum value reached so that we can compare volums of different stocks
    df["Volume_norm"] = df["Volume"] / df.groupby("Ticker")["Volume"].transform("max")

    
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Volume_norm"] = df["Volume_norm"].astype("float64")

    # Plot
    plt.figure(figsize=(15, 18))
    month_names = list(calendar.month_name)[1:]  # From January to December
    for month_num in range(1, 13):

        filtered_by_month = df[df["Month"] == month_num][["Volume_norm", "Day"]]

        if filtered_by_month.empty:
            continue

        filtered_by_days = filtered_by_month.groupby("Day")["Volume_norm"].mean() # Compute the mean across all years of the interval and across all stocks

        plt.subplot(4, 3, month_num)
        plt.plot(filtered_by_days.index, filtered_by_days.values, linewidth=2)

        plt.title(f"{month_names[month_num - 1]} Avg Volume ({y1}-{y2})")
        plt.xlabel("Day of Month")
        plt.ylabel("Normalized Volume")
        plt.grid(True)

    plt.tight_layout()
    plt.show()



def std_volume_within_months(df, y1, y2):

    
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter the data frame in order to keep the the data between year y1 and year y2 
    df = df[(df["Date"].dt.year >= y1) & (df["Date"].dt.year <= y2)]

    if df.empty:
        print(f"No data found between years {y1} and {y2}.")
        return None

    # Normalize the volume by dividing by the maximum value reached so that we can compare volums of different stocks
    df["Volume_norm"] = df["Volume"] / df.groupby("Ticker")["Volume"].transform("max")

    
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Volume_norm"] = df["Volume_norm"].astype("float64")

    # Plot
    plt.figure(figsize=(15, 18))
    month_names = list(calendar.month_name)[1:]  

    for month_num in range(1, 13):

        filtered_by_month = df[df["Month"] == month_num][["Volume_norm", "Day"]]

        if filtered_by_month.empty:
            continue

        
        filtered_by_days = filtered_by_month.groupby("Day")["Volume_norm"].std() # Compute the std across all years in [y1,y2] across all stocks

        plt.subplot(4, 3, month_num)
        plt.plot(filtered_by_days.index, filtered_by_days.values, linewidth=2)

        plt.title(f"{month_names[month_num - 1]} Std Volume ({y1}-{y2})")
        plt.xlabel("Day of Month")
        plt.ylabel("Std (Normalized Volume)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return filtered_by_days  




def avg_logreturn_within_months(df, y1, y2):

    
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter the data frame in order to keep the the data between year y1 and year y2 
    df = df[(df["Date"].dt.year >= y1) & (df["Date"].dt.year <= y2)]

    if df.empty:
        print(f"No data found between years {y1} and {y2}.")
        return None

    # We compute the log return here 
    df["LogReturn"] = np.log(1 + df["Return"])

    
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["LogReturn"] = df["LogReturn"].astype("float64")

    # Plot
    plt.figure(figsize=(15, 18))
    month_names = list(calendar.month_name)[1:]  

    for month_num in range(1, 13):

        filtered_by_month = df[df["Month"] == month_num][["LogReturn", "Day"]]

        if filtered_by_month.empty:
            continue

        
        filtered_by_days = filtered_by_month.groupby("Day")["LogReturn"].mean() # We compute the mean across the years of interval and across stocks 

        plt.subplot(4, 3, month_num)
        plt.plot(filtered_by_days.index, filtered_by_days.values, linewidth=2)

        plt.title(f"{month_names[month_num - 1]} Avg Log Return ({y1}-{y2})")
        plt.xlabel("Day of Month")
        plt.ylabel("Log Return")
        plt.grid(True)

    plt.tight_layout()
    plt.show()




def std_logreturn_within_months(df, y1, y2):

    
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter the data frame in order to keep the the data between year y1 and year y2 
    df = df[(df["Date"].dt.year >= y1) & (df["Date"].dt.year <= y2)]

    if df.empty:
        print(f"No data found between years {y1} and {y2}.")
        return None

    # We compute the log return here 
    df["LogReturn"] = np.log(1 + df["Return"])

    
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["LogReturn"] = df["LogReturn"].astype("float64")

    # Plot
    plt.figure(figsize=(15, 18))
    month_names = list(calendar.month_name)[1:]  

    for month_num in range(1, 13):

        filtered_by_month = df[df["Month"] == month_num][["LogReturn", "Day"]]

        if filtered_by_month.empty:
            continue

        
        filtered_by_days = filtered_by_month.groupby("Day")["LogReturn"].std() # computed std of the return across years of the interval and across stocks

        plt.subplot(4, 3, month_num)
        plt.plot(filtered_by_days.index, filtered_by_days.values, linewidth=2)

        plt.title(f"{month_names[month_num - 1]} Std Log Return ({y1}-{y2})")
        plt.xlabel("Day of Month")
        plt.ylabel("Std(Log Return)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
