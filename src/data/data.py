import pandas as pd
import os

EXCHANGE_NAMES = {
    'Q': 'NASDAQ Global (Large)',
    'N': 'NYSE (Large)',
    'A': 'NYSE American (Small)',
    'P': 'NYSE Arca',
    'Z': 'BATS'
}

DATA_FOLDERS = ['data/etfs_parquet', 'data/stocks_parquet']

def get_summer_winter_returns(df):
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
    return summer_returns, winter_returns
    
def load_all_stocks(directory='data/stocks_parquet'):
    all_data = []
    for file in os.listdir(directory):
        if file.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(directory, file))
            df['Date'] = pd.to_datetime(df['Date'])
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def load_metadata(directory='data'):
    file = "symbols_valid_meta.parquet"
    metadata = pd.read_parquet(os.path.join(directory, file))
    return metadata


def load_fed_funds(directory='data'):
    file = "FEDFUNDS.parquet"
    fed = pd.read_parquet(os.path.join(directory, file))
    fed = fed.rename(columns={
        "observation_date": "Date",
        "FEDFUNDS": "fed_funds"
    })
    fed["Date"] = pd.to_datetime(fed["Date"])

    return fed


def load_cpi_inflation(directory='data'):
    file = "FPCPITOTLZGUSA.parquet"
    cpi = pd.read_parquet(os.path.join(directory, file))
    cpi = cpi.rename(columns={
        "observation_date": "Date",
        "FPCPITOTLZGUSA": "inflation"
    })
    cpi["Date"] = pd.to_datetime(cpi["Date"])
    cpi["Year"] = cpi["Date"].dt.year
    return cpi[["Date", "Year", "inflation"]]


def prepare_data_for_analysis(df, metadata):
    """
    Clean dataset for analysis and add volatility
    """
    metadata = metadata.rename(columns={
        'Symbol': 'Ticker',
        'Listing Exchange': 'Exchange'
    })

    df = df.copy()
    metadata_exchange = metadata[['Ticker', 'Exchange']].copy()
    metadata_exchange = metadata_exchange.drop_duplicates(subset=['Ticker'])
    df_with_return = df.merge(metadata_exchange, on='Ticker', how='left')

    df_with_return['Month'] = df_with_return['Date'].dt.month
    df_with_return['Year'] = df_with_return['Date'].dt.year
    df_with_return = df_with_return.sort_values(['Ticker', 'Date'])

    df_with_return['Return'] = df_with_return.groupby(
        'Ticker')['Adj Close'].pct_change(fill_method=None)
    df_with_return = df_with_return.sort_values(["Ticker", "Date"])

    # adding volatility column
    df_with_return["volatility"] = (df_with_return.groupby("Ticker")["Return"].rolling(
        20).std().reset_index(level=0, drop=True))

    # add is_january flag
    df_with_return["is_january"] = df_with_return["Date"].dt.month == 1

    df_with_return = df_with_return.dropna(
        subset=['Return', 'Exchange', 'volatility'])

    df_with_return = df_with_return[(
        df_with_return['Return'] > -0.5) & (df_with_return['Return'] < 0.5)]

    return df_with_return


def add_interest_rate(df, fed):
    df = df.copy()
    fed_f = fed.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    fed_f["Date"] = pd.to_datetime(fed_f["Date"])

    df["YearMonth"] = df["Date"].dt.to_period("M")
    fed_f["YearMonth"] = fed_f["Date"].dt.to_period("M")
    fed_f = fed_f[["YearMonth", "fed_funds"]]
    df = df.merge(fed_f, on="YearMonth", how="left")
    df.drop(columns=["YearMonth"], inplace=True)

    return df


def compute_return(parquet_file, folder):
    """
    dataset with return added
    """
    data = pd.read_parquet(os.path.join(folder, parquet_file))
    
    data['Return'] = data['Adj Close'].pct_change(fill_method=None)   
    return data

def compute_return_all(parquet_files=DATA_FOLDERS):
    """Retourne une liste de DataFrames, un par fichier parquet"""
    all_data_with_return = []
    for folder in parquet_files:
        for file in os.listdir(folder):
            df = compute_return(file, folder)  
            all_data_with_return.append(df)  
    
    return all_data_with_return


def compute_log_returns(df: pd.DataFrame):
    """
    I calculated the daily log returns for each stock.
    I also added year and month info for later seasonal analysis.
    """
    d = df.sort_values(["Ticker","Date"]).copy() # I sorted by ticker and date to align prices
    for c in ["Adj Close","Open","Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")  # I ensured numeric values only
    # I filtered out rows with invalid or zero prices
    d = d[(d["Adj Close"]>0) & (d["Open"]>0) & (d["Close"]>0)]
    # I found the previous day's adjusted close for each ticker
    prev = d.groupby("Ticker")["Adj Close"].shift(1)
    # I computed the continuous compounding log return
    d["logret_cc"] = np.log(d["Adj Close"] / prev)
    # I extracted time-based features
    d["Year"]  = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    return d.dropna(subset=["logret_cc"])

def label_season_with_pivot(d: pd.DataFrame, pivot_month: int) -> pd.DataFrame:
    df = d.copy()
    """
    I labeled each record as 'summer' or 'winter' based on a chosen pivot month.
    I also created a 'CycleYear' column to align seasons that cross over calendar years.
    if pivot_month = 5 (May), then 'summer' = May–Oct, 'winter' = Nov – Apr.
    """
    # I calculated how far each record's month is from the pivot (values from 0 to 11)
    k = (df["Month"] - pivot_month) % 12
    # I labeled months within 6 months after the pivot as "summer", others as "winter"
    df["Season"] = np.where(k <= 5, "summer", "winter")
    # I shifted the year for months before the pivot
    df["CycleYear"] = np.where(df["Month"] < pivot_month, df["Year"] + 1, df["Year"])
    return df

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
        lambda row: tom_label(row["Year"], row["Month"],
                              row["Day"], row["LastDay"]),
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


def add_santa_claus_rally_flag(df):
    """ 
    Here we add a flag to the days that correspond to the effect
    The rally is defined as:
    - The last 5 trading days of December
    - The first 2 trading days of January
    """

    df = df.copy()
    df = df.sort_values(['Ticker', 'Date'])

    df['DayOfMonthRank'] = df.groupby(['Ticker', 'Year', 'Month'])[
        'Date'].rank(method='first')
    df['DayOfMonthRankEnd'] = df.groupby(['Ticker', 'Year', 'Month'])[
        'Date'].rank(method='first', ascending=False)

    is_end_dec = (df['Month'] == 12) & (df['DayOfMonthRankEnd'] <= 5)
    is_start_jan = (df['Month'] == 1) & (df['DayOfMonthRank'] <= 2)

    df['Is_SCR'] = is_end_dec | is_start_jan

    df = df.drop(columns=['DayOfMonthRank', 'DayOfMonthRankEnd'])

    return df


def get_us_holidays():
    """
    Returns a list of US federal holidays.
    Simple function that creates holiday dates for analysis.
    """
    holidays = [
        ('Christmas', 12, 25),
        ('New Year', 1, 1),
        ('Independence Day', 7, 4),
        ('Thanksgiving', 11, 28),  # Approximate - 4th Thursday of November
        ('Memorial Day', 5, 27),   # Approximate - last Monday of May
        ('Labor Day', 9, 2),       # Approximate - 1st Monday of September
    ]
    return holidays


def find_days_before_and_after_holidays(df_with_returns):
    """
    Marks days as 'before holiday', 'after holiday', or 'normal'.

    Parameters:
    - df_with_returns: DataFrame with Date column and return data

    Returns:
    - DataFrame with an additional 'holiday_period' column
    """
    # Make a copy to avoid modifying original
    df = df_with_returns.copy()

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Add columns for month and day
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    # Initialize holiday_period column
    df['holiday_period'] = 'normal'

    # Get holidays
    holidays = get_us_holidays()

    # For each holiday, mark the day before and after
    for holiday_name, month, day in holidays:
        # Create a date for this holiday in each year
        for year in df['Date'].dt.year.unique():
            try:
                holiday_date = pd.Timestamp(year=year, month=month, day=day)

                # Find the day before (previous trading day)
                day_before = holiday_date - pd.Timedelta(days=1)
                # If weekend, go back further
                while day_before.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    day_before -= pd.Timedelta(days=1)

                # Find the day after (next trading day)
                day_after = holiday_date + pd.Timedelta(days=1)
                # If weekend, go forward
                while day_after.weekday() >= 5:
                    day_after += pd.Timedelta(days=1)

                # Mark these days
                df.loc[df['Date'] == day_before,
                       'holiday_period'] = f'before_{holiday_name}'
                df.loc[df['Date'] == day_after,
                       'holiday_period'] = f'after_{holiday_name}'

            except:
                # Skip if date doesn't exist (like Feb 30)
                continue

    return df
