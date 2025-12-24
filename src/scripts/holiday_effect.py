import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np  
from scipy.stats import ttest_ind
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler # pour standardiser
import plotly.express as px

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
                df.loc[df['Date'] == day_before, 'holiday_period'] = f'before_{holiday_name}'
                df.loc[df['Date'] == day_after, 'holiday_period'] = f'after_{holiday_name}'
                
            except:
                # Skip if date doesn't exist (like Feb 30)
                continue
    
    return df


#avg return before vs normal days only
def plot_holiday_effect_simple(df_with_returns):
    """
    Bar chart comparing average returns on days before holidays vs normal days only.
    
    Shows only 2 bars:
    - Before holiday day
    - Normal days (excludes both before AND after holiday days)
    
    Parameters:
    - df_with_returns: DataFrame with Date and Return columns
    """
    # Mark holiday periods
    df = find_days_before_and_after_holidays(df_with_returns)
    
    # Separate returns by day type
    before_holiday_returns = df[df['holiday_period'].str.startswith('before_')]['Return']
    # Normal days only (exclude all before_ and after_ days)
    normal_returns = df[df['holiday_period'] == 'normal']['Return']
    
    # Calculate average returns
    avg_before_holiday = before_holiday_returns.mean() * 100
    avg_normal = normal_returns.mean() * 100
    
    # Prepare data for plotting
    day_types = ['Before Holiday', 'Normal Days']
    returns_values = [avg_before_holiday, avg_normal]
    colors = ['steelblue', 'lightgray']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(day_types, returns_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels and title
    ax.set_xlabel('Day Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Return: Before Holiday vs Normal Days', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_holiday_effect_by_decade_comparison(df_with_returns):
    """
    Grouped bar chart comparing holiday effect across decades.
    Also generates an interactive Plotly HTML file.
    """
    decades = [
        (1991, 2000, "1991–2000"),
        (2001, 2010, "2001–2010"),
        (2011, 2020, "2011–2020")
    ]
    
    colors = ['steelblue', 'darkorange', 'seagreen']
    bar_width = 0.22
    x = np.arange(2)  # Before Holiday, Normal Days

    plotly_data = []

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, ((start_year, end_year, label), color) in enumerate(zip(decades, colors)):
        df_filtered = df_with_returns.copy()
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
        df_filtered['Year'] = df_filtered['Date'].dt.year
        df_period = df_filtered[
            (df_filtered['Year'] >= start_year) &
            (df_filtered['Year'] <= end_year)
        ].copy()
        
        df = find_days_before_and_after_holidays(df_period)
        
        before_holiday_returns = df[df['holiday_period'].str.startswith('before_')]['Return']
        normal_returns = df[df['holiday_period'] == 'normal']['Return']
        
        avg_before = before_holiday_returns.mean() * 100
        avg_normal = normal_returns.mean() * 100
        
        values = [avg_before, avg_normal]

        # Matplotlib bars
        bars = ax.bar(
            x + (i - 1) * bar_width,
            values,
            width=bar_width,
            color=color,
            alpha=0.85,
            edgecolor='black',
            linewidth=1.2,
            label=label
        )
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}%",
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=9,
                fontweight='bold'
            )

        #  Data for Plotly
        plotly_data.append({"Decade": label, "Day Type": "Before Holiday", "Average Return (%)": avg_before})
        plotly_data.append({"Decade": label, "Day Type": "Normal Days", "Average Return (%)": avg_normal})

    # Matplotlib formatting
    ax.set_xticks(x)
    ax.set_xticklabels(['Before Holiday', 'Normal Days'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Holiday Effect by Decade\n(Before Holiday vs Normal Days)',
        fontsize=14,
        fontweight='bold'
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Decade')

    plt.tight_layout()
    plt.show()

    df_plotly = pd.DataFrame(plotly_data)

    fig_plotly = px.bar(
        df_plotly,
        x="Day Type",
        y="Average Return (%)",
        color="Decade",
        barmode="group",
        title="Holiday Effect by Decade (Before Holiday vs Normal Days)"
    )

    fig_plotly.update_xaxes(title="Day Type")
    fig_plotly.update_yaxes(title="Average Return (%)")
    fig_plotly.write_html(
        "plot_holiday_effect_by_decade.html",
        include_plotlyjs="cdn"
    )




def get_crisis_returns_by_day(df_with_returns, start_year, end_year):
    """
    Helper function: Calculate average returns for each relative day around holidays.
    
    Returns a dictionary with average returns for J-3 to J+3.
    """
    # Filter data by year range
    df = df_with_returns.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Get holidays
    holidays = get_us_holidays()
    
    # Dictionary to store returns by relative day
    relative_returns = {-3: [], -2: [], -1: [], 1: [], 2: [], 3: []}
    
    # For each holiday in each year
    for holiday_name, month, day in holidays:
        for year in df['Year'].unique():
            try:
                holiday_date = pd.Timestamp(year=year, month=month, day=day)
                
                # Find trading days before the holiday
                current_date = holiday_date - pd.Timedelta(days=1)
                days_before = []
                while len(days_before) < 3:
                    while current_date.weekday() >= 5:
                        current_date -= pd.Timedelta(days=1)
                    days_before.append(current_date)
                    current_date -= pd.Timedelta(days=1)
                
                # Find trading days after the holiday
                current_date = holiday_date + pd.Timedelta(days=1)
                days_after = []
                while len(days_after) < 3:
                    while current_date.weekday() >= 5:
                        current_date += pd.Timedelta(days=1)
                    days_after.append(current_date)
                    current_date += pd.Timedelta(days=1)
                
                # Get returns for days before
                for i, rel_day in enumerate([-1, -2, -3]):
                    ret = df.loc[df['Date'] == days_before[i], 'Return']
                    if len(ret) > 0:
                        relative_returns[rel_day].append(ret.values[0])
                
                # Get returns for days after
                for i, rel_day in enumerate([1, 2, 3]):
                    ret = df.loc[df['Date'] == days_after[i], 'Return']
                    if len(ret) > 0:
                        relative_returns[rel_day].append(ret.values[0])
                        
            except:
                continue
    
    # Calculate average returns
    avg_returns = {}
    for rel_day in [-3, -2, -1, 1, 2, 3]:
        if len(relative_returns[rel_day]) > 0:
            avg_returns[rel_day] = np.mean(relative_returns[rel_day]) * 100
        else:
            avg_returns[rel_day] = np.nan
    
    return avg_returns


def plot_crisis_holiday_heatmap(df_with_returns):
    """
    Heatmap showing holiday effect during 3 major financial crises.
    
    Crises analyzed:
    - 1987: Black Monday
    - 2000-2002: Dot-Com Bubble
    - 2008-2009: Global Financial Crisis
    
    Columns: J-3, J-2, J-1, J (Holiday), J+1, J+2, J+3
    Rows: Each crisis period
    Values: Average returns (%)
    
    Parameters:
    - df_with_returns: DataFrame with Date and Return columns
    """
    # Define crisis periods
    crises = {
        'Black Monday (1987)': (1987, 1987),
        'Dot-Com Bubble (2000-2002)': (2000, 2002),
        'Global Financial Crisis (2008-2009)': (2008, 2009)
    }
    
    # Column names for the heatmap
    columns = ['J-3', 'J-2', 'J-1', 'J (Holiday)', 'J+1', 'J+2', 'J+3']
    
    # Prepare data matrix
    data_matrix = []
    crisis_names = []
    
    for crisis_name, (start_year, end_year) in crises.items():
        # Get average returns for this crisis period
        avg_returns = get_crisis_returns_by_day(df_with_returns, start_year, end_year)
        
        # Build row: J-3, J-2, J-1, J (0 for holiday), J+1, J+2, J+3
        row = [
            avg_returns.get(-3, np.nan),
            avg_returns.get(-2, np.nan),
            avg_returns.get(-1, np.nan),
            0,  # Holiday - market closed
            avg_returns.get(1, np.nan),
            avg_returns.get(2, np.nan),
            avg_returns.get(3, np.nan)
        ]
        data_matrix.append(row)
        crisis_names.append(crisis_name)
    
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(data_matrix, index=crisis_names, columns=columns)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    sns.heatmap(df_heatmap, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                linewidths=1, linecolor='white', cbar_kws={'label': 'Average Return (%)'},
                annot_kws={'fontsize': 11, 'fontweight': 'bold'}, ax=ax)
    
    # Customize
    ax.set_title('Holiday Effect During Major Financial Crises\n(Average Returns Around Holidays)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Day Relative to Holiday', fontsize=12, fontweight='bold')
    ax.set_ylabel('Crisis Period', fontsize=12, fontweight='bold')
    
    # Highlight the holiday column
    ax.axvline(x=3, color='red', linewidth=3, linestyle='-')
    ax.axvline(x=4, color='red', linewidth=3, linestyle='-')
    
    plt.tight_layout()
    plt.show()
    
    for crisis_name, (start_year, end_year) in crises.items():
        avg_returns = get_crisis_returns_by_day(df_with_returns, start_year, end_year)
        pre_holiday_avg = np.nanmean([avg_returns.get(-3, np.nan), avg_returns.get(-2, np.nan), avg_returns.get(-1, np.nan)])
        post_holiday_avg = np.nanmean([avg_returns.get(1, np.nan), avg_returns.get(2, np.nan), avg_returns.get(3, np.nan)])
        
    return df_heatmap


def plot_preholiday_vs_normal_by_holiday(df_with_returns):
    """
    Bar chart comparing average returns on the day before each specific holiday vs normal days.
    
    Shows for each holiday (Christmas, New Year, Independence Day, etc.):
    - Average return on the day before that holiday
    - Average return on normal days (for comparison)
    
    Parameters:
    - df_with_returns: DataFrame with Date and Return columns
    """
    # Mark holiday periods
    df = find_days_before_and_after_holidays(df_with_returns)
    
    # Get list of holidays
    holidays = get_us_holidays()
    
    # Calculate normal day average (baseline)
    normal_returns = df[df['holiday_period'] == 'normal']['Return']
    avg_normal = normal_returns.mean() * 100
    
    # Prepare data for plotting
    holiday_names = []
    before_values = []
    
    for holiday_name, _, _ in holidays:
        # Get returns before this specific holiday
        before_returns = df[df['holiday_period'] == f'before_{holiday_name}']['Return']
        avg_before = before_returns.mean() * 100 if len(before_returns) > 0 else np.nan
        
        holiday_names.append(holiday_name)
        before_values.append(avg_before)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Set bar positions
    x = np.arange(len(holiday_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, before_values, width, label='Day Before Holiday', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, [avg_normal] * len(holiday_names), width, label='Normal Days', color='lightgray', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add labels and title
    ax.set_xlabel('Holiday', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Return: Day Before Each Holiday vs Normal Days', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(holiday_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    
    for i, holiday in enumerate(holiday_names):
        before = before_values[i]
        diff = before - avg_normal if not np.isnan(before) else np.nan
        print(f"{holiday:<20} {before:>10.4f}% {diff:>15.4f}%")
    
    # Return data as DataFrame
    results_df = pd.DataFrame({
        'Holiday': holiday_names,
        'Day Before Return (%)': before_values,
        'Normal Days Return (%)': [avg_normal] * len(holiday_names),
        'Difference (%)': [b - avg_normal for b in before_values]
    })
    
    return 


def plot_holiday_effect_exchange(df_with_returns):
    """
    Plot average returns on days before holidays vs normal days only, for different exchanges, in one graph, different colors. Precise the full exchange name in the legend.
    """
    # Exchange name mapping
    exchange_names = {
        'Q': 'NASDAQ',
        'N': 'NYSE',
        'A': 'NYSE American (ex-AMEX)',
        'P': 'ETFs (NYSE Arca)',
        'Z': 'BATS / Cboe'
    }
    plotly_data = []
    # Mark holiday periods
    df = find_days_before_and_after_holidays(df_with_returns)
    
    # Get list of exchanges
    exchanges = df['Exchange'].unique()
    
    # Prepare data for plotting
    day_types = ['Before Holiday', 'Normal Days']
    colors = plt.cm.get_cmap('tab10', len(exchanges))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot for each exchange
    for i, exchange in enumerate(exchanges):
        df_exchange = df[df['Exchange'] == exchange]
        
        before_holiday_returns = df_exchange[df_exchange['holiday_period'].str.startswith('before_')]['Return']
        normal_returns = df_exchange[df_exchange['holiday_period'] == 'normal']['Return']
        
        avg_before_holiday = before_holiday_returns.mean() * 100
        avg_normal = normal_returns.mean() * 100
        
        returns_values = [avg_before_holiday, avg_normal]
        plotly_data.append({
            "Exchange": exchange_names.get(exchange, exchange),
            "Difference (%)": avg_before_holiday - avg_normal
        })
        
        # Offset for bars
        offset = (i - len(exchanges)/2) * 0.15 + 0.075
        
        # Use full exchange name in legend
        exchange_label = exchange_names.get(exchange, exchange)
        bars = ax.bar(np.array([0, 1]) + offset, returns_values, width=0.15, color=colors(i), alpha=0.8, label=exchange_label)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)
            
    holiday_effect_exchange = pd.DataFrame(plotly_data)
    fig = px.bar(
        holiday_effect_exchange,
        x="Exchange",
        y="Difference (%)",
        title="Holiday Effect by Exchange (Before Holiday − Normal Days)"
    )
    fig.update_xaxes(title="Exchange", tickangle=-30)
    fig.update_yaxes(title="Average Return Difference (%)")
    fig.write_html("plot_holiday_effect_exchange.html", include_plotlyjs="cdn")
    # Add labels and title
    ax.set_xlabel('Day Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Return: Before Holiday vs Normal Days by Exchange', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(day_types)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.legend(title='Exchange')
    
    plt.tight_layout()
    plt.show()



def normality_check(df_with_returns):
    """
    Plot average returns distribution to check normality assumption for both samples (pre-holiday and normal days). y axis is probability density. Plot a smooth kernel density estimate (KDE) of pre-holiday returns and normal-day returns on the same figure, using probability density on the y-axis.
    """
    # Setup
    df = df_with_returns.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df_marked = find_days_before_and_after_holidays(df)
    
    # Separate groups
    pre_holiday_mask = df_marked['holiday_period'].astype(str).str.startswith('before_', na=False)
    pre_holiday_returns = df[pre_holiday_mask]['Return'].dropna()
    normal_returns = df[~pre_holiday_mask]['Return'].dropna()
    
    # Remove outliers (±50%)
    pre_holiday_returns = pre_holiday_returns[(pre_holiday_returns > -0.5) & (pre_holiday_returns < 0.5)]
    normal_returns = normal_returns[(normal_returns > -0.5) & (normal_returns < 0.5)]
    
    # Plot KDEs
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pre_holiday_returns, color='blue', label='Pre-Holiday Returns', fill=True, alpha=0.5)
    sns.kdeplot(normal_returns, color='orange', label='Normal Day Returns', fill=True, alpha=0.5)
    
    plt.title('KDE of Pre-Holiday Returns vs Normal Day Returns')
    plt.xlabel('Returns')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()
    


#hypothesis testing function
def hypothesis_test_pre_holiday_effect(df_with_returns):
    """
    One-tailed t-test: H1 - pre-holiday returns > normal day returns
    Returns and prints only the p-value.
    """
    
    # Setup
    df = df_with_returns.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df_marked = find_days_before_and_after_holidays(df)
    
    # Separate groups
    pre_holiday_mask = df_marked['holiday_period'].astype(str).str.startswith('before_', na=False)
    pre_holiday_returns = df[pre_holiday_mask]['Return'].dropna()
    normal_returns = df[~pre_holiday_mask]['Return'].dropna()
    
    # Remove outliers (±50%)
    pre_holiday_returns = pre_holiday_returns[(pre_holiday_returns > -0.5) & (pre_holiday_returns < 0.5)]
    normal_returns = normal_returns[(normal_returns > -0.5) & (normal_returns < 0.5)]
    
    # Welch's t-test
    t_stat, p_two_tailed = ttest_ind(pre_holiday_returns, normal_returns, equal_var=False)
    
    # One-tailed p-value (H1: pre-holiday > normal)
    p_value = p_two_tailed / 2 if t_stat > 0 else 1 - (p_two_tailed / 2)
    
    print("Hypothesis test")
    print(f"H₀: μ_return_pre-holiday = μ_return_normal")
    print(f"H₁: μ_return_pre-holiday > μ_return_normal (one-tailed)")
    print(f"\np-value: {p_value:.6f}")
    print(f"\nDecision ( α = 0.05):")
    if p_value < 0.05:
        print(f" We reject the null hypothesis and find statistically significant evidence that average stock returns on pre-holiday trading days are higher than returns on regular trading days (p = {p_value:.6f})")
    else:
        print(f" Fail to reject H₀ (p = {p_value:.6f})")    
    return p_value


def compute_volatility(df_with_returns):
    """
    create a copy of df_with_returns and compute the volatility (standard deviation) from the 'Return' column. Return df_with_return_volatility with an additional 'Volatility' column.
    """

    df = df_with_returns.copy()
    df['Volatility'] = df['Return'].rolling(window=21).std()  # 21 trading days ~ 1 month
    return df

def dataset_fedfunds_rate(df_with_returns, df_fedfunds):
    """
    Merge df_with_returns with df_fedfunds on Date column to include Fed Funds Rate data.
    Return merged DataFrame with an additional 'FEDFUNDS' column.
    """
    df_returns = df_with_returns.copy()
    df_returns['Date'] = pd.to_datetime(df_returns['Date'])
    df_fedfunds = df_fedfunds.copy()
    df_fedfunds['observation_date'] = pd.to_datetime(df_fedfunds['observation_date'])
    
    # Extract year and month for matching
    df_returns['Year'] = df_returns['Date'].dt.year
    df_returns['Month'] = df_returns['Date'].dt.month
    df_fedfunds['Year'] = df_fedfunds['observation_date'].dt.year
    df_fedfunds['Month'] = df_fedfunds['observation_date'].dt.month
    
    # Merge on Year and Month
    df_merged = pd.merge(df_returns, df_fedfunds[['Year', 'Month', 'FEDFUNDS']], on=['Year', 'Month'], how='left')
    
    # Remove temporary Year/Month columns if they weren't in original
    df_merged = df_merged.drop(columns=['Year', 'Month'])
    
    df_with_fedfunds = df_merged[df_merged['FEDFUNDS'].notna()]
    return df_merged

def holiday_effect_regression_extended(df_final):
    """
    OLS regression of Return on pre_holiday indicator, Volume, Volatility, and FEDFUNDS.
    Excludes rows where Volatility or other features are NaN.
    
    Parameters:
        df: DataFrame with columns Return, Volume, Volatility, FEDFUNDS (output of compute_volatility applied to dataset_fedfunds_rate)
    
    Returns:
        None (prints regression results)
    """

    df = df_final.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df = find_days_before_and_after_holidays(df)

    df["pre_holiday"] = (
        df["holiday_period"]
        .astype(str)
        .str.startswith("before_", na=False)
        .astype(int)
    )
    features = ["pre_holiday","Volume","Volatility","FEDFUNDS"]

    df_reg = df.dropna(subset=["Return"] + features).copy()
    df_reg.loc[:, features] = df_reg[features].astype("float32")
    scaler = StandardScaler()
    df_reg[["Volume", "Volatility", "FEDFUNDS"]] = scaler.fit_transform(
    df_reg[["Volume", "Volatility", "FEDFUNDS"]]
    )
   
    X = df_reg[features]
    X = sm.add_constant(X)
    y = df_reg["Return"]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    print(model.summary())

    return model


