import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from ..data.data import EXCHANGE_NAMES


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


def analyze_santa_claus_rally(df):
    """
    Performs a t-test to see if the Santa Claus Rally effect is significant.
    """

    rally_returns = df[df['Is_SCR'] == True]['Return']
    other_returns = df[df['Is_SCR'] == False]['Return']

    # print(f"Total 'Rally' observations: {len(rally_returns):,}")
    # print(f"Total 'Other' observations: {len(other_returns):,}")

    rally_mean = rally_returns.mean() * 100
    other_mean = other_returns.mean() * 100
    difference = rally_mean - other_mean

    t_stat, p_value = stats.ttest_ind(
        rally_returns, other_returns, equal_var=False)

    print("\n" + "="*50)
    print("SANTA CLAUS RALLY (SCR) SIGNIFICANCE TEST")
    print("="*50)
    print(f"  Avg. Daily Return (SCR Period):   {rally_mean:>8.4f}%")
    print(f"  Avg. Daily Return (Other Days):  {other_mean:>9.4f}%")
    print(f"  Difference (Difference):            {difference:>6.4f}%")
    print("-"*50)
    print(f"  T-Statistic:                     {t_stat:>9.3f}")
    print(f"  P-Value:                         {p_value:>10.5f}")
    print("="*50)

    return


def analyze_santa_claus_rally_by_decade(df):
    """
    Performs a t-test on the Santa Claus Rally for each decade.
    """

    df = df.copy()
    df['Decade'] = (df['Year'] // 10) * 10

    results = []

    for decade in sorted(df['Decade'].unique()):
        decade_data = df[df['Decade'] == decade]

        rally_returns = decade_data[decade_data['Is_SCR'] == True]['Return']
        other_returns = decade_data[decade_data['Is_SCR'] == False]['Return']

        if len(rally_returns) > 30 and len(other_returns) > 30:

            rally_mean = rally_returns.mean() * 100
            other_mean = other_returns.mean() * 100
            difference = rally_mean - other_mean

            t_stat, p_value = stats.ttest_ind(
                rally_returns, other_returns, equal_var=False)

            results.append({
                'Decade': f"{decade}s",
                'SCR_Mean': rally_mean,
                'Other_Mean': other_mean,
                'Difference': difference,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Significant': p_value < 0.05 and difference > 0
            })

    results_df = pd.DataFrame(results)

    print("\n" + "="*75)
    print("SANTA CLAUS RALLY (SCR) SIGNIFICANCE TEST BY DECADE")
    print("="*75)
    print(f"{'Decade':<10} {'SCR Mean':>10} {'Other Mean':>10} {'Difference':>10} {'T-Stat':>10} {'P-Value':>10}")
    print("-"*75)

    if results_df.empty:
        print("No decades with sufficient data to analyze.")
        print("="*75)
        return

    for _, row in results_df.iterrows():
        print(f"{row['Decade']:<10} {row['SCR_Mean']:>9.3f}% {row['Other_Mean']:>9.3f}% "
              f"{row['Difference']:>9.3f}% {row['T_Statistic']:>9.3f} {row['P_Value']:>9.4f}")

    print("="*75)

    return results_df


def graph_scr_by_decade(results_df):
    """
    Create a bar chart comparing the SCR Difference by decade.
    """

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['green' if sig else 'lightgray' for sig in results_df['Significant']]

    bars = ax.bar(results_df['Decade'], results_df['Difference'],
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, diff in zip(bars, results_df['Difference']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:.3f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax.set_ylabel(
        'SCR Difference (%) [Rally - Other Days]', fontweight='bold', fontsize=12)
    ax.set_xlabel('Decade', fontweight='bold', fontsize=12)
    ax.set_title('Santa Claus Rally Difference by Decade',
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def analyze_santa_claus_rally_by_exchange(df):
    """
    Performs a t-test on the Santa Claus Rally for each exchange.
    """

    df = df.copy()

    results = []

    for exchange in sorted(df['Exchange'].unique()):
        exchange_data = df[df['Exchange'] == exchange]

        rally_returns = exchange_data[exchange_data['Is_SCR']
                                      == True]['Return']
        other_returns = exchange_data[exchange_data['Is_SCR']
                                      == False]['Return']

        if len(rally_returns) > 30 and len(other_returns) > 30:

            rally_mean = rally_returns.mean() * 100
            other_mean = other_returns.mean() * 100
            difference = rally_mean - other_mean

            t_stat, p_value = stats.ttest_ind(
                rally_returns, other_returns, equal_var=False)

            name = EXCHANGE_NAMES.get(exchange, f"Exchange {exchange}")

            results.append({
                'Exchange': name,  # Use the full name
                'SCR_Mean': rally_mean,
                'Other_Mean': other_mean,
                'Premium': difference,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Significant': p_value < 0.05 and difference > 0
            })

    results_df = pd.DataFrame(results)

    print("\n" + "="*90)
    print("SANTA CLAUS RALLY (SCR) SIGNIFICANCE TEST BY EXCHANGE")
    print("="*90)
    print(f"{'Exchange':<25} {'SCR Mean':>10} {'Other Mean':>10} {'Premium':>10} {'T-Stat':>10} {'P-Value':>10}")
    print("-"*90)

    if results_df.empty:
        print("No exchanges with sufficient data to analyze.")
        print("="*90)
        return

    for _, row in results_df.iterrows():
        print(f"{row['Exchange']:<25} {row['SCR_Mean']:>9.3f}% {row['Other_Mean']:>9.3f}% "
              f"{row['Premium']:>9.3f}% {row['T_Statistic']:>9.3f} {row['P_Value']:>9.4f}")

    print("="*90)

    return results_df


def graph_scr_by_exchange(results_df):
    """
    Create a bar chart comparing the SCR premium by exchange.
    """

    if results_df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['green' if sig else 'lightgray' for sig in results_df['Significant']]

    bars = ax.bar(results_df['Exchange'], results_df['Premium'],
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, diff in zip(bars, results_df['Premium']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:.3f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax.set_ylabel(
        'SCR Premium (%) [Rally - Other Days]', fontweight='bold', fontsize=12)
    ax.set_xlabel('Exchange', fontweight='bold', fontsize=12)
    ax.set_title('Santa Claus Rally Premium by Exchange',
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
