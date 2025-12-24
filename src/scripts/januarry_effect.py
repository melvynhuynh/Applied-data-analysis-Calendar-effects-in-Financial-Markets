import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ..data.data import EXCHANGE_NAMES


def january_effect_significance_by_exchange(df):
    """
    Test if January effect is statistically significant for each exchange using T-test and P-value.
    """
    print("\n" + "="*90)
    print("STATISTICAL SIGNIFICANCE TEST - JANUARY EFFECT BY EXCHANGE")
    print("="*90)
    print(f"{'Exchange':<25} {'Jan Mean':>10} {'Other Mean':>12} {'Diff':>10} {'T-Stat':>10} {'P-Value':>10}")
    print("-"*90)

    results = []

    for exchange in sorted(df['Exchange'].unique()):
        exchange_data = df[df['Exchange'] == exchange]

        jan_returns = exchange_data[exchange_data['Month'] == 1]['Return']*100
        other_returns = exchange_data[exchange_data['Month']
                                      != 1]['Return']*100

        if len(jan_returns) > 30 and len(other_returns) > 30:
            t_stat, p_value = stats.ttest_ind(
                jan_returns, other_returns, equal_var=False)

            jan_mean = jan_returns.mean()
            other_mean = other_returns.mean()
            diff = jan_mean - other_mean

            name = EXCHANGE_NAMES.get(exchange, exchange)

            jan_se = jan_returns.sem()
            other_se = other_returns.sem()

            print(f"{name:<25} {jan_mean:>9.3f}% {other_mean:>9.3f}% {diff:>11.3f}% "
                  f"{t_stat:>10.3f} {p_value:>10.4f}")

            results.append({
                'Exchange': name,
                'January_Mean': jan_mean,
                'Other_Mean': other_mean,
                'January_SE': jan_se,
                'Other_SE': other_se,
                'Difference': diff,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'N_January': len(jan_returns),
                'N_Other': len(other_returns)
            })

    # print("="*90)

    # print("\nNumber of companies by exchange:")
    # exchange_counts = df.groupby('Exchange')['Ticker'].nunique()

    # for exchange, count in exchange_counts.items():
    #     name = EXCHANGE_NAMES.get(exchange, f'Exchange {exchange}')
    #     print(f"  {name:<20}: {count:>5,} companies")

    # print("\n" + "="*90)

    return pd.DataFrame(results)


def graph_january_effect_by_exchange(jan_effect_results):
    """
    Create a bar chart showing January vs Other months by exchange
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(jan_effect_results))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, jan_effect_results['January_Mean'], width,
                   label='January', color='pink', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    bars2 = ax.bar(x + width/2, jan_effect_results['Other_Mean'], width,
                   label='Other Months', color='blue', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ci_mult = 1.96

    # Add error bars (95% CI)
    ax.errorbar(x - width/2, jan_effect_results['January_Mean'],
                yerr=ci_mult * jan_effect_results['January_SE'],
                fmt='none', color='black', capsize=5, capthick=2, alpha=0.7)

    ax.errorbar(x + width/2, jan_effect_results['Other_Mean'],
                yerr=ci_mult * jan_effect_results['Other_SE'],
                fmt='none', color='black', capsize=5, capthick=2, alpha=0.7)

    ax.set_ylabel('Average Monthly Return (%) (95% Confidence Interval)',
                  fontsize=13, fontweight='bold')
    ax.set_xlabel('Exchange', fontsize=13, fontweight='bold')
    ax.set_title('January Effect by Exchange: January vs. Other Months',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(jan_effect_results['Exchange'], fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

    return


def calculate_january_effect_by_decade(df):
    """
    Compare January effect across different decades.
    """
    df = df.copy()
    df['Decade'] = (df['Year'] // 10) * 10

    results = []

    for decade in sorted(df['Decade'].unique()):
        decade_data = df[df['Decade'] == decade]

        jan_returns = decade_data[decade_data['Month'] == 1]['Return'] * 100
        other_returns = decade_data[decade_data['Month'] != 1]['Return'] * 100

        if len(jan_returns) > 30 and len(other_returns) > 30:
            t_stat, p_value = stats.ttest_ind(
                jan_returns, other_returns, equal_var=False)

            years_in_decade = decade_data['Year'].nunique()

            # Win rate: how often January returns are greater than other months each year in every decade
            yearly_wins = []
            for year in decade_data['Year'].unique():
                year_data = decade_data[decade_data['Year'] == year]
                jan_year = year_data[year_data['Month'] == 1]['Return'].mean()
                other_year = year_data[year_data['Month']
                                       != 1]['Return'].mean()
                yearly_wins.append(jan_year > other_year)

            win_rate = np.mean(yearly_wins) * 100 if yearly_wins else 0

            results.append({
                'Decade': f"{decade}s",
                'Jan_Mean': jan_returns.mean(),
                'Other_Mean': other_returns.mean(),
                'Difference': jan_returns.mean() - other_returns.mean(),
                'T_Stat': t_stat,
                'P_Value': p_value,
                'Win_Rate': win_rate,
                'N_Years': years_in_decade,
                'N_Jan_Obs': len(jan_returns),
                'N_Other_Obs': len(other_returns),
                'Significant': p_value < 0.05
            })

    jan_effect_decade_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("JANUARY EFFECT BY DECADE")
    print("="*80)
    print(f"{'Decade':<10} {'Jan Mean':>11} {'Other Mean':>11} {'Difference':>12} "
          f"{'Win Rate':>10} {'T-Stat':>8} {'P-Value':>9}")
    print("-"*80)

    for _, row in jan_effect_decade_results.iterrows():
        print(f"{row['Decade']:<10} {row['Jan_Mean']:>9.3f}% {row['Other_Mean']:>9.3f}% "
              f"{row['Difference']:>9.3f}% {row['Win_Rate']:>11.1f}% "
              f"{row['T_Stat']:>9.2f} {row['P_Value']:>9.4f}")
    print("="*80)
    print("\nWin Rate = % of years where January returns are greater than other months in that decade")
    return jan_effect_decade_results


def graph_january_effect_by_decade(results_df):
    """
    Create a bar chart comparing the January effect across different decades
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['green' if sig else 'lightgray' for sig in results_df['Significant']]
    bars = ax1.bar(results_df['Decade'], results_df['Difference'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, diff in zip(bars, results_df['Difference']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{diff:.3f}%',
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=10, fontweight='bold')

    # Plot 1: January by Decade
    ax1.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax1.set_ylabel('January Difference (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Decade', fontweight='bold', fontsize=12)
    ax1.set_title('January Effect by Decade', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Win rate
    ax2.bar(results_df['Decade'], results_df['Win_Rate'],
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2,
                label='Random (50%)', alpha=0.7)

    for i, (decade, rate) in enumerate(zip(results_df['Decade'], results_df['Win_Rate'])):
        ax2.text(i, rate + 2, f'{rate:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Win Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Decade', fontweight='bold', fontsize=12)
    ax2.set_title('% of Years January Beats Other Months',
                  fontweight='bold', fontsize=13)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

def january_year_by_year(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    daily_avg = (
        df.groupby(["Date", "Year", "is_january"])["Return"]
        .mean()
        .reset_index()
    )

    yearly_diff = (
        daily_avg
        .groupby("Year")
        .apply(
            lambda g: g.loc[g["is_january"], "Return"].mean()
                    - g.loc[~g["is_january"], "Return"].mean()
        )
    )

    plt.figure(figsize=(10, 5))
    plt.bar(yearly_diff.index, yearly_diff.values)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("Average January return − Average return (other months)")
    plt.title("Year-by-Year January Effect")

    plt.show()

    return
