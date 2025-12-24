import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.data import load_all_stocks, load_metadata, prepare_data_for_analysis
import seaborn as sns
from scipy import stats
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
DIRECTORY = '/ADA/2/stocks'

def load_data_monday():
    df_with_returnsdf = load_all_stocks()
    metadata = load_metadata()

    df_with_returns = prepare_data_for_analysis(df_with_returnsdf, metadata)
    df_by_date = df_with_returns.groupby("Date")['Return'].mean().reset_index()

    return df_with_returns, df_by_date

def plot_volume_per_day(df, first_year=1962, last_year=2020, gap=10):

    period_labels = []

    proportions_list = {
        0 : [],
        1 : [],
        2 : [],
        3 : [],
        4 : [],
        5 : [],
        6 : []
    }
    df = df[["Date", "Volume", "Year"]].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    for year in range(first_year, last_year, gap):
        
        if year + gap == last_year:
            upper_bound = year + gap
        else:
            upper_bound = year + gap - 1

        period_labels.append(f"{year}-{upper_bound}")

        df_period = df[(df["Year"] >= year) & (df["Year"] <= upper_bound)]
        
        for day in range(7):
            volume = df_period[df_period['Date'].dt.weekday == day]['Volume']
            volume_mean = volume.mean()

            proportions_list[day].append(volume_mean)
    
    del proportions_list[5]
    del proportions_list[6]
    df = pd.DataFrame(proportions_list, index=period_labels)

    periods = [f"{y}" for y in range(first_year, last_year, gap)]
    df_periods = pd.DataFrame(proportions_list, index=period_labels)

    #print(proportions_list)

    #plt.plot(periods, df)  # une ligne par colonne
    plt.figure(figsize=(12,6))
    ax = df_periods.plot(marker='o', ax=plt.gca())

    # Mettre Lundi en évidence
    if len(ax.lines) > 0:
        ax.lines[0].set_color('red')
        ax.lines[0].set_linewidth(2)


    plt.xlabel("Period")
    plt.ylabel("Average return per day")
    plt.title("Average return per weekday by period")
    plt.legend(title="Day of the week", labels=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()
    
    bar_volumes = {i: sum(proportions_list.get(i))/len(proportions_list.get(i)) for i in range(5)}
    threshold = sum(bar_volumes.values()) / len(bar_volumes)

    fig, ax = plt.subplots(figsize=(8,5))


    bars = ax.bar(
    bar_volumes.keys(), 
    bar_volumes.values(), 
    color="#27DAF5", 
    alpha=0.8,
    edgecolor='black', 
    linewidth=1.5,
    label="Average Volume"
    )


    line = ax.axhline(
        y=threshold, 
        linestyle='--', 
        color='red', 
        label="Overall Mean",  
        linewidth=1.5
    )


    ax.set_xticks(range(5))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri"])
    ax.set_ylabel("Average Volume")
    ax.set_title("Average Trading Volume per Weekday")
    ax.grid(True, alpha=0.3, axis='y')


    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.01*max(bar_volumes.values()), 
            f"{height:.0f}", 
            ha='center', va='bottom', fontsize=9
        )


    ax.legend(loc='lower left')

    plt.show()





def plot_return_per_day(df, first_year=1962, last_year=2020, gap=10):

    df = df[['Date', 'Return', 'Year']].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    proportions_list_return = {i: [] for i in range(7)} 
    period_labels = []

    for start_year in range(first_year, last_year, gap):
        if start_year == last_year:
            end_year = start_year + gap
        else:
            end_year = start_year + gap - 1

        period_labels.append(f"{start_year}-{end_year}")

        df_period = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
        if df_period.empty:

            for day in range(7):
                proportions_list_return[day].append(0)
            continue


        for day in range(7):
            percent_return = df_period[df_period['Date'].dt.weekday == day]['Return']
            percent_return = percent_return.replace([float('inf'), [float('-inf')]], 0)
            percent_return = percent_return.replace([float('nan'), [float('-nan')]], 0)


            if percent_return.empty:
                mean_return = 0.0
            else:
                mean_return = percent_return.mean()
            proportions_list_return[day].append(mean_return)


        
    del proportions_list_return[5]
    del proportions_list_return[6]


    df_plot = pd.DataFrame(proportions_list_return, index=period_labels)

    plt.figure(figsize=(12,6))
    ax = df_plot.plot(marker='o', ax=plt.gca())


    if len(ax.lines) > 0:
        ax.lines[0].set_color('red')
        ax.lines[0].set_linewidth(2)


    plt.xlabel("Period")
    plt.ylabel("Average return per day")
    plt.title("Average return per weekday by period")
    plt.legend(title="Day of the week", labels=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    bar_volumes = {i: sum(proportions_list_return.get(i))/len(proportions_list_return.get(i)) for i in range(5)}
    threshold = sum(bar_volumes.values()) / len(bar_volumes)

    fig, ax = plt.subplots(figsize=(8,5))


    bars = ax.bar(
    bar_volumes.keys(), 
    bar_volumes.values(), 
    color="#27F576", 
    alpha=0.8,
    edgecolor='black', 
    linewidth=1.5,
    label="Average Volume"
    )


    line = ax.axhline(
        y=threshold, 
        linestyle='--', 
        color='red', 
        label="Overall Mean",  
        linewidth=1.5
    )


    ax.set_xticks(range(5))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri"])
    ax.set_ylabel("Average Volume")
    ax.set_title("Average return per Weekday")
    ax.grid(True, alpha=0.3, axis='y')


    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.01*max(bar_volumes.values()), 
            f"{height:.0f}", 
            ha='center', va='bottom', fontsize=9
        )


    ax.legend(loc='upper left')

    plt.show()

def stat_mw_test(df):
    pvalues = np.zeros((5,5))
    df_test = df.copy()

    for i in range(5):
        for j in range(5):
            filtered_dfnz = df_test[df_test['day'] == i]
            filtered_df = df_test[df_test['day'] == j]
            #test = stats.kstest(filtered_dfnz["Return"],filtered_df["Return"], N=max(len(filtered_dfnz["Return"]) , len(filtered_df["Return"])))
            test = stats.mannwhitneyu(filtered_dfnz["Return"],filtered_df["Return"])

            pvalues[i][j]= test.pvalue
    log_pvalues = np.log(pvalues)

    return pvalues, log_pvalues

def stat_t_test(df):
    df_test=df.copy()
    pvalues = np.zeros((5,5))


    for i in range(5):
        for j in range(5):
            filtered_dfnz = df_test[df_test['day'] == i]
            filtered_df = df_test[df_test['day'] == j]
            #test = stats.kstest(filtered_dfnz["Return"],filtered_df["Return"], N=max(len(filtered_dfnz["Return"]) , len(filtered_df["Return"])))
            test = stats.ttest_ind(filtered_dfnz["Return"],filtered_df["Return"])

            pvalues[i][j]= test.pvalue
    log_pvalues = np.log(pvalues)

    return pvalues, log_pvalues


def plot_fluvial(df_copy, year_min=1960, year_max=2020):

    def get_groups(p_array):
        groups = [0, 1, 2, 3, 4]
        for i in range(1, 5):
            for j in range(0, i):
                if p_array[i, j] > 0.05:
                    groups[i] = j
            return groups

    df_aluvial = df_copy.copy()
    df_aluvial["Year"] = df_copy["Date"].dt.year
    print(df_aluvial.min())
    print(df_aluvial.max())
    year_min = 1960
    year_max = 2020
    all_groups = []
    for i in range(0, 60, 10):
        #print(str(year_min + i) +" et "+ str(year_min + i + 10))
        df_filtered = df_aluvial[(df_aluvial["Year"] > year_min + i) & (df_aluvial["Year"] <= year_min + i + 10)]
        pvalues, _ = stat_t_test(df_filtered)
        all_groups.append(get_groups(pvalues))
        """
        for l in all_groups:
            for i, g in enumerate(l):
                l[i] = week[g]
        """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    df = pd.DataFrame(all_groups).T
    noms_periodes = [f"{year_min + 10*i} - {year_min+(i+1)*10}" for i in range(df.shape[1])]
    df.columns = noms_periodes

    jours_liste = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df.insert(0, 'Day', jours_liste)
    df['Jour_ID'] = range(len(df))


    dimensions = [
        dict(values=df[col], label=col)
        for col in ['Day'] + noms_periodes
    ]

    fig = go.Figure(
        data=[
            go.Parcats(
                dimensions=dimensions,
                line=dict(
                    color=df['Jour_ID'],
                    colorscale='Agsunset',
                    shape='hspline'
                )
            )
        ]
    )

    fig.update_layout(
        template="plotly_white",
        title="Evolution of the significantly different average returns"
    )

    fig.show()
    fig.write_html("plot_ttest_monday.html", include_plotlyjs="cdn")
    
def plot_kde_return(distribution_list, colors, labels):

    plt.figure(figsize=(10, 6))
    for i, dist in enumerate(distribution_list):
        sns.kdeplot(distribution_list[i], color=colors[i], label=labels[i], fill=True, alpha=0.7)
    plt.legend()
    plt.grid()
    plt.show()

def plot_kde_per_day(df):
    week = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday"}

    df_by_date = df.copy()
    df_by_date = df_by_date.groupby("Date")['Return'].mean().reset_index()
    colors = ["#34A6F4", "#31C950","#ED6AFF","#C4CA1B", "#FE9A37"]
    distribution_list = []
    labels = []
    order = [4,2,3,1,0]
    for i in order:
        df_filtered=df_by_date.copy()
        df_filtered = df_by_date[df_by_date['Date'].dt.weekday == i]['Return']
        distribution_list.append(df_filtered)
        labels.append(week[i])
    plot_kde_return(distribution_list,colors, labels)

