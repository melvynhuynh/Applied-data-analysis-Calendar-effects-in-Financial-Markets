import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_ols(df, effect_col, controls=None, mode="cluster", hac_lags=5):
    if controls is None:
        controls = []

    data = df.copy()

    # cluster
    if mode == "cluster":

        cols = ["Return", effect_col, "Date", "Ticker"] + controls

        data = data[cols].copy()

        data["Return"] = pd.to_numeric(data["Return"], errors="coerce")
        data[effect_col] = data[effect_col].astype(int)

        for c in controls:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        for c in controls:
            mean_c = data[c].mean()
            std_c = data[c].std(ddof=0)
            data[c] = (data[c] - mean_c) / std_c

        y = data["Return"].astype(float)
        X = sm.add_constant(data[[effect_col] + controls].astype(float))

        date_codes, _ = pd.factorize(data["Date"])
        stock_codes, _ = pd.factorize(data["Ticker"])
        clusters = np.column_stack([date_codes, stock_codes])

        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters}
        )

        return model

    # time series
    elif mode == "ts":

        agg_cols = ["Return"] + controls + [effect_col]
        data = (
            data.groupby("Date")[agg_cols]
                .mean()
                .reset_index()
                .dropna()
        )

        for c in controls:
            mean_c = data[c].mean()
            std_c = data[c].std(ddof=0)
            data[c] = (data[c] - mean_c) / std_c

        y = data["Return"].astype(float)
        X = sm.add_constant(data[[effect_col] + controls].astype(float))

        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags}
        )

        return model

    else:
        raise ValueError("mode must be 'cluster' or 'ts'")
