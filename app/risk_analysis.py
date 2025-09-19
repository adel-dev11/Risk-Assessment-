import numpy as np
import pandas as pd
from data_preprocess import preprocess_data

def risk_analysis(df, lookback=90):
    window_data = df.tail(lookback).copy()

    vol = window_data["daily_return"].std()
    vol_annual = vol * np.sqrt(252)

    VaR_95 = np.percentile(window_data["daily_return"].dropna(), 5)

    window_data["CumReturn"] = (1 + window_data["daily_return"]).cumprod()
    window_data["RunningMax"] = window_data["CumReturn"].cummax()
    window_data["Drawdown"] = (window_data["CumReturn"] / window_data["RunningMax"]) - 1
    max_drawdown = window_data["Drawdown"].min()

    total_cum_return = window_data["CumReturn"].iloc[-1] - 1

    risk_summary = pd.DataFrame({
        "Metric": ["Volatility (daily)", "Volatility (annual)", "VaR 95%", "Max Drawdown", "Cumulative Return"],
        "Value": [vol, vol_annual, VaR_95, max_drawdown, total_cum_return]
    })

    return risk_summary

# if __name__ == "__main__":
#     file_path = input("دخل اسم ملف البيانات الخام (csv): ")
#     df, output_file = preprocess_data(file_path, save=True)
#     print(risk_analysis(df))
