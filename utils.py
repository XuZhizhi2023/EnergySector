from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates


def get_raw_data(ticker: str) -> pd.DataFrame:
    data = pd.read_csv(f"data/{ticker}.csv")[::-1]
    data.reset_index(inplace=True, drop=True)
    return data


def get_normalized_close_price(data: pd.DataFrame) -> None:
    data["norm_close_price"] = data["Price"] / data.loc[0, "Price"]


def get_return1_and_logvol20(data: pd.DataFrame) -> None:
    data["return"] = data["Price"] / data["Price"].shift(1) - 1.0
    data["logvol20"] = np.log(data["return"].rolling(20).std())


def build_datasets(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    datasets = {}
    for ticker in tickers:
        data = get_raw_data(ticker)
        get_normalized_close_price(data)
        get_return1_and_logvol20(data)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        datasets[ticker] = data
    return datasets


def plot_norm_price(
    tickers: List[str], datasets: Dict[str, pd.DataFrame], save_path: Optional[str] = None
) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot()
    for ticker in tickers:
        ax.plot(
            pd.to_datetime(datasets[ticker]["Date"]),
            datasets[ticker]["norm_close_price"],
            label=f"normalized close price of {ticker}",
        )
        ax.set_xlabel("date")
        ax.set_ylabel("normalized close price")
        ax.xaxis.set_major_formatter(mpl_dates.DateFormatter("%Y-%m-%d"))
        ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=500)
    return fig


def plot_logvol20(tickers: List[str], datasets: Dict[str, pd.DataFrame], save_path: Optional[str] = None) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot()
    for ticker in tickers:
        ax.plot(
            pd.to_datetime(datasets[ticker]["Date"]),
            datasets[ticker]["logvol20"],
            label=f"20-day log-volatility of {ticker}",
        )
        ax.set_xlabel("date")
        ax.set_ylabel("20-day log-volatility")
        ax.xaxis.set_major_formatter(mpl_dates.DateFormatter("%Y-%m-%d"))
        ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=500)
    return fig
