import pandas as pd
import numpy as np


def load_market_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date')
    return df


def compute_log_returns(df, price_col='Close'):
    df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def compute_rolling_volatility(df, window=30):
    df['Volatility_30D'] = df['Log_Return'].rolling(window).std()
    return df


def load_covid_data(filepath):
    covid = pd.read_csv(filepath)
    covid = covid[covid['country'] == 'India'].copy()
    covid['date'] = pd.to_datetime(covid['date'])
    covid = covid[['date', 'new_cases']]
    covid = covid.rename(columns={'date': 'Date'})
    return covid


def merge_market_covid(market_df, covid_df):
    merged = pd.merge(market_df, covid_df, on='Date', how='left')
    merged['new_cases'] = merged['new_cases'].fillna(0)
    merged['cases_7day_avg'] = merged['new_cases'].rolling(7).mean()
    return merged


def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)

