import pandas as pd
import numpy as np

# ----------------------------
# Load Market Data
# ----------------------------
nifty = pd.read_csv("data/raw/nifty50_raw.csv")
pharma = pd.read_csv("data/raw/nifty_pharma_raw.csv")

# ----------------------------
# Format Dates
# ----------------------------
nifty['Date'] = pd.to_datetime(nifty['Date'], dayfirst=True)
pharma['Date'] = pd.to_datetime(pharma['Date'], dayfirst=True)

# Sort by Date (important)
nifty = nifty.sort_values('Date')
pharma = pharma.sort_values('Date')

# ----------------------------
# Handle Missing Values
# ----------------------------
nifty = nifty.dropna()
pharma = pharma.dropna()

# ----------------------------
# Compute Log Returns (using Close)
# ----------------------------
nifty['Log_Return'] = np.log(nifty['Close'] / nifty['Close'].shift(1))
pharma['Log_Return'] = np.log(pharma['Close'] / pharma['Close'].shift(1))

# ----------------------------
# 30-Day Rolling Volatility
# ----------------------------
nifty['Volatility_30D'] = nifty['Log_Return'].rolling(30).std()
pharma['Volatility_30D'] = pharma['Log_Return'].rolling(30).std()

# ----------------------------
# Save Cleaned Data
# ----------------------------
nifty.to_csv("data/processed/nifty50_cleaned.csv", index=False)
pharma.to_csv("data/processed/nifty_pharma_cleaned.csv", index=False)

print("Data cleaning complete.")

