import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product


"""
Documentation for BestSMA Class

Class: BestSMA
Description:
    The `BestSMA` class is designed to implement a backtesting strategy for trading financial instruments 
    using two Simple Moving Averages (SMAs): a short SMA and a long SMA. The class provides methods to download historical data
    from Yahoo Finance, calculate SMAs, generate trading signals, and evaluate the strategy's performance based on cumulative
    returns.

Attributes:
    ticker (str): The ticker symbol of the financial instrument (e.g., 'AAPL' for Apple Inc.).
    start (str): The start date for the historical data in 'YYYY-MM-DD' format.
    end (str): The end date for the historical data in 'YYYY-MM-DD' format.
    interval (str): The interval of the historical data (e.g., '1d' for daily data).
    data (pd.DataFrame or None): A DataFrame to hold historical data and calculated indicators; initialized to None.

Methods:
    __init__(self, ticker, start, end, interval):
        Initializes the class with a ticker, start and end dates, and an interval for historical data.
    
    get_data(self):
        Downloads historical data from Yahoo Finance using the yfinance library and prepares it for analysis by keeping 
        only the adjusted close prices.
        
        Returns:
            pd.DataFrame: A DataFrame containing the historical data with the 'Close' column.
        
        Raises:
            ValueError: If data cannot be downloaded.

    test_strategy(self, sma_params):
        Tests a trading strategy based on the difference between two SMAs (short and long).

        Args:
            sma_params (tuple): A tuple containing two integers, `sma_s` (short SMA period) and `sma_l` (long SMA period).

        Returns:
            float: The cumulative performance of the strategy using the specified SMAs.

        Raises:
            ValueError: If the data is not loaded before calling this method.

Example Usage:
    ticker = 'AAPL'
    start = '2010-01-01'
    end = '2024-01-01'
    interval = '1d'

    # Instantiate the class
    best_sma = BestSMA(ticker, start, end, interval)

    # Get data
    best_sma.get_data()

    # Define ranges for short and long SMAs
    sma_s_range = range(10, 50, 1)
    sma_l_range = range(50, 250, 1)
    
    # Generate combinations of SMA parameters
    combinations = list(product(sma_s_range, sma_l_range))

    # Calculate the performance of each combination
    results = [best_sma.test_strategy(comb) for comb in combinations]

    # Create a DataFrame to store SMA combinations and their corresponding performance
    many_result = pd.DataFrame(data=combinations, columns=['sma_s', 'sma_l'])
    many_result['performance'] = results

    # Print or analyze the top 10 results based on performance
    print(many_result.nlargest(10, 'performance'))
"""
class BestSMA():
    def __init__(self, ticker, start, end, interval):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.data = None  # Initialize data as None

    def get_data(self):
        """Download data from Yahoo Finance and prepare it."""
        self.data = yf.download(self.ticker, self.start, self.end, self.interval)
        self.data = self.data[['Adj Close']]
        self.data.rename(columns=({'Adj Close': 'Close'}), inplace=True)
        return self.data

    def test_strategy(self, sma_params):
        """Test a strategy based on short and long SMA."""
        if self.data is None:
            raise ValueError("Data not loaded. Call get_data() first.")

        sma_s, sma_l = sma_params
        self.data['returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Calculate rolling averages for short and long SMAs
        self.data['sma_s'] = self.data['Close'].rolling(int(sma_s)).mean()
        self.data['sma_l'] = self.data['Close'].rolling(int(sma_l)).mean()
        # Drop rows with NaN values
        self.data.dropna(inplace=True)

        # Generate trading signals based on SMAs
        self.data['position'] = np.where(self.data['sma_s'] > self.data['sma_l'], 1, -1)
        # Calculate strategy returns
        self.data['strategy'] = self.data['position'].shift(1) * self.data['returns']
        self.data.dropna(inplace=True)

        # Calculate the cumulative returns of the strategy and sum them up
        return np.exp(self.data['strategy'].sum())

# Example usage:
ticker = 'AAPL'
start = '2010-01-01'
end = '2024-01-01'
interval = '1d'

# Instantiate the class
best_sma = BestSMA(ticker, start, end, interval)

# Get data
best_sma.get_data()

# Define ranges for short and long SMAs
sma_s_range = range(10, 50, 1)
sma_l_range = range(50, 250, 1)
# Generate combinations of SMA parameters
combinations = list(product(sma_s_range, sma_l_range))

# Ensure that results and combinations have the same length
results = [best_sma.test_strategy(comb) for comb in combinations]

# Create a DataFrame to store SMA combinations and their corresponding performance
many_result = pd.DataFrame(data=combinations, columns=['sma_s', 'sma_l'])
many_result['performance'] = results

# Print or analyze the top 10 results based on performance
print(many_result.nlargest(10, 'performance'))
