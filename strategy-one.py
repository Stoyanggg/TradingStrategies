import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


"""
StrategyOne implements a trading strategy based on simple technical analysis for backtesting historical price data. 
The strategy looks for three consecutive daily price drops and buys the asset on the fourth day. 
It uses customizable stop-loss and take-profit levels to manage trades. The class provides methods for fetching 
historical data, running backtests, testing on new data, calculating performance metrics, and plotting cumulative 
profit and loss (P&L) or percentage gain.

Attributes:
    symbol (str): The ticker symbol of the financial instrument (e.g., 'AAPL' for Apple Inc.).
    start (str): The start date for the historical data in 'YYYY-MM-DD' format.
    end (str): The end date for the historical data in 'YYYY-MM-DD' format.
    interval (str): The interval of the historical data (e.g., '1d' for daily data).
    data (pd.DataFrame or None): DataFrame to hold historical data and calculated indicators; initialized to None.
    account (float): Starting account balance; initialized to 100000.
    pnl (float): Track total profit and loss; initialized to 0.
    position (bool): To check if there is an open position; initialized to False.
    buy_price (float): Track the buy price; initialized to 0.
    shares (float): Track the number of shares bought; initialized to 0.
    cumulative_pnl (list): List to hold cumulative P&L values.

Methods:
    __init__(self, symbol, start, end, interval):
            Initializes the class with a symbol, start date, end date, and interval.
        
        get_data(self):
            Downloads and prepares historical data from Yahoo Finance.
        
        backtesting(self):
            Runs the backtest on historical data based on the defined trading strategy.
        
        run_backtest(self, start, end):
            Runs the entire strategy backtest for the specified period.
        
        test_on_new_data(self, start, end):
            Tests the strategy on new data using the same parameters.
        
        calculate_performance_metrics(self):
            Calculates and prints performance metrics including Sharpe Ratio, Maximum Drawdown, and Maximum Gain.
        
        plot_pnl(self):
            Plots the cumulative P&L of the strategy over time.
        
        plot_percentage_gain(self):
            Plots the cumulative returns of the strategy as a percentage over time.
"""


class StrategyOne():
    def __init__(self, symbol, start, end, interval):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.data = None  # Placeholder for historical data
        self.account = 100000  # Starting account balance
        self.pnl = 0  # Track total profit and loss
        self.position = False  # To check if we have an open position
        self.buy_price = 0  # Track the buy price
        self.shares = 0  # Track the number of shares bought
       

    def get_data(self):
        
        """Downloads and prepares the data."""
        
        data = yf.download(tickers=self.symbol, start=self.start, end=self.end, interval=self.interval)
        data = data[['Adj Close']]
        data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        data.dropna(inplace=True)
        self.data = data

    def backtesting(self):
        
        self.get_data()
        
        """Runs the backtest on historical data."""
        
        self.account = 100000  # Reset account balance
        self.pnl = 0  # Reset P&L
        self.position = False  # Reset position
        self.shares = 0  # Reset shares
        percentage = 5  # Percentage of account used per trade
        stop_loss = 5
        take_profit = 15
        self.cumulative_pnl = []
        
        # Loop through the data
        
        for i in range(len(self.data)): 
            
            # Check if the price has dropped for three consecutive days
            
            if (self.data['Close'][i-3] > self.data['Close'][i-2] and 
                self.data['Close'][i-2] > self.data['Close'][i-1] and 
                self.data['Close'][i-1] > self.data['Close'][i]) and not self.position:
                
                # Buy on the 4th day after three consecutive drops
                
                self.shares = (self.account * (percentage/100)) / self.data['Close'][i]
                self.buy_price = self.data['Close'][i]
                self.account -= self.shares * self.buy_price
                self.position = True
                
            
                print(f'Bought at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')

            # Check for stop loss and selling conditions if a position is open
            
            if self.position:
    # Selling Condition: If price goes up by 15% from the buy price
    
              if self.data['Close'][i] >= self.buy_price * (1 + take_profit/100):
                sell_price = self.data['Close'][i]
                profit_loss = self.shares * (sell_price - self.buy_price)
                self.pnl += profit_loss
                self.account += self.shares * sell_price
                self.shares = 0
                self.position = False  # Reset position after selling
                print(f'Sold at {round(sell_price,2)}, profit: {round(profit_loss,2)}, account after selling: {round(self.account,2)}')

              # Stop-Loss Condition: If price drops by 5% from the buy price
            
              elif self.data['Close'][i] <= self.buy_price * (1 - stop_loss/100):
                sell_price = self.data['Close'][i]
                profit_loss = self.shares * (sell_price - self.buy_price)
                self.pnl += profit_loss
                self.account += self.shares * sell_price
                self.shares = 0
                self.position = False  # Reset position after selling
                print(f'Stop-loss triggered, sold at {round(sell_price,2)}, loss: {round(profit_loss,2)}, account after selling: {round(self.account,2)}')

            self.cumulative_pnl.append(self.pnl)
            
        print(f"Total P&L: {round(self.pnl,2)}")
        print(f"Final Account Balance: {round(self.account,2)}")
        print(f'Open position: {self.position}')
        
    def run_backtest(self,start,end):
        
        self.start = start
        self.end = end
        
        """Runs the entire strategy backtest for the specified period."""
        
        self.get_data()
        self.backtesting()

    def test_on_new_data(self, start, end):
        
        """Tests the strategy on new data using the same parameters."""
        
        self.start = start
        self.end = end
        self.get_data()  # Re-fetch the new data
        self.backtesting()  # Backtest on the new data period

    def calculate_performance_metrics(self):
        
        returns = self.data['Close'].pct_change()
        sharpe_ratio = (returns.mean() / returns.std()) * (252**0.5)  # Annualized Sharpe Ratio
        
        # Calculate Maximum Drawdown
        
        cumulative_max = self.data['Close'].cummax()
        drawdown = (self.data['Close'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100  # Convert to percentage
        
        # Calculate Maximum Gain
        
        cumulative_min = self.data['Close'].cummin()
        gain = (self.data['Close'] - cumulative_min) / cumulative_min
        max_gain = gain.max() * 100  # Convert to percentage
        
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Maximum Gain: {max_gain:.2f}%")


    def plot_pnl(self):
        
        """Plots the cumulative P&L of the strategy over time."""

        plt.figure(figsize=(12, 6))
        plt.plot(self.cumulative_pnl, label='Cumulative P&L')
        plt.title('Strategy Cumulative P&L Over Time')
        plt.xlabel('Time (trading steps)')
        plt.ylabel('Cumulative P&L ($)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_percentage_gain(self):
        
        self.backtesting
        initial_capital = 100000
      
      # Convert cumulative P&L to percentage gain
        
        cumulative_returns = (np.array(self.cumulative_pnl) / initial_capital) * 100

      # Create a cumulative P&L plot
    
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns, label='Cumulative Returns (%)')
        plt.title('Strategy Cumulative Returns Over Time')
        plt.xlabel('Time (trading steps)')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True)
        plt.show()


strategy = StrategyOne('aapl','2010-01-01','2023-01-01','1d')

strategy.backtesting()

strategy.run_backtest('2010-01-01','2022-01-01')

strategy.plot_pnl()

strategy.test_on_new_data('2022-01-01','2023-01-01')

strategy.plot_pnl()

