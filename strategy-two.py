import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

"""
Documentation for StrategyTwo Class

Class: StrategyTwo
Description: The StrategyTwo class implements a backtesting strategy for trading financial instruments using 
technical indicators such as SMA (Simple Moving Average), MACD (Moving Average Convergence Divergence), 
and Bollinger Bands.

Attributes:
    symbol (str): The ticker symbol of the financial instrument.
    start (str): The start date for the historical data.
    end (str): The end date for the historical data.
    interval (str): The interval of the historical data.
    use_sma_short (bool): Flag to indicate whether to use the short SMA in the strategy.
    use_sma_long (bool): Flag to indicate whether to use the long SMA in the strategy.
    use_macd (bool): Flag to indicate whether to use the MACD in the strategy.
    use_bb (bool): Flag to indicate whether to use the Bollinger Bands in the strategy.
    pnl (float): The profit and loss from trading.
    percentage (float): The percentage of the account used per trade.
    stop_loss (float): The stop loss percentage.
    account (float): The starting account balance.
    position (str or None): The current position ('long', 'short', or None).
    shares (int): The number of shares bought or sold.
    buy_price (float): The price at which a position was entered.
    sma_short_period (int): The period for the short SMA.
    sma_long_period (int): The period for the long SMA.

Methods:
    __init__(self, symbol, start, end, interval, use_sma_short, use_sma_long, use_macd, use_bb):
        Initializes the class with a symbol, start date, end date, and interval.
    
    get_data(): 
        Retrieves historical data from Yahoo Finance and calculates the necessary indicators.
    
    backtesting(): 
        Runs the backtesting strategy based on the specified indicators and rules.
    
    run_backtest(start, end):
        Executes the backtest for the provided date range.
    
    test_on_new_data(start, end): 
        Tests the strategy on new data.
    
    calculate_performance_metrics(): 
        Computes performance metrics like Sharpe Ratio and Maximum Drawdown.
    
    plot_graph(): 
        Plots the price chart with indicators.
    
    plot_pnl(): 
        Plots the cumulative profit and loss.
    
    plot_percentage_gain(): 
        Plots the cumulative percentage returns.
"""



class StrategyTwo():
    def __init__(self, symbol, start, end, interval, use_sma_short= True, use_sma_long = True, use_macd= True, use_bb = False):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.use_sma_short = use_sma_short
        self.use_sma_long = use_sma_long
        self.use_macd = use_macd
        self.use_bb = use_bb
        self.pnl = 0
        self.percentage = 10  # Percentage of account used per trade
        self.stop_loss = 5  # Stop loss percentage
        self.account = 100000  # Starting account balance
        self.position = None  # Track the type of position: 'long', 'short', or None
        self.shares = 0
        self.buy_price = 0  # Initialize buy price
        self.sma_short_period = 10 # short SMA which we got from class BestSMA
        self.sma_long_period = 58  # long SMA which we got from class BestSMA

    def get_data(self):    
        
        data = yf.download(tickers=self.symbol, start=self.start, end=self.end)
        data = data[['Adj Close']]
        data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        
        if self.use_bb:
            data['Mean'] = data['Close'].rolling(window=10).mean()
            data['Std'] = data['Close'].rolling(window=10).std()
            data['Upper_Threshold'] = data['Mean'] + 2 * data['Std']
            data['Lower_Threshold'] = data['Mean'] - 2 * data['Std']

        if self.use_sma_short:
            data['Sma_short'] = data['Close'].rolling(window=self.sma_short_period).mean()
        if self.use_sma_long:
            data['Sma_long'] = data['Close'].rolling(window=self.sma_long_period).mean()
        if self.use_macd:
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        data.dropna(inplace=True)
        self.data = data

    def backtesting(self):
        
        self.get_data()  # Ensure data is loaded before backtesting
        self.cumulative_pnl = []  # Store cumulative P&L for each step
        self.pnl = 0
        self.percentage = 10  # Percentage of account used per trade
        self.stop_loss = 5  # Stop loss percentage
        self.account = 100000  # Starting account balance
        self.position = None  # Track the type of position: 'long', 'short', or None
        self.shares = 0
        
        for i in range(len(self.data)):
            
            # Long Position Entry Condition
            
            if self.position is None:
                if self.use_sma_short and self.use_sma_long and self.use_macd and self.use_bb:
                    if self.data['Sma_short'][i] > self.data['Sma_long'][i] and self.data['Close'][i] < self.data['Mean'][i] and self.data['Signal_Line'][i] > self.data['MACD'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account -= self.shares * self.buy_price  # Deduct cost of shares from the account
                        self.position = 'long'
                        print(f'Bought long at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')

                elif self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] > self.data['Sma_long'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account -= self.shares * self.buy_price
                        self.position = 'long'
                        print(f'Bought long at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')
                
                elif self.use_macd and self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] > self.data['Sma_long'][i] and self.data['Signal_Line'][i] > self.data['MACD'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account -= self.shares * self.buy_price
                        self.position = 'long'
                        print(f'Bought long at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')

                elif self.use_bb and self.use_macd :
                    if self.data['Close'][i] < self.data['Lower_Threshold'][i] and self.data['Signal_Line'][i] > self.data['MACD'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account -= self.shares * self.buy_price
                        self.position = 'long'
                        print(f'Bought long at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')
                
                elif self.use_bb and self.use_sma_short and self.use_long :
                    if self.data['Close'][i] < self.data['Mean'][i] and self.data['Sma_short'][i] > self.data['Sma_long'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account -= self.shares * self.buy_price
                        self.position = 'long'
                        print(f'Bought long at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')
                
            # Short Position Entry Condition
            
            elif self.position is None:
                if self.use_sma_short and self.use_sma_long and self.use_macd and self.use_bb:
                    if self.data['Sma_short'][i] < self.data['Sma_long'][i] and self.data['Signal_Line'][i] < self.data['MACD'][i] and self.data['Close'][i] > self.data['Mean'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account += self.shares * self.buy_price  # Receive cash from short selling
                        self.position = 'short'
                        print(f'Short sold at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after short selling: {round(self.account,2)}')
                
                elif self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] < self.data['Sma_long'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account += self.shares * self.buy_price
                        self.position = 'short'
                        print(f'Short sold at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after short selling: {round(self.account,2)}')

                elif self.use_macd and self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] < self.data['Sma_long'][i] and self.data['Signal_Line'][i] < self.data['MACD'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account += self.shares * self.buy_price
                        self.position = 'short'
                        print(f'Short sold at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after short selling: {round(self.account,2)}')

                elif self.use_macd and self.use_bb:
                    if self.data['Close'][i] > self.data['Mean'][i] and self.data['Signal_Line'][i] < self.data['MACD'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account += self.shares * self.buy_price
                        self.position = 'short'
                        print(f'Short sold at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after short selling: {round(self.account,2)}')


                elif self.use_bb and self.use_sma_short and self.use_long :
                    if self.data['Close'][i] > self.data['Mean'][i] and self.data['Sma_short'][i] < self.data['Sma_long'][i]:
                        self.shares = (self.account * (self.percentage / 100)) / self.data['Close'][i]
                        self.buy_price = self.data['Close'][i]
                        self.account -= self.shares * self.buy_price
                        self.position = 'short'
                        print(f'Bought long at {round(self.buy_price,2)}, shares: {round(self.shares,2)}, account after buying: {round(self.account,2)}')

            # Manage Long Position
            
            if self.position == 'long':
                
                # Stop Loss Condition for Long
                
                if self.data['Close'][i] <= self.buy_price * (1 - self.stop_loss / 100):
                    self.sell_price = self.data['Close'][i]
                    self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                    self.account += self.shares * self.sell_price
                    self.pnl += self.profit_loss
                    self.shares = 0
                    self.position = None
                    print(f'Stop loss for long triggered, sold at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, loss: {round(self.profit_loss,2)}')

                # Regular Selling Condition for Long
                
                elif self.use_sma_short and self.use_sma_long and self.use_macd and self.use_bb:
                    if self.data['Sma_short'][i] < self.data['Sma_long'][i] and self.data['Close'][i] > self.data['Mean'][i] and self.data['Signal_Line'][i] < self.data['MACD'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {self.sell_price}, account after selling: {self.account}, profit: {self.profit_loss}')

                elif self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] < self.data['Sma_long'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')
                
                elif self.use_macd and self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] < self.data['Sma_long'][i] and self.data['Signal_Line'][i] < self.data['MACD'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')
                
                elif self.use_macd and self.use_bb:
                    if self.data['Close'][i] > self.data['Mean'][i] and self.data['Signal_Line'][i] < self.data['MACD'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')
                
                elif self.use_sma_short and self.use_sma_long and self.use_bb:
                    if self.data['Close'][i] > self.data['Mean'][i] and self.data['Sma_short'][i] < self.data['Sma_long'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')
            
            # Manage Short Position
            
            elif self.position == 'short':
                
                # Stop Loss Condition for Short
                
                if self.data['Close'][i] >= self.buy_price * (1 + self.stop_loss / 100):
                    self.cover_price = self.data['Close'][i]
                    self.profit_loss = self.shares * (self.buy_price - self.cover_price)
                    self.account -= self.shares * self.cover_price  # Pay to cover the short position
                    self.pnl += self.profit_loss
                    self.shares = 0
                    self.position = None
                    print(f'Stop loss for short triggered, covered at {round(self.cover_price,2)}, account after covering: {round(self.account,2)}, loss: {round(self.profit_loss,2)}')

                # Regular Covering Condition for Short
                
                elif self.use_sma_short and self.use_sma_long and self.use_macd and self.use_bb:
                    if self.data['Sma_short'][i] > self.data['Sma_long'][i] and self.data['Close'][i] < self.data['Mean'][i] and self.data['Signal_Line'][i] > self.data['MACD'][i]:
                        self.cover_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.buy_price - self.cover_price)
                        self.account -= self.shares * self.cover_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Covered short at {self.cover_price}, account after covering: {self.account}, profit: {self.profit_loss}')
                
                elif self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] > self.data['Sma_long'][i]:
                        self.cover_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.buy_price - self.cover_price)
                        self.account -= self.shares * self.cover_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Covered short at {round(self.cover_price,2)}, account after covering: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')

                elif self.use_macd and self.use_sma_short and self.use_sma_long:
                    if self.data['Sma_short'][i] > self.data['Sma_long'][i] and self.data['Signal_Line'][i] > self.data['MACD'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')

                elif self.use_macd and self.use_bb:
                    if self.data['Close'][i] < self.data['Mean'][i] and self.data['Signal_Line'][i] > self.data['MACD'][i]:
                        self.cover_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.buy_price - self.cover_price)
                        self.account -= self.shares * self.cover_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Covered short at {self.cover_price}, account after covering: {self.account}, profit: {self.profit_loss}')
                elif self.use_sma_short and self.use_sma_long and self.use_bb:
                    if self.data['Close'][i] < self.data['Mean'][i] and self.data['Sma_short'][i] > self.data['Sma_long'][i]:
                        self.sell_price = self.data['Close'][i]
                        self.profit_loss = self.shares * (self.sell_price - self.buy_price)
                        self.account += self.shares * self.sell_price
                        self.pnl += self.profit_loss
                        self.shares = 0
                        self.position = None
                        print(f'Sold long at {round(self.sell_price,2)}, account after selling: {round(self.account,2)}, profit: {round(self.profit_loss,2)}')
        
            self.cumulative_pnl.append(self.pnl)

        print(f"Final Account Balance: {round(self.account,2)}")
        print(f"Total P&L: {round(self.pnl,2)}")
        print(f"Position status (Open: True, Closed: False): {self.position}")

    def run_backtest(self,start,end):
        
        self.start = start
        self.end = end
        
        """Runs the entire strategy backtest for the specified period."""
        
        self.get_data()
        self.backtesting()

    def test_on_new_data(self, start, end):
        self.start = start
        self.end = end
        
        """Tests the strategy on new data using the same parameters."""

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
    
    def plot_graph(self):
        
        plt.figure(figsize=(14, 8))
        
        # Plot the Close price
        plt.plot(self.data.index, self.data['Close'], label='Close', color='black', linewidth=1.5)
        
        # Plot the short-term moving average
        if 'Sma_short' in self.data.columns:
            plt.plot(self.data.index, self.data['Sma_short'], label=f'SMA {self.sma_short_period}', color='blue', linestyle='--')

        # Plot the long-term moving average
        if 'Sma_long' in self.data.columns:
            plt.plot(self.data.index, self.data['Sma_long'], label=f'SMA {self.sma_long_period}', color='red', linestyle='--')

        # Plot Bollinger Bands if available
        if 'Upper_Threshold' in self.data.columns and 'Lower_Threshold' in self.data.columns:
            plt.plot(self.data.index, self.data['Upper_Threshold'], label='Upper Bollinger Band', color='green', linestyle='--')
            plt.plot(self.data.index, self.data['Lower_Threshold'], label='Lower Bollinger Band', color='orange', linestyle='--')
            plt.fill_between(self.data.index, self.data['Lower_Threshold'], self.data['Upper_Threshold'], color='yellow', alpha=0.1)
        
        # Add titles and labels
        plt.title(f'{self.symbol} Price Chart with Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    


    def plot_pnl(self):
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.cumulative_pnl, label='Cumulative P&L')
        plt.title('Strategy Cumulative P&L Over Time')
        plt.xlabel('Time (trading steps)')
        plt.ylabel('Cumulative P&L ($)')
        plt.legend()
        plt.show()

        
    def plot_percentage_gain(self):
        
        initial_capital = 100000
        cumulative_returns = (np.array(self.cumulative_pnl) / initial_capital) * 100

        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns, label='Cumulative Returns (%)')
        plt.title('Strategy Cumulative Returns Over Time')
        plt.xlabel('Time (trading steps)')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True)
        plt.show()


  


# Execute the strategy by specifying the ticker symbol, start and end dates, and data interval. 
# You can customize these parameters to test different scenarios.
strategy_two = StrategyTwo('AAPL', '2010-01-01', '2024-01-01', '1d')

# Running the strategy over the entire specified period.
strategy_two.backtesting()

# Executing the backtest with the specified parameters.
strategy_two.run_backtest('2010-01-01', '2020-01-01')

# Plotting the strategy's Profit and Loss (P&L) over time.
strategy_two.plot_pnl()

# Testing the strategy on a new data period with the current configuration.
strategy_two.test_on_new_data('2020-01-01','2024-01-01')

# Plotting the strategy's Profit and Loss (P&L) over time.
strategy_two.plot_pnl()

