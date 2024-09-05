# Trading Strategies

This repository contains three different trading strategies implemented in Python. Each strategy uses different methodologies and indicators to make trading decisions. The following strategies are included:

## Strategies Overview

### Strategy One

**Description:**
Strategy One is designed to identify buying opportunities based on consecutive days of falling prices. The strategy buys on the fourth day after a specified number of consecutive days of price decline. 

**Key Features:**
- **Configurable Falling Days:** You can set the number of days the price should fall consecutively before making a buy decision. This can be adjusted to 2, 4, 5, or any number of days as needed.
- **Adjustable Stop Loss and Take Profit:** The stop loss and take profit percentages can be easily modified to suit different risk appetites and trading styles. For instance, you can set a stop loss between 2% and 4% and adjust the take profit accordingly.

**Usage:**
To use this strategy, modify the stop loss, take profit, and falling days parameters in the `strategy_one.py` file.

### Strategy Two

**Description:**
Strategy Two incorporates several well-known technical indicators, including Simple Moving Averages (SMA), Bollinger Bands (BB), and Moving Average Convergence Divergence (MACD). This strategy allows you to enable or disable the use of these indicators.

**Key Features:**
- **Configurable Indicators:** You can enable or disable SMA, BB, and MACD based on your preference. However, ensure that both short and long SMAs are either enabled or disabled together to maintain consistency in the strategy.
- **Flexible Indicator Settings:** Customize the parameters for each indicator as needed to optimize performance.

**Usage:**
Modify the `strategy_two.py` file to enable or disable the desired indicators and adjust their parameters.

### Best SMA Optimization

**Description:**
The `best_sma.py` file provides functionality to test and optimize the performance of different SMA configurations. This allows you to find the best combination of short and long SMAs that yield the highest returns.

**Key Features:**
- **SMA Performance Testing:** Evaluate various SMA settings to determine which combination of short and long SMAs produces the best results.
- **Integration with Strategy Two:** The optimized SMA parameters can be used in Strategy Two for improved performance.

**Usage:**
Run the `best_sma.py` file to test different SMA settings and use the results to configure Strategy Two.

## Getting Started

