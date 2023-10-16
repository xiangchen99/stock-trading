import yfinance as yf  # Yahoo Finance library for downloading stock data
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

# Function to download historical stock data
def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Function to calculate daily returns from stock prices
def calculate_daily_returns(stock_data):
    # Implement this method to calculate daily returns
    closing_prices = stock_data['Close']
    daily_returns = closing_prices.pct_change()
    return daily_returns

# Function to calculate moving averages
def calculate_moving_averages(stock_data, short_window, long_window):
    # Implement this method to calculate moving averages
    closing_prices = stock_data['Close']
    short_rolling = closing_prices.rolling(window=short_window).mean()
    long_rolling = closing_prices.rolling(window=long_window).mean()
    return short_rolling, long_rolling

# Function to calculate historical volatility
def calculate_volatility(daily_returns, window):
    # Implement this method to calculate volatility
    volatility = daily_returns.rolling(window=window).std()
    return volatility

# Function to visualize stock prices over time
def visualize_stock_prices(stock_data, title):
    # Implement this method to create a line chart of stock prices
    closing_prices = stock_data['Close']
    plt.plot(closing_prices)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Function to visualize candlestick chart
def visualize_candlestick_chart(stock_data, title):
    # Implement this method to create a candlestick chart
    mpf.plot(stock_data, type='candle', title=title)

# Function to visualize daily returns
def visualize_daily_returns(daily_returns, title):
    # Implement this method to create a histogram or density plot
    plt.plot(daily_returns)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Function to calculate the Sharpe ratio
def calculate_sharpe_ratio(daily_returns, risk_free_rate):
    # Implement this method to calculate the Sharpe ratio
    arr = np.array(daily_returns)
    mean = np.mean(arr)
    std = np.std(arr)
    return (mean - risk_free_rate) / std * np.sqrt(252)

# Main function for testing and visualization
def main():
    stock_symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    stock_data = download_stock_data(stock_symbol, start_date, end_date)
    
    # Use the methods you implement for data analysis and visualization

    # Example usage:
    daily_returns = calculate_daily_returns(stock_data)
    visualize_stock_prices(stock_data, f"{stock_symbol} Stock Prices")
    visualize_candlestick_chart(stock_data, f"{stock_symbol} Candlestick Chart")
    visualize_daily_returns(daily_returns, f"{stock_symbol} Daily Returns")
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.03  # Example risk-free rate
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    print(f"Sharpe Ratio: {sharpe_ratio}")

if __name__ == "__main__":
    main()
