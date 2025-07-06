import yfinance as yf

# Example: Fetch data for Apple (AAPL)
ticker = yf.Ticker("BAC")

# Get historical market data
data = ticker.history(period="5d")

# Print or visualize
print(data)
