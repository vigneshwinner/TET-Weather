import yfinance as yf

# Define tickers for front-month futures
commodities = {
    'Brent': 'BZ=F',
    'Henry_Hub': 'NG=F',
    'RBOB': 'RB=F',
    'Copper': 'HG=F',
    'Corn': 'C=F'
}

latest_prices = {}

for name, ticker in commodities.items():
    # Download most recent 1-min bar (for today)
    data = yf.download(ticker, period='1d', interval='1m')
    # Get the last available close price
    last_quote = data['Close'][-1]
    latest_prices[name] = last_quote

print("Latest (near-real-time) prices for front-month futures:")
for name, price in latest_prices.items():
    print(f"{name}: {price}")
