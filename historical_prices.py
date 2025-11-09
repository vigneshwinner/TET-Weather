import yfinance as yf

# Commodity tickers
commodities = {
    'Brent': 'BZ=F',
    'Henry_Hub': 'NG=F',
    'RBOB': 'RB=F',
    'Copper': 'HG=F',
    'Corn': 'C=F'
}

for name, ticker in commodities.items():
    print(f"\n{name} ({ticker}) - Daily prices for last 3 years:")
    data = yf.download(ticker, period='3y', interval='1d')
    if not data.empty:
        # Show just last 5 rows for a preview (remove .tail(5) for full listing)
        print(data[['Close']].tail(5))
        # Optionally: Save to CSV for each commodity
        data.to_csv(f"{name}_3yr.csv")
    else:
        print("No price data found!")