import pandas as pd
import sys
from datetime import datetime, timedelta

def get_local_data(symbol):
#    file_path = "market_data_full.parquet"
    file_path = "market_data_n_ind.parquet"
#    file_path = "market_data_sp500.parquet"
#    file_path = "market_data_sp500_n_ind.parquet"
    
    try:
        # 1. Load the Parquet file
        # Parquet is fast enough to load the whole thing, 
        # but we filter it immediately.
        df = pd.read_parquet(file_path)
        
        # 2. Filter for the specific symbol
        # We use .upper() to ensure case-insensitivity
        df_ticker = df[df['Ticker'] == symbol.upper()].copy()
        
        if df_ticker.empty:
            print(f"No data found for symbol: {symbol}")
            return

        # 3. Ensure Date is datetime objects and sorted
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
        df_ticker.sort_values('Date', inplace=True)

        # 4. Get the last 30 trading days
        last_30_days = df_ticker.tail(30)

        # 5. Format output for readability
        print(f"\nLast 30 Days of Candle Data for {symbol.upper()}:")
        print("-" * 80)
        # Formatting to 2 decimal places for price and 0 for volume
        pd.options.display.float_format = '{:.2f}'.format
        
        # Displaying the most important columns
#        print(last_30_days[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string(index=False))
        print(last_30_days[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'EMA_12']].to_string(index=False))
        print("-" * 80)
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run your download script first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Check if a ticker was provided via command line
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <TICKER>")
        print("Example: python get_candles.py NVDA")
    else:
        ticker_input = sys.argv[1]
        get_local_data(ticker_input)