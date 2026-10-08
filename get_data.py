import argparse
import pandas as pd
import yfinance as yf
import requests
import io # Required for Pandas 2.0+ compatibility

# Define headers to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_sp500_tickers():
    """Scrapes the S&P 500 list from Wikipedia using a fake User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # 1. Fetch content with headers
    r = requests.get(url, headers=HEADERS)
    
    # 2. Parse HTML using io.StringIO to avoid Pandas warnings
    tables = pd.read_html(io.StringIO(r.text))
    df = tables[0]
    
    # Clean tickers
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"Loaded {len(tickers)} S&P 500 tickers.")
    return tickers

def get_nasdaq_tickers():
    """Fetches NASDAQ tickers from FTP, with a Wikipedia fallback."""
    try:
        # FTP usually doesn't require headers, but it might be slow
        print("Attempting to fetch NASDAQ list via FTP...")
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df = pd.read_csv(url, sep="|")
        df = df[df['Test Issue'] == 'N']
        tickers = df['Symbol'].dropna().tolist()
        if isinstance(tickers[-1], str) and "File Creation Time" in tickers[-1]:
            tickers.pop()
        print(f"Loaded {len(tickers)} NASDAQ tickers via FTP.")
        return tickers
        
    except Exception as e:
        print(f"FTP failed ({e}). Falling back to Wikipedia NASDAQ-100...")
        
        # Fallback to Wikipedia (Needs Headers too!)
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        r = requests.get(url, headers=HEADERS)
        
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[4] # Table index varies, usually 4 for NASDAQ 100
        print(f"Loaded {len(df)} NASDAQ tickers via Wikipedia.")
        return df['Ticker'].tolist()

# ==========================================
# 2. CORE: Batch Download Function
# ==========================================
def download_and_process(tickers, chunk_size=100):
    all_data = []
    
    # Remove duplicates
    unique_tickers = list(set(tickers))
    total = len(unique_tickers)
    
    print(f"Starting download for {total} unique tickers...")

    for i in range(0, total, chunk_size):
        chunk = unique_tickers[i:i + chunk_size]
        print(f"Processing batch {i}/{total}...")
        
        try:
            # Download batch
            # group_by='ticker' ensures we get a clean MultiIndex
            # auto_adjust=True fixes splits/dividends automatically
            data = yf.download(chunk, period="5y", group_by='ticker', auto_adjust=True, threads=True)
            
            if data.empty:
                continue

            # data is "Wide" (Columns are Tickers). We need "Long" (Rows are Tickers).
            # The structure from yfinance is MultiIndex: (Ticker, PriceType)
            # We stack it to move Ticker from Column to Index
            data = data.stack(level=0)
            
            # Reset index so 'Date' and 'Ticker' become regular columns
            data.index.names = ['Date', 'Ticker']
            data.reset_index(inplace=True)
            
            all_data.append(data)
            
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")

    if not all_data:
        return pd.DataFrame()

    print("Concatenating all batches...")
    final_df = pd.concat(all_data)
    return final_df

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Get Lists
    # Configure the argument parser
    parser = argparse.ArgumentParser(description="Download financial market data.")
    parser.add_argument(
        '--market', 
        choices=['sp500', 'all'], 
        default='sp500', 
        help="Select market universe: 'sp500' (default) or 'all' (S&P 500 + NASDAQ)"
    )
    args = parser.parse_args()

    # Select the tickers based on user argument
    if args.market == 'all':
        sp500 = get_sp500_tickers()
        nasdaq = get_nasdaq_tickers()
        full_ticker_list = list(set(sp500 + nasdaq))
        filename = "market_data_full.parquet"
    else:
        full_ticker_list = get_sp500_tickers()
        filename = "market_data_sp500.parquet"
    
    # 2. Download
    # Note: Downloading 3000+ stocks may take 5-10 minutes depending on connection
    df_market = download_and_process(full_ticker_list, chunk_size=100)
    
    # 3. Clean up
    # Ensure Date is sorted for Time Series Analysis
    df_market.sort_values(['Ticker', 'Date'], inplace=True)
    
    # 4. Save
    print(f"Saving to {filename}...")
    df_market.to_parquet(filename, engine='pyarrow', compression='snappy')
    
    print("Done! Data sample:")
    print(df_market.head())
