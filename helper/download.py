import ccxt
import pandas as pd
from datetime import datetime
import time
import os

def fetch_ohlcv_data(symbol, timeframe='1m', start_date=None, market_type='spot'):
    # Create data directory structure
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    market_dir = os.path.join(base_dir, market_type)
    symbol_dir = os.path.join(market_dir, symbol.replace('/', ''))
    
    # Create directories if they don't exist
    for directory in [base_dir, market_dir, symbol_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    print(f"\nInitializing download for {symbol} {market_type} market data...")
    
    
    # Initialize exchange
    if market_type == 'spot':
        exchange = ccxt.binance()
    else:
        exchange = ccxt.binanceusdm()
    
    since = exchange.parse8601(start_date)
    current_time = exchange.milliseconds()
    
    all_ohlcv = []
    total_candles = 0
    start_time = time.time()
    
    print(f"Start date: {start_date}")
    print(f"Downloading {timeframe} candles...\n")
    
    while since < current_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if len(ohlcv) == 0:
                break
                
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            total_candles += len(ohlcv)
            
            if total_candles % 10000 == 0:
                current_date = datetime.fromtimestamp(since/1000).strftime('%Y-%m-%d %H:%M')
                print(f"Downloaded {total_candles} candles... Current date: {current_date}")
            
            exchange.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    clean_symbol = symbol.replace('/', '')
    start_date_clean = start_date.split()[0]
    end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
    filename = f"{clean_symbol}_{market_type}_{start_date_clean}_to_{end_date}_{timeframe}.csv"
    filepath = os.path.join(symbol_dir, filename)
    
    # Save to CSV with full path
    df.to_csv(filepath, index=False)
    print(f"\nDownload completed!")
    print(f"Total candles: {total_candles}")
    print(f"Data saved to: {filepath}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    
    return df

if __name__ == "__main__":
    symbol = "BTC/USDT"
    start_date = "2020-01-01 00:00:00"
    market_type = input("Enter market type (spot/futures): ").lower()
    
    while market_type not in ['spot', 'futures']:
        print("Invalid market type. Please enter 'spot' or 'futures'")
        market_type = input("Enter market type (spot/futures): ").lower()
    
    df = fetch_ohlcv_data(symbol, start_date=start_date, market_type=market_type)