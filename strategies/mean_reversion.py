import numpy as np
import pandas as pd
from numba import njit


@njit
def _calculate_slope_arr(prices, window):
    """Calculate slopes for entire array using Numba"""
    n = len(prices)
    slopes = np.zeros(n)
    x = np.arange(window)
    
    for i in range(window - 1, n):
        y = prices[i-window+1:i+1]
        x_mean = (window - 1) / 2
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slopes[i] = numerator / denominator if denominator != 0 else 0
    
    return slopes

def process_strategy(dfs, regression_window=20, bb_mult=2.0, vol_threshold=1.5):

    for key in dfs:
        df = dfs[key].copy()

        def find_regimes(df):
    
            slopes = _calculate_slope_arr(df['Close'].values, regression_window)
            df['Slope'] = slopes
            
            slope_std = pd.Series(slopes).rolling(regression_window).std().values
            df['Regime'] = 'Trend'
            df.loc[abs(df['Slope']) < slope_std, 'Regime'] = 'Range'
            
            df['MA'] = df['Close'].rolling(window=regression_window).mean()
            df['STD'] = df['Close'].rolling(window=regression_window).std()
            df['Upper_Band'] = df['MA'] + (df['STD'] * bb_mult)
            df['Lower_Band'] = df['MA'] - (df['STD'] * bb_mult)
            
            # Volatility
            df['ATR'] = calc_atr(df, regression_window)
            
            # Volume Rate calculation
            df['Vol_MA'] = df['Volume'].rolling(window=regression_window).mean()
            df['Vol_Rate'] = df['Volume'] / df['Vol_MA']
            
            df.dropna(inplace=True)

            return df

        def calc_atr(df, window):
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=window).mean()

        def generate_signals(df):
            df['signal'] = 0
            
            range_mask = (df['Regime'] == 'Range')
            
            long_condition = (
                (df['Close'] < df['Lower_Band']) & 
                range_mask &
                (df['Vol_Rate'] > vol_threshold)
            )
            
            short_condition = (
                (df['Close'] > df['Upper_Band']) & 
                range_mask &
                (df['Vol_Rate'] > vol_threshold)
            )
            
            df.loc[long_condition, 'signal'] = 1
            df.loc[short_condition, 'signal'] = -1
            
            return df

        def generate_exits(df):
            df['exit'] = 0
            
            # exit conditions:
            # 1. Price crosses back through moving average
            # 2. Volatility spike (ATR increases significantly)
            # 3. Regime changes
            
            long_exit = (
                ((df['Close'] > df['MA'])) | 
                (df['ATR'] > df['ATR'].rolling(regression_window).mean() * vol_threshold)
            )
            
            short_exit = (
                ((df['Close'] < df['MA'])) | 
                (df['ATR'] > df['ATR'].rolling(regression_window).mean() * vol_threshold)
            )
            
            regime_change = df['Regime'] != df['Regime'].shift(1)
            
            df.loc[long_exit, 'exit'] = 1
            df.loc[short_exit, 'exit'] = -1
            df.loc[regime_change, 'exit'] = df.loc[regime_change, 'signal']
            
            return df

        # Process data
        df = find_regimes(df.copy())
        df = generate_signals(df)
        df = generate_exits(df)

        # Clean up
        df = df.drop(columns=['MA', 'STD', 'Slope', 'Upper_Band', 'Lower_Band', 
                            'Vol_MA', 'Vol_Rate', 'ATR', 'ROPDMA'], errors='ignore')
    
    dfs[key] = df
    
    return dfs
