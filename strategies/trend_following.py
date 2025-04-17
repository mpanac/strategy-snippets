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

def process_strategy(dfs, regression_window=20, trend_strength=2.0, vol_threshold=1.5):
    for key in dfs:
        df = dfs[key].copy()

        def find_regimes(df):
            # Keep the original regime detection logic
            slopes = _calculate_slope_arr(df['Close'].values, regression_window)
            df['Slope'] = slopes
            
            slope_std = pd.Series(slopes).rolling(regression_window).std().values
            df['Regime'] = 'Trend'
            df.loc[abs(df['Slope']) < slope_std, 'Regime'] = 'Range'

            # Add Volume Rate calculation
            df['Vol_MA'] = df['Volume'].rolling(window=regression_window).mean()
            df['Vol_Rate'] = df['Volume'] / df['Vol_MA']
            
            df.dropna(inplace=True)

            return df
        
        def generate_signals(df):
            df['signal'] = 0
            
            trend_mask = (df['Regime'] == 'Trend')
            
            # Calculate acceleration of price movement
            df['Price_Chg'] = df['Close'].diff()
            df['Price_Acc'] = df['Price_Chg'].diff()
            
            # Calculate slope momentum
            df['Slope_Mom'] = df['Slope'].diff()
            
            # Long signals: Strong upward trend with acceleration
            long_condition = (
                trend_mask &
                (df['Slope'] > df['Slope'].rolling(regression_window).std() * trend_strength) &
                (df['Price_Acc'] > 0) &
                (df['Slope_Mom'] > 0) &
                (df['Vol_Rate'] > vol_threshold)
            )
            
            # Short signals: Strong downward trend with acceleration
            short_condition = (
                trend_mask &
                (df['Slope'] < -df['Slope'].rolling(regression_window).std() * trend_strength) &
                (df['Price_Acc'] < 0) &
                (df['Slope_Mom'] < 0) &
                (df['Vol_Rate'] > vol_threshold)
            )
            
            df.loc[long_condition, 'signal'] = 1
            df.loc[short_condition, 'signal'] = -1
            
            return df

        def generate_exits(df):
            df['exit'] = 0
            
            # Calculate trend strength decay
            df['Slope_Decay'] = df['Slope'].pct_change()
            
            # exit conditions:
            # 1. Trend strength significantly weakens
            # 2. Price acceleration reverses
            # 3. Slope momentum reverses
            # 4. Regime changes to range-bound
            
            long_exit = (
                ((df['Slope_Decay'] < -trend_strength) |
                (df['Price_Acc'] < 0) |
                (df['Slope_Mom'] < 0))
            )
            
            short_exit = (
                ((df['Slope_Decay'] > trend_strength) |
                (df['Price_Acc'] > 0) |
                (df['Slope_Mom'] > 0))
            )
            
            regime_change = (df['Regime'] != 'Trend') & (df['Regime'].shift(1) == 'Trend')
            
            df.loc[long_exit, 'exit'] = 1
            df.loc[short_exit, 'exit'] = -1
            df.loc[regime_change, 'exit'] = df.loc[regime_change, 'signal']
            
            return df

        # Process data
        df = find_regimes(df.copy())
        df = generate_signals(df)
        df = generate_exits(df)

        # Clean up
        df = df.drop(columns=['MA', 'Vol_MA', 'Price_Chg'], errors='ignore')

    dfs[key] = df
    
    return dfs