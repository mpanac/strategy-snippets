
# Function to split the data into windows for walk-forward optimization
# The returned values are then used in vectorBt to create the windows

def wfo_rolling_split_params(total_candles, insample_percentage, n):
    # Calculate the window length using the formula derived
    window_len = total_candles / (1 + (n - 1) * (1 - insample_percentage))

    # Set lengths tuple (just the in-sample percentage as per your request)
    set_lens = (insample_percentage,)
    
    # Return the number of windows, window length, and set lengths
    return int(n), int(window_len), set_lens