import vectorbt as vbt
from itertools import product
import os
import logging
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List
import multiprocessing as mp
import concurrent.futures

# Suppress the OMP warning
warnings.filterwarnings('ignore', message=".*omp_set_nested routine deprecated.*")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from functions.helpers import wfo_rolling_split_params
from functions.plots import create_combined_parameter_sharpe_plot
from strategies.trend_following import process_strategy

class WalkForwardOptimizer:
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        param_ranges: dict,
        n_windows: int = 10,
        in_sample_percentage: float = 0.8,
        allow_shorting: bool = False,
        allow_leverage: bool = False,
        timeframe: str = '30m',
        fees: float = 0.0005,
        init_cash: float = 100000,
        auto_select: bool = True,
        trades_share_ratio_selection: bool = False,
        results_dir: str = 'results',
        n_jobs: int = None 
    ):
        """Initialize the Walk Forward Optimizer."""
        self.data = data
        self.allow_shorting = allow_shorting
        self.allow_leverage = allow_leverage
        
        # Fix parameter ranges duplication
        self.param_ranges = {}
        for key, values in param_ranges.items():
            if isinstance(values, range):
                self.param_ranges[key] = list(values)
            else:
                # Round and remove duplicates for float values
                unique_values = np.unique(np.round(values, 3))
                self.param_ranges[key] = sorted(unique_values)
        
        self.n_windows = n_windows
        self.in_sample_percentage = in_sample_percentage
        self.timeframe = timeframe
        self.fees = fees
        self.init_cash = init_cash
        self.auto_select = auto_select
        self.trades_share_ratio_selection = trades_share_ratio_selection
        self.results_dir = results_dir
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()

    def create_portfolio(self, df: pd.DataFrame) -> vbt.Portfolio:
        """Creates a portfolio using vectorbt based on signals in the dataframe."""
        # Shifting all signals and exits by 1 to avoid lookahead bias
        long_entries = (df['signal'] == 1).shift(1).astype(bool) 
        short_entries = (df['signal'] == -1).shift(1).astype(bool) 
        
        exit_long_signal = (df['exit'] == 1).shift(1).astype(bool) 
        exit_short_signal = (df['exit'] == -1).shift(1).astype(bool) 

        if self.allow_shorting:
            # Combined exits (including opposite entry signals)
            long_exits = exit_long_signal # Exit long on short entries
            short_exits = exit_short_signal  # Exit short on long entries
            
            if self.allow_leverage:
                leverage = df['leverage'].shift(1) # shift to avoid lookahead bias
                return vbt.Portfolio.from_signals(
                    df['Open'], # Open price used for entry
                    entries=long_entries,
                    exits=long_exits,
                    short_entries=short_entries,
                    short_exits=short_exits,
                    freq=self.timeframe,
                    fees=self.fees,
                    init_cash=self.init_cash,
                    accumulate=False,
                    #upon_opposite_entry='ignore',
                    size=leverage,
                    size_type='percent'
                )
            else:
                return vbt.Portfolio.from_signals(
                    df['Open'],
                    entries=long_entries,
                    exits=long_exits,
                    short_entries=short_entries,
                    short_exits=short_exits,
                    freq=self.timeframe,
                    fees=self.fees,
                    init_cash=self.init_cash,
                    accumulate=False,
                    #upon_opposite_entry='ignore'
                )
        else:
            if self.allow_leverage:
                leverage = df['leverage'].shift(1)
                return vbt.Portfolio.from_signals(
                    df['Open'],
                    entries=long_entries,
                    exits=exit_long_signal,
                    freq=self.timeframe,
                    fees=self.fees,
                    init_cash=self.init_cash,
                    accumulate=False,
                    #upon_opposite_entry='ignore',
                    size=leverage,
                    size_type='percent'
                )
            else:
                return vbt.Portfolio.from_signals(
                    df['Open'],
                    entries=long_entries,
                    exits=exit_long_signal,
                    freq=self.timeframe,
                    fees=self.fees,
                    init_cash=self.init_cash,
                    accumulate=False,
                    #upon_opposite_entry='ignore'
                )

    def process_param_combination(
        self,
        df: pd.DataFrame,
        params: tuple,
        pair_key: str
    ) -> dict:
        """Process a single parameter combination and return the results."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Use param_ranges keys as parameter names
        param_dict = dict(zip(self.param_ranges.keys(), params))
        
        df_copy = df.copy()
        df_copy.index = df.index
        
        # Create single pair dictionary for detect_twap_signals
        single_pair_dict = {pair_key: df_copy}
        
        processed_data = process_strategy(
            single_pair_dict,
            **param_dict
        )[pair_key]
        
        pf = self.create_portfolio(processed_data)
        stats = pf.stats()
        
        return {
            'params': param_dict,
            'sharpe_ratio': stats['Sharpe Ratio'],
            'sortino_ratio': stats['Sortino Ratio'],
            'calmar_ratio': stats['Calmar Ratio'],
            'total_return': stats['Total Return [%]'] / 100,
            'max_drawdown': stats['Max Drawdown [%]'] / 100,
            'total_trades': stats['Total Trades'],
            'stats': stats,
            'processed_df': processed_data
        }

    def process_param_combinations_parallel(self, args):
        """Helper function to process parameter combinations in parallel."""
        params, in_sample, pair_key = args
        return self.process_param_combination(in_sample, params, pair_key)

    def process_window(self, pair_key: str, window_idx: int, in_sample: pd.DataFrame, out_sample: pd.DataFrame) -> dict:
        """Process a single window of the walk-forward optimization."""
        try:
            # 1. Validate input data
            if not isinstance(in_sample.index, pd.DatetimeIndex):
                in_sample.index = pd.to_datetime(in_sample.index)
            if not isinstance(out_sample.index, pd.DatetimeIndex):
                out_sample.index = pd.to_datetime(out_sample.index)
            
            # 2. Generate parameter combinations
            param_combinations = list(product(*self.param_ranges.values()))
            args_list = [(params, in_sample, pair_key) for params in param_combinations]
            
            # 3. Process combinations in parallel using a suitable chunksize 
            chunksize = max(1, len(args_list) // (self.n_jobs * 4))
            with mp.Pool(processes=self.n_jobs) as pool:
                results = []
                for i, result in enumerate(pool.imap_unordered(self.process_param_combinations_parallel, args_list, chunksize=chunksize), 1):
                    try:
                        if i % 10 == 0:
                            logger.info(f"Processing combination {i}/{len(param_combinations)}")
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in combination {i}: {str(e)}")
            
            if not results:
                raise ValueError(f"No valid results for window {window_idx+1}")

            # 4. Create results directory and plot
            subfolder = os.path.join(self.results_dir, pair_key)
            os.makedirs(subfolder, exist_ok=True)
            try:
                plot_filename = create_combined_parameter_sharpe_plot(results, window_idx, subfolder)
                logger.info(f"Parameter-Sharpe plot saved as {plot_filename}")
            except Exception as e:
                logger.error(f"Error creating plot: {str(e)}")
                plot_filename = None

            # 5. Process results and filter
            results_df = pd.DataFrame(results)
            results_df = results_df[results_df['total_trades'] > 0].sort_values(
                ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return'],
                ascending=[False, False, False, False]
            ).reset_index(drop=True)

            if len(results_df) == 0:
                raise ValueError("No valid parameter combinations found")

            # 6. Handle parameter selection
            if self.auto_select:
                if self.trades_share_ratio_selection:
                    # Compute trade_shares using total_return clipped to 0 to avoid negative contributions
                    results_df['trade_shares'] = results_df['total_trades'] * results_df['sharpe_ratio'] * results_df['total_return'].clip(lower=0)
                    best_row = results_df.sort_values('trade_shares', ascending=False).iloc[0]
                    chosen_params = best_row['params']
                    logger.info(f"Auto-selected parameters (via trades_share_ratio) for {pair_key}: {chosen_params}")
                else:
                    chosen_params = results_df.iloc[0]['params']
                    logger.info(f"Auto-selected parameters for {pair_key}: {chosen_params}")
            else:
                print(f"\nTop 5 parameter combinations by Sharpe ratio for window {window_idx+1}:")
                for idx, row in results_df.head().iterrows():
                    print(f"\nRank {idx + 1}:")
                    print(f"Parameters: {row['params']}")
                    print(f"Sharpe Ratio: {row['sharpe_ratio']:.4f}")
                    print(f"Total Return: {row['total_return']:.2%}")
                    print(f"Total Trades: {row['total_trades']}")

                chosen_params = {}
                print("\nPlease enter your chosen values for each parameter:")
                
                for param_name, param_range in self.param_ranges.items():
                    while True:
                        print(f"\nAvailable values for {param_name}: {list(param_range)}")
                        try:
                            value = input(f"Enter value for {param_name}: ")
                            value = int(value) if isinstance(param_range, range) else type(param_range[0])(value)
                            if value in param_range:
                                chosen_params[param_name] = value
                                break
                            else:
                                print(f"Error: Value must be one of the available options")
                        except ValueError:
                            print("Error: Invalid input. Please try again.")
                
                logger.info(f"Manually selected parameters for {pair_key}: {chosen_params}")

            # 7. Process out-of-sample data
            try:
                out_result = self.process_param_combination(out_sample, tuple(chosen_params.values()), pair_key)
            except Exception as e:
                logger.error(f"Error processing out-of-sample data: {str(e)}")
                raise

            # 8. Return results
            return {
                'window': window_idx,
                'in_sample_results': results_df.head(5).to_dict('records'),
                'chosen_params': chosen_params,
                'out_sample_result': out_result,
                'plot_filename': plot_filename,
                'out_sample_df': out_result['processed_df']
            }
            
        except Exception as e:
            logger.error(f"Error processing window {window_idx+1} for pair {pair_key}: {str(e)}")
            raise
    
    def optimize(self) -> Dict[str, List[dict]]:
        """Run the walk-forward optimization process."""
        results = {}
        
        for pair_key, df in self.data.items():
            print(f"\nProcessing pair: {pair_key}")
            
            n, window_len, set_lens = wfo_rolling_split_params(
                len(df), self.in_sample_percentage, self.n_windows)
            
            (in_samples, in_indexes), (out_samples, out_indexes) = df.vbt.rolling_split(
                n=n,
                window_len=window_len,
                set_lens=set_lens,
            )
            
            pair_results = []
            for i in range(min(len(in_samples), self.n_windows)):
                try:
                    result = self.process_window(pair_key, i, in_samples[i], out_samples[i])
                    pair_results.append(result)
                    
                    # Log results after window completion
                    logger.info(f"\n{pair_key} Window {i+1} Results:")
                    logger.info(f"In-sample top Sharpe: {result['in_sample_results'][0]['sharpe_ratio']:.4f}")
                    logger.info(f"Chosen parameters: {result['chosen_params']}")
                    logger.info(f"Out-of-sample Sharpe: {result['out_sample_result']['sharpe_ratio']:.4f}")
                except Exception as e:
                    logger.error(f"Error processing window {i+1} for pair {pair_key}: {e}")
                    continue
            
            results[pair_key] = pair_results
            
            # Save results for this pair
            try:
                self.save_pair_results(pair_key, pair_results)
            except Exception as e:
                logger.error(f"Error saving results for pair {pair_key}: {e}")
        
        return results

    def save_pair_results(self, pair_key: str, results: List[dict]):
        """Save results for a single pair to files."""
        subfolder = os.path.join(self.results_dir, pair_key)
        os.makedirs(subfolder, exist_ok=True)
        
        # Define output files
        output_file_md = os.path.join(subfolder, 'wfo_results.md')
        output_file_csv = os.path.join(subfolder, 'wfo_results.csv')
        output_file_oos = os.path.join(subfolder, f'{pair_key}_combined_oos.csv')
        
        # Combine and save out-of-sample results with identifiers
        combined_oos_dfs = []
        for r in results:
            df = r['out_sample_df'].copy()
            df['pair'] = pair_key
            df['window'] = r['window'] + 1
            combined_oos_dfs.append(df)
        
        combined_oos_df = pd.concat(combined_oos_dfs)
        # Define saving tasks as functions
        def save_oos():
            combined_oos_df.to_csv(output_file_oos)
        
        total_oos_stats = self.create_portfolio(combined_oos_df).stats()
        
        def save_md():
            with open(output_file_md, 'w') as f:
                f.write(f"# Walk-Forward Optimization Results for {pair_key}\n\n")
                for window_result in results:
                    f.write(f"## Window {window_result['window'] + 1}\n\n")
                    f.write("### Chosen Parameters\n")
                    for param, value in window_result['chosen_params'].items():
                        f.write(f"- {param}: {value}\n")
                    f.write("\n")
                    f.write("### Out-of-Sample Results\n")
                    stats = window_result['out_sample_result']['stats']
                    for metric in ['Total Return [%]', 'Max Drawdown [%]', 'Sharpe Ratio',
                                   'Sortino Ratio', 'Win Rate [%]', 'Total Trades']:
                        f.write(f"{metric}: {stats[metric]}\n")
                    f.write("\n")
                f.write("## Total Out-of-Sample Statistics\n\n")
                for metric in ['Total Return [%]', 'Max Drawdown [%]', 'Sharpe Ratio',
                               'Sortino Ratio', 'Win Rate [%]', 'Total Trades']:
                    f.write(f"{metric}: {total_oos_stats[metric]}\n")
        
        # Prepare window results CSV data
        window_results = []
        for window_result in results:
            window_data = {
                'pair': pair_key,
                'window': window_result['window'] + 1,
                **window_result['chosen_params'],
                **{f'OutOfSample_{k}': v for k, v in window_result['out_sample_result']['stats'].items()}
            }
            window_results.append(window_data)
        
        def save_csv():
            pd.DataFrame(window_results).to_csv(output_file_csv, index=False)
        
        # Run saving tasks concurrently to speed up the overall saving process
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(save_oos))
            futures.append(executor.submit(save_md))
            futures.append(executor.submit(save_csv))
            concurrent.futures.wait(futures)
        
        logger.info(f"Results saved for {pair_key} in directory: {subfolder}")