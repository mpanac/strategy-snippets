import vectorbt as vbt
from itertools import product
from tqdm import tqdm
import multiprocessing
import os
import logging
import sys
from functools import partial
import pandas as pd
import warnings

# Suppress the OMP warning
warnings.filterwarnings('ignore', message=".*omp_set_nested routine deprecated.*")

# Add the parent directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from helper_functions.plot import create_combined_parameter_sharpe_plot
from strategies.mean_reversion_regression import process_strategy

class WFOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                
    def optimize(self, in_ohlcv, out_ohlcv, in_indexes, out_indexes, 
                param_ranges, freq, fees, slippage, init_cash, auto_select, 
                subfolder):
        """
        Main optimization method that runs the walk-forward optimization process.
        
        Args:
            in_ohlcv (List[pd.DataFrame]): List of in-sample OHLCV dataframes
            out_ohlcv (List[pd.DataFrame]): List of out-of-sample OHLCV dataframes
            param_ranges (dict): Dictionary of parameter ranges to test
            freq (str): Timeframe frequency
            fees (float): Trading fees
            slippage (float): Slippage factor
            init_cash (float): Initial capital
            auto_select (bool): Whether to automatically select parameters
            subfolder (str): Output directory for results
        """
        num_cores = multiprocessing.cpu_count()
        self.logger.info(f"Starting optimization with {num_cores} cores")
        
        # Pre-calculate parameter combinations
        param_names = list(param_ranges.keys())
        param_combinations = list(product(*param_ranges.values()))
        
        results = []
        with multiprocessing.Pool(processes=num_cores) as pool:
            for window_idx in range(len(in_indexes)):
                try:
                    result = self._process_optimization_window(
                        window_idx=window_idx,
                        in_data=in_ohlcv[window_idx],
                        out_data=out_ohlcv[window_idx],
                        param_names=param_names,
                        param_combinations=param_combinations,
                        param_ranges=param_ranges,
                        pool=pool,
                        config={
                            'freq': freq,
                            'fees': fees,
                            'slippage': slippage,
                            'init_cash': init_cash,
                            'auto_select': auto_select,
                            'subfolder': subfolder
                        }
                    )
                    results.append(result)
                    self._log_window_results(window_idx, result)
                except Exception as e:
                    self.logger.error(f"Error processing window {window_idx}: {str(e)}")
                    raise
        
        self._save_results(results, 
                          os.path.join(subfolder, 'wfo_results.md'),
                          os.path.join(subfolder, 'wfo_results.csv'),
                          freq, fees, slippage, init_cash)
        return results

    def _process_optimization_window(self, window_idx, in_data, out_data, param_names,
                                   param_combinations, param_ranges, pool, config):
        """Process a single optimization window using parallel processing."""
        # Ensure data has datetime index
        in_data = self._ensure_datetime_index(in_data)
        
        # Create parallel processing function
        process_func = partial(
            self._process_param_combination,
            in_data,
            param_names=param_names,
            freq=config['freq'],
            fees=config['fees'],
            slippage=config['slippage'],
            init_cash=config['init_cash']
        )
        
        # Process parameter combinations in parallel
        results = list(tqdm(
            pool.imap(process_func, param_combinations),
            total=len(param_combinations),
            desc=f"Window {window_idx + 1}"
        ))
        
        # Process results
        results_df = self._process_results_df(results)
        
        # Create visualization
        plot_filename = create_combined_parameter_sharpe_plot(
            results, window_idx, config['subfolder']
        )
        
        # Select parameters
        chosen_params = (self._auto_select_params(results_df) 
                        if config['auto_select'] 
                        else self._manual_select_params(param_ranges))
        
        # Process out-of-sample data
        out_result = self._process_param_combination(
            out_data,
            tuple(chosen_params.values()),
            param_names,
            config['freq'],
            config['fees'],
            config['slippage'],
            config['init_cash']
        )
        
        return {
            'window': window_idx,
            'in_sample_results': results_df.head(5).to_dict('records'),
            'chosen_params': chosen_params,
            'out_sample_result': out_result,
            'plot_filename': plot_filename,
            'out_sample_df': out_result['processed_df']
        }

    @staticmethod
    def _ensure_datetime_index(df):
        """Ensure dataframe has a datetime index."""
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Timestamp' in df.columns:
                df.set_index('Timestamp', inplace=True)
            df.index = pd.to_datetime(df.index)
        return df

    def _process_results_df(self, results):
        """Process and sort results dataframe."""
        df = pd.DataFrame(results)
        return (df[df['total_trades'] > 0]
                .sort_values(['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return'],
                           ascending=[False, False, False, False])
                .reset_index(drop=True))

    def _log_window_results(self, window_idx, result):
        """Log the results of a window optimization."""
        self.logger.info(f"\nWindow {window_idx + 1} Results:")
        self.logger.info(f"In-sample top Sharpe: {result['in_sample_results'][0]['sharpe_ratio']:.4f}")
        self.logger.info(f"Chosen parameters: {result['chosen_params']}")
        self.logger.info(f"Out-of-sample Sharpe: {result['out_sample_result']['sharpe_ratio']:.4f}")

    def _process_window(self, i, in_ohlcv_i, out_ohlcv_i, param_ranges, freq, 
                       fees, slippage, init_cash, auto_select, subfolder, pool):
        """Process a single window of the walk-forward optimization."""
        if not isinstance(in_ohlcv_i.index, pd.DatetimeIndex):
            in_ohlcv_i.index = pd.to_datetime(in_ohlcv_i.index)
        
        param_combinations = list(product(*param_ranges.values()))
        
        # Create a list of parameter names
        param_names = list(param_ranges.keys())
        process_func = partial(self._process_param_combination, 
                             in_ohlcv_i, 
                             param_names=param_names,  # Pass parameter names
                             freq=freq, 
                             fees=fees, 
                             slippage=slippage, 
                             init_cash=init_cash)
        
        results = list(tqdm(pool.imap(process_func, param_combinations), 
                           total=len(param_combinations), 
                           desc=f"Processing window {i+1}"))
        
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df['total_trades'] > 0]
        results_df = results_df.sort_values(['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return'], 
                                          ascending=[False, False, False, False]).reset_index(drop=True)
        
        plot_filename = create_combined_parameter_sharpe_plot(results, i, subfolder)
        self.logger.info(f"Combined Parameter-Sharpe plot saved as {plot_filename}")
        
        if auto_select:
            chosen_params = self._auto_select_params(results_df)
            self.logger.info(f"Automatically selected parameters: {chosen_params}")
        else:
            self.logger.info("Review the plot and choose parameters for the out-of-sample test.")
            chosen_params = {}
            for param, values in param_ranges.items():
                while True:
                    try:
                        value = float(input(f"Enter value for {param}: "))
                        if value in values:
                            chosen_params[param] = value
                            break
                        else:
                            print(f"Value not in range. Please choose from: {values}")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        
        out_result = self._process_param_combination(out_ohlcv_i, tuple(chosen_params.values()), param_names, freq, fees, slippage, init_cash)
        
        return {
            'window': i,
            'in_sample_results': results_df.head(5).to_dict('records'),
            'chosen_params': chosen_params,
            'out_sample_result': out_result,
            'plot_filename': plot_filename,
            'out_sample_df': out_result['processed_df']
        }

    def _auto_select_params(self, results_df):
        """Select the top 1 result based on Sharpe ratio."""
        return results_df.iloc[0]['params']

    def _save_results(self, results, output_file_md, output_file_csv, freq, fees, slippage, init_cash):
        """Save results to markdown and CSV files."""
        subfolder = os.path.dirname(output_file_csv)
        
        # Combine out-of-sample dataframes and ensure correct index
        combined_oos_df = pd.concat([r['out_sample_df'] for r in results])
        if 'Timestamp' in combined_oos_df.columns:
            combined_oos_df.set_index('Timestamp', inplace=True)
        combined_oos_df.index = pd.to_datetime(combined_oos_df.index)
        
        total_oos_stats = self._calculate_total_oos_stats(combined_oos_df, freq, fees, slippage, init_cash)
        
        oos_df_path = os.path.join(subfolder, 'out_of_sample_df.csv')
        combined_oos_df.to_csv(oos_df_path)
        self.logger.info(f"Combined out-of-sample dataframe saved to {oos_df_path}")
        
        with open(output_file_md, 'w') as f:
            f.write("# Walk-Forward Optimization Results\n\n")
            
            # Write individual window results
            for window_result in results:
                f.write(f"## Window {window_result['window'] + 1}\n\n")
                
                f.write("### Chosen Parameters for Out-of-Sample Test\n")
                for param, value in window_result['chosen_params'].items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
                
                f.write("### Out-of-Sample Results\n")
                stats = window_result['out_sample_result']['stats']
                for key in ['Start', 'End', 'Total Return [%]', 'Benchmark Return [%]', 
                           'Max Drawdown [%]', 'Sharpe Ratio', 'Sortino Ratio', 
                           'Calmar Ratio', 'Win Rate [%]', 'Total Trades']:
                    if key in ['Start', 'End']:
                        # Format datetime values properly
                        f.write(f"{key}: {pd.to_datetime(stats[key]).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    elif key in ['Total Return [%]', 'Benchmark Return [%]', 'Max Drawdown [%]', 'Win Rate [%]']:
                        f.write(f"{key.split('[')[0].strip()}: {stats[key]:.2f}%\n")
                    elif key in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
                        f.write(f"{key}: {stats[key]:.4f}\n")
                    else:
                        f.write(f"{key}: {stats[key]}\n")
                f.write("\n")
            
            # Write total out-of-sample statistics
            f.write("## Total Out-of-Sample Statistics\n\n")
            for key in ['Start', 'End', 'Total Return [%]', 'Benchmark Return [%]', 
                       'Max Drawdown [%]', 'Sharpe Ratio', 'Sortino Ratio', 
                       'Calmar Ratio', 'Win Rate [%]', 'Total Trades']:
                if key in ['Start', 'End']:
                    # Format datetime values properly
                    f.write(f"{key}: {pd.to_datetime(total_oos_stats[key]).strftime('%Y-%m-%d %H:%M:%S')}\n")
                elif key in ['Total Return [%]', 'Benchmark Return [%]', 'Max Drawdown [%]', 'Win Rate [%]']:
                    f.write(f"{key.split('[')[0].strip()}: {total_oos_stats[key]:.2f}%\n")
                elif key in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
                    f.write(f"{key}: {total_oos_stats[key]:.4f}\n")
                else:
                    f.write(f"{key}: {total_oos_stats[key]}\n")
        
        self.logger.info(f"Results saved to markdown: {output_file_md}")
        self._save_results_to_csv(results, output_file_csv)

    @staticmethod
    def _create_portfolio(df, freq, fees, slippage, init_cash):
        """Creates a portfolio using vectorbt based on signals in the dataframe."""
        # Calculate entry/exit signals
        long_entries = ((df['Signal'].shift(1) == 1) & (df['Exit'].shift(1).isin([1, -1]) == False)).astype('bool')
        short_entries = ((df['Signal'].shift(1) == -1) & (df['Exit'].shift(1).isin([1, -1]) == False)).astype('bool')

        long_exits = (df['Exit'].shift(1) == 1).astype('bool')
        short_exits = (df['Exit'].shift(1) == -1).astype('bool')

        # Create portfolio
        return vbt.Portfolio.from_signals(
            df['Open'],
            entries=long_entries,
            short_entries=short_entries,
            exits=long_exits,
            short_exits=short_exits,
            freq=freq,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
            accumulate=False
        )

    @staticmethod
    def _process_param_combination(df, params, param_names, freq, fees, slippage, init_cash):
        """Process a single parameter combination and return the results."""
        # Ensure df has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create parameter dictionary by zipping names with values
        param_dict = dict(zip(param_names, params))
        
        # Rest of the method remains the same...
        df_copy = df.copy()
        df_copy.index = df.index
        
        # Process the dataframe with the given parameters
        df = process_strategy(df_copy, **param_dict)
        
        # Create portfolio using the unified function
        pf = WFOptimizer._create_portfolio(df, freq, fees, slippage, init_cash)
        
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
            'processed_df': df
        }

    def _calculate_total_oos_stats(self, combined_oos_df, freq, fees, slippage, init_cash):
        """Calculate statistics for the complete out-of-sample dataset."""
        return WFOptimizer._create_portfolio(combined_oos_df, freq, fees, slippage, init_cash).stats()

    def _save_results_to_csv(self, results, output_file):
        """Save results to CSV file."""
        data = []
        for window_result in results:
            window_data = {
                'Window': window_result['window'] + 1,
                **window_result['chosen_params'],
                **{f'OutOfSample_{k}': v for k, v in window_result['out_sample_result']['stats'].items()}
            }
            data.append(window_data)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to CSV: {output_file}")