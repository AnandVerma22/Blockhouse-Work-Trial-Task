import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import os

plt.rcParams['figure.figsize'] = [10, 6]  # Set default figure size

class OrderBookAnalyzer:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.ticker_data = {}
        self.load_all_data()
        
    def load_all_data(self):
        for ticker_dir in self.data_root.iterdir():
            if ticker_dir.is_dir():
                ticker = ticker_dir.name
                all_files = list(ticker_dir.glob('*.csv'))
                
                if not all_files:
                    print(f"No CSV files found for {ticker}")
                    continue
                    
                dfs = []
                for file in all_files:
                    try:
                        if ticker == 'CRWV':
                            df = pd.read_csv(
                                file,
                                parse_dates=['ts_event'],
                                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
                            )
                        else:
                            df = pd.read_csv(file, parse_dates=['ts_event'])
                        
                        df = df.dropna(subset=['ts_event'])
                        df['ticker'] = ticker
                        dfs.append(df)
                        print(f"Loaded {file.name} for {ticker}")
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
                
                if dfs:
                    self.ticker_data[ticker] = pd.concat(dfs, ignore_index=True)
                    print(f"Combined {len(dfs)} files for {ticker}")

    def calculate_mid_price(self, row: pd.Series) -> float:
        return (row['bid_px_00'] + row['ask_px_00']) / 2
    
    def calculate_slippage(self, row: pd.Series, quantity: float, side: str) -> float:
        mid_price = self.calculate_mid_price(row)
        executed_price = 0
        remaining_qty = quantity
        
        if side == 'buy':
            for i in range(10):
                px_col = f'ask_px_{i:02d}'
                sz_col = f'ask_sz_{i:02d}'
                
                if px_col not in row or sz_col not in row:
                    continue
                    
                available = row[sz_col]
                price = row[px_col]
                
                if remaining_qty <= 0:
                    break
                    
                if available > remaining_qty:
                    executed_price += remaining_qty * price
                    remaining_qty = 0
                else:
                    executed_price += available * price
                    remaining_qty -= available
            
            if remaining_qty > 0:
                executed_price += remaining_qty * price
                
            avg_price = executed_price / quantity
            return avg_price - mid_price
            
        elif side == 'sell':
            for i in range(10):
                px_col = f'bid_px_{i:02d}'
                sz_col = f'bid_sz_{i:02d}'
                
                if px_col not in row or sz_col not in row:
                    continue
                    
                available = row[sz_col]
                price = row[px_col]
                
                if remaining_qty <= 0:
                    break
                    
                if available > remaining_qty:
                    executed_price += remaining_qty * price
                    remaining_qty = 0
                else:
                    executed_price += available * price
                    remaining_qty -= available
            
            if remaining_qty > 0:
                executed_price += remaining_qty * price
                
            avg_price = executed_price / quantity
            return mid_price - avg_price
        
        else:
            raise ValueError("Side must be 'buy' or 'sell'")
    
    def estimate_temporary_impact(self, ticker: str, side: str = 'buy') -> Tuple[np.ndarray, np.ndarray]:
        if ticker not in self.ticker_data:
            raise ValueError(f"No data available for {ticker}")
            
        df = self.ticker_data[ticker]
        row = df.iloc[0]  # Use first snapshot
        
        x_values = np.linspace(1, 10000, 50)
        g_values = []
        
        for x in x_values:
            slippage = self.calculate_slippage(row, x, side)
            g_values.append(slippage)
            
        return x_values, np.array(g_values)
    
    def plot_impact_function(self, ticker: str, side: str = 'buy'):
        x, g = self.estimate_temporary_impact(ticker, side)
        
        fig, ax = plt.subplots()
        ax.plot(x, g, 'b-', label='Temporary Impact')
        ax.set_xlabel('Order Size (shares)')
        ax.set_ylabel('Slippage (per share)')
        ax.set_title(f'Temporary Impact Function for {ticker} ({side} side)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    def fit_impact_model(self, ticker: str, side: str = 'buy') -> Dict:
        x, g = self.estimate_temporary_impact(ticker, side)
        
        # Linear model
        linear_coef = np.polyfit(x, g, 1)
        
        # Square root model
        sqrt_x = np.sqrt(x)
        sqrt_coef = np.polyfit(sqrt_x, g, 1)
        
        # Power law model
        log_x = np.log(x[x > 0])
        log_g = np.log(g[x > 0])
        power_coef = np.polyfit(log_x, log_g, 1)
        power_b = power_coef[0]
        power_a = np.exp(power_coef[1])
        
        return {
            'linear': {'a': linear_coef[0], 'b': 1},
            'sqrt': {'a': sqrt_coef[0], 'b': 0.5},
            'power': {'a': power_a, 'b': power_b}
        }
    
    def optimal_execution(self, S: float, N: int, ticker: str, side: str = 'buy', model_type: str = 'sqrt') -> np.ndarray:
        models = self.fit_impact_model(ticker, side)
        
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
            
        a = models[model_type]['a']
        b = models[model_type]['b']
        
        # Equal time-weighted allocation (TWAP)
        x_twap = np.ones(N) * S / N
        
        # Adjust for impact
        t_values = np.arange(N)
        
        if model_type == 'linear':
            return x_twap
        elif model_type == 'sqrt':
            weights = 1 / (1 + 0.1 * t_values)
        elif model_type == 'power':
            weights = 1 / (1 + 0.05 * t_values ** (b))
            
        weights = weights / np.sum(weights)
        x_optimal = S * weights
        
        return x_optimal


if __name__ == "__main__":
    # Set up the data paths
    data_root = "data"  # Update this to your actual path
    
    # Create analyzer instance
    analyzer = OrderBookAnalyzer(data_root)
    
    # Ensure we have some tickers loaded
    if not analyzer.ticker_data:
        print("No ticker data loaded - check your data directory path")
    else:
        # Analyze each ticker
        for ticker in analyzer.ticker_data.keys():
            try:
                print(f"\nAnalyzing {ticker}")
                
                # Plot impact function
                analyzer.plot_impact_function(ticker)
                
                # Fit models
                models = analyzer.fit_impact_model(ticker)
                print(f"Models for {ticker}:")
                for model, params in models.items():
                    print(f"{model}: g(x) = {params['a']:.6f} * x^{params['b']:.2f}")
                
                # Example execution schedules
                S = 10000
                N = 390
                
                fig, ax = plt.subplots(figsize=(12, 6))
                for model_type in ['sqrt', 'power', 'linear']:
                    schedule = analyzer.optimal_execution(S, N, ticker, model_type=model_type)
                    ax.plot(schedule, label=f'{model_type} model')
                
                ax.plot([S/N]*N, 'k--', label='TWAP')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Shares to execute')
                ax.set_title(f'Execution Schedules for {ticker} (Total={S} shares)')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")