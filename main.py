import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    calculate_nlevel_imbalance, 
    calculate_weighted_imbalance,
    calculate_price_weighted_imbalance,
    generate_signals,
    calculate_returns,
    backtest_strategy,
    calculate_signal_statistics
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class OrderImbalanceStrategy:
    """
    Comprehensive Order Imbalance Trading Strategy Framework
    """
    
    def __init__(self, data_file: str = 'IC2507.csv'):
        """Initialize the strategy with data"""
        self.data_file = data_file
        self.df = None
        self.signals = None
        self.backtest_results = None
        
    def load_and_prepare_data(self):
        """Load and prepare the market data"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_file)
        self.df = self.df.sort_values(by=['TradingDay', 'UpdateTime']).reset_index(drop=True)
        
        # Create timestamp for better analysis
        if 'TradingDay' in self.df.columns and 'UpdateTime' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['TradingDay'].astype(str) + ' ' + 
                                                 self.df['UpdateTime'].astype(str))
        
        # Calculate mid price
        self.df['mid_price'] = (self.df['BidPrice1'] + self.df['AskPrice1']) / 2
        
        # Calculate spread
        self.df['spread'] = self.df['AskPrice1'] - self.df['BidPrice1']
        self.df['spread_bps'] = (self.df['spread'] / self.df['mid_price']) * 10000
        
        print(f"Data loaded: {len(self.df)} records")
        print(f"Date range: {self.df['TradingDay'].min()} to {self.df['TradingDay'].max()}")
        
        return self.df
    
    def calculate_imbalance_metrics(self, n_levels: int = 5):
        """Calculate various order imbalance metrics"""
        print(f"Calculating imbalance metrics for {n_levels} levels...")
        
        # Simple imbalance
        self.df['imbalance_simple'] = self.df.apply(
            lambda row: calculate_nlevel_imbalance(row, n_levels), axis=1
        )
        
        # Weighted imbalance (distance-based)
        self.df['imbalance_weighted'] = self.df.apply(
            lambda row: calculate_weighted_imbalance(row, n_levels), axis=1
        )
        
        # Price-weighted imbalance
        self.df['imbalance_price_weighted'] = self.df.apply(
            lambda row: calculate_price_weighted_imbalance(row, n_levels), axis=1
        )
        
        # Calculate moving averages of imbalance for smoothing
        self.df['imbalance_ma_10'] = self.df['imbalance_simple'].rolling(window=10).mean()
        self.df['imbalance_ma_50'] = self.df['imbalance_simple'].rolling(window=50).mean()
        
        return self.df
    
    def generate_trading_signals(self, method: str = 'threshold', imbalance_type: str = 'simple', **kwargs):
        """Generate trading signals based on order imbalance"""
        print(f"Generating {method} signals using {imbalance_type} imbalance...")
        
        # Select imbalance type
        imbalance_col = f'imbalance_{imbalance_type}'
        if imbalance_col not in self.df.columns:
            imbalance_col = 'imbalance_simple'
            
        imbalance = self.df[imbalance_col]
        
        # Generate signals
        self.signals = generate_signals(imbalance, method=method, **kwargs)
        self.df['signals'] = self.signals
        
        # Calculate signal statistics
        signal_stats = calculate_signal_statistics(self.signals, imbalance)
        print(f"Signal Statistics:")
        for key, value in signal_stats.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return self.signals
    
    def run_backtest(self, forward_periods: int = 1, transaction_cost: float = 0.0001):
        """Run backtest of the strategy"""
        print(f"Running backtest with {forward_periods} period(s) forward returns...")
        
        if self.signals is None:
            raise ValueError("Signals not generated. Call generate_trading_signals first.")
        
        # Calculate forward returns
        returns = calculate_returns(self.df, forward_periods=forward_periods)
        self.df['forward_returns'] = returns
        
        # Run backtest
        self.backtest_results = backtest_strategy(
            self.df, self.signals, returns, transaction_cost=transaction_cost
        )
        
        # Print results
        if 'error' not in self.backtest_results:
            print(f"Backtest Results:")
            print(f"  Total Return: {self.backtest_results['total_return']:.4f} ({self.backtest_results['total_return']*100:.2f}%)")
            print(f"  Annualized Return: {self.backtest_results['annualized_return']:.4f} ({self.backtest_results['annualized_return']*100:.2f}%)")
            print(f"  Volatility: {self.backtest_results['volatility']:.4f} ({self.backtest_results['volatility']*100:.2f}%)")
            print(f"  Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {self.backtest_results['max_drawdown']:.4f} ({self.backtest_results['max_drawdown']*100:.2f}%)")
            print(f"  Win Rate: {self.backtest_results['win_rate']:.4f} ({self.backtest_results['win_rate']*100:.2f}%)")
            print(f"  Information Ratio: {self.backtest_results['information_ratio']:.4f}")
            print(f"  Total Trades: {self.backtest_results['total_trades']}")
        else:
            print(f"Backtest Error: {self.backtest_results['error']}")
        
        return self.backtest_results
    
    def plot_imbalance_analysis(self, save_plots: bool = False):
        """Create comprehensive imbalance analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Imbalance histogram
        axes[0, 0].hist(self.df['imbalance_simple'].dropna(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Order Imbalance Distribution')
        axes[0, 0].set_xlabel('Imbalance Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # 2. Imbalance time series
        sample_data = self.df.head(1000)  # Show first 1000 points for clarity
        axes[0, 1].plot(sample_data.index, sample_data['imbalance_simple'], alpha=0.7, linewidth=0.8)
        axes[0, 1].plot(sample_data.index, sample_data['imbalance_ma_10'], color='red', linewidth=1.5, label='MA-10')
        axes[0, 1].set_title('Order Imbalance Time Series (First 1000 Points)')
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('Imbalance Value')
        axes[0, 1].legend()
        axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Imbalance vs Future Returns
        if 'forward_returns' in self.df.columns:
            valid_data = self.df[['imbalance_simple', 'forward_returns']].dropna()
            if len(valid_data) > 0:
                axes[1, 0].scatter(valid_data['imbalance_simple'], valid_data['forward_returns'], 
                                 alpha=0.5, s=1)
                axes[1, 0].set_title('Imbalance vs Forward Returns')
                axes[1, 0].set_xlabel('Imbalance Value')
                axes[1, 0].set_ylabel('Forward Returns')
                
                # Add correlation
                corr = valid_data['imbalance_simple'].corr(valid_data['forward_returns'])
                axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                              transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 4. Signal visualization
        if self.signals is not None:
            sample_signals = self.signals.head(1000)
            buy_signals = sample_signals[sample_signals == 1]
            sell_signals = sample_signals[sample_signals == -1]
            
            axes[1, 1].plot(sample_data.index, sample_data['mid_price'], color='black', alpha=0.7, linewidth=1)
            axes[1, 1].scatter(buy_signals.index, sample_data.loc[buy_signals.index, 'mid_price'], 
                             color='green', marker='^', s=30, label='Buy Signal')
            axes[1, 1].scatter(sell_signals.index, sample_data.loc[sell_signals.index, 'mid_price'], 
                             color='red', marker='v', s=30, label='Sell Signal')
            axes[1, 1].set_title('Trading Signals on Price Chart')
            axes[1, 1].set_xlabel('Time Index')
            axes[1, 1].set_ylabel('Mid Price')
            axes[1, 1].legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('imbalance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_backtest_results(self, save_plots: bool = False):
        """Plot backtest results"""
        if self.backtest_results is None or 'error' in self.backtest_results:
            print("No valid backtest results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative returns
        cumulative_returns = self.backtest_results['cumulative_returns']
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, color='blue', linewidth=2)
        axes[0, 0].set_title('Strategy Cumulative Returns')
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Strategy returns distribution
        strategy_returns = self.backtest_results['strategy_returns'].dropna()
        axes[0, 1].hist(strategy_returns, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Strategy Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(strategy_returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {strategy_returns.mean():.6f}')
        axes[0, 1].legend()
        
        # 3. Rolling Sharpe ratio
        rolling_sharpe = strategy_returns.rolling(window=100).mean() / strategy_returns.rolling(window=100).std()
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, color='purple', linewidth=1)
        axes[1, 0].set_title('Rolling Sharpe Ratio (100-period)')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Drawdown
        cumulative_returns = self.backtest_results['cumulative_returns']
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.6, color='red')
        axes[1, 1].set_title('Strategy Drawdown')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("ORDER IMBALANCE TRADING STRATEGY ANALYSIS")
        print("=" * 60)

        # Load and prepare data
        self.load_and_prepare_data()

        # Calculate imbalance metrics
        self.calculate_imbalance_metrics(n_levels=5)

        # Test different signal generation methods
        methods = [
            {'method': 'threshold', 'upper_threshold': 0.3, 'lower_threshold': -0.3},
            {'method': 'zscore', 'window': 100, 'threshold': 2},
            {'method': 'percentile', 'window': 100, 'upper_percentile': 80, 'lower_percentile': 20}
        ]

        best_sharpe = -np.inf
        best_method = None
        best_results = None
        best_method_params = None

        for i, method_params in enumerate(methods):
            print(f"\n{'=' * 40}")
            print(f"Testing Method {i + 1}: {method_params}")
            print(f"{'=' * 40}")

            # Create a copy to avoid modifying the original
            params_copy = method_params.copy()
            method = params_copy.pop('method')
            self.generate_trading_signals(method=method, **params_copy)

            # Run backtest
            results = self.run_backtest(forward_periods=1, transaction_cost=0.0001)

            # Track best performing method
            if (results and 'error' not in results and
                    results['sharpe_ratio'] > best_sharpe):
                best_sharpe = results['sharpe_ratio']
                best_method = method
                best_method_params = method_params.copy()
                best_results = results.copy()

        # Use best method for final analysis
        if best_method:
            print(f"\n{'=' * 60}")
            print(f"BEST PERFORMING METHOD: {best_method} with {best_method_params}")
            print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
            print(f"{'=' * 60}")

            # Re-run with best method for plotting
            best_params_copy = best_method_params.copy()
            best_params_copy.pop('method')  # Remove method since it's passed separately
            self.generate_trading_signals(method=best_method, **best_params_copy)
            self.run_backtest(forward_periods=1, transaction_cost=0.0001)

        # Create plots
        print("\nGenerating analysis plots...")
        self.plot_imbalance_analysis(save_plots=True)
        self.plot_backtest_results(save_plots=True)

        print("\nAnalysis complete! Check the generated plots for detailed results.")
        

# Main execution
if __name__ == "__main__":
    # Create strategy instance
    strategy = OrderImbalanceStrategy('IC2507.csv')
    
    # Run comprehensive analysis
    strategy.run_comprehensive_analysis()
    
    # Optional: Run custom analysis
    # strategy.load_and_prepare_data()
    # strategy.calculate_imbalance_metrics(n_levels=3)
    # strategy.generate_trading_signals(method='threshold', upper_threshold=0.4, lower_threshold=-0.4)
    # strategy.run_backtest(forward_periods=2, transaction_cost=0.0002)
    # strategy.plot_imbalance_analysis()
    # strategy.plot_backtest_results()

