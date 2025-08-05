"""
Example usage of the Order Imbalance Trading Strategy Framework

This script demonstrates how to use the framework for analyzing order imbalance
and backtesting trading strategies.
"""

import pandas as pd
import numpy as np
from main import OrderImbalanceStrategy
from advanced_analysis import AdvancedOrderImbalanceAnalysis, run_advanced_analysis

def create_sample_data():
    """Create sample order book data for demonstration purposes"""
    np.random.seed(42)
    n_rows = 10000
    
    # Base price
    base_price = 3000.0
    price_changes = np.random.normal(0, 0.001, n_rows)
    prices = base_price * (1 + price_changes.cumsum())
    
    data = []
    for i in range(n_rows):
        price = prices[i]
        spread = np.random.uniform(0.5, 2.0)
        
        # Generate order book levels
        bid_prices = [price - spread/2 - j*0.5 for j in range(5)]
        ask_prices = [price + spread/2 + j*0.5 for j in range(5)]
        
        # Generate volumes (with some imbalance)
        imbalance_factor = np.random.normal(0, 0.3)
        bid_volumes = [max(1, int(np.random.exponential(100) * (1 + imbalance_factor))) for _ in range(5)]
        ask_volumes = [max(1, int(np.random.exponential(100) * (1 - imbalance_factor))) for _ in range(5)]
        
        row = {
            'TradingDay': 20240101,
            'UpdateTime': f"09:30:{i%3600:02d}.{(i*200)%1000:03d}",
        }
        
        # Add bid/ask prices and volumes
        for j in range(5):
            row[f'BidPrice{j+1}'] = round(bid_prices[j], 1)
            row[f'AskPrice{j+1}'] = round(ask_prices[j], 1)
            row[f'BidVolume{j+1}'] = bid_volumes[j]
            row[f'AskVolume{j+1}'] = ask_volumes[j]
        
        data.append(row)
    
    return pd.DataFrame(data)

def example_basic_usage():
    """Demonstrate basic usage of the framework"""
    print("="*60)
    print("BASIC USAGE EXAMPLE")
    print("="*60)
    
    # Create sample data (replace with your actual data loading)
    print("Creating sample data...")
    df = create_sample_data()
    df.to_csv('sample_data.csv', index=False)
    
    # Initialize strategy
    strategy = OrderImbalanceStrategy('sample_data.csv')
    
    # Method 1: Run comprehensive analysis (recommended for beginners)
    print("\nRunning comprehensive analysis...")
    strategy.run_comprehensive_analysis()
    
    return strategy

def example_custom_analysis():
    """Demonstrate custom analysis workflow"""
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    # Initialize strategy
    strategy = OrderImbalanceStrategy('sample_data.csv')
    
    # Step-by-step analysis
    print("Loading and preparing data...")
    strategy.load_and_prepare_data()
    
    print("Calculating imbalance metrics...")
    strategy.calculate_imbalance_metrics(n_levels=3)  # Use 3 levels instead of 5
    
    print("Testing different signal generation methods...")
    
    # Test threshold method
    print("\n--- Threshold Method ---")
    strategy.generate_trading_signals(
        method='threshold',
        upper_threshold=0.4,
        lower_threshold=-0.4
    )
    results_threshold = strategy.run_backtest(forward_periods=1, transaction_cost=0.0001)
    
    # Test z-score method
    print("\n--- Z-Score Method ---")
    strategy.generate_trading_signals(
        method='zscore',
        window=50,
        threshold=2.0
    )
    results_zscore = strategy.run_backtest(forward_periods=1, transaction_cost=0.0001)
    
    # Test percentile method
    print("\n--- Percentile Method ---")
    strategy.generate_trading_signals(
        method='percentile',
        window=100,
        upper_percentile=85,
        lower_percentile=15
    )
    results_percentile = strategy.run_backtest(forward_periods=1, transaction_cost=0.0001)
    
    # Compare results
    print("\n" + "="*40)
    print("METHOD COMPARISON")
    print("="*40)
    methods = ['Threshold', 'Z-Score', 'Percentile']
    results = [results_threshold, results_zscore, results_percentile]
    
    for method, result in zip(methods, results):
        if result and 'error' not in result:
            print(f"\n{method}:")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"  Total Return: {result['total_return']:.4f}")
            print(f"  Win Rate: {result['win_rate']:.4f}")
            print(f"  Total Trades: {result['total_trades']}")
    
    # Create plots for the best method
    print("\nGenerating plots...")
    strategy.plot_imbalance_analysis(save_plots=True)
    strategy.plot_backtest_results(save_plots=True)
    
    return strategy

def example_advanced_analysis():
    """Demonstrate advanced analysis features"""
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS EXAMPLE")
    print("="*60)
    
    # Load data
    df = pd.read_csv('sample_data.csv')
    
    # Run advanced analysis
    results = run_advanced_analysis(df)
    
    # Display key findings
    if results['optimization_results'] is not None:
        best_params = results['optimization_results'].iloc[0]
        print(f"\nBest parameters found:")
        print(f"  Method: {best_params['method']}")
        print(f"  Sharpe Ratio: {best_params['sharpe_ratio']:.4f}")
        print(f"  Total Return: {best_params['total_return']:.4f}")
    
    return results

def example_parameter_optimization():
    """Demonstrate parameter optimization"""
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Load data
    df = pd.read_csv('sample_data.csv')
    
    # Initialize advanced analysis
    advanced = AdvancedOrderImbalanceAnalysis(df)
    
    # Define a smaller parameter grid for demonstration
    param_grid = {
        'method': ['threshold', 'zscore'],
        'upper_threshold': [0.2, 0.3, 0.4],
        'lower_threshold': [-0.2, -0.3, -0.4],
        'window': [50, 100],
        'threshold': [1.5, 2.0],
        'forward_periods': [1, 2],
        'transaction_cost': [0.0001, 0.0002]
    }
    
    print("Running parameter optimization...")
    optimization_results = advanced.parameter_optimization(param_grid)
    
    if optimization_results is not None and len(optimization_results) > 0:
        print(f"\nOptimization completed! Best 3 parameter combinations:")
        print(optimization_results.head(3)[['method', 'sharpe_ratio', 'total_return', 'win_rate']].to_string())
        
        # Plot optimization results
        advanced.plot_optimization_results(save_plots=True)
    
    return advanced

def main():
    """Run all examples"""
    print("Order Imbalance Trading Strategy Framework Examples")
    print("="*60)
    
    # Example 1: Basic usage
    strategy = example_basic_usage()
    
    # Example 2: Custom analysis
    strategy = example_custom_analysis()
    
    # Example 3: Advanced analysis
    advanced_results = example_advanced_analysis()
    
    # Example 4: Parameter optimization
    advanced = example_parameter_optimization()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED!")
    print("="*60)
    print("Check the generated plots:")
    print("- imbalance_analysis.png")
    print("- backtest_results.png") 
    print("- optimization_results.png")
    print("\nYou can now use this framework with your own IC2507.csv data file.")

if __name__ == "__main__":
    main()