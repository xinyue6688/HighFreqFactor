# Order Imbalance Trading Strategy Framework

A comprehensive framework for developing and backtesting order imbalance-based trading strategies for futures contracts using Level-2 market data.

## Overview

This framework provides tools for:
- Calculating various order imbalance metrics from Level-2 order book data
- Generating trading signals based on order imbalance
- Backtesting strategies with transaction costs
- Parameter optimization and performance analysis
- Advanced statistical analysis and regime detection

## Features

### Core Components

1. **Order Imbalance Calculation**
   - Simple n-level imbalance
   - Distance-weighted imbalance
   - Price-weighted imbalance

2. **Signal Generation Methods**
   - Fixed threshold method
   - Z-score normalized method
   - Rolling percentile method

3. **Backtesting Engine**
   - Forward-looking returns calculation
   - Transaction cost modeling
   - Comprehensive performance metrics

4. **Advanced Analysis**
   - Parameter optimization via grid search
   - Regime analysis (volatility, trend, volume)
   - Feature importance analysis using Random Forest
   - Statistical significance testing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from main import OrderImbalanceStrategy

# Initialize strategy
strategy = OrderImbalanceStrategy('IC2507.csv')

# Run comprehensive analysis
strategy.run_comprehensive_analysis()
```

### Custom Analysis

```python
# Load and prepare data
strategy.load_and_prepare_data()

# Calculate imbalance metrics
strategy.calculate_imbalance_metrics(n_levels=5)

# Generate signals using threshold method
strategy.generate_trading_signals(
    method='threshold', 
    upper_threshold=0.3, 
    lower_threshold=-0.3
)

# Run backtest
strategy.run_backtest(forward_periods=1, transaction_cost=0.0001)

# Create plots
strategy.plot_imbalance_analysis()
strategy.plot_backtest_results()
```

### Advanced Analysis

```python
from advanced_analysis import run_advanced_analysis

# Run comprehensive advanced analysis
results = run_advanced_analysis(strategy.df)
```

## Data Format

The framework expects CSV data with the following columns:
- `TradingDay`: Trading date
- `UpdateTime`: Time of update
- `BidPrice1`, `BidPrice2`, ..., `BidPrice5`: Bid prices for levels 1-5
- `AskPrice1`, `AskPrice2`, ..., `AskPrice5`: Ask prices for levels 1-5
- `BidVolume1`, `BidVolume2`, ..., `BidVolume5`: Bid volumes for levels 1-5
- `AskVolume1`, `AskVolume2`, ..., `AskVolume5`: Ask volumes for levels 1-5

## Order Imbalance Calculation

### Simple Imbalance
```
Imbalance = (Sum(BidVolumes) - Sum(AskVolumes)) / (Sum(BidVolumes) + Sum(AskVolumes))
```

### Weighted Imbalance
Gives higher weight to closer price levels:
```
Weight_i = 1/i  (where i is the level number)
```

### Price-Weighted Imbalance
Weights based on distance from mid-price:
```
Weight_i = 1 / (1 + |Price_i - MidPrice|)
```

## Signal Generation Methods

### 1. Threshold Method
- **Buy Signal**: Imbalance > upper_threshold
- **Sell Signal**: Imbalance < lower_threshold
- **Parameters**: `upper_threshold`, `lower_threshold`

### 2. Z-Score Method
- **Buy Signal**: Z-score > threshold
- **Sell Signal**: Z-score < -threshold
- **Parameters**: `window`, `threshold`

### 3. Percentile Method
- **Buy Signal**: Imbalance > rolling upper percentile
- **Sell Signal**: Imbalance < rolling lower percentile
- **Parameters**: `window`, `upper_percentile`, `lower_percentile`

## Performance Metrics

The framework calculates comprehensive performance metrics:

- **Total Return**: Cumulative strategy return
- **Annualized Return**: Annualized performance
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Information Ratio**: Excess return per unit of tracking error

## Parameter Optimization

The framework includes grid search optimization to find optimal parameters:

```python
from advanced_analysis import AdvancedOrderImbalanceAnalysis

advanced = AdvancedOrderImbalanceAnalysis(df)

# Define parameter grid
param_grid = {
    'method': ['threshold', 'zscore'],
    'upper_threshold': [0.2, 0.3, 0.4],
    'lower_threshold': [-0.2, -0.3, -0.4],
    'window': [50, 100, 200],
    'threshold': [1.5, 2.0, 2.5],
    'forward_periods': [1, 2, 3],
    'transaction_cost': [0.0001, 0.0002, 0.0005]
}

# Run optimization
results = advanced.parameter_optimization(param_grid)
```

## File Structure

```
.
├── main.py                 # Main strategy class and execution
├── utils.py               # Core utility functions
├── advanced_analysis.py   # Advanced analysis tools
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── IC2507.csv            # Your data file
```

## Key Classes and Functions

### OrderImbalanceStrategy
Main strategy class with methods:
- `load_and_prepare_data()`: Load and preprocess data
- `calculate_imbalance_metrics()`: Calculate various imbalance measures
- `generate_trading_signals()`: Generate buy/sell signals
- `run_backtest()`: Execute strategy backtest
- `plot_imbalance_analysis()`: Create analysis visualizations
- `run_comprehensive_analysis()`: Complete analysis pipeline

### AdvancedOrderImbalanceAnalysis
Advanced analysis class with methods:
- `parameter_optimization()`: Grid search parameter optimization
- `regime_analysis()`: Performance analysis by market regime
- `feature_importance_analysis()`: ML-based feature importance
- `statistical_tests()`: Statistical significance testing

## Research Ideas

The framework supports various research directions:

1. **Multi-timeframe Analysis**: Analyze imbalance across different time horizons
2. **Regime-Dependent Strategies**: Adapt parameters based on market conditions
3. **Machine Learning Enhancement**: Use ML models for signal generation
4. **Risk Management**: Implement dynamic position sizing and stop-losses
5. **Cross-Asset Analysis**: Compare imbalance patterns across different instruments

## Example Results

The framework generates detailed outputs including:

```
============================================================
ORDER IMBALANCE TRADING STRATEGY ANALYSIS
============================================================

Data loaded: 50000 records
Date range: 20240101 to 20240131

Signal Statistics:
  total_signals: 2341
  buy_signals: 1156
  sell_signals: 1185
  signal_frequency: 0.0468

Backtest Results:
  Total Return: 0.0234 (2.34%)
  Annualized Return: 0.2450 (24.50%)
  Volatility: 0.3200 (32.00%)
  Sharpe Ratio: 0.7656
  Max Drawdown: -0.0890 (-8.90%)
  Win Rate: 0.5234 (52.34%)
  Information Ratio: 0.4321
  Total Trades: 1247
```

## Contributing

Feel free to contribute improvements to the framework:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This framework is for research and educational purposes only. Past performance does not guarantee future results. Always conduct thorough testing before deploying any trading strategy with real capital.