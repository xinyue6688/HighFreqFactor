# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File: utils.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def calculate_nlevel_imbalance(row: pd.Series, n: int) -> float:
    """
    Calculate n-level order book imbalance
    
    Args:
        row: DataFrame row containing bid/ask volumes
        n: Number of levels to consider
    
    Returns:
        Order imbalance ratio between -1 and 1
    """
    bid_volume_sum = sum([row[f'BidVolume{i}'] for i in range(1, n + 1)])
    ask_volume_sum = sum([row[f'AskVolume{i}'] for i in range(1, n + 1)])
    
    if bid_volume_sum + ask_volume_sum == 0:
        return 0
    
    return (bid_volume_sum - ask_volume_sum) / (bid_volume_sum + ask_volume_sum)

def calculate_weighted_imbalance(row: pd.Series, n: int) -> float:
    """
    Calculate weighted order book imbalance (closer levels get higher weights)
    
    Args:
        row: DataFrame row containing bid/ask volumes and prices
        n: Number of levels to consider
    
    Returns:
        Weighted order imbalance ratio
    """
    bid_weighted_sum = 0
    ask_weighted_sum = 0
    
    for i in range(1, n + 1):
        weight = 1 / i  # Higher weight for closer levels
        bid_weighted_sum += row[f'BidVolume{i}'] * weight
        ask_weighted_sum += row[f'AskVolume{i}'] * weight
    
    total_weighted = bid_weighted_sum + ask_weighted_sum
    if total_weighted == 0:
        return 0
    
    return (bid_weighted_sum - ask_weighted_sum) / total_weighted

def calculate_price_weighted_imbalance(row: pd.Series, n: int) -> float:
    """
    Calculate price-weighted order book imbalance
    
    Args:
        row: DataFrame row containing bid/ask volumes and prices
        n: Number of levels to consider
    
    Returns:
        Price-weighted order imbalance ratio
    """
    mid_price = (row['BidPrice1'] + row['AskPrice1']) / 2
    bid_weighted_sum = 0
    ask_weighted_sum = 0
    
    for i in range(1, n + 1):
        # Weight by distance from mid price
        bid_weight = 1 / (1 + abs(row[f'BidPrice{i}'] - mid_price))
        ask_weight = 1 / (1 + abs(row[f'AskPrice{i}'] - mid_price))
        
        bid_weighted_sum += row[f'BidVolume{i}'] * bid_weight
        ask_weighted_sum += row[f'AskVolume{i}'] * ask_weight
    
    total_weighted = bid_weighted_sum + ask_weighted_sum
    if total_weighted == 0:
        return 0
    
    return (bid_weighted_sum - ask_weighted_sum) / total_weighted

def generate_signals(imbalance: pd.Series, method: str = 'threshold', **kwargs) -> pd.Series:
    """
    Generate trading signals based on order imbalance
    
    Args:
        imbalance: Series of order imbalance values
        method: Signal generation method ('threshold', 'zscore', 'percentile')
        **kwargs: Additional parameters for each method
    
    Returns:
        Series of trading signals (1: buy, -1: sell, 0: hold)
    """
    signals = pd.Series(0, index=imbalance.index)
    
    if method == 'threshold':
        upper_threshold = kwargs.get('upper_threshold', 0.3)
        lower_threshold = kwargs.get('lower_threshold', -0.3)
        
        signals[imbalance > upper_threshold] = 1   # Buy signal
        signals[imbalance < lower_threshold] = -1  # Sell signal
        
    elif method == 'zscore':
        window = kwargs.get('window', 100)
        threshold = kwargs.get('threshold', 2)
        
        rolling_mean = imbalance.rolling(window=window).mean()
        rolling_std = imbalance.rolling(window=window).std()
        zscore = (imbalance - rolling_mean) / rolling_std
        
        signals[zscore > threshold] = 1
        signals[zscore < -threshold] = -1
        
    elif method == 'percentile':
        window = kwargs.get('window', 100)
        upper_pct = kwargs.get('upper_percentile', 80)
        lower_pct = kwargs.get('lower_percentile', 20)
        
        rolling_upper = imbalance.rolling(window=window).quantile(upper_pct/100)
        rolling_lower = imbalance.rolling(window=window).quantile(lower_pct/100)
        
        signals[imbalance > rolling_upper] = 1
        signals[imbalance < rolling_lower] = -1
    
    return signals

def calculate_returns(df: pd.DataFrame, price_col: str = 'mid_price', forward_periods: int = 1) -> pd.Series:
    """
    Calculate forward returns for backtesting
    
    Args:
        df: DataFrame containing price data
        price_col: Column name for price data
        forward_periods: Number of periods to look forward
    
    Returns:
        Series of forward returns
    """
    if price_col not in df.columns:
        # Calculate mid price if not available
        df[price_col] = (df['BidPrice1'] + df['AskPrice1']) / 2
    
    returns = df[price_col].pct_change(periods=forward_periods).shift(-forward_periods)
    return returns

def backtest_strategy(df: pd.DataFrame, signals: pd.Series, returns: pd.Series, 
                     transaction_cost: float = 0.0001) -> Dict[str, Any]:
    """
    Backtest the order imbalance strategy
    
    Args:
        df: DataFrame with market data
        signals: Trading signals
        returns: Forward returns
        transaction_cost: Transaction cost per trade (as percentage)
    
    Returns:
        Dictionary containing backtest results
    """
    # Calculate strategy returns
    strategy_returns = signals.shift(1) * returns - abs(signals.diff()) * transaction_cost
    
    # Remove NaN values
    strategy_returns = strategy_returns.dropna()
    returns_clean = returns.dropna()
    
    if len(strategy_returns) == 0:
        return {"error": "No valid returns for backtesting"}
    
    # Performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annualized_return = (1 + strategy_returns.mean()) ** (252 * 24 * 60) - 1  # Assuming minute data
    volatility = strategy_returns.std() * np.sqrt(252 * 24 * 60)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = strategy_returns[strategy_returns > 0]
    total_trades = len(strategy_returns[strategy_returns != 0])
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Information Ratio (vs buy and hold)
    benchmark_returns = returns_clean
    excess_returns = strategy_returns - benchmark_returns.loc[strategy_returns.index]
    tracking_error = excess_returns.std()
    information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'information_ratio': information_ratio,
        'total_trades': total_trades,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns
    }

def calculate_signal_statistics(signals: pd.Series, imbalance: pd.Series) -> Dict[str, Any]:
    """
    Calculate statistics about signal generation
    
    Args:
        signals: Trading signals
        imbalance: Order imbalance values
    
    Returns:
        Dictionary containing signal statistics
    """
    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]
    
    stats = {
        'total_signals': len(signals[signals != 0]),
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'signal_frequency': len(signals[signals != 0]) / len(signals),
        'avg_imbalance_at_buy': imbalance[signals == 1].mean() if len(buy_signals) > 0 else np.nan,
        'avg_imbalance_at_sell': imbalance[signals == -1].mean() if len(sell_signals) > 0 else np.nan,
    }
    
    return stats