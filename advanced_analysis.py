import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from utils import (
    calculate_nlevel_imbalance,
    generate_signals,
    calculate_returns,
    backtest_strategy
)

class AdvancedOrderImbalanceAnalysis:
    """
    Advanced analysis tools for order imbalance strategy optimization and research
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with prepared DataFrame"""
        self.df = df.copy()
        self.optimization_results = None
        
    def parameter_optimization(self, param_grid: dict = None, n_levels: int = 5):
        """
        Optimize strategy parameters using grid search
        
        Args:
            param_grid: Dictionary of parameters to optimize
            n_levels: Number of order book levels to use
        
        Returns:
            DataFrame with optimization results
        """
        if param_grid is None:
            param_grid = {
                'method': ['threshold', 'zscore', 'percentile'],
                'upper_threshold': [0.2, 0.3, 0.4, 0.5],
                'lower_threshold': [-0.2, -0.3, -0.4, -0.5],
                'window': [50, 100, 200],
                'threshold': [1.5, 2.0, 2.5],
                'upper_percentile': [75, 80, 85, 90],
                'lower_percentile': [10, 15, 20, 25],
                'forward_periods': [1, 2, 3, 5],
                'transaction_cost': [0.0001, 0.0002, 0.0005]
            }
        
        print("Starting parameter optimization...")
        
        # Calculate imbalance
        self.df['imbalance'] = self.df.apply(
            lambda row: calculate_nlevel_imbalance(row, n_levels), axis=1
        )
        
        results = []
        
        # Generate parameter combinations based on method
        for method in param_grid['method']:
            if method == 'threshold':
                for upper_thresh, lower_thresh, forward_periods, tx_cost in product(
                    param_grid['upper_threshold'],
                    param_grid['lower_threshold'],
                    param_grid['forward_periods'],
                    param_grid['transaction_cost']
                ):
                    if upper_thresh > -lower_thresh:  # Ensure consistent thresholds
                        result = self._test_parameter_combination(
                            method, {'upper_threshold': upper_thresh, 'lower_threshold': lower_thresh},
                            forward_periods, tx_cost
                        )
                        if result:
                            results.append(result)
            
            elif method == 'zscore':
                for window, threshold, forward_periods, tx_cost in product(
                    param_grid['window'],
                    param_grid['threshold'],
                    param_grid['forward_periods'],
                    param_grid['transaction_cost']
                ):
                    result = self._test_parameter_combination(
                        method, {'window': window, 'threshold': threshold},
                        forward_periods, tx_cost
                    )
                    if result:
                        results.append(result)
            
            elif method == 'percentile':
                for window, upper_pct, lower_pct, forward_periods, tx_cost in product(
                    param_grid['window'],
                    param_grid['upper_percentile'],
                    param_grid['lower_percentile'],
                    param_grid['forward_periods'],
                    param_grid['transaction_cost']
                ):
                    if upper_pct > lower_pct:  # Ensure logical percentiles
                        result = self._test_parameter_combination(
                            method, {'window': window, 'upper_percentile': upper_pct, 'lower_percentile': lower_pct},
                            forward_periods, tx_cost
                        )
                        if result:
                            results.append(result)
        
        self.optimization_results = pd.DataFrame(results)
        
        if len(self.optimization_results) > 0:
            # Sort by Sharpe ratio
            self.optimization_results = self.optimization_results.sort_values('sharpe_ratio', ascending=False)
            
            print(f"Optimization complete! Tested {len(self.optimization_results)} parameter combinations.")
            print("\nTop 5 performing parameter combinations:")
            print(self.optimization_results.head().to_string())
        else:
            print("No valid parameter combinations found.")
        
        return self.optimization_results
    
    def _test_parameter_combination(self, method: str, method_params: dict, forward_periods: int, tx_cost: float):
        """Test a single parameter combination"""
        try:
            # Generate signals
            signals = generate_signals(self.df['imbalance'], method=method, **method_params)
            
            # Calculate returns
            returns = calculate_returns(self.df, forward_periods=forward_periods)
            
            # Run backtest
            backtest_result = backtest_strategy(self.df, signals, returns, transaction_cost=tx_cost)
            
            if 'error' not in backtest_result and backtest_result['total_trades'] > 10:
                return {
                    'method': method,
                    **method_params,
                    'forward_periods': forward_periods,
                    'transaction_cost': tx_cost,
                    'sharpe_ratio': backtest_result['sharpe_ratio'],
                    'total_return': backtest_result['total_return'],
                    'max_drawdown': backtest_result['max_drawdown'],
                    'win_rate': backtest_result['win_rate'],
                    'total_trades': backtest_result['total_trades'],
                    'information_ratio': backtest_result['information_ratio']
                }
        except Exception as e:
            pass  # Skip problematic parameter combinations
        
        return None
    
    def regime_analysis(self, regime_indicator: str = 'volatility', window: int = 100):
        """
        Analyze strategy performance across different market regimes
        
        Args:
            regime_indicator: Type of regime analysis ('volatility', 'trend', 'volume')
            window: Window for regime calculation
        
        Returns:
            Dictionary with regime analysis results
        """
        print(f"Performing {regime_indicator} regime analysis...")
        
        # Calculate mid price if not available
        if 'mid_price' not in self.df.columns:
            self.df['mid_price'] = (self.df['BidPrice1'] + self.df['AskPrice1']) / 2
        
        # Calculate regime indicators
        if regime_indicator == 'volatility':
            returns = self.df['mid_price'].pct_change()
            regime_metric = returns.rolling(window=window).std()
            regime_labels = pd.cut(regime_metric, bins=3, labels=['Low Vol', 'Medium Vol', 'High Vol'])
            
        elif regime_indicator == 'trend':
            ma_short = self.df['mid_price'].rolling(window=20).mean()
            ma_long = self.df['mid_price'].rolling(window=100).mean()
            trend = (ma_short - ma_long) / ma_long
            regime_labels = pd.cut(trend, bins=3, labels=['Downtrend', 'Sideways', 'Uptrend'])
            
        elif regime_indicator == 'volume':
            total_volume = self.df['BidVolume1'] + self.df['AskVolume1']
            volume_ma = total_volume.rolling(window=window).mean()
            regime_labels = pd.cut(volume_ma, bins=3, labels=['Low Volume', 'Medium Volume', 'High Volume'])
        
        # Calculate imbalance if not available
        if 'imbalance' not in self.df.columns:
            self.df['imbalance'] = self.df.apply(
                lambda row: calculate_nlevel_imbalance(row, 5), axis=1
            )
        
        # Analyze imbalance characteristics by regime
        regime_stats = {}
        for regime in regime_labels.cat.categories:
            regime_mask = regime_labels == regime
            regime_data = self.df[regime_mask]
            
            if len(regime_data) > 100:  # Minimum data points for analysis
                regime_stats[regime] = {
                    'count': len(regime_data),
                    'imbalance_mean': regime_data['imbalance'].mean(),
                    'imbalance_std': regime_data['imbalance'].std(),
                    'imbalance_skew': regime_data['imbalance'].skew(),
                    'imbalance_kurt': regime_data['imbalance'].kurtosis()
                }
        
        # Test strategy performance by regime (using best parameters if available)
        if self.optimization_results is not None and len(self.optimization_results) > 0:
            best_params = self.optimization_results.iloc[0]
            
            method = best_params['method']
            method_params = {k: v for k, v in best_params.items() 
                           if k not in ['method', 'forward_periods', 'transaction_cost', 'sharpe_ratio', 
                                      'total_return', 'max_drawdown', 'win_rate', 'total_trades', 'information_ratio']}
            
            regime_performance = {}
            for regime in regime_labels.cat.categories:
                regime_mask = regime_labels == regime
                if regime_mask.sum() > 100:
                    regime_df = self.df[regime_mask].copy()
                    
                    # Generate signals for this regime
                    signals = generate_signals(regime_df['imbalance'], method=method, **method_params)
                    returns = calculate_returns(regime_df, forward_periods=int(best_params['forward_periods']))
                    
                    backtest_result = backtest_strategy(
                        regime_df, signals, returns, 
                        transaction_cost=best_params['transaction_cost']
                    )
                    
                    if 'error' not in backtest_result:
                        regime_performance[regime] = backtest_result
            
            return {
                'regime_stats': regime_stats,
                'regime_performance': regime_performance,
                'regime_labels': regime_labels
            }
        
        return {'regime_stats': regime_stats, 'regime_labels': regime_labels}
    
    def feature_importance_analysis(self, n_levels: int = 5):
        """
        Analyze feature importance for predicting returns using Random Forest
        
        Args:
            n_levels: Number of order book levels to analyze
        
        Returns:
            DataFrame with feature importance
        """
        print("Analyzing feature importance...")
        
        # Create features
        features = []
        feature_names = []
        
        # Order book features
        for i in range(1, n_levels + 1):
            # Bid/Ask volumes
            features.append(self.df[f'BidVolume{i}'].values)
            feature_names.append(f'BidVolume{i}')
            features.append(self.df[f'AskVolume{i}'].values)
            feature_names.append(f'AskVolume{i}')
            
            # Bid/Ask prices (relative to mid)
            mid_price = (self.df['BidPrice1'] + self.df['AskPrice1']) / 2
            features.append((self.df[f'BidPrice{i}'] - mid_price) / mid_price)
            feature_names.append(f'BidPrice{i}_rel')
            features.append((self.df[f'AskPrice{i}'] - mid_price) / mid_price)
            feature_names.append(f'AskPrice{i}_rel')
        
        # Imbalance features
        if 'imbalance' not in self.df.columns:
            self.df['imbalance'] = self.df.apply(
                lambda row: calculate_nlevel_imbalance(row, n_levels), axis=1
            )
        
        features.append(self.df['imbalance'].values)
        feature_names.append('imbalance')
        
        # Lagged imbalance
        for lag in [1, 2, 5, 10]:
            lagged_imbalance = self.df['imbalance'].shift(lag)
            features.append(lagged_imbalance.values)
            feature_names.append(f'imbalance_lag_{lag}')
        
        # Spread features
        spread = self.df['AskPrice1'] - self.df['BidPrice1']
        mid_price = (self.df['BidPrice1'] + self.df['AskPrice1']) / 2
        spread_bps = (spread / mid_price) * 10000
        
        features.append(spread_bps.values)
        feature_names.append('spread_bps')
        
        # Moving averages of imbalance
        for window in [5, 10, 20]:
            ma_imbalance = self.df['imbalance'].rolling(window=window).mean()
            features.append(ma_imbalance.values)
            feature_names.append(f'imbalance_ma_{window}')
        
        # Combine features
        X = np.column_stack(features)
        
        # Target variable (forward returns)
        y = calculate_returns(self.df, forward_periods=1).values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 1000:
            print("Insufficient data for feature importance analysis")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(importance_df.head(10).to_string())
        
        return importance_df
    
    def statistical_tests(self):
        """
        Perform statistical tests on the order imbalance strategy
        
        Returns:
            Dictionary with test results
        """
        print("Performing statistical tests...")
        
        # Calculate imbalance if not available
        if 'imbalance' not in self.df.columns:
            self.df['imbalance'] = self.df.apply(
                lambda row: calculate_nlevel_imbalance(row, 5), axis=1
            )
        
        # Calculate returns
        returns = calculate_returns(self.df, forward_periods=1)
        
        # Remove NaN values
        valid_data = pd.DataFrame({
            'imbalance': self.df['imbalance'],
            'returns': returns
        }).dropna()
        
        if len(valid_data) < 100:
            return {"error": "Insufficient data for statistical tests"}
        
        tests = {}
        
        # 1. Correlation test
        correlation, p_value_corr = stats.pearsonr(valid_data['imbalance'], valid_data['returns'])
        tests['correlation'] = {
            'correlation': correlation,
            'p_value': p_value_corr,
            'significant': p_value_corr < 0.05
        }
        
        # 2. Test for normality of imbalance
        _, p_value_norm = stats.normaltest(valid_data['imbalance'])
        tests['imbalance_normality'] = {
            'p_value': p_value_norm,
            'is_normal': p_value_norm > 0.05
        }
        
        # 3. Test for stationarity (simplified using runs test)
        median_imbalance = valid_data['imbalance'].median()
        runs, n_runs = 0, 1
        for i in range(1, len(valid_data)):
            if ((valid_data['imbalance'].iloc[i] >= median_imbalance) != 
                (valid_data['imbalance'].iloc[i-1] >= median_imbalance)):
                n_runs += 1
        
        # Expected runs under null hypothesis
        n1 = (valid_data['imbalance'] >= median_imbalance).sum()
        n2 = len(valid_data) - n1
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
        
        if var_runs > 0:
            z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
            p_value_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
            tests['runs_test'] = {
                'z_statistic': z_runs,
                'p_value': p_value_runs,
                'is_random': p_value_runs > 0.05
            }
        
        # 4. Autocorrelation test
        def ljung_box_test(series, lags=10):
            """Simplified Ljung-Box test"""
            n = len(series)
            acf_vals = [series.autocorr(lag=i) for i in range(1, lags+1)]
            acf_vals = [x for x in acf_vals if not pd.isna(x)]
            
            if len(acf_vals) > 0:
                lb_stat = n * (n + 2) * sum([(acf**2) / (n - k) for k, acf in enumerate(acf_vals, 1)])
                p_value = 1 - stats.chi2.cdf(lb_stat, len(acf_vals))
                return lb_stat, p_value
            return np.nan, np.nan
        
        lb_stat, p_value_lb = ljung_box_test(valid_data['imbalance'])
        tests['autocorrelation'] = {
            'ljung_box_statistic': lb_stat,
            'p_value': p_value_lb,
            'has_autocorrelation': p_value_lb < 0.05 if not pd.isna(p_value_lb) else None
        }
        
        print("Statistical test results:")
        for test_name, results in tests.items():
            print(f"\n{test_name}:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        return tests
    
    def plot_optimization_results(self, save_plots: bool = False):
        """Plot parameter optimization results"""
        if self.optimization_results is None or len(self.optimization_results) == 0:
            print("No optimization results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sharpe ratio distribution by method
        for method in self.optimization_results['method'].unique():
            method_data = self.optimization_results[self.optimization_results['method'] == method]
            axes[0, 0].hist(method_data['sharpe_ratio'], alpha=0.6, label=method, bins=20)
        axes[0, 0].set_title('Sharpe Ratio Distribution by Method')
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Return vs Risk
        axes[0, 1].scatter(self.optimization_results['max_drawdown'], 
                          self.optimization_results['total_return'],
                          c=self.optimization_results['sharpe_ratio'], 
                          cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Return vs Risk (colored by Sharpe Ratio)')
        axes[0, 1].set_xlabel('Max Drawdown')
        axes[0, 1].set_ylabel('Total Return')
        
        # 3. Win rate vs Total trades
        axes[1, 0].scatter(self.optimization_results['total_trades'], 
                          self.optimization_results['win_rate'],
                          c=self.optimization_results['sharpe_ratio'], 
                          cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Win Rate vs Total Trades (colored by Sharpe Ratio)')
        axes[1, 0].set_xlabel('Total Trades')
        axes[1, 0].set_ylabel('Win Rate')
        
        # 4. Top 10 parameter combinations
        top_10 = self.optimization_results.head(10)
        y_pos = np.arange(len(top_10))
        axes[1, 1].barh(y_pos, top_10['sharpe_ratio'])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"{row['method'][:3]}-{i}" for i, (_, row) in enumerate(top_10.iterrows())])
        axes[1, 1].set_title('Top 10 Parameter Combinations by Sharpe Ratio')
        axes[1, 1].set_xlabel('Sharpe Ratio')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
def run_advanced_analysis(df: pd.DataFrame):
    """
    Run comprehensive advanced analysis
    
    Args:
        df: Prepared DataFrame with order book data
    """
    print("Starting Advanced Order Imbalance Analysis...")
    
    # Initialize advanced analysis
    advanced = AdvancedOrderImbalanceAnalysis(df)
    
    # 1. Parameter optimization
    print("\n" + "="*50)
    print("PARAMETER OPTIMIZATION")
    print("="*50)
    
    # Custom parameter grid for faster execution
    param_grid = {
        'method': ['threshold', 'zscore'],
        'upper_threshold': [0.2, 0.3, 0.4],
        'lower_threshold': [-0.2, -0.3, -0.4],
        'window': [50, 100],
        'threshold': [1.5, 2.0],
        'upper_percentile': [80, 85],
        'lower_percentile': [15, 20],
        'forward_periods': [1, 2],
        'transaction_cost': [0.0001, 0.0002]
    }
    
    optimization_results = advanced.parameter_optimization(param_grid)
    
    if optimization_results is not None and len(optimization_results) > 0:
        advanced.plot_optimization_results(save_plots=True)
    
    # 2. Statistical tests
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    statistical_results = advanced.statistical_tests()
    
    # 3. Feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    feature_importance = advanced.feature_importance_analysis()
    
    # 4. Regime analysis
    print("\n" + "="*50)
    print("REGIME ANALYSIS")
    print("="*50)
    
    regime_results = advanced.regime_analysis('volatility')
    
    if 'regime_performance' in regime_results:
        print("\nPerformance by volatility regime:")
        for regime, performance in regime_results['regime_performance'].items():
            print(f"\n{regime}:")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
            print(f"  Total Return: {performance['total_return']:.4f}")
            print(f"  Win Rate: {performance['win_rate']:.4f}")
    
    return {
        'optimization_results': optimization_results,
        'statistical_results': statistical_results,
        'feature_importance': feature_importance,
        'regime_results': regime_results
    }