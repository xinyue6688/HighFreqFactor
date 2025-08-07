import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TickBasedOrderImbalanceStrategy:
    """
    Tick-Based Order Imbalance Strategy
    """

    def __init__(self, data_file: str = 'IC2507.csv',
                 max_lot = 10,
                 backtest_price = ['midprice', 'counterprice'],
                 stop_loss_pct: float = 0.02,       
                 max_drawdown_pct: float = 0.15):  
        """Initialize imbalanced order strategy"""

        self.data_file = data_file
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.df = None
        self.signals = None
        self.backtest_results = None
        self.portfolio_history = None
        
    def load_and_prepare_data(self):
        """Load and prepare the market data without aggregation"""
        print("Loading and preparing tick data...")
        
        self.df = pd.read_csv(self.data_file)
        self.df = self.df.drop(columns=['SeqNo', 'SettlementPrice', ]).reset_index(drop=True)
        
        if 'TradingDay' in self.df.columns and 'UpdateTime' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(
                self.df['TradingDay'].astype(str) + ' ' +
                self.df['UpdateTime'].astype(str)
            )
        millisec_clean = pd.to_numeric(self.df['UpdateMillisec'], errors='coerce').fillna(0)
        millisec_offset = pd.to_timedelta(millisec_clean, unit='ms')
        self.df['timestamp'] += millisec_offset
        self.df = self.df.sort_values(by=['timestamp']).reset_index(drop=True)

        self.df['mid_price'] = (self.df['BidPrice1'] + self.df['AskPrice1']) / 2

        def calculate_return_for_day(group):
            group = group.copy()
            rt = group['mid_price'].shift(-1) / group['mid_price'] - 1
            return rt

        self.df['return'] = self.df.groupby('TradingDay', group_keys=False).apply(calculate_return_for_day)
        
        print(f"Data loaded: {len(self.df)} records")
        print(f"Date range: {self.df['TradingDay'].min()} to {self.df['TradingDay'].max()}")
        
        return self.df
    
    def calculate_imbalance_metrics(self):
        """Calculate order imbalance metrics"""
        print("Calculating imbalance metrics...")

        def calculate_imbalance_for_day(group):
            """Calculate imbalance for a single trading day"""
            bid_price = group['BidPrice1']
            ask_price = group['AskPrice1']
            bid_volume = group['BidVolume1']
            ask_volume = group['AskVolume1']

            bid_price_prev = bid_price.shift(1)
            ask_price_prev = ask_price.shift(1)
            bid_volume_prev = bid_volume.shift(1)
            ask_volume_prev = ask_volume.shift(1)

            delta_bid = (
                (bid_price > bid_price_prev) * bid_volume +
            (bid_price == bid_price_prev) * (bid_volume - bid_volume_prev)
            )

            delta_ask = (
            (ask_price < ask_price_prev) * ask_volume +
            (ask_price == ask_price_prev) * (ask_volume - ask_volume_prev)
            )

            imbalance = delta_bid - delta_ask

            return imbalance

        print("Grouping by trading day and calculating imbalance...")
        self.df['imbalance'] = np.nan

        for trading_day in self.df['TradingDay'].unique():
            day_mask = self.df['TradingDay'] == trading_day
            day_indices = self.df[day_mask].index

            day_data = self.df.loc[day_mask].copy()
            day_imbalance = calculate_imbalance_for_day(day_data)
            self.df.loc[day_indices, 'imbalance'] = day_imbalance

        return self.df

    def calculate_factor_return_corr(self, max_horizon=40, step=2):
        """
        Calculate correlations between imbalance factor and future returns
        at different tick horizons, ensuring calculations stay within trading days

        Parameters:
        -----------
        max_horizon : int
            Maximum number of ticks to look forward
        step : int
            Step size for horizon calculation (e.g., step=5 means check every 5 ticks)

        Returns:
        --------
        pd.DataFrame: DataFrame with horizons and their corresponding correlations
        """
        print(f"Calculating factor-return correlations up to {max_horizon} ticks...")

        if 'imbalance' not in self.df.columns:
            print("Imbalance not calculated. Running calculate_imbalance_metrics first...")
            self.calculate_imbalance_metrics()

        def calculate_returns_for_day(day_data, horizon):
            """Calculate h-tick forward returns within a single trading day"""
            day_data = day_data.reset_index(drop=True)  # Reset for clean indexing

            current_price = day_data['mid_price']
            future_price = current_price.shift(-horizon)

            # Calculate log returns (more stable for financial data)
            returns = np.log(future_price / current_price)

            # Set returns to NaN if we're looking beyond the day's data
            # This prevents using data from the next day
            returns.iloc[-horizon:] = np.nan if horizon > 0 else returns.iloc[-horizon:]

            return returns

        correlations = []
        horizons = range(1, max_horizon + 1, step)

        print("Calculating returns and correlations for each horizon...")

        for h in horizons:
            print(f"Processing horizon {h} ticks...", end='\r')

            # Initialize lists to collect imbalance and return values across all days
            all_imbalance_values = []
            all_return_values = []

            # Process each trading day separately
            for trading_day in self.df['TradingDay'].unique():
                day_mask = self.df['TradingDay'] == trading_day
                day_data = self.df[day_mask].copy()

                if len(day_data) <= h:  # Skip days with insufficient data
                    continue

                # Calculate returns for this day
                day_returns = calculate_returns_for_day(day_data, h)

                # Get imbalance values for this day
                day_imbalance = day_data['imbalance'].reset_index(drop=True)

                # Collect valid (non-NaN) pairs
                valid_mask = ~(day_imbalance.isna() | day_returns.isna())

                if valid_mask.sum() > 5:  # Need at least 5 valid observations per day
                    all_imbalance_values.extend(day_imbalance[valid_mask].values)
                    all_return_values.extend(day_returns[valid_mask].values)

            # Calculate correlation across all valid observations
            if len(all_imbalance_values) > 10:  # Need sufficient total observations
                corr = np.corrcoef(all_imbalance_values, all_return_values)[0, 1]

                correlations.append({
                    'horizon_ticks': h,
                    'correlation': corr,
                    'valid_observations': len(all_imbalance_values),
                    'trading_days_used': len([day for day in self.df['TradingDay'].unique()
                                              if (self.df['TradingDay'] == day).sum() > h])
                })

        print()

        # Create results DataFrame
        corr_results = pd.DataFrame(correlations)

        # Find optimal horizons
        if len(corr_results) > 0:
            best_corr = corr_results.loc[corr_results['correlation'].idxmax()]
            lowest_corr = corr_results.loc[corr_results['correlation'].idxmin()]

            print(f"\nCorrelation Analysis Results:")
            print(
                f"Highest correlation: {best_corr['correlation']:.4f} at {best_corr['horizon_ticks']} ticks")
            print(
                f"  ({best_corr['valid_observations']} observations from {best_corr['trading_days_used']} days)")
            print(
                f"Lowest correlation: {lowest_corr['correlation']:.4f} at {lowest_corr['horizon_ticks']} ticks")
            print(
                f"  ({lowest_corr['valid_observations']} observations from {lowest_corr['trading_days_used']} days)")

        # Store results for later use
        self.correlation_analysis = corr_results

        return corr_results

    def analyze_imbalance_buckets(self, num_buckets=10):
        df = self.df.copy()
        df = df.dropna(subset=['imbalance', 'return'])
        df['imbalance_bucket'] = pd.qcut(df['imbalance'], q=num_buckets, labels=False, duplicates='drop')
        bucket_returns = df.groupby('imbalance_bucket').agg(
            avg_return = ('return', 'mean'),
            imbalance_avg = ('imbalance', 'mean')
        ).reset_index()

        self.bucket_returns = bucket_returns

        df = self.bucket_returns.copy()
        df['avg_return_bps'] = df['avg_return'] * 10000  # Convert to basis points

        plt.figure(figsize=(8, 5))
        plt.bar(
            df['imbalance_bucket'],
            df['avg_return_bps'],
            width=0.6,
            color='skyblue',
            edgecolor='black'
        )

        plt.xlabel('Imbalance Bucket', fontsize=12)
        plt.ylabel('Average Return (bps)', fontsize=12)
        plt.title('Average Return by Imbalance Bucket', fontsize=14)
        plt.xticks(df['imbalance_bucket'], fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

        return bucket_returns

    def generate_signals(self):
        print("Generating tick-based signals using rolling 120-tick window...")

        df = self.df.copy()
        df['signal'] = 0
        window = 120

        for day in df['TradingDay'].unique():
            day_mask = df['TradingDay'] == day
            day_df = df.loc[day_mask].copy()

            imbalances = day_df['imbalance']

            # Rolling percentiles
            q25 = imbalances.rolling(window=window, min_periods=window).quantile(0.25)
            q75 = imbalances.rolling(window=window, min_periods=window).quantile(0.75)

            signal = pd.Series(0, index=day_df.index)
            signal[imbalances > q75] = 1
            signal[imbalances < q25] = -1

            df.loc[day_mask, 'signal'] = signal

        self.df = df

        print("Signal generation complete.")
        print(df['signal'].value_counts().to_dict())
        return df

    def run_trade_execution(self, max_lot=10, stop_loss_pct=0.02, transaction_cost_bps=2):
        print("Running execution logic with transaction costs...")

        df = self.df.copy()
        df['position'] = 0
        df['trade_lot'] = 0
        df['entry_price'] = np.nan
        df['pnl'] = 0.0
        df['cumulative_pnl'] = 0.0

        cumulative_pnl = 0.0
        cost_rate = transaction_cost_bps / 10000  # convert bps to decimal

        for day in df['TradingDay'].unique():
            day_mask = df['TradingDay'] == day
            day_df = df.loc[day_mask].copy()
            indices = day_df.index

            current_position = 0
            entry_price = None

            for i in indices:
                signal = df.at[i, 'signal']
                imbalance = df.at[i, 'imbalance']

                # Open position
                if current_position == 0 and signal != 0 and i >= 120:
                    window = df.loc[i - 120:i - 1, 'imbalance']
                    mean = window.mean()
                    std = window.std()
                    if std > 0:
                        z = (imbalance - mean) / std
                        lot = int(round(np.clip(abs(z) / 3 * max_lot, 0, max_lot)))

                        if lot > 0:
                            current_position = signal * lot
                            entry_price = df.at[i, 'AskPrice1'] if current_position > 0 else df.at[i, 'BidPrice1']

                            # Apply entry cost
                            entry_cost = abs(current_position) * entry_price * cost_rate
                            cumulative_pnl -= entry_cost

                            df.at[i, 'trade_lot'] = current_position
                            df.at[i, 'entry_price'] = entry_price

                # Manage position
                elif current_position != 0:
                    current_price = df.at[i, 'BidPrice1'] if current_position > 0 else df.at[i, 'AskPrice1']
                    return_pct = (current_price - entry_price) / entry_price * np.sign(current_position)

                    # Exit condition
                    if (signal * current_position < 0) or (return_pct < -stop_loss_pct):
                        pnl = return_pct * abs(current_position)
                        exit_cost = abs(current_position) * current_price * cost_rate
                        pnl -= exit_cost  # subtract transaction cost

                        cumulative_pnl += pnl
                        df.at[i, 'pnl'] = pnl
                        df.at[i, 'trade_lot'] = -current_position
                        current_position = 0
                        entry_price = None
                    else:
                        df.at[i, 'position'] = current_position

                df.at[i, 'cumulative_pnl'] = cumulative_pnl

            # End-of-day force close
            if current_position != 0:
                last_idx = indices[-1]
                close_price = df.at[last_idx, 'BidPrice1'] if current_position > 0 else df.at[last_idx, 'AskPrice1']
                return_pct = (close_price - entry_price) / entry_price * np.sign(current_position)
                pnl = return_pct * abs(current_position)
                exit_cost = abs(current_position) * close_price * cost_rate
                pnl -= exit_cost

                cumulative_pnl += pnl

                df.at[last_idx, 'pnl'] = pnl
                df.at[last_idx, 'trade_lot'] = -current_position
                df.at[last_idx, 'position'] = 0
                df.at[last_idx, 'cumulative_pnl'] = cumulative_pnl

                current_position = 0
                entry_price = None

        self.df = df
        print(f"Execution complete. Final Cumulative PnL: {cumulative_pnl:.2f}")

        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative PnL', linewidth=1.5)
        plt.xlabel('Time')
        plt.ylabel('Cumulative PnL')
        plt.title('Cumulative PnL Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return df

    def run_tick_based_backtest(self, transaction_cost: float = 0.0001):
        """Run backtest with tick-based position sizing and risk management"""
        print("Running tick-based backtest...")
        
        if self.signals is None:
            raise ValueError("Signals not generated. Call generate_tick_based_signals first.")
        
        self.df['returns'] = self.df['mid_price'].pct_change()
        
        portfolio = self._initialize_portfolio()
        
        portfolio_history = self._run_portfolio_simulation(portfolio, transaction_cost)
        
        self.backtest_results = self._calculate_performance_metrics(portfolio_history)
        
        self._print_backtest_results()
        
        return self.backtest_results
    
    def _initialize_portfolio(self):
        """Initialize portfolio with risk management parameters"""
        return {
            'capital': self.initial_capital,
            'position': 0,  # Current position size
            'entry_price': 0,  # Entry price for current position
            'max_position_value': self.initial_capital * self.max_position_size,
            'stop_loss_price': 0,  # Stop loss price
            'max_capital': self.initial_capital,  # Peak capital for drawdown calculation
            'trades': [],
            'equity_curve': []
        }
    
    def _run_portfolio_simulation(self, portfolio, transaction_cost):
        """Run portfolio simulation with position sizing and risk management"""
        portfolio_history = []
        
        for i in range(len(self.df)):
            current_price = self.df['mid_price'].iloc[i]
            signal = self.signals.iloc[i]
            returns = self.df['returns'].iloc[i] if i > 0 else 0
            
            if portfolio['position'] != 0:
                if portfolio['position'] > 0:  # Long position
                    pnl = portfolio['position'] * returns
                else:  # Short position
                    pnl = -portfolio['position'] * returns
                
                portfolio['capital'] += pnl
                
                # Check stop loss
                if self._check_stop_loss(portfolio, current_price):
                    portfolio = self._close_position(portfolio, current_price, transaction_cost)
            
            # Check max drawdown
            if portfolio['capital'] > portfolio['max_capital']:
                portfolio['max_capital'] = portfolio['capital']
            
            current_drawdown = (portfolio['max_capital'] - portfolio['capital']) / portfolio['max_capital']
            if current_drawdown > self.max_drawdown_pct:
                if portfolio['position'] != 0:
                    portfolio = self._close_position(portfolio, current_price, transaction_cost)
                break
            
            if signal != 0 and portfolio['position'] == 0:
                portfolio = self._open_position(portfolio, current_price, signal, transaction_cost)
            
            if i % 120 == 0:
                portfolio_history.append({
                    'timestamp': self.df.index[i],
                    'price': current_price,
                    'signal': signal,
                    'position': portfolio['position'],
                    'capital': portfolio['capital'],
                    'equity': portfolio['capital'] + (portfolio['position'] * current_price),
                    'drawdown': current_drawdown
                })
        
        return pd.DataFrame(portfolio_history)
    
    def _open_position(self, portfolio, price, signal, transaction_cost):
        """Open a new position with position sizing"""
        
        position_value = min(portfolio['max_position_value'], portfolio['capital'] * 0.1)
        position_size = position_value / price
        
        if signal == -1:  # Short position
            position_size = -position_size
        
        transaction_value = abs(position_size) * price * transaction_cost
        portfolio['capital'] -= transaction_value
        
        portfolio['position'] = position_size
        portfolio['entry_price'] = price
        portfolio['stop_loss_price'] = price * (1 - self.stop_loss_pct) if signal == 1 else price * (1 + self.stop_loss_pct)
        
        portfolio['trades'].append({
            'type': 'open',
            'price': price,
            'size': position_size,
            'capital': portfolio['capital']
        })
        
        return portfolio
    
    def _close_position(self, portfolio, price, transaction_cost):
        """Close current position"""
        if portfolio['position'] == 0:
            return portfolio
        
        # Calculate P&L
        if portfolio['position'] > 0:  # Long position
            pnl = portfolio['position'] * (price - portfolio['entry_price'])
        else:  # Short position
            pnl = -portfolio['position'] * (price - portfolio['entry_price'])
        
        # Apply transaction costs
        transaction_value = abs(portfolio['position']) * price * transaction_cost
        portfolio['capital'] += pnl - transaction_value
        
        portfolio['trades'].append({
            'type': 'close',
            'price': price,
            'size': portfolio['position'],
            'pnl': pnl,
            'capital': portfolio['capital']
        })
        
        # Reset position
        portfolio['position'] = 0
        portfolio['entry_price'] = 0
        portfolio['stop_loss_price'] = 0
        
        return portfolio
    
    def _check_stop_loss(self, portfolio, current_price):
        """Check if stop loss is triggered"""
        if portfolio['position'] == 0:
            return False
        
        if portfolio['position'] > 0:  # Long position
            return current_price <= portfolio['stop_loss_price']
        else:  # Short position
            return current_price >= portfolio['stop_loss_price']
    
    def _calculate_performance_metrics(self, portfolio_history):
        """Calculate comprehensive performance metrics"""
        if len(portfolio_history) == 0:
            return {"error": "No portfolio history"}
        
        # Calculate returns
        portfolio_history['returns'] = portfolio_history['equity'].pct_change()
        
        # Basic metrics
        total_return = (portfolio_history['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** 12 - 1  # Assuming minute data
        volatility = portfolio_history['returns'].std() * np.sqrt(252 * 24 * 60)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_history['equity'].expanding().max()
        drawdown = (portfolio_history['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate - handle case where no trades occurred
        if hasattr(portfolio_history, 'trades'):
            trades = [t for t in portfolio_history['trades'] if t['type'] == 'close']
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            total_trades = len(trades)
        else:
            win_rate = 0
            total_trades = 0
        
        # Risk metrics
        var_95 = portfolio_history['returns'].quantile(0.05)
        max_loss = portfolio_history['returns'].min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'var_95': var_95,
            'max_loss': max_loss,
            'final_capital': portfolio_history['equity'].iloc[-1],
            'portfolio_history': portfolio_history
        }
    
    def _print_backtest_results(self):
        """Print comprehensive backtest results"""
        if 'error' in self.backtest_results:
            print(f"Backtest Error: {self.backtest_results['error']}")
            return
        
        print(f"\nTick-Based Backtest Results:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital: ${self.backtest_results['final_capital']:,.2f}")
        print(f"  Total Return: {self.backtest_results['total_return']:.4f} ({self.backtest_results['total_return']*100:.2f}%)")
        print(f"  Annualized Return: {self.backtest_results['annualized_return']:.4f} ({self.backtest_results['annualized_return']*100:.2f}%)")
        print(f"  Volatility: {self.backtest_results['volatility']:.4f} ({self.backtest_results['volatility']*100:.2f}%)")
        print(f"  Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {self.backtest_results['max_drawdown']:.4f} ({self.backtest_results['max_drawdown']*100:.2f}%)")
        # print(f"  Win Rate: {self.backtest_results['win_rate']:.4f} ({self.backtest_results['win_rate']*100:.2f}%)")
        # print(f"  Total Trades: {self.backtest_results['total_trades']}")
        # print(f"  VaR (95%): {self.backtest_results['var_95']:.4f} ({self.backtest_results['var_95']*100:.2f}%)")
        # print(f"  Max Loss: {self.backtest_results['max_loss']:.4f} ({self.backtest_results['max_loss']*100:.2f}%)")
    
    def plot_tick_based_results(self, save_plots: bool = False):
        """Plot tick-based strategy results"""
        if self.backtest_results is None or 'error' in self.backtest_results:
            print("No valid backtest results to plot")
            return
        
        portfolio_history = self.backtest_results['portfolio_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity curve
        axes[0, 0].plot(portfolio_history.index, portfolio_history['equity'], color='blue', linewidth=2)
        axes[0, 0].axhline(self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title('Tick-Based Portfolio Equity Curve')
        axes[0, 0].set_xlabel('Time Index (Sampled)')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        rolling_max = portfolio_history['equity'].expanding().max()
        drawdown = (portfolio_history['equity'] - rolling_max) / rolling_max
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.6, color='red')
        axes[0, 1].set_title('Tick-Based Portfolio Drawdown')
        axes[0, 1].set_xlabel('Time Index (Sampled)')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Position size over time
        axes[1, 0].plot(portfolio_history.index, portfolio_history['position'], color='green', linewidth=1)
        axes[1, 0].set_title('Tick-Based Position Size Over Time')
        axes[1, 0].set_xlabel('Time Index (Sampled)')
        axes[1, 0].set_ylabel('Position Size')
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Returns distribution
        returns = portfolio_history['equity'].pct_change().dropna()
        axes[1, 1].hist(returns, bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('Tick-Based Portfolio Returns Distribution')
        axes[1, 1].set_xlabel('Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {returns.mean():.6f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('tick_based_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_signals_on_price(self, save_plot: bool = False):
        """Plot all buy and sell signals on the price line"""
        if self.df is None or 'signals' not in self.df.columns:
            print("No signals to plot.")
            return
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.df['mid_price'], label='Mid Price', color='black', linewidth=1)
        
        buy_signals = self.df[self.df['signals'] == 1]
        sell_signals = self.df[self.df['signals'] == -1]
        
        ax.scatter(buy_signals.index, buy_signals['mid_price'], color='green', marker='^', s=60, label='Buy Signal')
        ax.scatter(sell_signals.index, sell_signals['mid_price'], color='red', marker='v', s=60, label='Sell Signal')
        
        ax.set_title('Tick-Based Buy/Sell Signals on Price Line')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Mid Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plot:
            plt.savefig('tick_signals_on_price.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_comprehensive_tick_analysis(self):
        """Run comprehensive tick-based analysis"""
        print("="*60)
        print("TICK-BASED ORDER IMBALANCE STRATEGY ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Calculate imbalance metrics
        self.calculate_imbalance_metrics()

        # Calculate correlations between imbalance metrics and returns
        # self.calculate_factor_return_corr()

        # Imbalance buckets and returns
        self.analyze_imbalance_buckets()

        # Generate signals with tick-based parameters
        self.generate_signals()
        

        self.run_trade_execution()
        
        # Create plots
        print("\nGenerating tick-based analysis plots...")
        self.plot_tick_based_results(save_plots=True)
        self.plot_signals_on_price(save_plot=True)
        
        print("\nTick-based analysis complete!")


# Main execution
if __name__ == "__main__":
    # Create tick-based strategy instance
    strategy = TickBasedOrderImbalanceStrategy(
        data_file='IC2507.csv',
        stop_loss_pct=0.02,       # 2% stop loss
        max_drawdown_pct=0.15     # 15% max drawdown
    )
    
    # Run comprehensive tick analysis
    strategy.run_comprehensive_tick_analysis() 