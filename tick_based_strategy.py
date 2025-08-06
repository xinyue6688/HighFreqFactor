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

    # TODO: 交易参数要改
    def __init__(self, data_file: str = 'IC2507.csv', 
                 initial_capital: float = 1000000,  
                 max_position_size: float = 0.1,    
                 stop_loss_pct: float = 0.02,       
                 max_drawdown_pct: float = 0.15):  
        """Initialize imbalanced order strategy"""
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
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
        self.df = self.df.sort_values(by=['TradingDay', 'UpdateTime']).reset_index(drop=True)
        
        if 'TradingDay' in self.df.columns and 'UpdateTime' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['TradingDay'].astype(str) + ' ' + 
                                                 self.df['UpdateTime'].astype(str))
        
        self.df['mid_price'] = (self.df['BidPrice1'] + self.df['AskPrice1']) / 2
        
        self.df['spread'] = self.df['AskPrice1'] - self.df['BidPrice1']
        self.df['spread_bps'] = (self.df['spread'] / self.df['mid_price']) * 10000
        
        self.df['total_volume'] = self.df['BidVolume1'] + self.df['AskVolume1']
        
        print(f"Data loaded: {len(self.df)} records")
        print(f"Date range: {self.df['TradingDay'].min()} to {self.df['TradingDay'].max()}")
        
        return self.df
    
    def calculate_imbalance_metrics(self):
        """Calculate order imbalance metrics"""
        print("Calculating imbalance metrics...")

        bid_price = self.df['BidPrice1']
        ask_price = self.df['AskPrice1']
        bid_volume = self.df['BidVolume1']
        ask_volume = self.df['AskVolume1']

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

        self.df['imbalance'] = delta_bid - delta_ask

        self.df['imbalance_ma_10'] = self.df['imbalance'].rolling(window=10).mean()
        self.df['imbalance_ma_30'] = self.df['imbalance'].rolling(window=30).mean()
        
        self.df['imbalance_volatility'] = self.df['imbalance'].rolling(window=20).std()
        
        return self.df

    def generate_tick_based_signals(self, 
                                   imbalance_threshold: float = 0.65,
                                   min_volume: int = 20,  
                                   max_spread_bps: float = 15.0,  
                                ):
        
        """Generate signals with tick-based constraints"""
        print(f"Generating tick-based signals with threshold {imbalance_threshold}...")
        
        signals = pd.Series(0, index=self.df.index)
        
        volume_filter = self.df['total_volume'] >= min_volume
        spread_filter = self.df['spread_bps'] <= max_spread_bps
        
        
        for i in range(len(self.df)):
            if i < 1:  # Skip first tick
                continue
            
            # Volume and spread filters
            if not (volume_filter.iloc[i] and spread_filter.iloc[i]):
                continue
            
            if self.df['imbalance'].iloc[i] > imbalance_threshold:
                signals.iloc[i] = 1
                last_trade_idx = i
            elif self.df['imbalance'].iloc[i] < -imbalance_threshold:
                signals.iloc[i] = -1
                last_trade_idx = i
        
        self.signals = signals
        self.df['signals'] = signals
        
        # Calculate signal statistics
        total_signals = len(signals[signals != 0])
        buy_signals = len(signals[signals == 1])
        sell_signals = len(signals[signals == -1])
        
        print(f"Signal Statistics:")
        print(f"  Total Signals: {total_signals}")
        print(f"  Buy Signals: {buy_signals}")
        print(f"  Sell Signals: {sell_signals}")
        print(f"  Signal Frequency: {total_signals/len(signals):.4f}")
        
        return signals
    
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
        
        # Generate signals with tick-based parameters
        self.generate_tick_based_signals(
            imbalance_threshold=0.65,
            min_volume=20,
            max_spread_bps=15.0,
        )
        
        # Run tick-based backtest
        self.run_tick_based_backtest(transaction_cost=0.0001)
        
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
        initial_capital=1000000,  # $1M initial capital
        max_position_size=0.1,    # Max 10% per trade
        stop_loss_pct=0.02,       # 2% stop loss
        max_drawdown_pct=0.15     # 15% max drawdown
    )
    
    # Run comprehensive tick analysis
    strategy.run_comprehensive_tick_analysis() 