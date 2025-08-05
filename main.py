import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IC2507.csv')
df = df.sort_values(by = ['TradingDay', 'UpdateTime']).reset_index(drop = True)
# UpdateMillisec 字段含义，第200和700毫秒

def calculate_nlevel_imbalance(row, n):
    bid_volume_sum = sum([row[f'BidVolume{i}'] for i in range(1, n + 1)])
    ask_volume_sum = sum([row[f'AskVolume{i}'] for i in range(1, n + 1)])
    if bid_volume_sum + ask_volume_sum == 0:
        return 0
    return (bid_volume_sum - ask_volume_sum) / (bid_volume_sum + ask_volume_sum)

df['order_imbalance'] = df.apply(lambda row: calculate_nlevel_imbalance(row, n=5), axis=1)

def plot_order_imbalance_histogram(order_imbalance, n):
    plt.figure(figsize=(10, 6))
    plt.hist(order_imbalance, bins=50, range=(-1.1, 1.1))
    plt.title(f'订单簿不平衡直方图（L={n}）')
    plt.xlabel('订单簿不平衡值')
    plt.ylabel('频数')
    plt.show()

plot_order_imbalance_histogram(df['order_imbalance'], n=5)

