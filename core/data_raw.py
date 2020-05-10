import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
from core.model import Model
from core.data_processor import DataLoader
import os
import json

# pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
def plot_raw_ixic():
    configs = json.load(open('config_ixic.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    f = os.path.join('data', configs['data']['filename'])
    df = pd.read_csv(f, parse_dates=True, index_col=0)

    df_ohlc = df['Adj Close'].resample('30D').ohlc()
    df_volume = df['Volume'].resample('30D').sum()

    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    plt.show()

def plot_cpi():
    configs = json.load(open('config_cpitrnsl.json', 'r'))
    f = os.path.join('data', configs['data']['filename'])
    ax = plt.gca()
    df = pd.read_csv(f)
    df.plot(x='DATE', y='CPITRNSL', kind='line', ax=ax)
    plt.show()