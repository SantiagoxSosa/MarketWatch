import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## Lets get the stock data we want
def get_tickerData(ticker, start, end):
    """Get stock data from yfinance
        Parameters:
        ticker - STOCK TICKER
        start  - start of ticker time observation
        end - end of observation"""
    df = yf.download(ticker, start=start, end=end)
    return df

# Plotting of data
def plot_data(df):
    # Get moving averages
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Candlestick with SMAs', 'Volume'),
                        row_width = [0.7, 0.3])
    
    # Add sticks 
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='OHLC'),
                                 row=1, col=1)
    
    # Add SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'],
                             line=dict(color='blue', width=1),
                                       name='SMA 5'),
                                       row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
                            line=dict(color='green', width=1),
                                    name='SMA 20'),
                                    row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
                            line=dict(color='red', width=1),
                                    name='SMA 50'),
                                    row=1, col=1)
    

def prepare_data(df, look_back=10):
    features= ['Close','SMA5', 'SMA20','SMA50', 'Price_Change']
    df = df.dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df [features])

    X, y =[], []
    for i in range(look_back, len (scaled_data)-1):
        X.append(scaled_data[i-look_back: i]) 
        y.append(df['Target'].iloc[i])
    return np.array(X), np.array (y)