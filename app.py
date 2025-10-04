# app.py
import os
from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TensorFlow & sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# numeric safety
pd.options.mode.chained_assignment = None

# Data fetch (project or yfinance fallback)
try:
    from data_loader import fetch_stock_data
except Exception:
    fetch_stock_data = None

USE_YFINANCE = False
if fetch_stock_data is None:
    try:
        import yfinance as yf
        USE_YFINANCE = True
    except Exception:
        USE_YFINANCE = False
# UI config
st.set_page_config(page_title="Stock Market Prediction Interface", layout="wide")
st.title("📈 STOCK PRICE PREDICTION")

# ---------- Helpers ----------
@st.cache_data(ttl=3600)
def cached_fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    if fetch_stock_data:
        return fetch_stock_data(ticker, start=start, end=end)
    if USE_YFINANCE:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df.index = pd.to_datetime(df.index)
        return df
    raise RuntimeError("No data fetcher available.")

def calc_indicators(df):
    df = df.copy()
    if 'MA20' not in df: df['MA20'] = df['Close'].rolling(20).mean()
    if 'STD20' not in df: df['STD20'] = df['Close'].rolling(20).std()
    if 'BB_high' not in df: df['BB_high'] = df['MA20'] + 2*df['STD20']
    if 'BB_low' not in df: df['BB_low'] = df['MA20'] - 2*df['STD20']
    if 'RSI' not in df:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0,np.nan)
        df['RSI'] = 100 - 100/(1+rs)
        df['RSI'] = df['RSI'].fillna(50)
    if 'MACD' not in df:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def trend_arrow(curr, prev):
    if curr is None or prev is None or np.isnan(curr) or np.isnan(prev):
        return "N/A", "off"
    if curr > prev: return f"↑ {curr:.2f}", "normal"
    if curr < prev: return f"↓ {curr:.2f}", "inverse"
    return f"→ {curr:.2f}", "off"

# ---------- LSTM functions ----------
def prepare_lstm_data(series, lookback=60):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y

def build_and_train_model(ticker, lookback_days=60, epochs=5):
    df_pred = cached_fetch_data(ticker, start=(datetime.today().date() - timedelta(days=365)).isoformat(),
                                end=(datetime.today().date() + timedelta(days=1)).isoformat())
    df_pred = df_pred.sort_index()
    series = df_pred['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)
    X, y = prepare_lstm_data(series_scaled, lookback=lookback_days)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    last_seq = series_scaled[-lookback_days:].reshape(1,lookback_days,1)
    next_scaled = model.predict(last_seq, verbose=0)
    next_price = scaler.inverse_transform(next_scaled)[0][0]
    return next_price

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    ticker_input = st.text_input("Ticker for analysis", "").upper().strip()
    analyze_btn = st.button("Analyze")
    st.markdown("---")
    show_ma = st.checkbox("Show MA20", True)
    show_bb = st.checkbox("Show Bollinger Bands", True)
    show_rsi = st.checkbox("Show RSI", True)
    show_macd = st.checkbox("Show MACD", True)
    st.markdown("---")
    st.header("Next-Day Prediction")
    pred_ticker_input = st.text_input("Ticker for prediction", "").upper().strip()
    pred_lookback = st.slider("Lookback days (LSTM)", 20, 120, 60)
    pred_epochs = st.slider("LSTM epochs", 1, 30, 5)
    predict_btn = st.button("Predict Next-Day Price")

# ---------- Main ----------
start_date = (datetime.today().date() - timedelta(days=365)).isoformat()
end_date = (datetime.today().date() + timedelta(days=1)).isoformat()
df = None

if analyze_btn and ticker_input:
    try:
        df = cached_fetch_data(ticker_input, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found.")
        else:
            df = calc_indicators(df)
            st.success(f"Data fetched: {len(df)} records")
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df)>1 else latest

            # ---------- Metrics ----------
            col1,col2,col3,col4 = st.columns(4)
            for col,label,field in zip([col1,col2,col3,col4],
                                        ["Close","MA20","MA50","Volume"],
                                        ["Close","MA20","MA50","Volume"]):
                curr_val = float(latest.get(field,np.nan))
                prev_val = float(prev.get(field,np.nan))
                trend,color = trend_arrow(curr_val, prev_val)
                col.metric(label, f"{curr_val:.2f}", delta=trend, delta_color=color)

            # ---------- Plotly chart ----------
            rows = 3 if show_rsi or show_macd else 1
            row_heights = [0.5,0.25,0.25] if rows==3 else [1]
            main_chart = make_subplots(rows=rows,cols=1,shared_xaxes=True,
                                       vertical_spacing=0.02,row_heights=row_heights)
            main_chart.add_trace(go.Scatter(x=df.index,y=df['Close'],name='Close',line=dict(color='blue')),row=1,col=1)
            if show_ma and 'MA20' in df: main_chart.add_trace(go.Scatter(x=df.index,y=df['MA20'],name='MA20',line=dict(color='orange')),row=1,col=1)
            if show_bb and 'BB_high' in df and 'BB_low' in df:
                main_chart.add_trace(go.Scatter(x=df.index,y=df['BB_high'],name='BB High',line=dict(color='green',dash='dot')),row=1,col=1)
                main_chart.add_trace(go.Scatter(x=df.index,y=df['BB_low'],name='BB Low',line=dict(color='red',dash='dot')),row=1,col=1)
            if show_rsi and 'RSI' in df: main_chart.add_trace(go.Scatter(x=df.index,y=df['RSI'],name='RSI',line=dict(color='purple')),row=2,col=1)
            if show_macd and 'MACD' in df:
                main_chart.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD',line=dict(color='black')),row=3,col=1)
                main_chart.add_trace(go.Scatter(x=df.index,y=df['MACD_signal'],name='MACD Signal',line=dict(color='orange')),row=3,col=1)
            main_chart.update_layout(height=700,showlegend=True,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(main_chart,use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

# ---------- Next-Day Prediction ----------
if predict_btn:
    if pred_ticker_input:
        try:
            with st.spinner("Predicting next-day price..."):
                predicted_price = build_and_train_model(pred_ticker_input, lookback_days=pred_lookback, epochs=pred_epochs)
                st.success(f"Next-day predicted price for {pred_ticker_input}: {predicted_price:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Enter a ticker for prediction!")
