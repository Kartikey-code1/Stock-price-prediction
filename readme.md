# 📈 Stock Market Prediction Interface

An **interactive Streamlit application** for analyzing stock market data and predicting next-day stock prices using **LSTM deep learning**.  

This project demonstrates a **complete ML workflow**, from data fetching and feature engineering to model training and visualization.

---

## Features

- **Real-time Stock Analysis:**  
  Display key metrics such as Closing Price, MA20, MA50, Volume.

- **Interactive Charts:**  
  - Candlestick charts  
  - Bollinger Bands  
  - RSI (Relative Strength Index)  
  - MACD (Moving Average Convergence Divergence)

- **Next-Day Prediction:**  
  - Predicts next-day stock prices using an LSTM model  
  - Configurable **lookback days** and **training epochs**

- **Data Sources:**  
  - Fetches historical stock data from `yfinance`  
  - Optional integration with `data_loader.fetch_stock_data`  

---

## Tech Stack

- **Python 3.10+**  
- **Streamlit** – Frontend dashboard  
- **Pandas & Numpy** – Data manipulation  
- **Plotly** – Interactive charts  
- **TensorFlow/Keras** – LSTM model for prediction  
- **yfinance** – Historical stock data  

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd StockMarketPrediction
2.Create a virtual environment:

python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


3.Install dependencies:

pip install -r requirements.txt

Usage

1.Run the Streamlit app:

streamlit run app.py


2.Use the sidebar controls:

Enter a ticker symbol (e.g., RELIANCE.NS)

Select indicators to display (MA20, Bollinger Bands, RSI, MACD)

Enter a ticker for next-day prediction and configure lookback days and epochs

3.View metrics, interactive charts, and next-day predicted price in the main panel.

