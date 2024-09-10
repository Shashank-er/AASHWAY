import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM # type: ignore

# List of stock tickers
stock_list = ['GOOGL', 'AAPL', 'MSFT']  # You can add any other stocks you'd like

# fetching stock data, train model, and predict next day high/low
def predict_stock_price(stock_ticker):
    print(f"\nFetching and predicting for: {stock_ticker}")
    
    # Fetch stock data from yfinance (last 90 days for training)
    df = yf.download(stock_ticker, period='3mo', interval='1d')

    # Check if data is available
    if df.empty:
        print(f"No data found for {stock_ticker}. Skipping...")
        return
    
    # Fetch today's Open, high, low and Close(the latest row)

    todays_open = df['Open'].iloc[-1]
    todays_high = df['High'].iloc[-1]
    todays_low = df['Low'].iloc[-1]
    todays_close = df['Close'].iloc[-1]
    
    # Preparing data for prediction (using 'Open', 'High', 'Low', 'Close' features)
    data = df[['Open', 'High', 'Low', 'Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Preparing input and output for LSTM
    X_train = []
    y_train = []
    for i in range(60, len(scaled_data)-1):
        X_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i+1, :])  # Predicting all 4 features (Open, High, Low, Close)
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))  # Reshaping for LSTM (4 features: Open, High, Low, Close)

    # Building and training LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 4)))  # 4 features (Open, High, Low, Close)
    model.add(LSTM(units=50))
    model.add(Dense(units=4))  # Four outputs: next day's Open, High, Low, Close

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=15)

    # Predict next day's open, high, low, and close
    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    prediction = model.predict(X_test)

    # Inverse transform the predicted Open, High, Low, Close values
    predicted_open, predicted_high, predicted_low, predicted_close = scaler.inverse_transform(prediction)[0]

    # Printing today's and predicted values
    print(f"{stock_ticker}")
    print(f"Today's Open:{todays_open:.2f}, Today's High: {todays_high:.2f}, Today's Low: {todays_low:.2f}, Today's Close:{todays_close:.2f}")
    print(f"Predicted Next Day's Open: {predicted_open:.2f}, High: {predicted_high:.2f}, Low: {predicted_low:.2f}, Close: {predicted_close:.2f}")

# Looping through all stocks and predicts
for stock in stock_list:
    predict_stock_price(stock)
