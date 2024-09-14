**Stock Price Prediction Using LSTM**

In this project, we use Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), to predict the next day’s high and low stock prices for a given list of stocks using historical data. We fetch real-time stock data from Yahoo Finance using the `yfinance` library and apply a predictive model built with the Keras library.

**1. Data Collection**
We first gather historical stock data for 60 days using the `yfinance` API. The data includes four key attributes: Open, High, Low, and Close prices. These prices are normalized using the `MinMaxScaler` from the `sklearn.preprocessing` module to scale the data between 0 and 1, which makes it suitable for the LSTM model.

df = yf.download(stock_ticker, period='60d', interval='1d')
data = df[['Open', 'High', 'Low', 'Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

**2. Data Preparation**
LSTMs work on sequential data, so we prepare input data by using the previous 60 days of stock prices to predict the next day’s high and low. We create separate lists for the input data (`X_train`), and the high and low price predictions (`y_high_train`, `y_low_train`). 

`X_train`: A 3D array where each row contains 60 days of stock price data.
`y_high_train` and `y_low_train`: Contain the next day’s high and low prices respectively.


X_train = []
y_high_train = []
y_low_train = []
for i in range(60, len(scaled_data)-1):
    X_train.append(scaled_data[i-60:i])
    y_high_train.append(scaled_data[i+1, 1])  # Next day high
    y_low_train.append(scaled_data[i+1, 2])   # Next day low

The input data is then reshaped to match the input requirements of the LSTM model, which takes in 3D data of the shape (number of samples, time steps, features).

**3. Model Building**

We build an LSTM model using the `Sequential` API from Keras. The model consists of:
Two LSTM layers: Each with 50 units. The first layer returns sequences, while the second does not.
Dense layer: With 2 units to predict the next day’s high and low prices simultaneously.


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 4)))  # Input has 4 features
model.add(LSTM(units=50))
model.add(Dense(units=2))  # Output 2 values (next day's high and low)
```

The model is compiled with the `adam` optimizer and `mean_squared_error` loss function, which is common for regression problems.

**4. Model Training**
We train the model on the input data (`X_train`) with the corresponding output labels (`y_high_train` and `y_low_train`). The model runs for 10 epochs with a batch size of 64.

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, {'dense_1': y_high_train, 'dense_2': y_low_train}, batch_size=64, epochs=10)

**5. Prediction**
Once the model is trained, we use the last 60 days of stock data to predict the next day’s high and low prices. The scaled predictions are then transformed back to the original scale using `scaler.inverse_transform`.


last_60_days = scaled_data[-60:]
X_test = np.array([last_60_days])
prediction = model.predict(X_test)
predicted_high, predicted_low = scaler.inverse_transform(prediction)[0]

**6. Output**
Finally, we display the actual high and low prices for the current day and the predicted high and low for the next day.


print(f"{stock_ticker}")
print(f"Today's High: {todays_high:.2f}, Today's Low: {todays_low:.2f}")
print(f"Predicted Next Day's High: {predicted_high:.2f}, Predicted Next Day's Low: {predicted_low:.2f}")

**7. Looping Through Multiple Stocks**
The entire process (data fetching, model training, and prediction) is repeated for each stock in the `stock_list`, allowing for predictions on multiple stocks. The stock tickers are stored in a list, and the `predict_stock_price` function is called for each stock ticker in the loop.


stock_list = ['GOOGL', 'AAPL', 'MSFT']
for stock in stock_list:
    predict_stock_price(stock)

This approach uses machine learning to predict stock prices based on historical data, but it is important to note that stock markets are influenced by numerous factors, many of which cannot be captured by past data alone. Thus, predictions may not always be accurate and should be used cautiously.

