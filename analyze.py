
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import yfinance as yf

# Download Apple stock data
start_date = "2020-01-01"
end_date = "2023-12-15"
apple_data = yf.download("AAPL", start=start_date, end=end_date)

# Preprocess data
closing_prices = apple_data["Close"]
scaled_prices = (closing_prices - closing_prices.mean()) / closing_prices.std()

# Feature engineering
def create_features(data, window=20):
    features = []
    for i in range(len(data) - window):
        features.append([
            np.array(data[i - window : i]),
            np.mean(data[i - window : i]),
            np.std(data[i - window : i]),
            data[i - window : i].max() - data[i - window : i].min(),
        ])
    return np.array(features)


features = create_features(scaled_prices)

# Split data into training and testing sets
train_size = int(len(features) * 0.8)
X_train, X_test = features[:train_size], features[train_size:]

# Build and train LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(features.shape[1], features.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.fit(X_train, closing_prices[train_size:], epochs=50, batch_size=32)

# Predict future price and calculate buy/sell signal
predicted_price = model.predict(X_test[-1].reshape(1, features.shape[1], features.shape[2]))[0][0]
current_price = closing_prices.iloc[-1]

buy_threshold = 0.03  # Change this based on your risk tolerance
sell_threshold = -0.02  # Change this based on your risk tolerance

buy_signal = "BUY" if predicted_price > current_price * (1 + buy_threshold) else "HOLD"
sell_signal = "SELL" if predicted_price < current_price * (1 + sell_threshold) else "HOLD"

print(f"Predicted price: ${predicted_price:.2f} | Current price: ${current_price:.2f}")
print(f"Buy signal: {buy_signal}")
print(f"Sell signal: {sell_signal}")
