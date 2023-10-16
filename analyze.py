import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import tensorflow as tf

# Replace 'YOUR_API_KEY' with your Alpha Vantage API key
api_key = 'HDR7CEO4SGKD9LYL'
symbol = 'AAPL'  # Replace with the stock symbol you want to analyze

# Load the prices data into a pandas DataFrame
with open('data.json', 'r') as f:
    data = json.load(f)
prices = []
for date, values in data['Time Series (Daily)'].items():
    prices.append(float(values['4. close']))
df = pd.DataFrame(prices, columns=['price'])

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mae', 'mse'])

# Train the model
history = model.fit(train_df.index, train_df['price'], epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model
test_predictions = model.predict(test_df.index)
test_predictions = test_predictions * train_std + train_mean
test_mse = tf.keras.losses.mean_squared_error(test_df['price'], test_predictions).numpy()
test_mae = tf.keras.losses.mean_absolute_error(test_df['price'], test_predictions).numpy()
print(f'Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}')

# Use the trained model to predict tomorrow's price
last_price = df.iloc[-1]['price']
normalized_last_price = (last_price - train_mean) / train_std
tomorrow_index = df.index[-1] + 1
tomorrow_normalized_price = model.predict([tomorrow_index])[0][0]
tomorrow_price = tomorrow_normalized_price * train_std + train_mean
print(f'Tomorrow\'s predicted price: {tomorrow_price:.2f}')
