import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as data
from keras.models import load_model
from neuralprophet import NeuralProphet
import streamlit as st
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress TensorFlow info and warning messages
tf.get_logger().setLevel('ERROR')

# Customize title font color
st.markdown("<h1 style='text-align: center; color: red;'>Smart Stock Trading System</h1>", unsafe_allow_html=True)

# Add the developer information
st.markdown("<p style='text-align: center;  color: skyblue; '>Developed by Subhradeep Nath </p>", unsafe_allow_html=True)



user_input = st.text_input('Enter Stock Ticker', 'AAPL')


# Fetch historical data for the specified stock
stock_data = yf.download(user_input, start='2016-01-01', end='2024-06-03')
# Reset index to turn 'Date' into a regular column
stock_data.reset_index(inplace=True)

# Drop the 'Date' column and the 'Adj Close' column
modified_stock_data = stock_data.drop(columns=['Adj Close'])


#Describing Data
st.subheader('Stock Analysis Data from 2015 - 2024' )
st.dataframe(modified_stock_data)

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(stock_data.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = stock_data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = stock_data.Close.rolling(100).mean()
ma200 = stock_data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(stock_data.Close, 'b')
st.pyplot(fig)

#Splitting Data into Training and Testing
data_training = pd.DataFrame(stock_data['Close'][0:int(len(stock_data)*0.70)])
data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.70): int(len(stock_data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Load my model
model_path = 'C:/Users/Subhradeep Nath/OneDrive/Desktop/Stock Trend Predictor/stock trend prediction/streamlit/keras_model.h5'
model = tf.keras.models.load_model(model_path)


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

if model.compiled_metrics is not None:
    st.write("Model has been compiled with metrics")
else:
    st.write("Model has not been compiled with metrics")

#Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph

st.subheader('Predictions vs Original Chart (Days vs Price)')
fig2 = plt.figure(figsize=(12,6))
# Assuming y_test and y_predicted are arrays with corresponding timestamps or indices representing days
days = range(len(y_test))  # Generate days as indices if not available
plt.plot(days, y_test, 'g', label='Original Price')
plt.plot(days, y_predicted, 'r', label='Predicted Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

# Display the Matplotlib figure using st.pyplot()
st.pyplot(fig2)

# Fit the NeuralProphet model
stocks = modified_stock_data[['Date', 'Close']].copy()
stocks.columns = ['ds', 'y']
model_neuralprophet = NeuralProphet( epochs=50)
model_neuralprophet.fit(stocks)

# Make predictions with NeuralProphet
future = model_neuralprophet.make_future_dataframe(stocks, periods=300)
forecast = model_neuralprophet.predict(future)
actual_prediction = model_neuralprophet.predict(stocks)

# Plot predictions from the NeuralProphet model
st.subheader('Future Trend Forecasting (Year vs Price)')
fig3 = plt.figure(figsize=(18, 9))
plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label='Actual Prediction', c='r')
plt.plot(forecast['ds'], forecast['yhat1'], label='Future Prediction', c='b')
plt.plot(stocks['ds'], stocks['y'], label='Original', c='g')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)


