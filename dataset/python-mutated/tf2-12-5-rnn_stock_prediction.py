"""
This script shows how to predict stock prices using a basic RNN
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def MinMaxScaler(data):
    if False:
        for i in range(10):
            print('nop')
    ' Min Max Normalization\n\n    Parameters\n    ----------\n    data : numpy.ndarray\n        input data to be normalized\n        shape: [Batch size, dimension]\n\n    Returns\n    ----------\n    data : numpy.ndarry\n        normalized data\n        shape: [Batch size, dimension]\n\n    References\n    ----------\n    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html\n\n    '
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-07)
seq_length = 7
data_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500
xy = np.loadtxt('../data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

def build_dataset(time_series, seq_length):
    if False:
        print('Hello World!')
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, :]
        y = time_series[i + seq_length, [-1]]
        print(x, '->', y)
        dataX.append(x)
        dataY.append(y)
    return (np.array(dataX), np.array(dataY))
(trainX, trainY) = build_dataset(train_set, seq_length)
(testX, testY) = build_dataset(test_set, seq_length)
print(trainX.shape)
print(trainY.shape)
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=1, input_shape=(seq_length, data_dim)))
tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
tf.model.summary()
tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
tf.model.fit(trainX, trainY, epochs=iterations)
test_predict = tf.model.predict(testX)
plt.plot(testY)
plt.plot(test_predict)
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.show()