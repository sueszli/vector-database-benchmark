"""
Title: Timeseries anomaly detection using an Autoencoder
Author: [pavithrasv](https://github.com/pavithrasv)
Date created: 2020/05/31
Last modified: 2020/05/31
Description: Detect anomalies in a timeseries using an Autoencoder.
Accelerator: GPU
"""
'\n## Introduction\n\nThis script demonstrates how you can use a reconstruction convolutional\nautoencoder model to detect anomalies in timeseries data.\n'
'\n## Setup\n'
import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt
'\n## Load the data\n\nWe will use the [Numenta Anomaly Benchmark(NAB)](\nhttps://www.kaggle.com/boltzmannbrain/nab) dataset. It provides artificial\ntimeseries data containing labeled anomalous periods of behavior. Data are\nordered, timestamped, single-valued metrics.\n\nWe will use the `art_daily_small_noise.csv` file for training and the\n`art_daily_jumpsup.csv` file for testing. The simplicity of this dataset\nallows us to demonstrate anomaly detection effectively.\n'
master_url_root = 'https://raw.githubusercontent.com/numenta/NAB/master/data/'
df_small_noise_url_suffix = 'artificialNoAnomaly/art_daily_small_noise.csv'
df_small_noise_url = master_url_root + df_small_noise_url_suffix
df_small_noise = pd.read_csv(df_small_noise_url, parse_dates=True, index_col='timestamp')
df_daily_jumpsup_url_suffix = 'artificialWithAnomaly/art_daily_jumpsup.csv'
df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(df_daily_jumpsup_url, parse_dates=True, index_col='timestamp')
'\n## Quick look at the data\n'
print(df_small_noise.head())
print(df_daily_jumpsup.head())
'\n## Visualize the data\n### Timeseries data without anomalies\n\nWe will use the following data for training.\n'
(fig, ax) = plt.subplots()
df_small_noise.plot(legend=False, ax=ax)
plt.show()
'\n### Timeseries data with anomalies\n\nWe will use the following data for testing and see if the sudden jump up in the\ndata is detected as an anomaly.\n'
(fig, ax) = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
plt.show()
'\n## Prepare training data\n\nGet data values from the training timeseries data file and normalize the\n`value` data. We have a `value` for every 5 mins for 14 days.\n\n-   24 * 60 / 5 = **288 timesteps per day**\n-   288 * 14 = **4032 data points** in total\n'
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print('Number of training samples:', len(df_training_value))
'\n### Create sequences\nCreate sequences combining `TIME_STEPS` contiguous data values from the\ntraining data.\n'
TIME_STEPS = 288

def create_sequences(values, time_steps=TIME_STEPS):
    if False:
        return 10
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i:i + time_steps])
    return np.stack(output)
x_train = create_sequences(df_training_value.values)
print('Training input shape: ', x_train.shape)
'\n## Build a model\n\nWe will build a convolutional reconstruction autoencoder model. The model will\ntake input of shape `(batch_size, sequence_length, num_features)` and return\noutput of the same shape. In this case, `sequence_length` is 288 and\n`num_features` is 1.\n'
model = keras.Sequential([layers.Input(shape=(x_train.shape[1], x_train.shape[2])), layers.Conv1D(filters=32, kernel_size=7, padding='same', strides=2, activation='relu'), layers.Dropout(rate=0.2), layers.Conv1D(filters=16, kernel_size=7, padding='same', strides=2, activation='relu'), layers.Conv1DTranspose(filters=16, kernel_size=7, padding='same', strides=2, activation='relu'), layers.Dropout(rate=0.2), layers.Conv1DTranspose(filters=32, kernel_size=7, padding='same', strides=2, activation='relu'), layers.Conv1DTranspose(filters=1, kernel_size=7, padding='same')])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()
'\n## Train the model\n\nPlease note that we are using `x_train` as both the input and the target\nsince this is a reconstruction model.\n'
history = model.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')])
"\nLet's plot training and validation loss to see how the training went.\n"
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
"\n## Detecting anomalies\n\nWe will detect anomalies by determining how well our model can reconstruct\nthe input data.\n\n\n1.   Find MAE loss on training samples.\n2.   Find max MAE loss value. This is the worst our model has performed trying\nto reconstruct a sample. We will make this the `threshold` for anomaly\ndetection.\n3.   If the reconstruction loss for a sample is greater than this `threshold`\nvalue then we can infer that the model is seeing a pattern that it isn't\nfamiliar with. We will label this sample as an `anomaly`.\n\n\n"
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('No of samples')
plt.show()
threshold = np.max(train_mae_loss)
print('Reconstruction error threshold: ', threshold)
"\n### Compare recontruction\n\nJust for fun, let's see how our model has recontructed the first sample.\nThis is the 288 timesteps from day 1 of our training dataset.\n"
plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.show()
'\n### Prepare test data\n'
df_test_value = (df_daily_jumpsup - training_mean) / training_std
(fig, ax) = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()
x_test = create_sequences(df_test_value.values)
print('Test input shape: ', x_test.shape)
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape(-1)
plt.hist(test_mae_loss, bins=50)
plt.xlabel('test MAE loss')
plt.ylabel('No of samples')
plt.show()
anomalies = test_mae_loss > threshold
print('Number of anomaly samples: ', np.sum(anomalies))
print('Indices of anomaly samples: ', np.where(anomalies))
"\n## Plot anomalies\n\nWe now know the samples of the data which are anomalies. With this, we will\nfind the corresponding `timestamps` from the original test data. We will be\nusing the following method to do that:\n\nLet's say time_steps = 3 and we have 10 training values. Our `x_train` will\nlook like this:\n\n- 0, 1, 2\n- 1, 2, 3\n- 2, 3, 4\n- 3, 4, 5\n- 4, 5, 6\n- 5, 6, 7\n- 6, 7, 8\n- 7, 8, 9\n\nAll except the initial and the final time_steps-1 data values, will appear in\n`time_steps` number of samples. So, if we know that the samples\n[(3, 4, 5), (4, 5, 6), (5, 6, 7)] are anomalies, we can say that the data point\n5 is an anomaly.\n"
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1:data_idx]):
        anomalous_data_indices.append(data_idx)
"\nLet's overlay the anomalies on the original test data plot.\n"
df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
(fig, ax) = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color='r')
plt.show()