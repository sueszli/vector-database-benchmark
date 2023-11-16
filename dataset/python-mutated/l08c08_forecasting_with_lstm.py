import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def trend(time, slope=0):
    if False:
        return 10
    return slope * time

def seasonal_pattern(season_time):
    if False:
        print('Hello World!')
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    if False:
        i = 10
        return i + 15
    season_time = (time + phase) % period / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    if False:
        print('Hello World!')
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def ndarray_to_dataset(ndarray):
    if False:
        print('Hello World!')
    return tf.data.Dataset.from_tensor_slices(ndarray)

def sequential_window_dataset(series, window_size):
    if False:
        i = 10
        return i + 15
    series = tf.expand_dims(series, axis=-1)
    ds = ndarray_to_dataset(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)

def dataset_to_ndarray(dataset):
    if False:
        print('Hello World!')
    array = list(dataset.as_numpy_iterator())
    return np.ndarray(array)
time_range = 4 * 365 + 1
time = np.arange(time_range)
slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
noise_level = 5
noise = white_noise(time, noise_level, seed=42)
series += noise
tf.random.set_seed(42)
np.random.seed(42)
window_size = 30
test_set = sequential_window_dataset(series, window_size)
test_array = dataset_to_ndarray(test_set)
model = tf.keras.models.load_model('path/to/model')
rnn_forecast_nd = model.predict(test_array)