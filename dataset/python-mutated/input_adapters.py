import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module

class InputAdapter(adapter_module.Adapter):

    def check(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Record any information needed by transform.'
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to Input to be numpy.ndarray or tf.data.Dataset, but got {type}.'.format(type=type(x)))
        if isinstance(x, np.ndarray) and (not np.issubdtype(x.dtype, np.number)):
            raise TypeError('Expect the data to Input to be numerical, but got {type}.'.format(type=x.dtype))

class ImageAdapter(adapter_module.Adapter):

    def check(self, x):
        if False:
            while True:
                i = 10
        'Record any information needed by transform.'
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to ImageInput to be numpy.ndarray or tf.data.Dataset, but got {type}.'.format(type=type(x)))
        if isinstance(x, np.ndarray) and (not np.issubdtype(x.dtype, np.number)):
            raise TypeError('Expect the data to ImageInput to be numerical, but got {type}.'.format(type=x.dtype))

class TextAdapter(adapter_module.Adapter):

    def check(self, x):
        if False:
            while True:
                i = 10
        'Record any information needed by transform.'
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to TextInput to be numpy.ndarray or tf.data.Dataset, but got {type}.'.format(type=type(x)))

class StructuredDataAdapter(adapter_module.Adapter):

    def check(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError('Unsupported type {type} for {name}.'.format(type=type(x), name=self.__class__.__name__))

    def convert_to_dataset(self, dataset, batch_size):
        if False:
            i = 10
            return i + 15
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, np.ndarray) and dataset.dtype == object:
            dataset = dataset.astype(str)
        return super().convert_to_dataset(dataset, batch_size)

class TimeseriesAdapter(adapter_module.Adapter):

    def __init__(self, lookback=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.lookback = lookback

    def check(self, x):
        if False:
            return 10
        'Record any information needed by transform.'
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data in TimeseriesInput to be numpy.ndarray or tf.data.Dataset or pd.DataFrame, but got {type}.'.format(type=type(x)))

    def convert_to_dataset(self, dataset, batch_size):
        if False:
            i = 10
            return i + 15
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        return super().convert_to_dataset(dataset, batch_size)