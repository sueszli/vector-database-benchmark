import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module

class HeadAdapter(adapter_module.Adapter):

    def __init__(self, name, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.name = name

    def check(self, dataset):
        if False:
            return 10
        supported_types = (tf.data.Dataset, np.ndarray, pd.DataFrame, pd.Series)
        if not isinstance(dataset, supported_types):
            raise TypeError(f'Expect the target data of {self.name} to be tf.data.Dataset, np.ndarray, pd.DataFrame or pd.Series, but got {type(dataset)}.')

    def convert_to_dataset(self, dataset, batch_size):
        if False:
            print('Hello World!')
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, pd.Series):
            dataset = dataset.values
        return super().convert_to_dataset(dataset, batch_size)

class ClassificationAdapter(HeadAdapter):
    pass

class RegressionAdapter(HeadAdapter):
    pass

class SegmentationHeadAdapter(ClassificationAdapter):
    pass