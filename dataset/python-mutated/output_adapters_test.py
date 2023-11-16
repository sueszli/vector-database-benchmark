import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import output_adapters

def test_clf_head_transform_pd_series_to_dataset():
    if False:
        while True:
            i = 10
    adapter = output_adapters.ClassificationAdapter(name='a')
    y = adapter.adapt(pd.read_csv(test_utils.TEST_CSV_PATH).pop('survived'), batch_size=32)
    assert isinstance(y, tf.data.Dataset)

def test_clf_head_transform_df_to_dataset():
    if False:
        while True:
            i = 10
    adapter = output_adapters.ClassificationAdapter(name='a')
    y = adapter.adapt(pd.DataFrame(test_utils.generate_one_hot_labels(dtype='np', num_classes=10)), batch_size=32)
    assert isinstance(y, tf.data.Dataset)

def test_unsupported_types_error():
    if False:
        for i in range(10):
            print('nop')
    adapter = output_adapters.ClassificationAdapter(name='a')
    with pytest.raises(TypeError) as info:
        adapter.adapt(1, batch_size=32)
    assert 'Expect the target data of a to be tf' in str(info.value)

def test_reg_head_transform_pd_series():
    if False:
        while True:
            i = 10
    adapter = output_adapters.RegressionAdapter(name='a')
    y = adapter.adapt(pd.read_csv(test_utils.TEST_CSV_PATH).pop('survived'), batch_size=32)
    assert isinstance(y, tf.data.Dataset)

def test_reg_head_transform_1d_np():
    if False:
        return 10
    adapter = output_adapters.RegressionAdapter(name='a')
    y = adapter.adapt(np.random.rand(10), batch_size=32)
    assert isinstance(y, tf.data.Dataset)