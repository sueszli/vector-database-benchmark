import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils

def test_structured_data_input_unsupported_type_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError) as info:
        adapter = input_adapters.StructuredDataAdapter()
        adapter.adapt('unknown', batch_size=32)
    assert 'Unsupported type' in str(info.value)

def test_structured_data_input_transform_to_dataset():
    if False:
        for i in range(10):
            print('nop')
    x = tf.data.Dataset.from_tensor_slices(pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(str))
    adapter = input_adapters.StructuredDataAdapter()
    x = adapter.adapt(x, batch_size=32)
    assert isinstance(x, tf.data.Dataset)

def test_image_input_adapter_transform_to_dataset():
    if False:
        return 10
    x = test_utils.generate_data()
    adapter = input_adapters.ImageAdapter()
    assert isinstance(adapter.adapt(x, batch_size=32), tf.data.Dataset)

def test_image_input_unsupported_type():
    if False:
        print('Hello World!')
    x = 'unknown'
    adapter = input_adapters.ImageAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data to ImageInput to be numpy' in str(info.value)

def test_image_input_numerical():
    if False:
        print('Hello World!')
    x = np.array([[['unknown']]])
    adapter = input_adapters.ImageAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data to ImageInput to be numerical' in str(info.value)

def test_input_type_error():
    if False:
        print('Hello World!')
    x = 'unknown'
    adapter = input_adapters.InputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data to Input to be numpy' in str(info.value)

def test_input_numerical():
    if False:
        i = 10
        return i + 15
    x = np.array([[['unknown']]])
    adapter = input_adapters.InputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data to Input to be numerical' in str(info.value)

def test_text_adapt_unbatched_dataset():
    if False:
        for i in range(10):
            print('nop')
    x = tf.data.Dataset.from_tensor_slices(np.array(['a b c', 'b b c']))
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)
    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)

def test_text_adapt_batched_dataset():
    if False:
        while True:
            i = 10
    x = tf.data.Dataset.from_tensor_slices(np.array(['a b c', 'b b c'])).batch(32)
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)
    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)

def test_text_adapt_np():
    if False:
        print('Hello World!')
    x = np.array(['a b c', 'b b c'])
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)
    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)

def test_text_input_type_error():
    if False:
        print('Hello World!')
    x = 'unknown'
    adapter = input_adapters.TextAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data to TextInput to be numpy' in str(info.value)

def test_time_series_input_type_error():
    if False:
        print('Hello World!')
    x = 'unknown'
    adapter = input_adapters.TimeseriesAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert 'Expect the data in TimeseriesInput to be numpy' in str(info.value)

def test_time_series_input_transform_df_to_dataset():
    if False:
        i = 10
        return i + 15
    adapter = input_adapters.TimeseriesAdapter()
    x = adapter.adapt(pd.DataFrame(np.random.rand(100, 32)), batch_size=32)
    assert isinstance(x, tf.data.Dataset)