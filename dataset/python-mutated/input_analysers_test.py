import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers

def test_structured_data_input_less_col_name_error():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as info:
        analyser = input_analysers.StructuredDataAnalyser(column_names=list(range(8)))
        dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(20, 10)).batch(32)
        for x in dataset:
            analyser.update(x)
        analyser.finalize()
    assert 'Expect column_names to have length' in str(info.value)

def test_structured_data_infer_col_types():
    if False:
        while True:
            i = 10
    analyser = input_analysers.StructuredDataAnalyser(column_names=test_utils.COLUMN_NAMES, column_types=None)
    x = pd.read_csv(test_utils.TRAIN_CSV_PATH)
    x.pop('survived')
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(str)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.column_types == test_utils.COLUMN_TYPES

def test_dont_infer_specified_column_types():
    if False:
        i = 10
        return i + 15
    column_types = copy.copy(test_utils.COLUMN_TYPES)
    column_types.pop('sex')
    column_types['age'] = 'categorical'
    analyser = input_analysers.StructuredDataAnalyser(column_names=test_utils.COLUMN_NAMES, column_types=column_types)
    x = pd.read_csv(test_utils.TRAIN_CSV_PATH)
    x.pop('survived')
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(str)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.column_types['age'] == 'categorical'

def test_structured_data_input_with_illegal_dim():
    if False:
        print('Hello World!')
    analyser = input_analysers.StructuredDataAnalyser(column_names=test_utils.COLUMN_NAMES, column_types=None)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to StructuredDataInput to have shape' in str(info.value)

def test_image_input_analyser_shape_is_list_of_int():
    if False:
        print('Hello World!')
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32, 3)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert isinstance(analyser.shape, list)
    assert all(map(lambda x: isinstance(x, int), analyser.shape))

def test_image_input_with_three_dim():
    if False:
        i = 10
        return i + 15
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert len(analyser.shape) == 3

def test_image_input_with_illegal_dim():
    if False:
        for i in range(10):
            print('nop')
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to ImageInput to have shape' in str(info.value)

def test_text_input_with_illegal_dim():
    if False:
        return 10
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to TextInput to have shape' in str(info.value)

def test_text_analyzer_with_one_dim_doesnt_crash():
    if False:
        while True:
            i = 10
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(['a b c', 'b b c']).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()

def test_text_illegal_type_error():
    if False:
        while True:
            i = 10
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 1)).batch(32)
    with pytest.raises(TypeError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to TextInput to be strings' in str(info.value)

def test_time_series_input_with_illegal_dim():
    if False:
        print('Hello World!')
    analyser = input_analysers.TimeseriesAnalyser(column_names=test_utils.COLUMN_NAMES, column_types=None)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to TimeseriesInput to have shape' in str(info.value)