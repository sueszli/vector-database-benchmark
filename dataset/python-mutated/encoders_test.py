import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import encoders
from autokeras.utils import data_utils

def test_one_hot_encoder_deserialize_transforms_to_np():
    if False:
        for i in range(10):
            print('nop')
    encoder = encoders.OneHotEncoder(['a', 'b', 'c'])
    encoder.fit(np.array(['a', 'b', 'a']))
    encoder = preprocessors.deserialize(preprocessors.serialize(encoder))
    one_hot = encoder.transform(tf.data.Dataset.from_tensor_slices([['a'], ['c'], ['b']]).batch(2))
    for data in one_hot:
        assert data.shape[1:] == [3]

def test_one_hot_encoder_decode_to_same_string():
    if False:
        i = 10
        return i + 15
    encoder = encoders.OneHotEncoder(['a', 'b', 'c'])
    result = encoder.postprocess(np.eye(3))
    assert np.array_equal(result, np.array([['a'], ['b'], ['c']]))

def test_label_encoder_decode_to_same_string():
    if False:
        i = 10
        return i + 15
    encoder = encoders.LabelEncoder(['a', 'b'])
    result = encoder.postprocess([[0], [1]])
    assert np.array_equal(result, np.array([['a'], ['b']]))

def test_label_encoder_encode_to_correct_shape():
    if False:
        return 10
    encoder = encoders.LabelEncoder(['a', 'b'])
    dataset = tf.data.Dataset.from_tensor_slices([['a'], ['b']]).batch(32)
    result = encoder.transform(dataset)
    assert data_utils.dataset_shape(result).as_list() == [None, 1]