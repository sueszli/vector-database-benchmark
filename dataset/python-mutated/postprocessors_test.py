import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import postprocessors

def test_sigmoid_postprocess_to_zero_one():
    if False:
        for i in range(10):
            print('nop')
    postprocessor = postprocessors.SigmoidPostprocessor()
    y = postprocessor.postprocess(np.random.rand(10, 3))
    assert set(y.flatten().tolist()) == set([1, 0])

def test_sigmoid_transform_dataset_doesnt_change():
    if False:
        return 10
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)
    assert postprocessor.transform(dataset) is dataset

def test_sigmoid_deserialize_without_error():
    if False:
        while True:
            i = 10
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)
    postprocessor = preprocessors.deserialize(preprocessors.serialize(postprocessor))
    assert postprocessor.transform(dataset) is dataset

def test_softmax_postprocess_to_zero_one():
    if False:
        while True:
            i = 10
    postprocessor = postprocessors.SoftmaxPostprocessor()
    y = postprocessor.postprocess(np.random.rand(10, 3))
    assert set(y.flatten().tolist()) == set([1, 0])

def test_softmax_transform_dataset_doesnt_change():
    if False:
        return 10
    postprocessor = postprocessors.SoftmaxPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)
    assert postprocessor.transform(dataset) is dataset

def test_softmax_deserialize_without_error():
    if False:
        i = 10
        return i + 15
    postprocessor = postprocessors.SoftmaxPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)
    postprocessor = preprocessors.deserialize(preprocessors.serialize(postprocessor))
    assert postprocessor.transform(dataset) is dataset