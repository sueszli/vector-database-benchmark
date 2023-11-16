import numpy as np
import tensorflow as tf
from autokeras import hyper_preprocessors
from autokeras import preprocessors

def test_serialize_and_deserialize_default_hpps():
    if False:
        print('Hello World!')
    preprocessor = preprocessors.AddOneDimension()
    hyper_preprocessor = hyper_preprocessors.DefaultHyperPreprocessor(preprocessor)
    hyper_preprocessor = hyper_preprocessors.deserialize(hyper_preprocessors.serialize(hyper_preprocessor))
    assert isinstance(hyper_preprocessor.preprocessor, preprocessors.AddOneDimension)

def test_serialize_and_deserialize_default_hpps_categorical():
    if False:
        i = 10
        return i + 15
    x_train = np.array([['a', 'ab', 2.1], ['b', 'bc', 1.0], ['a', 'bc', 'nan']])
    preprocessor = preprocessors.CategoricalToNumericalPreprocessor(column_names=['column_a', 'column_b', 'column_c'], column_types={'column_a': 'categorical', 'column_b': 'categorical', 'column_c': 'numerical'})
    hyper_preprocessor = hyper_preprocessors.DefaultHyperPreprocessor(preprocessor)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
    hyper_preprocessor.preprocessor.fit(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    hyper_preprocessor = hyper_preprocessors.deserialize(hyper_preprocessors.serialize(hyper_preprocessor))
    assert isinstance(hyper_preprocessor.preprocessor, preprocessors.CategoricalToNumericalPreprocessor)
    results = hyper_preprocessor.preprocessor.transform(dataset)
    for result in results:
        assert result[0][0] == result[2][0]
        assert result[0][0] != result[1][0]
        assert result[0][1] != result[1][1]
        assert result[0][1] != result[2][1]
        assert result[2][2] == 0
        assert result.dtype == tf.float32