import pytest
import operator
import tempfile
from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils.np_utils import to_categorical
import numpy as np
from bigdl.nano.tf.keras import Model, InferenceOptimizer
from bigdl.nano.utils.common import compare_version

class TestModelQuantize(TestCase):

    def test_model_quantize_ptq(self):
        if False:
            return 10
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_labels = to_categorical(train_labels, num_classes=10)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        q_model = InferenceOptimizer.quantize(model, x=train_dataset)
        assert q_model
        output = q_model(train_examples[0:10])
        assert output.shape == (10, 10)
        q_model = InferenceOptimizer.quantize(model, x=train_dataset, metric=tf.keras.metrics.CategoricalAccuracy(), tuning_strategy='basic', accuracy_criterion={'relative': 0.99, 'higher_is_better': True})
        assert q_model
        output = q_model(train_examples[0:10])
        assert output.shape == (10, 10)
        with tempfile.TemporaryDirectory() as tmp_dir:
            InferenceOptimizer.save(q_model, tmp_dir)
            loaded_model = InferenceOptimizer.load(tmp_dir, model)
        output2 = loaded_model(train_examples[0:10])
        assert output2.shape == (10, 10)
        np.testing.assert_almost_equal(output.numpy(), output2.numpy(), decimal=5)
        invalid_approach = 'dynamic'
        with pytest.raises(RuntimeError, match="Only 'static' approach is supported now."):
            InferenceOptimizer.quantize(model, x=None, approach=invalid_approach)

    def test_model_quantize_without_dataset(self):
        if False:
            return 10
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_labels = to_categorical(train_labels, num_classes=10)
        q_model = InferenceOptimizer.quantize(model, x=train_examples, y=train_labels)
        assert q_model(train_examples[0:10]).shape == (10, 10)

    def test_model_quantize_with_only_x(self):
        if False:
            print('Hello World!')
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        q_model = InferenceOptimizer.quantize(model, x=train_examples)
        assert q_model(train_examples[0:10]).shape == (10, 10)
        train_tensor = tf.convert_to_tensor(train_examples)
        q_model = InferenceOptimizer.quantize(model, x=train_tensor)
        assert q_model(train_examples[0:10]).shape == (10, 10)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)
        q_model = InferenceOptimizer.quantize(model, x=train_dataset)
        assert q_model(train_examples[0:10]).shape == (10, 10)
        train_dataset = tf.data.Dataset.from_tensors(train_examples[0])
        q_model = InferenceOptimizer.quantize(model, x=train_dataset)
        assert q_model(train_examples[0:10]).shape == (10, 10)

    def test_model_quantize_tuning(self):
        if False:
            return 10
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_labels = to_categorical(train_labels, num_classes=10)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        if compare_version('neural_compressor', operator.ge, '1.14.1'):

            def eval_func(model):
                if False:
                    return 10
                infer = model.signatures['serving_default']
                output_dict_keys = infer.structured_outputs.keys()
                output_name = list(output_dict_keys)[0]
                inputs = np.array(train_examples)
                input_tensor = tf.constant(inputs, dtype=tf.float32)
                preds = infer(input_tensor)[output_name]
                acc = tf.keras.metrics.CategoricalAccuracy()(preds, train_labels).numpy()
                return acc
            q_model = InferenceOptimizer.quantize(model, x=train_dataset, eval_func=eval_func, accuracy_criterion={'relative': 0.99, 'higher_is_better': True})
            assert q_model
            output = q_model(train_examples[0:10])
            assert output.shape == (10, 10)
        q_model = InferenceOptimizer.quantize(model, x=train_dataset, metric=tf.keras.metrics.CategoricalAccuracy(), tuning_strategy='basic', accuracy_criterion={'relative': 0.99, 'higher_is_better': True})
        assert q_model
        output = q_model(train_examples[0:10])
        assert output.shape == (10, 10)