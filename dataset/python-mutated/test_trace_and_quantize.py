from unittest import TestCase
import tempfile
import operator
import numpy as np
import tensorflow as tf
import pytest
from tensorflow.keras.metrics import MeanSquaredError, CategoricalAccuracy
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from bigdl.nano.tf.keras import InferenceOptimizer

class MyModel(tf.keras.Model):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.x = x

    def call(self, inputs):
        if False:
            print('Hello World!')
        x = self.dense1(inputs)
        return self.dense2(x)

    def get_x(self):
        if False:
            for i in range(10):
                print('nop')
        return self.x

    @staticmethod
    def do_nothing():
        if False:
            return 10
        pass

class MyModelCannotComputeOutputShape(tf.keras.Model):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.dense = tf.keras.layers.Dense(4, activation=tf.nn.relu)

    def call(self, inputs):
        if False:
            return 10
        return self.dense(inputs)

    def compute_output_shape(self, input_shape):
        if False:
            print('Hello World!')
        '\n        Older versions of TensorFlow required custom layers to implement the\n        `compute_output_shape` method. If it was not implemented, calling its\n        `compute_output_shape` would throw a `NotImplementedError` exception.\n        We cannot reproduce this behavior in newer versions of TensorFlow,\n        so we manually throw this exception to simulate this behavior.\n        '
        raise NotImplementedError()

def ModelWithConv2DTranspose():
    if False:
        while True:
            i = 10
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    outputs = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class TestTraceAndQuantize(TestCase):

    def test_attribute_access_after_trace(self):
        if False:
            print('Hello World!')
        x = 100
        model = MyModel(x)
        traced_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        traced_model.do_nothing()
        assert traced_model.get_x() == traced_model.x == x
        traced_model(np.random.random((1, 4)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == traced_model.x == x
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with pytest.raises(AttributeError):
            new_model.do_nothing()
        with pytest.raises(AttributeError):
            assert new_model.get_x()
        model = MyModel(x)
        traced_model = InferenceOptimizer.trace(model, accelerator='openvino', input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        traced_model.do_nothing()
        assert traced_model.get_x() == traced_model.x == x
        traced_model(np.random.random((1, 4)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == traced_model.x == x
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with pytest.raises(AttributeError):
            new_model.do_nothing()
        with pytest.raises(AttributeError):
            assert new_model.get_x()

    def test_attribute_access_after_quantize(self):
        if False:
            return 10
        x = 100
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model, accelerator='onnxruntime', input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32), x=np.random.random((100, 4)), y=np.random.random((100, 5)))
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
        quantized_model(np.random.random((1, 4)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == quantized_model.x == x
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with pytest.raises(AttributeError):
            new_model.do_nothing()
        with pytest.raises(AttributeError):
            assert new_model.get_x()
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model, accelerator='openvino', input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32), x=np.random.random((100, 4)), y=np.random.random((100, 5)))
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
        quantized_model(np.random.random((1, 4)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == quantized_model.x == x
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with pytest.raises(AttributeError):
            new_model.do_nothing()
        with pytest.raises(AttributeError):
            assert new_model.get_x()
        from bigdl.nano.utils.common import compare_version
        INC_LESS_14 = compare_version('neural_compressor', operator.lt, '1.14')
        if INC_LESS_14:
            return
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model, accelerator=None, input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32), x=np.random.random((100, 4)), y=np.random.random((100, 5)))
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
        quantized_model(np.random.random((1, 4)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == quantized_model.x == x

    def test_evaluate(self):
        if False:
            i = 10
            return i + 15
        inputs = tf.keras.Input(shape=(28 * 28,), name='digits')
        x = layers.Dense(10, name='dense_logits')(inputs)
        outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(), metrics=CategoricalAccuracy())
        x = np.random.random((100, 28 * 28))
        y = np.random.randint(0, 10, 100)
        inc_q_model = InferenceOptimizer.quantize(model, x=x, y=y)
        inc_q_model.evaluate(x=x, y=y)
        ov_t_model = InferenceOptimizer.trace(model, accelerator='openvino')
        ov_t_model.evaluate(x=x, y=y)
        ov_q_model = InferenceOptimizer.quantize(model, accelerator='openvino', x=x, y=y)
        ov_q_model.evaluate(x=x, y=y)
        ort_t_model = InferenceOptimizer.trace(model, accelerator='onnxruntime')
        ort_t_model.evaluate(x=x, y=y)
        ort_q_model = InferenceOptimizer.quantize(model, accelerator='onnxruntime', x=x, y=y)
        ort_q_model.evaluate(x=x, y=y)
        for m in [inc_q_model, ov_t_model, ov_q_model, ort_t_model, ort_q_model]:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                InferenceOptimizer.save(m, tmp_dir_name)
                new_model = InferenceOptimizer.load(tmp_dir_name, model)
                new_model.evaluate(x=x, y=y)

    def test_quantize_bf16(self):
        if False:
            return 10
        model = MyModel(100)
        model.compile(loss='mse', metrics=MeanSquaredError())
        ori_model_policies = []
        for layer in model.layers:
            ori_model_policies.append(layer._dtype_policy)
        x = np.random.random((100, 4))
        model(x)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        for (idx, layer) in enumerate(model.layers):
            assert layer._dtype_policy == ori_model_policies[idx]
        from bigdl.nano.utils.common import _avx512_checker
        if _avx512_checker():
            output = bf16_model(x)
            assert output.dtype == tf.float32
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
        if _avx512_checker():
            output = load_model(x)
            assert output.dtype == tf.float32
        model = MobileNetV2(weights='imagenet')
        ori_model_config = model.get_config()
        x = np.random.rand(32, 224, 224, 3)
        model(x)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        assert ori_model_config == model.get_config()
        from bigdl.nano.utils.common import _avx512_checker
        if _avx512_checker():
            output = bf16_model(x)
            assert output.dtype == tf.bfloat16
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
        if _avx512_checker():
            output = load_model(x)
            assert output.dtype == tf.bfloat16
        model = ModelWithConv2DTranspose()
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        inputs = tf.random.normal((1, 32, 32, 3))
        if _avx512_checker():
            output1 = model(inputs)
            output2 = bf16_model(inputs)
            assert output1.dtype == tf.float32
            assert output2.dtype == tf.bfloat16
            np.testing.assert_allclose(output1, tf.cast(output2, tf.float32), atol=0.01)

    def test_model_cannot_compute_output_shape(self):
        if False:
            return 10
        model = MyModelCannotComputeOutputShape()
        x = np.random.random((100, 4))
        y = np.random.random((100, 4))
        ov_t_model = InferenceOptimizer.trace(model, accelerator='openvino', input_spec=tf.TensorSpec(shape=(None, 4)))
        ort_t_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_spec=tf.TensorSpec(shape=(None, 4)))
        inc_q_model = InferenceOptimizer.quantize(model, x=x, y=y, input_spec=tf.TensorSpec(shape=(None, 4)))