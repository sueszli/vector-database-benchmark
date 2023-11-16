import os
import shutil
import tempfile
import unittest
from collections import OrderedDict
import numpy as np
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
try:
    import onnxruntime as ort
    import torch
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
    import tensorflow as tf
    import tf2onnx
    from tensorflow.keras import layers
    from sklearn import linear_model
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from apache_beam.ml.inference.base import PredictionResult
    from apache_beam.ml.inference.base import RunInference
    from apache_beam.ml.inference.onnx_inference import default_numpy_inference_fn
    from apache_beam.ml.inference.onnx_inference import OnnxModelHandlerNumpy
except ImportError:
    raise unittest.SkipTest('Onnx dependencies are not installed')
try:
    from apache_beam.io.gcp.gcsfilesystem import GCSFileSystem
except ImportError:
    GCSFileSystem = None

class PytorchLinearRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.linear(x)
        return out

    def generate(self, x):
        if False:
            i = 10
            return i + 15
        out = self.linear(x) + 0.5
        return out

class TestDataAndModel:

    def get_one_feature_samples(self):
        if False:
            while True:
                i = 10
        return [np.array([1], dtype='float32'), np.array([5], dtype='float32'), np.array([-3], dtype='float32'), np.array([10.0], dtype='float32')]

    def get_one_feature_predictions(self):
        if False:
            i = 10
            return i + 15
        return [PredictionResult(ex, pred) for (ex, pred) in zip(self.get_one_feature_samples(), [example * 2.0 + 0.5 for example in self.get_one_feature_samples()])]

    def get_two_feature_examples(self):
        if False:
            return 10
        return [np.array([1, 5], dtype='float32'), np.array([3, 10], dtype='float32'), np.array([-14, 0], dtype='float32'), np.array([0.5, 0.5], dtype='float32')]

    def get_two_feature_predictions(self):
        if False:
            while True:
                i = 10
        return [PredictionResult(ex, pred) for (ex, pred) in zip(self.get_two_feature_examples(), [f1 * 2.0 + f2 * 3 + 0.5 for (f1, f2) in self.get_two_feature_examples()])]

    def get_torch_one_feature_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = PytorchLinearRegression(input_dim=1, output_dim=1)
        model.load_state_dict(OrderedDict([('linear.weight', torch.Tensor([[2.0]])), ('linear.bias', torch.Tensor([0.5]))]))
        return model

    def get_tf_one_feature_model(self):
        if False:
            for i in range(10):
                print('nop')
        params = [np.array([[2.0]], dtype='float32'), np.array([0.5], dtype='float32')]
        linear_layer = layers.Dense(units=1, weights=params)
        linear_model = tf.keras.Sequential([linear_layer])
        return linear_model

    def get_sklearn_one_feature_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = [[0], [1]]
        y = [0.5, 2.5]
        model = linear_model.LinearRegression()
        model.fit(x, y)
        return model

    def get_torch_two_feature_model(self):
        if False:
            i = 10
            return i + 15
        model = PytorchLinearRegression(input_dim=2, output_dim=1)
        model.load_state_dict(OrderedDict([('linear.weight', torch.Tensor([[2.0, 3]])), ('linear.bias', torch.Tensor([0.5]))]))
        return model

    def get_tf_two_feature_model(self):
        if False:
            while True:
                i = 10
        params = [np.array([[2.0], [3]]), np.array([0.5], dtype='float32')]
        linear_layer = layers.Dense(units=1, weights=params)
        linear_model = tf.keras.Sequential([linear_layer])
        return linear_model

    def get_sklearn_two_feature_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = [[1, 5], [3, 2], [1, 0]]
        y = [17.5, 12.5, 2.5]
        model = linear_model.LinearRegression()
        model.fit(x, y)
        return model

def _compare_prediction_result(a, b):
    if False:
        return 10
    example_equal = np.array_equal(a.example, b.example)
    if isinstance(a.inference, dict):
        return all((x == y for (x, y) in zip(a.inference.values(), b.inference.values()))) and example_equal
    return a.inference == b.inference and example_equal

def _to_numpy(tensor):
    if False:
        return 10
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class TestOnnxModelHandler(OnnxModelHandlerNumpy):

    def __init__(self, model_uri: str, session_options=None, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], provider_options=None, *, inference_fn=default_numpy_inference_fn, large_model=False, **kwargs):
        if False:
            return 10
        self._model_uri = model_uri
        self._session_options = session_options
        self._providers = providers
        self._provider_options = provider_options
        self._model_inference_fn = inference_fn
        self._env_vars = kwargs.get('env_vars', {})
        self._large_model = large_model

class OnnxTestBase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmpdir = tempfile.mkdtemp()
        self.test_data_and_model = TestDataAndModel()

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmpdir)

@pytest.mark.uses_onnx
class OnnxPytorchRunInferenceTest(OnnxTestBase):

    def test_onnx_pytorch_run_inference(self):
        if False:
            print('Hello World!')
        examples = self.test_data_and_model.get_one_feature_samples()
        expected_predictions = self.test_data_and_model.get_one_feature_predictions()
        model = self.test_data_and_model.get_torch_one_feature_model()
        path = os.path.join(self.tmpdir, 'my_onnx_pytorch_path')
        dummy_input = torch.randn(4, 1, requires_grad=True)
        torch.onnx.export(model, dummy_input, path, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        inference_runner = TestOnnxModelHandler(path)
        inference_session = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        predictions = inference_runner.run_inference(examples, inference_session)
        for (actual, expected) in zip(predictions, expected_predictions):
            self.assertEqual(actual, expected)

    def test_num_bytes(self):
        if False:
            i = 10
            return i + 15
        inference_runner = TestOnnxModelHandler('dummy')
        batched_examples_int = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        self.assertEqual(batched_examples_int[0].itemsize * 3, inference_runner.get_num_bytes(batched_examples_int))
        batched_examples_float = [np.array([1, 5], dtype=np.float32), np.array([3, 10], dtype=np.float32), np.array([-14, 0], dtype=np.float32), np.array([0.5, 0.5], dtype=np.float32)]
        self.assertEqual(batched_examples_float[0].itemsize * 4, inference_runner.get_num_bytes(batched_examples_float))

    def test_namespace(self):
        if False:
            print('Hello World!')
        inference_runner = TestOnnxModelHandler('dummy')
        self.assertEqual('BeamML_Onnx', inference_runner.get_metrics_namespace())

@pytest.mark.uses_onnx
class OnnxTensorflowRunInferenceTest(OnnxTestBase):

    def test_onnx_tensorflow_run_inference(self):
        if False:
            return 10
        examples = self.test_data_and_model.get_one_feature_samples()
        expected_predictions = self.test_data_and_model.get_one_feature_predictions()
        linear_model = self.test_data_and_model.get_tf_one_feature_model()
        path = os.path.join(self.tmpdir, 'my_onnx_tf_path')
        spec = (tf.TensorSpec((None, 1), tf.float32, name='input'),)
        (_, _) = tf2onnx.convert.from_keras(linear_model, input_signature=spec, opset=13, output_path=path)
        inference_runner = TestOnnxModelHandler(path)
        inference_session = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        predictions = inference_runner.run_inference(examples, inference_session)
        for (actual, expected) in zip(predictions, expected_predictions):
            self.assertEqual(actual, expected)

@pytest.mark.uses_onnx
class OnnxSklearnRunInferenceTest(OnnxTestBase):

    def save_model(self, model, input_dim, path):
        if False:
            while True:
                i = 10
        initial_type = [('float_input', FloatTensorType([None, input_dim]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open(path, 'wb') as f:
            f.write(onx.SerializeToString())

    def test_onnx_sklearn_run_inference(self):
        if False:
            while True:
                i = 10
        examples = self.test_data_and_model.get_one_feature_samples()
        expected_predictions = self.test_data_and_model.get_one_feature_predictions()
        linear_model = self.test_data_and_model.get_sklearn_one_feature_model()
        path = os.path.join(self.tmpdir, 'my_onnx_sklearn_path')
        self.save_model(linear_model, 1, path)
        inference_runner = TestOnnxModelHandler(path)
        inference_session = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        predictions = inference_runner.run_inference(examples, inference_session)
        for (actual, expected) in zip(predictions, expected_predictions):
            self.assertEqual(actual, expected)

@pytest.mark.uses_onnx
class OnnxPytorchRunInferencePipelineTest(OnnxTestBase):

    def exportModelToOnnx(self, model, path):
        if False:
            while True:
                i = 10
        dummy_input = torch.randn(4, 2, requires_grad=True)
        torch.onnx.export(model, dummy_input, path, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    def test_pipeline_local_model_simple(self):
        if False:
            i = 10
            return i + 15
        with TestPipeline() as pipeline:
            path = os.path.join(self.tmpdir, 'my_onnx_pytorch_path')
            model = self.test_data_and_model.get_torch_two_feature_model()
            self.exportModelToOnnx(model, path)
            model_handler = TestOnnxModelHandler(path)
            pcoll = pipeline | 'start' >> beam.Create(self.test_data_and_model.get_two_feature_examples())
            predictions = pcoll | RunInference(model_handler)
            assert_that(predictions, equal_to(self.test_data_and_model.get_two_feature_predictions(), equals_fn=_compare_prediction_result))

    def test_model_handler_sets_env_vars(self):
        if False:
            while True:
                i = 10
        with TestPipeline() as pipeline:
            path = os.path.join(self.tmpdir, 'my_onnx_pytorch_path')
            model = self.test_data_and_model.get_torch_two_feature_model()
            self.exportModelToOnnx(model, path)
            model_handler = OnnxModelHandlerNumpy(model_uri=path, env_vars={'FOO': 'bar'})
            self.assertFalse('FOO' in os.environ)
            _ = pipeline | 'start' >> beam.Create(self.test_data_and_model.get_two_feature_examples()) | RunInference(model_handler)
            pipeline.run()
            self.assertTrue('FOO' in os.environ)
            self.assertTrue('bar'.equals(os.environ['FOO']))

    def test_model_handler_large_model(self):
        if False:
            print('Hello World!')
        with TestPipeline() as pipeline:

            def onxx_numpy_inference_fn(inference_session: ort.InferenceSession, batch, inference_args=None):
                if False:
                    i = 10
                    return i + 15
                multi_process_shared_loaded = 'multi_process_shared' in str(type(inference_session))
                if not multi_process_shared_loaded:
                    raise Exception(f'Loaded model of type {type(model)}, was ' + 'expecting multi_process_shared_model')
                return default_numpy_inference_fn(inference_session, batch, inference_args)
            path = os.path.join(self.tmpdir, 'my_onnx_pytorch_path')
            model = self.test_data_and_model.get_torch_two_feature_model()
            self.exportModelToOnnx(model, path)
            model_handler = OnnxModelHandlerNumpy(model_uri=path, env_vars={'FOO': 'bar'}, inference_fn=onxx_numpy_inference_fn, large_model=True)
            self.assertFalse('FOO' in os.environ)
            _ = pipeline | 'start' >> beam.Create(self.test_data_and_model.get_two_feature_examples()) | RunInference(model_handler)
            pipeline.run()

    @unittest.skipIf(GCSFileSystem is None, 'GCP dependencies are not installed')
    def test_pipeline_gcs_model(self):
        if False:
            for i in range(10):
                print('nop')
        with TestPipeline() as pipeline:
            examples = self.test_data_and_model.get_one_feature_samples()
            expected_predictions = self.test_data_and_model.get_one_feature_predictions()
            gs_path = 'gs://apache-beam-ml/models/torch_2xplus5_onnx'
            model_handler = TestOnnxModelHandler(gs_path)
            pcoll = pipeline | 'start' >> beam.Create(examples)
            predictions = pcoll | RunInference(model_handler)
            assert_that(predictions, equal_to(expected_predictions, equals_fn=_compare_prediction_result))

    def test_invalid_input_type(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(InvalidArgument, 'Got invalid dimensions for input'):
            with TestPipeline() as pipeline:
                examples = [np.array([1], dtype='float32')]
                path = os.path.join(self.tmpdir, 'my_onnx_pytorch_path')
                model = self.test_data_and_model.get_torch_two_feature_model()
                self.exportModelToOnnx(model, path)
                model_handler = TestOnnxModelHandler(path)
                pcoll = pipeline | 'start' >> beam.Create(examples)
                pcoll | RunInference(model_handler)

@pytest.mark.uses_onnx
class OnnxTensorflowRunInferencePipelineTest(OnnxTestBase):

    def exportModelToOnnx(self, model, path):
        if False:
            while True:
                i = 10
        spec = (tf.TensorSpec((None, 2), tf.float32, name='input'),)
        (_, _) = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=path)

    def test_pipeline_local_model_simple(self):
        if False:
            return 10
        with TestPipeline() as pipeline:
            path = os.path.join(self.tmpdir, 'my_onnx_tensorflow_path')
            model = self.test_data_and_model.get_tf_two_feature_model()
            self.exportModelToOnnx(model, path)
            model_handler = TestOnnxModelHandler(path)
            pcoll = pipeline | 'start' >> beam.Create(self.test_data_and_model.get_two_feature_examples())
            predictions = pcoll | RunInference(model_handler)
            assert_that(predictions, equal_to(self.test_data_and_model.get_two_feature_predictions(), equals_fn=_compare_prediction_result))

    @unittest.skipIf(GCSFileSystem is None, 'GCP dependencies are not installed')
    def test_pipeline_gcs_model(self):
        if False:
            return 10
        with TestPipeline() as pipeline:
            examples = self.test_data_and_model.get_one_feature_samples()
            expected_predictions = self.test_data_and_model.get_one_feature_predictions()
            gs_path = 'gs://apache-beam-ml/models/tf_2xplus5_onnx'
            model_handler = TestOnnxModelHandler(gs_path)
            pcoll = pipeline | 'start' >> beam.Create(examples)
            predictions = pcoll | RunInference(model_handler)
            assert_that(predictions, equal_to(expected_predictions, equals_fn=_compare_prediction_result))

    def test_invalid_input_type(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(InvalidArgument, 'Got invalid dimensions for input'):
            with TestPipeline() as pipeline:
                examples = [np.array([1], dtype='float32')]
                path = os.path.join(self.tmpdir, 'my_onnx_tensorflow_path')
                model = self.test_data_and_model.get_tf_two_feature_model()
                self.exportModelToOnnx(model, path)
                model_handler = TestOnnxModelHandler(path)
                pcoll = pipeline | 'start' >> beam.Create(examples)
                pcoll | RunInference(model_handler)

@pytest.mark.uses_onnx
class OnnxSklearnRunInferencePipelineTest(OnnxTestBase):

    def save_model(self, model, input_dim, path):
        if False:
            return 10
        initial_type = [('float_input', FloatTensorType([None, input_dim]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open(path, 'wb') as f:
            f.write(onx.SerializeToString())

    def test_pipeline_local_model_simple(self):
        if False:
            while True:
                i = 10
        with TestPipeline() as pipeline:
            path = os.path.join(self.tmpdir, 'my_onnx_sklearn_path')
            model = self.test_data_and_model.get_sklearn_two_feature_model()
            self.save_model(model, 2, path)
            model_handler = TestOnnxModelHandler(path)
            pcoll = pipeline | 'start' >> beam.Create(self.test_data_and_model.get_two_feature_examples())
            predictions = pcoll | RunInference(model_handler)
            assert_that(predictions, equal_to(self.test_data_and_model.get_two_feature_predictions(), equals_fn=_compare_prediction_result))

    @unittest.skipIf(GCSFileSystem is None, 'GCP dependencies are not installed')
    def test_pipeline_gcs_model(self):
        if False:
            print('Hello World!')
        with TestPipeline() as pipeline:
            examples = self.test_data_and_model.get_one_feature_samples()
            expected_predictions = self.test_data_and_model.get_one_feature_predictions()
            gs_path = 'gs://apache-beam-ml/models/skl_2xplus5_onnx'
            model_handler = TestOnnxModelHandler(gs_path)
            pcoll = pipeline | 'start' >> beam.Create(examples)
            predictions = pcoll | RunInference(model_handler)
            assert_that(predictions, equal_to(expected_predictions, equals_fn=_compare_prediction_result))

    def test_invalid_input_type(self):
        if False:
            return 10
        with self.assertRaises(InvalidArgument):
            with TestPipeline() as pipeline:
                examples = [np.array([1], dtype='float32')]
                path = os.path.join(self.tmpdir, 'my_onnx_sklearn_path')
                model = self.test_data_and_model.get_sklearn_two_feature_model()
                self.save_model(model, 2, path)
                model_handler = TestOnnxModelHandler(path)
                pcoll = pipeline | 'start' >> beam.Create(examples)
                pcoll | RunInference(model_handler)
if __name__ == '__main__':
    unittest.main()