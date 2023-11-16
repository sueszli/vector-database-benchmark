"""Tests for tensorflow_model_server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import subprocess
import sys
import time
sys.path = [i for i in sys.path if 'bazel-out' not in i] + [i for i in sys.path if 'bazel-out' in i]
import grpc
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import flags
from tensorflow.python.profiler import profiler_client
from tensorflow.python.saved_model import signature_constants
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.model_servers.test_util import tensorflow_model_server_test_base
FLAGS = flags.FLAGS
RPC_TIMEOUT = 20.0
GRPC_SOCKET_PATH = '/tmp/tf-serving.sock'

class TensorflowModelServerTest(tensorflow_model_server_test_base.TensorflowModelServerTestBase):
    """This class defines integration test cases for tensorflow_model_server."""

    @staticmethod
    def __TestSrcDirPath(relative_path=''):
        if False:
            i = 10
            return i + 15
        return os.path.join(os.environ['TEST_SRCDIR'], 'tf_serving/tensorflow_serving', relative_path)

    def __BuildModelConfigFile(self):
        if False:
            for i in range(10):
                print('nop')
        'Write a config file to disk for use in tests.\n\n    Substitutes placeholder for test directory with test directory path\n    in the configuration template file and writes it out to another file\n    used by the test.\n    '
        with open(self._GetGoodModelConfigTemplate(), 'r') as template_file:
            config = template_file.read().replace('${TEST_HALF_PLUS_TWO_DIR}', self._GetSavedModelBundlePath())
            config = config.replace('${TEST_HALF_PLUS_THREE_DIR}', self._GetSavedModelHalfPlusThreePath())
        with open(self._GetGoodModelConfigFile(), 'w') as config_file:
            config_file.write(config)

    def setUp(self):
        if False:
            while True:
                i = 10
        'Sets up integration test parameters.'
        self.testdata_dir = TensorflowModelServerTest.__TestSrcDirPath('servables/tensorflow/testdata')
        self.temp_dir = tf.test.get_temp_dir()
        self.server_proc = None
        self.__BuildModelConfigFile()

    def tearDown(self):
        if False:
            print('Hello World!')
        'Deletes created configuration file.'
        os.remove(self._GetGoodModelConfigFile())

    def testGetModelStatus(self):
        if False:
            return 10
        'Test ModelService.GetModelStatus implementation.'
        model_path = self._GetSavedModelBundlePath()
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path)[1]
        print('Sending GetModelStatus request...')
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = 'default'
        channel = grpc.insecure_channel(model_server_address)
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        result = stub.GetModelStatus(request, RPC_TIMEOUT)
        self.assertEqual(1, len(result.model_version_status))
        self.assertEqual(123, result.model_version_status[0].version)
        self.assertEqual(0, result.model_version_status[0].status.error_code)

    def testGetModelMetadata(self):
        if False:
            return 10
        'Test PredictionService.GetModelMetadata implementation.'
        model_path = self._GetSavedModelBundlePath()
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path)[1]
        print('Sending GetModelMetadata request...')
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = 'default'
        request.metadata_field.append('signature_def')
        channel = grpc.insecure_channel(model_server_address)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        result = stub.GetModelMetadata(request, RPC_TIMEOUT)
        self.assertEqual('default', result.model_spec.name)
        self.assertEqual(self._GetModelVersion(model_path), result.model_spec.version.value)
        self.assertEqual(1, len(result.metadata))
        self.assertIn('signature_def', result.metadata)

    def _TestClassify(self, model_path):
        if False:
            return 10
        'Test PredictionService.Classify implementation.'
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path)[1]
        print('Sending Classify request...')
        request = classification_pb2.ClassificationRequest()
        request.model_spec.name = 'default'
        request.model_spec.signature_name = 'classify_x_to_y'
        example = request.input.example_list.examples.add()
        example.features.feature['x'].float_list.value.extend([2.0])
        channel = grpc.insecure_channel(model_server_address)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        result = stub.Classify(request, RPC_TIMEOUT)
        self.assertEqual(1, len(result.result.classifications))
        self.assertEqual(1, len(result.result.classifications[0].classes))
        expected_output = 3.0
        self.assertEqual(expected_output, result.result.classifications[0].classes[0].score)
        self._VerifyModelSpec(result.model_spec, request.model_spec.name, request.model_spec.signature_name, self._GetModelVersion(model_path))

    def testClassify(self):
        if False:
            print('Hello World!')
        'Test PredictionService.Classify implementation for TF1 model.'
        self._TestClassify(self._GetSavedModelBundlePath())

    def testClassifyTf2(self):
        if False:
            while True:
                i = 10
        'Test PredictionService.Classify implementation for TF2 model.'
        self._TestClassify(self._GetSavedModelHalfPlusTwoTf2())

    def _TestRegress(self, model_path):
        if False:
            print('Hello World!')
        'Test PredictionService.Regress implementation.'
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path)[1]
        print('Sending Regress request...')
        request = regression_pb2.RegressionRequest()
        request.model_spec.name = 'default'
        request.model_spec.signature_name = 'regress_x_to_y'
        example = request.input.example_list.examples.add()
        example.features.feature['x'].float_list.value.extend([2.0])
        channel = grpc.insecure_channel(model_server_address)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        result = stub.Regress(request, RPC_TIMEOUT)
        self.assertEqual(1, len(result.result.regressions))
        expected_output = 3.0
        self.assertEqual(expected_output, result.result.regressions[0].value)
        self._VerifyModelSpec(result.model_spec, request.model_spec.name, request.model_spec.signature_name, self._GetModelVersion(model_path))

    def testRegress(self):
        if False:
            print('Hello World!')
        'Test PredictionService.Regress implementation for TF1 model.'
        self._TestRegress(self._GetSavedModelBundlePath())

    def testRegressTf2(self):
        if False:
            return 10
        'Test PredictionService.Regress implementation for TF2 model.'
        self._TestRegress(self._GetSavedModelHalfPlusTwoTf2())

    def _TestMultiInference(self, model_path):
        if False:
            while True:
                i = 10
        'Test PredictionService.MultiInference implementation.'
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path)[1]
        print('Sending MultiInference request...')
        request = inference_pb2.MultiInferenceRequest()
        request.tasks.add().model_spec.name = 'default'
        request.tasks[0].model_spec.signature_name = 'regress_x_to_y'
        request.tasks[0].method_name = 'tensorflow/serving/regress'
        request.tasks.add().model_spec.name = 'default'
        request.tasks[1].model_spec.signature_name = 'classify_x_to_y'
        request.tasks[1].method_name = 'tensorflow/serving/classify'
        example = request.input.example_list.examples.add()
        example.features.feature['x'].float_list.value.extend([2.0])
        channel = grpc.insecure_channel(model_server_address)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        result = stub.MultiInference(request, RPC_TIMEOUT)
        self.assertEqual(2, len(result.results))
        expected_output = 3.0
        self.assertEqual(expected_output, result.results[0].regression_result.regressions[0].value)
        self.assertEqual(expected_output, result.results[1].classification_result.classifications[0].classes[0].score)
        for i in range(2):
            self._VerifyModelSpec(result.results[i].model_spec, request.tasks[i].model_spec.name, request.tasks[i].model_spec.signature_name, self._GetModelVersion(model_path))

    def testMultiInference(self):
        if False:
            i = 10
            return i + 15
        'Test PredictionService.MultiInference implementation for TF1 model.'
        self._TestMultiInference(self._GetSavedModelBundlePath())

    def testMultiInferenceTf2(self):
        if False:
            i = 10
            return i + 15
        'Test PredictionService.MultiInference implementation for TF2 model.'
        self._TestMultiInference(self._GetSavedModelHalfPlusTwoTf2())

    def testPredictSavedModel(self):
        if False:
            while True:
                i = 10
        'Test PredictionService.Predict implementation with SavedModel.'
        self._TestPredict(self._GetSavedModelBundlePath())

    def _TestBadModel(self):
        if False:
            while True:
                i = 10
        'Helper method to test against a bad model export.'
        model_path = (os.path.join(self.testdata_dir, 'bad_half_plus_two'),)
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path, wait_for_server_ready=False)[1]
        with self.assertRaises(grpc.RpcError) as ectxt:
            self.VerifyPredictRequest(model_server_address, expected_output=3.0, expected_version=self._GetModelVersion(model_path), signature_name='')
        self.assertIs(grpc.StatusCode.FAILED_PRECONDITION, ectxt.exception.code())

    def _TestBadModelUpconvertedSavedModel(self):
        if False:
            while True:
                i = 10
        'Test Predict against a bad upconverted SavedModel model export.'
        self._TestBadModel()

    def testGoodModelConfig(self):
        if False:
            i = 10
            return i + 15
        'Test server configuration from file works with valid configuration.'
        model_server_address = TensorflowModelServerTest.RunServer(None, None, model_config_file=self._GetGoodModelConfigFile())[1]
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_two', expected_output=3.0, expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_two', expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_three', expected_output=4.0, expected_version=self._GetModelVersion(self._GetSavedModelHalfPlusThreePath()))
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_three', expected_output=4.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelHalfPlusThreePath()))

    def testBadModelConfig(self):
        if False:
            return 10
        'Test server model configuration from file fails for invalid file.'
        proc = TensorflowModelServerTest.RunServer(None, None, model_config_file=self._GetBadModelConfigFile(), pipe=subprocess.PIPE, wait_for_server_ready=False)[0]
        error_message = 'Error parsing text-format tensorflow.serving.ModelServerConfig'
        error_message = error_message.encode('utf-8')
        self.assertNotEqual(proc.stderr, None)
        self.assertGreater(proc.stderr.read().find(error_message), -1)

    def testModelConfigReload(self):
        if False:
            print('Hello World!')
        'Test model server polls filesystem for model configuration.'
        base_config_proto = '\n    model_config_list: {{\n      config: {{\n        name: "{name}",\n        base_path: "{model_path}",\n        model_platform: "tensorflow"\n      }}\n    }}\n    '
        config_path = os.path.join(FLAGS.test_tmpdir, 'model_config.txt')
        with open(config_path, 'w') as f:
            f.write(base_config_proto.format(name='half_plus_two', model_path=self._GetSavedModelBundlePath()))
        poll_period = 1
        model_server_address = TensorflowModelServerTest.RunServer(None, None, model_config_file=config_path, model_config_file_poll_period=poll_period)[1]
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_two', expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))
        with open(config_path, 'w') as f:
            f.write(base_config_proto.format(name='half_plus_three', model_path=self._GetSavedModelHalfPlusThreePath()))
        time.sleep(poll_period + 1)
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_three', expected_output=4.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelHalfPlusThreePath()))

    def testModelConfigReloadWithZeroPollPeriod(self):
        if False:
            print('Hello World!')
        'Test model server does not poll filesystem for model config.'
        base_config_proto = '\n    model_config_list: {{\n      config: {{\n        name: "{name}",\n        base_path: "{model_path}",\n        model_platform: "tensorflow"\n      }}\n    }}\n    '
        config_path = os.path.join(FLAGS.test_tmpdir, 'model_config.txt')
        with open(config_path, 'w') as f:
            f.write(base_config_proto.format(name='half_plus_two', model_path=self._GetSavedModelBundlePath()))
        poll_period = 0
        model_server_address = TensorflowModelServerTest.RunServer(None, None, model_config_file=config_path, model_config_file_poll_period=poll_period)[1]
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_two', expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))
        with open(config_path, 'w') as f:
            f.write(base_config_proto.format(name='half_plus_three', model_path=self._GetSavedModelHalfPlusThreePath()))
        time.sleep(poll_period + 1)
        self.VerifyPredictRequest(model_server_address, model_name='half_plus_two', expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelHalfPlusThreePath()))

    def testGoodGrpcChannelArgs(self):
        if False:
            while True:
                i = 10
        'Test server starts with grpc_channel_arguments specified.'
        model_server_address = TensorflowModelServerTest.RunServer('default', self._GetSavedModelBundlePath(), grpc_channel_arguments='grpc.max_connection_age_ms=2000,grpc.lb_policy_name=grpclb')[1]
        self.VerifyPredictRequest(model_server_address, expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelHalfPlusThreePath()))

    def testClassifyREST(self):
        if False:
            print('Hello World!')
        'Test Classify implementation over REST API.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path)[2].split(':')
        url = 'http://{}:{}/v1/models/default:classify'.format(host, port)
        json_req = {'signature_name': 'classify_x_to_y', 'examples': [{'x': 2.0}]}
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, json_req)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        self.assertEqual(json.loads(resp_data.decode()), {'results': [[['', 3.0]]]})

    def testRegressREST(self):
        if False:
            i = 10
            return i + 15
        'Test Regress implementation over REST API.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path)[2].split(':')
        url = 'http://{}:{}/v1/models/default:regress'.format(host, port)
        json_req = {'signature_name': 'regress_x_to_y', 'examples': [{'x': 2.0}]}
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, json_req)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        self.assertEqual(json.loads(resp_data.decode()), {'results': [3.0]})

    def testPredictREST(self):
        if False:
            i = 10
            return i + 15
        'Test Predict implementation over REST API.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path)[2].split(':')
        url = 'http://{}:{}/v1/models/default:predict'.format(host, port)
        json_req = {'instances': [2.0, 3.0, 4.0]}
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, json_req)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        self.assertEqual(json.loads(resp_data.decode()), {'predictions': [3.0, 3.5, 4.0]})

    def testPredictColumnarREST(self):
        if False:
            i = 10
            return i + 15
        'Test Predict implementation over REST API with columnar inputs.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path)[2].split(':')
        url = 'http://{}:{}/v1/models/default:predict'.format(host, port)
        json_req = {'inputs': [2.0, 3.0, 4.0]}
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, json_req)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        self.assertEqual(json.loads(resp_data.decode()), {'outputs': [3.0, 3.5, 4.0]})

    def testGetStatusREST(self):
        if False:
            i = 10
            return i + 15
        'Test ModelStatus implementation over REST API with columnar inputs.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path)[2].split(':')
        url = 'http://{}:{}/v1/models/default'.format(host, port)
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, None)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        self.assertEqual(json.loads(resp_data.decode()), {'model_version_status': [{'version': '123', 'state': 'AVAILABLE', 'status': {'error_code': 'OK', 'error_message': ''}}]})

    def testGetModelMetadataREST(self):
        if False:
            i = 10
            return i + 15
        'Test ModelStatus implementation over REST API with columnar inputs.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path)[2].split(':')
        url = 'http://{}:{}/v1/models/default/metadata'.format(host, port)
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, None)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        try:
            model_metadata_file = self._GetModelMetadataFile()
            with open(model_metadata_file) as f:
                expected_metadata = json.load(f)
                self.assertEqual(tensorflow_model_server_test_base.SortedObject(json.loads(resp_data.decode())), tensorflow_model_server_test_base.SortedObject(expected_metadata))
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))

    def testPrometheusEndpoint(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ModelStatus implementation over REST API with columnar inputs.'
        model_path = self._GetSavedModelBundlePath()
        (host, port) = TensorflowModelServerTest.RunServer('default', model_path, monitoring_config_file=self._GetMonitoringConfigFile())[2].split(':')
        url = 'http://{}:{}/monitoring/prometheus/metrics'.format(host, port)
        resp_data = None
        try:
            resp_data = tensorflow_model_server_test_base.CallREST(url, None)
        except Exception as e:
            self.fail('Request failed with error: {}'.format(e))
        self.assertIn('# TYPE', resp_data.decode('utf-8') if resp_data is not None else None)

    def testPredictUDS(self):
        if False:
            for i in range(10):
                print('nop')
        'Test saved model prediction over a Unix domain socket.'
        _ = TensorflowModelServerTest.RunServer('default', self._GetSavedModelBundlePath())
        model_server_address = 'unix:%s' % GRPC_SOCKET_PATH
        self.VerifyPredictRequest(model_server_address, expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetSavedModelHalfPlusThreePath()))

    def testPredictOnTfLite(self):
        if False:
            while True:
                i = 10
        'Test saved model prediction on a TF Lite mode.'
        model_server_address = TensorflowModelServerTest.RunServer('default', self._GetTfLiteModelPath(), model_type='tflite')[1]
        self.VerifyPredictRequest(model_server_address, expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetTfLiteModelPath()))

    def testPredictWithSignatureDefOnTfLite(self):
        if False:
            i = 10
            return i + 15
        'Test saved model prediction on a TF Lite mode.'
        model_server_address = TensorflowModelServerTest.RunServer('default', self._GetTfLiteModelWithSigDefPath(), model_type='tflite')[1]
        self.VerifyPredictRequest(model_server_address, expected_output=3.0, specify_output=False, expected_version=self._GetModelVersion(self._GetTfLiteModelPath()))

    def test_tf_saved_model_save(self):
        if False:
            return 10
        base_path = os.path.join(self.get_temp_dir(), 'tf_saved_model_save')
        export_path = os.path.join(base_path, '00000123')
        root = tf.train.Checkpoint()
        root.v1 = tf.Variable(3.0)
        root.v2 = tf.Variable(2.0)
        root.f = tf.function(lambda x: {'y': root.v1 * root.v2 * x})
        to_save = root.f.get_concrete_function(tf.TensorSpec(None, tf.float32))
        tf.saved_model.experimental.save(root, export_path, to_save)
        (_, model_server_address, _) = TensorflowModelServerTest.RunServer('default', base_path)
        expected_version = self._GetModelVersion(base_path)
        self.VerifyPredictRequest(model_server_address, expected_output=12.0, specify_output=False, expected_version=expected_version)

    def test_tf_saved_model_save_multiple_signatures(self):
        if False:
            while True:
                i = 10
        base_path = os.path.join(self.get_temp_dir(), 'tf_saved_model_save')
        export_path = os.path.join(base_path, '00000123')
        root = tf.train.Checkpoint()
        root.f = tf.function(lambda x: {'y': 1.0}, input_signature=[tf.TensorSpec(None, tf.float32)])
        root.g = tf.function(lambda x: {'y': 2.0}, input_signature=[tf.TensorSpec(None, tf.float32)])
        tf.saved_model.experimental.save(root, export_path, signatures={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: root.f, 'custom_signature_key': root.g})
        (_, model_server_address, _) = TensorflowModelServerTest.RunServer('default', base_path)
        expected_version = self._GetModelVersion(base_path)
        self.VerifyPredictRequest(model_server_address, expected_output=2.0, expected_version=expected_version, signature_name='custom_signature_key')
        self.VerifyPredictRequest(model_server_address, expected_output=1.0, expected_version=expected_version)

    def test_sequential_keras_saved_model_save(self):
        if False:
            i = 10
            return i + 15
        'Test loading a simple SavedModel created with Keras Sequential API.'
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(dtype='float32', shape=(1,), name='x'))
        model.add(tf.keras.layers.Lambda(lambda x: x, name='y'))
        base_path = os.path.join(self.get_temp_dir(), 'keras_sequential_saved_model_save')
        export_path = os.path.join(base_path, '00000123')
        tf.saved_model.save(model, export_path)
        (_, model_server_address, _) = TensorflowModelServerTest.RunServer('default', base_path)
        expected_version = self._GetModelVersion(base_path)
        self.VerifyPredictRequest(model_server_address, batch_input=True, specify_output=False, expected_output=2.0, expected_version=expected_version)

    def test_distrat_sequential_keras_saved_model_save(self):
        if False:
            for i in range(10):
                print('nop')
        'Test loading a Keras SavedModel with tf.distribute.'
        tensorflow_model_server_test_base.SetVirtualCpus(2)
        strategy = tf.distribute.MirroredStrategy(devices=('/cpu:0', '/cpu:1'))
        with strategy.scope():
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(dtype='float32', shape=(1,), name='x'))
            model.add(tf.keras.layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros'))
            model.add(tf.keras.layers.Lambda(lambda x: x, name='y'))
        base_path = os.path.join(self.get_temp_dir(), 'keras_sequential_saved_model_save')
        export_path = os.path.join(base_path, '00000123')
        tf.saved_model.save(model, export_path)
        (_, model_server_address, _) = TensorflowModelServerTest.RunServer('default', base_path)
        expected_version = self._GetModelVersion(base_path)
        self.VerifyPredictRequest(model_server_address, batch_input=True, specify_output=False, expected_output=2.0, expected_version=expected_version)

    def test_profiler_service_with_valid_trace_request(self):
        if False:
            i = 10
            return i + 15
        'Test integration with profiler service by sending tracing requests.'
        model_path = self._GetSavedModelBundlePath()
        (_, grpc_addr, rest_addr) = TensorflowModelServerTest.RunServer('default', model_path)
        url = 'http://{}/v1/models/default:predict'.format(rest_addr)
        json_req = '{"instances": [2.0, 3.0, 4.0]}'
        exec_command = "wget {} --content-on-error=on -O- --post-data  '{}' --header='Content-Type:application/json'".format(url, json_req)
        repeat_command = 'for n in {{1..3}}; do {} & sleep 1; done;'.format(exec_command)
        proc = subprocess.Popen(repeat_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logdir = os.path.join(self.temp_dir, 'logs')
        worker_list = ''
        duration_ms = 1000
        num_tracing_attempts = 10
        os.makedirs(logdir)
        profiler_client.trace(grpc_addr, logdir, duration_ms, worker_list, num_tracing_attempts)
        (out, err) = proc.communicate()
        print("stdout: '{}' | stderr: '{}'".format(out, err))

    def test_tf_text(self):
        if False:
            while True:
                i = 10
        'Test TF Text.'
        model_path = os.path.join(flags.os.environ['TEST_SRCDIR'], 'tf_serving/tensorflow_serving', 'servables/tensorflow/testdata', 'tf_text_regression')
        model_server_address = TensorflowModelServerTest.RunServer('default', model_path)[1]
        self.VerifyPredictRequest(model_server_address, expected_output=3.0, expected_version=self._GetModelVersion(model_path), rpc_timeout=600)
if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()