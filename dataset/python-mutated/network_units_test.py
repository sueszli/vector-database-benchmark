"""Tests for network_units."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from dragnn.protos import spec_pb2
from dragnn.python import network_units

class NetworkUnitsConverterTest(test_util.TensorFlowTestCase):

    def testConvertNetworkStateTensorarray(self):
        if False:
            i = 10
            return i + 15
        with self.test_session() as session:
            ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False)
            ta = ta.write(0, [[0.0, 0.0]] * 2)
            ta = ta.write(1, [[1.0, 10.0]] * 2)
            ta = ta.write(2, [[2.0, 20.0]] * 2)
            ta = ta.write(3, [[3.0, 30.0]] * 2)
            tensor = network_units.convert_network_state_tensorarray(ta)
            actual = session.run(tensor)
            self.assertEqual(actual.shape, (6, 2))
            expected = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
            self.assertAllEqual(actual, expected)

class MockComponent(object):

    def __init__(self, master, component_spec):
        if False:
            return 10
        self.master = master
        self.spec = component_spec
        self.name = component_spec.name
        self.beam_size = 1
        self.num_actions = 45
        self._attrs = {}

    def attr(self, name):
        if False:
            return 10
        return self._attrs[name]

    def get_variable(self, name):
        if False:
            for i in range(10):
                print('nop')
        return tf.get_variable(name)

class MockMaster(object):

    def __init__(self, build_runtime_graph=False):
        if False:
            while True:
                i = 10
        self.spec = spec_pb2.MasterSpec()
        self.hyperparams = spec_pb2.GridPoint()
        self.lookup_component = {'previous': MockComponent(self, spec_pb2.ComponentSpec())}
        self.build_runtime_graph = build_runtime_graph

class MockNetwork(object):

    def __init__(self, **dims):
        if False:
            while True:
                i = 10
        self._dims = dims

    def get_layer_size(self, name):
        if False:
            print('Hello World!')
        return self._dims[name]

class NetworkUnitsLookupTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        tf.reset_default_graph()
        self._master = MockMaster()
        self._master.spec = spec_pb2.MasterSpec()
        component_spec = self._master.spec.component.add()
        component_spec.name = 'fake_linked'
        component_spec.backend.registered_name = 'FakeComponent'
        linked_feature = component_spec.linked_feature.add()
        linked_feature.source_component = 'fake_linked'
        linked_feature.source_translator = 'identity'
        linked_feature.embedding_dim = -1
        linked_feature.size = 2
        self._linked_component = MockComponent(self._master, component_spec)
        component_spec = self._master.spec.component.add()
        component_spec.name = 'fake_fixed'
        component_spec.backend.registered_name = 'FakeComponent'
        fixed_feature = component_spec.fixed_feature.add()
        fixed_feature.fml = 'input.word'
        fixed_feature.embedding_dim = 1
        fixed_feature.size = 1
        self._fixed_component = MockComponent(self._master, component_spec)

    def testExportFixedFeaturesNetworkWithEnabledEmbeddingMatrix(self):
        if False:
            i = 10
            return i + 15
        network = network_units.ExportFixedFeaturesNetwork(self._fixed_component)
        self.assertEqual(1, len(network.params))

    def testExportFixedFeaturesNetworkWithDisabledEmbeddingMatrix(self):
        if False:
            print('Hello World!')
        self._fixed_component.spec.fixed_feature[0].embedding_dim = -1
        network = network_units.ExportFixedFeaturesNetwork(self._fixed_component)
        self.assertEqual(0, len(network.params))

class GetAttrsWithDefaultsTest(test_util.TensorFlowTestCase):

    def MakeAttrs(self, defaults, key=None, value=None):
        if False:
            print('Hello World!')
        'Returns attrs based on the |defaults| and one |key|,|value| override.'
        spec = spec_pb2.RegisteredModuleSpec()
        if key and value:
            spec.parameters[key] = value
        return network_units.get_attrs_with_defaults(spec.parameters, defaults)

    def testFalseValues(self):
        if False:
            print('Hello World!')

        def _assert_attr_is_false(value=None):
            if False:
                return 10
            key = 'foo'
            attrs = self.MakeAttrs({key: False}, key, value)
            self.assertFalse(attrs[key])
        _assert_attr_is_false()
        _assert_attr_is_false('false')
        _assert_attr_is_false('False')
        _assert_attr_is_false('FALSE')
        _assert_attr_is_false('no')
        _assert_attr_is_false('whatever')
        _assert_attr_is_false('   ')
        _assert_attr_is_false('')

    def testTrueValues(self):
        if False:
            return 10

        def _assert_attr_is_true(value=None):
            if False:
                print('Hello World!')
            key = 'foo'
            attrs = self.MakeAttrs({key: False}, key, value)
            self.assertTrue(attrs[key])
        _assert_attr_is_true('true')
        _assert_attr_is_true('True')
        _assert_attr_is_true('TRUE')

class LstmNetworkTest(test_util.TensorFlowTestCase):
    test_spec_1 = '\n      component {\n        name: \'bi_lstm\'\n        backend { registered_name: \'TestComponent\' }\n        fixed_feature {\n          name: \'words\'\n          fml: \'words\'\n          size: 1\n          embedding_dim: 32\n          vocabulary_size: 1079813,\n        }\n        network_unit {\n          registered_name: \'LSTMNetwork\'\n          parameters {\n            key: "hidden_layer_sizes"\n            value: "128"\n          }\n        }\n      }\n    '
    test_spec_linked = '\n      component {\n        name: \'bi_lstm\'\n        backend { registered_name: \'TestComponent\' }\n        fixed_feature {\n          name: \'words\'\n          fml: \'words\'\n          size: 1\n          embedding_dim: 32\n          vocabulary_size: 1079813,\n        }\n        linked_feature {\n          name: \'lstm_h\'\n          fml: \'bias(0)\'\n          embedding_dim: -1\n          size: 1\n          source_component: \'bi_lstm\'\n          source_translator: \'history\'\n          source_layer: \'lstm_h\'\n        }\n        linked_feature {\n          name: \'lstm_c\'\n          fml: \'bias(0)\'\n          embedding_dim: -1\n          size: 1\n          source_component: \'bi_lstm\'\n          source_translator: \'history\'\n          source_layer: \'lstm_c\'\n        }\n        network_unit {\n          registered_name: \'LSTMNetwork\'\n          parameters {\n            key: "hidden_layer_sizes"\n            value: "128"\n          }\n        }\n      }\n    '

    def setUp(self):
        if False:
            while True:
                i = 10
        tf.reset_default_graph()

    def construct_lstm_network_unit(self, master):
        if False:
            return 10
        "Helper to construct a LSTMNetwork. Doesn't call create() yet."
        component = MockComponent(master, master.spec.component[0])
        with tf.variable_scope('bi_lstm'):
            lstm_network_unit = network_units.LSTMNetwork(component)
        return lstm_network_unit

    def get_context_tensor_arrays(self, lstm_network_unit):
        if False:
            i = 10
            return i + 15
        context_tensor_arrays = []
        for context_layer in lstm_network_unit.context_layers:
            context_tensor_arrays.append(context_layer.create_array(1))
        return context_tensor_arrays

    def fixed_word_embeddings(self):
        if False:
            i = 10
            return i + 15
        'Helper for returning fixed embeddings, for 1 word feature.'
        words_tensor = tf.constant([[1.0] * 32], dtype=tf.float32)
        return [network_units.NamedTensor(words_tensor, 'words')]

    def testCanCreate(self):
        if False:
            for i in range(10):
                print('nop')
        "Smoke test that the create() function doesn't raise errors."
        master = MockMaster()
        master.spec = spec_pb2.MasterSpec()
        text_format.Parse(self.test_spec_1, master.spec)
        lstm_network_unit = self.construct_lstm_network_unit(master)
        with tf.variable_scope('bi_lstm', reuse=True):
            lstm_network_unit.create(self.fixed_word_embeddings(), [], self.get_context_tensor_arrays(lstm_network_unit), None, True)

    def testCanCreateLinked(self):
        if False:
            return 10
        "Smoke test that the create() function doesn't raise errors."
        master = MockMaster()
        master.spec = spec_pb2.MasterSpec()
        text_format.Parse(self.test_spec_linked, master.spec)
        lstm_network_unit = self.construct_lstm_network_unit(master)
        with tf.variable_scope('bi_lstm', reuse=True):
            lstm_network_unit.create(self.fixed_word_embeddings(), [], self.get_context_tensor_arrays(lstm_network_unit), None, True)

    def testRuntimeConcatentatedMatrices(self):
        if False:
            i = 10
            return i + 15
        'Test generation of concatenated matrices.'
        master = MockMaster(build_runtime_graph=False)
        master.spec = spec_pb2.MasterSpec()
        text_format.Parse(self.test_spec_1, master.spec)
        lstm_network_unit = self.construct_lstm_network_unit(master)
        with tf.variable_scope('bi_lstm', reuse=True):
            lstm_network_unit.create(self.fixed_word_embeddings(), [], self.get_context_tensor_arrays(lstm_network_unit), None, False)
            x_to_ico = lstm_network_unit.derived_params[0]()
            h_to_ico = lstm_network_unit.derived_params[1]()
            ico_bias = lstm_network_unit.derived_params[2]()
            self.assertEqual(x_to_ico.shape, (32, 384))
            self.assertEqual(x_to_ico.op.name, 'bi_lstm/x_to_ico')
            self.assertEqual(h_to_ico.shape, (128, 384))
            self.assertEqual(h_to_ico.op.name, 'bi_lstm/h_to_ico')
            self.assertEqual(ico_bias.shape, (384,))
            self.assertEqual(ico_bias.op.name, 'bi_lstm/ico_bias')

    def testRuntimeConcatentatedMatricesLinked(self):
        if False:
            while True:
                i = 10
        'Test generation of concatenated matrices.'
        master = MockMaster(build_runtime_graph=False)
        master.spec = spec_pb2.MasterSpec()
        text_format.Parse(self.test_spec_linked, master.spec)
        lstm_network_unit = self.construct_lstm_network_unit(master)
        with tf.variable_scope('bi_lstm', reuse=True):
            lstm_network_unit.create(self.fixed_word_embeddings(), [], self.get_context_tensor_arrays(lstm_network_unit), None, False)
            x_to_ico = lstm_network_unit.derived_params[0]()
            h_to_ico = lstm_network_unit.derived_params[1]()
            ico_bias = lstm_network_unit.derived_params[2]()
            self.assertEqual(x_to_ico.shape, (32, 384))
            self.assertEqual(h_to_ico.shape, (128, 384))
            self.assertEqual(ico_bias.shape, (384,))

class GatherNetworkTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            return 10
        tf.reset_default_graph()
        self._master = MockMaster()
        self._master.spec = spec_pb2.MasterSpec()
        text_format.Parse("\n      component {\n        name: 'test'\n        backend { registered_name: 'TestComponent' }\n        linked_feature {\n          name: 'indices'\n          fml: 'input.focus'\n          size: 1\n          embedding_dim: -1\n          source_component: 'previous'\n          source_translator: 'identity'\n          source_layer: 'index_layer'\n        }\n        linked_feature {\n          name: 'features'\n          fml: 'input.focus'\n          size: 1\n          embedding_dim: -1\n          source_component: 'previous'\n          source_translator: 'identity'\n          source_layer: 'feature_layer'\n        }\n        network_unit {\n          registered_name: 'GatherNetwork'\n        }\n      }\n    ", self._master.spec)
        self._component = MockComponent(self._master, self._master.spec.component[0])
        self._master.lookup_component['previous'].network = MockNetwork(index_layer=1, feature_layer=2)

    def testConstantPadding(self):
        if False:
            for i in range(10):
                print('nop')
        with tf.Graph().as_default(), self.test_session():
            with tf.variable_scope('test_scope'):
                network = network_units.GatherNetwork(self._component)
            indices = tf.constant([[1], [2], [0], [-1], [0], [-1]], dtype=tf.int64)
            features = tf.constant([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5], [6.0, 6.5]], dtype=tf.float32)
            fixed_embeddings = []
            linked_embeddings = [network_units.NamedTensor(indices, 'indices', 1), network_units.NamedTensor(features, 'features', 2)]
            with tf.variable_scope('test_scope', reuse=True):
                outputs = network.create(fixed_embeddings, linked_embeddings, None, None, True, 2)
            gathered = outputs[0]
            self.assertAllEqual(gathered.eval(), [[2.0, 2.5], [3.0, 3.5], [1.0, 1.5], [0.0, 0.0], [4.0, 4.5], [0.0, 0.0]])

    def testTrainablePadding(self):
        if False:
            for i in range(10):
                print('nop')
        self._component.spec.network_unit.parameters['trainable_padding'] = 'true'
        with tf.Graph().as_default(), self.test_session():
            with tf.variable_scope('test_scope'):
                network = network_units.GatherNetwork(self._component)
            indices = tf.constant([[1], [2], [0], [-1], [0], [-1]], dtype=tf.int64)
            features = tf.constant([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5], [6.0, 6.5]], dtype=tf.float32)
            fixed_embeddings = []
            linked_embeddings = [network_units.NamedTensor(indices, 'indices', 1), network_units.NamedTensor(features, 'features', 2)]
            with tf.variable_scope('test_scope', reuse=True):
                outputs = network.create(fixed_embeddings, linked_embeddings, None, None, True, 2)
            gathered = outputs[0]
            tf.global_variables_initializer().run()
            self.assertAllEqual(gathered[0].eval(), [2.0, 2.5])
            self.assertAllEqual(gathered[1].eval(), [3.0, 3.5])
            self.assertAllEqual(gathered[2].eval(), [1.0, 1.5])
            tf.logging.info('padding = %s', gathered[3].eval())
            self.assertAllEqual(gathered[4].eval(), [4.0, 4.5])
            tf.logging.info('padding = %s', gathered[5].eval())
            self.assertAllEqual(gathered[3].eval(), gathered[5].eval())

class IdentityInitializerTest(test_util.TensorFlowTestCase):

    def IdentityInitializerHelper(self, shape, expected, divisor=1.0, std=0.0001):
        if False:
            return 10
        "Tests identity initialization by comparing expected to actual array.\n\n    Tests the given expected array against the result of calling\n    network_units.add_var_initialized() with the given params and\n    init_type='identity'.\n\n    Args:\n      shape: shape of the array\n      expected: expected contents of the array to initialize\n      divisor: numerator for identity initialization where the last two dims\n        of the array are not equal; should divide both of the last two dims\n      std: standard deviation for random normal samples\n    "
        with tf.Graph().as_default(), self.test_session() as session:
            np.random.seed(4)
            tensor = network_units.add_var_initialized('tensor', shape, 'identity', divisor=divisor, stddev=std)
            session.run(tf.global_variables_initializer())
            actual = session.run(tensor)
            self.assertAllClose(actual, expected, 1e-08, 1e-08)

    def IdentityInitializerSquareHelper(self, shape, middles):
        if False:
            i = 10
            return i + 15
        'Tests identity initialization when last two dims are equal.\n\n    When the last two dims of the array are equal, identity initialization\n    should simply set the center matrix in the last two dimensions to the\n    identity, with all other entries set to zero.\n\n    Args:\n      shape: shape of the array to initialize\n      middles: indices into the middle of all axes except the last two. It\n          must be the case that len(middles) == len(shape) - 2.\n    '
        expected = np.zeros(shape, dtype='float32')
        expected[[[m] for m in middles]] = np.eye(shape[-1])
        self.IdentityInitializerHelper(shape, expected)

    def testIdentityInitializerSquareRank2(self):
        if False:
            print('Hello World!')
        shape = (3, 3)
        expected = np.eye(shape[-1]).astype('float32')
        self.IdentityInitializerHelper(shape, expected)

    def testIdentityInitializerSquareRank3(self):
        if False:
            while True:
                i = 10
        shape = (2, 4, 4)
        middles = [1]
        self.IdentityInitializerSquareHelper(shape, middles)

    def testIdentityInitializerSquareRank4(self):
        if False:
            i = 10
            return i + 15
        shape = (2, 3, 4, 4)
        middles = [1, 1]
        self.IdentityInitializerSquareHelper(shape, middles)

    def testIdentityInitializerSquareRank5(self):
        if False:
            print('Hello World!')
        shape = (2, 3, 4, 5, 5)
        middles = [1, 1, 2]
        self.IdentityInitializerSquareHelper(shape, middles)

    def testIdentityInitializerNonSquareRank2FirstDimLarger(self):
        if False:
            for i in range(10):
                print('nop')
        divisor = 3.0
        std = 0.001
        shape = (6, 3)
        m = divisor / shape[-1]
        expected = [[m, 0.000499951362, -0.00099590898], [m, -0.000418301526, -0.00158457726], [-0.000647706795, m, 0.000332250027], [-0.00114747661, m, -8.79869258e-05], [0.000425072387, 0.000332253141, m], [0.000350997143, -0.000606887275, m]]
        self.IdentityInitializerHelper(shape, expected, divisor, std)

    def testIdentityInitializerNonSquareRank2FirstDimSmaller(self):
        if False:
            print('Hello World!')
        divisor = 2.0
        std = 0.001
        shape = (2, 4)
        m = divisor / shape[-1]
        expected = [[m, m, -0.00099590898, 0.000693598529], [-0.000418301526, -0.00158457726, m, m]]
        self.IdentityInitializerHelper(shape, expected, divisor, std)

    def testIdentityInitializerNonSquareRank3(self):
        if False:
            while True:
                i = 10
        divisor = 2.0
        std = 0.001
        shape = (2, 2, 6)
        m = divisor / shape[-1]
        expected = [[[5.05617063e-05, 0.000499951362, -0.00099590898, 0.000693598529, -0.000418301526, -0.00158457726], [-0.000647706795, 0.000598575163, 0.000332250027, -0.00114747661, 0.00061866967, -8.79869258e-05]], [[m, m, m, 0.000350997143, -0.000606887275, 0.0015469793], [0.000723341596, 4.61355667e-05, -0.000982991653, m, m, m]]]
        self.IdentityInitializerHelper(shape, expected, divisor, std)

    def testIdentityInitializerNonSquareRank4(self):
        if False:
            print('Hello World!')
        divisor = 2.0
        std = 0.001
        shape = (2, 3, 2, 8)
        m = divisor / float(shape[-1])
        expected = [[[[5.05617063e-05, 0.000499951362, -0.00099590898, 0.000693598529, -0.000418301526, -0.00158457726, -0.000647706795, 0.000598575163], [0.000332250027, -0.00114747661, 0.00061866967, -8.79869258e-05, 0.000425072387, 0.000332253141, -0.00115681626, 0.000350997143]], [[-0.000606887275, 0.0015469793, 0.000723341596, 4.61355667e-05, -0.000982991653, 5.44327377e-05, 0.000159892938, -0.0012089482], [0.00222336012, 0.000394295203, 0.00169235771, -0.0011128122, 0.0016357475, -0.00136096554, -0.000651225855, 0.000542451337]], [[4.80062481e-05, -0.0023580736, -0.00110558409, 0.000837836356, 0.00208787085, 0.000914840959, -0.000276203355, 0.000796511886], [-0.00114379858, 0.000509919773, -0.00134746032, -9.36010019e-06, -0.000130704633, 0.000802086608, -0.000302963977, 0.00120200263]]], [[[-0.000196745284, 0.000836528721, 0.000786602264, -0.00184087583, 3.75474883e-05, 3.5928053e-05, -0.000778739923, 0.000179410708], [-0.00145553437, 0.000556185201, 0.000509778853, 0.000300445536, 0.00247658417, 0.000352343399, 6.74710027e-05, -0.000732264714]], [[m, m, m, m, 0.000158469542, 0.00199008291, 0.00116418756, 0.000242660157], [0.00137992005, -5.45587063e-05, 0.000795233937, 1.90899627e-05, m, m, m, m]], [[-0.00109712186, -0.000528196048, -0.00237977528, -0.000607683673, -0.00107529014, 0.00202240516, -0.000564875314, -0.00154292909], [0.000870841788, -0.000175210531, 4.86030076e-05, 0.000188646198, 0.000209313483, -0.000374444906, 0.000954698597, 0.00052324764]]]]
        self.IdentityInitializerHelper(shape, expected, divisor, std)

class FeatureIdDropoutTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            return 10
        tf.reset_default_graph()

    def testApplyFeatureIdDropout(self):
        if False:
            i = 10
            return i + 15
        channel = spec_pb2.FixedFeatureChannel()
        text_format.Parse('\n      vocabulary_size: 10\n      dropout_id: 8\n      dropout_keep_probability: [0.0, 0.25, 0.5, 0.75, 1.0]\n    ', channel)
        with tf.Graph().as_default(), self.test_session():
            with tf.variable_scope('test_scope'):
                ids = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.int64)
                weights = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
                tensors = network_units.apply_feature_id_dropout(ids, weights, channel)
                perturbed_ids = tensors[0].eval()
                tf.logging.info('perturbed_ids = %s', perturbed_ids)
                self.assertEqual(perturbed_ids[0], channel.dropout_id)
                self.assertTrue(perturbed_ids[1] in (1, channel.dropout_id))
                self.assertTrue(perturbed_ids[2] in (2, channel.dropout_id))
                self.assertTrue(perturbed_ids[3] in (3, channel.dropout_id))
                self.assertAllEqual(perturbed_ids[4:], [4, 5, 6, 7, 8, 9])

    def testApplyFeatureIdDropoutSkip(self):
        if False:
            while True:
                i = 10
        channel = spec_pb2.FixedFeatureChannel()
        text_format.Parse('\n      vocabulary_size: 2\n      dropout_id: 2\n      dropout_keep_probability: [0.0, 1.0]\n    ', channel)
        with tf.Graph().as_default(), self.test_session():
            with tf.variable_scope('test_scope'):
                ids = tf.constant([0, 1], dtype=tf.int64)
                weights = tf.constant([1, 1], dtype=tf.float32)
                tensors = network_units.apply_feature_id_dropout(ids, weights, channel)
                (perturbed_ids, perturbed_weights) = (tensors[0].eval(), tensors[1].eval())
                tf.logging.info('perturbed_ids = %s', perturbed_ids)
                tf.logging.info('perturbed_weights = %s', perturbed_weights)
                self.assertEqual(perturbed_ids[0], channel.dropout_id)
                self.assertEqual(perturbed_weights[0], 0)
                self.assertEqual(perturbed_ids[1], 1)
                self.assertEqual(perturbed_weights[1], 1)
if __name__ == '__main__':
    googletest.main()