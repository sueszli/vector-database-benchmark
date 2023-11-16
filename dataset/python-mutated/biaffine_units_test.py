"""Tests for biaffine_units."""
import tensorflow as tf
from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import biaffine_units
from dragnn.python import network_units
_BATCH_SIZE = 11
_NUM_TOKENS = 22
_TOKEN_DIM = 33

class MockNetwork(object):

    def __init__(self):
        if False:
            return 10
        pass

    def get_layer_size(self, unused_name):
        if False:
            print('Hello World!')
        return _TOKEN_DIM

class MockComponent(object):

    def __init__(self, master, component_spec):
        if False:
            while True:
                i = 10
        self.master = master
        self.spec = component_spec
        self.name = component_spec.name
        self.network = MockNetwork()
        self.beam_size = 1
        self.num_actions = 45
        self._attrs = {}

    def attr(self, name):
        if False:
            return 10
        return self._attrs[name]

    def get_variable(self, name):
        if False:
            print('Hello World!')
        return tf.get_variable(name)

class MockMaster(object):

    def __init__(self):
        if False:
            return 10
        self.spec = spec_pb2.MasterSpec()
        self.hyperparams = spec_pb2.GridPoint()
        self.lookup_component = {'previous': MockComponent(self, spec_pb2.ComponentSpec())}

def _make_biaffine_spec():
    if False:
        for i in range(10):
            print('nop')
    'Returns a ComponentSpec that the BiaffineDigraphNetwork works on.'
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse('\n    name: "test_component"\n    backend { registered_name: "TestComponent" }\n    linked_feature {\n      name: "sources"\n      fml: "input.focus"\n      source_translator: "identity"\n      source_component: "previous"\n      source_layer: "sources"\n      size: 1\n      embedding_dim: -1\n    }\n    linked_feature {\n      name: "targets"\n      fml: "input.focus"\n      source_translator: "identity"\n      source_component: "previous"\n      source_layer: "targets"\n      size: 1\n      embedding_dim: -1\n    }\n    network_unit {\n      registered_name: "biaffine_units.BiaffineDigraphNetwork"\n    }\n  ', component_spec)
    return component_spec

class BiaffineDigraphNetworkTest(tf.test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        tf.reset_default_graph()

    def testCanCreate(self):
        if False:
            i = 10
            return i + 15
        'Tests that create() works on a good spec.'
        with tf.Graph().as_default(), self.test_session():
            master = MockMaster()
            component = MockComponent(master, _make_biaffine_spec())
            with tf.variable_scope(component.name, reuse=None):
                component.network = biaffine_units.BiaffineDigraphNetwork(component)
            with tf.variable_scope(component.name, reuse=True):
                sources = network_units.NamedTensor(tf.zeros([_BATCH_SIZE * _NUM_TOKENS, _TOKEN_DIM]), 'sources')
                targets = network_units.NamedTensor(tf.zeros([_BATCH_SIZE * _NUM_TOKENS, _TOKEN_DIM]), 'targets')
                component.network.create(fixed_embeddings=[], linked_embeddings=[sources, targets], context_tensor_arrays=None, attention_tensor=None, during_training=True, stride=_BATCH_SIZE)

    def testDerivedParametersForRuntime(self):
        if False:
            i = 10
            return i + 15
        'Test generation of derived parameters for the runtime.'
        with tf.Graph().as_default(), self.test_session():
            master = MockMaster()
            component = MockComponent(master, _make_biaffine_spec())
            with tf.variable_scope(component.name, reuse=None):
                component.network = biaffine_units.BiaffineDigraphNetwork(component)
            with tf.variable_scope(component.name, reuse=True):
                self.assertEqual(len(component.network.derived_params), 2)
                root_weights = component.network.derived_params[0]()
                root_bias = component.network.derived_params[1]()
                self.assertAllEqual(root_weights.shape.as_list(), [1, _TOKEN_DIM])
                self.assertAllEqual(root_bias.shape.as_list(), [1, 1])
if __name__ == '__main__':
    tf.test.main()