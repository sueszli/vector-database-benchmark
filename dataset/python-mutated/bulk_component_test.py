"""Tests for bulk_component.

Verifies that:
1. BulkFeatureExtractor and BulkAnnotator both raise NotImplementedError when
   non-identity translator configured.
2. BulkFeatureExtractor and BulkAnnotator both raise RuntimeError when
   recurrent linked features are configured.
3. BulkAnnotator raises RuntimeError when fixed features are configured.
4. BulkFeatureIdExtractor raises ValueError when linked features are configured,
   or when the fixed features are invalid.
"""
import os.path
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import bulk_component
from dragnn.python import component
from dragnn.python import dragnn_ops
from dragnn.python import network_units
from syntaxnet import sentence_pb2

class MockNetworkUnit(object):

    def get_layer_size(self, unused_layer_name):
        if False:
            i = 10
            return i + 15
        return 64

class MockComponent(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.name = 'mock'
        self.network = MockNetworkUnit()

class MockMaster(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.spec = spec_pb2.MasterSpec()
        self.hyperparams = spec_pb2.GridPoint()
        self.lookup_component = {'mock': MockComponent()}
        self.build_runtime_graph = False

def _create_fake_corpus():
    if False:
        print('Hello World!')
    'Returns a list of fake serialized sentences for tests.'
    num_docs = 4
    corpus = []
    for num_tokens in range(1, num_docs + 1):
        sentence = sentence_pb2.Sentence()
        sentence.text = 'x' * num_tokens
        for i in range(num_tokens):
            token = sentence.token.add()
            token.word = 'x'
            token.start = i
            token.end = i
        corpus.append(sentence.SerializeToString())
    return corpus

class BulkComponentTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        tf.reset_default_graph()
        self.master = MockMaster()
        self.master_state = component.MasterState(handle=tf.constant(['foo', 'bar']), current_batch_size=2)
        self.network_states = {'mock': component.NetworkState(), 'test': component.NetworkState()}

    def testFailsOnNonIdentityTranslator(self):
        if False:
            while True:
                i = 10
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        linked_feature {\n          name: "features" embedding_dim: -1 size: 1\n          source_translator: "history"\n          source_component: "mock"\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        with self.assertRaises(NotImplementedError):
            comp.build_greedy_training(self.master_state, self.network_states)
        self.setUp()
        comp = bulk_component.BulkAnnotatorComponentBuilder(self.master, component_spec)
        with self.assertRaises(NotImplementedError):
            comp.build_greedy_training(self.master_state, self.network_states)

    def testFailsOnRecurrentLinkedFeature(self):
        if False:
            i = 10
            return i + 15
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "FeedForwardNetwork"\n          parameters {\n            key: \'hidden_layer_sizes\' value: \'64\'\n          }\n        }\n        linked_feature {\n          name: "features" embedding_dim: -1 size: 1\n          source_translator: "identity"\n          source_component: "test"\n          source_layer: "layer_0"\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        with self.assertRaises(RuntimeError):
            comp.build_greedy_training(self.master_state, self.network_states)
        self.setUp()
        comp = bulk_component.BulkAnnotatorComponentBuilder(self.master, component_spec)
        with self.assertRaises(RuntimeError):
            comp.build_greedy_training(self.master_state, self.network_states)

    def testConstantFixedFeatureFailsIfNotPretrained(self):
        if False:
            while True:
                i = 10
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: 32 size: 1\n          is_constant: true\n        }\n        component_builder {\n          registered_name: "bulk_component.BulkFeatureExtractorComponentBuilder"\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        with self.assertRaisesRegexp(ValueError, 'Constant embeddings must be pretrained'):
            comp.build_greedy_training(self.master_state, self.network_states)
        with self.assertRaisesRegexp(ValueError, 'Constant embeddings must be pretrained'):
            comp.build_greedy_inference(self.master_state, self.network_states, during_training=True)
        with self.assertRaisesRegexp(ValueError, 'Constant embeddings must be pretrained'):
            comp.build_greedy_inference(self.master_state, self.network_states, during_training=False)

    def testNormalFixedFeaturesAreDifferentiable(self):
        if False:
            for i in range(10):
                print('nop')
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: 32 size: 1\n          pretrained_embedding_matrix { part {} }\n          vocab { part {} }\n        }\n        component_builder {\n          registered_name: "bulk_component.BulkFeatureExtractorComponentBuilder"\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        with tf.variable_scope(comp.name, reuse=True):
            fixed_embedding_matrix = tf.get_variable(network_units.fixed_embeddings_name(0))
        comp.build_greedy_training(self.master_state, self.network_states)
        activations = self.network_states[comp.name].activations
        outputs = activations[comp.network.layers[0].name].bulk_tensor
        gradients = tf.gradients(outputs, fixed_embedding_matrix)
        self.assertEqual(len(gradients), 1)
        self.assertFalse(gradients[0] is None)

    def testConstantFixedFeaturesAreNotDifferentiableButOthersAre(self):
        if False:
            return 10
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "constant" embedding_dim: 32 size: 1\n          is_constant: true\n          pretrained_embedding_matrix { part {} }\n          vocab { part {} }\n        }\n        fixed_feature {\n          name: "trainable" embedding_dim: 32 size: 1\n          pretrained_embedding_matrix { part {} }\n          vocab { part {} }\n        }\n        component_builder {\n          registered_name: "bulk_component.BulkFeatureExtractorComponentBuilder"\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        with tf.variable_scope(comp.name, reuse=True):
            constant_embedding_matrix = tf.get_variable(network_units.fixed_embeddings_name(0))
            trainable_embedding_matrix = tf.get_variable(network_units.fixed_embeddings_name(1))
        comp.build_greedy_training(self.master_state, self.network_states)
        activations = self.network_states[comp.name].activations
        outputs = activations[comp.network.layers[0].name].bulk_tensor
        constant_gradients = tf.gradients(outputs, constant_embedding_matrix)
        self.assertEqual(len(constant_gradients), 1)
        self.assertTrue(constant_gradients[0] is None)
        trainable_gradients = tf.gradients(outputs, trainable_embedding_matrix)
        self.assertEqual(len(trainable_gradients), 1)
        self.assertFalse(trainable_gradients[0] is None)

    def testFailsOnFixedFeature(self):
        if False:
            while True:
                i = 10
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "annotate"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: 32 size: 1\n        }\n        ', component_spec)
        with tf.Graph().as_default():
            comp = bulk_component.BulkAnnotatorComponentBuilder(self.master, component_spec)
            with self.assertRaises(RuntimeError):
                comp.build_greedy_training(self.master_state, self.network_states)

    def testBulkFeatureIdExtractorOkWithOneFixedFeature(self):
        if False:
            i = 10
            return i + 15
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: -1 size: 1\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, component_spec)
        self.network_states[component_spec.name] = component.NetworkState()
        comp.build_greedy_training(self.master_state, self.network_states)
        self.network_states[component_spec.name] = component.NetworkState()
        comp.build_greedy_inference(self.master_state, self.network_states)

    def testBulkFeatureIdExtractorFailsOnLinkedFeature(self):
        if False:
            for i in range(10):
                print('nop')
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: -1 size: 1\n        }\n        linked_feature {\n          name: "linked" embedding_dim: -1 size: 1\n          source_translator: "identity"\n          source_component: "mock"\n        }\n        ', component_spec)
        with self.assertRaises(ValueError):
            unused_comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, component_spec)

    def testBulkFeatureIdExtractorOkWithMultipleFixedFeatures(self):
        if False:
            i = 10
            return i + 15
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed1" embedding_dim: -1 size: 1\n        }\n        fixed_feature {\n          name: "fixed2" embedding_dim: -1 size: 1\n        }\n        fixed_feature {\n          name: "fixed3" embedding_dim: -1 size: 1\n        }\n        ', component_spec)
        comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, component_spec)
        self.network_states[component_spec.name] = component.NetworkState()
        comp.build_greedy_training(self.master_state, self.network_states)
        self.network_states[component_spec.name] = component.NetworkState()
        comp.build_greedy_inference(self.master_state, self.network_states)

    def testBulkFeatureIdExtractorFailsOnEmbeddedFixedFeature(self):
        if False:
            for i in range(10):
                print('nop')
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: 2 size: 1\n        }\n        ', component_spec)
        with self.assertRaises(ValueError):
            unused_comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, component_spec)

    def testBulkFeatureIdExtractorExtractFocusWithOffset(self):
        if False:
            return 10
        path = os.path.join(tf.test.get_temp_dir(), 'label-map')
        with open(path, 'w') as label_map_file:
            label_map_file.write('0\n')
        master_spec = spec_pb2.MasterSpec()
        text_format.Parse('\n        component {\n          name: "test"\n          transition_system {\n            registered_name: "shift-only"\n          }\n          resource {\n            name: "label-map"\n            part {\n              file_pattern: "%s"\n              file_format: "text"\n            }\n          }\n          network_unit {\n            registered_name: "ExportFixedFeaturesNetwork"\n          }\n          backend {\n            registered_name: "SyntaxNetComponent"\n          }\n          fixed_feature {\n            name: "focus1" embedding_dim: -1 size: 1 fml: "input.focus"\n            predicate_map: "none"\n          }\n          fixed_feature {\n            name: "focus2" embedding_dim: -1 size: 1 fml: "input(1).focus"\n            predicate_map: "none"\n          }\n          fixed_feature {\n            name: "focus3" embedding_dim: -1 size: 1 fml: "input(2).focus"\n            predicate_map: "none"\n          }\n        }\n        ' % path, master_spec)
        corpus = _create_fake_corpus()
        corpus = tf.constant(corpus, shape=[len(corpus)])
        handle = dragnn_ops.get_session(container='test', master_spec=master_spec.SerializeToString(), grid_point='')
        handle = dragnn_ops.attach_data_reader(handle, corpus)
        handle = dragnn_ops.init_component_data(handle, beam_size=1, component='test')
        batch_size = dragnn_ops.batch_size(handle, component='test')
        master_state = component.MasterState(handle, batch_size)
        extractor = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, master_spec.component[0])
        network_state = component.NetworkState()
        self.network_states['test'] = network_state
        handle = extractor.build_greedy_inference(master_state, self.network_states)
        focus1 = network_state.activations['focus1'].bulk_tensor
        focus2 = network_state.activations['focus2'].bulk_tensor
        focus3 = network_state.activations['focus3'].bulk_tensor
        with self.test_session() as sess:
            (focus1, focus2, focus3) = sess.run([focus1, focus2, focus3])
            tf.logging.info('focus1=\n%s', focus1)
            tf.logging.info('focus2=\n%s', focus2)
            tf.logging.info('focus3=\n%s', focus3)
            self.assertAllEqual(focus1, [[0], [-1], [-1], [-1], [0], [1], [-1], [-1], [0], [1], [2], [-1], [0], [1], [2], [3]])
            self.assertAllEqual(focus2, [[-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [1], [2], [-1], [-1], [1], [2], [3], [-1]])
            self.assertAllEqual(focus3, [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [2], [-1], [-1], [-1], [2], [3], [-1], [-1]])

    def testBuildLossFailsOnNoExamples(self):
        if False:
            while True:
                i = 10
        logits = tf.constant([[0.5], [-0.5], [0.5], [-0.5]])
        gold = tf.constant([-1, -1, -1, -1])
        result = bulk_component.build_cross_entropy_loss(logits, gold)
        with self.test_session() as sess:
            with self.assertRaises(tf.errors.InvalidArgumentError):
                sess.run(result)

    def testPreCreateCalledBeforeCreate(self):
        if False:
            while True:
                i = 10
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        ', component_spec)

        class AssertPreCreateBeforeCreateNetwork(network_units.NetworkUnitInterface):
            """Mock that asserts that .create() is called before .pre_create()."""

            def __init__(self, comp, test_fixture):
                if False:
                    for i in range(10):
                        print('nop')
                super(AssertPreCreateBeforeCreateNetwork, self).__init__(comp)
                self._test_fixture = test_fixture
                self._pre_create_called = False

            def get_logits(self, network_tensors):
                if False:
                    while True:
                        i = 10
                return tf.zeros([2, 1], dtype=tf.float32)

            def pre_create(self, *unused_args):
                if False:
                    while True:
                        i = 10
                self._pre_create_called = True

            def create(self, *unused_args, **unuesd_kwargs):
                if False:
                    i = 10
                    return i + 15
                self._test_fixture.assertTrue(self._pre_create_called)
                return []
        builder = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        builder.network = AssertPreCreateBeforeCreateNetwork(builder, self)
        builder.build_greedy_training(component.MasterState(['foo', 'bar'], 2), self.network_states)
        self.setUp()
        builder = bulk_component.BulkFeatureExtractorComponentBuilder(self.master, component_spec)
        builder.network = AssertPreCreateBeforeCreateNetwork(builder, self)
        builder.build_greedy_inference(component.MasterState(['foo', 'bar'], 2), self.network_states)
        self.setUp()
        builder = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, component_spec)
        builder.network = AssertPreCreateBeforeCreateNetwork(builder, self)
        builder.build_greedy_training(component.MasterState(['foo', 'bar'], 2), self.network_states)
        self.setUp()
        builder = bulk_component.BulkFeatureIdExtractorComponentBuilder(self.master, component_spec)
        builder.network = AssertPreCreateBeforeCreateNetwork(builder, self)
        builder.build_greedy_inference(component.MasterState(['foo', 'bar'], 2), self.network_states)
        self.setUp()
        builder = bulk_component.BulkAnnotatorComponentBuilder(self.master, component_spec)
        builder.network = AssertPreCreateBeforeCreateNetwork(builder, self)
        builder.build_greedy_training(component.MasterState(['foo', 'bar'], 2), self.network_states)
        self.setUp()
        builder = bulk_component.BulkAnnotatorComponentBuilder(self.master, component_spec)
        builder.network = AssertPreCreateBeforeCreateNetwork(builder, self)
        builder.build_greedy_inference(component.MasterState(['foo', 'bar'], 2), self.network_states)
if __name__ == '__main__':
    googletest.main()