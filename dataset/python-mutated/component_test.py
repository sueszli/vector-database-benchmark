"""Tests for component.py.
"""
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import component

class MockNetworkUnit(object):

    def get_layer_size(self, unused_layer_name):
        if False:
            i = 10
            return i + 15
        return 64

class MockComponent(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.name = 'mock'
        self.network = MockNetworkUnit()

class MockMaster(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.spec = spec_pb2.MasterSpec()
        self.hyperparams = spec_pb2.GridPoint()
        self.lookup_component = {'mock': MockComponent()}
        self.build_runtime_graph = False

class ComponentTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            return 10
        tf.reset_default_graph()
        self.master = MockMaster()
        self.master_state = component.MasterState(handle=tf.constant(['foo', 'bar']), current_batch_size=2)
        self.network_states = {'mock': component.NetworkState(), 'test': component.NetworkState()}

    def testSoftmaxCrossEntropyLoss(self):
        if False:
            i = 10
            return i + 15
        logits = tf.constant([[0.0, 2.0, -1.0], [-5.0, 1.0, -1.0], [3.0, 1.0, -2.0]])
        gold_labels = tf.constant([1, -1, 1])
        (cost, correct, total, logits, gold_labels) = component.build_softmax_cross_entropy_loss(logits, gold_labels)
        with self.test_session() as sess:
            (cost, correct, total, logits, gold_labels) = sess.run([cost, correct, total, logits, gold_labels])
            self.assertAlmostEqual(cost, 2.3027, 4)
            self.assertEqual(correct, 1)
            self.assertEqual(total, 2)
            self.assertAllEqual(logits, [[0.0, 2.0, -1.0], [3.0, 1.0, -2.0]])
            self.assertAllEqual(gold_labels, [1, 1])

    def testSigmoidCrossEntropyLoss(self):
        if False:
            i = 10
            return i + 15
        indices = tf.constant([0, 0, 1])
        gold_labels = tf.constant([0, 1, 2])
        probs = tf.constant([0.6, 0.7, 0.2])
        logits = tf.constant([[0.9, -0.3, 0.1], [-0.5, 0.4, 2.0]])
        (cost, correct, total, gold_labels) = component.build_sigmoid_cross_entropy_loss(logits, gold_labels, indices, probs)
        with self.test_session() as sess:
            (cost, correct, total, gold_labels) = sess.run([cost, correct, total, gold_labels])
            self.assertAlmostEqual(cost, 3.1924, 4)
            self.assertEqual(correct, 1)
            self.assertEqual(total, 3)
            self.assertAllEqual(gold_labels, [0, 1, 2])

    def testGraphConstruction(self):
        if False:
            i = 10
            return i + 15
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: 32 size: 1\n        }\n        component_builder {\n          registered_name: "component.DynamicComponentBuilder"\n        }\n        ', component_spec)
        comp = component.DynamicComponentBuilder(self.master, component_spec)
        comp.build_greedy_training(self.master_state, self.network_states)

    def testGraphConstructionWithSigmoidLoss(self):
        if False:
            for i in range(10):
                print('nop')
        component_spec = spec_pb2.ComponentSpec()
        text_format.Parse('\n        name: "test"\n        network_unit {\n          registered_name: "IdentityNetwork"\n        }\n        fixed_feature {\n          name: "fixed" embedding_dim: 32 size: 1\n        }\n        component_builder {\n          registered_name: "component.DynamicComponentBuilder"\n          parameters {\n            key: "loss_function"\n            value: "sigmoid_cross_entropy"\n          }\n        }\n        ', component_spec)
        comp = component.DynamicComponentBuilder(self.master, component_spec)
        comp.build_greedy_training(self.master_state, self.network_states)
        op_names = [op.name for op in tf.get_default_graph().get_operations()]
        self.assertTrue('train_test/compute_loss/sigmoid_cross_entropy_with_logits' in op_names)
if __name__ == '__main__':
    googletest.main()