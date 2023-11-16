"""Tests for google3.third_party.tensorflow_models.slim.nets.mobilenet.mobilenet_v3."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
import tensorflow as tf
from nets.mobilenet import mobilenet_v3

class MobilenetV3Test(absltest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MobilenetV3Test, self).setUp()
        tf.reset_default_graph()

    def testMobilenetV3Large(self):
        if False:
            print('Hello World!')
        (logits, endpoints) = mobilenet_v3.mobilenet(tf.placeholder(tf.float32, (1, 224, 224, 3)))
        self.assertEqual(endpoints['layer_19'].shape, [1, 1, 1, 1280])
        self.assertEqual(logits.shape, [1, 1001])

    def testMobilenetV3Small(self):
        if False:
            for i in range(10):
                print('nop')
        (_, endpoints) = mobilenet_v3.mobilenet(tf.placeholder(tf.float32, (1, 224, 224, 3)), conv_defs=mobilenet_v3.V3_SMALL)
        self.assertEqual(endpoints['layer_15'].shape, [1, 1, 1, 1024])

    def testMobilenetEdgeTpu(self):
        if False:
            print('Hello World!')
        (_, endpoints) = mobilenet_v3.edge_tpu(tf.placeholder(tf.float32, (1, 224, 224, 3)))
        self.assertIn('Inference mode is created by default', mobilenet_v3.edge_tpu.__doc__)
        self.assertEqual(endpoints['layer_24'].shape, [1, 7, 7, 1280])
        self.assertStartsWith(endpoints['layer_24'].name, 'MobilenetEdgeTPU')

    def testMobilenetEdgeTpuChangeScope(self):
        if False:
            for i in range(10):
                print('nop')
        (_, endpoints) = mobilenet_v3.edge_tpu(tf.placeholder(tf.float32, (1, 224, 224, 3)), scope='Scope')
        self.assertStartsWith(endpoints['layer_24'].name, 'Scope')

    def testMobilenetV3BaseOnly(self):
        if False:
            i = 10
            return i + 15
        (result, endpoints) = mobilenet_v3.mobilenet(tf.placeholder(tf.float32, (1, 224, 224, 3)), conv_defs=mobilenet_v3.V3_LARGE, base_only=True, final_endpoint='layer_17')
        self.assertEqual(endpoints['layer_17'].shape, [1, 7, 7, 960])
        self.assertEqual(result, endpoints['layer_17'])
if __name__ == '__main__':
    absltest.main()