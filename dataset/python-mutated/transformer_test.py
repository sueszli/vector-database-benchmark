"""Test Transformer model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from official.transformer.model import model_params
from official.transformer.v2 import transformer

class TransformerV2Test(tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.params = params = model_params.TINY_PARAMS
        params['batch_size'] = params['default_batch_size'] = 16
        params['use_synthetic_data'] = True
        params['hidden_size'] = 12
        params['num_hidden_layers'] = 2
        params['filter_size'] = 14
        params['num_heads'] = 2
        params['vocab_size'] = 41
        params['extra_decode_length'] = 2
        params['beam_size'] = 3
        params['dtype'] = tf.float32

    def test_create_model_train(self):
        if False:
            print('Hello World!')
        model = transformer.create_model(self.params, True)
        (inputs, outputs) = (model.inputs, model.outputs)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(inputs[1].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None, 41])
        self.assertEqual(outputs[0].dtype, tf.float32)

    def test_create_model_not_train(self):
        if False:
            print('Hello World!')
        model = transformer.create_model(self.params, False)
        (inputs, outputs) = (model.inputs, model.outputs)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None])
        self.assertEqual(outputs[0].dtype, tf.int32)
        self.assertEqual(outputs[1].shape.as_list(), [None])
        self.assertEqual(outputs[1].dtype, tf.float32)
if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()