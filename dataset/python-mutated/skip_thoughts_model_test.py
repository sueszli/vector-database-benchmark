"""Tests for tensorflow_models.skip_thoughts.skip_thoughts_model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from skip_thoughts import configuration
from skip_thoughts import skip_thoughts_model

class SkipThoughtsModel(skip_thoughts_model.SkipThoughtsModel):
    """Subclass of SkipThoughtsModel without the disk I/O."""

    def build_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mode == 'encode':
            return super(SkipThoughtsModel, self).build_inputs()
        else:
            self.encode_ids = tf.random_uniform([self.config.batch_size, 15], minval=0, maxval=self.config.vocab_size, dtype=tf.int64)
            self.decode_pre_ids = tf.random_uniform([self.config.batch_size, 15], minval=0, maxval=self.config.vocab_size, dtype=tf.int64)
            self.decode_post_ids = tf.random_uniform([self.config.batch_size, 15], minval=0, maxval=self.config.vocab_size, dtype=tf.int64)
            self.encode_mask = tf.ones_like(self.encode_ids)
            self.decode_pre_mask = tf.ones_like(self.decode_pre_ids)
            self.decode_post_mask = tf.ones_like(self.decode_post_ids)

class SkipThoughtsModelTest(tf.test.TestCase):

    def setUp(self):
        if False:
            return 10
        super(SkipThoughtsModelTest, self).setUp()
        self._model_config = configuration.model_config()

    def _countModelParameters(self):
        if False:
            return 10
        'Counts the number of parameters in the model at top level scope.'
        counter = {}
        for v in tf.global_variables():
            name = v.op.name.split('/')[0]
            num_params = v.get_shape().num_elements()
            if not num_params:
                self.fail('Could not infer num_elements from Variable %s' % v.op.name)
            counter[name] = counter.get(name, 0) + num_params
        return counter

    def _checkModelParameters(self):
        if False:
            while True:
                i = 10
        'Verifies the number of parameters in the model.'
        param_counts = self._countModelParameters()
        expected_param_counts = {'word_embedding': 12400000, 'encoder': 21772800, 'decoder_pre': 21772800, 'decoder_post': 21772800, 'logits': 48020000, 'global_step': 1}
        self.assertDictEqual(expected_param_counts, param_counts)

    def _checkOutputs(self, expected_shapes, feed_dict=None):
        if False:
            for i in range(10):
                print('nop')
        'Verifies that the model produces expected outputs.\n\n    Args:\n      expected_shapes: A dict mapping Tensor or Tensor name to expected output\n        shape.\n      feed_dict: Values of Tensors to feed into Session.run().\n    '
        fetches = expected_shapes.keys()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run(fetches, feed_dict)
        for (index, output) in enumerate(outputs):
            tensor = fetches[index]
            expected = expected_shapes[tensor]
            actual = output.shape
            if expected != actual:
                self.fail('Tensor %s has shape %s (expected %s).' % (tensor, actual, expected))

    def testBuildForTraining(self):
        if False:
            while True:
                i = 10
        model = SkipThoughtsModel(self._model_config, mode='train')
        model.build()
        self._checkModelParameters()
        expected_shapes = {model.encode_ids: (128, 15), model.decode_pre_ids: (128, 15), model.decode_post_ids: (128, 15), model.encode_mask: (128, 15), model.decode_pre_mask: (128, 15), model.decode_post_mask: (128, 15), model.encode_emb: (128, 15, 620), model.decode_pre_emb: (128, 15, 620), model.decode_post_emb: (128, 15, 620), model.thought_vectors: (128, 2400), model.target_cross_entropy_losses[0]: (1920,), model.target_cross_entropy_losses[1]: (1920,), model.target_cross_entropy_loss_weights[0]: (1920,), model.target_cross_entropy_loss_weights[1]: (1920,), model.total_loss: ()}
        self._checkOutputs(expected_shapes)

    def testBuildForEval(self):
        if False:
            return 10
        model = SkipThoughtsModel(self._model_config, mode='eval')
        model.build()
        self._checkModelParameters()
        expected_shapes = {model.encode_ids: (128, 15), model.decode_pre_ids: (128, 15), model.decode_post_ids: (128, 15), model.encode_mask: (128, 15), model.decode_pre_mask: (128, 15), model.decode_post_mask: (128, 15), model.encode_emb: (128, 15, 620), model.decode_pre_emb: (128, 15, 620), model.decode_post_emb: (128, 15, 620), model.thought_vectors: (128, 2400), model.target_cross_entropy_losses[0]: (1920,), model.target_cross_entropy_losses[1]: (1920,), model.target_cross_entropy_loss_weights[0]: (1920,), model.target_cross_entropy_loss_weights[1]: (1920,), model.total_loss: ()}
        self._checkOutputs(expected_shapes)

    def testBuildForEncode(self):
        if False:
            print('Hello World!')
        model = SkipThoughtsModel(self._model_config, mode='encode')
        model.build()
        encode_emb = np.random.rand(64, 15, 620)
        encode_mask = np.ones((64, 15), dtype=np.int64)
        feed_dict = {model.encode_emb: encode_emb, model.encode_mask: encode_mask}
        expected_shapes = {model.thought_vectors: (64, 2400)}
        self._checkOutputs(expected_shapes, feed_dict)
if __name__ == '__main__':
    tf.test.main()