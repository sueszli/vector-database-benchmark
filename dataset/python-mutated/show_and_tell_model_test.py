"""Tests for tensorflow_models.im2txt.show_and_tell_model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from im2txt import configuration
from im2txt import show_and_tell_model

class ShowAndTellModel(show_and_tell_model.ShowAndTellModel):
    """Subclass of ShowAndTellModel without the disk I/O."""

    def build_inputs(self):
        if False:
            return 10
        if self.mode == 'inference':
            return super(ShowAndTellModel, self).build_inputs()
        else:
            self.images = tf.random_uniform(shape=[self.config.batch_size, self.config.image_height, self.config.image_width, 3], minval=-1, maxval=1)
            self.input_seqs = tf.random_uniform([self.config.batch_size, 15], minval=0, maxval=self.config.vocab_size, dtype=tf.int64)
            self.target_seqs = tf.random_uniform([self.config.batch_size, 15], minval=0, maxval=self.config.vocab_size, dtype=tf.int64)
            self.input_mask = tf.ones_like(self.input_seqs)

class ShowAndTellModelTest(tf.test.TestCase):

    def setUp(self):
        if False:
            return 10
        super(ShowAndTellModelTest, self).setUp()
        self._model_config = configuration.ModelConfig()

    def _countModelParameters(self):
        if False:
            i = 10
            return i + 15
        'Counts the number of parameters in the model at top level scope.'
        counter = {}
        for v in tf.global_variables():
            name = v.op.name.split('/')[0]
            num_params = v.get_shape().num_elements()
            assert num_params
            counter[name] = counter.get(name, 0) + num_params
        return counter

    def _checkModelParameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Verifies the number of parameters in the model.'
        param_counts = self._countModelParameters()
        expected_param_counts = {'InceptionV3': 21802784, 'image_embedding': 1048576, 'seq_embedding': 6144000, 'lstm': 2099200, 'logits': 6156000, 'global_step': 1}
        self.assertDictEqual(expected_param_counts, param_counts)

    def _checkOutputs(self, expected_shapes, feed_dict=None):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        model = ShowAndTellModel(self._model_config, mode='train')
        model.build()
        self._checkModelParameters()
        expected_shapes = {model.images: (32, 299, 299, 3), model.input_seqs: (32, 15), model.target_seqs: (32, 15), model.input_mask: (32, 15), model.image_embeddings: (32, 512), model.seq_embeddings: (32, 15, 512), model.total_loss: (), model.target_cross_entropy_losses: (480,), model.target_cross_entropy_loss_weights: (480,)}
        self._checkOutputs(expected_shapes)

    def testBuildForEval(self):
        if False:
            print('Hello World!')
        model = ShowAndTellModel(self._model_config, mode='eval')
        model.build()
        self._checkModelParameters()
        expected_shapes = {model.images: (32, 299, 299, 3), model.input_seqs: (32, 15), model.target_seqs: (32, 15), model.input_mask: (32, 15), model.image_embeddings: (32, 512), model.seq_embeddings: (32, 15, 512), model.total_loss: (), model.target_cross_entropy_losses: (480,), model.target_cross_entropy_loss_weights: (480,)}
        self._checkOutputs(expected_shapes)

    def testBuildForInference(self):
        if False:
            return 10
        model = ShowAndTellModel(self._model_config, mode='inference')
        model.build()
        self._checkModelParameters()
        images_feed = np.random.rand(1, 299, 299, 3)
        feed_dict = {model.images: images_feed}
        expected_shapes = {model.image_embeddings: (1, 512), 'lstm/initial_state:0': (1, 1024)}
        self._checkOutputs(expected_shapes, feed_dict)
        input_feed = np.random.randint(0, 10, size=3)
        state_feed = np.random.rand(3, 1024)
        feed_dict = {'input_feed:0': input_feed, 'lstm/state_feed:0': state_feed}
        expected_shapes = {'lstm/state:0': (3, 1024), 'softmax:0': (3, 12000)}
        self._checkOutputs(expected_shapes, feed_dict)
if __name__ == '__main__':
    tf.test.main()