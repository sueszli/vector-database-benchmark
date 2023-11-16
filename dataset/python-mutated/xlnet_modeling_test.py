from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
import tensorflow as tf
from official.nlp.xlnet import xlnet_modeling

class PositionalEmbeddingLayerTest(tf.test.TestCase):

    def test_positional_embedding(self):
        if False:
            return 10
        'A low-dimensional example is tested.\n\n     With len(pos_seq)=2 and d_model=4:\n\n       pos_seq  = [[1.], [0.]]\n       inv_freq = [1., 0.01]\n       pos_seq x inv_freq = [[1, 0.01], [0., 0.]]\n       pos_emb = [[sin(1.), sin(0.01), cos(1.), cos(0.01)],\n                  [sin(0.), sin(0.), cos(0.), cos(0.)]]\n               = [[0.84147096, 0.00999983, 0.54030228, 0.99994999],\n                 [0., 0., 1., 1.]]\n    '
        target = np.array([[[0.84147096, 0.00999983, 0.54030228, 0.99994999]], [[0.0, 0.0, 1.0, 1.0]]])
        d_model = 4
        pos_seq = tf.range(1, -1, -1.0)
        pos_emb_layer = xlnet_modeling.PositionalEmbedding(d_model)
        pos_emb = pos_emb_layer(pos_seq=pos_seq, batch_size=None).numpy().astype(float)
        logging.info(pos_emb)
        self.assertAllClose(pos_emb, target)
if __name__ == '__main__':
    assert tf.version.VERSION.startswith('2.')
    tf.test.main()