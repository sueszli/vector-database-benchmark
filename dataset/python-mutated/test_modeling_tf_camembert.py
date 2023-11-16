from __future__ import annotations
import unittest
from transformers import is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow
if is_tf_available():
    import numpy as np
    import tensorflow as tf
    from transformers import TFCamembertModel

@require_tf
@require_sentencepiece
@require_tokenizers
class TFCamembertModelIntegrationTest(unittest.TestCase):

    @slow
    def test_output_embeds_base_model(self):
        if False:
            print('Hello World!')
        model = TFCamembertModel.from_pretrained('jplu/tf-camembert-base')
        input_ids = tf.convert_to_tensor([[5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]], dtype=tf.int32)
        output = model(input_ids)['last_hidden_state']
        expected_shape = tf.TensorShape((1, 10, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = tf.convert_to_tensor([[[-0.0254, 0.0235, 0.1027], [0.0606, -0.1811, -0.0418], [-0.1561, -0.1127, 0.2687]]], dtype=tf.float32)
        self.assertTrue(np.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=0.0001))