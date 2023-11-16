import unittest
from transformers import AutoTokenizer, is_flax_available
from transformers.testing_utils import require_flax, require_sentencepiece, require_tokenizers, slow
if is_flax_available():
    import jax.numpy as jnp
    from transformers import FlaxXLMRobertaModel

@require_sentencepiece
@require_tokenizers
@require_flax
class FlaxXLMRobertaModelIntegrationTest(unittest.TestCase):

    @slow
    def test_flax_xlm_roberta_base(self):
        if False:
            while True:
                i = 10
        model = FlaxXLMRobertaModel.from_pretrained('xlm-roberta-base')
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        text = 'The dog is cute and lives in the garden house'
        input_ids = jnp.array([tokenizer.encode(text)])
        expected_output_shape = (1, 12, 768)
        expected_output_values_last_dim = jnp.array([[-0.0101, 0.1218, -0.0803, 0.0801, 0.1327, 0.0776, -0.1215, 0.2383, 0.3338, 0.3106, 0.03, 0.0252]])
        output = model(input_ids)['last_hidden_state']
        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(jnp.allclose(output[:, :, -1], expected_output_values_last_dim, atol=0.001))