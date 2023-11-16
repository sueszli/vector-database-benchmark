import unittest
from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow
if is_torch_available():
    import torch
    from transformers import XLMRobertaModel

@require_sentencepiece
@require_tokenizers
@require_torch
class XLMRobertaModelIntegrationTest(unittest.TestCase):

    @slow
    def test_xlm_roberta_base(self):
        if False:
            for i in range(10):
                print('nop')
        model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        expected_output_shape = torch.Size((1, 12, 768))
        expected_output_values_last_dim = torch.tensor([[-0.0101, 0.1218, -0.0803, 0.0801, 0.1327, 0.0776, -0.1215, 0.2383, 0.3338, 0.3106, 0.03, 0.0252]])
        with torch.no_grad():
            output = model(input_ids)['last_hidden_state'].detach()
        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(torch.allclose(output[:, :, -1], expected_output_values_last_dim, atol=0.001))

    @slow
    def test_xlm_roberta_large(self):
        if False:
            while True:
                i = 10
        model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        expected_output_shape = torch.Size((1, 12, 1024))
        expected_output_values_last_dim = torch.tensor([[-0.0699, -0.0318, 0.0705, -0.1241, 0.0999, -0.052, 0.1004, -0.1838, -0.4704, 0.1437, 0.0821, 0.0126]])
        with torch.no_grad():
            output = model(input_ids)['last_hidden_state'].detach()
        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(torch.allclose(output[:, :, -1], expected_output_values_last_dim, atol=0.001))