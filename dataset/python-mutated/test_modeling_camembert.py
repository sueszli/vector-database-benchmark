import unittest
from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
if is_torch_available():
    import torch
    from transformers import CamembertModel

@require_torch
@require_sentencepiece
@require_tokenizers
class CamembertModelIntegrationTest(unittest.TestCase):

    @slow
    def test_output_embeds_base_model(self):
        if False:
            while True:
                i = 10
        model = CamembertModel.from_pretrained('camembert-base')
        model.to(torch_device)
        input_ids = torch.tensor([[5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]], device=torch_device, dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids)['last_hidden_state']
        expected_shape = torch.Size((1, 10, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[[-0.0254, 0.0235, 0.1027], [0.0606, -0.1811, -0.0418], [-0.1561, -0.1127, 0.2687]]], device=torch_device, dtype=torch.float)
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=0.0001))