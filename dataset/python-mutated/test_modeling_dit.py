import unittest
from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
if is_torch_available():
    import torch
    from transformers import AutoModelForImageClassification
if is_vision_available():
    from transformers import AutoImageProcessor

@require_torch
@require_vision
class DiTIntegrationTest(unittest.TestCase):

    @slow
    def test_for_image_classification(self):
        if False:
            print('Hello World!')
        image_processor = AutoImageProcessor.from_pretrained('microsoft/dit-base-finetuned-rvlcdip')
        model = AutoModelForImageClassification.from_pretrained('microsoft/dit-base-finetuned-rvlcdip')
        model.to(torch_device)
        from datasets import load_dataset
        dataset = load_dataset('nielsr/rvlcdip-demo')
        image = dataset['train'][0]['image'].convert('RGB')
        inputs = image_processor(image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        expected_shape = torch.Size((1, 16))
        self.assertEqual(logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.4158, -0.4092, -0.4347], device=torch_device, dtype=torch.float)
        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=0.0001))