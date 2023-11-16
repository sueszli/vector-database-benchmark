import unittest
import torch
try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub

@unittest.skipIf(not huggingface_hub, 'Requires huggingface_hub install')
class TestHuggingFaceHub(unittest.TestCase):

    @torch.no_grad()
    def test_hf_fastspeech2(self):
        if False:
            return 10
        hf_model_id = 'facebook/fastspeech2-en-ljspeech'
        (models, cfg, task) = load_model_ensemble_and_task_from_hf_hub(hf_model_id)
        self.assertTrue(len(models) > 0)
if __name__ == '__main__':
    unittest.main()