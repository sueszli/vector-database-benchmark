import unittest
from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel
PEFT_MODELS_TO_TEST = [('peft-internal-testing/test-lora-subfolder', 'test')]

class PeftHubFeaturesTester(unittest.TestCase):

    def test_subfolder(self):
        if False:
            return 10
        '\n        Test if subfolder argument works as expected\n        '
        for (model_id, subfolder) in PEFT_MODELS_TO_TEST:
            config = PeftConfig.from_pretrained(model_id, subfolder=subfolder)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, model_id, subfolder=subfolder)
            self.assertTrue(isinstance(model, PeftModel))