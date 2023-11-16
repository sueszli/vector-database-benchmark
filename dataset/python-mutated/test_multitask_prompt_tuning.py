import importlib
import os
import tempfile
from unittest import TestCase
import torch
from torch.testing import assert_close
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.multitask_prompt_tuning import MultitaskPromptTuningConfig
from peft.utils.other import prepare_model_for_int8_training
from peft.utils.save_and_load import get_peft_model_state_dict
from tests.testing_common import PeftCommonTester

def is_llama_available() -> bool:
    if False:
        print('Hello World!')
    "Check if Llama is available in the transformers library (it's not in earlier versions)."
    try:
        return importlib.util.find_spec('transformers.models.llama.modeling_llama') is not None
    except ModuleNotFoundError:
        return False
if is_llama_available():
    from transformers import LlamaConfig, LlamaForCausalLM

class MultiTaskPromptTuningTester(TestCase, PeftCommonTester):
    """
    Tests for the AdaptionPrompt model.

    Some of these tests were adapted from `test_peft_model.py` (which has been refactored since), but since we haven't
    checked in the test checkpoints for Llama into `hf-internal-testing`, we separate them for now.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Check that llama is available in transformers package before running each test.'
        if not is_llama_available():
            self.skipTest('Llama not available in transformers. Skipping test.')

    @staticmethod
    def _create_test_llama_config():
        if False:
            for i in range(10):
                print('nop')
        'Create a test config for a small Llama model for testing.'
        return LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=8, num_hidden_layers=8, num_attention_heads=4, use_cache=False)

    @classmethod
    def _create_multitask_prompt_tuning_config(cls) -> MultitaskPromptTuningConfig:
        if False:
            return 10
        return MultitaskPromptTuningConfig(task_type='CAUSAL_LM', num_virtual_tokens=50, num_tasks=3, prompt_tuning_init_text='classify the following into either positive or negative, or entailment, neutral or contradiction:')

    def test_prepare_for_training(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)
        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)
        self.assertTrue(not dummy_output.requires_grad)

    def test_prepare_for_int8_training(self) -> None:
        if False:
            i = 10
            return i + 15
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = prepare_model_for_int8_training(model)
        model = model.to(self.torch_device)
        for param in model.parameters():
            self.assertTrue(not param.requires_grad)
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                if False:
                    print('Hello World!')
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)
        self.assertTrue(dummy_output.requires_grad)

    def test_save_pretrained(self) -> None:
        if False:
            return 10
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())
            self.assertEqual(len(list(state_dict.keys())), 3)
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_model.safetensors')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'pytorch_model.bin')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))

    def test_save_pretrained_regression(self) -> None:
        if False:
            while True:
                i = 10
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, safe_serialization=False)
            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())
            self.assertEqual(len(list(state_dict.keys())), 3)
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_model.bin')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'pytorch_model.bin')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))

    def test_generate(self) -> None:
        if False:
            return 10
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        task_ids = torch.LongTensor([1, 2]).to(self.torch_device)
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)
        with self.assertRaises(TypeError):
            _ = model.generate(input_ids, attention_mask=attention_mask)

    def test_use_cache(self) -> None:
        if False:
            return 10
        'Test that MultiTaskPromptTuning works when Llama config use_cache=True.'
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        task_ids = torch.LongTensor([1, 2]).to(self.torch_device)
        original = LlamaForCausalLM(self._create_test_llama_config())
        mpt = get_peft_model(original, self._create_multitask_prompt_tuning_config())
        mpt = mpt.to(self.torch_device)
        expected = mpt.generate(input_ids=input_ids, max_length=8, task_ids=task_ids)
        mpt.base_model.config.use_cache = True
        actual = mpt.generate(input_ids=input_ids, max_length=8, task_ids=task_ids)
        assert_close(expected, actual, rtol=0, atol=0)

    def test_bf16_inference(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that MultiTaskPromptTuning works when Llama using a half-precision model.'
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        task_ids = torch.tensor([1, 2]).to(self.torch_device)
        original = LlamaForCausalLM.from_pretrained('trl-internal-testing/tiny-random-LlamaForCausalLM', torch_dtype=torch.bfloat16)
        mpt = get_peft_model(original, self._create_multitask_prompt_tuning_config())
        mpt = mpt.to(self.torch_device)
        _ = mpt.generate(input_ids=input_ids, task_ids=task_ids)