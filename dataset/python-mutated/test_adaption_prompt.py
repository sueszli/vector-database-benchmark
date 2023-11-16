import importlib
import os
import tempfile
import unittest
from unittest import TestCase
import torch
from torch.testing import assert_close
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.adaption_prompt import AdaptionPromptConfig
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
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

class AdaptionPromptTester(TestCase, PeftCommonTester):
    """
    Tests for the AdaptionPrompt model.

    Some of these tests were adapted from `test_peft_model.py` (which has been refactored since), but since we haven't
    checked in the test checkpoints for Llama into `hf-internal-testing`, we separate them for now.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if not is_llama_available():
            self.skipTest('Llama not available in transformers. Skipping test.')

    @staticmethod
    def _create_test_llama_config():
        if False:
            print('Hello World!')
        'Create a test config for a small Llama model for testing.'
        return LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=8, num_hidden_layers=8, num_attention_heads=4, use_cache=False)

    def test_attributes(self) -> None:
        if False:
            return 10
        model = LlamaModel(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4)
        model = get_peft_model(model, config)
        self.assertTrue(hasattr(model, 'save_pretrained'))
        self.assertTrue(hasattr(model, 'from_pretrained'))
        self.assertTrue(hasattr(model, 'push_to_hub'))

    def test_prepare_for_training(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)
        self.assertTrue(not dummy_output.requires_grad)

    def test_prepare_for_int8_training(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = prepare_model_for_int8_training(model)
        model = model.to(self.torch_device)
        for param in model.parameters():
            self.assertTrue(not param.requires_grad)
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                if False:
                    i = 10
                    return i + 15
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)
        self.assertTrue(dummy_output.requires_grad)

    def test_save_pretrained_regression(self) -> None:
        if False:
            print('Hello World!')
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, safe_serialization=False)
            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())
            self.assertEqual(len(list(state_dict.keys())), 4)
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_model.bin')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'model.safetensors')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))

    def test_save_pretrained(self) -> None:
        if False:
            while True:
                i = 10
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())
            self.assertEqual(len(list(state_dict.keys())), 4)
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_model.safetensors')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'model.safetensors')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))

    def test_save_pretrained_selected_adapters(self) -> None:
        if False:
            i = 10
            return i + 15
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        new_adapter_config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM')
        model.add_adapter('new_adapter', new_adapter_config)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            model_from_pretrained.load_adapter(tmp_dirname, 'new_adapter')
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())
            self.assertEqual(len(list(state_dict.keys())), 4)
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_model.safetensors')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'model.safetensors')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))

    def test_generate(self) -> None:
        if False:
            i = 10
            return i + 15
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        with self.assertRaises(TypeError):
            _ = model.generate(input_ids, attention_mask=attention_mask)

    def test_sequence_adapter_ops(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test sequence of adapter operations.'
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        target_ids = torch.LongTensor([[0, 0, 0], [0, 0, 0]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        original = LlamaForCausalLM(self._create_test_llama_config())
        original = original.to(self.torch_device)
        original_before = original(input_ids=input_ids, attention_mask=attention_mask)
        adapted = get_peft_model(original, AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM'))
        adapted = adapted.to(self.torch_device)
        default_before = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        assert_close(original_before.logits, default_before.logits, rtol=0, atol=0)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=1)
        optimizer.zero_grad()
        default_before.loss.backward()
        optimizer.step()
        default_after = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        self.assertFalse(torch.allclose(default_before.logits, default_after.logits))
        with adapted.disable_adapter():
            default_disabled = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            assert_close(original_before.logits, default_disabled.logits, rtol=0, atol=0)
        adapted.add_adapter('adapter 1', AdaptionPromptConfig(adapter_layers=3, adapter_len=8, task_type='CAUSAL_LM'))
        adapter_1_before = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        assert_close(original_before.logits, adapter_1_before.logits, rtol=0, atol=0)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=1)
        optimizer.zero_grad()
        adapter_1_before.loss.backward()
        optimizer.step()
        adapter_1_after = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        self.assertFalse(torch.allclose(adapter_1_before.logits, adapter_1_after.logits))
        self.assertFalse(torch.allclose(original_before.logits, adapter_1_after.logits))
        self.assertFalse(torch.allclose(default_after.logits, adapter_1_after.logits))
        with adapted.disable_adapter():
            adapter_1_disabled = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            assert_close(original_before.logits, adapter_1_disabled.logits, rtol=0, atol=0)
        adapted.set_adapter('default')
        default_after_set = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        assert_close(default_after.logits, default_after_set.logits, rtol=0, atol=0)
        self.assertFalse(torch.allclose(original_before.logits, default_after_set.logits))
        self.assertFalse(torch.allclose(adapter_1_after.logits, default_after_set.logits))

    def test_add_and_set_while_disabled(self):
        if False:
            i = 10
            return i + 15
        'Test that adding and setting adapters while disabled works as intended.'
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        target_ids = torch.LongTensor([[0, 0, 0], [0, 0, 0]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        original = LlamaForCausalLM(self._create_test_llama_config())
        original = original.to(self.torch_device)
        original_before = original(input_ids=input_ids, attention_mask=attention_mask)
        adapted = get_peft_model(original, AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM'))
        adapted = adapted.to(self.torch_device)
        with adapted.disable_adapter():
            adapted.add_adapter('adapter 1', AdaptionPromptConfig(adapter_layers=3, adapter_len=8, task_type='CAUSAL_LM'))
        adapter_1_before = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        assert_close(original_before.logits, adapter_1_before.logits, rtol=0, atol=0)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=1)
        optimizer.zero_grad()
        adapter_1_before.loss.backward()
        optimizer.step()
        adapter_1_after = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        self.assertFalse(torch.allclose(original_before.logits, adapter_1_after.logits))
        adapted.set_adapter('default')
        with adapted.disable_adapter():
            adapted.set_adapter('adapter 1')
        adapter_1_after_set = adapted(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        assert_close(adapter_1_after.logits, adapter_1_after_set.logits, rtol=0, atol=0)

    def test_use_cache(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that AdaptionPrompt works when Llama config use_cache=True.'
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        original = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=8, num_hidden_layers=8, num_attention_heads=4, use_cache=False))
        adapted = get_peft_model(original, AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM'))
        adapted = adapted.to(self.torch_device)
        expected = adapted.generate(input_ids=input_ids, max_length=8)
        adapted.base_model.config.use_cache = True
        actual = adapted.generate(input_ids=input_ids, max_length=8)
        assert_close(expected, actual, rtol=0, atol=0)

    def test_bf16_inference(self) -> None:
        if False:
            print('Hello World!')
        'Test that AdaptionPrompt works when Llama using a half-precision model.'
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        original = LlamaForCausalLM.from_pretrained('trl-internal-testing/tiny-random-LlamaForCausalLM', torch_dtype=torch.bfloat16)
        adapted = get_peft_model(original, AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type='CAUSAL_LM'))
        adapted = adapted.to(self.torch_device)
        _ = adapted.generate(input_ids=input_ids)

    @unittest.expectedFailure
    def test_disable_adapter(self):
        if False:
            i = 10
            return i + 15
        llama_config = self._create_test_llama_config()
        model = LlamaForCausalLM(llama_config).to(self.torch_device)
        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        output_before = model(dummy_input).logits
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4, task_type='CAUSAL_LM')
        model = get_peft_model(model, config).to(self.torch_device)
        output_peft = model(dummy_input).logits
        self.assertFalse(torch.allclose(output_before, output_peft))
        with model.disable_adapter():
            output_peft_disabled = model(dummy_input).logits
        self.assertTrue(torch.allclose(output_before, output_peft_disabled))