import unittest
from parameterized import parameterized
from transformers import AutoModel
from peft import IA3Config, LoraConfig, get_peft_model
from peft.tuners.tuners_utils import check_target_module_exists, inspect_matched_modules
TEST_CASES = [('', [], None, None, False), ('', ['foo'], None, None, False), ('foo', [], None, None, False), ('foo', ['foo'], None, None, True), ('foo', ['bar'], None, None, False), ('foo', ['foo', 'bar'], None, None, True), ('foo', 'foo', None, None, True), ('foo', '.*oo', None, None, True), ('foo', 'fo.*', None, None, True), ('foo', '.*bar.*', None, None, False), ('foobar', '.*oba.*', None, None, True), ('foo.bar.1.baz', ['baz'], [1], ['bar'], True), ('foo.bar.1.baz', ['baz'], [0], ['bar'], False), ('foo.bar.1.baz', ['baz'], [2], ['bar'], False), ('foo.bar.10.baz', ['baz'], [0], ['bar'], False), ('foo.bar.10.baz', ['baz'], [1], ['bar'], False), ('foo.bar.1.baz', ['baz'], [0, 1, 2], ['bar'], True), ('foo.bar.1.baz', ['baz', 'spam'], [1], ['bar'], True), ('foo.bar.1.baz', ['baz', 'spam'], [0, 1, 2], ['bar'], True), ('foo.bar.1.baz', ['baz'], [1], [], True), ('foo.bar.1.baz', ['baz'], [1], ['ar'], True), ('transformer.h.1.attn.attention.q_proj.foo', ['q_proj'], None, [], False), ('transformer.h.1.attn.attention.q_proj', [], None, [], False), ('transformer.h.1.attn.attention.q_proj', ['q_proj'], None, [], True), ('transformer.h.1.attn.attention.q_proj', ['q_proj', 'v_proj'], None, [], True), ('transformer.h.1.attn.attention.resid_dropout', ['q_proj', 'v_proj'], None, [], False), ('transformer.h.1.attn.attention.q_proj', ['q_proj'], [1], ['h'], True), ('transformer.h.1.attn.attention.q_proj', ['q_proj'], [0], ['h'], False), ('transformer.h.1.attn.attention.q_proj', ['q_proj'], [2], ['h'], False), ('transformer.h.1.attn.attention.q_proj', ['q_proj'], [0, 1, 2], ['h'], True), ('transformer.h.1.attn.attention.q_proj', ['q_proj', 'v_proj'], [0, 1, 2], ['h'], True), ('foo.bar.q_proj', ['q_proj'], None, [], True), ('foo.bar.1.baz', ['baz'], [1], ['foo'], False), ('foo.bar.1.baz', ['baz'], [1], ['baz'], False), ('bar.1.baz', ['baz'], [1], ['bar'], False), ('foo.bar.001.baz', ['baz'], [1], ['bar'], True), ('foo.bar.1.spam.2.baz', ['baz'], [1], ['bar'], True), ('foo.bar.2.spam.1.baz', ['baz'], [1], ['bar'], False), ('blocks.1.weight', ['weight'], [1], ['blocks'], False), ('blocks.1.bias', ['weight'], [1], ['blocks'], False), ('mlp.blocks.1.weight', ['weight'], [1], ['blocks'], True), ('mlp.blocks.1.bias', ['weight'], [1], ['blocks'], False)]

class PeftCustomKwargsTester(unittest.TestCase):
    """
    Test if the PeftModel is instantiated with correct behaviour for custom kwargs. This includes:
    - test if regex matching works correctly
    - test if adapters handle custom kwargs the right way e.g. IA3 for `feedforward_modules`

    """
    transformers_class = AutoModel

    @parameterized.expand(TEST_CASES)
    def test_regex_matching_valid(self, key, target_modules, layers_to_transform, layers_pattern, expected_result):
        if False:
            i = 10
            return i + 15
        model_id = 'peft-internal-testing/tiny-OPTForCausalLM-lora'
        config = LoraConfig(base_model_name_or_path=model_id, target_modules=target_modules, layers_pattern=layers_pattern, layers_to_transform=layers_to_transform)
        actual_result = bool(check_target_module_exists(config, key))
        self.assertEqual(actual_result, expected_result)

    def test_module_matching_lora(self):
        if False:
            return 10
        model_id = 'hf-internal-testing/tiny-random-BloomForCausalLM'
        model = self.transformers_class.from_pretrained(model_id)
        config = LoraConfig()
        peft_model = get_peft_model(model, config)
        output = inspect_matched_modules(peft_model)
        matched = output['matched']
        expected = ['h.0.self_attention.query_key_value', 'h.1.self_attention.query_key_value', 'h.2.self_attention.query_key_value', 'h.3.self_attention.query_key_value', 'h.4.self_attention.query_key_value']
        self.assertEqual(matched, expected)
        unmatched = output['unmatched']
        for key in expected:
            self.assertFalse(key in unmatched)

    def test_feedforward_matching_ia3(self):
        if False:
            print('Hello World!')
        model_id = 'hf-internal-testing/tiny-random-T5ForConditionalGeneration'
        model = self.transformers_class.from_pretrained(model_id)
        config_kwargs = {'target_modules': '.*encoder.*block.0.*(SelfAttention|EncDecAttention|DenseReluDense).(k|q|v|wo|wi)$', 'feedforward_modules': ['wo', 'wi']}
        config = IA3Config(base_model_name_or_path=model_id, **config_kwargs)
        peft_model = get_peft_model(model, config)
        output = inspect_matched_modules(peft_model)
        matched = output['matched']
        expected = ['encoder.block.0.layer.0.SelfAttention.q', 'encoder.block.0.layer.0.SelfAttention.k', 'encoder.block.0.layer.0.SelfAttention.v', 'encoder.block.0.layer.1.DenseReluDense.wi', 'encoder.block.0.layer.1.DenseReluDense.wo']
        expected_feedforward = ['encoder.block.0.layer.1.DenseReluDense.wi', 'encoder.block.0.layer.1.DenseReluDense.wo']
        self.assertEqual(matched, expected)
        module_dict = dict(model.named_modules())
        for key in matched:
            module = module_dict[key]
            if key in expected_feedforward:
                self.assertTrue(module.is_feedforward)
            else:
                self.assertFalse(module.is_feedforward)