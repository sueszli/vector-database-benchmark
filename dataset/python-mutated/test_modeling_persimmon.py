""" Testing suite for the PyTorch Persimmon model. """
import gc
import unittest
from parameterized import parameterized
from transformers import PersimmonConfig, is_torch_available, set_seed
from transformers.testing_utils import backend_empty_cache, require_torch, require_torch_accelerator, require_torch_fp16, slow, torch_device
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import AutoTokenizer, PersimmonForCausalLM, PersimmonForSequenceClassification, PersimmonModel

class PersimmonModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_input_mask=True, use_token_type_ids=False, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, pad_token_id=0, scope=None):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)
        config = self.get_config()
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

    def get_config(self):
        if False:
            return 10
        return PersimmonConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, is_decoder=False, initializer_range=self.initializer_range, pad_token_id=self.pad_token_id)

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = PersimmonModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            return 10
        config.add_cross_attention = True
        model = PersimmonModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        result = model(input_ids, attention_mask=input_mask, encoder_hidden_states=encoder_hidden_states)
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            return 10
        model = PersimmonForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            for i in range(10):
                print('nop')
        config.is_decoder = True
        config.add_cross_attention = True
        model = PersimmonForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        outputs = model(input_ids, attention_mask=input_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)
        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_hidden_states=True)['hidden_states'][0]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, output_hidden_states=True)['hidden_states'][0]
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()
        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=0.001))

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_torch
class PersimmonModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PersimmonModel, PersimmonForCausalLM, PersimmonForSequenceClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': PersimmonModel, 'text-classification': PersimmonForSequenceClassification} if is_torch_available() else {}
    all_generative_model_classes = (PersimmonForCausalLM,) if is_torch_available() else ()
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = PersimmonModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PersimmonConfig, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ['absolute', 'relative_key', 'relative_key_query']:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_persimmon_sequence_classification_model(self):
        if False:
            i = 10
            return i + 15
        (config, input_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict['input_ids']
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = PersimmonForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_persimmon_sequence_classification_model_for_single_label(self):
        if False:
            print('Hello World!')
        (config, input_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = 'single_label_classification'
        input_ids = input_dict['input_ids']
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = PersimmonForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_persimmon_sequence_classification_model_for_multi_label(self):
        if False:
            print('Hello World!')
        (config, input_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = 'multi_label_classification'
        input_ids = input_dict['input_ids']
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size).to(torch.float)
        model = PersimmonForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @unittest.skip('Persimmon buffers include complex numbers, which breaks this test')
    def test_save_load_fast_init_from_base(self):
        if False:
            return 10
        pass

    @parameterized.expand([('linear',), ('dynamic',)])
    def test_model_rope_scaling(self, scaling_type):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)
        set_seed(42)
        original_model = PersimmonModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state
        set_seed(42)
        config.rope_scaling = {'type': scaling_type, 'factor': 10.0}
        scaled_model = PersimmonModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state
        if scaling_type == 'dynamic':
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-05))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-05))
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-05))

@require_torch
class PersimmonIntegrationTest(unittest.TestCase):

    @slow
    def test_model_8b_chat_logits(self):
        if False:
            i = 10
            return i + 15
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = PersimmonForCausalLM.from_pretrained('adept/persimmon-8b-chat', load_in_8bit=True, device_map={'': 0}, torch_dtype=torch.float16)
        out = model(torch.tensor([input_ids], device=torch_device)).logits
        EXPECTED_MEAN = torch.tensor([[-11.4726, -11.1495, -11.2694, -11.2223, -10.9452, -11.0663, -11.0031, -11.1028]])
        torch.testing.assert_close(out.cpu().to(torch.float32).mean(-1), EXPECTED_MEAN, atol=0.0001, rtol=0.0001)
        EXPECTED_SLICE = torch.tensor([-16.9062, -16.9062, -16.9062, -16.9062, -16.8906, -16.9062, -16.9531, -16.9062, -16.9062, -16.9062, -16.9531, -16.9062, -16.9531, -16.9062, -16.9062, -16.9062, -16.9062, -16.9062, -16.9531, -16.9062, -16.9062, -16.9062, -16.9062, -16.9062, -16.9062, -16.9531, -16.9062, -16.9531, -16.9062, -16.9062], dtype=torch.float16)
        torch.testing.assert_close(out.cpu()[0, 0, :30], EXPECTED_SLICE, atol=1e-05, rtol=1e-05)
        backend_empty_cache(torch_device)
        del model
        gc.collect()

    @slow
    @require_torch_accelerator
    @require_torch_fp16
    def test_model_8b_chat_greedy_generation(self):
        if False:
            for i in range(10):
                print('nop')
        EXPECTED_TEXT_COMPLETION = 'human: Simply put, the theory of relativity states that?\n\nadept: The theory of relativity states that the laws of physics are the same for all observers, regardless of their relative motion.'
        prompt = 'human: Simply put, the theory of relativity states that?\n\nadept:'
        tokenizer = AutoTokenizer.from_pretrained('adept/persimmon-8b-chat', use_fast=False)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)
        model = PersimmonForCausalLM.from_pretrained('adept/persimmon-8b-chat', load_in_8bit=True, device_map={'': 0}, torch_dtype=torch.float16)
        generated_ids = model.generate(input_ids, max_new_tokens=64)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
        backend_empty_cache(torch_device)
        del model
        gc.collect()