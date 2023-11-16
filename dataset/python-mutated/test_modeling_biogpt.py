""" Testing suite for the PyTorch BioGPT model. """
import math
import unittest
from transformers import BioGptConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import BioGptForCausalLM, BioGptForSequenceClassification, BioGptForTokenClassification, BioGptModel, BioGptTokenizer
    from transformers.models.biogpt.modeling_biogpt import BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST

class BioGptModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_input_mask=True, use_token_type_ids=False, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, scope=None):
        if False:
            print('Hello World!')
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        return BioGptConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, is_decoder=False, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        model = BioGptModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            print('Hello World!')
        model = BioGptForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_biogpt_model_attention_mask_past(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        if False:
            for i in range(10):
                print('nop')
        model = BioGptModel(config=config)
        model.to(torch_device)
        model.eval()
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0
        (output, past) = model(input_ids, attention_mask=attn_mask).to_tuple()
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)], dim=1)
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)['last_hidden_state']
        output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask)['last_hidden_state']
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=0.001))

    def create_and_check_biogpt_model_past_large_inputs(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        if False:
            print('Hello World!')
        model = BioGptModel(config=config).to(torch_device).eval()
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        (output, past_key_values) = outputs.to_tuple()
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)
        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)['last_hidden_state']
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)['last_hidden_state']
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()
        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=0.001))

    def create_and_check_forward_and_backwards(self, config, input_ids, input_mask, head_mask, token_type_ids, *args, gradient_checkpointing=False):
        if False:
            i = 10
            return i + 15
        model = BioGptForCausalLM(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def create_and_check_biogpt_weight_initialization(self, config, *args):
        if False:
            return 10
        model = BioGptModel(config)
        model_std = model.config.initializer_range / math.sqrt(2 * model.config.num_hidden_layers)
        for key in model.state_dict().keys():
            if 'c_proj' in key and 'weight' in key:
                self.parent.assertLessEqual(abs(torch.std(model.state_dict()[key]) - model_std), 0.001)
                self.parent.assertLessEqual(abs(torch.mean(model.state_dict()[key]) - 0.0), 0.01)

    def create_and_check_biogpt_for_token_classification(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        if False:
            while True:
                i = 10
        config.num_labels = self.num_labels
        model = BioGptForTokenClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_torch
class BioGptModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (BioGptModel, BioGptForCausalLM, BioGptForSequenceClassification, BioGptForTokenClassification) if is_torch_available() else ()
    all_generative_model_classes = (BioGptForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': BioGptModel, 'text-classification': BioGptForSequenceClassification, 'text-generation': BioGptForCausalLM, 'token-classification': BioGptForTokenClassification, 'zero-shot': BioGptForSequenceClassification} if is_torch_available() else {}
    test_pruning = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = BioGptModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BioGptConfig, hidden_size=37)

    def test_config(self):
        if False:
            return 10
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
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

    def test_biogpt_model_att_mask_past(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_biogpt_model_attention_mask_past(*config_and_inputs)

    def test_biogpt_gradient_checkpointing(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs, gradient_checkpointing=True)

    def test_biogpt_model_past_with_large_inputs(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_biogpt_model_past_large_inputs(*config_and_inputs)

    def test_biogpt_weight_initialization(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_biogpt_weight_initialization(*config_and_inputs)

    def test_biogpt_token_classification_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_biogpt_for_token_classification(*config_and_inputs)

    @slow
    def test_batch_generation(self):
        if False:
            for i in range(10):
                print('nop')
        model = BioGptForCausalLM.from_pretrained('microsoft/biogpt')
        model.to(torch_device)
        tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        sentences = ['Hello, my dog is a little', 'Today, I']
        inputs = tokenizer(sentences, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(torch_device)
        outputs = model.generate(input_ids=input_ids, attention_mask=inputs['attention_mask'].to(torch_device))
        inputs_non_padded = tokenizer(sentences[0], return_tensors='pt').input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded)
        num_paddings = inputs_non_padded.shape[-1] - inputs['attention_mask'][-1].long().sum().cpu().item()
        inputs_padded = tokenizer(sentences[1], return_tensors='pt').input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)
        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)
        expected_output_sentence = ['Hello, my dog is a little bit bigger than a little bit.', 'Today, I have a good idea of how to use the information']
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        if False:
            print('Hello World!')
        for model_name in BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BioGptModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_biogpt_sequence_classification_model(self):
        if False:
            print('Hello World!')
        (config, input_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict['input_ids']
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = BioGptForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_biogpt_sequence_classification_model_for_multi_label(self):
        if False:
            print('Hello World!')
        (config, input_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = 'multi_label_classification'
        input_ids = input_dict['input_ids']
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size).to(torch.float)
        model = BioGptForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

@require_torch
class BioGptModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_lm_head_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = BioGptForCausalLM.from_pretrained('microsoft/biogpt')
        input_ids = torch.tensor([[2, 4805, 9, 656, 21]])
        output = model(input_ids)[0]
        vocab_size = 42384
        expected_shape = torch.Size((1, 5, vocab_size))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[[-9.5236, -9.8918, 10.4557], [-11.0469, -9.6423, 8.1022], [-8.8664, -7.8826, 5.5325]]])
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=0.0001))

    @slow
    def test_biogpt_generation(self):
        if False:
            return 10
        tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
        model = BioGptForCausalLM.from_pretrained('microsoft/biogpt')
        model.to(torch_device)
        torch.manual_seed(0)
        tokenized = tokenizer('COVID-19 is', return_tensors='pt').to(torch_device)
        output_ids = model.generate(**tokenized, min_length=100, max_length=1024, num_beams=5, early_stopping=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        EXPECTED_OUTPUT_STR = 'COVID-19 is a global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of coronavirus disease 2019 (COVID-19), which has spread to more than 200 countries and territories, including the United States (US), Canada, Australia, New Zealand, the United Kingdom (UK), and the United States of America (USA), as of March 11, 2020, with more than 800,000 confirmed cases and more than 800,000 deaths.'
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)