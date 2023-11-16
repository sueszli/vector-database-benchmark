""" Testing suite for the PyTorch RemBERT model. """
import unittest
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import RemBertConfig, RemBertForCausalLM, RemBertForMaskedLM, RemBertForMultipleChoice, RemBertForQuestionAnswering, RemBertForSequenceClassification, RemBertForTokenClassification, RemBertModel
    from transformers.models.rembert.modeling_rembert import REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST

class RemBertModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_input_mask=True, use_token_type_ids=True, use_labels=True, vocab_size=99, hidden_size=32, input_embedding_size=18, output_embedding_size=43, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, scope=None):
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
        self.input_embedding_size = input_embedding_size
        self.output_embedding_size = output_embedding_size
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
        config = RemBertConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, input_embedding_size=self.input_embedding_size, output_embedding_size=self.output_embedding_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, is_decoder=False, initializer_range=self.initializer_range)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

    def prepare_config_and_inputs_for_decoder(self):
        if False:
            return 10
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = self.prepare_config_and_inputs()
        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask)

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        model = RemBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            for i in range(10):
                print('nop')
        config.add_cross_attention = True
        model = RemBertModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            return 10
        model = RemBertForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = RemBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            return 10
        config.is_decoder = True
        config.add_cross_attention = True
        model = RemBertForCausalLM(config=config)
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

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        model = RemBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.num_labels = self.num_labels
        model = RemBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        config.num_labels = self.num_labels
        model = RemBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        config.num_choices = self.num_choices
        model = RemBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(multiple_choice_inputs_ids, attention_mask=multiple_choice_input_mask, token_type_ids=multiple_choice_token_type_ids, labels=choice_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_torch
class RemBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (RemBertModel, RemBertForMaskedLM, RemBertForCausalLM, RemBertForMultipleChoice, RemBertForQuestionAnswering, RemBertForSequenceClassification, RemBertForTokenClassification) if is_torch_available() else ()
    all_generative_model_classes = (RemBertForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': RemBertModel, 'fill-mask': RemBertForMaskedLM, 'question-answering': RemBertForQuestionAnswering, 'text-classification': RemBertForSequenceClassification, 'text-generation': RemBertForCausalLM, 'token-classification': RemBertForTokenClassification, 'zero-shot': RemBertForSequenceClassification} if is_torch_available() else {}

    def setUp(self):
        if False:
            return 10
        self.model_tester = RemBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RemBertConfig, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ['absolute', 'relative_key', 'relative_key_query']:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_question_answering(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_model_as_decoder(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_as_decoder_with_default_input_mask(self):
        if False:
            print('Hello World!')
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask) = self.model_tester.prepare_config_and_inputs_for_decoder()
        input_mask = None
        self.model_tester.create_and_check_model_as_decoder(config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask)

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = RemBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

@require_torch
class RemBertModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_model(self):
        if False:
            return 10
        model = RemBertModel.from_pretrained('google/rembert')
        input_ids = torch.tensor([[312, 56498, 313, 2125, 313]])
        segment_ids = torch.tensor([[0, 0, 0, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, token_type_ids=segment_ids, output_hidden_states=True)
        hidden_size = 1152
        expected_shape = torch.Size((1, 5, hidden_size))
        self.assertEqual(output['last_hidden_state'].shape, expected_shape)
        expected_implementation = torch.tensor([[[0.0754, -0.2022, 0.1904], [-0.3354, -0.3692, -0.4791], [-0.2314, -0.6729, -0.0749], [-0.0396, -0.3105, -0.4234], [-0.1571, -0.0525, 0.5353]]])
        self.assertTrue(torch.allclose(output['last_hidden_state'][:, :, :3], expected_implementation, atol=0.0001))