import unittest
from transformers import LongformerConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import LongformerForMaskedLM, LongformerForMultipleChoice, LongformerForQuestionAnswering, LongformerForSequenceClassification, LongformerForTokenClassification, LongformerModel, LongformerSelfAttention

class LongformerModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_input_mask=True, use_token_type_ids=True, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, scope=None, attention_window=4):
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
        self.attention_window = attention_window
        self.key_length = self.attention_window + 2

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
            print('Hello World!')
        return LongformerConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, initializer_range=self.initializer_range, attention_window=self.attention_window)

    def get_pipeline_config(self):
        if False:
            i = 10
            return i + 15
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_attention_mask_determinism(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        output_with_mask = model(input_ids, attention_mask=attention_mask)['last_hidden_state']
        output_without_mask = model(input_ids)['last_hidden_state']
        self.parent.assertTrue(torch.allclose(output_with_mask[0, 0, :5], output_without_mask[0, 0, :5], atol=0.0001))

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_global_attention_mask(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        global_attention_mask = input_mask.clone()
        global_attention_mask[:, input_mask.shape[-1] // 2] = 0
        global_attention_mask = global_attention_mask.to(torch_device)
        result = model(input_ids, attention_mask=input_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask)
        result = model(input_ids, global_attention_mask=global_attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        model = LongformerForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        model = LongformerForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, global_attention_mask=input_mask, token_type_ids=token_type_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        config.num_labels = self.num_labels
        model = LongformerForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.num_labels = self.num_labels
        model = LongformerForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        config.num_choices = self.num_choices
        model = LongformerForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(multiple_choice_inputs_ids, attention_mask=multiple_choice_input_mask, global_attention_mask=multiple_choice_input_mask, token_type_ids=multiple_choice_token_type_ids, labels=choice_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, -1] = 1
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask, 'global_attention_mask': global_attention_mask}
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_question_answering(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        input_ids[input_ids == config.sep_token_id] = torch.randint(0, config.vocab_size, (1,)).item()
        input_ids[:, -3:] = config.sep_token_id
        input_mask = torch.ones_like(input_ids)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

@require_torch
class LongformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_pruning = False
    test_torchscript = False
    all_model_classes = (LongformerModel, LongformerForMaskedLM, LongformerForSequenceClassification, LongformerForQuestionAnswering, LongformerForTokenClassification, LongformerForMultipleChoice) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': LongformerModel, 'fill-mask': LongformerForMaskedLM, 'question-answering': LongformerForQuestionAnswering, 'text-classification': LongformerForSequenceClassification, 'token-classification': LongformerForTokenClassification, 'zero-shot': LongformerForSequenceClassification} if is_torch_available() else {}
    model_split_percents = [0.6, 0.7, 0.9]

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            return 10
        if pipeline_test_casse_name == 'QAPipelineTests' and tokenizer_name is not None and (not tokenizer_name.endswith('Fast')):
            return True
        return False

    def setUp(self):
        if False:
            return 10
        self.model_tester = LongformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongformerConfig, hidden_size=37)

    def test_config(self):
        if False:
            while True:
                i = 10
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask_determinism(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_attention_mask_determinism(*config_and_inputs)

    def test_model_global_attention_mask(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_global_attention_mask(*config_and_inputs)

    def test_for_masked_lm(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_question_answering()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            for i in range(10):
                print('nop')
        return

@require_torch
@require_sentencepiece
@require_tokenizers
class LongformerModelIntegrationTest(unittest.TestCase):

    def _get_hidden_states(self):
        if False:
            while True:
                i = 10
        return torch.tensor([[[0.498332758, 2.69175139, -0.00708081422, 1.04915401, -1.83476661, 0.767220476, 0.298580543, 0.0284803992], [-0.758357372, 0.420635998, -0.0404739919, 0.159924145, 2.05135748, -1.15997978, 0.537166397, 0.262873606], [-1.69438001, 0.41757466, -1.49196962, -1.76483717, -0.194566312, -1.71183858, 0.772903565, -1.11557056], [0.544028163, 0.205466114, -0.363045868, 0.241865062, 0.320348382, -0.905611176, -0.192690727, -1.19917547]]], dtype=torch.float32, device=torch_device)

    def test_diagonalize(self):
        if False:
            i = 10
            return i + 15
        hidden_states = self._get_hidden_states()
        hidden_states = hidden_states.reshape((1, 8, 4))
        chunked_hidden_states = LongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        window_overlap_size = chunked_hidden_states.shape[2]
        self.assertTrue(window_overlap_size == 4)
        padded_hidden_states = LongformerSelfAttention._pad_and_diagonalize(chunked_hidden_states)
        self.assertTrue(padded_hidden_states.shape[-1] == chunked_hidden_states.shape[-1] + window_overlap_size - 1)
        self.assertTrue(torch.allclose(padded_hidden_states[0, 0, 0, :4], chunked_hidden_states[0, 0, 0], atol=0.001))
        self.assertTrue(torch.allclose(padded_hidden_states[0, 0, 0, 4:], torch.zeros((3,), device=torch_device, dtype=torch.float32), atol=0.001))
        self.assertTrue(torch.allclose(padded_hidden_states[0, 0, -1, 3:], chunked_hidden_states[0, 0, -1], atol=0.001))
        self.assertTrue(torch.allclose(padded_hidden_states[0, 0, -1, :3], torch.zeros((3,), device=torch_device, dtype=torch.float32), atol=0.001))

    def test_pad_and_transpose_last_two_dims(self):
        if False:
            print('Hello World!')
        hidden_states = self._get_hidden_states()
        self.assertEqual(hidden_states.shape, (1, 4, 8))
        padding = (0, 0, 0, 1)
        padded_hidden_states = LongformerSelfAttention._pad_and_transpose_last_two_dims(hidden_states, padding)
        self.assertEqual(padded_hidden_states.shape, (1, 8, 5))
        expected_added_dim = torch.zeros((5,), device=torch_device, dtype=torch.float32)
        self.assertTrue(torch.allclose(expected_added_dim, padded_hidden_states[0, -1, :], atol=1e-06))
        self.assertTrue(torch.allclose(hidden_states[0, -1, :], padded_hidden_states.view(1, -1)[0, 24:32], atol=1e-06))

    def test_chunk(self):
        if False:
            i = 10
            return i + 15
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = hidden_states.reshape((batch_size, seq_length, hidden_size))
        chunked_hidden_states = LongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        expected_slice_along_seq_length = torch.tensor([0.4983, -0.7584, -1.6944], device=torch_device, dtype=torch.float32)
        expected_slice_along_chunk = torch.tensor([0.4983, -1.8348, -0.7584, 2.0514], device=torch_device, dtype=torch.float32)
        self.assertTrue(torch.allclose(chunked_hidden_states[0, :, 0, 0], expected_slice_along_seq_length, atol=0.001))
        self.assertTrue(torch.allclose(chunked_hidden_states[0, 0, :, 0], expected_slice_along_chunk, atol=0.001))
        self.assertEqual(chunked_hidden_states.shape, (1, 3, 4, 4))

    def test_mask_invalid_locations(self):
        if False:
            while True:
                i = 10
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = hidden_states.reshape((batch_size, seq_length, hidden_size))
        chunked_hidden_states = LongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        hid_states_1 = chunked_hidden_states.clone()
        LongformerSelfAttention._mask_invalid_locations(hid_states_1, 1)
        self.assertTrue(torch.isinf(hid_states_1).sum().item() == 8)
        hid_states_2 = chunked_hidden_states.clone()
        LongformerSelfAttention._mask_invalid_locations(hid_states_2, 2)
        self.assertTrue(torch.isinf(hid_states_2).sum().item() == 24)
        hid_states_3 = chunked_hidden_states.clone()[:, :, :, :3]
        LongformerSelfAttention._mask_invalid_locations(hid_states_3, 2)
        self.assertTrue(torch.isinf(hid_states_3).sum().item() == 24)
        hid_states_4 = chunked_hidden_states.clone()[:, :, 2:, :]
        LongformerSelfAttention._mask_invalid_locations(hid_states_4, 2)
        self.assertTrue(torch.isinf(hid_states_4).sum().item() == 12)

    def test_layer_local_attn(self):
        if False:
            return 10
        model = LongformerModel.from_pretrained('patrickvonplaten/longformer-random-tiny')
        model.eval()
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = self._get_hidden_states()
        (batch_size, seq_length, hidden_size) = hidden_states.size()
        attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=torch_device)
        attention_mask[:, -2:] = -10000
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        output_hidden_states = layer(hidden_states, attention_mask=attention_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn)[0]
        self.assertEqual(output_hidden_states.shape, (1, 4, 8))
        self.assertTrue(torch.allclose(output_hidden_states[0, 1], torch.tensor([0.0019, 0.0122, -0.0171, -0.0256, -0.03, 0.0173, -0.0115, 0.0048], dtype=torch.float32, device=torch_device), atol=0.001))

    def test_layer_global_attn(self):
        if False:
            i = 10
            return i + 15
        model = LongformerModel.from_pretrained('patrickvonplaten/longformer-random-tiny')
        model.eval()
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = torch.cat([self._get_hidden_states(), self._get_hidden_states() - 0.5], dim=0)
        (batch_size, seq_length, hidden_size) = hidden_states.size()
        attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=torch_device)
        attention_mask[0, -2:] = 10000.0
        attention_mask[0, -1:] = -10000.0
        attention_mask[1, 1:] = 10000.0
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        output_hidden_states = layer(hidden_states, attention_mask=attention_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn)[0]
        self.assertEqual(output_hidden_states.shape, (2, 4, 8))
        self.assertTrue(torch.allclose(output_hidden_states[0, 2], torch.tensor([-0.0651, -0.0393, 0.0309, -0.0342, -0.0066, -0.0155, -0.0209, -0.0494], dtype=torch.float32, device=torch_device), atol=0.001))
        self.assertTrue(torch.allclose(output_hidden_states[1, -2], torch.tensor([-0.0405, -0.0384, 0.0396, -0.0374, -0.0341, 0.0136, 0.0014, -0.0571], dtype=torch.float32, device=torch_device), atol=0.001))

    def test_layer_attn_probs(self):
        if False:
            for i in range(10):
                print('nop')
        model = LongformerModel.from_pretrained('patrickvonplaten/longformer-random-tiny')
        model.eval()
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = torch.cat([self._get_hidden_states(), self._get_hidden_states() - 0.5], dim=0)
        (batch_size, seq_length, hidden_size) = hidden_states.size()
        attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=torch_device)
        attention_mask[0, -2:] = 10000.0
        attention_mask[0, -1:] = -10000.0
        attention_mask[1, 1:] = 10000.0
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        (output_hidden_states, local_attentions, global_attentions) = layer(hidden_states, attention_mask=attention_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=True)
        self.assertEqual(local_attentions.shape, (2, 4, 2, 8))
        self.assertEqual(global_attentions.shape, (2, 2, 3, 4))
        self.assertTrue(torch.all(local_attentions[0, 2:4, :, :] == 0))
        self.assertTrue(torch.all(local_attentions[1, 1:4, :, :] == 0))
        self.assertTrue(torch.all(torch.abs(global_attentions[0, :, :2, :].sum(dim=-1) - 1) < 1e-06))
        self.assertTrue(torch.all(torch.abs(global_attentions[1, :, :1, :].sum(dim=-1) - 1) < 1e-06))
        self.assertTrue(torch.allclose(local_attentions[0, 0, 0, :], torch.tensor([0.3328, 0.0, 0.0, 0.0, 0.0, 0.3355, 0.3318, 0.0], dtype=torch.float32, device=torch_device), atol=0.001))
        self.assertTrue(torch.allclose(local_attentions[1, 0, 0, :], torch.tensor([0.2492, 0.2502, 0.2502, 0.0, 0.0, 0.2505, 0.0, 0.0], dtype=torch.float32, device=torch_device), atol=0.001))
        self.assertTrue(torch.all(torch.abs(global_attentions.sum(dim=-1) - 1) < 1e-06))
        self.assertTrue(torch.allclose(global_attentions[0, 0, 1, :], torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32, device=torch_device), atol=0.001))
        self.assertTrue(torch.allclose(global_attentions[1, 0, 0, :], torch.tensor([0.2497, 0.25, 0.2499, 0.2504], dtype=torch.float32, device=torch_device), atol=0.001))

    @slow
    def test_inference_no_head(self):
        if False:
            for i in range(10):
                print('nop')
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        model.to(torch_device)
        input_ids = torch.tensor([[0, 20920, 232, 328, 1437, 2]], dtype=torch.long, device=torch_device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        output = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]
        expected_output_slice = torch.tensor([0.0549, 0.1087, -0.1119, -0.0368, 0.025], device=torch_device)
        self.assertTrue(torch.allclose(output[0, 0, -5:], expected_output_slice, atol=0.0001))
        self.assertTrue(torch.allclose(output_without_mask[0, 0, -5:], expected_output_slice, atol=0.0001))

    @slow
    def test_inference_no_head_long(self):
        if False:
            while True:
                i = 10
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        model.to(torch_device)
        input_ids = torch.tensor([[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[:, [1, 4, 21]] = 1
        output = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)[0]
        expected_output_sum = torch.tensor(74585.8594, device=torch_device)
        expected_output_mean = torch.tensor(0.0243, device=torch_device)
        self.assertTrue(torch.allclose(output.sum(), expected_output_sum, atol=0.0001))
        self.assertTrue(torch.allclose(output.mean(), expected_output_mean, atol=0.0001))

    @slow
    def test_inference_masked_lm_long(self):
        if False:
            while True:
                i = 10
        model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
        model.to(torch_device)
        input_ids = torch.tensor([[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device)
        input_ids = input_ids.to(torch_device)
        (loss, prediction_scores) = model(input_ids, labels=input_ids).to_tuple()
        expected_loss = torch.tensor(0.0074, device=torch_device)
        expected_prediction_scores_sum = torch.tensor(-610480000.0, device=torch_device)
        expected_prediction_scores_mean = torch.tensor(-3.0348, device=torch_device)
        self.assertTrue(torch.allclose(loss, expected_loss, atol=0.0001))
        self.assertTrue(torch.allclose(prediction_scores.sum(), expected_prediction_scores_sum, atol=0.0001))
        self.assertTrue(torch.allclose(prediction_scores.mean(), expected_prediction_scores_mean, atol=0.0001))