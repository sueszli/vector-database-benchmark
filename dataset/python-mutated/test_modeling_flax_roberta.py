import unittest
import numpy as np
from transformers import RobertaConfig, is_flax_available
from transformers.testing_utils import require_flax, slow
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
if is_flax_available():
    from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaForCausalLM, FlaxRobertaForMaskedLM, FlaxRobertaForMultipleChoice, FlaxRobertaForQuestionAnswering, FlaxRobertaForSequenceClassification, FlaxRobertaForTokenClassification, FlaxRobertaModel

class FlaxRobertaModelTester(unittest.TestCase):

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_attention_mask=True, use_token_type_ids=True, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_choices=4):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
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
        self.num_choices = num_choices

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        config = RobertaConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, is_decoder=False, initializer_range=self.initializer_range)
        return (config, input_ids, token_type_ids, attention_mask)

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, attention_mask) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_decoder(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, attention_mask) = config_and_inputs
        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
        return (config, input_ids, token_type_ids, encoder_hidden_states, encoder_attention_mask)

@require_flax
class FlaxRobertaModelTest(FlaxModelTesterMixin, unittest.TestCase):
    test_head_masking = True
    all_model_classes = (FlaxRobertaModel, FlaxRobertaForCausalLM, FlaxRobertaForMaskedLM, FlaxRobertaForSequenceClassification, FlaxRobertaForTokenClassification, FlaxRobertaForMultipleChoice, FlaxRobertaForQuestionAnswering) if is_flax_available() else ()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = FlaxRobertaModelTester(self)

    @slow
    def test_model_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained('roberta-base', from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)