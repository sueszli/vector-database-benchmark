from __future__ import annotations
import unittest
from transformers import RoFormerConfig, is_tf_available
from transformers.testing_utils import require_tf, slow
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import TFRoFormerForCausalLM, TFRoFormerForMaskedLM, TFRoFormerForMultipleChoice, TFRoFormerForQuestionAnswering, TFRoFormerForSequenceClassification, TFRoFormerForTokenClassification, TFRoFormerModel
    from transformers.models.roformer.modeling_tf_roformer import TFRoFormerSelfAttention, TFRoFormerSinusoidalPositionalEmbedding

class TFRoFormerModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_input_mask=True, use_token_type_ids=True, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, scope=None):
        if False:
            return 10
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = 'gelu'
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None

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
        config = RoFormerConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, initializer_range=self.initializer_range, return_dict=True)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = TFRoFormerModel(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        inputs = [input_ids, input_mask]
        result = model(inputs)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_lm_head(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        config.is_decoder = True
        model = TFRoFormerForCausalLM(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        prediction_scores = model(inputs)['logits']
        self.parent.assertListEqual(list(prediction_scores.numpy().shape), [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        model = TFRoFormerForMaskedLM(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.num_labels = self.num_labels
        model = TFRoFormerForSequenceClassification(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        config.num_choices = self.num_choices
        model = TFRoFormerForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        inputs = {'input_ids': multiple_choice_inputs_ids, 'attention_mask': multiple_choice_input_mask, 'token_type_ids': multiple_choice_token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        config.num_labels = self.num_labels
        model = TFRoFormerForTokenClassification(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        model = TFRoFormerForQuestionAnswering(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_tf
class TFRoFormerModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFRoFormerModel, TFRoFormerForCausalLM, TFRoFormerForMaskedLM, TFRoFormerForQuestionAnswering, TFRoFormerForSequenceClassification, TFRoFormerForTokenClassification, TFRoFormerForMultipleChoice) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFRoFormerModel, 'fill-mask': TFRoFormerForMaskedLM, 'question-answering': TFRoFormerForQuestionAnswering, 'text-classification': TFRoFormerForSequenceClassification, 'text-generation': TFRoFormerForCausalLM, 'token-classification': TFRoFormerForTokenClassification, 'zero-shot': TFRoFormerForSequenceClassification} if is_tf_available() else {}
    test_head_masking = False
    test_onnx = False

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            i = 10
            return i + 15
        if pipeline_test_casse_name == 'TextGenerationPipelineTests':
            return True
        return False

    def setUp(self):
        if False:
            return 10
        self.model_tester = TFRoFormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RoFormerConfig, hidden_size=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_causal_lm(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head(*config_and_inputs)

    def test_for_multiple_choice(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        model = TFRoFormerModel.from_pretrained('junnyu/roformer_chinese_base')
        self.assertIsNotNone(model)

@require_tf
class TFRoFormerModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_masked_lm(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFRoFormerForMaskedLM.from_pretrained('junnyu/roformer_chinese_base')
        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        vocab_size = 50000
        expected_shape = [1, 6, vocab_size]
        self.assertEqual(output.shape, expected_shape)
        print(output[:, :3, :3])
        expected_slice = tf.constant([[[-0.12053341, -1.0264901, 0.29221946], [-1.5133783, 0.197433, 0.15190607], [-5.0135403, -3.900256, -0.84038764]]])
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=0.0001)

@require_tf
class TFRoFormerSinusoidalPositionalEmbeddingTest(unittest.TestCase):
    tolerance = 0.0001

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        input_ids = tf.constant([[4, 10]])
        emb1 = TFRoFormerSinusoidalPositionalEmbedding(num_positions=6, embedding_dim=6)
        emb = emb1(input_ids.shape)
        desired_weights = tf.constant([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.8415, 0.0464, 0.0022, 0.5403, 0.9989, 1.0]])
        tf.debugging.assert_near(emb, desired_weights, atol=self.tolerance)

    def test_positional_emb_weights_against_roformer(self):
        if False:
            return 10
        desired_weights = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.8415, 0.8219, 0.802, 0.7819, 0.7617], [0.9093, 0.9364, 0.9581, 0.9749, 0.987]])
        emb1 = TFRoFormerSinusoidalPositionalEmbedding(num_positions=512, embedding_dim=512)
        emb1([2, 16, 512])
        weights = emb1.weight[:3, :5]
        tf.debugging.assert_near(weights, desired_weights, atol=self.tolerance)

@require_tf
class TFRoFormerSelfAttentionRotaryPositionEmbeddingTest(unittest.TestCase):
    tolerance = 0.0001

    def test_apply_rotary_position_embeddings(self):
        if False:
            i = 10
            return i + 15
        query_layer = tf.reshape(tf.range(2 * 12 * 16 * 64, dtype=tf.float32), shape=(2, 12, 16, 64)) / 100
        key_layer = -tf.reshape(tf.range(2 * 12 * 16 * 64, dtype=tf.float32), shape=(2, 12, 16, 64)) / 100
        embed_positions = TFRoFormerSinusoidalPositionalEmbedding(num_positions=32, embedding_dim=64)
        sinusoidal_pos = embed_positions([2, 16, 768])[None, None, :, :]
        (query_layer, key_layer) = TFRoFormerSelfAttention.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)
        desired_query_layer = tf.constant([[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], [-0.2012, 0.8897, 0.0263, 0.9401, 0.2074, 0.9463, 0.3481, 0.9343], [-1.7057, 0.6271, -1.2145, 1.3897, -0.6303, 1.7647, -0.1173, 1.8985], [-2.1731, -1.6397, -2.7358, 0.2854, -2.184, 1.7183, -1.3018, 2.4871], [0.2717, -3.6173, -2.9206, -2.1988, -3.6638, 0.3858, -2.9155, 2.298], [3.9859, -2.158, -0.7984, -4.4904, -4.1181, -2.0252, -4.4782, 1.1253]])
        desired_key_layer = tf.constant([[0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07], [0.2012, -0.8897, -0.0263, -0.9401, -0.2074, -0.9463, -0.3481, -0.9343], [1.7057, -0.6271, 1.2145, -1.3897, 0.6303, -1.7647, 0.1173, -1.8985], [2.1731, 1.6397, 2.7358, -0.2854, 2.184, -1.7183, 1.3018, -2.4871], [-0.2717, 3.6173, 2.9206, 2.1988, 3.6638, -0.3858, 2.9155, -2.298], [-3.9859, 2.158, 0.7984, 4.4904, 4.1181, 2.0252, 4.4782, -1.1253]])
        tf.debugging.assert_near(query_layer[0, 0, :6, :8], desired_query_layer, atol=self.tolerance)
        tf.debugging.assert_near(key_layer[0, 0, :6, :8], desired_key_layer, atol=self.tolerance)