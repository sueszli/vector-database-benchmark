from __future__ import annotations
import unittest
from transformers import is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import LongformerConfig, TFLongformerForMaskedLM, TFLongformerForMultipleChoice, TFLongformerForQuestionAnswering, TFLongformerForSequenceClassification, TFLongformerForTokenClassification, TFLongformerModel, TFLongformerSelfAttention
    from transformers.tf_utils import shape_list

class TFLongformerModelTester:

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
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
        self.attention_window = 4
        self.key_length = self.attention_window + 2

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
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
        config = LongformerConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, initializer_range=self.initializer_range, attention_window=self.attention_window)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

    def create_and_check_attention_mask_determinism(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        model = TFLongformerModel(config=config)
        attention_mask = tf.ones(input_ids.shape, dtype=tf.int64)
        output_with_mask = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]
        tf.debugging.assert_near(output_with_mask[0, 0, :5], output_without_mask[0, 0, :5], rtol=0.0001)

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        config.return_dict = True
        model = TFLongformerModel(config=config)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertListEqual(shape_list(result.last_hidden_state), [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertListEqual(shape_list(result.pooler_output), [self.batch_size, self.hidden_size])

    def create_and_check_model_with_global_attention_mask(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        config.return_dict = True
        model = TFLongformerModel(config=config)
        half_input_mask_length = shape_list(input_mask)[-1] // 2
        global_attention_mask = tf.concat([tf.zeros_like(input_mask)[:, :half_input_mask_length], tf.ones_like(input_mask)[:, half_input_mask_length:]], axis=-1)
        result = model(input_ids, attention_mask=input_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask)
        result = model(input_ids, global_attention_mask=global_attention_mask)
        self.parent.assertListEqual(shape_list(result.last_hidden_state), [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertListEqual(shape_list(result.pooler_output), [self.batch_size, self.hidden_size])

    def create_and_check_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        config.return_dict = True
        model = TFLongformerForMaskedLM(config=config)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertListEqual(shape_list(result.logits), [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.return_dict = True
        model = TFLongformerForQuestionAnswering(config=config)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        self.parent.assertListEqual(shape_list(result.start_logits), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(shape_list(result.end_logits), [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.num_labels = self.num_labels
        model = TFLongformerForSequenceClassification(config=config)
        output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels).logits
        self.parent.assertListEqual(shape_list(output), [self.batch_size, self.num_labels])

    def create_and_check_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        config.num_labels = self.num_labels
        model = TFLongformerForTokenClassification(config=config)
        output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels).logits
        self.parent.assertListEqual(shape_list(output), [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_multiple_choice(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.num_choices = self.num_choices
        model = TFLongformerForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        output = model(multiple_choice_inputs_ids, attention_mask=multiple_choice_input_mask, global_attention_mask=multiple_choice_input_mask, token_type_ids=multiple_choice_token_type_ids, labels=choice_labels).logits
        self.parent.assertListEqual(list(output.shape), [self.batch_size, self.num_choices])

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        global_attention_mask = tf.concat([tf.zeros_like(input_ids)[:, :-1], tf.ones_like(input_ids)[:, -1:]], axis=-1)
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask, 'global_attention_mask': global_attention_mask}
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_question_answering(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        input_ids = tf.where(input_ids == config.sep_token_id, 0, input_ids)
        input_ids = tf.concat([input_ids[:, :-3], tf.ones_like(input_ids)[:, -3:] * config.sep_token_id], axis=-1)
        input_mask = tf.ones_like(input_ids)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

@require_tf
class TFLongformerModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFLongformerModel, TFLongformerForMaskedLM, TFLongformerForQuestionAnswering, TFLongformerForSequenceClassification, TFLongformerForMultipleChoice, TFLongformerForTokenClassification) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFLongformerModel, 'fill-mask': TFLongformerForMaskedLM, 'question-answering': TFLongformerForQuestionAnswering, 'text-classification': TFLongformerForSequenceClassification, 'token-classification': TFLongformerForTokenClassification, 'zero-shot': TFLongformerForSequenceClassification} if is_tf_available() else {}
    test_head_masking = False
    test_onnx = False

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            while True:
                i = 10
        if pipeline_test_casse_name == 'QAPipelineTests' and tokenizer_name is not None and (not tokenizer_name.endswith('Fast')):
            return True
        return False

    def setUp(self):
        if False:
            return 10
        self.model_tester = TFLongformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongformerConfig, hidden_size=37)

    def test_config(self):
        if False:
            while True:
                i = 10
        self.config_tester.run_common_tests()

    def test_model_attention_mask_determinism(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_attention_mask_determinism(*config_and_inputs)

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_global_attention_mask(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_global_attention_mask(*config_and_inputs)

    def test_for_masked_lm(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_question_answering()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        if False:
            print('Hello World!')
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
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    @unittest.skip('Longformer keeps using potentially symbolic tensors in conditionals and breaks tracing.')
    def test_saved_model_creation(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip('Longformer keeps using potentially symbolic tensors in conditionals and breaks tracing.')
    def test_compile_tf_model(self):
        if False:
            return 10
        pass

@require_tf
@require_sentencepiece
@require_tokenizers
class TFLongformerModelIntegrationTest(unittest.TestCase):

    def _get_hidden_states(self):
        if False:
            while True:
                i = 10
        return tf.convert_to_tensor([[[0.498332758, 2.69175139, -0.00708081422, 1.04915401, -1.83476661, 0.767220476, 0.298580543, 0.0284803992], [-0.758357372, 0.420635998, -0.0404739919, 0.159924145, 2.05135748, -1.15997978, 0.537166397, 0.262873606], [-1.69438001, 0.41757466, -1.49196962, -1.76483717, -0.194566312, -1.71183858, 0.772903565, -1.11557056], [0.544028163, 0.205466114, -0.363045868, 0.241865062, 0.320348382, -0.905611176, -0.192690727, -1.19917547]]], dtype=tf.float32)

    def test_diagonalize(self):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self._get_hidden_states()
        hidden_states = tf.reshape(hidden_states, (1, 8, 4))
        chunked_hidden_states = TFLongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        window_overlap_size = shape_list(chunked_hidden_states)[2]
        self.assertTrue(window_overlap_size == 4)
        padded_hidden_states = TFLongformerSelfAttention._pad_and_diagonalize(chunked_hidden_states)
        self.assertTrue(shape_list(padded_hidden_states)[-1] == shape_list(chunked_hidden_states)[-1] + window_overlap_size - 1)
        tf.debugging.assert_near(padded_hidden_states[0, 0, 0, :4], chunked_hidden_states[0, 0, 0], rtol=0.001)
        tf.debugging.assert_near(padded_hidden_states[0, 0, 0, 4:], tf.zeros((3,), dtype=tf.float32), rtol=0.001)
        tf.debugging.assert_near(padded_hidden_states[0, 0, -1, 3:], chunked_hidden_states[0, 0, -1], rtol=0.001)
        tf.debugging.assert_near(padded_hidden_states[0, 0, -1, :3], tf.zeros((3,), dtype=tf.float32), rtol=0.001)

    def test_pad_and_transpose_last_two_dims(self):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self._get_hidden_states()
        self.assertEqual(shape_list(hidden_states), [1, 4, 8])
        paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]], dtype=tf.int64)
        hidden_states = TFLongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        padded_hidden_states = TFLongformerSelfAttention._pad_and_transpose_last_two_dims(hidden_states, paddings)
        self.assertTrue(shape_list(padded_hidden_states) == [1, 1, 8, 5])
        expected_added_dim = tf.zeros((5,), dtype=tf.float32)
        tf.debugging.assert_near(expected_added_dim, padded_hidden_states[0, 0, -1, :], rtol=1e-06)
        tf.debugging.assert_near(hidden_states[0, 0, -1, :], tf.reshape(padded_hidden_states, (1, -1))[0, 24:32], rtol=1e-06)

    def test_mask_invalid_locations(self):
        if False:
            i = 10
            return i + 15
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length, hidden_size))
        hidden_states = TFLongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        hid_states_1 = TFLongformerSelfAttention._mask_invalid_locations(hidden_states, 1)
        hid_states_2 = TFLongformerSelfAttention._mask_invalid_locations(hidden_states, 2)
        hid_states_3 = TFLongformerSelfAttention._mask_invalid_locations(hidden_states[:, :, :, :3], 2)
        hid_states_4 = TFLongformerSelfAttention._mask_invalid_locations(hidden_states[:, :, 2:, :], 2)
        self.assertTrue(tf.math.reduce_sum(tf.cast(tf.math.is_inf(hid_states_1), tf.int64)) == 8)
        self.assertTrue(tf.math.reduce_sum(tf.cast(tf.math.is_inf(hid_states_2), tf.int64)) == 24)
        self.assertTrue(tf.math.reduce_sum(tf.cast(tf.math.is_inf(hid_states_3), tf.int64)) == 24)
        self.assertTrue(tf.math.reduce_sum(tf.cast(tf.math.is_inf(hid_states_4), tf.int64)) == 12)

    def test_chunk(self):
        if False:
            return 10
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length, hidden_size))
        chunked_hidden_states = TFLongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        expected_slice_along_seq_length = tf.convert_to_tensor([0.4983, -0.7584, -1.6944], dtype=tf.float32)
        expected_slice_along_chunk = tf.convert_to_tensor([0.4983, -1.8348, -0.7584, 2.0514], dtype=tf.float32)
        self.assertTrue(shape_list(chunked_hidden_states) == [1, 3, 4, 4])
        tf.debugging.assert_near(chunked_hidden_states[0, :, 0, 0], expected_slice_along_seq_length, rtol=0.001, atol=0.0001)
        tf.debugging.assert_near(chunked_hidden_states[0, 0, :, 0], expected_slice_along_chunk, rtol=0.001, atol=0.0001)

    def test_layer_local_attn(self):
        if False:
            print('Hello World!')
        model = TFLongformerModel.from_pretrained('patrickvonplaten/longformer-random-tiny')
        layer = model.longformer.encoder.layer[0].attention.self_attention
        hidden_states = self._get_hidden_states()
        (batch_size, seq_length, hidden_size) = hidden_states.shape
        attention_mask = tf.zeros((batch_size, seq_length), dtype=tf.float32)
        is_index_global_attn = tf.math.greater(attention_mask, 1)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)
        attention_mask = tf.where(tf.range(4)[None, :, None, None] > 1, -10000.0, attention_mask[:, :, None, None])
        is_index_masked = tf.math.less(attention_mask[:, :, 0, 0], 0)
        layer_head_mask = None
        output_hidden_states = layer([hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn])[0]
        expected_slice = tf.convert_to_tensor([0.00188, 0.012196, -0.017051, -0.025571, -0.02996, 0.017297, -0.011521, 0.004848], dtype=tf.float32)
        self.assertEqual(output_hidden_states.shape, (1, 4, 8))
        tf.debugging.assert_near(output_hidden_states[0, 1], expected_slice, rtol=0.001, atol=0.0001)

    def test_layer_global_attn(self):
        if False:
            i = 10
            return i + 15
        model = TFLongformerModel.from_pretrained('patrickvonplaten/longformer-random-tiny')
        layer = model.longformer.encoder.layer[0].attention.self_attention
        hidden_states = self._get_hidden_states()
        hidden_states = tf.concat([self._get_hidden_states(), self._get_hidden_states() - 0.5], axis=0)
        (batch_size, seq_length, hidden_size) = hidden_states.shape
        attention_mask_1 = tf.zeros((1, 1, 1, seq_length), dtype=tf.float32)
        attention_mask_2 = tf.zeros((1, 1, 1, seq_length), dtype=tf.float32)
        attention_mask_1 = tf.where(tf.range(4)[None, :, None, None] > 1, 10000.0, attention_mask_1)
        attention_mask_1 = tf.where(tf.range(4)[None, :, None, None] > 2, -10000.0, attention_mask_1)
        attention_mask_2 = tf.where(tf.range(4)[None, :, None, None] > 0, 10000.0, attention_mask_2)
        attention_mask = tf.concat([attention_mask_1, attention_mask_2], axis=0)
        is_index_masked = tf.math.less(attention_mask[:, :, 0, 0], 0)
        is_index_global_attn = tf.math.greater(attention_mask[:, :, 0, 0], 0)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)
        layer_head_mask = None
        output_hidden_states = layer([hidden_states, -tf.math.abs(attention_mask), layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn])[0]
        self.assertEqual(output_hidden_states.shape, (2, 4, 8))
        expected_slice_0 = tf.convert_to_tensor([-0.06508, -0.039306, 0.030934, -0.03417, -0.00656, -0.01553, -0.02088, -0.04938], dtype=tf.float32)
        expected_slice_1 = tf.convert_to_tensor([-0.04055, -0.038399, 0.0396, -0.03735, -0.03415, 0.01357, 0.00145, -0.05709], dtype=tf.float32)
        tf.debugging.assert_near(output_hidden_states[0, 2], expected_slice_0, rtol=0.001, atol=0.0001)
        tf.debugging.assert_near(output_hidden_states[1, -2], expected_slice_1, rtol=0.001, atol=0.0001)

    def test_layer_attn_probs(self):
        if False:
            i = 10
            return i + 15
        model = TFLongformerModel.from_pretrained('patrickvonplaten/longformer-random-tiny')
        layer = model.longformer.encoder.layer[0].attention.self_attention
        hidden_states = tf.concat([self._get_hidden_states(), self._get_hidden_states() - 0.5], axis=0)
        (batch_size, seq_length, hidden_size) = hidden_states.shape
        attention_mask_1 = tf.zeros((1, 1, 1, seq_length), dtype=tf.float32)
        attention_mask_2 = tf.zeros((1, 1, 1, seq_length), dtype=tf.float32)
        attention_mask_1 = tf.where(tf.range(4)[None, :, None, None] > 1, 10000.0, attention_mask_1)
        attention_mask_1 = tf.where(tf.range(4)[None, :, None, None] > 2, -10000.0, attention_mask_1)
        attention_mask_2 = tf.where(tf.range(4)[None, :, None, None] > 0, 10000.0, attention_mask_2)
        attention_mask = tf.concat([attention_mask_1, attention_mask_2], axis=0)
        is_index_masked = tf.math.less(attention_mask[:, :, 0, 0], 0)
        is_index_global_attn = tf.math.greater(attention_mask[:, :, 0, 0], 0)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)
        layer_head_mask = None
        (output_hidden_states, local_attentions, global_attentions) = layer([hidden_states, -tf.math.abs(attention_mask), layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn])
        self.assertEqual(local_attentions.shape, (2, 4, 2, 8))
        self.assertEqual(global_attentions.shape, (2, 2, 3, 4))
        self.assertTrue((local_attentions[0, 2:4, :, :] == 0).numpy().tolist())
        self.assertTrue((local_attentions[1, 1:4, :, :] == 0).numpy().tolist())
        self.assertTrue((tf.math.abs(tf.math.reduce_sum(global_attentions[0, :, :2, :], axis=-1) - 1) < 1e-06).numpy().tolist())
        self.assertTrue((tf.math.abs(tf.math.reduce_sum(global_attentions[1, :, :1, :], axis=-1) - 1) < 1e-06).numpy().tolist())
        tf.debugging.assert_near(local_attentions[0, 0, 0, :], tf.convert_to_tensor([0.3328, 0.0, 0.0, 0.0, 0.0, 0.3355, 0.3318, 0.0], dtype=tf.float32), rtol=0.001, atol=0.0001)
        tf.debugging.assert_near(local_attentions[1, 0, 0, :], tf.convert_to_tensor([0.2492, 0.2502, 0.2502, 0.0, 0.0, 0.2505, 0.0, 0.0], dtype=tf.float32), rtol=0.001, atol=0.0001)
        self.assertTrue((tf.math.abs(tf.math.reduce_sum(global_attentions, axis=-1) - 1) < 1e-06).numpy().tolist())
        tf.debugging.assert_near(global_attentions[0, 0, 1, :], tf.convert_to_tensor([0.25, 0.25, 0.25, 0.25], dtype=tf.float32), rtol=0.001, atol=0.0001)
        tf.debugging.assert_near(global_attentions[1, 0, 0, :], tf.convert_to_tensor([0.2497, 0.25, 0.2499, 0.2504], dtype=tf.float32), rtol=0.001, atol=0.0001)

    @slow
    def test_inference_no_head(self):
        if False:
            i = 10
            return i + 15
        model = TFLongformerModel.from_pretrained('allenai/longformer-base-4096')
        input_ids = tf.convert_to_tensor([[0, 20920, 232, 328, 1437, 2]], dtype=tf.int64)
        attention_mask = tf.ones(shape_list(input_ids), dtype=tf.int64)
        output = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]
        expected_output_slice = tf.convert_to_tensor([0.0549, 0.1087, -0.1119, -0.0368, 0.025], dtype=tf.float32)
        tf.debugging.assert_near(output[0, 0, -5:], expected_output_slice, rtol=0.001, atol=0.0001)
        tf.debugging.assert_near(output_without_mask[0, 0, -5:], expected_output_slice, rtol=0.001, atol=0.0001)

    @slow
    def test_inference_no_head_long(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFLongformerModel.from_pretrained('allenai/longformer-base-4096')
        input_ids = tf.convert_to_tensor([[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=tf.int64)
        attention_mask = tf.ones(shape_list(input_ids), dtype=tf.int64)
        global_attention_mask = tf.zeros(shape_list(input_ids), dtype=tf.int64)
        global_attention_mask = tf.tensor_scatter_nd_update(global_attention_mask, tf.constant([[0, 1], [0, 4], [0, 21]], dtype=tf.int64), tf.constant([1, 1, 1], dtype=tf.int64))
        output = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)[0]
        expected_output_sum = tf.constant(74585.875)
        expected_output_mean = tf.constant(0.024267)
        tf.debugging.assert_near(tf.reduce_sum(output), expected_output_sum, rtol=0.0001, atol=0.0001)
        tf.debugging.assert_near(tf.reduce_mean(output), expected_output_mean, rtol=0.0001, atol=0.0001)

    @slow
    def test_inference_masked_lm_long(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFLongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
        input_ids = tf.convert_to_tensor([[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=tf.int64)
        output = model(input_ids, labels=input_ids)
        loss = output.loss
        prediction_scores = output.logits
        expected_loss = tf.constant(0.0073798)
        expected_prediction_scores_sum = tf.constant(-610476600.0)
        expected_prediction_scores_mean = tf.constant(-3.03477)
        tf.debugging.assert_near(tf.reduce_mean(loss), expected_loss, rtol=0.0001, atol=0.0001)
        tf.debugging.assert_near(tf.reduce_sum(prediction_scores), expected_prediction_scores_sum, rtol=0.0001, atol=0.0001)
        tf.debugging.assert_near(tf.reduce_mean(prediction_scores), expected_prediction_scores_mean, rtol=0.0001, atol=0.0001)

    @slow
    def test_inference_masked_lm(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFLongformerForMaskedLM.from_pretrained('lysandre/tiny-longformer-random')
        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        expected_shape = [1, 6, 10]
        self.assertEqual(output.shape, expected_shape)
        print(output[:, :3, :3])
        expected_slice = tf.constant([[[-0.04926379, 0.0367098, 0.02099686], [0.03940692, 0.01547744, -0.01448723], [0.03495252, -0.05900355, -0.01675752]]])
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=0.0001)