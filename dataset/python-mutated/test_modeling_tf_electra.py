from __future__ import annotations
import unittest
from transformers import ElectraConfig, is_tf_available
from transformers.testing_utils import require_tf, slow
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers.models.electra.modeling_tf_electra import TFElectraForMaskedLM, TFElectraForMultipleChoice, TFElectraForPreTraining, TFElectraForQuestionAnswering, TFElectraForSequenceClassification, TFElectraForTokenClassification, TFElectraModel

class TFElectraModelTester:

    def __init__(self, parent):
        if False:
            print('Hello World!')
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
        self.embedding_size = 128

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
        config = ElectraConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, initializer_range=self.initializer_range)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels)

    def prepare_config_and_inputs_for_decoder(self):
        if False:
            print('Hello World!')
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = self.prepare_config_and_inputs()
        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
        return (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask)

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = TFElectraModel(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        inputs = [input_ids, input_mask]
        result = model(inputs)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_causal_lm_base_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        config.is_decoder = True
        model = TFElectraModel(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        inputs = [input_ids, input_mask]
        result = model(inputs)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            while True:
                i = 10
        config.add_cross_attention = True
        model = TFElectraModel(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids, 'encoder_hidden_states': encoder_hidden_states, 'encoder_attention_mask': encoder_attention_mask}
        result = model(inputs)
        inputs = [input_ids, input_mask]
        result = model(inputs, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_causal_lm_base_model_past(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        config.is_decoder = True
        model = TFElectraModel(config=config)
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)
        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)
        past_key_values = outputs.past_key_values
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        output_from_no_past = model(next_input_ids, output_hidden_states=True).hidden_states[0]
        output_from_past = model(next_tokens, past_key_values=past_key_values, output_hidden_states=True).hidden_states[0]
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-06)

    def create_and_check_causal_lm_base_model_past_with_attn_mask(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.is_decoder = True
        model = TFElectraModel(config=config)
        half_seq_length = self.seq_length // 2
        attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
        attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
        attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)
        outputs = model(input_ids, attention_mask=attn_mask, use_cache=True)
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        past_key_values = outputs.past_key_values
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).numpy() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, self.seq_length), config.vocab_size)
        vector_condition = tf.range(self.seq_length) == self.seq_length - random_seq_idx_to_change
        condition = tf.transpose(tf.broadcast_to(tf.expand_dims(vector_condition, -1), (self.seq_length, self.batch_size)))
        input_ids = tf.where(condition, random_other_next_tokens, input_ids)
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        attn_mask = tf.concat([attn_mask, tf.ones((attn_mask.shape[0], 1), dtype=tf.int32)], axis=1)
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[0]
        output_from_past = model(next_tokens, past_key_values=past_key_values, attention_mask=attn_mask, output_hidden_states=True).hidden_states[0]
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-06)

    def create_and_check_causal_lm_base_model_past_large_inputs(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        config.is_decoder = True
        model = TFElectraModel(config=config)
        input_ids = input_ids[:1, :]
        input_mask = input_mask[:1, :]
        self.batch_size = 1
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_attn_mask], axis=-1)
        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True).hidden_states[0]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values, output_hidden_states=True).hidden_states[0]
        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=0.001)

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels, encoder_hidden_states, encoder_attention_mask):
        if False:
            for i in range(10):
                print('nop')
        config.add_cross_attention = True
        model = TFElectraModel(config=config)
        input_ids = input_ids[:1, :]
        input_mask = input_mask[:1, :]
        encoder_hidden_states = encoder_hidden_states[:1, :, :]
        encoder_attention_mask = encoder_attention_mask[:1, :]
        self.batch_size = 1
        outputs = model(input_ids, attention_mask=input_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_attn_mask], axis=-1)
        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_hidden_states=True).hidden_states[0]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, output_hidden_states=True).hidden_states[0]
        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=0.001)

    def create_and_check_for_masked_lm(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        model = TFElectraForMaskedLM(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_pretraining(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        model = TFElectraForPreTraining(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        config.num_labels = self.num_labels
        model = TFElectraForSequenceClassification(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        config.num_choices = self.num_choices
        model = TFElectraForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        inputs = {'input_ids': multiple_choice_inputs_ids, 'attention_mask': multiple_choice_input_mask, 'token_type_ids': multiple_choice_token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = TFElectraForQuestionAnswering(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_token_classification(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        config.num_labels = self.num_labels
        model = TFElectraForTokenClassification(config=config)
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_tf
class TFElectraModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFElectraModel, TFElectraForMaskedLM, TFElectraForPreTraining, TFElectraForTokenClassification, TFElectraForMultipleChoice, TFElectraForSequenceClassification, TFElectraForQuestionAnswering) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFElectraModel, 'fill-mask': TFElectraForMaskedLM, 'question-answering': TFElectraForQuestionAnswering, 'text-classification': TFElectraForSequenceClassification, 'token-classification': TFElectraForTokenClassification, 'zero-shot': TFElectraForSequenceClassification} if is_tf_available() else {}
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = TFElectraModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ElectraConfig, hidden_size=37)

    def test_config(self):
        if False:
            while True:
                i = 10
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
        'Test the base model'
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_causal_lm_base_model(self):
        if False:
            return 10
        'Test the base model of the causal LM model\n\n        is_deocder=True, no cross_attention, no encoder outputs\n        '
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_base_model(*config_and_inputs)

    def test_model_as_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the base model as a decoder (of an encoder-decoder architecture)\n\n        is_deocder=True + cross_attention + pass encoder outputs\n        '
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_causal_lm_base_model_past(self):
        if False:
            return 10
        'Test causal LM base model with `past_key_values`'
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_base_model_past(*config_and_inputs)

    def test_causal_lm_base_model_past_with_attn_mask(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the causal LM base model with `past_key_values` and `attention_mask`'
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_base_model_past_with_attn_mask(*config_and_inputs)

    def test_causal_lm_base_model_past_with_large_inputs(self):
        if False:
            return 10
        'Test the causal LM base model with `past_key_values` and a longer decoder sequence length'
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_base_model_past_large_inputs(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        if False:
            i = 10
            return i + 15
        'Similar to `test_causal_lm_base_model_past_with_large_inputs` but with cross-attention'
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_masked_lm(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_pretraining(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_token_classification(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in ['google/electra-small-discriminator']:
            model = TFElectraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

@require_tf
class TFElectraModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_masked_lm(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFElectraForPreTraining.from_pretrained('lysandre/tiny-electra-random')
        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        expected_shape = [1, 6]
        self.assertEqual(output.shape, expected_shape)
        print(output[:, :3])
        expected_slice = tf.constant([[-0.24651965, 0.8835437, 1.823782]])
        tf.debugging.assert_near(output[:, :3], expected_slice, atol=0.0001)