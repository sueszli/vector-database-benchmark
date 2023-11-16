from __future__ import annotations
import random
import unittest
from transformers import TransfoXLConfig, is_tf_available
from transformers.testing_utils import require_tf, slow
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST, TFTransfoXLForSequenceClassification, TFTransfoXLLMHeadModel, TFTransfoXLModel

class TFTransfoXLModelTester:

    def __init__(self, parent):
        if False:
            return 10
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.mem_len = 30
        self.key_length = self.seq_length + self.mem_len
        self.clamp_len = 15
        self.is_training = True
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.d_embed = 32
        self.num_attention_heads = 4
        self.d_head = 8
        self.d_inner = 128
        self.div_val = 2
        self.num_hidden_layers = 2
        self.scope = None
        self.seed = 1
        self.eos_token_id = 0
        self.num_labels = 3
        self.pad_token_id = self.vocab_size - 1
        self.init_range = 0.01

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        config = TransfoXLConfig(vocab_size=self.vocab_size, mem_len=self.mem_len, clamp_len=self.clamp_len, cutoffs=self.cutoffs, d_model=self.hidden_size, d_embed=self.d_embed, n_head=self.num_attention_heads, d_head=self.d_head, d_inner=self.d_inner, div_val=self.div_val, n_layer=self.num_hidden_layers, eos_token_id=self.eos_token_id, pad_token_id=self.vocab_size - 1, init_range=self.init_range, num_labels=self.num_labels)
        return (config, input_ids_1, input_ids_2, lm_labels)

    def set_seed(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def create_and_check_transfo_xl_model(self, config, input_ids_1, input_ids_2, lm_labels):
        if False:
            return 10
        model = TFTransfoXLModel(config)
        (hidden_states_1, mems_1) = model(input_ids_1).to_tuple()
        inputs = {'input_ids': input_ids_2, 'mems': mems_1}
        (hidden_states_2, mems_2) = model(inputs).to_tuple()
        self.parent.assertEqual(hidden_states_1.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(hidden_states_2.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertListEqual([mem.shape for mem in mems_1], [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers)
        self.parent.assertListEqual([mem.shape for mem in mems_2], [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers)

    def create_and_check_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, lm_labels):
        if False:
            return 10
        model = TFTransfoXLLMHeadModel(config)
        (lm_logits_1, mems_1) = model(input_ids_1).to_tuple()
        inputs = {'input_ids': input_ids_1, 'labels': lm_labels}
        (_, mems_1) = model(inputs).to_tuple()
        (lm_logits_2, mems_2) = model([input_ids_2, mems_1]).to_tuple()
        inputs = {'input_ids': input_ids_1, 'mems': mems_1, 'labels': lm_labels}
        (_, mems_2) = model(inputs).to_tuple()
        self.parent.assertEqual(lm_logits_1.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual([mem.shape for mem in mems_1], [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers)
        self.parent.assertEqual(lm_logits_2.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual([mem.shape for mem in mems_2], [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers)

    def create_and_check_transfo_xl_for_sequence_classification(self, config, input_ids_1, input_ids_2, lm_labels):
        if False:
            print('Hello World!')
        model = TFTransfoXLForSequenceClassification(config)
        result = model(input_ids_1)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids_1, input_ids_2, lm_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids_1}
        return (config, inputs_dict)

@require_tf
class TFTransfoXLModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFTransfoXLModel, TFTransfoXLLMHeadModel, TFTransfoXLForSequenceClassification) if is_tf_available() else ()
    all_generative_model_classes = () if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFTransfoXLModel, 'text-classification': TFTransfoXLForSequenceClassification, 'text-generation': TFTransfoXLLMHeadModel, 'zero-shot': TFTransfoXLForSequenceClassification} if is_tf_available() else {}
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False
    test_mismatched_shapes = False

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            for i in range(10):
                print('nop')
        if pipeline_test_casse_name == 'TextGenerationPipelineTests':
            return True
        return False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = TFTransfoXLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TransfoXLConfig, d_embed=37)

    def test_config(self):
        if False:
            return 10
        self.config_tester.run_common_tests()

    def test_transfo_xl_model(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_model(*config_and_inputs)

    def test_transfo_xl_lm_head(self):
        if False:
            i = 10
            return i + 15
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_lm_head(*config_and_inputs)

    def test_transfo_xl_sequence_classification_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_for_sequence_classification(*config_and_inputs)

    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        list_other_models_with_output_ebd = [TFTransfoXLForSequenceClassification]
        for model_class in self.all_model_classes:
            model = model_class(config)
            assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)
            if model_class in list_other_models_with_output_ebd:
                x = model.get_output_embeddings()
                assert isinstance(x, tf.keras.layers.Layer)
                name = model.get_bias()
                assert name is None
            else:
                x = model.get_output_embeddings()
                assert x is None
                name = model.get_bias()
                assert name is None

    def test_xla_mode(self):
        if False:
            return 10
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_name in TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFTransfoXLModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @unittest.skip(reason="This model doesn't play well with fit() due to not returning a single loss.")
    def test_dataset_conversion(self):
        if False:
            while True:
                i = 10
        pass

@require_tf
class TFTransfoXLModelLanguageGenerationTest(unittest.TestCase):

    @unittest.skip('Skip test until #12651 is resolved.')
    @slow
    def test_lm_generate_transfo_xl_wt103(self):
        if False:
            return 10
        model = TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        input_ids = tf.convert_to_tensor([[33, 1297, 2, 1, 1009, 4, 1109, 11739, 4762, 358, 5, 25, 245, 22, 1706, 17, 20098, 5, 3215, 21, 37, 1110, 3, 13, 1041, 4, 24, 603, 490, 2, 71477, 20098, 104447, 2, 20961, 1, 2604, 4, 1, 329, 3, 6224, 831, 16002, 2, 8, 603, 78967, 29546, 23, 803, 20, 25, 416, 5, 8, 232, 4, 277, 6, 1855, 4601, 3, 29546, 54, 8, 3609, 5, 57211, 49, 4, 1, 277, 18, 8, 1755, 15691, 3, 341, 25, 416, 693, 42573, 71, 17, 401, 94, 31, 17919, 2, 29546, 7873, 18, 1, 435, 23, 11011, 755, 5, 5167, 3, 7983, 98, 84, 2, 29546, 3267, 8, 3609, 4, 1, 4865, 1075, 2, 6087, 71, 6, 346, 8, 5854, 3, 29546, 824, 1400, 1868, 2, 19, 160, 2, 311, 8, 5496, 2, 20920, 17, 25, 15097, 3, 24, 24, 0]], dtype=tf.int32)
        expected_output_ids = [33, 1297, 2, 1, 1009, 4, 1109, 11739, 4762, 358, 5, 25, 245, 22, 1706, 17, 20098, 5, 3215, 21, 37, 1110, 3, 13, 1041, 4, 24, 603, 490, 2, 71477, 20098, 104447, 2, 20961, 1, 2604, 4, 1, 329, 3, 6224, 831, 16002, 2, 8, 603, 78967, 29546, 23, 803, 20, 25, 416, 5, 8, 232, 4, 277, 6, 1855, 4601, 3, 29546, 54, 8, 3609, 5, 57211, 49, 4, 1, 277, 18, 8, 1755, 15691, 3, 341, 25, 416, 693, 42573, 71, 17, 401, 94, 31, 17919, 2, 29546, 7873, 18, 1, 435, 23, 11011, 755, 5, 5167, 3, 7983, 98, 84, 2, 29546, 3267, 8, 3609, 4, 1, 4865, 1075, 2, 6087, 71, 6, 346, 8, 5854, 3, 29546, 824, 1400, 1868, 2, 19, 160, 2, 311, 8, 5496, 2, 20920, 17, 25, 15097, 3, 24, 24, 0, 33, 1, 1857, 2, 1, 1009, 4, 1109, 11739, 4762, 358, 5, 25, 245, 28, 1110, 3, 13, 1041, 4, 24, 603, 490, 2, 71477, 20098, 104447, 2, 20961, 1, 2604, 4, 1, 329, 3, 0]
        output_ids = model.generate(input_ids, max_length=200, do_sample=False)
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)