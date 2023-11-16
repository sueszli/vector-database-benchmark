""" Testing suite for the TensorFlow Blip model. """
from __future__ import annotations
import inspect
import tempfile
import unittest
import numpy as np
import requests
from transformers import BlipConfig, BlipTextConfig, BlipVisionConfig
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import is_tf_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import TFBlipForConditionalGeneration, TFBlipForImageTextRetrieval, TFBlipForQuestionAnswering, TFBlipModel, TFBlipTextModel, TFBlipVisionModel
    from transformers.models.blip.modeling_tf_blip import TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import BlipProcessor

class TFBlipVisionModelTester:

    def __init__(self, parent, batch_size=12, image_size=30, patch_size=2, num_channels=3, is_training=True, hidden_size=32, projection_dim=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, dropout=0.1, attention_dropout=0.1, initializer_range=1e-10, scope=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return (config, pixel_values)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return BlipVisionConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, projection_dim=self.projection_dim, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, pixel_values):
        if False:
            i = 10
            return i + 15
        model = TFBlipVisionModel(config=config)
        result = model(pixel_values)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_tf
class TFBlipVisionModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Blip does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (TFBlipVisionModel,) if is_tf_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = TFBlipVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    @unittest.skip(reason='Blip does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    def test_forward_signature(self):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model_common_attributes(self):
        if False:
            print('Hello World!')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), tf.keras.layers.Layer)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, tf.keras.layers.Layer))

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='BlipVisionModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_from_base(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='BlipVisionModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_to_base(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_name in TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFBlipVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

class TFBlipTextModelTester:

    def __init__(self, parent, batch_size=12, seq_length=7, is_training=True, use_input_mask=True, use_labels=True, vocab_size=99, hidden_size=32, projection_dim=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, dropout=0.1, attention_dropout=0.1, max_position_embeddings=512, initializer_range=0.02, bos_token_id=0, scope=None):
        if False:
            print('Hello World!')
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        if input_mask is not None:
            input_mask = input_mask.numpy()
            (batch_size, seq_length) = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for (batch_idx, start_index) in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0
            input_mask = tf.convert_to_tensor(input_mask)
        config = self.get_config()
        return (config, input_ids, input_mask)

    def get_config(self):
        if False:
            print('Hello World!')
        return BlipTextConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, projection_dim=self.projection_dim, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, max_position_embeddings=self.max_position_embeddings, initializer_range=self.initializer_range, bos_token_id=self.bos_token_id)

    def create_and_check_model(self, config, input_ids, input_mask):
        if False:
            while True:
                i = 10
        model = TFBlipTextModel(config=config)
        result = model(input_ids, attention_mask=input_mask, training=False)
        result = model(input_ids, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_tf
class TFBlipTextModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlipTextModel,) if is_tf_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = TFBlipTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipTextConfig, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Blip does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='BlipTextModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_from_base(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='BlipTextModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_to_base(self):
        if False:
            while True:
                i = 10
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_name in TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFBlipTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_pt_tf_model_equivalence(self, allow_missing_keys=True):
        if False:
            return 10
        super().test_pt_tf_model_equivalence(allow_missing_keys=allow_missing_keys)

class TFBlipModelTester:

    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if False:
            while True:
                i = 10
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}
        self.parent = parent
        self.text_model_tester = TFBlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = TFBlipVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        (text_config, input_ids, attention_mask) = self.text_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return (config, input_ids, attention_mask, pixel_values)

    def get_config(self):
        if False:
            while True:
                i = 10
        return BlipConfig.from_text_vision_configs(self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        if False:
            while True:
                i = 10
        model = TFBlipModel(config)
        result = model(input_ids, pixel_values, attention_mask, training=False)
        self.parent.assertEqual(result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size))
        self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values, 'return_loss': True}
        return (config, inputs_dict)

@require_tf
class TFBlipModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlipModel,) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFBlipModel, 'image-to-text': TFBlipForConditionalGeneration} if is_tf_available() else {}
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_onnx = False

    def setUp(self):
        if False:
            return 10
        self.model_tester = TFBlipModelTester(self)

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Hidden_states is tested in individual model tests')
    def test_hidden_states_output(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Inputs_embeds is tested in individual model tests')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='BlipModel does not have input/output embeddings')
    def test_model_common_attributes(self):
        if False:
            return 10
        pass

    def test_load_vision_text_config(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_name in TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFBlipModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_pt_tf_model_equivalence(self, allow_missing_keys=True):
        if False:
            i = 10
            return i + 15
        super().test_pt_tf_model_equivalence(allow_missing_keys=allow_missing_keys)

    @unittest.skip('Matt: Re-enable this test when we have a proper export function for TF models.')
    def test_saved_model_creation(self):
        if False:
            i = 10
            return i + 15
        pass

class BlipTextRetrievalModelTester:

    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if False:
            for i in range(10):
                print('nop')
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}
        self.parent = parent
        self.text_model_tester = TFBlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = TFBlipVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        (text_config, input_ids, attention_mask) = self.text_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return (config, input_ids, attention_mask, pixel_values)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return BlipConfig.from_text_vision_configs(self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        if False:
            i = 10
            return i + 15
        model = TFBlipModel(config)
        result = model(input_ids, pixel_values, attention_mask, training=False)
        self.parent.assertEqual(result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size))
        self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values}
        return (config, inputs_dict)

class BlipTextImageModelsModelTester:

    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if False:
            while True:
                i = 10
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}
        self.parent = parent
        self.text_model_tester = TFBlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = TFBlipVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        (text_config, input_ids, attention_mask) = self.text_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return (config, input_ids, attention_mask, pixel_values)

    def get_config(self):
        if False:
            return 10
        return BlipConfig.from_text_vision_configs(self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        if False:
            while True:
                i = 10
        model = TFBlipModel(config)
        result = model(input_ids, pixel_values, attention_mask, training=False)
        self.parent.assertEqual(result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size))
        self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'labels': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values}
        return (config, inputs_dict)

class BlipVQAModelsModelTester:

    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if False:
            print('Hello World!')
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}
        self.parent = parent
        self.text_model_tester = TFBlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = TFBlipVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        (text_config, input_ids, attention_mask) = self.text_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return (config, input_ids, attention_mask, pixel_values)

    def get_config(self):
        if False:
            print('Hello World!')
        return BlipConfig.from_text_vision_configs(self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        if False:
            return 10
        model = TFBlipModel(config)
        result = model(input_ids, pixel_values, attention_mask, training=False)
        self.parent.assertEqual(result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size))
        self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'decoder_input_ids': input_ids, 'labels': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_tf
@require_vision
class TFBlipVQAModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlipForQuestionAnswering,) if is_tf_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_onnx = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = BlipVQAModelsModelTester(self)

    def _prepare_inputs_for_vqa(self):
        if False:
            print('Hello World!')
        (_, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict['labels'] = inputs_dict['input_ids']
        inputs_dict['decoder_input_ids'] = inputs_dict['input_ids']
        inputs_dict.pop('return_loss')
        return inputs_dict

    def test_class_name_consistency(self):
        if False:
            print('Hello World!')
        '\n        Tests that all VQA models have a class name that ends with "ForQuestionAnswering"\n        '
        for model_class in self.all_model_classes:
            model = model_class(self.model_tester.get_config())
            self.assertTrue(model.__class__.__name__.endswith('ForQuestionAnswering'), f"Class name should end with 'ForVisualQuestionAnswering' got {model.__class__.__name__}")

    def test_training(self):
        if False:
            while True:
                i = 10
        '\n        Tests that all VQA models can be trained on a single batch\n        '
        for model_class in self.all_model_classes:
            model = model_class(self.model_tester.get_config())
            loss = model(**self.model_tester.prepare_config_and_inputs_for_common()[1], training=True).loss
            self.assertIsNotNone(loss, 'Loss should not be None')

    @unittest.skip(reason='Hidden_states is tested in individual model tests')
    def test_hidden_states_output(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Inputs_embeds is tested in individual model tests')
    def test_inputs_embeds(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='BlipModel does not have input/output embeddings')
    def test_model_common_attributes(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Tested in individual model tests')
    def test_compile_tf_model(self):
        if False:
            return 10
        pass

    @unittest.skip("Model doesn't have a clean loss output.")
    def test_keras_fit(self):
        if False:
            return 10
        pass

@require_tf
class TFBlipTextRetrievalModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlipForImageTextRetrieval,) if is_tf_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_onnx = False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = BlipTextRetrievalModelTester(self)

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Hidden_states is tested in individual model tests')
    def test_hidden_states_output(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='Inputs_embeds is tested in individual model tests')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='BlipModel does not have input/output embeddings')
    def test_model_common_attributes(self):
        if False:
            return 10
        pass

    def test_training(self):
        if False:
            while True:
                i = 10
        if not self.model_tester.is_training:
            return
        for model_class in self.all_model_classes[:-1]:
            (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            inputs['labels'] = inputs['input_ids']
            loss = model(**inputs, training=True).loss
            self.assertTrue(loss is not None)

    def test_load_vision_text_config(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_name in TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFBlipModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @unittest.skip(reason='Tested in individual model tests')
    def test_compile_tf_model(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip("Model doesn't have a clean loss output.")
    def test_keras_fit(self):
        if False:
            while True:
                i = 10
        pass

@require_tf
class TFBlipTextImageModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlipForConditionalGeneration,) if is_tf_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_onnx = False

    def setUp(self):
        if False:
            return 10
        self.model_tester = BlipTextImageModelsModelTester(self)

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Hidden_states is tested in individual model tests')
    def test_hidden_states_output(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='Inputs_embeds is tested in individual model tests')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_forward_signature(self):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            arg_names = [*signature.parameters.keys()]
            if model.config.is_encoder_decoder:
                expected_arg_names = ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask']
                expected_arg_names.extend(['head_mask', 'decoder_head_mask', 'cross_attn_head_mask', 'encoder_outputs'] if 'head_mask' and 'decoder_head_mask' and ('cross_attn_head_mask' in arg_names) else ['encoder_outputs'])
                self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ['input_ids'] if model_class != TFBlipForConditionalGeneration else ['pixel_values']
                self.assertListEqual(arg_names[:1], expected_arg_names)

    @unittest.skip(reason='Tested in individual model tests')
    def test_compile_tf_model(self):
        if False:
            return 10
        pass

    @unittest.skip('Has some odd input names!')
    def test_keras_fit(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='BlipModel does not have input/output embeddings')
    def test_model_common_attributes(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_training(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.model_tester.is_training:
            return
        for model_class in self.all_model_classes[:-1]:
            (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            inputs['labels'] = inputs['input_ids']
            loss = model(**inputs, training=True).loss
            self.assertIsNotNone(loss)

    def test_load_vision_text_config(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFBlipModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        print('Hello World!')
    url = 'https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@require_vision
@require_tf
@slow
class TFBlipModelIntegrationTest(unittest.TestCase):

    def test_inference_image_captioning(self):
        if False:
            return 10
        model = TFBlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        image = prepare_img()
        inputs = processor(images=image, return_tensors='tf')
        predictions = model.generate(**inputs)
        self.assertEqual(predictions[0].numpy().tolist(), [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102])
        context = ['a picture of']
        inputs = processor(images=image, text=context, return_tensors='tf')
        predictions = model.generate(**inputs)
        self.assertEqual(predictions[0].numpy().tolist(), [30522, 1037, 3861, 1997, 1037, 2450, 1998, 2014, 3899, 2006, 1996, 3509, 102])

    def test_inference_vqa(self):
        if False:
            return 10
        model = TFBlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')
        processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
        image = prepare_img()
        text = 'how many dogs are in the picture?'
        inputs = processor(image, text=text, return_tensors='tf')
        out = model.generate(**inputs)
        self.assertEqual(out[0].numpy().tolist(), [30522, 1015, 102])

    def test_inference_itm(self):
        if False:
            print('Hello World!')
        model = TFBlipForImageTextRetrieval.from_pretrained('Salesforce/blip-itm-base-coco')
        processor = BlipProcessor.from_pretrained('Salesforce/blip-itm-base-coco')
        image = prepare_img()
        text = 'A woman and her dog sitting in a beach'
        inputs = processor(image, text, return_tensors='tf')
        out_itm = model(**inputs)
        out = model(**inputs, use_itm_head=False, training=False)
        expected_scores = tf.convert_to_tensor([[0.0029, 0.9971]])
        self.assertTrue(np.allclose(tf.nn.softmax(out_itm[0]).numpy(), expected_scores, rtol=0.001, atol=0.001))
        self.assertTrue(np.allclose(out[0], tf.convert_to_tensor([[0.5162]]), rtol=0.001, atol=0.001))