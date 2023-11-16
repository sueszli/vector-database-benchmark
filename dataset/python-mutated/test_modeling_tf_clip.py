""" Testing suite for the TensorFlow CLIP model. """
from __future__ import annotations
import inspect
import os
import tempfile
import unittest
from importlib import import_module
import requests
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import is_tf_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import TFCLIPModel, TFCLIPTextModel, TFCLIPVisionModel, TFSharedEmbeddings
    from transformers.models.clip.modeling_tf_clip import TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import CLIPProcessor

class TFCLIPVisionModelTester:

    def __init__(self, parent, batch_size=12, image_size=30, patch_size=2, num_channels=3, is_training=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, dropout=0.1, attention_dropout=0.1, initializer_range=0.02, scope=None):
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
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            return 10
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return (config, pixel_values)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return CLIPVisionConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, pixel_values):
        if False:
            return 10
        model = TFCLIPVisionModel(config=config)
        result = model(pixel_values, training=False)
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
class TFCLIPVisionModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (TFCLIPVisionModel,) if is_tf_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = TFCLIPVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_graph_mode_with_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    def test_model_common_attributes(self):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), tf.keras.layers.Layer)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, tf.keras.layers.Layer))

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

    def test_model(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        image_size = (self.model_tester.image_size, self.model_tester.image_size)
        patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        for model_class in self.all_model_classes:
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = False
            config.return_dict = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            out_len = len(outputs)
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))
            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_len, seq_len])

    def test_hidden_states_output(self):
        if False:
            i = 10
            return i + 15

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                for i in range(10):
                    print('nop')
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
            self.assertEqual(len(hidden_states), expected_num_layers)
            image_size = (self.model_tester.image_size, self.model_tester.image_size)
            patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
            num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
            seq_length = num_patches + 1
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [seq_length, self.model_tester.hidden_size])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCLIPVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_saved_model_creation_extended(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True
        if hasattr(config, 'use_cache'):
            config.use_cache = True
        image_size = (self.model_tester.image_size, self.model_tester.image_size)
        patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        for model_class in self.all_model_classes:
            class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)
            num_out = len(model(class_inputs_dict))
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=True)
                saved_model_dir = os.path.join(tmpdirname, 'saved_model', '1')
                model = tf.keras.models.load_model(saved_model_dir)
                outputs = model(class_inputs_dict)
                output_hidden_states = outputs['hidden_states']
                output_attentions = outputs['attentions']
                self.assertEqual(len(outputs), num_out)
                expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
                self.assertEqual(len(output_hidden_states), expected_num_layers)
                self.assertEqual(len(output_attentions), self.model_tester.num_hidden_layers)
                image_size = (self.model_tester.image_size, self.model_tester.image_size)
                patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
                num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
                seq_len = num_patches + 1
                self.assertListEqual(list(output_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_len, seq_len])
                self.assertListEqual(list(output_hidden_states[0].shape[-2:]), [seq_len, self.model_tester.hidden_size])

class TFCLIPTextModelTester:

    def __init__(self, parent, batch_size=12, seq_length=7, is_training=True, use_input_mask=True, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, dropout=0.1, attention_dropout=0.1, max_position_embeddings=512, initializer_range=0.02, scope=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
            input_mask = tf.concat([tf.ones_like(input_mask[:, :1], dtype=input_mask.dtype), input_mask[:, 1:]], axis=-1)
        config = self.get_config()
        return (config, input_ids, input_mask)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return CLIPTextConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, max_position_embeddings=self.max_position_embeddings, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, input_ids, input_mask):
        if False:
            while True:
                i = 10
        model = TFCLIPTextModel(config=config)
        result = model(input_ids, attention_mask=input_mask, training=False)
        result = model(input_ids, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_tf
class TFCLIPTextModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFCLIPTextModel,) if is_tf_available() else ()
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = TFCLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPTextConfig, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_name in TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCLIPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_saved_model_creation_extended(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True
        if hasattr(config, 'use_cache'):
            config.use_cache = True
        for model_class in self.all_model_classes:
            class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)
            num_out = len(model(class_inputs_dict))
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=True)
                saved_model_dir = os.path.join(tmpdirname, 'saved_model', '1')
                model = tf.keras.models.load_model(saved_model_dir)
                outputs = model(class_inputs_dict)
                output_hidden_states = outputs['hidden_states']
                output_attentions = outputs['attentions']
                self.assertEqual(len(outputs), num_out)
                expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
                self.assertEqual(len(output_hidden_states), expected_num_layers)
                self.assertListEqual(list(output_hidden_states[0].shape[-2:]), [self.model_tester.seq_length, self.model_tester.hidden_size])
                self.assertEqual(len(output_attentions), self.model_tester.num_hidden_layers)
                seq_length = self.model_tester.seq_length
                key_length = getattr(self.model_tester, 'key_length', seq_length)
                self.assertListEqual(list(output_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_length, key_length])

class TFCLIPModelTester:

    def __init__(self, parent, is_training=True):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.text_model_tester = TFCLIPTextModelTester(parent)
        self.vision_model_tester = TFCLIPVisionModelTester(parent)
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
            while True:
                i = 10
        return CLIPConfig.from_text_vision_configs(self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        if False:
            i = 10
            return i + 15
        model = TFCLIPModel(config)
        result = model(input_ids, pixel_values, attention_mask, training=False)
        self.parent.assertEqual(result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size))
        self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values, 'return_loss': True}
        return (config, inputs_dict)

@require_tf
class TFCLIPModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFCLIPModel,) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFCLIPModel} if is_tf_available() else {}
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_onnx = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = TFCLIPModelTester(self)

    def test_model(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_model_common_attributes(self):
        if False:
            return 10
        pass

    def test_keras_save_load(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        if self.__class__.__name__ == 'TFCLIPModelTest':
            inputs_dict.pop('return_loss', None)
        tf_main_layer_classes = {module_member for model_class in self.all_model_classes for module in (import_module(model_class.__module__),) for module_member_name in dir(module) if module_member_name.endswith('MainLayer') and module_member_name[:-len('MainLayer')] == model_class.__name__[:-len('Model')] for module_member in (getattr(module, module_member_name),) if isinstance(module_member, type) and tf.keras.layers.Layer in module_member.__bases__ and getattr(module_member, '_keras_serializable', False)}
        for main_layer_class in tf_main_layer_classes:
            if 'T5' in main_layer_class.__name__:
                shared = TFSharedEmbeddings(99, 32, name='shared')
                config.use_cache = inputs_dict.pop('use_cache', None)
                main_layer = main_layer_class(config, embed_tokens=shared)
            else:
                main_layer = main_layer_class(config)
            symbolic_inputs = {name: tf.keras.Input(tensor.shape[1:], dtype=tensor.dtype) for (name, tensor) in inputs_dict.items()}
            model = tf.keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, 'keras_model.h5')
                model.save(filepath)
                if 'T5' in main_layer_class.__name__:
                    model = tf.keras.models.load_model(filepath, custom_objects={main_layer_class.__name__: main_layer_class, 'TFSharedEmbeddings': TFSharedEmbeddings})
                else:
                    model = tf.keras.models.load_model(filepath, custom_objects={main_layer_class.__name__: main_layer_class})
                assert isinstance(model, tf.keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCLIPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @unittest.skip(reason="Currently `saved_model` doesn't work with nested outputs.")
    @slow
    def test_saved_model_creation(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason="Currently `saved_model` doesn't work with nested outputs.")
    @slow
    def test_saved_model_creation_extended(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason="`saved_model` doesn't work with nested outputs so no preparation happens.")
    @slow
    def test_prepare_serving_output(self):
        if False:
            i = 10
            return i + 15
        pass

def prepare_img():
    if False:
        print('Hello World!')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@require_vision
@require_tf
class TFCLIPModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference(self):
        if False:
            i = 10
            return i + 15
        model_name = 'openai/clip-vit-base-patch32'
        model = TFCLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        image = prepare_img()
        inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, padding=True, return_tensors='tf')
        outputs = model(**inputs, training=False)
        self.assertEqual(outputs.logits_per_image.shape, tf.TensorShape((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])))
        self.assertEqual(outputs.logits_per_text.shape, tf.TensorShape((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])))
        expected_logits = tf.constant([[24.5701, 19.3049]])
        tf.debugging.assert_near(outputs.logits_per_image, expected_logits, atol=0.001)