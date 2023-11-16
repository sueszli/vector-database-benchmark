""" Testing suite for the TensorFlow ViTMAE model. """
from __future__ import annotations
import copy
import inspect
import json
import math
import os
import tempfile
import unittest
from importlib import import_module
import numpy as np
from transformers import ViTMAEConfig
from transformers.file_utils import cached_property, is_tf_available, is_vision_available
from transformers.testing_utils import require_tf, require_vision, slow
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import TFViTMAEForPreTraining, TFViTMAEModel
if is_vision_available():
    from PIL import Image
    from transformers import ViTImageProcessor

class TFViTMAEModelTester:

    def __init__(self, parent, batch_size=13, image_size=30, patch_size=2, num_channels=3, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, type_sequence_label_size=10, initializer_range=0.02, num_labels=3, mask_ratio=0.6, scope=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.mask_ratio = mask_ratio
        self.scope = scope
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = int(math.ceil((1 - mask_ratio) * (num_patches + 1)))

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return ViTMAEConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, decoder_hidden_size=self.hidden_size, decoder_num_hidden_layers=self.num_hidden_layers, decoder_num_attention_heads=self.num_attention_heads, decoder_intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range, mask_ratio=self.mask_ratio)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        model = TFViTMAEModel(config=config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(self, config, pixel_values, labels):
        if False:
            print('Hello World!')
        model = TFViTMAEForPreTraining(config)
        result = model(pixel_values, training=False)
        num_patches = (self.image_size // self.patch_size) ** 2
        expected_num_channels = self.patch_size ** 2 * self.num_channels
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, expected_num_channels))
        config.num_channels = 1
        model = TFViTMAEForPreTraining(config)
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values, training=False)
        expected_num_channels = self.patch_size ** 2
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, expected_num_channels))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_tf
class TFViTMAEModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViTMAE does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (TFViTMAEModel, TFViTMAEForPreTraining) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFViTMAEModel} if is_tf_available() else {}
    test_pruning = False
    test_onnx = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = TFViTMAEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTMAEConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    @unittest.skip(reason='ViTMAE does not use inputs_embeds')
    def test_inputs_embeds(self):
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
            print('Hello World!')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_keyword_and_dict_args(self):
        if False:
            return 10
        np.random.seed(2)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        num_patches = int((config.image_size // config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs_dict = model(inputs, noise=noise)
            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            outputs_keywords = model(**inputs_keywords, noise=noise)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()
            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-06)

    def test_numpy_arrays_inputs(self):
        if False:
            while True:
                i = 10
        np.random.seed(2)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        num_patches = int((config.image_size // config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))

        def prepare_numpy_arrays(inputs_dict):
            if False:
                i = 10
                return i + 15
            inputs_np_dict = {}
            for (k, v) in inputs_dict.items():
                if tf.is_tensor(v):
                    inputs_np_dict[k] = v.numpy()
                else:
                    inputs_np_dict[k] = np.array(k)
            return inputs_np_dict
        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs_np = prepare_numpy_arrays(inputs)
            output_for_dict_input = model(inputs_np, noise=noise)
            output_for_kw_input = model(**inputs_np, noise=noise)
            self.assert_outputs_same(output_for_dict_input, output_for_kw_input)

    def check_pt_tf_models(self, tf_model, pt_model, tf_inputs_dict):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2)
        num_patches = int((tf_model.config.image_size // tf_model.config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        tf_noise = tf.constant(noise)
        tf_inputs_dict['noise'] = tf_noise
        super().check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)

    def test_keras_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        tf_main_layer_classes = {module_member for model_class in self.all_model_classes for module in (import_module(model_class.__module__),) for module_member_name in dir(module) if module_member_name.endswith('MainLayer') and module_member_name[:-len('MainLayer')] == model_class.__name__[:-len('Model')] for module_member in (getattr(module, module_member_name),) if isinstance(module_member, type) and tf.keras.layers.Layer in module_member.__bases__ and getattr(module_member, '_keras_serializable', False)}
        num_patches = int((config.image_size // config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        noise = tf.convert_to_tensor(noise)
        inputs_dict.update({'noise': noise})
        for main_layer_class in tf_main_layer_classes:
            main_layer = main_layer_class(config)
            symbolic_inputs = {name: tf.keras.Input(tensor.shape[1:], dtype=tensor.dtype) for (name, tensor) in inputs_dict.items()}
            model = tf.keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, 'keras_model.h5')
                model.save(filepath)
                model = tf.keras.models.load_model(filepath, custom_objects={main_layer_class.__name__: main_layer_class})
                assert isinstance(model, tf.keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    @slow
    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        num_patches = int((config.image_size // config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        for model_class in self.all_model_classes:
            model = model_class(config)
            model_input = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(model_input, noise=noise)
            if model_class.__name__ == 'TFViTMAEModel':
                out_2 = outputs.last_hidden_state.numpy()
                out_2[np.isnan(out_2)] = 0
            else:
                out_2 = outputs.logits.numpy()
                out_2[np.isnan(out_2)] = 0
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)
                after_outputs = model(model_input, noise=noise)
                if model_class.__name__ == 'TFViTMAEModel':
                    out_1 = after_outputs['last_hidden_state'].numpy()
                    out_1[np.isnan(out_1)] = 0
                else:
                    out_1 = after_outputs['logits'].numpy()
                    out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-05)

    def test_save_load_config(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        num_patches = int((config.image_size // config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        for model_class in self.all_model_classes:
            model = model_class(config)
            model_inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(model_inputs, noise=noise)
            model_config = model.get_config()
            json.dumps(model_config)
            new_model = model_class.from_config(model.get_config())
            _ = model_class.from_config(model.config)
            _ = new_model(model_inputs)
            new_model.set_weights(model.get_weights())
            after_outputs = new_model(model_inputs, noise=noise)
            self.assert_outputs_same(after_outputs, outputs)

    @unittest.skip(reason='ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load\n    to get deterministic results.')
    def test_determinism(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load')
    def test_model_outputs_equivalence(self):
        if False:
            return 10
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        model = TFViTMAEModel.from_pretrained('google/vit-base-patch16-224')
        self.assertIsNotNone(model)

def prepare_img():
    if False:
        i = 10
        return i + 15
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_tf
@require_vision
class TFViTMAEModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            i = 10
            return i + 15
        return ViTImageProcessor.from_pretrained('facebook/vit-mae-base') if is_vision_available() else None

    @slow
    def test_inference_for_pretraining(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(2)
        model = TFViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='tf')
        vit_mae_config = ViTMAEConfig()
        num_patches = int((vit_mae_config.image_size // vit_mae_config.patch_size) ** 2)
        noise = np.random.uniform(size=(1, num_patches))
        outputs = model(**inputs, noise=noise)
        expected_shape = tf.convert_to_tensor([1, 196, 768])
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = tf.convert_to_tensor([[-0.0548, -1.7023, -0.9325], [0.3721, -0.567, -0.2233], [0.8235, -1.3878, -0.3524]])
        tf.debugging.assert_near(outputs.logits[0, :3, :3], expected_slice, atol=0.0001)