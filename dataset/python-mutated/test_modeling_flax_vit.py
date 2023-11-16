import inspect
import unittest
import numpy as np
from transformers import ViTConfig, is_flax_available
from transformers.testing_utils import require_flax, slow
from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor
if is_flax_available():
    import jax
    from transformers.models.vit.modeling_flax_vit import FlaxViTForImageClassification, FlaxViTModel

class FlaxViTModelTester(unittest.TestCase):

    def __init__(self, parent, batch_size=13, image_size=30, patch_size=2, num_channels=3, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, type_sequence_label_size=10, initializer_range=0.02):
        if False:
            print('Hello World!')
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
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = ViTConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range)
        return (config, pixel_values)

    def create_and_check_model(self, config, pixel_values):
        if False:
            i = 10
            return i + 15
        model = FlaxViTModel(config=config)
        result = model(pixel_values)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values):
        if False:
            while True:
                i = 10
        config.num_labels = self.type_sequence_label_size
        model = FlaxViTForImageClassification(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))
        config.num_channels = 1
        model = FlaxViTForImageClassification(config)
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_flax
class FlaxViTModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxViTModel, FlaxViTForImageClassification) if is_flax_available() else ()

    def setUp(self) -> None:
        if False:
            return 10
        self.model_tester = FlaxViTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTConfig, has_text_modality=False, hidden_size=37)

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

    def test_for_image_classification(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_forward_signature(self):
        if False:
            for i in range(10):
                print('nop')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_jit_compilation(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(pixel_values, **kwargs):
                    if False:
                        while True:
                            i = 10
                    return model(pixel_values=pixel_values, **kwargs)
                with self.subTest('JIT Enabled'):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()
                with self.subTest('JIT Disabled'):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(outputs), len(jitted_outputs))
                for (jitted_output, output) in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)

    @slow
    def test_model_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained('google/vit-base-patch16-224')
            outputs = model(np.ones((1, 3, 224, 224)))
            self.assertIsNotNone(outputs)