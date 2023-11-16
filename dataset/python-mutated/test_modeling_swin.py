""" Testing suite for the PyTorch Swin model. """
import collections
import inspect
import unittest
from transformers import SwinConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import SwinBackbone, SwinForImageClassification, SwinForMaskedImageModeling, SwinModel
    from transformers.models.swin.modeling_swin import SWIN_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

class SwinModelTester:

    def __init__(self, parent, batch_size=13, image_size=32, patch_size=2, num_channels=3, embed_dim=16, depths=[1, 2, 1], num_heads=[2, 2, 4], window_size=2, mlp_ratio=2.0, qkv_bias=True, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, drop_path_rate=0.1, hidden_act='gelu', use_absolute_embeddings=False, patch_norm=True, initializer_range=0.02, layer_norm_eps=1e-05, is_training=True, scope=None, use_labels=True, type_sequence_label_size=10, encoder_stride=8, out_features=['stage1', 'stage2'], out_indices=[1, 2]):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.patch_norm = patch_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.scope = scope
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.encoder_stride = encoder_stride
        self.out_features = out_features
        self.out_indices = out_indices

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            print('Hello World!')
        return SwinConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, embed_dim=self.embed_dim, depths=self.depths, num_heads=self.num_heads, window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, drop_path_rate=self.drop_path_rate, hidden_act=self.hidden_act, use_absolute_embeddings=self.use_absolute_embeddings, path_norm=self.patch_norm, layer_norm_eps=self.layer_norm_eps, initializer_range=self.initializer_range, encoder_stride=self.encoder_stride, out_features=self.out_features, out_indices=self.out_indices)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        model = SwinModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        expected_seq_len = (config.image_size // config.patch_size) ** 2 // 4 ** (len(config.depths) - 1)
        expected_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, expected_dim))

    def create_and_check_backbone(self, config, pixel_values, labels):
        if False:
            return 10
        model = SwinBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], 16, 16])
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        config.out_features = None
        model = SwinBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, model.channels[-1], 4, 4])
        self.parent.assertEqual(len(model.channels), 1)

    def create_and_check_for_masked_image_modeling(self, config, pixel_values, labels):
        if False:
            for i in range(10):
                print('nop')
        model = SwinForMaskedImageModeling(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size))
        config.num_channels = 1
        model = SwinForMaskedImageModeling(config)
        model.to(torch_device)
        model.eval()
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 1, self.image_size, self.image_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            return 10
        config.num_labels = self.type_sequence_label_size
        model = SwinForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))
        config.num_channels = 1
        model = SwinForImageClassification(config)
        model.to(torch_device)
        model.eval()
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class SwinModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (SwinModel, SwinBackbone, SwinForImageClassification, SwinForMaskedImageModeling) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': SwinModel, 'image-classification': SwinForImageClassification} if is_torch_available() else {}
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = SwinModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SwinConfig, embed_dim=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        if False:
            i = 10
            return i + 15
        return

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Got `CUDA error: misaligned address` with PyTorch 2.0.0.')
    def test_multi_gpu_data_parallel_forward(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_training_gradient_checkpointing(self):
        if False:
            while True:
                i = 10
        super().test_training_gradient_checkpointing()

    def test_backbone(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_for_masked_image_modeling(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @unittest.skip(reason='Swin does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='Swin Transformer does not use feedforward chunking')
    def test_feed_forward_chunking(self):
        if False:
            return 10
        pass

    def test_model_common_attributes(self):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        if False:
            print('Hello World!')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_attention_outputs(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        for model_class in self.all_model_classes:
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            expected_num_attentions = len(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            window_size_squared = config.window_size ** 2
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_heads[0], window_size_squared, window_size_squared])
            out_len = len(outputs)
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            added_hidden_states = 1 if model_class.__name__ == 'SwinBackbone' else 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))
            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_heads[0], window_size_squared, window_size_squared])

    def check_hidden_states_output(self, inputs_dict, config, model_class, image_size):
        if False:
            while True:
                i = 10
        model = model_class(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
        hidden_states = outputs.hidden_states
        expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', len(self.model_tester.depths) + 1)
        self.assertEqual(len(hidden_states), expected_num_layers)
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.assertListEqual(list(hidden_states[0].shape[-2:]), [num_patches, self.model_tester.embed_dim])
        if not model_class.__name__ == 'SwinBackbone':
            reshaped_hidden_states = outputs.reshaped_hidden_states
            self.assertEqual(len(reshaped_hidden_states), expected_num_layers)
            (batch_size, num_channels, height, width) = reshaped_hidden_states[0].shape
            reshaped_hidden_states = reshaped_hidden_states[0].view(batch_size, num_channels, height * width).permute(0, 2, 1)
            self.assertListEqual(list(reshaped_hidden_states.shape[-2:]), [num_patches, self.model_tester.embed_dim])

    def test_hidden_states_output(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        image_size = self.model_tester.image_size if isinstance(self.model_tester.image_size, collections.abc.Iterable) else (self.model_tester.image_size, self.model_tester.image_size)
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

    def test_hidden_states_output_with_padding(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.patch_size = 3
        image_size = self.model_tester.image_size if isinstance(self.model_tester.image_size, collections.abc.Iterable) else (self.model_tester.image_size, self.model_tester.image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        padded_height = image_size[0] + patch_size[0] - image_size[0] % patch_size[0]
        padded_width = image_size[1] + patch_size[1] - image_size[1] % patch_size[1]
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, (padded_height, padded_width))
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            self.check_hidden_states_output(inputs_dict, config, model_class, (padded_height, padded_width))

    @slow
    def test_model_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_name in SWIN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SwinModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_initialization(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                if 'embeddings' not in name and param.requires_grad:
                    self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

@require_vision
@require_torch
class SwinModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            return 10
        return AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224') if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            i = 10
            return i + 15
        model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224').to(torch_device)
        image_processor = self.default_image_processor
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.0948, -0.6454, -0.0921]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))

@require_torch
class SwinBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (SwinBackbone,) if is_torch_available() else ()
    config_class = SwinConfig

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = SwinModelTester(self)