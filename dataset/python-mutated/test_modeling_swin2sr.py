""" Testing suite for the PyTorch Swin2SR model. """
import inspect
import unittest
from transformers import Swin2SRConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import Swin2SRForImageSuperResolution, Swin2SRModel
    from transformers.models.swin2sr.modeling_swin2sr import SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import Swin2SRImageProcessor

class Swin2SRModelTester:

    def __init__(self, parent, batch_size=13, image_size=32, patch_size=1, num_channels=3, num_channels_out=1, embed_dim=16, depths=[1, 2, 1], num_heads=[2, 2, 4], window_size=2, mlp_ratio=2.0, qkv_bias=True, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, drop_path_rate=0.1, hidden_act='gelu', use_absolute_embeddings=False, patch_norm=True, initializer_range=0.02, layer_norm_eps=1e-05, is_training=True, scope=None, use_labels=False, upscale=2):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_channels_out = num_channels_out
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
        self.upscale = upscale
        self.num_hidden_layers = len(depths)
        self.hidden_size = embed_dim
        self.seq_length = (image_size // patch_size) ** 2

    def prepare_config_and_inputs(self):
        if False:
            return 10
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            while True:
                i = 10
        return Swin2SRConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, num_channels_out=self.num_channels_out, embed_dim=self.embed_dim, depths=self.depths, num_heads=self.num_heads, window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, drop_path_rate=self.drop_path_rate, hidden_act=self.hidden_act, use_absolute_embeddings=self.use_absolute_embeddings, path_norm=self.patch_norm, layer_norm_eps=self.layer_norm_eps, initializer_range=self.initializer_range, upscale=self.upscale)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            for i in range(10):
                print('nop')
        model = Swin2SRModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.embed_dim, self.image_size, self.image_size))

    def create_and_check_for_image_super_resolution(self, config, pixel_values, labels):
        if False:
            while True:
                i = 10
        model = Swin2SRForImageSuperResolution(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        expected_image_size = self.image_size * self.upscale
        self.parent.assertEqual(result.reconstruction.shape, (self.batch_size, self.num_channels_out, expected_image_size, expected_image_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class Swin2SRModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Swin2SRModel, Swin2SRForImageSuperResolution) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': Swin2SRModel} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = Swin2SRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Swin2SRConfig, embed_dim=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_image_super_resolution(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_super_resolution(*config_and_inputs)

    @unittest.skip(reason='Got `CUDA error: misaligned address` with PyTorch 2.0.0.')
    def test_multi_gpu_data_parallel_forward(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='Swin2SR does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='Swin2SR does not support training yet')
    def test_training(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='Swin2SR does not support training yet')
    def test_training_gradient_checkpointing(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            print('Hello World!')
        pass

    def test_model_common_attributes(self):
        if False:
            print('Hello World!')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        if False:
            return 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Swin2SRModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_initialization(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                if 'logit_scale' in name:
                    continue
                if param.requires_grad:
                    self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def test_attention_outputs(self):
        if False:
            for i in range(10):
                print('nop')
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
            self.assertEqual(out_len + 1, len(outputs))
            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_heads[0], window_size_squared, window_size_squared])

@require_vision
@require_torch
@slow
class Swin2SRModelIntegrationTest(unittest.TestCase):

    def test_inference_image_super_resolution_head(self):
        if False:
            print('Hello World!')
        processor = Swin2SRImageProcessor()
        model = Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2SR-classical-sr-x2-64').to(torch_device)
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        inputs = processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size([1, 3, 976, 1296])
        self.assertEqual(outputs.reconstruction.shape, expected_shape)
        expected_slice = torch.tensor([[0.5458, 0.5546, 0.5638], [0.5526, 0.5565, 0.5651], [0.5396, 0.5426, 0.5621]]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=0.0001))