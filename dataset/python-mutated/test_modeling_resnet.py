""" Testing suite for the PyTorch ResNet model. """
import inspect
import unittest
from transformers import ResNetConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import ResNetBackbone, ResNetForImageClassification, ResNetModel
    from transformers.models.resnet.modeling_resnet import RESNET_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

class ResNetModelTester:

    def __init__(self, parent, batch_size=3, image_size=32, num_channels=3, embeddings_size=10, hidden_sizes=[10, 20, 30, 40], depths=[1, 1, 2, 1], is_training=True, use_labels=True, hidden_act='relu', num_labels=3, scope=None, out_features=['stage2', 'stage3', 'stage4'], out_indices=[2, 3, 4]):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.scope = scope
        self.num_stages = len(hidden_sizes)
        self.out_features = out_features
        self.out_indices = out_indices

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            print('Hello World!')
        return ResNetConfig(num_channels=self.num_channels, embeddings_size=self.embeddings_size, hidden_sizes=self.hidden_sizes, depths=self.depths, hidden_act=self.hidden_act, num_labels=self.num_labels, out_features=self.out_features, out_indices=self.out_indices)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            while True:
                i = 10
        model = ResNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        config.num_labels = self.num_labels
        model = ResNetForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_backbone(self, config, pixel_values, labels):
        if False:
            return 10
        model = ResNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 4, 4])
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])
        config.out_features = None
        model = ResNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[-1], 1, 1])
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class ResNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ResNet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (ResNetModel, ResNetForImageClassification, ResNetBackbone) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': ResNetModel, 'image-classification': ResNetForImageClassification} if is_torch_available() else {}
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        if False:
            return 10
        self.model_tester = ResNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ResNetConfig, has_text_modality=False)

    def test_config(self):
        if False:
            while True:
                i = 10
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        if False:
            return 10
        return

    @unittest.skip(reason='ResNet does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='ResNet does not support input and output embeddings')
    def test_model_common_attributes(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_forward_signature(self):
        if False:
            while True:
                i = 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_initialization(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for (name, module) in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.assertTrue(torch.all(module.weight == 1), msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    self.assertTrue(torch.all(module.bias == 0), msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def test_hidden_states_output(self):
        if False:
            print('Hello World!')

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                return 10
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            expected_num_stages = self.model_tester.num_stages
            self.assertEqual(len(hidden_states), expected_num_stages + 1)
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [self.model_tester.image_size // 4, self.model_tester.image_size // 4])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        layers_type = ['basic', 'bottleneck']
        for model_class in self.all_model_classes:
            for layer_type in layers_type:
                config.layer_type = layer_type
                inputs_dict['output_hidden_states'] = True
                check_hidden_states_output(inputs_dict, config, model_class)
                del inputs_dict['output_hidden_states']
                config.output_hidden_states = True
                check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason='ResNet does not use feedforward chunking')
    def test_feed_forward_chunking(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_for_image_classification(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in RESNET_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ResNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        i = 10
        return i + 15
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class ResNetModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            while True:
                i = 10
        return AutoImageProcessor.from_pretrained(RESNET_PRETRAINED_MODEL_ARCHIVE_LIST[0]) if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            print('Hello World!')
        model = ResNetForImageClassification.from_pretrained(RESNET_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-11.1069, -9.7877, -8.3777]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))

@require_torch
class ResNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (ResNetBackbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = ResNetConfig

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = ResNetModelTester(self)