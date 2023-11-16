""" Testing suite for the PyTorch ConvNext model. """
import inspect
import unittest
from transformers import ConvNextConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import ConvNextBackbone, ConvNextForImageClassification, ConvNextModel
    from transformers.models.convnext.modeling_convnext import CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

class ConvNextModelTester:

    def __init__(self, parent, batch_size=13, image_size=32, num_channels=3, num_stages=4, hidden_sizes=[10, 20, 30, 40], depths=[2, 2, 3, 2], is_training=True, use_labels=True, intermediate_size=37, hidden_act='gelu', num_labels=10, initializer_range=0.02, out_features=['stage2', 'stage3', 'stage4'], out_indices=[2, 3, 4], scope=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.out_features = out_features
        self.out_indices = out_indices
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            return 10
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            print('Hello World!')
        return ConvNextConfig(num_channels=self.num_channels, hidden_sizes=self.hidden_sizes, depths=self.depths, num_stages=self.num_stages, hidden_act=self.hidden_act, is_decoder=False, initializer_range=self.initializer_range, out_features=self.out_features, out_indices=self.out_indices, num_labels=self.num_labels)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        model = ConvNextModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            while True:
                i = 10
        model = ConvNextForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_backbone(self, config, pixel_values, labels):
        if False:
            print('Hello World!')
        model = ConvNextBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 4, 4])
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])
        config.out_features = None
        model = ConvNextBackbone(config=config)
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
class ConvNextModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ConvNext does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (ConvNextModel, ConvNextForImageClassification, ConvNextBackbone) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': ConvNextModel, 'image-classification': ConvNextForImageClassification} if is_torch_available() else {}
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = ConvNextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ConvNextConfig, has_text_modality=False, hidden_size=37)

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

    @unittest.skip(reason='ConvNext does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='ConvNext does not support input and output embeddings')
    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='ConvNext does not use feedforward chunking')
    def test_feed_forward_chunking(self):
        if False:
            print('Hello World!')
        pass

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

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_hidden_states_output(self):
        if False:
            return 10

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                for i in range(10):
                    print('nop')
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
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ConvNextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        for i in range(10):
            print('nop')
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class ConvNextModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            return 10
        return AutoImageProcessor.from_pretrained('facebook/convnext-tiny-224') if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            return 10
        model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224').to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.026, -0.4739, 0.1911]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))

@require_torch
class ConvNextBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (ConvNextBackbone,) if is_torch_available() else ()
    config_class = ConvNextConfig
    has_attentions = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = ConvNextModelTester(self)