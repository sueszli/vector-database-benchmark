""" Testing suite for the PyTorch GLPN model. """
import inspect
import unittest
from transformers import is_torch_available, is_vision_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import MODEL_MAPPING, GLPNConfig, GLPNForDepthEstimation, GLPNModel
    from transformers.models.glpn.modeling_glpn import GLPN_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import GLPNImageProcessor

class GLPNConfigTester(ConfigTester):

    def create_and_test_config_common_properties(self):
        if False:
            for i in range(10):
                print('nop')
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, 'hidden_sizes'))
        self.parent.assertTrue(hasattr(config, 'num_attention_heads'))
        self.parent.assertTrue(hasattr(config, 'num_encoder_blocks'))

class GLPNModelTester:

    def __init__(self, parent, batch_size=13, image_size=64, num_channels=3, num_encoder_blocks=4, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], hidden_sizes=[16, 32, 64, 128], downsampling_rates=[1, 4, 8, 16], num_attention_heads=[1, 2, 4, 8], is_training=True, use_labels=True, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, initializer_range=0.02, decoder_hidden_size=16, num_labels=3, scope=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.sr_ratios = sr_ratios
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.decoder_hidden_size = decoder_hidden_size
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            print('Hello World!')
        return GLPNConfig(image_size=self.image_size, num_channels=self.num_channels, num_encoder_blocks=self.num_encoder_blocks, depths=self.depths, hidden_sizes=self.hidden_sizes, num_attention_heads=self.num_attention_heads, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, initializer_range=self.initializer_range, decoder_hidden_size=self.decoder_hidden_size)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        model = GLPNModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        expected_height = expected_width = self.image_size // (self.downsampling_rates[-1] * 2)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], expected_height, expected_width))

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        config.num_labels = self.num_labels
        model = GLPNForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class GLPNModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (GLPNModel, GLPNForDepthEstimation) if is_torch_available() else ()
    pipeline_model_mapping = {'depth-estimation': GLPNForDepthEstimation, 'feature-extraction': GLPNModel} if is_torch_available() else {}
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False

    def setUp(self):
        if False:
            return 10
        self.model_tester = GLPNModelTester(self)
        self.config_tester = GLPNConfigTester(self, config_class=GLPNConfig)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_depth_estimation(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    @unittest.skip('GLPN does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip('GLPN does not have get_input_embeddings method and get_output_embeddings methods')
    def test_model_common_attributes(self):
        if False:
            while True:
                i = 10
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

    def test_attention_outputs(self):
        if False:
            print('Hello World!')
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
            expected_num_attentions = sum(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            expected_seq_len = (self.model_tester.image_size // 4) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (4 * self.model_tester.sr_ratios[0])) ** 2
            self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads[0], expected_seq_len, expected_reduced_seq_len])
            expected_seq_len = (self.model_tester.image_size // 32) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (32 * self.model_tester.sr_ratios[-1])) ** 2
            self.assertListEqual(list(attentions[-1].shape[-3:]), [self.model_tester.num_attention_heads[-1], expected_seq_len, expected_reduced_seq_len])
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
            expected_seq_len = (self.model_tester.image_size // 4) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (4 * self.model_tester.sr_ratios[0])) ** 2
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads[0], expected_seq_len, expected_reduced_seq_len])

    def test_hidden_states_output(self):
        if False:
            return 10

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                while True:
                    i = 10
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.hidden_states
            expected_num_layers = self.model_tester.num_encoder_blocks
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(list(hidden_states[0].shape[-3:]), [self.model_tester.hidden_sizes[0], self.model_tester.image_size // 4, self.model_tester.image_size // 4])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_training(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.model_tester.is_training:
            return
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        for model_class in self.all_model_classes:
            if model_class in get_values(MODEL_MAPPING):
                continue
            if model_class.__name__ == 'GLPNForDepthEstimation':
                (batch_size, num_channels, height, width) = inputs_dict['pixel_values'].shape
                inputs_dict['labels'] = torch.zeros([self.model_tester.batch_size, height, width], device=torch_device).long()
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_model_from_pretrained(self):
        if False:
            print('Hello World!')
        for model_name in GLPN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GLPNModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        i = 10
        return i + 15
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
@slow
class GLPNModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_depth_estimation(self):
        if False:
            for i in range(10):
                print('nop')
        image_processor = GLPNImageProcessor.from_pretrained(GLPN_PRETRAINED_MODEL_ARCHIVE_LIST[0])
        model = GLPNForDepthEstimation.from_pretrained(GLPN_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size([1, 480, 640])
        self.assertEqual(outputs.predicted_depth.shape, expected_shape)
        expected_slice = torch.tensor([[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=0.0001))