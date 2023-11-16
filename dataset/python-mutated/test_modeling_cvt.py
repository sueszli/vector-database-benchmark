""" Testing suite for the PyTorch CvT model. """
import inspect
import unittest
from math import floor
from transformers import CvtConfig
from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import CvtForImageClassification, CvtModel
    from transformers.models.cvt.modeling_cvt import CVT_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

class CvtConfigTester(ConfigTester):

    def create_and_test_config_common_properties(self):
        if False:
            print('Hello World!')
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, 'embed_dim'))
        self.parent.assertTrue(hasattr(config, 'num_heads'))

class CvtModelTester:

    def __init__(self, parent, batch_size=13, image_size=64, num_channels=3, embed_dim=[16, 32, 48], num_heads=[1, 2, 3], depth=[1, 2, 10], patch_sizes=[7, 3, 3], patch_stride=[4, 2, 2], patch_padding=[2, 1, 1], stride_kv=[2, 2, 2], cls_token=[False, False, True], attention_drop_rate=[0.0, 0.0, 0.0], initializer_range=0.02, layer_norm_eps=1e-12, is_training=True, use_labels=True, num_labels=2):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stride_kv = stride_kv
        self.depth = depth
        self.cls_token = cls_token
        self.attention_drop_rate = attention_drop_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

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
            while True:
                i = 10
        return CvtConfig(image_size=self.image_size, num_labels=self.num_labels, num_channels=self.num_channels, embed_dim=self.embed_dim, num_heads=self.num_heads, patch_sizes=self.patch_sizes, patch_padding=self.patch_padding, patch_stride=self.patch_stride, stride_kv=self.stride_kv, depth=self.depth, cls_token=self.cls_token, attention_drop_rate=self.attention_drop_rate, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            for i in range(10):
                print('nop')
        model = CvtModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        image_size = (self.image_size, self.image_size)
        (height, width) = (image_size[0], image_size[1])
        for i in range(len(self.depth)):
            height = floor((height + 2 * self.patch_padding[i] - self.patch_sizes[i]) / self.patch_stride[i] + 1)
            width = floor((width + 2 * self.patch_padding[i] - self.patch_sizes[i]) / self.patch_stride[i] + 1)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.embed_dim[-1], height, width))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            return 10
        config.num_labels = self.num_labels
        model = CvtForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class CvtModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Cvt does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (CvtModel, CvtForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': CvtModel, 'image-classification': CvtForImageClassification} if is_torch_available() else {}
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = CvtModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CvtConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        if False:
            print('Hello World!')
        return

    @unittest.skip(reason='Cvt does not output attentions')
    def test_attention_outputs(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Cvt does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Cvt does not support input and output embeddings')
    def test_model_common_attributes(self):
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
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        if False:
            while True:
                i = 10

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                for i in range(10):
                    print('nop')
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.hidden_states
            expected_num_layers = len(self.model_tester.depth)
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(list(hidden_states[0].shape[-3:]), [self.model_tester.embed_dim[0], self.model_tester.image_size // 4, self.model_tester.image_size // 4])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in CVT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CvtModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        while True:
            i = 10
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class CvtModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            i = 10
            return i + 15
        return AutoImageProcessor.from_pretrained(CVT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

    @slow
    def test_inference_image_classification_head(self):
        if False:
            for i in range(10):
                print('nop')
        model = CvtForImageClassification.from_pretrained(CVT_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([0.9285, 0.9015, -0.315]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))