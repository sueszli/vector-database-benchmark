""" Testing suite for the PyTorch MobileViT model. """
import inspect
import unittest
from transformers import MobileViTConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import MobileViTForImageClassification, MobileViTForSemanticSegmentation, MobileViTModel
    from transformers.models.mobilevit.modeling_mobilevit import MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import MobileViTImageProcessor

class MobileViTConfigTester(ConfigTester):

    def create_and_test_config_common_properties(self):
        if False:
            while True:
                i = 10
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, 'hidden_sizes'))
        self.parent.assertTrue(hasattr(config, 'neck_hidden_sizes'))
        self.parent.assertTrue(hasattr(config, 'num_attention_heads'))

class MobileViTModelTester:

    def __init__(self, parent, batch_size=13, image_size=32, patch_size=2, num_channels=3, last_hidden_size=32, num_attention_heads=4, hidden_act='silu', conv_kernel_size=3, output_stride=32, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, classifier_dropout_prob=0.1, initializer_range=0.02, is_training=True, use_labels=True, num_labels=10, scope=None):
        if False:
            print('Hello World!')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.last_hidden_size = last_hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.use_labels = use_labels
        self.is_training = is_training
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)
        config = self.get_config()
        return (config, pixel_values, labels, pixel_labels)

    def get_config(self):
        if False:
            return 10
        return MobileViTConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, num_attention_heads=self.num_attention_heads, hidden_act=self.hidden_act, conv_kernel_size=self.conv_kernel_size, output_stride=self.output_stride, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, classifier_dropout_prob=self.classifier_dropout_prob, initializer_range=self.initializer_range, hidden_sizes=[12, 16, 20], neck_hidden_sizes=[8, 8, 16, 16, 32, 32, 32])

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        if False:
            return 10
        model = MobileViTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.last_hidden_size, self.image_size // self.output_stride, self.image_size // self.output_stride))

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        if False:
            i = 10
            return i + 15
        config.num_labels = self.num_labels
        model = MobileViTForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        if False:
            for i in range(10):
                print('nop')
        config.num_labels = self.num_labels
        model = MobileViTForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels, self.image_size // self.output_stride, self.image_size // self.output_stride))
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels, self.image_size // self.output_stride, self.image_size // self.output_stride))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels, pixel_labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class MobileViTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MobileViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': MobileViTModel, 'image-classification': MobileViTForImageClassification, 'image-segmentation': MobileViTForSemanticSegmentation} if is_torch_available() else {}
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = MobileViTModelTester(self)
        self.config_tester = MobileViTConfigTester(self, config_class=MobileViTConfig, has_text_modality=False)

    def test_config(self):
        if False:
            while True:
                i = 10
        self.config_tester.run_common_tests()

    @unittest.skip(reason='MobileViT does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='MobileViT does not support input and output embeddings')
    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='MobileViT does not output attentions')
    def test_attention_outputs(self):
        if False:
            while True:
                i = 10
        pass

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

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        if False:
            while True:
                i = 10

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
            expected_num_stages = 5
            self.assertEqual(len(hidden_states), expected_num_stages)
            divisor = 2
            for i in range(len(hidden_states)):
                self.assertListEqual(list(hidden_states[i].shape[-2:]), [self.model_tester.image_size // divisor, self.model_tester.image_size // divisor])
                divisor *= 2
            self.assertEqual(self.model_tester.output_stride, divisor // 2)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MobileViTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        while True:
            i = 10
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class MobileViTModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            while True:
                i = 10
        return MobileViTImageProcessor.from_pretrained('apple/mobilevit-xx-small') if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            print('Hello World!')
        model = MobileViTForImageClassification.from_pretrained('apple/mobilevit-xx-small').to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-1.9364, -1.2327, -0.4653]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))

    @slow
    def test_inference_semantic_segmentation(self):
        if False:
            return 10
        model = MobileViTForSemanticSegmentation.from_pretrained('apple/deeplabv3-mobilevit-xx-small')
        model = model.to(torch_device)
        image_processor = MobileViTImageProcessor.from_pretrained('apple/deeplabv3-mobilevit-xx-small')
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        expected_shape = torch.Size((1, 21, 32, 32))
        self.assertEqual(logits.shape, expected_shape)
        expected_slice = torch.tensor([[[6.9713, 6.9786, 7.2422], [7.2893, 7.2825, 7.4446], [7.658, 7.8797, 7.942]], [[-10.6869, -10.325, -10.3471], [-10.4228, -9.9868, -9.7132], [-11.0405, -11.0221, -10.7318]], [[-3.3089, -2.8539, -2.674], [-3.2706, -2.5621, -2.5108], [-3.2534, -2.6615, -2.6651]]], device=torch_device)
        self.assertTrue(torch.allclose(logits[0, :3, :3, :3], expected_slice, atol=0.0001))

    @slow
    def test_post_processing_semantic_segmentation(self):
        if False:
            print('Hello World!')
        model = MobileViTForSemanticSegmentation.from_pretrained('apple/deeplabv3-mobilevit-xx-small')
        model = model.to(torch_device)
        image_processor = MobileViTImageProcessor.from_pretrained('apple/deeplabv3-mobilevit-xx-small')
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        outputs.logits = outputs.logits.detach().cpu()
        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[(50, 60)])
        expected_shape = torch.Size((50, 60))
        self.assertEqual(segmentation[0].shape, expected_shape)
        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = torch.Size((32, 32))
        self.assertEqual(segmentation[0].shape, expected_shape)