""" Testing suite for the PyTorch YOLOS model. """
import inspect
import unittest
from transformers import YolosConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import YolosForObjectDetection, YolosModel
    from transformers.models.yolos.modeling_yolos import YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

class YolosModelTester:

    def __init__(self, parent, batch_size=13, image_size=[30, 30], patch_size=2, num_channels=3, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, type_sequence_label_size=10, initializer_range=0.02, num_labels=3, scope=None, n_targets=8, num_detection_tokens=10):
        if False:
            return 10
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
        self.num_labels = num_labels
        self.scope = scope
        self.n_targets = n_targets
        self.num_detection_tokens = num_detection_tokens
        num_patches = image_size[1] // patch_size * (image_size[0] // patch_size)
        self.expected_seq_len = num_patches + 1 + self.num_detection_tokens

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        labels = None
        if self.use_labels:
            labels = []
            for i in range(self.batch_size):
                target = {}
                target['class_labels'] = torch.randint(high=self.num_labels, size=(self.n_targets,), device=torch_device)
                target['boxes'] = torch.rand(self.n_targets, 4, device=torch_device)
                labels.append(target)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return YolosConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range, num_detection_tokens=self.num_detection_tokens, num_labels=self.num_labels)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            for i in range(10):
                print('nop')
        model = YolosModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.expected_seq_len, self.hidden_size))

    def create_and_check_for_object_detection(self, config, pixel_values, labels):
        if False:
            return 10
        model = YolosForObjectDetection(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values=pixel_values)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_detection_tokens, self.num_labels + 1))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_detection_tokens, 4))
        result = model(pixel_values=pixel_values, labels=labels)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_detection_tokens, self.num_labels + 1))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_detection_tokens, 4))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class YolosModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as YOLOS does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (YolosModel, YolosForObjectDetection) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': YolosModel, 'object-detection': YolosForObjectDetection} if is_torch_available() else {}
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        if False:
            i = 10
            return i + 15
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            if model_class.__name__ == 'YolosForObjectDetection':
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target['class_labels'] = torch.ones(size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long)
                    target['boxes'] = torch.ones(self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float)
                    labels.append(target)
                inputs_dict['labels'] = labels
        return inputs_dict

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = YolosModelTester(self)
        self.config_tester = ConfigTester(self, config_class=YolosConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
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
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = self.model_tester.expected_seq_len
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
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))
            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_len, seq_len])

    def test_hidden_states_output(self):
        if False:
            return 10

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                return 10
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.hidden_states
            expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
            self.assertEqual(len(hidden_states), expected_num_layers)
            seq_length = self.model_tester.expected_seq_len
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [seq_length, self.model_tester.hidden_size])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_object_detection(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_object_detection(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            print('Hello World!')
        for model_name in YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = YolosModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        i = 10
        return i + 15
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class YolosModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            return 10
        return AutoImageProcessor.from_pretrained('hustvl/yolos-small') if is_vision_available() else None

    @slow
    def test_inference_object_detection_head(self):
        if False:
            while True:
                i = 10
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small').to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(inputs.pixel_values)
        expected_shape = torch.Size((1, 100, 92))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice_logits = torch.tensor([[-24.0248, -10.3024, -14.829], [-42.0392, -16.82, -27.4334], [-27.2743, -11.8154, -18.7148]], device=torch_device)
        expected_slice_boxes = torch.tensor([[0.2559, 0.5455, 0.4706], [0.2989, 0.7279, 0.1875], [0.7732, 0.4017, 0.4462]], device=torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=0.0001))
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=0.0001))
        results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=[image.size[::-1]])[0]
        expected_scores = torch.tensor([0.9994, 0.979, 0.9964, 0.9972, 0.9861]).to(torch_device)
        expected_labels = [75, 75, 17, 63, 17]
        expected_slice_boxes = torch.tensor([335.0609, 79.3848, 375.4216, 187.2495]).to(torch_device)
        self.assertEqual(len(results['scores']), 5)
        self.assertTrue(torch.allclose(results['scores'], expected_scores, atol=0.0001))
        self.assertSequenceEqual(results['labels'].tolist(), expected_labels)
        self.assertTrue(torch.allclose(results['boxes'][0, :], expected_slice_boxes))