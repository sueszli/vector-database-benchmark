""" Testing suite for the PyTorch ViT Hybrid model. """
import inspect
import unittest
from transformers import ViTHybridConfig
from transformers.testing_utils import require_accelerate, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import ViTHybridForImageClassification, ViTHybridImageProcessor, ViTHybridModel
    from transformers.models.vit_hybrid.modeling_vit_hybrid import VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image

class ViTHybridModelTester:

    def __init__(self, parent, batch_size=13, image_size=64, patch_size=2, num_channels=3, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, type_sequence_label_size=10, initializer_range=0.02, backbone_featmap_shape=[1, 16, 4, 4], scope=None):
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
        self.scope = scope
        self.backbone_featmap_shape = backbone_featmap_shape
        num_patches = (self.image_size // 32) ** 2
        self.seq_length = num_patches + 1

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
            for i in range(10):
                print('nop')
        backbone_config = {'global_padding': 'same', 'layer_type': 'bottleneck', 'depths': [3, 4, 9], 'out_features': ['stage1', 'stage2', 'stage3'], 'embedding_dynamic_padding': True, 'hidden_sizes': [4, 8, 16, 32], 'num_groups': 2}
        return ViTHybridConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range, backbone_featmap_shape=self.backbone_featmap_shape, backbone_config=backbone_config)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            print('Hello World!')
        model = ViTHybridModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            print('Hello World!')
        config.num_labels = self.type_sequence_label_size
        model = ViTHybridForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class ViTHybridModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (ViTHybridModel, ViTHybridForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': ViTHybridModel, 'image-classification': ViTHybridForImageClassification} if is_torch_available() else {}
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    model_split_percents = [0.5, 0.9]

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = ViTHybridModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTHybridConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    @unittest.skip(reason='ViT does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
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

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_initialization(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, module) in model.named_modules():
                if module.__class__.__name__ == 'ViTHybridPatchEmbeddings':
                    backbone_params = [f'{name}.{key}' for key in module.state_dict().keys()]
                    break
            for (name, param) in model.named_parameters():
                if param.requires_grad:
                    if name in backbone_params:
                        continue
                    self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ViTHybridModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        return 10
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class ViTModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            return 10
        return ViTHybridImageProcessor.from_pretrained(VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST[0]) if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            print('Hello World!')
        model = ViTHybridForImageClassification.from_pretrained(VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-1.909, -0.4993, -0.2389]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))

    @slow
    @require_accelerate
    def test_accelerate_inference(self):
        if False:
            while True:
                i = 10
        image_processor = ViTHybridImageProcessor.from_pretrained('google/vit-hybrid-base-bit-384')
        model = ViTHybridForImageClassification.from_pretrained('google/vit-hybrid-base-bit-384', device_map='auto')
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        self.assertTrue(model.config.id2label[predicted_class_idx], 'tabby, tabby cat')