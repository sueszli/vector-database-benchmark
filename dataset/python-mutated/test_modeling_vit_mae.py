""" Testing suite for the PyTorch ViTMAE model. """
import inspect
import math
import tempfile
import unittest
import numpy as np
from transformers import ViTMAEConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import ViTMAEForPreTraining, ViTMAEModel
    from transformers.models.vit.modeling_vit import VIT_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import ViTImageProcessor

class ViTMAEModelTester:

    def __init__(self, parent, batch_size=13, image_size=30, patch_size=2, num_channels=3, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, type_sequence_label_size=10, initializer_range=0.02, num_labels=3, mask_ratio=0.6, scope=None):
        if False:
            while True:
                i = 10
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
        self.mask_ratio = mask_ratio
        self.scope = scope
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = int(math.ceil((1 - mask_ratio) * (num_patches + 1)))

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
            i = 10
            return i + 15
        return ViTMAEConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range, mask_ratio=self.mask_ratio, decoder_hidden_size=self.hidden_size, decoder_intermediate_size=self.intermediate_size, decoder_num_attention_heads=self.num_attention_heads, decoder_num_hidden_layers=self.num_hidden_layers)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            print('Hello World!')
        model = ViTMAEModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(self, config, pixel_values, labels):
        if False:
            return 10
        model = ViTMAEForPreTraining(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        num_patches = (self.image_size // self.patch_size) ** 2
        expected_num_channels = self.patch_size ** 2 * self.num_channels
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, expected_num_channels))
        config.num_channels = 1
        model = ViTMAEForPreTraining(config)
        model.to(torch_device)
        model.eval()
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        expected_num_channels = self.patch_size ** 2
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, expected_num_channels))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class ViTMAEModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViTMAE does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (ViTMAEModel, ViTMAEForPreTraining) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': ViTMAEModel} if is_torch_available() else {}
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = ViTMAEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTMAEConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.run_common_tests()

    @unittest.skip(reason='ViTMAE does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_model_common_attributes(self):
        if False:
            while True:
                i = 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

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
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def check_pt_tf_models(self, tf_model, pt_model, pt_inputs_dict):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2)
        num_patches = int((pt_model.config.image_size // pt_model.config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        pt_noise = torch.from_numpy(noise)
        pt_inputs_dict['noise'] = pt_noise
        super().check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)

    def test_save_load(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            torch.manual_seed(2)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                torch.manual_seed(2)
                with torch.no_grad():
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-05)

    @unittest.skip(reason='ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load\n    to get deterministic results.')
    def test_determinism(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load\n    to get deterministic results.')
    def test_save_load_fast_init_from_base(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load\n    to get deterministic results.')
    def test_save_load_fast_init_to_base(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load')
    def test_model_outputs_equivalence(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in VIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ViTMAEModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        for i in range(10):
            print('nop')
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class ViTMAEModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            for i in range(10):
                print('nop')
        return ViTImageProcessor.from_pretrained('facebook/vit-mae-base') if is_vision_available() else None

    @slow
    def test_inference_for_pretraining(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(2)
        model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base').to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        vit_mae_config = ViTMAEConfig()
        num_patches = int((vit_mae_config.image_size // vit_mae_config.patch_size) ** 2)
        noise = np.random.uniform(size=(1, num_patches))
        with torch.no_grad():
            outputs = model(**inputs, noise=torch.from_numpy(noise).to(device=torch_device))
        expected_shape = torch.Size((1, 196, 768))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([[-0.0548, -1.7023, -0.9325], [0.3721, -0.567, -0.2233], [0.8235, -1.3878, -0.3524]])
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice.to(torch_device), atol=0.0001))