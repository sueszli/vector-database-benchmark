""" Testing suite for the PyTorch TimeSformer model. """
import copy
import inspect
import unittest
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import TimesformerConfig
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING, TimesformerForVideoClassification, TimesformerModel
    from transformers.models.timesformer.modeling_timesformer import TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from transformers import VideoMAEImageProcessor

class TimesformerModelTester:

    def __init__(self, parent, batch_size=13, image_size=10, num_channels=3, patch_size=2, num_frames=2, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, num_labels=10, initializer_range=0.02, attention_type='divided_space_time', scope=None):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.attention_type = attention_type
        self.initializer_range = initializer_range
        self.scope = scope
        self.num_labels = num_labels
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.seq_length = num_frames * self.num_patches_per_frame + 1

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        pixel_values = floats_tensor([self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            print('Hello World!')
        config = TimesformerConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, num_frames=self.num_frames, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, initializer_range=self.initializer_range, attention_type=self.attention_type)
        config.num_labels = self.num_labels
        return config

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            for i in range(10):
                print('nop')
        model = TimesformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_video_classification(self, config, pixel_values, labels):
        if False:
            print('Hello World!')
        model = TimesformerForVideoClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        expected_shape = torch.Size((self.batch_size, self.num_labels))
        self.parent.assertEqual(result.logits.shape, expected_shape)

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class TimesformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as TimeSformer does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (TimesformerModel, TimesformerForVideoClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': TimesformerModel, 'video-classification': TimesformerForVideoClassification} if is_torch_available() else {}
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = TimesformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimesformerConfig, has_text_modality=False, hidden_size=37)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        if False:
            print('Hello World!')
        inputs_dict = copy.deepcopy(inputs_dict)
        if return_labels:
            if model_class in get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING):
                inputs_dict['labels'] = torch.zeros(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
        return inputs_dict

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    @unittest.skip(reason='TimeSformer does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_video_classification(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_video_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_name in TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TimesformerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_attention_outputs(self):
        if False:
            return 10
        if not self.has_attentions:
            pass
        else:
            (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True
            for model_class in self.all_model_classes:
                seq_len = self.model_tester.seq_length
                num_frames = self.model_tester.num_frames
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
                self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_len // num_frames + 1, seq_len // num_frames + 1])
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
                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, seq_len // num_frames + 1, seq_len // num_frames + 1])

    def test_hidden_states_output(self):
        if False:
            i = 10
            return i + 15

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                i = 10
                return i + 15
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.hidden_states
            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)
            seq_length = self.model_tester.seq_length
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [seq_length, self.model_tester.hidden_size])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

def prepare_video():
    if False:
        for i in range(10):
            print('nop')
    file = hf_hub_download(repo_id='hf-internal-testing/spaghetti-video', filename='eating_spaghetti.npy', repo_type='dataset')
    video = np.load(file)
    return list(video)

@require_torch
@require_vision
class TimesformerModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            for i in range(10):
                print('nop')
        return VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]) if is_vision_available() else None

    @slow
    def test_inference_for_video_classification(self):
        if False:
            i = 10
            return i + 15
        model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k400').to(torch_device)
        image_processor = self.default_image_processor
        video = prepare_video()
        inputs = image_processor(video[:8], return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 400))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.3016, -0.7713, -0.4205]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))