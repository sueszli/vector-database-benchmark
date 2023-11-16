""" Testing suite for the PyTorch Audio Spectrogram Transformer (AST) model. """
import inspect
import unittest
from huggingface_hub import hf_hub_download
from transformers import ASTConfig
from transformers.testing_utils import require_torch, require_torchaudio, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_torchaudio_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import ASTForAudioClassification, ASTModel
    from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST
if is_torchaudio_available():
    import torchaudio
    from transformers import ASTFeatureExtractor

class ASTModelTester:

    def __init__(self, parent, batch_size=13, patch_size=2, max_length=24, num_mel_bins=16, is_training=True, use_labels=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, type_sequence_label_size=10, initializer_range=0.02, scope=None, frequency_stride=2, time_stride=2):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
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
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        frequency_out_dimension = (self.num_mel_bins - self.patch_size) // self.frequency_stride + 1
        time_out_dimension = (self.max_length - self.patch_size) // self.time_stride + 1
        num_patches = frequency_out_dimension * time_out_dimension
        self.seq_length = num_patches + 2

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        input_values = floats_tensor([self.batch_size, self.max_length, self.num_mel_bins])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        config = self.get_config()
        return (config, input_values, labels)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return ASTConfig(patch_size=self.patch_size, max_length=self.max_length, num_mel_bins=self.num_mel_bins, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range, frequency_stride=self.frequency_stride, time_stride=self.time_stride)

    def create_and_check_model(self, config, input_values, labels):
        if False:
            return 10
        model = ASTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_values, labels) = config_and_inputs
        inputs_dict = {'input_values': input_values}
        return (config, inputs_dict)

@require_torch
class ASTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as AST does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (ASTModel, ASTForAudioClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'audio-classification': ASTForAudioClassification, 'feature-extraction': ASTModel} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            while True:
                i = 10
        if pipeline_test_casse_name == 'AudioClassificationPipelineTests':
            return True
        return False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = ASTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ASTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.run_common_tests()

    @unittest.skip(reason='AST does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['input_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ASTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_audio():
    if False:
        for i in range(10):
            print('nop')
    filepath = hf_hub_download(repo_id='nielsr/audio-spectogram-transformer-checkpoint', filename='sample_audio.flac', repo_type='dataset')
    (audio, sampling_rate) = torchaudio.load(filepath)
    return (audio, sampling_rate)

@require_torch
@require_torchaudio
class ASTModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_feature_extractor(self):
        if False:
            while True:
                i = 10
        return ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593') if is_torchaudio_available() else None

    @slow
    def test_inference_audio_classification(self):
        if False:
            return 10
        feature_extractor = self.default_feature_extractor
        model = ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593').to(torch_device)
        feature_extractor = self.default_feature_extractor
        (audio, sampling_rate) = prepare_audio()
        audio = audio.squeeze().numpy()
        inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = torch.Size((1, 527))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.876, -7.0042, -8.6602]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001))