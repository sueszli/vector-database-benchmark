""" Testing suite for the PyTorch UniSpeech model. """
import math
import unittest
import numpy as np
import pytest
from datasets import load_dataset
from transformers import UniSpeechConfig, is_torch_available
from transformers.testing_utils import require_soundfile, require_torch, slow, torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import UniSpeechForCTC, UniSpeechForPreTraining, UniSpeechForSequenceClassification, UniSpeechModel, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

class UniSpeechModelTester:

    def __init__(self, parent, batch_size=13, seq_length=1024, is_training=False, hidden_size=16, feat_extract_norm='group', feat_extract_dropout=0.0, feat_extract_activation='gelu', conv_dim=(32, 32, 32), conv_stride=(4, 4, 4), conv_kernel=(8, 8, 8), conv_bias=False, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_hidden_layers=2, num_attention_heads=2, hidden_dropout_prob=0.1, intermediate_size=20, layer_norm_eps=1e-05, hidden_act='gelu', initializer_range=0.02, vocab_size=32, do_stable_layer_norm=False, scope=None):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.scope = scope
        output_seq_length = self.seq_length
        for (kernel, stride) in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()
        return (config, input_values, attention_mask)

    def get_config(self):
        if False:
            print('Hello World!')
        return UniSpeechConfig(hidden_size=self.hidden_size, feat_extract_norm=self.feat_extract_norm, feat_extract_dropout=self.feat_extract_dropout, feat_extract_activation=self.feat_extract_activation, conv_dim=self.conv_dim, conv_stride=self.conv_stride, conv_kernel=self.conv_kernel, conv_bias=self.conv_bias, num_conv_pos_embeddings=self.num_conv_pos_embeddings, num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, hidden_dropout_prob=self.hidden_dropout_prob, intermediate_size=self.intermediate_size, layer_norm_eps=self.layer_norm_eps, hidden_act=self.hidden_act, initializer_range=self.initializer_range, vocab_size=self.vocab_size)

    def create_and_check_model(self, config, input_values, attention_mask):
        if False:
            for i in range(10):
                print('nop')
        model = UniSpeechModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size))

    def create_and_check_batch_inference(self, config, input_values, *args):
        if False:
            return 10
        model = UniSpeechModel(config=config)
        model.to(torch_device)
        model.eval()
        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.bool)
        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i]:] = 0.0
            attention_mask[i, input_lengths[i]:] = 0.0
        batch_outputs = model(input_values, attention_mask=attention_mask).last_hidden_state
        for i in range(input_values.shape[0]):
            input_slice = input_values[i:i + 1, :input_lengths[i]]
            output = model(input_slice).last_hidden_state
            batch_output = batch_outputs[i:i + 1, :output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=0.001))

    def check_ctc_loss(self, config, input_values, *args):
        if False:
            print('Hello World!')
        model = UniSpeechForCTC(config=config)
        model.to(torch_device)
        model.eval()
        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)
        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i]:] = 0.0
            attention_mask[i, input_lengths[i]:] = 0
        model.config.ctc_loss_reduction = 'sum'
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()
        model.config.ctc_loss_reduction = 'mean'
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()
        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_seq_classifier_loss(self, config, input_values, *args):
        if False:
            while True:
                i = 10
        model = UniSpeechForSequenceClassification(config=config)
        model.to(torch_device)
        model.eval()
        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)
        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i]:] = 0.0
            attention_mask[i, input_lengths[i]:] = 0
        masked_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()
        unmasked_loss = model(input_values, labels=labels).loss.item()
        self.parent.assertTrue(isinstance(masked_loss, float))
        self.parent.assertTrue(isinstance(unmasked_loss, float))
        self.parent.assertTrue(masked_loss != unmasked_loss)

    def check_ctc_training(self, config, input_values, *args):
        if False:
            return 10
        config.ctc_zero_infinity = True
        model = UniSpeechForCTC(config=config)
        model.to(torch_device)
        model.train()
        model.freeze_feature_encoder()
        input_values = input_values[:3]
        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i]:] = 0.0
            if max_length_labels[i] < labels.shape[-1]:
                labels[i, max_length_labels[i] - 1:] = -100
        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())
        loss.backward()

    def check_seq_classifier_training(self, config, input_values, *args):
        if False:
            print('Hello World!')
        config.ctc_zero_infinity = True
        model = UniSpeechForSequenceClassification(config=config)
        model.to(torch_device)
        model.train()
        model.freeze_base_model()
        input_values = input_values[:3]
        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i]:] = 0.0
        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())
        loss.backward()

    def check_labels_out_of_vocab(self, config, input_values, *args):
        if False:
            i = 10
            return i + 15
        model = UniSpeechForCTC(config)
        model.to(torch_device)
        model.train()
        input_values = input_values[:3]
        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)
        with pytest.raises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        (config, input_values, attention_mask) = self.prepare_config_and_inputs()
        inputs_dict = {'input_values': input_values, 'attention_mask': attention_mask}
        return (config, inputs_dict)

@require_torch
class UniSpeechRobustModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (UniSpeechForCTC, UniSpeechModel, UniSpeechForSequenceClassification, UniSpeechForPreTraining) if is_torch_available() else ()
    pipeline_model_mapping = {'audio-classification': UniSpeechForSequenceClassification, 'automatic-speech-recognition': UniSpeechForCTC, 'feature-extraction': UniSpeechModel} if is_torch_available() else {}
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = UniSpeechModelTester(self, conv_stride=(3, 3, 3), feat_extract_norm='layer', do_stable_layer_norm=True)
        self.config_tester = ConfigTester(self, config_class=UniSpeechConfig, hidden_size=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_batched_inference(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_batch_inference(*config_and_inputs)

    def test_ctc_loss_inference(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_seq_classifier_loss_inference(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_ctc_train(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_seq_classifier_train(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    def test_inputs_embeds(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_forward_signature(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_resize_tokens_embeddings(self):
        if False:
            return 10
        pass

    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)
        model.config.layerdrop = 0.0
        input_values = inputs_dict['input_values']
        input_lengths = torch.tensor([input_values.shape[1] for _ in range(input_values.shape[0])], dtype=torch.long, device=torch_device)
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
        labels = ids_tensor((input_values.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
        inputs_dict['attention_mask'] = torch.ones_like(inputs_dict['attention_mask'])
        inputs_dict['labels'] = labels
        outputs = model(**inputs_dict)
        output = outputs[0]
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]
        hidden_states.retain_grad()
        attentions.retain_grad()
        output.flatten()[0].backward(retain_graph=True)
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_initialization(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                uniform_init_parms = ['conv.weight', 'conv.parametrizations.weight', 'masked_spec_embed', 'codevectors', 'quantizer.weight_proj.weight', 'project_hid.weight', 'project_hid.bias', 'project_q.weight', 'project_q.bias', 'feature_projection.projection.weight', 'feature_projection.projection.bias']
                if param.requires_grad:
                    if any((x in name for x in uniform_init_parms)):
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item() <= 1.0, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def _mock_init_weights(self, module):
        if False:
            print('Hello World!')
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, 'weight_g') and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, 'weight_v') and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, 'codevectors') and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, 'masked_spec_embed') and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    def test_mask_feature_prob_ctc(self):
        if False:
            i = 10
            return i + 15
        model = UniSpeechForCTC.from_pretrained('hf-internal-testing/tiny-random-unispeech', mask_feature_prob=0.2, mask_feature_length=2)
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained('hf-internal-testing/tiny-random-unispeech', return_attention_mask=True)
        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16000 * s) for s in batch_duration_in_seconds]
        batch = processor(input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors='pt')
        logits = model(input_values=batch['input_values'].to(torch_device), attention_mask=batch['attention_mask'].to(torch_device)).logits
        self.assertEqual(logits.shape, (4, 1498, 32))

    def test_mask_time_prob_ctc(self):
        if False:
            while True:
                i = 10
        model = UniSpeechForCTC.from_pretrained('hf-internal-testing/tiny-random-unispeech', mask_time_prob=0.2, mask_time_length=2)
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained('hf-internal-testing/tiny-random-unispeech', return_attention_mask=True)
        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16000 * s) for s in batch_duration_in_seconds]
        batch = processor(input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors='pt')
        logits = model(input_values=batch['input_values'].to(torch_device), attention_mask=batch['attention_mask'].to(torch_device)).logits
        self.assertEqual(logits.shape, (4, 1498, 32))

    def test_mask_time_feature_prob_ctc_single_batch(self):
        if False:
            i = 10
            return i + 15
        model = UniSpeechForCTC.from_pretrained('hf-internal-testing/tiny-random-unispeech', mask_time_prob=0.2, mask_feature_prob=0.2, mask_time_length=2, mask_feature_length=2)
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained('hf-internal-testing/tiny-random-unispeech', return_attention_mask=True)
        batch_duration_in_seconds = [6]
        input_features = [np.random.random(16000 * s) for s in batch_duration_in_seconds]
        batch = processor(input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors='pt')
        logits = model(input_values=batch['input_values'].to(torch_device), attention_mask=batch['attention_mask'].to(torch_device)).logits
        self.assertEqual(logits.shape, (1, 1498, 32))

    @unittest.skip(reason='Feed forward chunking is not implemented')
    def test_feed_forward_chunking(self):
        if False:
            print('Hello World!')
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        model = UniSpeechModel.from_pretrained('microsoft/unispeech-large-1500h-cv')
        self.assertIsNotNone(model)

@require_torch
@require_soundfile
@slow
class UniSpeechModelIntegrationTest(unittest.TestCase):

    def _load_datasamples(self, num_samples):
        if False:
            print('Hello World!')
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').filter(lambda x: x['id'] in [f'1272-141231-000{i}' for i in range(num_samples)])[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def _load_superb(self, task, num_samples):
        if False:
            for i in range(10):
                print('nop')
        ds = load_dataset('anton-l/superb_dummy', task, split='test')
        return ds[:num_samples]

    def test_inference_pretraining(self):
        if False:
            i = 10
            return i + 15
        model = UniSpeechForPreTraining.from_pretrained('microsoft/unispeech-large-1500h-cv')
        model.to(torch_device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        input_speech = self._load_datasamples(2)
        inputs_dict = feature_extractor(input_speech, return_tensors='pt', padding=True)
        with torch.no_grad():
            torch.manual_seed(0)
            outputs = model(inputs_dict.input_values.to(torch_device), attention_mask=inputs_dict.attention_mask.to(torch_device))
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
        self.assertTrue(cosine_sim.mean() > 0.5)
        expected_cosine_sim_slice = torch.tensor([[0.829, 0.8335, 0.8815, 0.858, 0.8249], [0.8892, 0.9221, 0.8711, 0.8601, 0.8482]], device=torch_device)
        self.assertTrue(torch.allclose(cosine_sim[:, :5], expected_cosine_sim_slice, atol=0.001))