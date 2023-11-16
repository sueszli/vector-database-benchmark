""" Testing suite for the PyTorch Hubert model. """
import math
import os
import pickle
import tempfile
import unittest
import pytest
from transformers import HubertConfig, is_torch_available
from transformers.testing_utils import require_soundfile, require_torch, slow, torch_device
from transformers.utils import is_torch_fx_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import HubertForCTC, HubertForSequenceClassification, HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    from transformers.models.hubert.modeling_hubert import _compute_mask_indices
if is_torch_fx_available():
    from transformers.utils.fx import symbolic_trace

class HubertModelTester:

    def __init__(self, parent, batch_size=13, seq_length=1024, is_training=False, hidden_size=16, feat_extract_norm='group', feat_extract_dropout=0.0, feat_extract_activation='gelu', conv_dim=(32, 32, 32), conv_stride=(4, 4, 4), conv_kernel=(8, 8, 8), conv_bias=False, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_hidden_layers=2, num_attention_heads=2, hidden_dropout_prob=0.1, intermediate_size=20, layer_norm_eps=1e-05, hidden_act='gelu', initializer_range=0.02, vocab_size=32, do_stable_layer_norm=False, scope=None):
        if False:
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()
        return (config, input_values, attention_mask)

    def get_config(self):
        if False:
            print('Hello World!')
        return HubertConfig(hidden_size=self.hidden_size, feat_extract_norm=self.feat_extract_norm, feat_extract_dropout=self.feat_extract_dropout, feat_extract_activation=self.feat_extract_activation, conv_dim=self.conv_dim, conv_stride=self.conv_stride, conv_kernel=self.conv_kernel, conv_bias=self.conv_bias, num_conv_pos_embeddings=self.num_conv_pos_embeddings, num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, hidden_dropout_prob=self.hidden_dropout_prob, intermediate_size=self.intermediate_size, layer_norm_eps=self.layer_norm_eps, hidden_act=self.hidden_act, initializer_range=self.initializer_range, vocab_size=self.vocab_size, do_stable_layer_norm=self.do_stable_layer_norm)

    def create_and_check_model(self, config, input_values, attention_mask):
        if False:
            i = 10
            return i + 15
        model = HubertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size))

    def create_and_check_batch_inference(self, config, input_values, *args):
        if False:
            while True:
                i = 10
        model = HubertModel(config=config)
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
            return 10
        model = HubertForCTC(config=config)
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
            i = 10
            return i + 15
        model = HubertForSequenceClassification(config=config)
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
            for i in range(10):
                print('nop')
        config.ctc_zero_infinity = True
        model = HubertForCTC(config=config)
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
            i = 10
            return i + 15
        config.ctc_zero_infinity = True
        model = HubertForSequenceClassification(config=config)
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
            while True:
                i = 10
        model = HubertForCTC(config)
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
            print('Hello World!')
        (config, input_values, attention_mask) = self.prepare_config_and_inputs()
        inputs_dict = {'input_values': input_values, 'attention_mask': attention_mask}
        return (config, inputs_dict)

@require_torch
class HubertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (HubertForCTC, HubertForSequenceClassification, HubertModel) if is_torch_available() else ()
    pipeline_model_mapping = {'audio-classification': HubertForSequenceClassification, 'automatic-speech-recognition': HubertForCTC, 'feature-extraction': HubertModel} if is_torch_available() else {}
    fx_compatible = True
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = HubertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HubertConfig, hidden_size=37)

    def test_config(self):
        if False:
            return 10
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_ctc_loss_inference(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_seq_classifier_loss_inference(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_ctc_train(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_seq_classifier_train(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_forward_signature(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_resize_tokens_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_model_common_attributes(self):
        if False:
            while True:
                i = 10
        pass

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            while True:
                i = 10
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
                uniform_init_parms = ['conv.weight', 'conv.parametrizations.weight', 'masked_spec_embed', 'quantizer.weight_proj.weight']
                if param.requires_grad:
                    if any((x in name for x in uniform_init_parms)):
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item() <= 1.0, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def _create_and_check_torch_fx_tracing(self, config, inputs_dict, output_loss=False):
        if False:
            i = 10
            return i + 15
        self.skipTest('torch 2.1 breaks torch fx tests for wav2vec2/hubert.')
        if not is_torch_fx_available() or not self.fx_compatible:
            return
        configs_no_init = _config_zero_init(config)
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=output_loss)
            try:
                if model.config.is_encoder_decoder:
                    model.config.use_cache = False
                    labels = inputs.get('labels', None)
                    input_names = ['attention_mask', 'decoder_attention_mask', 'decoder_input_ids', 'input_features', 'input_ids', 'input_values']
                    if labels is not None:
                        input_names.append('labels')
                    filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                    input_names = list(filtered_inputs.keys())
                    model_output = model(**filtered_inputs)
                    traced_model = symbolic_trace(model, input_names)
                    traced_output = traced_model(**filtered_inputs)
                else:
                    input_names = ['attention_mask', 'bbox', 'input_features', 'input_ids', 'input_values', 'pixel_values', 'token_type_ids', 'visual_feats', 'visual_pos']
                    labels = inputs.get('labels', None)
                    start_positions = inputs.get('start_positions', None)
                    end_positions = inputs.get('end_positions', None)
                    if labels is not None:
                        input_names.append('labels')
                    if start_positions is not None:
                        input_names.append('start_positions')
                    if end_positions is not None:
                        input_names.append('end_positions')
                    filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                    input_names = list(filtered_inputs.keys())
                    model_output = model(**filtered_inputs)
                    traced_model = symbolic_trace(model, input_names)
                    traced_output = traced_model(**filtered_inputs)
            except Exception as e:
                self.fail(f"Couldn't trace module: {e}")

            def flatten_output(output):
                if False:
                    i = 10
                    return i + 15
                flatten = []
                for x in output:
                    if isinstance(x, (tuple, list)):
                        flatten += flatten_output(x)
                    elif not isinstance(x, torch.Tensor):
                        continue
                    else:
                        flatten.append(x)
                return flatten
            model_output = flatten_output(model_output)
            traced_output = flatten_output(traced_output)
            num_outputs = len(model_output)
            for i in range(num_outputs):
                self.assertTrue(torch.allclose(model_output[i], traced_output[i]), f"traced {i}th output doesn't match model {i}th output for {model_class}")
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pkl_file_name = os.path.join(tmp_dir_name, 'model.pkl')
                try:
                    with open(pkl_file_name, 'wb') as f:
                        pickle.dump(traced_model, f)
                    with open(pkl_file_name, 'rb') as f:
                        loaded = pickle.load(f)
                except Exception as e:
                    self.fail(f"Couldn't serialize / deserialize the traced model: {e}")
                loaded_output = loaded(**filtered_inputs)
                loaded_output = flatten_output(loaded_output)
                for i in range(num_outputs):
                    self.assertTrue(torch.allclose(model_output[i], loaded_output[i]), f"serialized model {i}th output doesn't match model {i}th output for {model_class}")

    def _mock_init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, 'weight_g') and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, 'weight_v') and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, 'masked_spec_embed') and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    @unittest.skip(reason='Feed forward chunking is not implemented')
    def test_feed_forward_chunking(self):
        if False:
            while True:
                i = 10
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        model = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        self.assertIsNotNone(model)

@require_torch
class HubertRobustModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (HubertForCTC, HubertForSequenceClassification, HubertModel) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = HubertModelTester(self, conv_stride=(3, 3, 3), feat_extract_norm='layer', do_stable_layer_norm=True)
        self.config_tester = ConfigTester(self, config_class=HubertConfig, hidden_size=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
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
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_ctc_train(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_seq_classifier_train(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    def test_forward_signature(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_resize_tokens_embeddings(self):
        if False:
            print('Hello World!')
        pass

    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                uniform_init_parms = ['conv.weight', 'conv.parametrizations.weight', 'masked_spec_embed', 'quantizer.weight_proj.weight']
                if param.requires_grad:
                    if any((x in name for x in uniform_init_parms)):
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item() <= 1.0, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def _mock_init_weights(self, module):
        if False:
            return 10
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, 'weight_g') and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, 'weight_v') and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, 'masked_spec_embed') and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    @unittest.skip(reason='Feed forward chunking is not implemented')
    def test_feed_forward_chunking(self):
        if False:
            print('Hello World!')
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        model = HubertModel.from_pretrained('facebook/hubert-large-ls960-ft')
        self.assertIsNotNone(model)

@require_torch
class HubertUtilsTest(unittest.TestCase):

    def test_compute_mask_indices(self):
        if False:
            print('Hello World!')
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1
        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        mask = torch.from_numpy(mask).to(torch_device)
        self.assertListEqual(mask.sum(axis=-1).tolist(), [mask_prob * sequence_length for _ in range(batch_size)])

    def test_compute_mask_indices_overlap(self):
        if False:
            return 10
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4
        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        mask = torch.from_numpy(mask).to(torch_device)
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

@require_torch
@require_soundfile
@slow
class HubertModelIntegrationTest(unittest.TestCase):

    def _load_datasamples(self, num_samples):
        if False:
            print('Hello World!')
        from datasets import load_dataset
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').filter(lambda x: x['id'] in [f'1272-141231-000{i}' for i in range(num_samples)])[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def _load_superb(self, task, num_samples):
        if False:
            while True:
                i = 10
        from datasets import load_dataset
        ds = load_dataset('anton-l/superb_dummy', task, split='test')
        return ds[:num_samples]

    def test_inference_ctc_batched(self):
        if False:
            for i in range(10):
                print('nop')
        model = HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft', torch_dtype=torch.float16).to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft', do_lower_case=True)
        input_speech = self._load_datasamples(2)
        inputs = processor(input_speech, return_tensors='pt', padding=True)
        input_values = inputs.input_values.half().to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)
        EXPECTED_TRANSCRIPTIONS = ['a man said to the universe sir i exist', "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_keyword_spotting(self):
        if False:
            print('Hello World!')
        model = HubertForSequenceClassification.from_pretrained('superb/hubert-base-superb-ks', torch_dtype=torch.float16).to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained('superb/hubert-base-superb-ks')
        input_data = self._load_superb('ks', 4)
        inputs = processor(input_data['speech'], return_tensors='pt', padding=True)
        input_values = inputs.input_values.half().to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        (predicted_logits, predicted_ids) = torch.max(outputs.logits, dim=-1)
        expected_labels = [2, 6, 10, 9]
        expected_logits = torch.tensor([7.6692, 17.7795, 11.1562, 11.8232], dtype=torch.float16, device=torch_device)
        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        self.assertTrue(torch.allclose(predicted_logits, expected_logits, atol=0.03))

    def test_inference_intent_classification(self):
        if False:
            print('Hello World!')
        model = HubertForSequenceClassification.from_pretrained('superb/hubert-base-superb-ic', torch_dtype=torch.float16).to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained('superb/hubert-base-superb-ic')
        input_data = self._load_superb('ic', 4)
        inputs = processor(input_data['speech'], return_tensors='pt', padding=True)
        input_values = inputs.input_values.half().to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        (predicted_logits_action, predicted_ids_action) = torch.max(outputs.logits[:, :6], dim=-1)
        (predicted_logits_object, predicted_ids_object) = torch.max(outputs.logits[:, 6:20], dim=-1)
        (predicted_logits_location, predicted_ids_location) = torch.max(outputs.logits[:, 20:24], dim=-1)
        expected_labels_action = [1, 0, 4, 3]
        expected_logits_action = torch.tensor([5.9052, 12.5865, 4.484, 10.024], dtype=torch.float16, device=torch_device)
        expected_labels_object = [1, 10, 3, 4]
        expected_logits_object = torch.tensor([5.5316, 11.7946, 8.1672, 23.2415], dtype=torch.float16, device=torch_device)
        expected_labels_location = [0, 0, 0, 1]
        expected_logits_location = torch.tensor([5.2053, 8.9577, 10.0447, 8.1481], dtype=torch.float16, device=torch_device)
        self.assertListEqual(predicted_ids_action.tolist(), expected_labels_action)
        self.assertListEqual(predicted_ids_object.tolist(), expected_labels_object)
        self.assertListEqual(predicted_ids_location.tolist(), expected_labels_location)
        self.assertTrue(torch.allclose(predicted_logits_action, expected_logits_action, atol=0.3))
        self.assertTrue(torch.allclose(predicted_logits_object, expected_logits_object, atol=0.3))
        self.assertTrue(torch.allclose(predicted_logits_location, expected_logits_location, atol=0.3))

    def test_inference_speaker_identification(self):
        if False:
            i = 10
            return i + 15
        model = HubertForSequenceClassification.from_pretrained('superb/hubert-base-superb-sid', torch_dtype=torch.float16).to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained('superb/hubert-base-superb-sid')
        input_data = self._load_superb('si', 4)
        output_logits = []
        with torch.no_grad():
            for example in input_data['speech']:
                input = processor(example, return_tensors='pt', padding=True)
                output = model(input.input_values.half().to(torch_device), attention_mask=None)
                output_logits.append(output.logits[0])
        output_logits = torch.stack(output_logits)
        (predicted_logits, predicted_ids) = torch.max(output_logits, dim=-1)
        expected_labels = [5, 1, 1, 3]
        expected_logits = torch.tensor([78231.5547, 123166.6094, 122785.4141, 84851.2969], dtype=torch.float16, device=torch_device)
        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        self.assertTrue(torch.allclose(predicted_logits, expected_logits, atol=10))

    def test_inference_emotion_recognition(self):
        if False:
            i = 10
            return i + 15
        model = HubertForSequenceClassification.from_pretrained('superb/hubert-base-superb-er', torch_dtype=torch.float16).to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained('superb/hubert-base-superb-er')
        input_data = self._load_superb('er', 4)
        inputs = processor(input_data['speech'], return_tensors='pt', padding=True)
        input_values = inputs.input_values.half().to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        (predicted_logits, predicted_ids) = torch.max(outputs.logits, dim=-1)
        expected_labels = [1, 1, 2, 2]
        expected_logits = torch.tensor([2.8384, 2.3389, 3.8564, 4.5558], dtype=torch.float16, device=torch_device)
        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        self.assertTrue(torch.allclose(predicted_logits, expected_logits, atol=0.1))

    def test_inference_distilhubert(self):
        if False:
            i = 10
            return i + 15
        model = HubertModel.from_pretrained('ntu-spml/distilhubert').to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained('ntu-spml/distilhubert')
        input_speech = self._load_datasamples(1)
        inputs = processor(input_speech, return_tensors='pt', padding=True)
        input_values = inputs.input_values.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values).last_hidden_state
        expected_outputs_first = torch.tensor([[[-0.3505, 0.1167, 0.0608, 0.1294], [-0.3085, 0.0481, 0.1106, 0.0955], [-0.3107, -0.0391, 0.0739, 0.136], [-0.2385, -0.1795, -0.0928, 0.2389]]], device=torch_device)
        expected_outputs_last = torch.tensor([[[-0.0732, 0.0255, 0.0529, -0.1372], [-0.0812, 0.1259, 0.0564, -0.0438], [-0.0054, 0.0758, -0.0002, -0.1617], [0.0133, -0.032, -0.0687, 0.0062]]], device=torch_device)
        expected_output_sum = -3776.073
        self.assertTrue(torch.allclose(outputs[:, :4, :4], expected_outputs_first, atol=0.005))
        self.assertTrue(torch.allclose(outputs[:, -4:, -4:], expected_outputs_last, atol=0.005))
        self.assertTrue(abs(outputs.sum() - expected_output_sum) < 0.1)