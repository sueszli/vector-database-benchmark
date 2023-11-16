""" Testing suite for the PyTorch SpeechT5 model. """
import copy
import inspect
import tempfile
import unittest
from transformers import SpeechT5Config, SpeechT5HifiGanConfig
from transformers.testing_utils import is_torch_available, require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.trainer_utils import set_seed
from transformers.utils import cached_property
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import SpeechT5ForSpeechToSpeech, SpeechT5ForSpeechToText, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Model, SpeechT5Processor

def prepare_inputs_dict(config, input_ids=None, input_values=None, decoder_input_ids=None, decoder_input_values=None, attention_mask=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None):
    if False:
        for i in range(10):
            print('nop')
    if input_ids is not None:
        encoder_dict = {'input_ids': input_ids}
    else:
        encoder_dict = {'input_values': input_values}
    if decoder_input_ids is not None:
        decoder_dict = {'decoder_input_ids': decoder_input_ids}
    else:
        decoder_dict = {'decoder_input_values': decoder_input_values}
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    return {**encoder_dict, **decoder_dict, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask}

@require_torch
class SpeechT5ModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=False, vocab_size=81, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=4):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        input_values = floats_tensor([self.batch_size, self.seq_length, self.hidden_size], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        decoder_input_values = floats_tensor([self.batch_size, self.seq_length, self.hidden_size], scale=1.0)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()
        inputs_dict = prepare_inputs_dict(config, input_values=input_values, decoder_input_values=decoder_input_values, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.prepare_config_and_inputs()
        return (config, inputs_dict)

    def get_config(self):
        if False:
            return 10
        return SpeechT5Config(vocab_size=self.vocab_size, hidden_size=self.hidden_size, encoder_layers=self.num_hidden_layers, decoder_layers=self.num_hidden_layers, encoder_attention_heads=self.num_attention_heads, decoder_attention_heads=self.num_attention_heads, encoder_ffn_dim=self.intermediate_size, decoder_ffn_dim=self.intermediate_size)

    def create_and_check_model_forward(self, config, inputs_dict):
        if False:
            print('Hello World!')
        model = SpeechT5Model(config=config).to(torch_device).eval()
        input_values = inputs_dict['input_values']
        attention_mask = inputs_dict['attention_mask']
        decoder_input_values = inputs_dict['decoder_input_values']
        result = model(input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

@require_torch
class SpeechT5ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5Model,) if is_torch_available() else ()
    pipeline_model_mapping = {'automatic-speech-recognition': SpeechT5ForSpeechToText, 'feature-extraction': SpeechT5Model} if is_torch_available() else {}
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    input_name = 'input_values'

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = SpeechT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        if False:
            while True:
                i = 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['input_values', 'attention_mask', 'decoder_input_values', 'decoder_attention_mask']
            expected_arg_names.extend(['head_mask', 'decoder_head_mask', 'cross_attn_head_mask', 'encoder_outputs'] if 'head_mask' and 'decoder_head_mask' and ('cross_attn_head_mask' in arg_names) else ['encoder_outputs'])
            self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)

    def test_inputs_embeds(self):
        if False:
            return 10
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
        pass

    @slow
    def test_torchscript_output_attentions(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    def test_torchscript_simple(self):
        if False:
            i = 10
            return i + 15
        pass

@require_torch
class SpeechT5ForSpeechToTextTester:

    def __init__(self, parent, batch_size=13, encoder_seq_length=1024, decoder_seq_length=7, is_training=False, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=4, conv_dim=(32, 32, 32), conv_stride=(4, 4, 4), conv_kernel=(8, 8, 8), conv_bias=False, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, vocab_size=81):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        if False:
            return 10
        input_values = floats_tensor([self.batch_size, self.encoder_seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size).clamp(2)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])
        config = self.get_config()
        inputs_dict = prepare_inputs_dict(config, input_values=input_values, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.prepare_config_and_inputs()
        return (config, inputs_dict)

    def get_config(self):
        if False:
            while True:
                i = 10
        return SpeechT5Config(hidden_size=self.hidden_size, encoder_layers=self.num_hidden_layers, decoder_layers=self.num_hidden_layers, encoder_attention_heads=self.num_attention_heads, decoder_attention_heads=self.num_attention_heads, encoder_ffn_dim=self.intermediate_size, decoder_ffn_dim=self.intermediate_size, conv_dim=self.conv_dim, conv_stride=self.conv_stride, conv_kernel=self.conv_kernel, conv_bias=self.conv_bias, num_conv_pos_embeddings=self.num_conv_pos_embeddings, num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups, vocab_size=self.vocab_size)

    def create_and_check_model_forward(self, config, inputs_dict):
        if False:
            for i in range(10):
                print('nop')
        model = SpeechT5ForSpeechToText(config=config).to(torch_device).eval()
        input_values = inputs_dict['input_values']
        attention_mask = inputs_dict['attention_mask']
        decoder_input_ids = inputs_dict['decoder_input_ids']
        result = model(input_values, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.decoder_seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        if False:
            while True:
                i = 10
        model = SpeechT5ForSpeechToText(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict['decoder_input_ids']
        attention_mask = inputs_dict['decoder_attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        (output, past_key_values) = outputs.to_tuple()
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size).clamp(2)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)
        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)['last_hidden_state']
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)['last_hidden_state']
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()
        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=0.01))

@require_torch
class SpeechT5ForSpeechToTextTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5ForSpeechToText,) if is_torch_available() else ()
    all_generative_model_classes = (SpeechT5ForSpeechToText,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    input_name = 'input_values'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = SpeechT5ForSpeechToTextTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                (model2, info) = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info['missing_keys'], [])

    def test_model_forward(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_attention_outputs(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = getattr(self.model_tester, 'seq_length', None)
        decoder_seq_length = getattr(self.model_tester, 'decoder_seq_length', seq_len)
        encoder_seq_length = getattr(self.model_tester, 'encoder_seq_length', seq_len)
        decoder_key_length = getattr(self.model_tester, 'decoder_key_length', decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, 'key_length', encoder_seq_length)
        for model_class in self.all_model_classes:
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            subsampled_encoder_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(encoder_key_length)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length])
            out_len = len(outputs)
            correct_outlen = 5
            if 'labels' in inputs_dict:
                correct_outlen += 1
            if 'past_key_values' in outputs:
                correct_outlen += 1
            self.assertEqual(out_len, correct_outlen)
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(decoder_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length])
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(cross_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, decoder_seq_length, subsampled_encoder_key_length])
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))
            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length])

    def test_forward_signature(self):
        if False:
            return 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['input_values', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask']
            expected_arg_names.extend(['head_mask', 'decoder_head_mask', 'cross_attn_head_mask', 'encoder_outputs'] if 'head_mask' and 'decoder_head_mask' and ('cross_attn_head_mask' in arg_names) else ['encoder_outputs'])
            self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)

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
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
            self.assertEqual(len(hidden_states), expected_num_layers)
            if hasattr(self.model_tester, 'encoder_seq_length'):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length
            subsampled_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(seq_length)
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [subsampled_seq_length, self.model_tester.hidden_size])
            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states
                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, 'seq_length', None)
                decoder_seq_length = getattr(self.model_tester, 'decoder_seq_length', seq_len)
                self.assertListEqual(list(hidden_states[0].shape[-2:]), [decoder_seq_length, self.model_tester.hidden_size])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                uniform_init_parms = ['conv.weight', 'conv.parametrizations.weight', 'masked_spec_embed', 'feature_projection.projection.weight', 'feature_projection.projection.bias']
                if param.requires_grad:
                    if any((x in name for x in uniform_init_parms)):
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item() <= 1.0, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_resize_embeddings_untied(self):
        if False:
            print('Hello World!')
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return
        original_config.tie_word_embeddings = False
        if original_config.tie_word_embeddings:
            return
        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)
            if model.get_output_embeddings() is None:
                continue
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            model(**self._prepare_for_class(inputs_dict, model_class))
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            if 'decoder_input_ids' in inputs_dict:
                inputs_dict['decoder_input_ids'].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

    def test_resize_tokens_embeddings(self):
        if False:
            print('Hello World!')
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return
        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            if self.model_tester.is_training is False:
                model.eval()
            model_vocab_size = config.vocab_size
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            model(**self._prepare_for_class(inputs_dict, model_class))
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)
            if 'decoder_input_ids' in inputs_dict:
                inputs_dict['decoder_input_ids'].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))
            models_equal = True
            for (p1, p2) in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False
            self.assertTrue(models_equal)

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            print('Hello World!')
        pass

    def test_training(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_training_gradient_checkpointing(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            i = 10
            return i + 15
        pass

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
        if hasattr(module, 'masked_spec_embed') and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class SpeechT5ForSpeechToTextIntegrationTests(unittest.TestCase):

    @cached_property
    def default_processor(self):
        if False:
            print('Hello World!')
        return SpeechT5Processor.from_pretrained('microsoft/speecht5_asr')

    def _load_datasamples(self, num_samples):
        if False:
            print('Hello World!')
        from datasets import load_dataset
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').select(range(num_samples))[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def test_generation_librispeech(self):
        if False:
            print('Hello World!')
        model = SpeechT5ForSpeechToText.from_pretrained('microsoft/speecht5_asr')
        model.to(torch_device)
        processor = self.default_processor
        input_speech = self._load_datasamples(1)
        input_values = processor(audio=input_speech, return_tensors='pt').input_values.to(torch_device)
        generated_ids = model.generate(input_values)
        generated_transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        EXPECTED_TRANSCRIPTIONS = ['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
        self.assertListEqual(generated_transcript, EXPECTED_TRANSCRIPTIONS)

    def test_generation_librispeech_batched(self):
        if False:
            print('Hello World!')
        model = SpeechT5ForSpeechToText.from_pretrained('microsoft/speecht5_asr')
        model.to(torch_device)
        processor = self.default_processor
        input_speech = self._load_datasamples(4)
        inputs = processor(audio=input_speech, return_tensors='pt', padding=True)
        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        generated_ids = model.generate(input_values, attention_mask=attention_mask)
        generated_transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        EXPECTED_TRANSCRIPTIONS = ['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel', "nor is mister quilter's manner less interesting than his matter", 'he tells us that at this festive season of the year with christmas and rosebeaf looming before us similars drawn from eating and its results occur most readily to the mind', "he has grave doubts whether sir frederick latin's work is really greek after all and can discover in it but little of rocky ithica"]
        self.assertListEqual(generated_transcripts, EXPECTED_TRANSCRIPTIONS)

@require_torch
class SpeechT5ForTextToSpeechTester:

    def __init__(self, parent, batch_size=13, encoder_seq_length=7, decoder_seq_length=1024, is_training=False, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=4, vocab_size=81, num_mel_bins=20, reduction_factor=2, speech_decoder_postnet_layers=2, speech_decoder_postnet_units=32, speech_decoder_prenet_units=32):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.reduction_factor = reduction_factor
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_prenet_units = speech_decoder_prenet_units

    def prepare_config_and_inputs(self):
        if False:
            return 10
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size).clamp(2)
        attention_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])
        decoder_input_values = floats_tensor([self.batch_size, self.decoder_seq_length, self.num_mel_bins], scale=1.0)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])
        config = self.get_config()
        inputs_dict = prepare_inputs_dict(config, input_ids=input_ids, decoder_input_values=decoder_input_values, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.prepare_config_and_inputs()
        return (config, inputs_dict)

    def get_config(self):
        if False:
            return 10
        return SpeechT5Config(hidden_size=self.hidden_size, encoder_layers=self.num_hidden_layers, decoder_layers=self.num_hidden_layers, encoder_attention_heads=self.num_attention_heads, decoder_attention_heads=self.num_attention_heads, encoder_ffn_dim=self.intermediate_size, decoder_ffn_dim=self.intermediate_size, vocab_size=self.vocab_size, num_mel_bins=self.num_mel_bins, reduction_factor=self.reduction_factor, speech_decoder_postnet_layers=self.speech_decoder_postnet_layers, speech_decoder_postnet_units=self.speech_decoder_postnet_units, speech_decoder_prenet_units=self.speech_decoder_prenet_units)

    def create_and_check_model_forward(self, config, inputs_dict):
        if False:
            i = 10
            return i + 15
        model = SpeechT5ForTextToSpeech(config=config).to(torch_device).eval()
        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']
        decoder_input_values = inputs_dict['decoder_input_values']
        result = model(input_ids, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(result.spectrogram.shape, (self.batch_size, self.decoder_seq_length * self.reduction_factor, self.num_mel_bins))

@require_torch
class SpeechT5ForTextToSpeechTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5ForTextToSpeech,) if is_torch_available() else ()
    all_generative_model_classes = (SpeechT5ForTextToSpeech,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    input_name = 'input_ids'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = SpeechT5ForTextToSpeechTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                (model2, info) = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info['missing_keys'], [])

    def test_model_forward(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        if False:
            return 10
        pass

    def test_determinism(self):
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
            expected_arg_names = ['input_ids', 'attention_mask', 'decoder_input_values', 'decoder_attention_mask']
            expected_arg_names.extend(['head_mask', 'decoder_head_mask', 'cross_attn_head_mask', 'encoder_outputs'] if 'head_mask' and 'decoder_head_mask' and ('cross_attn_head_mask' in arg_names) else ['encoder_outputs'])
            self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)

    def test_initialization(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                uniform_init_parms = ['conv.weight']
                if param.requires_grad:
                    if any((x in name for x in uniform_init_parms)):
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item() <= 1.0, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    def test_model_outputs_equivalence(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    def test_torchscript_output_attentions(self):
        if False:
            return 10
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        if False:
            print('Hello World!')
        pass

    @slow
    def test_torchscript_simple(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_training(self):
        if False:
            return 10
        pass

    def test_training_gradient_checkpointing(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            while True:
                i = 10
        pass

    def _mock_init_weights(self, module):
        if False:
            while True:
                i = 10
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, 'weight_g') and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, 'weight_v') and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(3)

@require_torch
@require_sentencepiece
@require_tokenizers
class SpeechT5ForTextToSpeechIntegrationTests(unittest.TestCase):

    @cached_property
    def default_model(self):
        if False:
            i = 10
            return i + 15
        return SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')

    @cached_property
    def default_processor(self):
        if False:
            i = 10
            return i + 15
        return SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')

    @cached_property
    def default_vocoder(self):
        if False:
            for i in range(10):
                print('nop')
        return SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    def test_generation(self):
        if False:
            i = 10
            return i + 15
        model = self.default_model
        model.to(torch_device)
        processor = self.default_processor
        set_seed(555)
        speaker_embeddings = torch.zeros((1, 512)).to(torch_device)
        input_text = 'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
        input_ids = processor(text=input_text, return_tensors='pt').input_ids.to(torch_device)
        generated_speech = model.generate_speech(input_ids, speaker_embeddings=speaker_embeddings)
        self.assertEqual(generated_speech.shape, (230, model.config.num_mel_bins))
        set_seed(555)
        generated_speech_with_generate = model.generate(input_ids, attention_mask=None, speaker_embeddings=speaker_embeddings)
        self.assertEqual(generated_speech_with_generate.shape, (230, model.config.num_mel_bins))

    def test_batch_generation(self):
        if False:
            i = 10
            return i + 15
        model = self.default_model
        model.to(torch_device)
        processor = self.default_processor
        vocoder = self.default_vocoder
        set_seed(555)
        input_text = ['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel', "nor is mister quilter's manner less interesting than his matter", 'he tells us that at this festive season of the year with christmas and rosebeaf looming before us']
        inputs = processor(text=input_text, padding='max_length', max_length=128, return_tensors='pt').to(torch_device)
        speaker_embeddings = torch.zeros((1, 512), device=torch_device)
        (spectrograms, spectrogram_lengths) = model.generate_speech(input_ids=inputs['input_ids'], speaker_embeddings=speaker_embeddings, attention_mask=inputs['attention_mask'], return_output_lengths=True)
        self.assertEqual(spectrograms.shape, (3, 262, model.config.num_mel_bins))
        waveforms = vocoder(spectrograms)
        waveform_lengths = [int(waveforms.size(1) / max(spectrogram_lengths)) * i for i in spectrogram_lengths]
        set_seed(555)
        (waveforms_with_vocoder, waveform_lengths_with_vocoder) = model.generate_speech(input_ids=inputs['input_ids'], speaker_embeddings=speaker_embeddings, attention_mask=inputs['attention_mask'], vocoder=vocoder, return_output_lengths=True)
        self.assertTrue(torch.allclose(waveforms, waveforms_with_vocoder, atol=1e-08))
        self.assertEqual(waveform_lengths, waveform_lengths_with_vocoder)
        set_seed(555)
        waveforms_with_vocoder_no_lengths = model.generate_speech(input_ids=inputs['input_ids'], speaker_embeddings=speaker_embeddings, attention_mask=inputs['attention_mask'], vocoder=vocoder, return_output_lengths=False)
        self.assertTrue(torch.allclose(waveforms_with_vocoder_no_lengths, waveforms_with_vocoder, atol=1e-08))
        for (i, text) in enumerate(input_text):
            set_seed(555)
            inputs = processor(text=text, padding='max_length', max_length=128, return_tensors='pt').to(torch_device)
            spectrogram = model.generate_speech(input_ids=inputs['input_ids'], speaker_embeddings=speaker_embeddings)
            self.assertEqual(spectrogram.shape, spectrograms[i][:spectrogram_lengths[i]].shape)
            self.assertTrue(torch.allclose(spectrogram, spectrograms[i][:spectrogram_lengths[i]], atol=0.005))
            waveform = vocoder(spectrogram)
            self.assertEqual(waveform.shape, waveforms[i][:waveform_lengths[i]].shape)
            set_seed(555)
            waveform_with_vocoder = model.generate_speech(input_ids=inputs['input_ids'], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
            self.assertTrue(torch.allclose(waveform, waveform_with_vocoder, atol=1e-08))

@require_torch
class SpeechT5ForSpeechToSpeechTester:

    def __init__(self, parent, batch_size=13, encoder_seq_length=1024, decoder_seq_length=1024, is_training=False, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=4, conv_dim=(32, 32, 32), conv_stride=(4, 4, 4), conv_kernel=(8, 8, 8), conv_bias=False, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, vocab_size=81, num_mel_bins=20, reduction_factor=2, speech_decoder_postnet_layers=2, speech_decoder_postnet_units=32, speech_decoder_prenet_units=32):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.reduction_factor = reduction_factor
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_prenet_units = speech_decoder_prenet_units

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        input_values = floats_tensor([self.batch_size, self.encoder_seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])
        decoder_input_values = floats_tensor([self.batch_size, self.decoder_seq_length, self.num_mel_bins], scale=1.0)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])
        config = self.get_config()
        inputs_dict = prepare_inputs_dict(config, input_values=input_values, decoder_input_values=decoder_input_values, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)
        return (config, inputs_dict)

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        (config, inputs_dict) = self.prepare_config_and_inputs()
        return (config, inputs_dict)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return SpeechT5Config(hidden_size=self.hidden_size, encoder_layers=self.num_hidden_layers, decoder_layers=self.num_hidden_layers, encoder_attention_heads=self.num_attention_heads, decoder_attention_heads=self.num_attention_heads, encoder_ffn_dim=self.intermediate_size, decoder_ffn_dim=self.intermediate_size, conv_dim=self.conv_dim, conv_stride=self.conv_stride, conv_kernel=self.conv_kernel, conv_bias=self.conv_bias, num_conv_pos_embeddings=self.num_conv_pos_embeddings, num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups, vocab_size=self.vocab_size, num_mel_bins=self.num_mel_bins, reduction_factor=self.reduction_factor, speech_decoder_postnet_layers=self.speech_decoder_postnet_layers, speech_decoder_postnet_units=self.speech_decoder_postnet_units, speech_decoder_prenet_units=self.speech_decoder_prenet_units)

    def create_and_check_model_forward(self, config, inputs_dict):
        if False:
            i = 10
            return i + 15
        model = SpeechT5ForSpeechToSpeech(config=config).to(torch_device).eval()
        input_values = inputs_dict['input_values']
        attention_mask = inputs_dict['attention_mask']
        decoder_input_values = inputs_dict['decoder_input_values']
        result = model(input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(result.spectrogram.shape, (self.batch_size, self.decoder_seq_length * self.reduction_factor, self.num_mel_bins))

@require_torch
class SpeechT5ForSpeechToSpeechTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5ForSpeechToSpeech,) if is_torch_available() else ()
    all_generative_model_classes = (SpeechT5ForSpeechToSpeech,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    input_name = 'input_values'

    def setUp(self):
        if False:
            return 10
        self.model_tester = SpeechT5ForSpeechToSpeechTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                (model2, info) = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info['missing_keys'], [])

    def test_model_forward(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_determinism(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_attention_outputs(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = getattr(self.model_tester, 'seq_length', None)
        decoder_seq_length = getattr(self.model_tester, 'decoder_seq_length', seq_len)
        encoder_seq_length = getattr(self.model_tester, 'encoder_seq_length', seq_len)
        decoder_key_length = getattr(self.model_tester, 'decoder_key_length', decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, 'key_length', encoder_seq_length)
        for model_class in self.all_model_classes:
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            subsampled_encoder_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(encoder_key_length)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length])
            out_len = len(outputs)
            correct_outlen = 5
            if 'labels' in inputs_dict:
                correct_outlen += 1
            if 'past_key_values' in outputs:
                correct_outlen += 1
            self.assertEqual(out_len, correct_outlen)
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(decoder_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length])
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(cross_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, decoder_seq_length, subsampled_encoder_key_length])
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))
            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length])

    def test_forward_signature(self):
        if False:
            return 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['input_values', 'attention_mask', 'decoder_input_values', 'decoder_attention_mask']
            expected_arg_names.extend(['head_mask', 'decoder_head_mask', 'cross_attn_head_mask', 'encoder_outputs'] if 'head_mask' and 'decoder_head_mask' and ('cross_attn_head_mask' in arg_names) else ['encoder_outputs'])
            self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)

    def test_hidden_states_output(self):
        if False:
            for i in range(10):
                print('nop')

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                i = 10
                return i + 15
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
            self.assertEqual(len(hidden_states), expected_num_layers)
            if hasattr(self.model_tester, 'encoder_seq_length'):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length
            subsampled_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(seq_length)
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [subsampled_seq_length, self.model_tester.hidden_size])
            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states
                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, 'seq_length', None)
                decoder_seq_length = getattr(self.model_tester, 'decoder_seq_length', seq_len)
                self.assertListEqual(list(hidden_states[0].shape[-2:]), [decoder_seq_length, self.model_tester.hidden_size])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                uniform_init_parms = ['conv.weight', 'conv.parametrizations.weight', 'masked_spec_embed', 'feature_projection.projection.weight', 'feature_projection.projection.bias']
                if param.requires_grad:
                    if any((x in name for x in uniform_init_parms)):
                        self.assertTrue(-1.0 <= ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item() <= 1.0, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def test_inputs_embeds(self):
        if False:
            while True:
                i = 10
        pass

    def test_model_common_attributes(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_model_outputs_equivalence(self):
        if False:
            print('Hello World!')
        pass

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @slow
    def test_torchscript_output_attentions(self):
        if False:
            i = 10
            return i + 15
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @slow
    def test_torchscript_simple(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_training(self):
        if False:
            print('Hello World!')
        pass

    def test_training_gradient_checkpointing(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            print('Hello World!')
        pass

    def _mock_init_weights(self, module):
        if False:
            i = 10
            return i + 15
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

@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class SpeechT5ForSpeechToSpeechIntegrationTests(unittest.TestCase):

    @cached_property
    def default_processor(self):
        if False:
            print('Hello World!')
        return SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')

    def _load_datasamples(self, num_samples):
        if False:
            print('Hello World!')
        from datasets import load_dataset
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').select(range(num_samples))[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def test_generation_librispeech(self):
        if False:
            return 10
        model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
        model.to(torch_device)
        processor = self.default_processor
        input_speech = self._load_datasamples(1)
        input_values = processor(audio=input_speech, return_tensors='pt').input_values.to(torch_device)
        speaker_embeddings = torch.zeros((1, 512), device=torch_device)
        generated_speech = model.generate_speech(input_values, speaker_embeddings=speaker_embeddings)
        self.assertEqual(generated_speech.shape[1], model.config.num_mel_bins)
        self.assertGreaterEqual(generated_speech.shape[0], 300)
        self.assertLessEqual(generated_speech.shape[0], 310)

class SpeechT5HifiGanTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=False, num_mel_bins=20):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_mel_bins = num_mel_bins

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        input_values = floats_tensor([self.seq_length, self.num_mel_bins], scale=1.0)
        config = self.get_config()
        return (config, input_values)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return SpeechT5HifiGanConfig(model_in_dim=self.num_mel_bins, upsample_initial_channel=32)

    def create_and_check_model(self, config, input_values):
        if False:
            i = 10
            return i + 15
        model = SpeechT5HifiGan(config=config).to(torch_device).eval()
        result = model(input_values)
        self.parent.assertEqual(result.shape, (self.seq_length * 256,))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        (config, input_values) = self.prepare_config_and_inputs()
        inputs_dict = {'spectrogram': input_values}
        return (config, inputs_dict)

@require_torch
class SpeechT5HifiGanTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5HifiGan,) if is_torch_available() else ()
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_resize_position_embeddings = False
    test_head_masking = False
    test_mismatched_shapes = False
    test_missing_keys = False
    test_model_parallel = False
    is_encoder_decoder = False
    has_attentions = False
    input_name = 'spectrogram'

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = SpeechT5HifiGanTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5HifiGanConfig)

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        if False:
            return 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['spectrogram']
            self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)

    def test_hidden_states_output(self):
        if False:
            return 10
        pass

    def test_initialization(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_model_common_attributes(self):
        if False:
            print('Hello World!')
        pass

    def test_model_outputs_equivalence(self):
        if False:
            while True:
                i = 10
        pass

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_save_load_fast_init_from_base(self):
        if False:
            return 10
        pass

    def test_save_load_fast_init_to_base(self):
        if False:
            print('Hello World!')
        pass

    def test_batched_inputs_outputs(self):
        if False:
            return 10
        (config, inputs) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            batched_inputs = inputs['spectrogram'].unsqueeze(0).repeat(2, 1, 1)
            with torch.no_grad():
                batched_outputs = model(batched_inputs.to(torch_device))
            self.assertEqual(batched_inputs.shape[0], batched_outputs.shape[0], msg='Got different batch dims for input and output')

    def test_unbatched_inputs_outputs(self):
        if False:
            print('Hello World!')
        (config, inputs) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(inputs['spectrogram'].to(torch_device))
            self.assertTrue(outputs.dim() == 1, msg='Got un-batched inputs but batched output')