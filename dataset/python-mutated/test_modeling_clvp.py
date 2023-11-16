""" Testing suite for the PyTorch Clvp model. """
import gc
import tempfile
import unittest
import datasets
import numpy as np
from transformers import ClvpConfig, ClvpDecoderConfig, ClvpEncoderConfig
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask
if is_torch_available():
    import torch
    from transformers import ClvpEncoder, ClvpForCausalLM, ClvpModel, ClvpModelForConditionalGeneration
    from transformers.models.clvp.modeling_clvp import CLVP_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers import ClvpFeatureExtractor, ClvpTokenizer

class ClvpEncoderTester:

    def __init__(self, parent, batch_size=2, seq_length=7, is_training=False, use_input_mask=True, use_labels=True, vocab_size=50, hidden_size=128, projection_dim=16, num_hidden_layers=2, num_attention_heads=4, intermediate_size=32, dropout=0.1, attention_dropout=0.1, initializer_range=0.02, scope=None):
        if False:
            print('Hello World!')
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1

    def get_config(self):
        if False:
            while True:
                i = 10
        encoder_config = ClvpEncoderConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, projection_dim=self.projection_dim, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, initializer_range=self.initializer_range, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id)
        return encoder_config

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        if input_mask is not None:
            (batch_size, seq_length) = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for (batch_idx, start_index) in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0
        encoder_config = self.get_config()
        return (encoder_config, input_ids, input_mask)

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        config_and_inputs = self.prepare_config_and_inputs()
        (speech_config, input_ids, input_mask) = config_and_inputs
        inputs_dict = {'input_ids': input_ids.to(torch_device), 'attention_mask': input_mask.to(torch_device)}
        return (speech_config, inputs_dict)

    def create_and_check_model(self, speech_config, input_ids, input_mask):
        if False:
            i = 10
            return i + 15
        text_config = ClvpEncoderConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, projection_dim=self.projection_dim, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, initializer_range=self.initializer_range)
        text_encoder_model = ClvpEncoder(config=text_config)
        text_encoder_model.to(torch_device)
        text_encoder_model.eval()
        with torch.no_grad():
            result = text_encoder_model(input_ids, attention_mask=input_mask)
            result = text_encoder_model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result[0].shape, (self.batch_size, self.projection_dim))
        speech_encoder_model = ClvpEncoder(config=speech_config)
        speech_encoder_model.to(torch_device)
        speech_encoder_model.eval()
        with torch.no_grad():
            result = speech_encoder_model(input_ids, attention_mask=input_mask)
            result = speech_encoder_model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result[0].shape, (self.batch_size, self.projection_dim))

@require_torch
class ClvpEncoderTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ClvpEncoder,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = ClvpEncoderTester(self)
        self.encoder_config_tester = ConfigTester(self, config_class=ClvpEncoderConfig, hidden_size=32)

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_config(self):
        if False:
            return 10
        self.encoder_config_tester.run_common_tests()

    def test_model(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='ClvpEncoder does not output loss')
    def test_training(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='ClvpEncoder does not output loss')
    def test_training_gradient_checkpointing(self):
        if False:
            while True:
                i = 10
        pass

class ClvpDecoderTester:

    def __init__(self, parent, batch_size=2, seq_length=3, is_training=False, vocab_size=300, max_position_embeddings=256, max_text_tokens=256, use_input_mask=True, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, bos_token_id=97, eos_token_id=98, relative_attention_num_buckets=4, relative_attention_max_distance=16):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_text_tokens = max_text_tokens
        self.use_input_mask = use_input_mask
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

    def get_config(self):
        if False:
            while True:
                i = 10
        decoder_config = ClvpDecoderConfig(vocab_size=self.vocab_size, max_position_embeddings=self.max_position_embeddings, max_text_tokens=self.max_text_tokens, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id, relative_attention_num_buckets=self.relative_attention_num_buckets, relative_attention_max_distance=self.relative_attention_max_distance)
        return decoder_config

    def prepare_config_and_inputs(self):
        if False:
            return 10
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        if input_mask is not None:
            (batch_size, seq_length) = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for (batch_idx, start_index) in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0
        decoder_config = self.get_config()
        return (decoder_config, input_ids, input_mask)

    def create_and_check_model(self, config, input_ids, attention_mask):
        if False:
            print('Hello World!')
        model = ClvpForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask) = config_and_inputs
        inputs_dict = {'input_ids': input_ids.to(torch_device), 'attention_mask': attention_mask.to(torch_device)}
        return (config, inputs_dict)

@require_torch
class ClvpDecoderTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (ClvpModel, ClvpForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (ClvpForCausalLM,) if is_torch_available() else ()
    test_pruning = False

    def setUp(self):
        if False:
            return 10
        self.model_tester = ClvpDecoderTester(self)
        self.decoder_config_tester = ConfigTester(self, config_class=ClvpDecoderConfig, hidden_size=32)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        if False:
            while True:
                i = 10
        if return_labels and model_class == ClvpForCausalLM:
            inputs_dict['labels'] = torch.zeros([self.model_tester.batch_size, self.model_tester.seq_length], device=torch_device).long()
        return inputs_dict

    def test_training(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        model = ClvpForCausalLM(config)
        model.to(torch_device)
        model.train()
        inputs = self._prepare_for_class(inputs_dict, ClvpForCausalLM, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()

    def test_training_gradient_checkpointing(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.use_cache = False
        config.return_dict = True
        model = ClvpForCausalLM(config)
        model.to(torch_device)
        model.gradient_checkpointing_enable()
        model.train()
        inputs = self._prepare_for_class(inputs_dict, ClvpForCausalLM, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()

class ClvpModelForConditionalGenerationTester:

    def __init__(self, parent, is_training=False):
        if False:
            return 10
        self.parent = parent
        self.clvp_encoder_tester = ClvpEncoderTester(parent)
        self.is_training = is_training

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        decoder_config = ClvpDecoderConfig(vocab_size=50, max_position_embeddings=30, max_text_tokens=30, hidden_size=128, num_hidden_layers=1, num_attention_heads=2, bos_token_id=97, eos_token_id=98, relative_attention_num_buckets=4, relative_attention_max_distance=16)
        text_config = self.clvp_encoder_tester.get_config()
        speech_config = self.clvp_encoder_tester.get_config()
        speech_config.vocab_size = 300
        return ClvpConfig.from_sub_model_configs(text_config, speech_config, decoder_config, projection_dim=16)

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        (_, input_ids, attention_mask) = self.clvp_encoder_tester.prepare_config_and_inputs()
        ds = datasets.load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        ds = ds.cast_column('audio', datasets.Audio(sampling_rate=22050))
        (_, audio, sr) = ds.sort('id').select(range(1))[:1]['audio'][0].values()
        feature_extractor = ClvpFeatureExtractor()
        input_features = feature_extractor(raw_speech=audio, sampling_rate=sr, return_tensors='pt')['input_features'].to(torch_device)
        config = self.get_config()
        return (config, input_ids, attention_mask, input_features)

    def create_and_check_model(self, config, input_ids, attention_mask, input_features):
        if False:
            print('Hello World!')
        model = ClvpModelForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, input_features=input_features, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits_per_speech.shape, (2, self.clvp_encoder_tester.batch_size))
        self.parent.assertEqual(result.logits_per_text.shape, (self.clvp_encoder_tester.batch_size, 2))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, input_features) = config_and_inputs
        inputs_dict = {'input_ids': input_ids.to(torch_device), 'attention_mask': attention_mask.to(torch_device), 'input_features': input_features.to(torch_device), 'return_loss': False}
        return (config, inputs_dict)

@require_torch
class ClvpModelForConditionalGenerationTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ClvpModelForConditionalGeneration,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = ClvpModelForConditionalGenerationTester(self)
        self.clvp_config_tester = ConfigTester(self, config_class=ClvpConfig, hidden_size=32)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

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
                i = 10
                return i + 15
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            decoder_hidden_states = outputs.decoder_hidden_states
            text_encoder_hidden_states = outputs.text_encoder_hidden_states
            speech_encoder_hidden_states = outputs.speech_encoder_hidden_states
            expected_decoder_num_layers = config.decoder_config.num_hidden_layers + 1
            self.assertEqual(len(decoder_hidden_states), expected_decoder_num_layers)
            expected_speech_encoder_num_layers = config.text_config.num_hidden_layers + 1
            self.assertEqual(len(text_encoder_hidden_states), expected_speech_encoder_num_layers)
            expected_text_encoder_num_layers = config.speech_config.num_hidden_layers + 1
            self.assertEqual(len(speech_encoder_hidden_states), expected_text_encoder_num_layers)
            self.assertEqual(decoder_hidden_states[0].shape[-1], config.decoder_config.hidden_size)
            self.assertListEqual(list(text_encoder_hidden_states[0].shape[-2:]), [self.model_tester.clvp_encoder_tester.seq_length, config.text_config.hidden_size])
            self.assertEqual(speech_encoder_hidden_states[0].shape[-1], config.speech_config.hidden_size)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='ClvpModelForConditionalGeneration does not have get_input_embeddings')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='ClvpModelForConditionalGeneration does not have get_input_embeddings')
    def test_model_common_attributes(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_initialization(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                if param.requires_grad:
                    if name == 'logit_scale':
                        expected_value = np.log(1 / 0.07)
                        returned_value = param.data.item()
                        self.assertAlmostEqual(returned_value, expected_value, delta=0.001, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        expected_range = [0.0, 1.0]
                        returned_range = ((param.data.mean() * 1000000000.0).round() / 1000000000.0).item()
                        self.assertIn(returned_range, expected_range, msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def test_load_speech_text_decoder_config(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            encoder_config = ClvpEncoderConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), encoder_config.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            decoder_config = ClvpDecoderConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.decoder_config.to_dict(), decoder_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ClvpModelForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)

@slow
@require_torch
class ClvpIntegrationTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.text = 'This is an example text.'
        ds = datasets.load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        ds = ds.cast_column('audio', datasets.Audio(sampling_rate=22050))
        (_, self.speech_samples, self.sr) = ds.sort('id').select(range(1))[:1]['audio'][0].values()
        self.model = ClvpModelForConditionalGeneration.from_pretrained('susnato/clvp_dev').to(torch_device)
        self.model.eval()
        tokenizer = ClvpTokenizer.from_pretrained('susnato/clvp_dev')
        feature_extractor = ClvpFeatureExtractor.from_pretrained('susnato/clvp_dev')
        tokenizer_output = tokenizer(self.text, return_tensors='pt')
        self.text_tokens = tokenizer_output['input_ids'].to(torch_device)
        self.input_features = feature_extractor(raw_speech=self.speech_samples, sampling_rate=self.sr, return_tensors='pt')['input_features'].to(torch_device)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_conditional_encoder(self):
        if False:
            print('Hello World!')
        with torch.no_grad():
            conditioning_encoder_outputs = self.model.conditioning_encoder(input_features=self.input_features, input_ids=self.text_tokens).to('cpu')
        self.assertEqual(conditioning_encoder_outputs.shape, torch.Size((self.input_features.shape[0], 18, self.model.config.decoder_config.hidden_size)))
        EXPECTED_OUTPUTS = torch.tensor([[-0.8582, 0.5228, 1.9944], [-0.0465, -1.1017, -0.0093], [-0.0466, -0.603, -0.128]])
        self.assertTrue(torch.allclose(conditioning_encoder_outputs[0, :3, :3], EXPECTED_OUTPUTS, atol=0.0001))

    def test_decoder_model_generate(self):
        if False:
            while True:
                i = 10
        autoregressive_model_output = self.model.speech_decoder_model.generate(input_ids=self.text_tokens).cpu()
        EXPECTED_OUTPUTS = torch.tensor([[147, 2, 54, 2, 43, 2, 169, 122, 29, 64, 2, 136, 37, 33, 9, 8193]])
        self.assertTrue(torch.allclose(autoregressive_model_output, EXPECTED_OUTPUTS))

    def test_text_and_speech_encoder_models(self):
        if False:
            while True:
                i = 10
        text_embeds = self.model.text_encoder_model(input_ids=self.text_tokens, return_dict=True)[0].cpu()
        EXPECTED_TEXT_EMBEDS = torch.tensor([1.806, -2.7928, 3.2021, -1.5673, 2.3284, -3.2065, -1.3368, 2.2322, -1.7667, 0.41505, 2.4119, -0.0058133, -4.6367, 0.1645, 6.7459, 6.6292, 1.1046, 3.6196, -10.496, 5.4924])
        self.assertTrue(torch.allclose(text_embeds[0, :20], EXPECTED_TEXT_EMBEDS, atol=0.0001))
        speech_embeds = self.model.speech_encoder_model(input_ids=self.text_tokens, return_dict=True)[0].cpu()
        EXPECTED_SPEECH_EMBEDS = torch.tensor([4.6143, -5.5784, 0.8983, -3.9665, -0.6714, -1.0665, -1.1277, 1.5619, 2.6322, -7.2008, -2.4932, 0.3265, -1.4738, 0.1425, 5.0825, 4.176, -5.4708, 2.1935, -6.0044, 3.954])
        self.assertTrue(torch.allclose(speech_embeds[0, :20], EXPECTED_SPEECH_EMBEDS, atol=0.0001))

    def test_full_model_integration(self):
        if False:
            for i in range(10):
                print('nop')
        full_model_output = self.model.generate(input_ids=self.text_tokens, input_features=self.input_features, do_sample=False, num_beams=4, num_return_sequences=4, max_new_tokens=10).speech_ids.cpu()
        EXPECTED_OUTPUTS = torch.tensor([[1953, 1080, 612], [1953, 1953, 612], [1953, 612, 716]])
        self.assertTrue(torch.allclose(full_model_output[-3:, -3:], EXPECTED_OUTPUTS))