import tempfile
import unittest
import timeout_decorator
from parameterized import parameterized
from transformers import FSMTConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, require_torch_fp16, slow, torch_device
from transformers.utils import cached_property
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import FSMTForConditionalGeneration, FSMTModel, FSMTTokenizer
    from transformers.models.fsmt.modeling_fsmt import SinusoidalPositionalEmbedding, _prepare_fsmt_decoder_inputs, invert_mask, shift_tokens_right
    from transformers.pipelines import TranslationPipeline

class FSMTModelTester:

    def __init__(self, parent, src_vocab_size=99, tgt_vocab_size=99, langs=['ru', 'en'], batch_size=13, seq_length=7, is_training=False, use_labels=False, hidden_size=16, num_hidden_layers=2, num_attention_heads=4, intermediate_size=4, hidden_act='relu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=20, bos_token_id=0, pad_token_id=1, eos_token_id=2):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.langs = langs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        torch.manual_seed(0)
        self.vocab_size = self.src_vocab_size

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.src_vocab_size).clamp(3)
        input_ids[:, -1] = 2
        config = self.get_config()
        inputs_dict = prepare_fsmt_inputs_dict(config, input_ids)
        return (config, inputs_dict)

    def get_config(self):
        if False:
            while True:
                i = 10
        return FSMTConfig(vocab_size=self.src_vocab_size, src_vocab_size=self.src_vocab_size, tgt_vocab_size=self.tgt_vocab_size, langs=self.langs, d_model=self.hidden_size, encoder_layers=self.num_hidden_layers, decoder_layers=self.num_hidden_layers, encoder_attention_heads=self.num_attention_heads, decoder_attention_heads=self.num_attention_heads, encoder_ffn_dim=self.intermediate_size, decoder_ffn_dim=self.intermediate_size, dropout=self.hidden_dropout_prob, attention_dropout=self.attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, eos_token_id=self.eos_token_id, bos_token_id=self.bos_token_id, pad_token_id=self.pad_token_id)

    def prepare_config_and_inputs_for_common(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.prepare_config_and_inputs()
        inputs_dict['decoder_input_ids'] = inputs_dict['input_ids']
        inputs_dict['decoder_attention_mask'] = inputs_dict['attention_mask']
        inputs_dict['use_cache'] = False
        return (config, inputs_dict)

def prepare_fsmt_inputs_dict(config, input_ids, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None):
    if False:
        return 10
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask}

@require_torch
class FSMTModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FSMTModel, FSMTForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (FSMTForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {'conversational': FSMTForConditionalGeneration, 'feature-extraction': FSMTModel, 'summarization': FSMTForConditionalGeneration, 'text2text-generation': FSMTForConditionalGeneration, 'translation': FSMTForConditionalGeneration} if is_torch_available() else {}
    is_encoder_decoder = True
    test_pruning = False
    test_missing_keys = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = FSMTModelTester(self)
        self.langs = ['en', 'ru']
        config = {'langs': self.langs, 'src_vocab_size': 10, 'tgt_vocab_size': 20}
        config['vocab_size'] = 99
        self.config_tester = ConfigTester(self, config_class=FSMTConfig, **config)

    def test_config(self):
        if False:
            print('Hello World!')
        self.config_tester.run_common_tests()

    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Embedding)
            model.set_input_embeddings(nn.Embedding(10, 10))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.modules.sparse.Embedding))

    def test_initialization_more(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        model = FSMTModel(config)
        model.to(torch_device)
        model.eval()

        def _check_var(module):
            if False:
                while True:
                    i = 10
            'Check that we initialized various parameters from N(0, config.init_std).'
            self.assertAlmostEqual(torch.std(module.weight).item(), config.init_std, 2)
        _check_var(model.encoder.embed_tokens)
        _check_var(model.encoder.layers[0].self_attn.k_proj)
        _check_var(model.encoder.layers[0].fc1)

    def test_advanced_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        config.use_cache = False
        inputs_dict['input_ids'][:, -2:] = config.pad_token_id
        (decoder_input_ids, decoder_attn_mask, causal_mask) = _prepare_fsmt_decoder_inputs(config, inputs_dict['input_ids'])
        model = FSMTModel(config).to(torch_device).eval()
        decoder_features_with_created_mask = model(**inputs_dict)[0]
        decoder_features_with_passed_mask = model(decoder_attention_mask=invert_mask(decoder_attn_mask), decoder_input_ids=decoder_input_ids, **inputs_dict)[0]
        _assert_tensors_equal(decoder_features_with_passed_mask, decoder_features_with_created_mask)
        useless_mask = torch.zeros_like(decoder_attn_mask)
        decoder_features = model(decoder_attention_mask=useless_mask, **inputs_dict)[0]
        self.assertTrue(isinstance(decoder_features, torch.Tensor))
        self.assertEqual(decoder_features.size(), (self.model_tester.batch_size, self.model_tester.seq_length, config.tgt_vocab_size))
        if decoder_attn_mask.min().item() < -1000.0:
            self.assertFalse((decoder_features_with_created_mask == decoder_features).all().item())
        decoder_features_with_long_encoder_mask = model(inputs_dict['input_ids'], attention_mask=inputs_dict['attention_mask'].long())[0]
        _assert_tensors_equal(decoder_features_with_long_encoder_mask, decoder_features_with_created_mask)

    def test_save_load_missing_keys(self):
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

    @unittest.skip('Test has a segmentation fault on torch 1.8.0')
    def test_export_to_onnx(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        model = FSMTModel(config).to(torch_device)
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.onnx.export(model, (inputs_dict['input_ids'], inputs_dict['attention_mask']), f'{tmpdirname}/fsmt_test.onnx', export_params=True, opset_version=12, input_names=['input_ids', 'attention_mask'])

    def test_ensure_weights_are_shared(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs()
        config.tie_word_embeddings = True
        model = FSMTForConditionalGeneration(config)
        self.assertEqual(len({model.get_output_embeddings().weight.data_ptr(), model.get_input_embeddings().weight.data_ptr(), model.base_model.decoder.output_projection.weight.data_ptr()}), 1)
        config.tie_word_embeddings = False
        model = FSMTForConditionalGeneration(config)
        self.assertEqual(len({model.get_output_embeddings().weight.data_ptr(), model.get_input_embeddings().weight.data_ptr(), model.base_model.decoder.output_projection.weight.data_ptr()}), 2)

    @unittest.skip("can't be implemented for FSMT due to dual vocab.")
    def test_resize_tokens_embeddings(self):
        if False:
            return 10
        pass

    @unittest.skip('Passing inputs_embeds not implemented for FSMT.')
    def test_inputs_embeds(self):
        if False:
            return 10
        pass

    @unittest.skip("model weights aren't tied in FSMT.")
    def test_tie_model_weights(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip('TODO: Decoder embeddings cannot be resized at the moment')
    def test_resize_embeddings_untied(self):
        if False:
            i = 10
            return i + 15
        pass

@require_torch
class FSMTHeadTests(unittest.TestCase):
    src_vocab_size = 99
    tgt_vocab_size = 99
    langs = ['ru', 'en']

    def _get_config(self):
        if False:
            i = 10
            return i + 15
        return FSMTConfig(src_vocab_size=self.src_vocab_size, tgt_vocab_size=self.tgt_vocab_size, langs=self.langs, d_model=24, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=48, eos_token_id=2, pad_token_id=1, bos_token_id=0)

    def _get_config_and_data(self):
        if False:
            while True:
                i = 10
        input_ids = torch.tensor([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 82, 2], [5, 97, 17, 39, 94, 40, 2], [76, 83, 94, 25, 70, 78, 2], [87, 59, 41, 35, 48, 66, 2], [55, 13, 16, 58, 5, 2, 1], [64, 27, 31, 51, 12, 75, 2], [52, 64, 86, 17, 83, 39, 2], [48, 61, 9, 24, 71, 82, 2], [26, 1, 60, 48, 22, 13, 2], [21, 5, 62, 28, 14, 76, 2], [45, 98, 37, 86, 59, 48, 2], [70, 70, 50, 9, 28, 0, 2]], dtype=torch.long, device=torch_device)
        batch_size = input_ids.shape[0]
        config = self._get_config()
        return (config, input_ids, batch_size)

    def test_generate_beam_search(self):
        if False:
            while True:
                i = 10
        input_ids = torch.tensor([[71, 82, 2], [68, 34, 2]], dtype=torch.long, device=torch_device)
        config = self._get_config()
        lm_model = FSMTForConditionalGeneration(config).to(torch_device)
        lm_model.eval()
        max_length = 5
        new_input_ids = lm_model.generate(input_ids.clone(), do_sample=True, num_return_sequences=1, num_beams=2, no_repeat_ngram_size=3, max_length=max_length)
        self.assertEqual(new_input_ids.shape, (input_ids.shape[0], max_length))

    def test_shift_tokens_right(self):
        if False:
            print('Hello World!')
        input_ids = torch.tensor([[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]], dtype=torch.long)
        shifted = shift_tokens_right(input_ids, 1)
        n_pad_before = input_ids.eq(1).float().sum()
        n_pad_after = shifted.eq(1).float().sum()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(torch.eq(shifted[:, 0], 2).all())

    @require_torch_fp16
    def test_generate_fp16(self):
        if False:
            i = 10
            return i + 15
        (config, input_ids, batch_size) = self._get_config_and_data()
        attention_mask = input_ids.ne(1).to(torch_device)
        model = FSMTForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_dummy_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        (config, *_) = self._get_config_and_data()
        model = FSMTForConditionalGeneration(config).eval().to(torch_device)
        model(**model.dummy_inputs)

    def test_prepare_fsmt_decoder_inputs(self):
        if False:
            print('Hello World!')
        (config, *_) = self._get_config_and_data()
        input_ids = _long_tensor([4, 4, 2])
        decoder_input_ids = _long_tensor([[26388, 2, config.pad_token_id]])
        causal_mask_dtype = torch.float32
        ignore = torch.finfo(causal_mask_dtype).min
        (decoder_input_ids, decoder_attn_mask, causal_mask) = _prepare_fsmt_decoder_inputs(config, input_ids, decoder_input_ids, causal_mask_dtype=causal_mask_dtype)
        expected_causal_mask = torch.tensor([[0, ignore, ignore], [0, 0, ignore], [0, 0, 0]]).to(input_ids.device)
        self.assertEqual(decoder_attn_mask.size(), decoder_input_ids.size())
        self.assertTrue(torch.eq(expected_causal_mask, causal_mask).all())

def _assert_tensors_equal(a, b, atol=1e-12, prefix=''):
    if False:
        i = 10
        return i + 15
    'If tensors not close, or a and b arent both tensors, raise a nice Assertion error.'
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        if len(prefix) > 0:
            prefix = f'{prefix}: '
        raise AssertionError(f'{prefix}{a} != {b}')

def _long_tensor(tok_lst):
    if False:
        return 10
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)
TOLERANCE = 0.0001
pairs = [['en-ru'], ['ru-en'], ['en-de'], ['de-en']]

@require_torch
@require_sentencepiece
@require_tokenizers
class FSMTModelIntegrationTests(unittest.TestCase):
    tokenizers_cache = {}
    models_cache = {}
    default_mname = 'facebook/wmt19-en-ru'

    @cached_property
    def default_tokenizer(self):
        if False:
            i = 10
            return i + 15
        return self.get_tokenizer(self.default_mname)

    @cached_property
    def default_model(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_model(self.default_mname)

    def get_tokenizer(self, mname):
        if False:
            i = 10
            return i + 15
        if mname not in self.tokenizers_cache:
            self.tokenizers_cache[mname] = FSMTTokenizer.from_pretrained(mname)
        return self.tokenizers_cache[mname]

    def get_model(self, mname):
        if False:
            return 10
        if mname not in self.models_cache:
            self.models_cache[mname] = FSMTForConditionalGeneration.from_pretrained(mname).to(torch_device)
            if torch_device == 'cuda':
                self.models_cache[mname].half()
        return self.models_cache[mname]

    @slow
    def test_inference_no_head(self):
        if False:
            i = 10
            return i + 15
        tokenizer = self.default_tokenizer
        model = FSMTModel.from_pretrained(self.default_mname).to(torch_device)
        src_text = 'My friend computer will translate this for me'
        input_ids = tokenizer([src_text], return_tensors='pt')['input_ids']
        input_ids = _long_tensor(input_ids).to(torch_device)
        inputs_dict = prepare_fsmt_inputs_dict(model.config, input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 10, model.config.tgt_vocab_size))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[-1.5753, -1.5753, 2.8975], [-0.954, -0.954, 1.0299], [-3.3131, -3.3131, 0.5219]]).to(torch_device)
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def translation_setup(self, pair):
        if False:
            for i in range(10):
                print('nop')
        text = {'en': "Machine learning is great, isn't it?", 'ru': 'Машинное обучение - это здорово, не так ли?', 'de': 'Maschinelles Lernen ist großartig, oder?'}
        (src, tgt) = pair.split('-')
        print(f'Testing {src} -> {tgt}')
        mname = f'facebook/wmt19-{pair}'
        src_text = text[src]
        tgt_text = text[tgt]
        tokenizer = self.get_tokenizer(mname)
        model = self.get_model(mname)
        return (tokenizer, model, src_text, tgt_text)

    @parameterized.expand(pairs)
    @slow
    def test_translation_direct(self, pair):
        if False:
            while True:
                i = 10
        (tokenizer, model, src_text, tgt_text) = self.translation_setup(pair)
        input_ids = tokenizer.encode(src_text, return_tensors='pt').to(torch_device)
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert decoded == tgt_text, f'\n\ngot: {decoded}\nexp: {tgt_text}\n'

    @parameterized.expand(pairs)
    @slow
    def test_translation_pipeline(self, pair):
        if False:
            i = 10
            return i + 15
        (tokenizer, model, src_text, tgt_text) = self.translation_setup(pair)
        pipeline = TranslationPipeline(model, tokenizer, framework='pt', device=torch_device)
        output = pipeline([src_text])
        self.assertEqual([tgt_text], [x['translation_text'] for x in output])

@require_torch
class TestSinusoidalPositionalEmbeddings(unittest.TestCase):
    padding_idx = 1
    tolerance = 0.0001

    def test_basic(self):
        if False:
            while True:
                i = 10
        input_ids = torch.tensor([[4, 10]], dtype=torch.long, device=torch_device)
        emb1 = SinusoidalPositionalEmbedding(num_positions=6, embedding_dim=6, padding_idx=self.padding_idx).to(torch_device)
        emb = emb1(input_ids)
        desired_weights = torch.tensor([[0.9093, 0.019999, 0.0002, -0.41615, 0.9998, 1.0], [0.14112, 0.029995, 0.0003, -0.98999, 0.99955, 1.0]]).to(torch_device)
        self.assertTrue(torch.allclose(emb[0], desired_weights, atol=self.tolerance), msg=f'\nexp:\n{desired_weights}\ngot:\n{emb[0]}\n')

    def test_odd_embed_dim(self):
        if False:
            return 10
        SinusoidalPositionalEmbedding(num_positions=4, embedding_dim=5, padding_idx=self.padding_idx).to(torch_device)
        SinusoidalPositionalEmbedding(num_positions=5, embedding_dim=4, padding_idx=self.padding_idx).to(torch_device)

    @unittest.skip('different from marian (needs more research)')
    def test_positional_emb_weights_against_marian(self):
        if False:
            for i in range(10):
                print('nop')
        desired_weights = torch.tensor([[0, 0, 0, 0, 0], [0.84147096, 0.82177866, 0.8018049, 0.78165019, 0.76140374], [0.90929741, 0.93651021, 0.95829457, 0.97505713, 0.98720258]])
        emb1 = SinusoidalPositionalEmbedding(num_positions=512, embedding_dim=512, padding_idx=self.padding_idx).to(torch_device)
        weights = emb1.weights.data[:3, :5]
        self.assertTrue(torch.allclose(weights, desired_weights, atol=self.tolerance), msg=f'\nexp:\n{desired_weights}\ngot:\n{weights}\n')
        input_ids = torch.tensor([[4, 10, self.padding_idx, self.padding_idx, self.padding_idx]], dtype=torch.long, device=torch_device)
        no_cache_pad_zero = emb1(input_ids)[0]
        self.assertTrue(torch.allclose(torch.tensor(desired_weights, device=torch_device), no_cache_pad_zero[:3, :5], atol=0.001))