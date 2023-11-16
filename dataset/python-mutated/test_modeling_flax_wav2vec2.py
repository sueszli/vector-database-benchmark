import inspect
import math
import multiprocessing
import traceback
import unittest
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2Config, is_flax_available
from transformers.testing_utils import CaptureLogger, is_flaky, is_librosa_available, is_pt_flax_cross_test, is_pyctcdecode_available, require_flax, require_librosa, require_pyctcdecode, require_soundfile, run_test_in_subprocess, slow
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, random_attention_mask
if is_flax_available():
    import jax
    import jax.numpy as jnp
    import optax
    from flax.traverse_util import flatten_dict
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_flax_wav2vec2 import FlaxWav2Vec2ForCTC, FlaxWav2Vec2ForPreTraining, FlaxWav2Vec2GumbelVectorQuantizer, FlaxWav2Vec2Model, _compute_mask_indices, _sample_negative_indices
if is_pyctcdecode_available():
    import pyctcdecode.decoder
    from transformers import Wav2Vec2ProcessorWithLM
    from transformers.models.wav2vec2_with_lm import processing_wav2vec2_with_lm
if is_librosa_available():
    import librosa

def _test_wav2vec2_with_lm_invalid_pool(in_queue, out_queue, timeout):
    if False:
        i = 10
        return i + 15
    error = None
    try:
        _ = in_queue.get(timeout=timeout)
        ds = load_dataset('common_voice', 'es', split='test', streaming=True)
        sample = next(iter(ds))
        resampled_audio = librosa.resample(sample['audio']['array'], 48000, 16000)
        model = FlaxWav2Vec2ForCTC.from_pretrained('patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm')
        processor = Wav2Vec2ProcessorWithLM.from_pretrained('patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm')
        input_values = processor(resampled_audio, return_tensors='np').input_values
        logits = model(input_values).logits
        with CaptureLogger(pyctcdecode.decoder.logger) as cl, multiprocessing.get_context('spawn').Pool(1) as pool:
            transcription = processor.batch_decode(np.array(logits), pool).text
        unittest.TestCase().assertIn('Falling back to sequential decoding.', cl.out)
        unittest.TestCase().assertEqual(transcription[0], 'bien y qué regalo vas a abrir primero')
        multiprocessing.set_start_method('spawn', force=True)
        with CaptureLogger(processing_wav2vec2_with_lm.logger) as cl:
            transcription = processor.batch_decode(np.array(logits)).text
        unittest.TestCase().assertIn('Falling back to sequential decoding.', cl.out)
        unittest.TestCase().assertEqual(transcription[0], 'bien y qué regalo vas a abrir primero')
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()

class FlaxWav2Vec2ModelTester:

    def __init__(self, parent, batch_size=13, seq_length=1024, is_training=False, hidden_size=24, feat_extract_norm='layer', feat_extract_dropout=0.0, feat_extract_activation='gelu', conv_dim=(32, 32, 32), conv_stride=(4, 4, 4), conv_kernel=(8, 8, 8), conv_bias=False, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_hidden_layers=2, num_attention_heads=2, hidden_dropout_prob=0.1, intermediate_size=20, layer_norm_eps=1e-05, hidden_act='gelu', initializer_range=0.02, vocab_size=32, do_stable_layer_norm=True, scope=None):
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
            return 10
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = Wav2Vec2Config(do_stable_layer_norm=self.do_stable_layer_norm, hidden_size=self.hidden_size, feat_extract_norm=self.feat_extract_norm, feat_extract_dropout=self.feat_extract_dropout, feat_extract_activation=self.feat_extract_activation, conv_dim=self.conv_dim, conv_stride=self.conv_stride, conv_kernel=self.conv_kernel, conv_bias=self.conv_bias, num_conv_pos_embeddings=self.num_conv_pos_embeddings, num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, hidden_dropout_prob=self.hidden_dropout_prob, intermediate_size=self.intermediate_size, layer_norm_eps=self.layer_norm_eps, hidden_act=self.hidden_act, initializer_range=self.initializer_range, vocab_size=self.vocab_size)
        return (config, input_values, attention_mask)

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_values, attention_mask) = config_and_inputs
        inputs_dict = {'input_values': input_values, 'attention_mask': attention_mask}
        return (config, inputs_dict)

@require_flax
class FlaxWav2Vec2ModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxWav2Vec2Model, FlaxWav2Vec2ForCTC, FlaxWav2Vec2ForPreTraining) if is_flax_available() else ()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = FlaxWav2Vec2ModelTester(self)

    def test_train(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        input_values = inputs_dict['input_values']
        attention_mask = inputs_dict['attention_mask']
        model = FlaxWav2Vec2ForPreTraining(config)
        features_shape = (input_values.shape[0], model._get_feat_extract_output_lengths(np.array(input_values.shape[1])))
        (batch_size, sequence_length) = features_shape[:2]
        mask_prob = 0.5
        mask_length = 4
        mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        (dropout_rng, gumbel_rng) = jax.random.split(jax.random.PRNGKey(0))
        output = model(input_values, attention_mask=attention_mask, mask_time_indices=mask_time_indices, train=True, dropout_rng=dropout_rng, gumbel_rng=gumbel_rng)[0]
        self.assertTrue(output.shape == (batch_size, sequence_length, model.config.proj_codevector_dim))

    def test_forward_signature(self):
        if False:
            return 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['input_values', 'attention_mask']
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_jit_compilation(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(input_values, attention_mask=None, **kwargs):
                    if False:
                        print('Hello World!')
                    return model(input_values=input_values, attention_mask=attention_mask, **kwargs)
                with self.subTest('JIT Enabled'):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()
                with self.subTest('JIT Disabled'):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(outputs), len(jitted_outputs))
                for (jitted_output, output) in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)

    def test_freeze_feature_encoder(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        input_values = inputs_dict['input_values']
        attention_mask = inputs_dict['attention_mask']
        model = FlaxWav2Vec2ForPreTraining(config)
        params = model.params

        def compute_loss(params, input_values, attention_mask, freeze_feature_encoder: bool=False, epsilon: float=1e-08):
            if False:
                while True:
                    i = 10
            outputs = model(input_values, attention_mask=attention_mask, freeze_feature_encoder=freeze_feature_encoder, params=params)
            cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, epsilon=epsilon)
            loss = cosine_sim.sum()
            return (loss, outputs.to_tuple())
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        ((loss, outputs), grads) = grad_fn(params, input_values, attention_mask, freeze_feature_encoder=False)
        ((loss_frozen, outputs_frozen), grads_frozen) = grad_fn(params, input_values, attention_mask, freeze_feature_encoder=True)
        for (output, output_frozen) in zip(outputs, outputs_frozen):
            self.assertTrue((output == output_frozen).all())
        self.assertEqual(loss, loss_frozen)
        grads = flatten_dict(grads)
        grads_frozen = flatten_dict(grads_frozen)
        self.assertEqual(grads.keys(), grads_frozen.keys())
        feature_extractor_grads = tuple((grads[k] for k in grads if 'feature_extractor' in k))
        feature_extractor_grads_frozen = tuple((grads_frozen[k] for k in grads_frozen if 'feature_extractor' in k))
        for (feature_extractor_grad, feature_extractor_grad_frozen) in zip(feature_extractor_grads, feature_extractor_grads_frozen):
            self.assertTrue((feature_extractor_grad_frozen == 0.0).all())
            self.assertTrue((feature_extractor_grad > 0.0).any())
        grads = tuple((grads[k] for k in grads if 'feature_extractor' not in k))
        grads_frozen = tuple((grads_frozen[k] for k in grads_frozen if 'feature_extractor' not in k))
        for (grad, grad_frozen) in zip(grads, grads_frozen):
            self.assertTrue((grad == grad_frozen).all())

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained('facebook/wav2vec2-large-960h-lv60-self', from_pt=True)
            outputs = model(np.ones((1, 1024), dtype='f4'))
            self.assertIsNotNone(outputs)

    @is_pt_flax_cross_test
    @is_flaky()
    def test_equivalence_pt_to_flax(self):
        if False:
            print('Hello World!')
        super().test_equivalence_pt_to_flax()

@require_flax
class FlaxWav2Vec2UtilsTest(unittest.TestCase):

    def test_compute_mask_indices(self):
        if False:
            while True:
                i = 10
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1
        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        self.assertListEqual(mask.sum(axis=-1).tolist(), [mask_prob * sequence_length for _ in range(batch_size)])

    def test_compute_mask_indices_overlap(self):
        if False:
            while True:
                i = 10
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4
        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

    def test_compute_mask_indices_attn_mask_overlap(self):
        if False:
            i = 10
            return i + 15
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4
        attention_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
        attention_mask[:2, sequence_length // 2:] = 0
        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length, attention_mask=attention_mask)
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)
        self.assertTrue(mask[:2, sequence_length // 2:].sum() == 0)

    def test_compute_perplexity(self):
        if False:
            i = 10
            return i + 15
        probs = np.arange(100).reshape(2, 5, 10) / 100
        ppl = FlaxWav2Vec2GumbelVectorQuantizer._compute_perplexity(probs)
        self.assertTrue(abs(ppl.item() - 141.4291) < 0.001)
        mask = np.ones((2,), dtype=bool)
        mask[0] = 0
        ppl = FlaxWav2Vec2GumbelVectorQuantizer._compute_perplexity(probs, mask)
        self.assertTrue(abs(ppl.item() - 58.6757) < 0.001)

    def test_sample_negatives(self):
        if False:
            i = 10
            return i + 15
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3
        features = (np.arange(sequence_length * hidden_size) // hidden_size).reshape(sequence_length, hidden_size)
        features = np.broadcast_to(features[None, :], (batch_size, sequence_length, hidden_size))
        negative_indices = _sample_negative_indices(features.shape, num_negatives)
        features = features.reshape(-1, hidden_size)
        sampled_negatives = features[negative_indices.reshape(-1)]
        negatives = sampled_negatives.reshape(batch_size, sequence_length, num_negatives, hidden_size).transpose(2, 0, 1, 3)
        self.assertTrue(negatives.shape == (num_negatives, batch_size, sequence_length, hidden_size))
        for negative in negatives:
            self.assertTrue((negative - features.reshape(negative.shape) == 0).sum() == 0.0)
        self.assertEqual(np.unique(negatives, axis=-1).shape, (num_negatives, batch_size, sequence_length, 1))

    def test_sample_negatives_with_attn_mask(self):
        if False:
            i = 10
            return i + 15
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3
        features = (np.arange(sequence_length * hidden_size) // hidden_size).reshape(sequence_length, hidden_size)
        attention_mask = np.ones((batch_size, sequence_length), dtype=np.int8)
        attention_mask[-1, sequence_length // 2:] = 0
        forbidden_indices = (np.arange(sequence_length // 2, sequence_length, dtype=np.int32) + (batch_size - 1) * sequence_length).tolist()
        features = np.broadcast_to(features[None, :], (batch_size, sequence_length, hidden_size))
        negative_indices = _sample_negative_indices(features.shape, num_negatives, attention_mask=attention_mask)
        self.assertTrue(all((idx not in negative_indices for idx in forbidden_indices)))
        features = features.reshape(-1, hidden_size)
        sampled_negatives = features[negative_indices.reshape(-1)]
        negatives = sampled_negatives.reshape(batch_size, sequence_length, num_negatives, hidden_size).transpose(2, 0, 1, 3)
        self.assertTrue(negatives.shape == (num_negatives, batch_size, sequence_length, hidden_size))
        for negative in negatives:
            self.assertTrue((negative - features.reshape(negative.shape) == 0).sum() == 0.0)
        self.assertEqual(np.unique(negatives, axis=-1).shape, (num_negatives, batch_size, sequence_length, 1))

@require_flax
@require_soundfile
@slow
class FlaxWav2Vec2ModelIntegrationTest(unittest.TestCase):

    def _load_datasamples(self, num_samples):
        if False:
            i = 10
            return i + 15
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').filter(lambda x: x['id'] in [f'1272-141231-000{i}' for i in range(num_samples)])[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def test_inference_ctc_robust_batched(self):
        if False:
            return 10
        model = FlaxWav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self', from_pt=True)
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self', do_lower_case=True)
        input_speech = self._load_datasamples(4)
        inputs = processor(input_speech, return_tensors='np', padding=True)
        input_values = inputs.input_values
        attention_mask = inputs.attention_mask
        logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = jnp.argmax(logits, axis=-1)
        predicted_trans = processor.batch_decode(predicted_ids)
        EXPECTED_TRANSCRIPTIONS = ['a man said to the universe sir i exist', "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore", 'the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about', 'his instant panic was followed by a small sharp blow high on his chest']
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_pretrained(self):
        if False:
            print('Hello World!')
        model = FlaxWav2Vec2ForPreTraining.from_pretrained('facebook/wav2vec2-large-lv60', from_pt=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-lv60', return_attention_mask=True)
        input_speech = self._load_datasamples(2)
        inputs_dict = feature_extractor(input_speech, return_tensors='np', padding=True)
        features_shape = (inputs_dict['input_values'].shape[0], model._get_feat_extract_output_lengths(np.array(inputs_dict['input_values'].shape[1])))
        mask_time_indices = _compute_mask_indices(features_shape, model.config.mask_time_prob, model.config.mask_time_length, min_masks=2)
        outputs = model(inputs_dict.input_values, attention_mask=inputs_dict.attention_mask, mask_time_indices=mask_time_indices)
        cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, epsilon=1e-08)
        cosine_sim_masked = cosine_sim[mask_time_indices]
        config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-large-lv60')
        model_rand = FlaxWav2Vec2ForPreTraining(config)
        outputs_rand = model_rand(inputs_dict.input_values, attention_mask=inputs_dict.attention_mask, mask_time_indices=mask_time_indices)
        cosine_sim_rand = optax.cosine_similarity(outputs_rand.projected_states, outputs_rand.projected_quantized_states)
        cosine_sim_masked_rand = cosine_sim_rand[mask_time_indices]
        self.assertTrue(cosine_sim_masked.mean().item() - 5 * cosine_sim_masked_rand.mean().item() > 0)

    @require_pyctcdecode
    @require_librosa
    def test_wav2vec2_with_lm(self):
        if False:
            i = 10
            return i + 15
        ds = load_dataset('common_voice', 'es', split='test', streaming=True)
        sample = next(iter(ds))
        resampled_audio = librosa.resample(sample['audio']['array'], 48000, 16000)
        model = FlaxWav2Vec2ForCTC.from_pretrained('patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm')
        processor = Wav2Vec2ProcessorWithLM.from_pretrained('patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm')
        input_values = processor(resampled_audio, return_tensors='np').input_values
        logits = model(input_values).logits
        transcription = processor.batch_decode(np.array(logits)).text
        self.assertEqual(transcription[0], 'bien y qué regalo vas a abrir primero')

    @require_pyctcdecode
    @require_librosa
    def test_wav2vec2_with_lm_pool(self):
        if False:
            return 10
        ds = load_dataset('common_voice', 'es', split='test', streaming=True)
        sample = next(iter(ds))
        resampled_audio = librosa.resample(sample['audio']['array'], 48000, 16000)
        model = FlaxWav2Vec2ForCTC.from_pretrained('patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm')
        processor = Wav2Vec2ProcessorWithLM.from_pretrained('patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm')
        input_values = processor(resampled_audio, return_tensors='np').input_values
        logits = model(input_values).logits
        with multiprocessing.get_context('fork').Pool(2) as pool:
            transcription = processor.batch_decode(np.array(logits), pool).text
        self.assertEqual(transcription[0], 'bien y qué regalo vas a abrir primero')
        with CaptureLogger(processing_wav2vec2_with_lm.logger) as cl, multiprocessing.get_context('fork').Pool(2) as pool:
            transcription = processor.batch_decode(np.array(logits), pool, num_processes=2).text
        self.assertIn('num_process', cl.out)
        self.assertIn('it will be ignored', cl.out)
        self.assertEqual(transcription[0], 'bien y qué regalo vas a abrir primero')

    @require_pyctcdecode
    @require_librosa
    def test_wav2vec2_with_lm_invalid_pool(self):
        if False:
            i = 10
            return i + 15
        run_test_in_subprocess(test_case=self, target_func=_test_wav2vec2_with_lm_invalid_pool, inputs=None)