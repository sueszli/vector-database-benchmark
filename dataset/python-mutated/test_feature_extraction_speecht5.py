"""Tests for the SpeechT5 feature extractors."""
import itertools
import random
import unittest
import numpy as np
from transformers import BatchFeature, SpeechT5FeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin
if is_torch_available():
    import torch
global_rng = random.Random()

def floats_list(shape, scale=1.0, rng=None, name=None):
    if False:
        return 10
    'Creates a random float32 tensor'
    if rng is None:
        rng = global_rng
    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)
    return values

@require_torch
class SpeechT5FeatureExtractionTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, feature_size=1, padding_value=0.0, sampling_rate=16000, do_normalize=True, num_mel_bins=80, hop_length=16, win_length=64, win_function='hann_window', fmin=80, fmax=7600, mel_floor=1e-10, return_attention_mask=True):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.do_normalize = do_normalize
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.fmin = fmin
        self.fmax = fmax
        self.mel_floor = mel_floor
        self.return_attention_mask = return_attention_mask

    def prepare_feat_extract_dict(self):
        if False:
            print('Hello World!')
        return {'feature_size': self.feature_size, 'padding_value': self.padding_value, 'sampling_rate': self.sampling_rate, 'do_normalize': self.do_normalize, 'num_mel_bins': self.num_mel_bins, 'hop_length': self.hop_length, 'win_length': self.win_length, 'win_function': self.win_function, 'fmin': self.fmin, 'fmax': self.fmax, 'mel_floor': self.mel_floor, 'return_attention_mask': self.return_attention_mask}

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        if False:
            for i in range(10):
                print('nop')

        def _flatten(list_of_lists):
            if False:
                while True:
                    i = 10
            return list(itertools.chain(*list_of_lists))
        if equal_length:
            speech_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            speech_inputs = [_flatten(floats_list((x, self.feature_size))) for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs

    def prepare_inputs_for_target(self, equal_length=False, numpify=False):
        if False:
            for i in range(10):
                print('nop')
        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.num_mel_bins)) for _ in range(self.batch_size)]
        else:
            speech_inputs = [floats_list((x, self.num_mel_bins)) for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs

@require_torch
class SpeechT5FeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = SpeechT5FeatureExtractor

    def setUp(self):
        if False:
            return 10
        self.feat_extract_tester = SpeechT5FeatureExtractionTester(self)

    def _check_zero_mean_unit_variance(self, input_vector):
        if False:
            return 10
        self.assertTrue(np.all(np.mean(input_vector, axis=0) < 0.001))
        self.assertTrue(np.all(np.abs(np.var(input_vector, axis=0) - 1) < 0.001))

    def test_call(self):
        if False:
            while True:
                i = 10
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]
        encoded_sequences_1 = feat_extract(speech_inputs[0], return_tensors='np').input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs[0], return_tensors='np').input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=0.001))
        encoded_sequences_1 = feat_extract(speech_inputs, return_tensors='np').input_values
        encoded_sequences_2 = feat_extract(np_speech_inputs, return_tensors='np').input_values
        for (enc_seq_1, enc_seq_2) in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=0.001))

    def test_zero_mean_unit_variance_normalization_np(self):
        if False:
            while True:
                i = 10
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        paddings = ['longest', 'max_length', 'do_not_pad']
        max_lengths = [None, 1600, None]
        for (max_length, padding) in zip(max_lengths, paddings):
            processed = feat_extract(speech_inputs, padding=padding, max_length=max_length, return_tensors='np')
            input_values = processed.input_values
            self._check_zero_mean_unit_variance(input_values[0][:800])
            self.assertTrue(input_values[0][800:].sum() < 1e-06)
            self._check_zero_mean_unit_variance(input_values[1][:1000])
            self.assertTrue(input_values[0][1000:].sum() < 1e-06)
            self._check_zero_mean_unit_variance(input_values[2][:1200])

    def test_zero_mean_unit_variance_normalization(self):
        if False:
            while True:
                i = 10
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        lengths = range(800, 1400, 200)
        speech_inputs = [floats_list((1, x))[0] for x in lengths]
        paddings = ['longest', 'max_length', 'do_not_pad']
        max_lengths = [None, 1600, None]
        for (max_length, padding) in zip(max_lengths, paddings):
            processed = feat_extract(speech_inputs, max_length=max_length, padding=padding)
            input_values = processed.input_values
            self._check_zero_mean_unit_variance(input_values[0][:800])
            self._check_zero_mean_unit_variance(input_values[1][:1000])
            self._check_zero_mean_unit_variance(input_values[2][:1200])

    def test_zero_mean_unit_variance_normalization_trunc_np_max_length(self):
        if False:
            print('Hello World!')
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = feat_extract(speech_inputs, truncation=True, max_length=1000, padding='max_length', return_tensors='np')
        input_values = processed.input_values
        self._check_zero_mean_unit_variance(input_values[0, :800])
        self._check_zero_mean_unit_variance(input_values[1])
        self._check_zero_mean_unit_variance(input_values[2])

    def test_zero_mean_unit_variance_normalization_trunc_np_longest(self):
        if False:
            print('Hello World!')
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = feat_extract(speech_inputs, truncation=True, max_length=1000, padding='longest', return_tensors='np')
        input_values = processed.input_values
        self._check_zero_mean_unit_variance(input_values[0, :800])
        self._check_zero_mean_unit_variance(input_values[1, :1000])
        self._check_zero_mean_unit_variance(input_values[2])
        self.assertTrue(input_values.shape == (3, 1000))
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = feat_extract(speech_inputs, truncation=True, max_length=2000, padding='longest', return_tensors='np')
        input_values = processed.input_values
        self._check_zero_mean_unit_variance(input_values[0, :800])
        self._check_zero_mean_unit_variance(input_values[1, :1000])
        self._check_zero_mean_unit_variance(input_values[2])
        self.assertTrue(input_values.shape == (3, 1200))

    def test_double_precision_pad(self):
        if False:
            return 10
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()
        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{'input_values': inputs}], return_tensors='np')
            self.assertTrue(np_processed.input_values.dtype == np.float32)
            pt_processed = feature_extractor.pad([{'input_values': inputs}], return_tensors='pt')
            self.assertTrue(pt_processed.input_values.dtype == torch.float32)

    def test_call_target(self):
        if False:
            while True:
                i = 10
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]
        input_values = feature_extractor(audio_target=np_speech_inputs, padding=True, return_tensors='np').input_values
        self.assertTrue(input_values.ndim == 3)
        self.assertTrue(input_values.shape[-1] == feature_extractor.num_mel_bins)
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors='np').input_values
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors='np').input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=0.001))
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors='np').input_values
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors='np').input_values
        for (enc_seq_1, enc_seq_2) in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=0.001))
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors='np').input_values
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors='np').input_values
        for (enc_seq_1, enc_seq_2) in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=0.001))

    def test_batch_feature_target(self):
        if False:
            return 10
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_target()
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        input_name = feat_extract.model_input_names[0]
        processed_features = BatchFeature({input_name: speech_inputs})
        self.assertTrue(all((len(x) == len(y) for (x, y) in zip(speech_inputs, processed_features[input_name]))))
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_target(equal_length=True)
        processed_features = BatchFeature({input_name: speech_inputs}, tensor_type='np')
        batch_features_input = processed_features[input_name]
        if len(batch_features_input.shape) < 3:
            batch_features_input = batch_features_input[:, :, None]
        self.assertTrue(batch_features_input.shape == (self.feat_extract_tester.batch_size, len(speech_inputs[0]), self.feat_extract_tester.num_mel_bins))

    @require_torch
    def test_batch_feature_target_pt(self):
        if False:
            while True:
                i = 10
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_target(equal_length=True)
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        input_name = feat_extract.model_input_names[0]
        processed_features = BatchFeature({input_name: speech_inputs}, tensor_type='pt')
        batch_features_input = processed_features[input_name]
        if len(batch_features_input.shape) < 3:
            batch_features_input = batch_features_input[:, :, None]
        self.assertTrue(batch_features_input.shape == (self.feat_extract_tester.batch_size, len(speech_inputs[0]), self.feat_extract_tester.num_mel_bins))

    @require_torch
    def test_padding_accepts_tensors_target_pt(self):
        if False:
            for i in range(10):
                print('nop')
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_target()
        input_name = feat_extract.model_input_names[0]
        processed_features = BatchFeature({input_name: speech_inputs})
        feat_extract.feature_size = feat_extract.num_mel_bins
        input_np = feat_extract.pad(processed_features, padding='longest', return_tensors='np')[input_name]
        input_pt = feat_extract.pad(processed_features, padding='longest', return_tensors='pt')[input_name]
        self.assertTrue(abs(input_np.astype(np.float32).sum() - input_pt.numpy().astype(np.float32).sum()) < 0.01)

    def test_attention_mask_target(self):
        if False:
            while True:
                i = 10
        feat_dict = self.feat_extract_dict
        feat_dict['return_attention_mask'] = True
        feat_extract = self.feature_extraction_class(**feat_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_target()
        input_lengths = [len(x) for x in speech_inputs]
        input_name = feat_extract.model_input_names[0]
        processed = BatchFeature({input_name: speech_inputs})
        feat_extract.feature_size = feat_extract.num_mel_bins
        processed = feat_extract.pad(processed, padding='longest', return_tensors='np')
        self.assertIn('attention_mask', processed)
        self.assertListEqual(list(processed.attention_mask.shape), list(processed[input_name].shape[:2]))
        self.assertListEqual(processed.attention_mask.sum(-1).tolist(), input_lengths)

    def test_attention_mask_with_truncation_target(self):
        if False:
            print('Hello World!')
        feat_dict = self.feat_extract_dict
        feat_dict['return_attention_mask'] = True
        feat_extract = self.feature_extraction_class(**feat_dict)
        speech_inputs = self.feat_extract_tester.prepare_inputs_for_target()
        input_lengths = [len(x) for x in speech_inputs]
        input_name = feat_extract.model_input_names[0]
        processed = BatchFeature({input_name: speech_inputs})
        max_length = min(input_lengths)
        feat_extract.feature_size = feat_extract.num_mel_bins
        processed_pad = feat_extract.pad(processed, padding='max_length', max_length=max_length, truncation=True, return_tensors='np')
        self.assertIn('attention_mask', processed_pad)
        self.assertListEqual(list(processed_pad.attention_mask.shape), [processed_pad[input_name].shape[0], max_length])
        self.assertListEqual(processed_pad.attention_mask[:, :max_length].sum(-1).tolist(), [max_length for x in speech_inputs])

    def _load_datasamples(self, num_samples):
        if False:
            for i in range(10):
                print('nop')
        from datasets import load_dataset
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').select(range(num_samples))[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def test_integration(self):
        if False:
            while True:
                i = 10
        EXPECTED_INPUT_VALUES = torch.tensor([0.0023804, 0.0020752, 0.0019836, 0.0021057, 0.0016174, 0.00030518, 9.1553e-05, 0.00033569, 0.00097656, 0.0018311, 0.0020142, 0.0021057, 0.0017395, 0.00045776, -0.00039673, 0.00045776, 0.0010071, 9.1553e-05, 0.00048828, 0.0011597, 0.00073242, 0.00094604, 0.0018005, 0.0018311, 0.00088501, 0.00042725, 0.00048828, 0.00073242, 0.0010986, 0.0021057])
        input_speech = self._load_datasamples(1)
        feature_extractor = SpeechT5FeatureExtractor()
        input_values = feature_extractor(input_speech, return_tensors='pt').input_values
        self.assertEquals(input_values.shape, (1, 93680))
        self.assertTrue(torch.allclose(input_values[0, :30], EXPECTED_INPUT_VALUES, atol=1e-06))

    def test_integration_target(self):
        if False:
            for i in range(10):
                print('nop')
        EXPECTED_INPUT_VALUES = torch.tensor([-2.687, -3.0104, -3.1356, -3.5352, -3.0044, -3.0353, -3.4719, -3.6777, -3.152, -2.9435, -2.6553, -2.8795, -2.9944, -2.5921, -3.0279, -3.0386, -3.0864, -3.1291, -3.2353, -2.7444, -2.6831, -2.7287, -3.1761, -3.1571, -3.2726, -3.0582, -3.1007, -3.4533, -3.4695, -3.0998])
        input_speech = self._load_datasamples(1)
        feature_extractor = SpeechT5FeatureExtractor()
        input_values = feature_extractor(audio_target=input_speech, return_tensors='pt').input_values
        self.assertEquals(input_values.shape, (1, 366, 80))
        self.assertTrue(torch.allclose(input_values[0, 0, :30], EXPECTED_INPUT_VALUES, atol=0.0001))