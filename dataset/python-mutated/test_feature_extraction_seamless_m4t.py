import itertools
import os
import random
import tempfile
import unittest
import numpy as np
from datasets import load_dataset
from transformers import SeamlessM4TFeatureExtractor, is_speech_available
from transformers.testing_utils import check_json_file_has_correct_format, require_torch
from transformers.utils.import_utils import is_torch_available
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin
if is_torch_available():
    import torch
global_rng = random.Random()

def floats_list(shape, scale=1.0, rng=None, name=None):
    if False:
        i = 10
        return i + 15
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
class SeamlessM4TFeatureExtractionTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, feature_size=10, padding_value=0.0, sampling_rate=4000, return_attention_mask=True, do_normalize=True, stride=2):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.feature_size = feature_size
        self.stride = stride
        self.num_mel_bins = feature_size

    def prepare_feat_extract_dict(self):
        if False:
            print('Hello World!')
        return {'feature_size': self.feature_size, 'num_mel_bins': self.num_mel_bins, 'padding_value': self.padding_value, 'sampling_rate': self.sampling_rate, 'stride': self.stride, 'return_attention_mask': self.return_attention_mask, 'do_normalize': self.do_normalize}

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
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            speech_inputs = [floats_list((x, self.feature_size)) for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs

@require_torch
class SeamlessM4TFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = SeamlessM4TFeatureExtractor if is_speech_available() else None

    def setUp(self):
        if False:
            return 10
        self.feat_extract_tester = SeamlessM4TFeatureExtractionTester(self)

    def test_feat_extract_from_and_save_pretrained(self):
        if False:
            return 10
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)
        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertDictEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        if False:
            while True:
                i = 10
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, 'feat_extract.json')
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)
        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        self.assertEqual(dict_first, dict_second)

    def test_call(self):
        if False:
            print('Hello World!')
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]
        input_features = feature_extractor(np_speech_inputs, padding=True, return_tensors='np').input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[0] == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size * feature_extractor.stride)
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors='np').input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors='np').input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=0.001))
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors='np').input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors='np').input_features
        for (enc_seq_1, enc_seq_2) in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=0.001))
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors='np').input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors='np').input_features
        for (enc_seq_1, enc_seq_2) in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=0.001))

    @require_torch
    def test_double_precision_pad(self):
        if False:
            return 10
        import torch
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()
        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{'input_features': inputs}], return_tensors='np')
            self.assertTrue(np_processed.input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{'input_features': inputs}], return_tensors='pt')
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)

    def _load_datasample(self, id):
        if False:
            for i in range(10):
                print('nop')
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_sample = ds.sort('id')[id]['audio']['array']
        return torch.from_numpy(speech_sample).unsqueeze(0)

    def test_integration(self):
        if False:
            i = 10
            return i + 15
        EXPECTED_INPUT_FEATURES = torch.tensor([-1.5621, -1.4236, -1.3335, -1.3991, -1.2881, -1.1133, -0.971, -0.8895, -0.828, -0.7376, -0.7194, -0.6896, -0.6849, -0.6788, -0.6545, -0.661, -0.6566, -0.5738, -0.5252, -0.5533, -0.5887, -0.6116, -0.5971, -0.4956, -0.2881, -0.1512, 0.0299, 0.1762, 0.2728, 0.2236])
        input_speech = self._load_datasample(10)
        feature_extractor = SeamlessM4TFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors='pt').input_features
        feature_extractor(input_speech, return_tensors='pt').input_features[0, 5, :30]
        self.assertEqual(input_features.shape, (1, 279, 160))
        self.assertTrue(torch.allclose(input_features[0, 5, :30], EXPECTED_INPUT_FEATURES, atol=0.0001))

    def test_zero_mean_unit_variance_normalization_trunc_np_longest(self):
        if False:
            i = 10
            return i + 15
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio = self._load_datasample(1)
        audio = (audio - audio.min()) / (audio.max() - audio.min()) * 65535
        audio = feat_extract.zero_mean_unit_var_norm([audio], attention_mask=None)[0]
        self.assertTrue((audio.mean() < 0.001).all())
        self.assertTrue(((audio.var() - 1).abs() < 0.001).all())