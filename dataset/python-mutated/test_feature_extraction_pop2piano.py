import os
import tempfile
import unittest
import numpy as np
from datasets import load_dataset
from transformers.testing_utils import check_json_file_has_correct_format, require_essentia, require_librosa, require_scipy, require_tf, require_torch
from transformers.utils.import_utils import is_essentia_available, is_librosa_available, is_scipy_available, is_torch_available
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin
requirements_available = is_torch_available() and is_essentia_available() and is_scipy_available() and is_librosa_available()
if requirements_available:
    import torch
    from transformers import Pop2PianoFeatureExtractor

class Pop2PianoFeatureExtractionTester(unittest.TestCase):

    def __init__(self, parent, n_bars=2, sample_rate=22050, use_mel=True, padding_value=0, vocab_size_special=4, vocab_size_note=128, vocab_size_velocity=2, vocab_size_time=100):
        if False:
            print('Hello World!')
        self.parent = parent
        self.n_bars = n_bars
        self.sample_rate = sample_rate
        self.use_mel = use_mel
        self.padding_value = padding_value
        self.vocab_size_special = vocab_size_special
        self.vocab_size_note = vocab_size_note
        self.vocab_size_velocity = vocab_size_velocity
        self.vocab_size_time = vocab_size_time

    def prepare_feat_extract_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'n_bars': self.n_bars, 'sample_rate': self.sample_rate, 'use_mel': self.use_mel, 'padding_value': self.padding_value, 'vocab_size_special': self.vocab_size_special, 'vocab_size_note': self.vocab_size_note, 'vocab_size_velocity': self.vocab_size_velocity, 'vocab_size_time': self.vocab_size_time}

@require_torch
@require_essentia
@require_librosa
@require_scipy
class Pop2PianoFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = Pop2PianoFeatureExtractor if requirements_available else None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.feat_extract_tester = Pop2PianoFeatureExtractionTester(self)

    def test_feat_extract_from_and_save_pretrained(self):
        if False:
            print('Hello World!')
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)
        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = feat_extract_first.use_mel
        mel_2 = feat_extract_second.use_mel
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_feat_extract_to_json_file(self):
        if False:
            for i in range(10):
                print('nop')
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, 'feat_extract.json')
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)
        dict_first = feat_extract_first.to_dict()
        dict_second = feat_extract_second.to_dict()
        mel_1 = feat_extract_first.use_mel
        mel_2 = feat_extract_second.use_mel
        self.assertTrue(np.allclose(mel_1, mel_2))
        self.assertEqual(dict_first, dict_second)

    def test_call(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input = np.zeros([1000000], dtype=np.float32)
        input_features = feature_extractor(speech_input, sampling_rate=16000, return_tensors='np')
        self.assertTrue(input_features.input_features.ndim == 3)
        self.assertEqual(input_features.input_features.shape[-1], 512)
        self.assertTrue(input_features.beatsteps.ndim == 2)
        self.assertTrue(input_features.extrapolated_beatstep.ndim == 2)

    def test_integration(self):
        if False:
            print('Hello World!')
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').select([0])['audio']
        input_speech = [x['array'] for x in speech_samples][0]
        sampling_rate = [x['sampling_rate'] for x in speech_samples][0]
        feaure_extractor = Pop2PianoFeatureExtractor.from_pretrained('sweetcocoa/pop2piano')
        input_features = feaure_extractor(input_speech, sampling_rate=sampling_rate, return_tensors='pt').input_features
        EXPECTED_INPUT_FEATURES = torch.tensor([[-7.1493, -6.8701, -4.3214], [-5.9473, -5.7548, -3.8438], [-6.1324, -5.9018, -4.3778]])
        self.assertTrue(torch.allclose(input_features[0, :3, :3], EXPECTED_INPUT_FEATURES, atol=0.0001))

    def test_attention_mask(self):
        if False:
            print('Hello World!')
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input1 = np.zeros([1000000], dtype=np.float32)
        speech_input2 = np.random.randint(low=0, high=10, size=500000).astype(np.float32)
        input_features = feature_extractor([speech_input1, speech_input2], sampling_rate=[44100, 16000], return_tensors='np', return_attention_mask=True)
        self.assertTrue(hasattr(input_features, 'attention_mask'))
        self.assertTrue(input_features['attention_mask'].ndim == 2)
        self.assertEqual(input_features['attention_mask_beatsteps'].shape[0], 2)
        self.assertEqual(input_features['attention_mask_extrapolated_beatstep'].shape[0], 2)
        self.assertTrue(np.max(input_features['attention_mask']) == 1)
        self.assertTrue(np.max(input_features['attention_mask_beatsteps']) == 1)
        self.assertTrue(np.max(input_features['attention_mask_extrapolated_beatstep']) == 1)
        self.assertTrue(np.min(input_features['attention_mask']) == 0)
        self.assertTrue(np.min(input_features['attention_mask_beatsteps']) == 0)
        self.assertTrue(np.min(input_features['attention_mask_extrapolated_beatstep']) == 0)

    def test_batch_feature(self):
        if False:
            while True:
                i = 10
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input1 = np.zeros([1000000], dtype=np.float32)
        speech_input2 = np.ones([2000000], dtype=np.float32)
        speech_input3 = np.random.randint(low=0, high=10, size=500000).astype(np.float32)
        input_features = feature_extractor([speech_input1, speech_input2, speech_input3], sampling_rate=[44100, 16000, 48000], return_attention_mask=True)
        self.assertEqual(len(input_features['input_features'].shape), 3)
        self.assertEqual(input_features['beatsteps'].shape[0], 3)
        self.assertEqual(input_features['extrapolated_beatstep'].shape[0], 3)

    def test_batch_feature_np(self):
        if False:
            return 10
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input1 = np.zeros([1000000], dtype=np.float32)
        speech_input2 = np.ones([2000000], dtype=np.float32)
        speech_input3 = np.random.randint(low=0, high=10, size=500000).astype(np.float32)
        input_features = feature_extractor([speech_input1, speech_input2, speech_input3], sampling_rate=[44100, 16000, 48000], return_tensors='np', return_attention_mask=True)
        self.assertEqual(type(input_features['input_features']), np.ndarray)
        self.assertEqual(len(input_features['input_features'].shape), 3)

    def test_batch_feature_pt(self):
        if False:
            print('Hello World!')
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input1 = np.zeros([1000000], dtype=np.float32)
        speech_input2 = np.ones([2000000], dtype=np.float32)
        speech_input3 = np.random.randint(low=0, high=10, size=500000).astype(np.float32)
        input_features = feature_extractor([speech_input1, speech_input2, speech_input3], sampling_rate=[44100, 16000, 48000], return_tensors='pt', return_attention_mask=True)
        self.assertEqual(type(input_features['input_features']), torch.Tensor)
        self.assertEqual(len(input_features['input_features'].shape), 3)

    @require_tf
    def test_batch_feature_tf(self):
        if False:
            while True:
                i = 10
        import tensorflow as tf
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input1 = np.zeros([1000000], dtype=np.float32)
        speech_input2 = np.ones([2000000], dtype=np.float32)
        speech_input3 = np.random.randint(low=0, high=10, size=500000).astype(np.float32)
        input_features = feature_extractor([speech_input1, speech_input2, speech_input3], sampling_rate=[44100, 16000, 48000], return_tensors='tf', return_attention_mask=True)
        self.assertTrue(tf.is_tensor(input_features['input_features']))
        self.assertEqual(len(input_features['input_features'].shape), 3)

    @unittest.skip('Pop2PianoFeatureExtractor does not supports padding externally (while processing audios in batches padding is automatically applied to max_length)')
    def test_padding_accepts_tensors_pt(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip('Pop2PianoFeatureExtractor does not supports padding externally (while processing audios in batches padding is automatically applied to max_length)')
    def test_padding_accepts_tensors_tf(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip('Pop2PianoFeatureExtractor does not supports padding externally (while processing audios in batches padding is automatically applied to max_length)')
    def test_padding_from_list(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip('Pop2PianoFeatureExtractor does not supports padding externally (while processing audios in batches padding is automatically applied to max_length)')
    def test_padding_from_array(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip('Pop2PianoFeatureExtractor does not support truncation')
    def test_attention_mask_with_truncation(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip('Pop2PianoFeatureExtractor does not supports truncation')
    def test_truncation_from_array(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip('Pop2PianoFeatureExtractor does not supports truncation')
    def test_truncation_from_list(self):
        if False:
            for i in range(10):
                print('nop')
        pass