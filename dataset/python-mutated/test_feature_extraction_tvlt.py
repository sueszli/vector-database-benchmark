""" Testing suite for the TVLT feature extraction. """
import itertools
import random
import unittest
import numpy as np
from transformers import TvltFeatureExtractor, is_datasets_available
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.utils.import_utils import is_torch_available
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin
if is_torch_available():
    import torch
if is_datasets_available():
    from datasets import load_dataset
global_rng = random.Random()

def floats_list(shape, scale=1.0, rng=None, name=None):
    if False:
        print('Hello World!')
    'Creates a random float32 tensor'
    if rng is None:
        rng = global_rng
    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)
    return values

class TvltFeatureExtractionTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, spectrogram_length=2048, feature_size=128, num_audio_channels=1, hop_length=512, chunk_length=30, sampling_rate=44100):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.spectrogram_length = spectrogram_length
        self.feature_size = feature_size
        self.num_audio_channels = num_audio_channels
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate

    def prepare_feat_extract_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'spectrogram_length': self.spectrogram_length, 'feature_size': self.feature_size, 'num_audio_channels': self.num_audio_channels, 'hop_length': self.hop_length, 'chunk_length': self.chunk_length, 'sampling_rate': self.sampling_rate}

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        if False:
            for i in range(10):
                print('nop')

        def _flatten(list_of_lists):
            if False:
                i = 10
                return i + 15
            return list(itertools.chain(*list_of_lists))
        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            speech_inputs = [floats_list((x, self.feature_size)) for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs

@require_torch
@require_torchaudio
class TvltFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = TvltFeatureExtractor

    def setUp(self):
        if False:
            print('Hello World!')
        self.feat_extract_tester = TvltFeatureExtractionTester(self)

    def test_feat_extract_properties(self):
        if False:
            for i in range(10):
                print('nop')
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, 'spectrogram_length'))
        self.assertTrue(hasattr(feature_extractor, 'feature_size'))
        self.assertTrue(hasattr(feature_extractor, 'num_audio_channels'))
        self.assertTrue(hasattr(feature_extractor, 'hop_length'))
        self.assertTrue(hasattr(feature_extractor, 'chunk_length'))
        self.assertTrue(hasattr(feature_extractor, 'sampling_rate'))

    def test_call(self):
        if False:
            return 10
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]
        encoded_audios = feature_extractor(np_speech_inputs[0], return_tensors='np', sampling_rate=44100).audio_values
        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)
        encoded_audios = feature_extractor(np_speech_inputs, return_tensors='np', sampling_rate=44100).audio_values
        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)
        encoded_audios = feature_extractor(np_speech_inputs, return_tensors='np', sampling_rate=44100, mask_audio=True).audio_values
        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
        encoded_audios = feature_extractor(np_speech_inputs, return_tensors='np', sampling_rate=44100).audio_values
        self.assertTrue(encoded_audios.ndim == 4)
        self.assertTrue(encoded_audios.shape[-1] == feature_extractor.feature_size)
        self.assertTrue(encoded_audios.shape[-2] <= feature_extractor.spectrogram_length)
        self.assertTrue(encoded_audios.shape[-3] == feature_extractor.num_channels)

    def _load_datasamples(self, num_samples):
        if False:
            print('Hello World!')
        ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
        speech_samples = ds.sort('id').select(range(num_samples))[:num_samples]['audio']
        return [x['array'] for x in speech_samples]

    def test_integration(self):
        if False:
            i = 10
            return i + 15
        input_speech = self._load_datasamples(1)
        feature_extractor = TvltFeatureExtractor()
        audio_values = feature_extractor(input_speech, return_tensors='pt').audio_values
        self.assertEquals(audio_values.shape, (1, 1, 192, 128))
        expected_slice = torch.tensor([[-0.3032, -0.2708], [-0.4434, -0.4007]])
        self.assertTrue(torch.allclose(audio_values[0, 0, :2, :2], expected_slice, atol=0.0001))