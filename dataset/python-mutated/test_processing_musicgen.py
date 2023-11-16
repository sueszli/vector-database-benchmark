"""Tests for the MusicGen processor."""
import random
import shutil
import tempfile
import unittest
import numpy as np
from transformers import T5Tokenizer, T5TokenizerFast
from transformers.testing_utils import require_sentencepiece, require_torch
from transformers.utils.import_utils import is_speech_available, is_torch_available
if is_torch_available():
    pass
if is_speech_available():
    from transformers import EncodecFeatureExtractor, MusicgenProcessor
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
@require_sentencepiece
class MusicgenProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.checkpoint = 'facebook/musicgen-small'
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return T5Tokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        if False:
            return 10
        return EncodecFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        if False:
            return 10
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = MusicgenProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        processor.save_pretrained(self.tmpdirname)
        processor = MusicgenProcessor.from_pretrained(self.tmpdirname)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, T5TokenizerFast)
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, EncodecFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        if False:
            print('Hello World!')
        processor = MusicgenProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)
        tokenizer_add_kwargs = self.get_tokenizer(bos_token='(BOS)', eos_token='(EOS)')
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)
        processor = MusicgenProcessor.from_pretrained(self.tmpdirname, bos_token='(BOS)', eos_token='(EOS)', do_normalize=False, padding_value=1.0)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, T5TokenizerFast)
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, EncodecFeatureExtractor)

    def test_feature_extractor(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = MusicgenProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        raw_speech = floats_list((3, 1000))
        input_feat_extract = feature_extractor(raw_speech, return_tensors='np')
        input_processor = processor(raw_speech, return_tensors='np')
        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=0.01)

    def test_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = MusicgenProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        input_str = 'This is a test string'
        encoded_processor = processor(text=input_str)
        encoded_tok = tokenizer(input_str)
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = MusicgenProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]
        decoded_processor = processor.batch_decode(sequences=predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)
        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        if False:
            while True:
                i = 10
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = MusicgenProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.assertListEqual(processor.model_input_names, feature_extractor.model_input_names, msg='`processor` and `feature_extractor` model input names do not match')

    def test_decode_audio(self):
        if False:
            print('Hello World!')
        feature_extractor = self.get_feature_extractor(padding_side='left')
        tokenizer = self.get_tokenizer()
        processor = MusicgenProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        raw_speech = [floats_list((1, x))[0] for x in range(5, 20, 5)]
        padding_mask = processor(raw_speech).padding_mask
        generated_speech = np.asarray(floats_list((3, 20)))[:, None, :]
        decoded_audios = processor.batch_decode(generated_speech, padding_mask=padding_mask)
        self.assertIsInstance(decoded_audios, list)
        for audio in decoded_audios:
            self.assertIsInstance(audio, np.ndarray)
        self.assertTrue(decoded_audios[0].shape == (1, 10))
        self.assertTrue(decoded_audios[1].shape == (1, 15))
        self.assertTrue(decoded_audios[2].shape == (1, 20))