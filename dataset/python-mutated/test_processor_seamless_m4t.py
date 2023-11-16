import shutil
import tempfile
import unittest
from transformers import SeamlessM4TFeatureExtractor, SeamlessM4TProcessor
from transformers.models.seamless_m4t import SeamlessM4TTokenizer, SeamlessM4TTokenizerFast
from transformers.testing_utils import require_torch
from .test_feature_extraction_seamless_m4t import floats_list

@require_torch
class SeamlessM4TProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.checkpoint = 'facebook/hf-seamless-m4t-medium'
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        if False:
            while True:
                i = 10
        return SeamlessM4TTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return SeamlessM4TFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        if False:
            return 10
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = SeamlessM4TProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        processor.save_pretrained(self.tmpdirname)
        processor = SeamlessM4TProcessor.from_pretrained(self.tmpdirname)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        tokenizer_instance = isinstance(processor.tokenizer, SeamlessM4TTokenizerFast) or isinstance(processor.tokenizer, SeamlessM4TTokenizer)
        self.assertTrue(tokenizer_instance)
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, SeamlessM4TFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        if False:
            return 10
        processor = SeamlessM4TProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)
        tokenizer_add_kwargs = self.get_tokenizer(bos_token='(BOS)', eos_token='(EOS)')
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)
        processor = SeamlessM4TProcessor.from_pretrained(self.tmpdirname, bos_token='(BOS)', eos_token='(EOS)', do_normalize=False, padding_value=1.0)
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, SeamlessM4TFeatureExtractor)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        tokenizer_instance = isinstance(processor.tokenizer, SeamlessM4TTokenizerFast) or isinstance(processor.tokenizer, SeamlessM4TTokenizer)
        self.assertTrue(tokenizer_instance)

    def test_feature_extractor(self):
        if False:
            print('Hello World!')
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = SeamlessM4TProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        raw_speech = floats_list((3, 1000))
        input_feat_extract = feature_extractor(raw_speech, return_tensors='np')
        input_processor = processor(audios=raw_speech, return_tensors='np')
        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=0.01)

    def test_tokenizer(self):
        if False:
            print('Hello World!')
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = SeamlessM4TProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        input_str = 'This is a test string'
        encoded_processor = processor(text=input_str)
        encoded_tok = tokenizer(input_str)
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        if False:
            for i in range(10):
                print('nop')
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        processor = SeamlessM4TProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]
        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)
        self.assertListEqual(decoded_tok, decoded_processor)