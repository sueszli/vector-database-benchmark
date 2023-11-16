import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile
from huggingface_hub import HfFolder, Repository, create_repo, delete_repo
from requests.exceptions import HTTPError
import transformers
from transformers import CONFIG_MAPPING, FEATURE_EXTRACTOR_MAPPING, PROCESSOR_MAPPING, TOKENIZER_MAPPING, AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoTokenizer, BertTokenizer, ProcessorMixin, Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.testing_utils import TOKEN, USER, get_tests_dir, is_staging_test
from transformers.tokenization_utils import TOKENIZER_CONFIG_FILE
from transformers.utils import FEATURE_EXTRACTOR_NAME, is_tokenizers_available
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig
from test_module.custom_feature_extraction import CustomFeatureExtractor
from test_module.custom_processing import CustomProcessor
from test_module.custom_tokenization import CustomTokenizer
SAMPLE_PROCESSOR_CONFIG = get_tests_dir('fixtures/dummy_feature_extractor_config.json')
SAMPLE_VOCAB = get_tests_dir('fixtures/vocab.json')
SAMPLE_PROCESSOR_CONFIG_DIR = get_tests_dir('fixtures')

class AutoFeatureExtractorTest(unittest.TestCase):
    vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', 'bla', 'blou']

    def setUp(self):
        if False:
            while True:
                i = 10
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_processor_from_model_shortcut(self):
        if False:
            return 10
        processor = AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_repo(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()
            processor = AutoProcessor.from_pretrained('facebook/wav2vec2-base-960h')
            model_config.save_pretrained(tmpdirname)
            processor.save_pretrained(tmpdirname)
            processor = AutoProcessor.from_pretrained(tmpdirname)
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_extractor_config(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            copyfile(SAMPLE_PROCESSOR_CONFIG, os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME))
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, 'vocab.json'))
            processor = AutoProcessor.from_pretrained(tmpdirname)
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_feat_extr_processor_class(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
            processor = Wav2Vec2Processor(feature_extractor, tokenizer)
            processor.save_pretrained(tmpdirname)
            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), 'r') as f:
                config_dict = json.load(f)
                config_dict.pop('processor_class')
            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), 'w') as f:
                f.write(json.dumps(config_dict))
            processor = AutoProcessor.from_pretrained(tmpdirname)
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_tokenizer_processor_class(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
            processor = Wav2Vec2Processor(feature_extractor, tokenizer)
            processor.save_pretrained(tmpdirname)
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), 'r') as f:
                config_dict = json.load(f)
                config_dict.pop('processor_class')
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), 'w') as f:
                f.write(json.dumps(config_dict))
            processor = AutoProcessor.from_pretrained(tmpdirname)
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_model_config(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config(processor_class='Wav2Vec2Processor')
            model_config.save_pretrained(tmpdirname)
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, 'vocab.json'))
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), 'w') as f:
                f.write('{}')
            processor = AutoProcessor.from_pretrained(tmpdirname)
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_from_pretrained_dynamic_processor(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor')
        with self.assertRaises(ValueError):
            processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor', trust_remote_code=False)
        processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor', trust_remote_code=True)
        self.assertTrue(processor.special_attribute_present)
        self.assertEqual(processor.__class__.__name__, 'NewProcessor')
        feature_extractor = processor.feature_extractor
        self.assertTrue(feature_extractor.special_attribute_present)
        self.assertEqual(feature_extractor.__class__.__name__, 'NewFeatureExtractor')
        tokenizer = processor.tokenizer
        self.assertTrue(tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizerFast')
            new_processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor', trust_remote_code=True, use_fast=False)
            new_tokenizer = new_processor.tokenizer
            self.assertTrue(new_tokenizer.special_attribute_present)
            self.assertEqual(new_tokenizer.__class__.__name__, 'NewTokenizer')
        else:
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')

    def test_new_processor_registration(self):
        if False:
            return 10
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, CustomFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            AutoProcessor.register(CustomConfig, CustomProcessor)
            with self.assertRaises(ValueError):
                AutoProcessor.register(Wav2Vec2Config, Wav2Vec2Processor)
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                vocab_file = os.path.join(tmp_dir, 'vocab.txt')
                with open(vocab_file, 'w', encoding='utf-8') as vocab_writer:
                    vocab_writer.write(''.join([x + '\n' for x in self.vocab_tokens]))
                tokenizer = CustomTokenizer(vocab_file)
            processor = CustomProcessor(feature_extractor, tokenizer)
            with tempfile.TemporaryDirectory() as tmp_dir:
                processor.save_pretrained(tmp_dir)
                new_processor = AutoProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_processor, CustomProcessor)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_processor_conflict(self):
        if False:
            return 10

        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            special_attribute_present = False

        class NewTokenizer(BertTokenizer):
            special_attribute_present = False

        class NewProcessor(ProcessorMixin):
            feature_extractor_class = 'AutoFeatureExtractor'
            tokenizer_class = 'AutoTokenizer'
            special_attribute_present = False
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoProcessor.register(CustomConfig, NewProcessor)
            processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor')
            self.assertEqual(processor.__class__.__name__, 'NewProcessor')
            self.assertFalse(processor.special_attribute_present)
            self.assertFalse(processor.feature_extractor.special_attribute_present)
            self.assertFalse(processor.tokenizer.special_attribute_present)
            processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor', trust_remote_code=False)
            self.assertEqual(processor.__class__.__name__, 'NewProcessor')
            self.assertFalse(processor.special_attribute_present)
            self.assertFalse(processor.feature_extractor.special_attribute_present)
            self.assertFalse(processor.tokenizer.special_attribute_present)
            processor = AutoProcessor.from_pretrained('hf-internal-testing/test_dynamic_processor', trust_remote_code=True)
            self.assertEqual(processor.__class__.__name__, 'NewProcessor')
            self.assertTrue(processor.special_attribute_present)
            self.assertTrue(processor.feature_extractor.special_attribute_present)
            self.assertTrue(processor.tokenizer.special_attribute_present)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]
            if CustomConfig in PROCESSOR_MAPPING._extra_content:
                del PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_auto_processor_creates_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        processor = AutoProcessor.from_pretrained('hf-internal-testing/tiny-random-bert')
        self.assertEqual(processor.__class__.__name__, 'BertTokenizerFast')

    def test_auto_processor_creates_image_processor(self):
        if False:
            while True:
                i = 10
        processor = AutoProcessor.from_pretrained('hf-internal-testing/tiny-random-convnext')
        self.assertEqual(processor.__class__.__name__, 'ConvNextImageProcessor')

@is_staging_test
class ProcessorPushToHubTester(unittest.TestCase):
    vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', 'bla', 'blou']

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        try:
            delete_repo(token=cls._token, repo_id='test-processor')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='valid_org/test-processor-org')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='test-dynamic-processor')
        except HTTPError:
            pass

    def test_push_to_hub(self):
        if False:
            print('Hello World!')
        processor = Wav2Vec2Processor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(os.path.join(tmp_dir, 'test-processor'), push_to_hub=True, token=self._token)
            new_processor = Wav2Vec2Processor.from_pretrained(f'{USER}/test-processor')
            for (k, v) in processor.feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_processor.feature_extractor, k))
            self.assertDictEqual(new_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())

    def test_push_to_hub_in_organization(self):
        if False:
            for i in range(10):
                print('nop')
        processor = Wav2Vec2Processor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(os.path.join(tmp_dir, 'test-processor-org'), push_to_hub=True, token=self._token, organization='valid_org')
            new_processor = Wav2Vec2Processor.from_pretrained('valid_org/test-processor-org')
            for (k, v) in processor.feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_processor.feature_extractor, k))
            self.assertDictEqual(new_processor.tokenizer.get_vocab(), processor.tokenizer.get_vocab())

    def test_push_to_hub_dynamic_processor(self):
        if False:
            print('Hello World!')
        CustomFeatureExtractor.register_for_auto_class()
        CustomTokenizer.register_for_auto_class()
        CustomProcessor.register_for_auto_class()
        feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_PROCESSOR_CONFIG_DIR)
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, 'vocab.txt')
            with open(vocab_file, 'w', encoding='utf-8') as vocab_writer:
                vocab_writer.write(''.join([x + '\n' for x in self.vocab_tokens]))
            tokenizer = CustomTokenizer(vocab_file)
        processor = CustomProcessor(feature_extractor, tokenizer)
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_repo(f'{USER}/test-dynamic-processor', token=self._token)
            repo = Repository(tmp_dir, clone_from=f'{USER}/test-dynamic-processor', token=self._token)
            processor.save_pretrained(tmp_dir)
            self.assertDictEqual(processor.feature_extractor.auto_map, {'AutoFeatureExtractor': 'custom_feature_extraction.CustomFeatureExtractor', 'AutoProcessor': 'custom_processing.CustomProcessor'})
            with open(os.path.join(tmp_dir, 'tokenizer_config.json')) as f:
                tokenizer_config = json.load(f)
            self.assertDictEqual(tokenizer_config['auto_map'], {'AutoTokenizer': ['custom_tokenization.CustomTokenizer', None], 'AutoProcessor': 'custom_processing.CustomProcessor'})
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'custom_feature_extraction.py')))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'custom_tokenization.py')))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'custom_processing.py')))
            repo.push_to_hub()
        new_processor = AutoProcessor.from_pretrained(f'{USER}/test-dynamic-processor', trust_remote_code=True)
        self.assertEqual(new_processor.__class__.__name__, 'CustomProcessor')