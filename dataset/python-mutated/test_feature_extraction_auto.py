import json
import sys
import tempfile
import unittest
from pathlib import Path
import transformers
from transformers import CONFIG_MAPPING, FEATURE_EXTRACTOR_MAPPING, AutoConfig, AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, get_tests_dir
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig
from test_module.custom_feature_extraction import CustomFeatureExtractor
SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir('fixtures')
SAMPLE_FEATURE_EXTRACTION_CONFIG = get_tests_dir('fixtures/dummy_feature_extractor_config.json')
SAMPLE_CONFIG = get_tests_dir('fixtures/dummy-config.json')

class AutoFeatureExtractorTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_feature_extractor_from_model_shortcut(self):
        if False:
            print('Hello World!')
        config = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_directory_from_key(self):
        if False:
            return 10
        config = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_directory_from_config(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()
            config_dict = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR).to_dict()
            config_dict.pop('feature_extractor_type')
            config = Wav2Vec2FeatureExtractor(**config_dict)
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)
            config = AutoFeatureExtractor.from_pretrained(tmpdirname)
            dict_as_saved = json.loads(config.to_json_string())
            self.assertTrue('_processor_class' not in dict_as_saved)
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_file(self):
        if False:
            return 10
        config = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG)
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_repo_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(EnvironmentError, 'bert-base is not a local folder and is not a valid model identifier'):
            _ = AutoFeatureExtractor.from_pretrained('bert-base')

    def test_revision_not_found(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(EnvironmentError, 'aaaaaa is not a valid git identifier \\(branch name, tag name or commit id\\)'):
            _ = AutoFeatureExtractor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision='aaaaaa')

    def test_feature_extractor_not_found(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(EnvironmentError, 'hf-internal-testing/config-no-model does not appear to have a file named preprocessor_config.json.'):
            _ = AutoFeatureExtractor.from_pretrained('hf-internal-testing/config-no-model')

    def test_from_pretrained_dynamic_feature_extractor(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            feature_extractor = AutoFeatureExtractor.from_pretrained('hf-internal-testing/test_dynamic_feature_extractor')
        with self.assertRaises(ValueError):
            feature_extractor = AutoFeatureExtractor.from_pretrained('hf-internal-testing/test_dynamic_feature_extractor', trust_remote_code=False)
        feature_extractor = AutoFeatureExtractor.from_pretrained('hf-internal-testing/test_dynamic_feature_extractor', trust_remote_code=True)
        self.assertEqual(feature_extractor.__class__.__name__, 'NewFeatureExtractor')
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(tmp_dir)
            reloaded_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertEqual(reloaded_feature_extractor.__class__.__name__, 'NewFeatureExtractor')

    def test_new_feature_extractor_registration(self):
        if False:
            return 10
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, CustomFeatureExtractor)
            with self.assertRaises(ValueError):
                AutoFeatureExtractor.register(Wav2Vec2Config, Wav2Vec2FeatureExtractor)
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                feature_extractor.save_pretrained(tmp_dir)
                new_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_feature_extractor, CustomFeatureExtractor)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_feature_extractor_conflict(self):
        if False:
            while True:
                i = 10

        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            is_local = True
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            feature_extractor = AutoFeatureExtractor.from_pretrained('hf-internal-testing/test_dynamic_feature_extractor')
            self.assertEqual(feature_extractor.__class__.__name__, 'NewFeatureExtractor')
            self.assertTrue(feature_extractor.is_local)
            feature_extractor = AutoFeatureExtractor.from_pretrained('hf-internal-testing/test_dynamic_feature_extractor', trust_remote_code=False)
            self.assertEqual(feature_extractor.__class__.__name__, 'NewFeatureExtractor')
            self.assertTrue(feature_extractor.is_local)
            feature_extractor = AutoFeatureExtractor.from_pretrained('hf-internal-testing/test_dynamic_feature_extractor', trust_remote_code=True)
            self.assertEqual(feature_extractor.__class__.__name__, 'NewFeatureExtractor')
            self.assertTrue(not hasattr(feature_extractor, 'is_local'))
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]