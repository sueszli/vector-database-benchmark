import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from huggingface_hub import HfFolder, delete_repo
from requests.exceptions import HTTPError
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers.testing_utils import TOKEN, USER, get_tests_dir, is_staging_test
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from test_module.custom_feature_extraction import CustomFeatureExtractor
SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir('fixtures')

class FeatureExtractorUtilTester(unittest.TestCase):

    def test_cached_files_are_used_when_internet_is_down(self):
        if False:
            while True:
                i = 10
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        _ = Wav2Vec2FeatureExtractor.from_pretrained('hf-internal-testing/tiny-random-wav2vec2')
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            _ = Wav2Vec2FeatureExtractor.from_pretrained('hf-internal-testing/tiny-random-wav2vec2')
            mock_head.assert_called()

    def test_legacy_load_from_url(self):
        if False:
            for i in range(10):
                print('nop')
        _ = Wav2Vec2FeatureExtractor.from_pretrained('https://huggingface.co/hf-internal-testing/tiny-random-wav2vec2/resolve/main/preprocessor_config.json')

@is_staging_test
class FeatureExtractorPushToHubTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        try:
            delete_repo(token=cls._token, repo_id='test-feature-extractor')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='valid_org/test-feature-extractor-org')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='test-dynamic-feature-extractor')
        except HTTPError:
            pass

    def test_push_to_hub(self):
        if False:
            print('Hello World!')
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        feature_extractor.push_to_hub('test-feature-extractor', token=self._token)
        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f'{USER}/test-feature-extractor')
        for (k, v) in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))
        delete_repo(token=self._token, repo_id='test-feature-extractor')
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(tmp_dir, repo_id='test-feature-extractor', push_to_hub=True, token=self._token)
        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f'{USER}/test-feature-extractor')
        for (k, v) in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_in_organization(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        feature_extractor.push_to_hub('valid_org/test-feature-extractor', token=self._token)
        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('valid_org/test-feature-extractor')
        for (k, v) in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))
        delete_repo(token=self._token, repo_id='valid_org/test-feature-extractor')
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(tmp_dir, repo_id='valid_org/test-feature-extractor-org', push_to_hub=True, token=self._token)
        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('valid_org/test-feature-extractor-org')
        for (k, v) in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_dynamic_feature_extractor(self):
        if False:
            while True:
                i = 10
        CustomFeatureExtractor.register_for_auto_class()
        feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        feature_extractor.push_to_hub('test-dynamic-feature-extractor', token=self._token)
        self.assertDictEqual(feature_extractor.auto_map, {'AutoFeatureExtractor': 'custom_feature_extraction.CustomFeatureExtractor'})
        new_feature_extractor = AutoFeatureExtractor.from_pretrained(f'{USER}/test-dynamic-feature-extractor', trust_remote_code=True)
        self.assertEqual(new_feature_extractor.__class__.__name__, 'CustomFeatureExtractor')