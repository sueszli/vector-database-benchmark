import json
import os
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from requests.exceptions import HTTPError
from transformers.utils import CONFIG_NAME, FLAX_WEIGHTS_NAME, TF2_WEIGHTS_NAME, TRANSFORMERS_CACHE, WEIGHTS_NAME, cached_file, get_file_from_repo, has_file
RANDOM_BERT = 'hf-internal-testing/tiny-random-bert'
CACHE_DIR = os.path.join(TRANSFORMERS_CACHE, 'models--hf-internal-testing--tiny-random-bert')
FULL_COMMIT_HASH = '9b8c223d42b2188cb49d29af482996f9d0f3e5a6'
GATED_REPO = 'hf-internal-testing/dummy-gated-model'
README_FILE = 'README.md'

class GetFromCacheTests(unittest.TestCase):

    def test_cached_file(self):
        if False:
            for i in range(10):
                print('nop')
        archive_file = cached_file(RANDOM_BERT, CONFIG_NAME)
        self.assertTrue(os.path.isdir(CACHE_DIR))
        for subfolder in ['blobs', 'refs', 'snapshots']:
            self.assertTrue(os.path.isdir(os.path.join(CACHE_DIR, subfolder)))
        with open(os.path.join(CACHE_DIR, 'refs', 'main')) as f:
            main_commit = f.read()
        self.assertEqual(archive_file, os.path.join(CACHE_DIR, 'snapshots', main_commit, CONFIG_NAME))
        self.assertTrue(os.path.isfile(archive_file))
        new_archive_file = cached_file(RANDOM_BERT, CONFIG_NAME)
        self.assertEqual(archive_file, new_archive_file)
        archive_file = cached_file(RANDOM_BERT, CONFIG_NAME, revision='9b8c223')
        self.assertEqual(archive_file, os.path.join(CACHE_DIR, 'snapshots', FULL_COMMIT_HASH, CONFIG_NAME))

    def test_cached_file_errors(self):
        if False:
            return 10
        with self.assertRaisesRegex(EnvironmentError, 'is not a valid model identifier'):
            _ = cached_file('tiny-random-bert', CONFIG_NAME)
        with self.assertRaisesRegex(EnvironmentError, 'is not a valid git identifier'):
            _ = cached_file(RANDOM_BERT, CONFIG_NAME, revision='aaaa')
        with self.assertRaisesRegex(EnvironmentError, 'does not appear to have a file named'):
            _ = cached_file(RANDOM_BERT, 'conf')

    def test_non_existence_is_cached(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(EnvironmentError, 'does not appear to have a file named'):
            _ = cached_file(RANDOM_BERT, 'conf')
        with open(os.path.join(CACHE_DIR, 'refs', 'main')) as f:
            main_commit = f.read()
        self.assertTrue(os.path.isfile(os.path.join(CACHE_DIR, '.no_exist', main_commit, 'conf')))
        path = cached_file(RANDOM_BERT, 'conf', _raise_exceptions_for_missing_entries=False)
        self.assertIsNone(path)
        path = cached_file(RANDOM_BERT, 'conf', local_files_only=True, _raise_exceptions_for_missing_entries=False)
        self.assertIsNone(path)
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            path = cached_file(RANDOM_BERT, 'conf', _raise_exceptions_for_connection_errors=False)
            self.assertIsNone(path)
            mock_head.assert_called()

    def test_has_file(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(has_file('hf-internal-testing/tiny-bert-pt-only', WEIGHTS_NAME))
        self.assertFalse(has_file('hf-internal-testing/tiny-bert-pt-only', TF2_WEIGHTS_NAME))
        self.assertFalse(has_file('hf-internal-testing/tiny-bert-pt-only', FLAX_WEIGHTS_NAME))

    def test_get_file_from_repo_distant(self):
        if False:
            print('Hello World!')
        self.assertIsNone(get_file_from_repo('bert-base-cased', 'ahah.txt'))
        with self.assertRaisesRegex(EnvironmentError, 'is not a valid model identifier'):
            get_file_from_repo('bert-base-case', CONFIG_NAME)
        with self.assertRaisesRegex(EnvironmentError, 'is not a valid git identifier'):
            get_file_from_repo('bert-base-cased', CONFIG_NAME, revision='ahaha')
        resolved_file = get_file_from_repo('bert-base-cased', CONFIG_NAME)
        config = json.loads(open(resolved_file, 'r').read())
        self.assertEqual(config['hidden_size'], 768)

    def test_get_file_from_repo_local(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / 'a.txt'
            filename.touch()
            self.assertEqual(get_file_from_repo(tmp_dir, 'a.txt'), str(filename))
            self.assertIsNone(get_file_from_repo(tmp_dir, 'b.txt'))

    def test_get_file_gated_repo(self):
        if False:
            i = 10
            return i + 15
        'Test download file from a gated repo fails with correct message when not authenticated.'
        with self.assertRaisesRegex(EnvironmentError, 'You are trying to access a gated repo.'):
            cached_file(GATED_REPO, 'gated_file.txt', token=False)

    def test_has_file_gated_repo(self):
        if False:
            i = 10
            return i + 15
        'Test check file existence from a gated repo fails with correct message when not authenticated.'
        with self.assertRaisesRegex(EnvironmentError, 'is a gated repository'):
            has_file(GATED_REPO, 'gated_file.txt', token=False)