import unittest
from unittest.mock import patch
import os
import tempfile
import warnings
import torch
import torch.hub as hub
from torch.testing._internal.common_utils import retry, IS_SANDCASTLE, TestCase

def sum_of_state_dict(state_dict):
    if False:
        return 10
    s = 0
    for v in state_dict.values():
        s += v.sum()
    return s
SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = 'https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones'

@unittest.skipIf(IS_SANDCASTLE, 'Sandcastle cannot ping external')
class TestHub(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.previous_hub_dir = torch.hub.get_dir()
        self.tmpdir = tempfile.TemporaryDirectory('hub_dir')
        torch.hub.set_dir(self.tmpdir.name)
        self.trusted_list_path = os.path.join(torch.hub.get_dir(), 'trusted_list')

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        torch.hub.set_dir(self.previous_hub_dir)
        self.tmpdir.cleanup()

    def _assert_trusted_list_is_empty(self):
        if False:
            return 10
        with open(self.trusted_list_path) as f:
            assert not f.readlines()

    def _assert_in_trusted_list(self, line):
        if False:
            return 10
        with open(self.trusted_list_path) as f:
            assert line in (l.strip() for l in f.readlines())

    @retry(Exception, tries=3)
    def test_load_from_github(self):
        if False:
            for i in range(10):
                print('nop')
        hub_model = hub.load('ailzhang/torchhub_example', 'mnist', source='github', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_local_dir(self):
        if False:
            return 10
        local_dir = hub._get_cache_or_reload('ailzhang/torchhub_example', force_reload=False, trust_repo=True, calling_fn=None)
        hub_model = hub.load(local_dir, 'mnist', source='local', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_branch(self):
        if False:
            print('Hello World!')
        hub_model = hub.load('ailzhang/torchhub_example:ci/test_slash', 'mnist', pretrained=True, verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_get_set_dir(self):
        if False:
            i = 10
            return i + 15
        previous_hub_dir = torch.hub.get_dir()
        with tempfile.TemporaryDirectory('hub_dir') as tmpdir:
            torch.hub.set_dir(tmpdir)
            self.assertEqual(torch.hub.get_dir(), tmpdir)
            self.assertNotEqual(previous_hub_dir, tmpdir)
            hub_model = hub.load('ailzhang/torchhub_example', 'mnist', pretrained=True, verbose=False)
            self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)
            assert os.path.exists(os.path.join(tmpdir, 'ailzhang_torchhub_example_master'))
        new_dir = os.path.join('~', 'hub')
        torch.hub.set_dir(new_dir)
        self.assertEqual(torch.hub.get_dir(), os.path.expanduser(new_dir))

    @retry(Exception, tries=3)
    def test_list_entrypoints(self):
        if False:
            print('Hello World!')
        entry_lists = hub.list('ailzhang/torchhub_example', trust_repo=True)
        self.assertObjectIn('mnist', entry_lists)

    @retry(Exception, tries=3)
    def test_download_url_to_file(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'temp')
            hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, f, progress=False)
            loaded_state = torch.load(f)
            self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
            f_ref = os.path.join(tmpdir, 'reference')
            open(f_ref, 'w').close()
            expected_permissions = oct(os.stat(f_ref).st_mode & 511)
            actual_permissions = oct(os.stat(f).st_mode & 511)
            assert actual_permissions == expected_permissions

    @retry(Exception, tries=3)
    def test_load_state_dict_from_url(self):
        if False:
            print('Hello World!')
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
        file_name = 'the_file_name'
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name)
        expected_file_path = os.path.join(torch.hub.get_dir(), 'checkpoints', file_name)
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL, weights_only=True)
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_legacy_zip_checkpoint(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            hub_model = hub.load('ailzhang/torchhub_example', 'mnist_zip', pretrained=True, verbose=False)
            self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)
            assert any(('will be deprecated in favor of default zipfile' in str(w) for w in ws))

    @retry(Exception, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        if False:
            i = 10
            return i + 15
        hub_model = hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', pretrained=True, verbose=False, trust_repo=True)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_hub_parse_repo_info(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(torch.hub._parse_repo_info('a/b:c'), ('a', 'b', 'c'))
        self.assertEqual(torch.hub._parse_repo_info('pytorch/vision'), ('pytorch', 'vision', 'main'))
        self.assertEqual(torch.hub._parse_repo_info('ailzhang/torchhub_example'), ('ailzhang', 'torchhub_example', 'master'))

    @retry(Exception, tries=3)
    def test_load_commit_from_forked_repo(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, "If it's a commit from a forked repo"):
            torch.hub.load('pytorch/vision:4e2c216', 'resnet18')

    @retry(Exception, tries=3)
    @patch('builtins.input', return_value='')
    def test_trust_repo_false_emptystring(self, patched_input):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(Exception, 'Untrusted repository.'):
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=False)
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()
        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, 'Untrusted repository.'):
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=False)
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch('builtins.input', return_value='no')
    def test_trust_repo_false_no(self, patched_input):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, 'Untrusted repository.'):
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=False)
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()
        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, 'Untrusted repository.'):
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=False)
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch('builtins.input', return_value='y')
    def test_trusted_repo_false_yes(self, patched_input):
        if False:
            i = 10
            return i + 15
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=False)
        self._assert_in_trusted_list('ailzhang_torchhub_example')
        patched_input.assert_called_once()
        patched_input.reset_mock()
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo='check')
        patched_input.assert_not_called()
        patched_input.reset_mock()
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=False)
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch('builtins.input', return_value='no')
    def test_trust_repo_check_no(self, patched_input):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(Exception, 'Untrusted repository.'):
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo='check')
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()
        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, 'Untrusted repository.'):
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo='check')
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch('builtins.input', return_value='y')
    def test_trust_repo_check_yes(self, patched_input):
        if False:
            i = 10
            return i + 15
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo='check')
        self._assert_in_trusted_list('ailzhang_torchhub_example')
        patched_input.assert_called_once()
        patched_input.reset_mock()
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo='check')
        patched_input.assert_not_called()

    @retry(Exception, tries=3)
    def test_trust_repo_true(self):
        if False:
            for i in range(10):
                print('nop')
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=True)
        self._assert_in_trusted_list('ailzhang_torchhub_example')

    @retry(Exception, tries=3)
    def test_trust_repo_builtin_trusted_owners(self):
        if False:
            return 10
        torch.hub.load('pytorch/vision', 'resnet18', trust_repo='check')
        self._assert_trusted_list_is_empty()

    @retry(Exception, tries=3)
    def test_trust_repo_none(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=None)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'You are about to download and run code from an untrusted repository' in str(w[-1].message)
        self._assert_trusted_list_is_empty()

    @retry(Exception, tries=3)
    def test_trust_repo_legacy(self):
        if False:
            i = 10
            return i + 15
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo=True)
        os.remove(self.trusted_list_path)
        torch.hub.load('ailzhang/torchhub_example', 'mnist_zip_1_6', trust_repo='check')
        self._assert_trusted_list_is_empty()