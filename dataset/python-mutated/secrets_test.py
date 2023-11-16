"""st.secrets unit tests."""
import os
import tempfile
import unittest
from collections.abc import Mapping as MappingABC
from collections.abc import MutableMapping as MutableMappingABC
from typing import Mapping, MutableMapping
from unittest.mock import MagicMock, mock_open, patch
from parameterized import parameterized
from toml import TomlDecodeError
from streamlit.runtime.secrets import SECRETS_FILE_LOCS, AttrDict, Secrets
from tests.exception_capturing_thread import call_on_threads
MOCK_TOML = '\n# Everything in this section will be available as an environment variable\ndb_username="Jane"\ndb_password="12345qwerty"\n\n# Sub-sections are not loaded into os.environ\n[subsection]\nemail="eng@streamlit.io"\n'
MOCK_SECRETS_FILE_LOC = '/mock/secrets.toml'

class SecretsTest(unittest.TestCase):
    """Tests for st.secrets with a single secrets.toml file"""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self._prev_environ = dict(os.environ)
        self.secrets = Secrets([MOCK_SECRETS_FILE_LOC])

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        os.environ.clear()
        os.environ.update(self._prev_environ)

    @patch('streamlit.watcher.path_watcher.watch_file')
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_access_secrets(self, *mocks):
        if False:
            while True:
                i = 10
        self.assertEqual(self.secrets['db_username'], 'Jane')
        self.assertEqual(self.secrets['subsection']['email'], 'eng@streamlit.io')
        self.assertEqual(self.secrets['subsection'].email, 'eng@streamlit.io')

    @parameterized.expand([[False, "Secrets(file_paths=['/mock/secrets.toml'])"], [True, "{'db_username': 'Jane', 'db_password': '12345qwerty', 'subsection': {'email': 'eng@streamlit.io'}}"]])
    @patch('streamlit.watcher.path_watcher.watch_file')
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_repr_secrets(self, runtime_exists, secrets_repr, *mocks):
        if False:
            while True:
                i = 10
        with patch('streamlit.runtime.exists', return_value=runtime_exists):
            self.assertEqual(repr(self.secrets), secrets_repr)

    @patch('streamlit.watcher.path_watcher.watch_file')
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_access_secrets_via_attribute(self, *mocks):
        if False:
            print('Hello World!')
        self.assertEqual(self.secrets.db_username, 'Jane')
        self.assertEqual(self.secrets.subsection['email'], 'eng@streamlit.io')
        self.assertEqual(self.secrets.subsection.email, 'eng@streamlit.io')

    def test_secrets_file_location(self):
        if False:
            return 10
        "Verify that we're looking for secrets.toml in the right place."
        self.assertEqual(['/mock/home/folder/.streamlit/secrets.toml', os.path.abspath('./.streamlit/secrets.toml')], SECRETS_FILE_LOCS)

    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_os_environ(self, _):
        if False:
            while True:
                i = 10
        'os.environ gets patched when we load our secrets.toml'
        self.assertEqual(os.environ.get('db_username'), None)
        self.secrets.load_if_toml_exists()
        self.assertEqual(os.environ['db_username'], 'Jane')
        self.assertEqual(os.environ['db_password'], '12345qwerty')
        self.assertEqual(os.environ.get('subsection'), None)

    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_load_if_toml_exists_returns_true_if_parse_succeeds(self, _):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.secrets.load_if_toml_exists())

    def test_load_if_toml_exists_returns_false_if_parse_fails(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.secrets.load_if_toml_exists())

    @patch('streamlit.error')
    def test_missing_toml_error(self, mock_st_error):
        if False:
            return 10
        'Secrets access raises an error, and calls st.error, if\n        secrets.toml is missing.\n        '
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = FileNotFoundError()
            with self.assertRaises(FileNotFoundError):
                self.secrets.get('no_such_secret', None)
        mock_st_error.assert_called_once_with(f'No secrets files found. Valid paths for a secrets.toml file are: {MOCK_SECRETS_FILE_LOC}')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid_toml')
    @patch('streamlit.error')
    def test_malformed_toml_error(self, mock_st_error, _):
        if False:
            while True:
                i = 10
        'Secrets access raises an error, and calls st.error, if\n        secrets.toml is malformed.\n        '
        with self.assertRaises(TomlDecodeError):
            self.secrets.get('no_such_secret', None)
        mock_st_error.assert_called_once_with(f'Error parsing secrets file at /mock/secrets.toml')

    @patch('streamlit.watcher.path_watcher.watch_file')
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_getattr_nonexistent(self, *mocks):
        if False:
            return 10
        'Verify that access to missing attribute raises  AttributeError.'
        with self.assertRaises(AttributeError):
            self.secrets.nonexistent_secret
        with self.assertRaises(AttributeError):
            self.secrets.subsection.nonexistent_secret

    @patch('streamlit.watcher.path_watcher.watch_file')
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_getattr_raises_exception_on_attr_dict(self, *mocks):
        if False:
            for i in range(10):
                print('nop')
        'Verify that assignment to nested secrets raises TypeError.'
        with self.assertRaises(TypeError):
            self.secrets.subsection['new_secret'] = '123'
        with self.assertRaises(TypeError):
            self.secrets.subsection.new_secret = '123'

    @patch('streamlit.watcher.path_watcher.watch_file')
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_getitem_nonexistent(self, *mocks):
        if False:
            i = 10
            return i + 15
        'Verify that access to missing key via dict notation raises KeyError.'
        with self.assertRaises(KeyError):
            self.secrets['nonexistent_secret']
        with self.assertRaises(KeyError):
            self.secrets['subsection']['nonexistent_secret']

    @patch('streamlit.watcher.path_watcher.watch_file')
    def test_reload_secrets_when_file_changes(self, mock_watch_file):
        if False:
            while True:
                i = 10
        'When secrets.toml is loaded, the secrets file gets watched.'
        with patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML):
            self.assertEqual('Jane', self.secrets['db_username'])
            self.assertEqual('12345qwerty', self.secrets['db_password'])
            self.assertEqual('Jane', os.environ['db_username'])
            self.assertEqual('12345qwerty', os.environ['db_password'])
        mock_watch_file.assert_called_once_with(MOCK_SECRETS_FILE_LOC, self.secrets._on_secrets_file_changed, watcher_type='poll')
        self.secrets.file_change_listener.send = MagicMock()
        new_mock_toml = "db_username='Joan'"
        with patch('builtins.open', new_callable=mock_open, read_data=new_mock_toml):
            self.secrets._on_secrets_file_changed(MOCK_SECRETS_FILE_LOC)
            self.secrets.file_change_listener.send.assert_called_once()
            self.assertEqual('Joan', self.secrets['db_username'])
            self.assertIsNone(self.secrets.get('db_password'))
            self.assertEqual('Joan', os.environ['db_username'])
            self.assertIsNone(os.environ.get('db_password'))

class MultipleSecretsFilesTest(unittest.TestCase):
    """Tests for st.secrets with multiple secrets.toml files."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        (self._fd1, self._path1) = tempfile.mkstemp()
        (self._fd2, self._path2) = tempfile.mkstemp()
        self._prev_environ = dict(os.environ)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        os.environ.clear()
        os.environ.update(self._prev_environ)
        os.remove(self._path1)
        os.remove(self._path2)

    @patch('streamlit.error')
    def test_no_secrets_files_explodes(self, mock_st_error):
        if False:
            for i in range(10):
                print('nop')
        'Validate that an error is thrown if none of the given secrets.toml files exist.'
        secrets_file_locations = ['/mock1/secrets.toml', '/mock2/secrets.toml']
        secrets = Secrets(secrets_file_locations)
        with self.assertRaises(FileNotFoundError):
            secrets.get('no_such_secret', None)
        mock_st_error.assert_called_once_with(f'No secrets files found. Valid paths for a secrets.toml file are: /mock1/secrets.toml, /mock2/secrets.toml')

    @patch('streamlit.runtime.secrets._LOGGER')
    def test_only_one_secrets_file_fine(self, patched_logger):
        if False:
            i = 10
            return i + 15
        with os.fdopen(self._fd1, 'w') as tmp:
            tmp.write(MOCK_TOML)
        secrets_file_locations = [self._path1, '/mock2/secrets.toml']
        secrets = Secrets(secrets_file_locations)
        self.assertEqual(secrets.db_username, 'Jane')
        patched_logger.info.assert_not_called()

    @patch('streamlit.runtime.secrets._LOGGER')
    def test_secret_overwriting(self, patched_logger):
        if False:
            return 10
        'Test that if both global and project-level secrets.toml files exist, secrets\n        from both are present in st.secrets, and secrets from the project-level file\n        "win" when secrets have conflicting names.\n        '
        with os.fdopen(self._fd1, 'w') as tmp:
            tmp.write(MOCK_TOML)
        with os.fdopen(self._fd2, 'w') as tmp:
            tmp.write('\ndb_password="54321dvorak"\nhi="I\'m new"\n\n[subsection]\nemail2="eng2@streamlit.io"\n')
        secrets_file_locations = [self._path1, self._path2]
        secrets = Secrets(secrets_file_locations)
        self.assertEqual(secrets.db_username, 'Jane')
        self.assertEqual(secrets.db_password, '54321dvorak')
        self.assertEqual(secrets.hi, "I'm new")
        self.assertEqual(secrets.subsection, {'email2': 'eng2@streamlit.io'})
        patched_logger.info.assert_called_once_with(f'Secrets found in multiple locations: {self._path1}, {self._path2}. When multiple secret.toml files exist, local secrets will take precedence over global secrets.')

class SecretsThreadingTests(unittest.TestCase):
    NUM_THREADS = 50

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._prev_environ = dict(os.environ)
        self.secrets = Secrets(MOCK_SECRETS_FILE_LOC)

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        os.environ.clear()
        os.environ.update(self._prev_environ)

    @patch('streamlit.watcher.path_watcher.watch_file', MagicMock())
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_access_secrets(self, _):
        if False:
            return 10
        'Accessing secrets is thread-safe.'

        def access_secrets(_: int) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.secrets['db_username'], 'Jane')
            self.assertEqual(self.secrets['subsection']['email'], 'eng@streamlit.io')
            self.assertEqual(self.secrets['subsection'].email, 'eng@streamlit.io')
        call_on_threads(access_secrets, num_threads=self.NUM_THREADS)

    @patch('streamlit.watcher.path_watcher.watch_file', MagicMock())
    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_reload_secrets(self, _):
        if False:
            for i in range(10):
                print('nop')
        'Re-parsing the secrets file is thread-safe.'

        def reload_secrets(_: int) -> None:
            if False:
                return 10
            self.secrets._reset()
            self.assertEqual(self.secrets['db_username'], 'Jane')
        call_on_threads(reload_secrets, num_threads=self.NUM_THREADS)

class AttrDictTest(unittest.TestCase):

    def test_attr_dict_is_mapping_but_not_built_in_dict(self):
        if False:
            print('Hello World!')
        'Verify that AttrDict implements Mapping, but not built-in Dict'
        attr_dict = AttrDict({'x': {'y': 'z'}})
        self.assertIsInstance(attr_dict.x, Mapping)
        self.assertIsInstance(attr_dict.x, MappingABC)
        self.assertNotIsInstance(attr_dict.x, MutableMapping)
        self.assertNotIsInstance(attr_dict.x, MutableMappingABC)
        self.assertNotIsInstance(attr_dict.x, dict)

    def test_attr_dict_to_dict(self):
        if False:
            return 10
        d = {'x': {'y': 'z'}}
        attr_dict = AttrDict(d)
        assert attr_dict.to_dict() == d
        attr_dict.to_dict()['x']['y'] = 'zed'
        assert attr_dict.x.y == 'z'
        assert d['x']['y'] == 'z'