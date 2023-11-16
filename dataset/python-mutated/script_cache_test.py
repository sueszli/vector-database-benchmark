import os.path
import unittest
from unittest import mock
from unittest.mock import Mock
from streamlit import source_util
from streamlit.runtime.scriptrunner.script_cache import ScriptCache

def _get_script_path(name: str) -> str:
    if False:
        while True:
            i = 10
    return os.path.join(os.path.dirname(__file__), 'test_data', name)

class ScriptCacheTest(unittest.TestCase):

    def test_load_valid_script(self):
        if False:
            return 10
        '`get_bytecode` works as expected.'
        cache = ScriptCache()
        result = cache.get_bytecode(_get_script_path('good_script.py'))
        self.assertIsNotNone(result)
        exec(result)

    @mock.patch('streamlit.runtime.scriptrunner.script_cache.open_python_file')
    def test_returns_cached_data(self, mock_open_python_file: Mock):
        if False:
            return 10
        '`get_bytecode` caches its results.'
        mock_open_python_file.side_effect = source_util.open_python_file
        cache = ScriptCache()
        result = cache.get_bytecode(_get_script_path('good_script.py'))
        self.assertIsNotNone(result)
        mock_open_python_file.assert_called_once()
        mock_open_python_file.reset_mock()
        self.assertIs(cache.get_bytecode(_get_script_path('good_script.py')), result)
        mock_open_python_file.assert_not_called()

    def test_clear(self):
        if False:
            return 10
        '`clear` removes cached entries.'
        cache = ScriptCache()
        cache.get_bytecode(_get_script_path('good_script.py'))
        self.assertEqual(1, len(cache._cache))
        cache.clear()
        self.assertEqual(0, len(cache._cache))

    def test_file_not_found_error(self):
        if False:
            i = 10
            return i + 15
        "An exception is thrown when a script file doesn't exist."
        cache = ScriptCache()
        with self.assertRaises(FileNotFoundError):
            cache.get_bytecode(_get_script_path('not_a_valid_path.py'))

    def test_syntax_error(self):
        if False:
            print('Hello World!')
        'An exception is thrown when a script has a compile error.'
        cache = ScriptCache()
        with self.assertRaises(SyntaxError):
            cache.get_bytecode(_get_script_path('compile_error.py.txt'))