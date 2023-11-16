import ast
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import IO
from unittest.mock import MagicMock, mock_open, patch
from .. import module_loader

class ModuleLoaderTest(unittest.TestCase):

    @patch('builtins.open')
    def test_load_module(self, open: MagicMock) -> None:
        if False:
            for i in range(10):
                print('nop')
        valid_path: str = '/valid'
        invalid_syntax_path: str = '/syntax'
        invalid_path: str = '/invalid'
        valid_syntax: str = textwrap.dedent('\n            def my_function():\n                pass\n        ')
        invalid_syntax: str = textwrap.dedent('\n            def: () my_function:\n                pass\n        ')

        def _open_implementation(path: str, mode: str) -> IO[str]:
            if False:
                i = 10
                return i + 15
            if path == valid_path:
                return mock_open(read_data=valid_syntax).return_value
            elif path == invalid_syntax_path:
                return mock_open(read_data=invalid_syntax).return_value
            else:
                raise FileNotFoundError(path)
        open.side_effect = _open_implementation
        module = module_loader.load_module(valid_path)
        self.assertIsInstance(module, ast.Module)
        self.assertEqual(module.body[0].name, 'my_function')
        module = module_loader.load_module(invalid_syntax_path)
        self.assertIsNone(module)
        module = module_loader.load_module(invalid_path)
        self.assertIsNone(module)

    def test_find_all_paths(self) -> None:
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            nested_directories = directory / 'dir/dir/dir/dir/'
            stub_directory = directory / 'stubs'
            nested_directories.mkdir(parents=True)
            stub_directory.mkdir()
            garbage_file = directory / 'garbage.yp'
            no_nest = directory / 'file.py'
            one_nest = directory / 'dir/file.py'
            many_nest = directory / 'dir/dir/dir/dir/file.py'
            py_file = directory / 'stubs/file.py'
            pyi_file = directory / 'stubs/file.pyi'
            garbage_file.touch()
            no_nest.touch()
            one_nest.touch()
            many_nest.touch()
            py_file.touch()
            pyi_file.touch()
            self.assertListEqual(sorted([str(no_nest), str(one_nest), str(many_nest), str(pyi_file)]), sorted(module_loader.find_all_paths(directory_name)))