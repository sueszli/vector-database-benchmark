from __future__ import annotations
from units.mock.procenv import ModuleTestCase
from unittest.mock import patch
import builtins
realimport = builtins.__import__

class TestGetModulePath(ModuleTestCase):

    def test_module_utils_basic_get_module_path(self):
        if False:
            i = 10
            return i + 15
        from ansible.module_utils.basic import get_module_path
        with patch('os.path.realpath', return_value='/path/to/foo/'):
            self.assertEqual(get_module_path(), '/path/to/foo')