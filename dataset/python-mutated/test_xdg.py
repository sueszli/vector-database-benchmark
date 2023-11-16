import os
import stat
import sys
from .base import TempAppDirTestCase
from http_prompt import xdg

class TestXDG(TempAppDirTestCase):

    def test_get_app_data_home(self):
        if False:
            while True:
                i = 10
        path = xdg.get_data_dir()
        expected_path = os.path.join(os.environ[self.homes['data']], 'http-prompt')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.exists(path))
        if sys.platform != 'win32':
            mask = stat.S_IMODE(os.stat(path).st_mode)
            self.assertTrue(mask & stat.S_IRWXU)
            self.assertFalse(mask & stat.S_IRWXG)
            self.assertFalse(mask & stat.S_IRWXO)

    def test_get_app_config_home(self):
        if False:
            i = 10
            return i + 15
        path = xdg.get_config_dir()
        expected_path = os.path.join(os.environ[self.homes['config']], 'http-prompt')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.exists(path))
        if sys.platform != 'win32':
            mask = stat.S_IMODE(os.stat(path).st_mode)
            self.assertTrue(mask & stat.S_IRWXU)
            self.assertFalse(mask & stat.S_IRWXG)
            self.assertFalse(mask & stat.S_IRWXO)

    def test_get_resource_data_dir(self):
        if False:
            for i in range(10):
                print('nop')
        path = xdg.get_data_dir('something')
        expected_path = os.path.join(os.environ[self.homes['data']], 'http-prompt', 'something')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.exists(path))
        with open(os.path.join(path, 'test'), 'wb') as f:
            f.write(b'hello')

    def test_get_resource_config_dir(self):
        if False:
            i = 10
            return i + 15
        path = xdg.get_config_dir('something')
        expected_path = os.path.join(os.environ[self.homes['config']], 'http-prompt', 'something')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.exists(path))
        with open(os.path.join(path, 'test'), 'wb') as f:
            f.write(b'hello')