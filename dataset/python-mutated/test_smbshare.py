import os
import shutil
from pathlib import Path
from unittest import TestCase, skipUnless
from golem.core.common import is_windows
from golem.docker import smbshare

@skipUnless(is_windows(), 'Windows only')
class TestGetShareName(TestCase):
    DEFAULT_SHARE_PATH = 'C:\\Users\\Public\\AppData\\Local\\golem\\golem\\default\\rinkeby\\ComputerRes'
    DEFAULT_SHARE_NAME = 'C97956C9B0D048CCC69B36413DBC994E'

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        current_path = Path('C:\\')
        for dirname in cls.DEFAULT_SHARE_PATH.split('\\')[1:]:
            current_path /= dirname
            if not current_path.is_dir():
                cls._path_to_remove = current_path
                break
        else:
            cls._path_to_remove = None
        os.makedirs(cls.DEFAULT_SHARE_PATH, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        shutil.rmtree(cls._path_to_remove)

    def _assert_share_name(self, path, share_name=DEFAULT_SHARE_NAME):
        if False:
            print('Hello World!')
        self.assertEqual(smbshare.get_share_name(Path(path)), share_name)

    def test_normal_path(self):
        if False:
            i = 10
            return i + 15
        self._assert_share_name(self.DEFAULT_SHARE_PATH)

    def test_trailing_backslash(self):
        if False:
            i = 10
            return i + 15
        self._assert_share_name('C:\\Users\\Public\\AppData\\Local\\golem\\golem\\default\\rinkeby\\ComputerRes\\')

    def test_slashes(self):
        if False:
            for i in range(10):
                print('nop')
        self._assert_share_name('C:/Users/Public/AppData/Local/golem/golem/default/rinkeby/ComputerRes')

    def test_dots(self):
        if False:
            print('Hello World!')
        self._assert_share_name('C:\\Users\\Public\\AppData\\Local\\golem\\golem\\default\\rinkeby\\ComputerRes\\.\\tmp\\..')

    def test_letter_case(self):
        if False:
            for i in range(10):
                print('nop')
        self._assert_share_name('c:\\USERS\\puBlic\\appdata\\LOCAL\\GoLeM\\golem\\DEFAULT\\rinkeBY\\computerres')

    def test_shortened_path(self):
        if False:
            i = 10
            return i + 15
        self._assert_share_name('C:\\Users\\Public\\AppData\\Local\\golem\\golem\\default\\rinkeby\\COMPUT~1')