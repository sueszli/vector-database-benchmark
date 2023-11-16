import os.path
import unittest
from test.picardtestcase import PicardTestCase
from picard.const.appdirs import cache_folder, config_folder, plugin_folder
from picard.const.sys import IS_LINUX, IS_MACOS, IS_WIN

class AppPathsTest(PicardTestCase):

    def assert_home_path_equals(self, expected, actual):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(os.path.normpath(os.path.expanduser(expected)), actual)

    @unittest.skipUnless(IS_WIN, 'Windows test')
    def test_config_folder_win(self):
        if False:
            print('Hello World!')
        self.assert_home_path_equals('~/AppData/Local/MusicBrainz/Picard', config_folder())

    @unittest.skipUnless(IS_MACOS, 'macOS test')
    def test_config_folder_macos(self):
        if False:
            print('Hello World!')
        self.assert_home_path_equals('~/Library/Preferences/MusicBrainz/Picard', config_folder())

    @unittest.skipUnless(IS_LINUX, 'Linux test')
    def test_config_folder_linux(self):
        if False:
            print('Hello World!')
        self.assert_home_path_equals('~/.config/MusicBrainz/Picard', config_folder())

    @unittest.skipUnless(IS_WIN, 'Windows test')
    def test_cache_folder_win(self):
        if False:
            return 10
        self.assert_home_path_equals('~/AppData/Local/MusicBrainz/Picard/cache', cache_folder())

    @unittest.skipUnless(IS_MACOS, 'macOS test')
    def test_cache_folder_macos(self):
        if False:
            i = 10
            return i + 15
        self.assert_home_path_equals('~/Library/Caches/MusicBrainz/Picard', cache_folder())

    @unittest.skipUnless(IS_LINUX, 'Linux test')
    def test_cache_folder_linux(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_home_path_equals('~/.cache/MusicBrainz/Picard', cache_folder())

    @unittest.skipUnless(IS_WIN, 'Windows test')
    def test_plugin_folder_win(self):
        if False:
            return 10
        self.assert_home_path_equals('~/AppData/Local/MusicBrainz/Picard/plugins', plugin_folder())

    @unittest.skipUnless(IS_MACOS, 'macOS test')
    def test_plugin_folder_macos(self):
        if False:
            print('Hello World!')
        self.assert_home_path_equals('~/Library/Preferences/MusicBrainz/Picard/plugins', plugin_folder())

    @unittest.skipUnless(IS_LINUX, 'Linux test')
    def test_plugin_folder_linux(self):
        if False:
            return 10
        self.assert_home_path_equals('~/.config/MusicBrainz/Picard/plugins', plugin_folder())