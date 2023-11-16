import ctypes
import os
import sys
import unittest
from test.helper import TestHelper
from beets import util

class FetchartCliTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_beets()
        self.load_plugins('fetchart')
        self.config['fetchart']['cover_names'] = 'cÃ¶ver.jpg'
        self.config['art_filename'] = 'mycover'
        self.album = self.add_album()
        self.cover_path = os.path.join(self.album.path, b'mycover.jpg')

    def tearDown(self):
        if False:
            return 10
        self.unload_plugins()
        self.teardown_beets()

    def check_cover_is_stored(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.album['artpath'], self.cover_path)
        with open(util.syspath(self.cover_path)) as f:
            self.assertEqual(f.read(), 'IMAGE')

    def hide_file_windows(self):
        if False:
            for i in range(10):
                print('nop')
        hidden_mask = 2
        success = ctypes.windll.kernel32.SetFileAttributesW(self.cover_path, hidden_mask)
        if not success:
            self.skipTest('unable to set file attributes')

    def test_set_art_from_folder(self):
        if False:
            for i in range(10):
                print('nop')
        self.touch(b'c\xc3\xb6ver.jpg', dir=self.album.path, content='IMAGE')
        self.run_command('fetchart')
        self.album.load()
        self.check_cover_is_stored()

    def test_filesystem_does_not_pick_up_folder(self):
        if False:
            print('Hello World!')
        os.makedirs(os.path.join(self.album.path, b'mycover.jpg'))
        self.run_command('fetchart')
        self.album.load()
        self.assertEqual(self.album['artpath'], None)

    def test_filesystem_does_not_pick_up_ignored_file(self):
        if False:
            i = 10
            return i + 15
        self.touch(b'co_ver.jpg', dir=self.album.path, content='IMAGE')
        self.config['ignore'] = ['*_*']
        self.run_command('fetchart')
        self.album.load()
        self.assertEqual(self.album['artpath'], None)

    def test_filesystem_picks_up_non_ignored_file(self):
        if False:
            while True:
                i = 10
        self.touch(b'cover.jpg', dir=self.album.path, content='IMAGE')
        self.config['ignore'] = ['*_*']
        self.run_command('fetchart')
        self.album.load()
        self.check_cover_is_stored()

    def test_filesystem_does_not_pick_up_hidden_file(self):
        if False:
            for i in range(10):
                print('nop')
        self.touch(b'.cover.jpg', dir=self.album.path, content='IMAGE')
        if sys.platform == 'win32':
            self.hide_file_windows()
        self.config['ignore'] = []
        self.config['ignore_hidden'] = True
        self.run_command('fetchart')
        self.album.load()
        self.assertEqual(self.album['artpath'], None)

    def test_filesystem_picks_up_non_hidden_file(self):
        if False:
            return 10
        self.touch(b'cover.jpg', dir=self.album.path, content='IMAGE')
        self.config['ignore_hidden'] = True
        self.run_command('fetchart')
        self.album.load()
        self.check_cover_is_stored()

    def test_filesystem_picks_up_hidden_file(self):
        if False:
            return 10
        self.touch(b'.cover.jpg', dir=self.album.path, content='IMAGE')
        if sys.platform == 'win32':
            self.hide_file_windows()
        self.config['ignore'] = []
        self.config['ignore_hidden'] = False
        self.run_command('fetchart')
        self.album.load()
        self.check_cover_is_stored()

def suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')