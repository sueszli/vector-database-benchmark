import os
import shutil
import tempfile
import unittest
from shlex import quote
from test import _common, helper
import beets

class PlaylistTestHelper(helper.TestHelper):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_beets()
        self.lib = beets.library.Library(':memory:')
        self.music_dir = os.path.expanduser(os.path.join('~', 'Music'))
        i1 = _common.item()
        i1.path = beets.util.normpath(os.path.join(self.music_dir, 'a', 'b', 'c.mp3'))
        i1.title = 'some item'
        i1.album = 'some album'
        self.lib.add(i1)
        self.lib.add_album([i1])
        i2 = _common.item()
        i2.path = beets.util.normpath(os.path.join(self.music_dir, 'd', 'e', 'f.mp3'))
        i2.title = 'another item'
        i2.album = 'another album'
        self.lib.add(i2)
        self.lib.add_album([i2])
        i3 = _common.item()
        i3.path = beets.util.normpath(os.path.join(self.music_dir, 'x', 'y', 'z.mp3'))
        i3.title = 'yet another item'
        i3.album = 'yet another album'
        self.lib.add(i3)
        self.lib.add_album([i3])
        self.playlist_dir = tempfile.mkdtemp()
        self.config['directory'] = self.music_dir
        self.config['playlist']['playlist_dir'] = self.playlist_dir
        self.setup_test()
        self.load_plugins('playlist')

    def setup_test(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.unload_plugins()
        shutil.rmtree(self.playlist_dir)
        self.teardown_beets()

class PlaylistQueryTestHelper(PlaylistTestHelper):

    def test_name_query_with_absolute_paths_in_playlist(self):
        if False:
            return 10
        q = 'playlist:absolute'
        results = self.lib.items(q)
        self.assertEqual({i.title for i in results}, {'some item', 'another item'})

    def test_path_query_with_absolute_paths_in_playlist(self):
        if False:
            return 10
        q = 'playlist:{}'.format(quote(os.path.join(self.playlist_dir, 'absolute.m3u')))
        results = self.lib.items(q)
        self.assertEqual({i.title for i in results}, {'some item', 'another item'})

    def test_name_query_with_relative_paths_in_playlist(self):
        if False:
            for i in range(10):
                print('nop')
        q = 'playlist:relative'
        results = self.lib.items(q)
        self.assertEqual({i.title for i in results}, {'some item', 'another item'})

    def test_path_query_with_relative_paths_in_playlist(self):
        if False:
            print('Hello World!')
        q = 'playlist:{}'.format(quote(os.path.join(self.playlist_dir, 'relative.m3u')))
        results = self.lib.items(q)
        self.assertEqual({i.title for i in results}, {'some item', 'another item'})

    def test_name_query_with_nonexisting_playlist(self):
        if False:
            return 10
        q = 'playlist:nonexisting'
        results = self.lib.items(q)
        self.assertEqual(set(results), set())

    def test_path_query_with_nonexisting_playlist(self):
        if False:
            return 10
        q = 'playlist:{}'.format(quote(os.path.join(self.playlist_dir, self.playlist_dir, 'nonexisting.m3u')))
        results = self.lib.items(q)
        self.assertEqual(set(results), set())

class PlaylistTestRelativeToLib(PlaylistQueryTestHelper, unittest.TestCase):

    def setup_test(self):
        if False:
            while True:
                i = 10
        with open(os.path.join(self.playlist_dir, 'absolute.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join(self.music_dir, 'a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'd', 'e', 'f.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'nonexisting.mp3')))
        with open(os.path.join(self.playlist_dir, 'relative.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join('a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join('d', 'e', 'f.mp3')))
            f.write('{}\n'.format('nonexisting.mp3'))
        self.config['playlist']['relative_to'] = 'library'

class PlaylistTestRelativeToDir(PlaylistQueryTestHelper, unittest.TestCase):

    def setup_test(self):
        if False:
            while True:
                i = 10
        with open(os.path.join(self.playlist_dir, 'absolute.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join(self.music_dir, 'a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'd', 'e', 'f.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'nonexisting.mp3')))
        with open(os.path.join(self.playlist_dir, 'relative.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join('a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join('d', 'e', 'f.mp3')))
            f.write('{}\n'.format('nonexisting.mp3'))
        self.config['playlist']['relative_to'] = self.music_dir

class PlaylistTestRelativeToPls(PlaylistQueryTestHelper, unittest.TestCase):

    def setup_test(self):
        if False:
            print('Hello World!')
        with open(os.path.join(self.playlist_dir, 'absolute.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join(self.music_dir, 'a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'd', 'e', 'f.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'nonexisting.mp3')))
        with open(os.path.join(self.playlist_dir, 'relative.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.relpath(os.path.join(self.music_dir, 'a', 'b', 'c.mp3'), start=self.playlist_dir)))
            f.write('{}\n'.format(os.path.relpath(os.path.join(self.music_dir, 'd', 'e', 'f.mp3'), start=self.playlist_dir)))
            f.write('{}\n'.format(os.path.relpath(os.path.join(self.music_dir, 'nonexisting.mp3'), start=self.playlist_dir)))
        self.config['playlist']['relative_to'] = 'playlist'
        self.config['playlist']['playlist_dir'] = self.playlist_dir

class PlaylistUpdateTestHelper(PlaylistTestHelper):

    def setup_test(self):
        if False:
            while True:
                i = 10
        with open(os.path.join(self.playlist_dir, 'absolute.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join(self.music_dir, 'a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'd', 'e', 'f.mp3')))
            f.write('{}\n'.format(os.path.join(self.music_dir, 'nonexisting.mp3')))
        with open(os.path.join(self.playlist_dir, 'relative.m3u'), 'w') as f:
            f.write('{}\n'.format(os.path.join('a', 'b', 'c.mp3')))
            f.write('{}\n'.format(os.path.join('d', 'e', 'f.mp3')))
            f.write('{}\n'.format('nonexisting.mp3'))
        self.config['playlist']['auto'] = True
        self.config['playlist']['relative_to'] = 'library'

class PlaylistTestItemMoved(PlaylistUpdateTestHelper, unittest.TestCase):

    def test_item_moved(self):
        if False:
            print('Hello World!')
        results = self.lib.items('path:{}'.format(quote(os.path.join(self.music_dir, 'd', 'e', 'f.mp3'))))
        item = results[0]
        beets.plugins.send('item_moved', item=item, source=item.path, destination=beets.util.bytestring_path(os.path.join(self.music_dir, 'g', 'h', 'i.mp3')))
        results = self.lib.items('path:{}'.format(quote(os.path.join(self.music_dir, 'x', 'y', 'z.mp3'))))
        item = results[0]
        beets.plugins.send('item_moved', item=item, source=item.path, destination=beets.util.bytestring_path(os.path.join(self.music_dir, 'u', 'v', 'w.mp3')))
        beets.plugins.send('cli_exit', lib=self.lib)
        playlist_path = os.path.join(self.playlist_dir, 'absolute.m3u')
        with open(playlist_path) as f:
            lines = [line.strip() for line in f.readlines()]
        self.assertEqual(lines, [os.path.join(self.music_dir, 'a', 'b', 'c.mp3'), os.path.join(self.music_dir, 'g', 'h', 'i.mp3'), os.path.join(self.music_dir, 'nonexisting.mp3')])
        playlist_path = os.path.join(self.playlist_dir, 'relative.m3u')
        with open(playlist_path) as f:
            lines = [line.strip() for line in f.readlines()]
        self.assertEqual(lines, [os.path.join('a', 'b', 'c.mp3'), os.path.join('g', 'h', 'i.mp3'), 'nonexisting.mp3'])

class PlaylistTestItemRemoved(PlaylistUpdateTestHelper, unittest.TestCase):

    def test_item_removed(self):
        if False:
            print('Hello World!')
        results = self.lib.items('path:{}'.format(quote(os.path.join(self.music_dir, 'd', 'e', 'f.mp3'))))
        item = results[0]
        beets.plugins.send('item_removed', item=item)
        results = self.lib.items('path:{}'.format(quote(os.path.join(self.music_dir, 'x', 'y', 'z.mp3'))))
        item = results[0]
        beets.plugins.send('item_removed', item=item)
        beets.plugins.send('cli_exit', lib=self.lib)
        playlist_path = os.path.join(self.playlist_dir, 'absolute.m3u')
        with open(playlist_path) as f:
            lines = [line.strip() for line in f.readlines()]
        self.assertEqual(lines, [os.path.join(self.music_dir, 'a', 'b', 'c.mp3'), os.path.join(self.music_dir, 'nonexisting.mp3')])
        playlist_path = os.path.join(self.playlist_dir, 'relative.m3u')
        with open(playlist_path) as f:
            lines = [line.strip() for line in f.readlines()]
        self.assertEqual(lines, [os.path.join('a', 'b', 'c.mp3'), 'nonexisting.mp3'])

def suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')