"""Tests for the 'ftintitle' plugin."""
import unittest
from test.helper import TestHelper
from beetsplug import ftintitle

class FtInTitlePluginFunctional(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set up configuration'
        self.setup_beets()
        self.load_plugins('ftintitle')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.unload_plugins()
        self.teardown_beets()

    def _ft_add_item(self, path, artist, title, aartist):
        if False:
            for i in range(10):
                print('nop')
        return self.add_item(path=path, artist=artist, artist_sort=artist, title=title, albumartist=aartist)

    def _ft_set_config(self, ftformat, drop=False, auto=True):
        if False:
            print('Hello World!')
        self.config['ftintitle']['format'] = ftformat
        self.config['ftintitle']['drop'] = drop
        self.config['ftintitle']['auto'] = auto

    def test_functional_drop(self):
        if False:
            return 10
        item = self._ft_add_item('/', 'Alice ft Bob', 'Song 1', 'Alice')
        self.run_command('ftintitle', '-d')
        item.load()
        self.assertEqual(item['artist'], 'Alice')
        self.assertEqual(item['title'], 'Song 1')

    def test_functional_not_found(self):
        if False:
            print('Hello World!')
        item = self._ft_add_item('/', 'Alice ft Bob', 'Song 1', 'George')
        self.run_command('ftintitle', '-d')
        item.load()
        self.assertEqual(item['artist'], 'Alice ft Bob')
        self.assertEqual(item['title'], 'Song 1')

    def test_functional_custom_format(self):
        if False:
            for i in range(10):
                print('nop')
        self._ft_set_config('feat. {0}')
        item = self._ft_add_item('/', 'Alice ft Bob', 'Song 1', 'Alice')
        self.run_command('ftintitle')
        item.load()
        self.assertEqual(item['artist'], 'Alice')
        self.assertEqual(item['title'], 'Song 1 feat. Bob')
        self._ft_set_config('featuring {0}')
        item = self._ft_add_item('/', 'Alice feat. Bob', 'Song 1', 'Alice')
        self.run_command('ftintitle')
        item.load()
        self.assertEqual(item['artist'], 'Alice')
        self.assertEqual(item['title'], 'Song 1 featuring Bob')
        self._ft_set_config('with {0}')
        item = self._ft_add_item('/', 'Alice feat Bob', 'Song 1', 'Alice')
        self.run_command('ftintitle')
        item.load()
        self.assertEqual(item['artist'], 'Alice')
        self.assertEqual(item['title'], 'Song 1 with Bob')

class FtInTitlePluginTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up configuration'
        ftintitle.FtInTitlePlugin()

    def test_find_feat_part(self):
        if False:
            i = 10
            return i + 15
        test_cases = [{'artist': 'Alice ft. Bob', 'album_artist': 'Alice', 'feat_part': 'Bob'}, {'artist': 'Alice feat Bob', 'album_artist': 'Alice', 'feat_part': 'Bob'}, {'artist': 'Alice featuring Bob', 'album_artist': 'Alice', 'feat_part': 'Bob'}, {'artist': 'Alice & Bob', 'album_artist': 'Alice', 'feat_part': 'Bob'}, {'artist': 'Alice and Bob', 'album_artist': 'Alice', 'feat_part': 'Bob'}, {'artist': 'Alice With Bob', 'album_artist': 'Alice', 'feat_part': 'Bob'}, {'artist': 'Alice defeat Bob', 'album_artist': 'Alice', 'feat_part': None}, {'artist': 'Alice & Bob', 'album_artist': 'Bob', 'feat_part': 'Alice'}, {'artist': 'Alice ft. Bob', 'album_artist': 'Bob', 'feat_part': 'Alice'}, {'artist': 'Alice ft. Carol', 'album_artist': 'Bob', 'feat_part': None}]
        for test_case in test_cases:
            feat_part = ftintitle.find_feat_part(test_case['artist'], test_case['album_artist'])
            self.assertEqual(feat_part, test_case['feat_part'])

    def test_split_on_feat(self):
        if False:
            return 10
        parts = ftintitle.split_on_feat('Alice ft. Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice feat Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice feat. Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice featuring Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice & Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice and Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice With Bob')
        self.assertEqual(parts, ('Alice', 'Bob'))
        parts = ftintitle.split_on_feat('Alice defeat Bob')
        self.assertEqual(parts, ('Alice defeat Bob', None))

    def test_contains_feat(self):
        if False:
            return 10
        self.assertTrue(ftintitle.contains_feat('Alice ft. Bob'))
        self.assertTrue(ftintitle.contains_feat('Alice feat. Bob'))
        self.assertTrue(ftintitle.contains_feat('Alice feat Bob'))
        self.assertTrue(ftintitle.contains_feat('Alice featuring Bob'))
        self.assertTrue(ftintitle.contains_feat('Alice & Bob'))
        self.assertTrue(ftintitle.contains_feat('Alice and Bob'))
        self.assertTrue(ftintitle.contains_feat('Alice With Bob'))
        self.assertFalse(ftintitle.contains_feat('Alice defeat Bob'))
        self.assertFalse(ftintitle.contains_feat('Aliceft.Bob'))

def suite():
    if False:
        i = 10
        return i + 15
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')