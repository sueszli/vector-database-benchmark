"""Tests for the 'ihate' plugin"""
import unittest
from beets import importer
from beets.library import Item
from beetsplug.ihate import IHatePlugin

class IHatePluginTest(unittest.TestCase):

    def test_hate(self):
        if False:
            print('Hello World!')
        match_pattern = {}
        test_item = Item(genre='TestGenre', album='TestAlbum', artist='TestArtist')
        task = importer.SingletonImportTask(None, test_item)
        self.assertFalse(IHatePlugin.do_i_hate_this(task, match_pattern))
        match_pattern = ['artist:bad_artist', 'artist:TestArtist']
        self.assertTrue(IHatePlugin.do_i_hate_this(task, match_pattern))
        match_pattern = ['album:test', 'artist:testartist']
        self.assertTrue(IHatePlugin.do_i_hate_this(task, match_pattern))
        match_pattern = ['album:notthis genre:testgenre']
        self.assertFalse(IHatePlugin.do_i_hate_this(task, match_pattern))
        match_pattern = ['album:notthis genre:testgenre', 'artist:testartist album:notthis']
        self.assertFalse(IHatePlugin.do_i_hate_this(task, match_pattern))
        match_pattern = ['album:testalbum genre:testgenre', 'artist:testartist album:notthis']
        self.assertTrue(IHatePlugin.do_i_hate_this(task, match_pattern))

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')