"""Tests for the 'albumtypes' plugin."""
import unittest
from test.helper import TestHelper
from beets.autotag.mb import VARIOUS_ARTISTS_ID
from beetsplug.albumtypes import AlbumTypesPlugin

class AlbumTypesPluginTest(unittest.TestCase, TestHelper):
    """Tests for albumtypes plugin."""

    def setUp(self):
        if False:
            print('Hello World!')
        'Set up tests.'
        self.setup_beets()
        self.load_plugins('albumtypes')

    def tearDown(self):
        if False:
            print('Hello World!')
        'Tear down tests.'
        self.unload_plugins()
        self.teardown_beets()

    def test_renames_types(self):
        if False:
            while True:
                i = 10
        'Tests if the plugin correctly renames the specified types.'
        self._set_config(types=[('ep', 'EP'), ('remix', 'Remix')], ignore_va=[], bracket='()')
        album = self._create_album(album_types=['ep', 'remix'])
        subject = AlbumTypesPlugin()
        result = subject._atypes(album)
        self.assertEqual('(EP)(Remix)', result)
        return

    def test_returns_only_specified_types(self):
        if False:
            print('Hello World!')
        'Tests if the plugin returns only non-blank types given in config.'
        self._set_config(types=[('ep', 'EP'), ('soundtrack', '')], ignore_va=[], bracket='()')
        album = self._create_album(album_types=['ep', 'remix', 'soundtrack'])
        subject = AlbumTypesPlugin()
        result = subject._atypes(album)
        self.assertEqual('(EP)', result)

    def test_respects_type_order(self):
        if False:
            i = 10
            return i + 15
        'Tests if the types are returned in the same order as config.'
        self._set_config(types=[('remix', 'Remix'), ('ep', 'EP')], ignore_va=[], bracket='()')
        album = self._create_album(album_types=['ep', 'remix'])
        subject = AlbumTypesPlugin()
        result = subject._atypes(album)
        self.assertEqual('(Remix)(EP)', result)
        return

    def test_ignores_va(self):
        if False:
            while True:
                i = 10
        'Tests if the specified type is ignored for VA albums.'
        self._set_config(types=[('ep', 'EP'), ('soundtrack', 'OST')], ignore_va=['ep'], bracket='()')
        album = self._create_album(album_types=['ep', 'soundtrack'], artist_id=VARIOUS_ARTISTS_ID)
        subject = AlbumTypesPlugin()
        result = subject._atypes(album)
        self.assertEqual('(OST)', result)

    def test_respects_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests if the plugin uses the default values if config not given.'
        album = self._create_album(album_types=['ep', 'single', 'soundtrack', 'live', 'compilation', 'remix'], artist_id=VARIOUS_ARTISTS_ID)
        subject = AlbumTypesPlugin()
        result = subject._atypes(album)
        self.assertEqual('[EP][Single][OST][Live][Remix]', result)

    def _set_config(self, types: [(str, str)], ignore_va: [str], bracket: str):
        if False:
            i = 10
            return i + 15
        self.config['albumtypes']['types'] = types
        self.config['albumtypes']['ignore_va'] = ignore_va
        self.config['albumtypes']['bracket'] = bracket

    def _create_album(self, album_types: [str], artist_id: str=0):
        if False:
            return 10
        return self.add_album(albumtypes=album_types, mb_albumartistid=artist_id)