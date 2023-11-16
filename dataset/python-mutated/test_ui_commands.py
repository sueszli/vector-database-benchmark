"""Test module for file ui/commands.py
"""
import os
import shutil
import unittest
from test import _common
from beets import library, ui
from beets.ui import commands
from beets.util import syspath

class QueryTest(_common.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.libdir = os.path.join(self.temp_dir, b'testlibdir')
        os.mkdir(syspath(self.libdir))
        self.lib = library.Library(':memory:', self.libdir)
        self.otherdir = os.path.join(self.temp_dir, b'testotherdir')

    def add_item(self, filename=b'srcfile', templatefile=b'full.mp3'):
        if False:
            for i in range(10):
                print('nop')
        itempath = os.path.join(self.libdir, filename)
        shutil.copy(syspath(os.path.join(_common.RSRC, templatefile)), syspath(itempath))
        item = library.Item.from_path(itempath)
        self.lib.add(item)
        return (item, itempath)

    def add_album(self, items):
        if False:
            return 10
        album = self.lib.add_album(items)
        return album

    def check_do_query(self, num_items, num_albums, q=(), album=False, also_items=True):
        if False:
            while True:
                i = 10
        (items, albums) = commands._do_query(self.lib, q, album, also_items)
        self.assertEqual(len(items), num_items)
        self.assertEqual(len(albums), num_albums)

    def test_query_empty(self):
        if False:
            return 10
        with self.assertRaises(ui.UserError):
            commands._do_query(self.lib, (), False)

    def test_query_empty_album(self):
        if False:
            return 10
        with self.assertRaises(ui.UserError):
            commands._do_query(self.lib, (), True)

    def test_query_item(self):
        if False:
            i = 10
            return i + 15
        self.add_item()
        self.check_do_query(1, 0, album=False)
        self.add_item()
        self.check_do_query(2, 0, album=False)

    def test_query_album(self):
        if False:
            while True:
                i = 10
        (item, itempath) = self.add_item()
        self.add_album([item])
        self.check_do_query(1, 1, album=True)
        self.check_do_query(0, 1, album=True, also_items=False)
        (item, itempath) = self.add_item()
        (item2, itempath) = self.add_item()
        self.add_album([item, item2])
        self.check_do_query(3, 2, album=True)
        self.check_do_query(0, 2, album=True, also_items=False)

class FieldsTest(_common.LibTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.io.install()

    def tearDown(self):
        if False:
            return 10
        self.io.restore()

    def remove_keys(self, l, text):
        if False:
            for i in range(10):
                print('nop')
        for i in text:
            try:
                l.remove(i)
            except ValueError:
                pass

    def test_fields_func(self):
        if False:
            for i in range(10):
                print('nop')
        commands.fields_func(self.lib, [], [])
        items = library.Item.all_keys()
        albums = library.Album.all_keys()
        output = self.io.stdout.get().split()
        self.remove_keys(items, output)
        self.remove_keys(albums, output)
        self.assertEqual(len(items), 0)
        self.assertEqual(len(albums), 0)

def suite():
    if False:
        return 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')