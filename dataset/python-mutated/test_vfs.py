"""Tests for the virtual filesystem builder.."""
import unittest
from test import _common
from beets import library, vfs

class VFSTest(_common.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.lib = library.Library(':memory:', path_formats=[('default', 'albums/$album/$title'), ('singleton:true', 'tracks/$artist/$title')])
        self.lib.add(_common.item())
        self.lib.add_album([_common.item()])
        self.tree = vfs.libtree(self.lib)

    def test_singleton_item(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.tree.dirs['tracks'].dirs['the artist'].files['the title'], 1)

    def test_album_item(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.tree.dirs['albums'].dirs['the album'].files['the title'], 2)

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')