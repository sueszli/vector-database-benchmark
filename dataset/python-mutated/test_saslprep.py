from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import unittest
from pymongo.saslprep import saslprep

class TestSASLprep(unittest.TestCase):

    def test_saslprep(self):
        if False:
            print('Hello World!')
        try:
            import stringprep
        except ImportError:
            self.assertRaises(TypeError, saslprep, 'anything...')
            self.assertEqual(saslprep(b'user'), b'user')
        else:
            self.assertEqual(saslprep('I\xadX'), 'IX')
            self.assertEqual(saslprep('user'), 'user')
            self.assertEqual(saslprep('USER'), 'USER')
            self.assertEqual(saslprep('ª'), 'a')
            self.assertEqual(saslprep('Ⅸ'), 'IX')
            self.assertRaises(ValueError, saslprep, '\x07')
            self.assertRaises(ValueError, saslprep, 'ا1')
            self.assertEqual(saslprep(b'user'), b'user')