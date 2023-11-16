"""Run the unit tests for WriteConcern."""
from __future__ import annotations
import collections
import unittest
from pymongo.errors import ConfigurationError
from pymongo.write_concern import WriteConcern

class TestWriteConcern(unittest.TestCase):

    def test_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ConfigurationError, WriteConcern, j=True, fsync=True)
        self.assertRaises(ConfigurationError, WriteConcern, w=0, j=True)

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        concern = WriteConcern(j=True, wtimeout=3000)
        self.assertEqual(concern, WriteConcern(j=True, wtimeout=3000))
        self.assertNotEqual(concern, WriteConcern())

    def test_equality_to_none(self):
        if False:
            for i in range(10):
                print('nop')
        concern = WriteConcern()
        self.assertNotEqual(concern, None)
        self.assertTrue(concern != None)

    def test_equality_compatible_type(self):
        if False:
            print('Hello World!')

        class _FakeWriteConcern:

            def __init__(self, **document):
                if False:
                    return 10
                self.document = document

            def __eq__(self, other):
                if False:
                    return 10
                try:
                    return self.document == other.document
                except AttributeError:
                    return NotImplemented

            def __ne__(self, other):
                if False:
                    return 10
                try:
                    return self.document != other.document
                except AttributeError:
                    return NotImplemented
        self.assertEqual(WriteConcern(j=True), _FakeWriteConcern(j=True))
        self.assertEqual(_FakeWriteConcern(j=True), WriteConcern(j=True))
        self.assertEqual(WriteConcern(j=True), _FakeWriteConcern(j=True))
        self.assertEqual(WriteConcern(wtimeout=42), _FakeWriteConcern(wtimeout=42))
        self.assertNotEqual(WriteConcern(wtimeout=42), _FakeWriteConcern(wtimeout=2000))

    def test_equality_incompatible_type(self):
        if False:
            while True:
                i = 10
        _fake_type = collections.namedtuple('NotAWriteConcern', ['document'])
        self.assertNotEqual(WriteConcern(j=True), _fake_type({'j': True}))
if __name__ == '__main__':
    unittest.main()