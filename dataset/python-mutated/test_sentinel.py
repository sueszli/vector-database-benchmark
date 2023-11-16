from copy import copy, deepcopy
from pickle import loads, dumps
import sys
from unittest import TestCase
from weakref import ref
from zipline.utils.sentinel import sentinel

class SentinelTestCase(TestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        sentinel._cache.clear()

    def test_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(sentinel('a').__name__, 'a')

    def test_doc(self):
        if False:
            while True:
                i = 10
        self.assertEqual(sentinel('a', 'b').__doc__, 'b')

    def test_doc_differentiates(self):
        if False:
            for i in range(10):
                print('nop')
        line = sys._getframe().f_lineno
        a = sentinel('sentinel-name', 'original-doc')
        with self.assertRaises(ValueError) as e:
            sentinel(a.__name__, 'new-doc')
        msg = str(e.exception)
        self.assertIn(a.__name__, msg)
        self.assertIn(a.__doc__, msg)
        self.assertIn('%s:%s' % (__file__.rstrip('c'), line + 1), msg)

    def test_memo(self):
        if False:
            print('Hello World!')
        self.assertIs(sentinel('a'), sentinel('a'))

    def test_copy(self):
        if False:
            print('Hello World!')
        a = sentinel('a')
        self.assertIs(copy(a), a)

    def test_deepcopy(self):
        if False:
            print('Hello World!')
        a = sentinel('a')
        self.assertIs(deepcopy(a), a)

    def test_repr(self):
        if False:
            while True:
                i = 10
        self.assertEqual(repr(sentinel('a')), "sentinel('a')")

    def test_new(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            type(sentinel('a'))()

    def test_pickle_roundtrip(self):
        if False:
            print('Hello World!')
        a = sentinel('a')
        self.assertIs(loads(dumps(a)), a)

    def test_weakreferencable(self):
        if False:
            while True:
                i = 10
        ref(sentinel('a'))