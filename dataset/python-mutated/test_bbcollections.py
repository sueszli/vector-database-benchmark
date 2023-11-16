from twisted.trial import unittest
from buildbot.util import bbcollections

class KeyedSets(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.ks = bbcollections.KeyedSets()

    def test_getitem_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.ks['x'], set())

    def test_add(self):
        if False:
            return 10
        self.ks.add('y', 2)
        self.assertEqual(self.ks['y'], set([2]))

    def test_add_twice(self):
        if False:
            while True:
                i = 10
        self.ks.add('z', 2)
        self.ks.add('z', 4)
        self.assertEqual(self.ks['z'], set([2, 4]))

    def test_discard_noError(self):
        if False:
            return 10
        self.ks.add('full', 12)
        self.ks.discard('empty', 13)
        self.ks.discard('full', 13)
        self.assertEqual(self.ks['full'], set([12]))

    def test_discard_existing(self):
        if False:
            i = 10
            return i + 15
        self.ks.add('yarn', 'red')
        self.ks.discard('yarn', 'red')
        self.assertEqual(self.ks['yarn'], set([]))

    def test_contains_true(self):
        if False:
            i = 10
            return i + 15
        self.ks.add('yarn', 'red')
        self.assertTrue('yarn' in self.ks)

    def test_contains_false(self):
        if False:
            return 10
        self.assertFalse('yarn' in self.ks)

    def test_contains_setNamesNotContents(self):
        if False:
            print('Hello World!')
        self.ks.add('yarn', 'red')
        self.assertFalse('red' in self.ks)

    def test_pop_exists(self):
        if False:
            while True:
                i = 10
        self.ks.add('names', 'pop')
        self.ks.add('names', 'coke')
        self.ks.add('names', 'soda')
        popped = self.ks.pop('names')
        remaining = self.ks['names']
        self.assertEqual((popped, remaining), (set(['pop', 'coke', 'soda']), set()))

    def test_pop_missing(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.ks.pop('flavors'), set())