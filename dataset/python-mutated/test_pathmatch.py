from twisted.trial import unittest
from buildbot.util import pathmatch

class Matcher(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.m = pathmatch.Matcher()

    def test_dupe_path(self):
        if False:
            i = 10
            return i + 15

        def set():
            if False:
                for i in range(10):
                    print('nop')
            self.m['abc,'] = 1
        set()
        with self.assertRaises(AssertionError):
            set()

    def test_empty(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(KeyError):
            self.m['abc',]

    def test_diff_length(self):
        if False:
            for i in range(10):
                print('nop')
        self.m['abc', 'def'] = 2
        self.m['ab', 'cd', 'ef'] = 3
        self.assertEqual(self.m['abc', 'def'], (2, {}))

    def test_same_length(self):
        if False:
            i = 10
            return i + 15
        self.m['abc', 'def'] = 2
        self.m['abc', 'efg'] = 3
        self.assertEqual(self.m['abc', 'efg'], (3, {}))

    def test_pattern_variables(self):
        if False:
            i = 10
            return i + 15
        self.m['A', ':a', 'B', ':b'] = 'AB'
        self.assertEqual(self.m['A', 'a', 'B', 'b'], ('AB', {'a': 'a', 'b': 'b'}))

    def test_pattern_variables_underscore(self):
        if False:
            while True:
                i = 10
        self.m['A', ':a_a_a'] = 'AB'
        self.assertEqual(self.m['A', 'a'], ('AB', {'a_a_a': 'a'}))

    def test_pattern_variables_num(self):
        if False:
            print('Hello World!')
        self.m['A', 'n:a', 'B', 'n:b'] = 'AB'
        self.assertEqual(self.m['A', '10', 'B', '-20'], ('AB', {'a': 10, 'b': -20}))

    def test_pattern_variables_ident(self):
        if False:
            return 10
        self.m['A', 'i:a', 'B', 'i:b'] = 'AB'
        self.assertEqual(self.m['A', 'abc', 'B', 'x-z-B'], ('AB', {'a': 'abc', 'b': 'x-z-B'}))

    def test_pattern_variables_num_invalid(self):
        if False:
            i = 10
            return i + 15
        self.m['A', 'n:a'] = 'AB'
        with self.assertRaises(KeyError):
            self.m['A', '1x0']

    def test_pattern_variables_ident_invalid(self):
        if False:
            i = 10
            return i + 15
        self.m['A', 'i:a'] = 'AB'
        with self.assertRaises(KeyError):
            self.m['A', '10']

    def test_pattern_variables_ident_num_distinguised(self):
        if False:
            for i in range(10):
                print('nop')
        self.m['A', 'n:a'] = 'num'
        self.m['A', 'i:a'] = 'ident'
        self.assertEqual(self.m['A', '123'], ('num', {'a': 123}))
        self.assertEqual(self.m['A', 'abc'], ('ident', {'a': 'abc'}))

    def test_prefix_matching(self):
        if False:
            return 10
        self.m['A', ':a'] = 'A'
        self.m['A', ':a', 'B', ':b'] = 'AB'
        self.assertEqual((self.m['A', 'a1', 'B', 'b'], self.m['A', 'a2']), (('AB', {'a': 'a1', 'b': 'b'}), ('A', {'a': 'a2'})))

    def test_dirty_again(self):
        if False:
            for i in range(10):
                print('nop')
        self.m['abc', 'def'] = 2
        self.assertEqual(self.m['abc', 'def'], (2, {}))
        self.m['abc', 'efg'] = 3
        self.assertEqual(self.m['abc', 'def'], (2, {}))
        self.assertEqual(self.m['abc', 'efg'], (3, {}))