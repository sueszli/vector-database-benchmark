import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary

class NgramModelVocabularyTests(unittest.TestCase):
    """tests Vocabulary Class"""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.vocab = Vocabulary(['z', 'a', 'b', 'c', 'f', 'd', 'e', 'g', 'a', 'd', 'b', 'e', 'w'], unk_cutoff=2)

    def test_truthiness(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.vocab)

    def test_cutoff_value_set_correctly(self):
        if False:
            return 10
        self.assertEqual(self.vocab.cutoff, 2)

    def test_unable_to_change_cutoff(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AttributeError):
            self.vocab.cutoff = 3

    def test_cutoff_setter_checks_value(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError) as exc_info:
            Vocabulary('abc', unk_cutoff=0)
        expected_error_msg = 'Cutoff value cannot be less than 1. Got: 0'
        self.assertEqual(expected_error_msg, str(exc_info.exception))

    def test_counts_set_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.vocab.counts['a'], 2)
        self.assertEqual(self.vocab.counts['b'], 2)
        self.assertEqual(self.vocab.counts['c'], 1)

    def test_membership_check_respects_cutoff(self):
        if False:
            while True:
                i = 10
        self.assertTrue('a' in self.vocab)
        self.assertFalse('c' in self.vocab)
        self.assertFalse('z' in self.vocab)

    def test_vocab_len_respects_cutoff(self):
        if False:
            return 10
        self.assertEqual(5, len(self.vocab))

    def test_vocab_iter_respects_cutoff(self):
        if False:
            i = 10
            return i + 15
        vocab_counts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'w', 'z']
        vocab_items = ['a', 'b', 'd', 'e', '<UNK>']
        self.assertCountEqual(vocab_counts, list(self.vocab.counts.keys()))
        self.assertCountEqual(vocab_items, list(self.vocab))

    def test_update_empty_vocab(self):
        if False:
            while True:
                i = 10
        empty = Vocabulary(unk_cutoff=2)
        self.assertEqual(len(empty), 0)
        self.assertFalse(empty)
        self.assertIn(empty.unk_label, empty)
        empty.update(list('abcde'))
        self.assertIn(empty.unk_label, empty)

    def test_lookup(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.vocab.lookup('a'), 'a')
        self.assertEqual(self.vocab.lookup('c'), '<UNK>')

    def test_lookup_iterables(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.vocab.lookup(['a', 'b']), ('a', 'b'))
        self.assertEqual(self.vocab.lookup(('a', 'b')), ('a', 'b'))
        self.assertEqual(self.vocab.lookup(('a', 'c')), ('a', '<UNK>'))
        self.assertEqual(self.vocab.lookup(map(str, range(3))), ('<UNK>', '<UNK>', '<UNK>'))

    def test_lookup_empty_iterables(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.vocab.lookup(()), ())
        self.assertEqual(self.vocab.lookup([]), ())
        self.assertEqual(self.vocab.lookup(iter([])), ())
        self.assertEqual(self.vocab.lookup((n for n in range(0, 0))), ())

    def test_lookup_recursive(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.vocab.lookup([['a', 'b'], ['a', 'c']]), (('a', 'b'), ('a', '<UNK>')))
        self.assertEqual(self.vocab.lookup([['a', 'b'], 'c']), (('a', 'b'), '<UNK>'))
        self.assertEqual(self.vocab.lookup([[[[['a', 'b']]]]]), ((((('a', 'b'),),),),))

    def test_lookup_None(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            self.vocab.lookup(None)
        with self.assertRaises(TypeError):
            list(self.vocab.lookup([None, None]))

    def test_lookup_int(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.vocab.lookup(1)
        with self.assertRaises(TypeError):
            list(self.vocab.lookup([1, 2]))

    def test_lookup_empty_str(self):
        if False:
            return 10
        self.assertEqual(self.vocab.lookup(''), '<UNK>')

    def test_eqality(self):
        if False:
            while True:
                i = 10
        v1 = Vocabulary(['a', 'b', 'c'], unk_cutoff=1)
        v2 = Vocabulary(['a', 'b', 'c'], unk_cutoff=1)
        v3 = Vocabulary(['a', 'b', 'c'], unk_cutoff=1, unk_label='blah')
        v4 = Vocabulary(['a', 'b'], unk_cutoff=1)
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertNotEqual(v1, v4)

    def test_str(self):
        if False:
            print('Hello World!')
        self.assertEqual(str(self.vocab), "<Vocabulary with cutoff=2 unk_label='<UNK>' and 5 items>")

    def test_creation_with_counter(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.vocab, Vocabulary(Counter(['z', 'a', 'b', 'c', 'f', 'd', 'e', 'g', 'a', 'd', 'b', 'e', 'w']), unk_cutoff=2))

    @unittest.skip(reason='Test is known to be flaky as it compares (runtime) performance.')
    def test_len_is_constant(self):
        if False:
            print('Hello World!')
        small_vocab = Vocabulary('abcde')
        from nltk.corpus.europarl_raw import english
        large_vocab = Vocabulary(english.words())
        small_vocab_len_time = timeit('len(small_vocab)', globals=locals())
        large_vocab_len_time = timeit('len(large_vocab)', globals=locals())
        self.assertAlmostEqual(small_vocab_len_time, large_vocab_len_time, places=1)