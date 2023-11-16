import unittest
import math

class WordPermutation:
    """'
    Based on a given word generate all possible permutations 
    of the characters that composes that word.
    Pre-condition: The word is intended to be a single word, if it's a phrase it will be treated as a single word.
    """

    def __init__(self, word) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.word = word
        self.permutations = set()

    def generate_permutations(self):
        if False:
            i = 10
            return i + 15
        "'\n        Generates all possible permutations for the given word\n        "
        self._permutations_recursive('', self.word)
        return self.permutations

    def _permutations_recursive(self, current, remaining):
        if False:
            print('Hello World!')
        if len(remaining) == 0:
            self.permutations.add(current)
            return
        for i in range(len(remaining)):
            next_char = remaining[i]
            new_current = current + next_char
            new_remaining = remaining[:i] + remaining[i + 1:]
            self._permutations_recursive(new_current, new_remaining)

class WordPermutationTestSuite(unittest.TestCase):
    """'
    Test Case for word permutation class
    """

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.word = 'abcd'
        self.word_permutation = WordPermutation(word=self.word)
        return super().setUp()

    def test_initialization(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.word_permutation.word, self.word)

    def test_permutations_contains_word(self):
        if False:
            i = 10
            return i + 15
        permutations = self.word_permutation.generate_permutations()
        self.assertIn(self.word, permutations)

    def test_permutations_simple_aa(self):
        if False:
            print('Hello World!')
        word = 'aa'
        self.word_permutation = WordPermutation(word)
        permutations = self.word_permutation.generate_permutations()
        self.assertIn(word, permutations)

    def test_permutations_simple_aa_count(self):
        if False:
            print('Hello World!')
        word = 'aa'
        self.word_permutation = WordPermutation(word)
        permutations = self.word_permutation.generate_permutations()
        self.assertEqual(len(permutations), 1)

    def test_permutations_count_unique_chars(self):
        if False:
            for i in range(10):
                print('nop')
        permutations = self.word_permutation.generate_permutations()
        self.assertEqual(len(permutations), math.factorial(len(self.word)))

    def test_permutations_count_repeated_chars(self):
        if False:
            for i in range(10):
                print('nop')
        word = 'abcdeee'
        self.word_permutation = WordPermutation(word)
        self.word_permutation.generate_permutations()
        self.assertEqual(len(self.word_permutation.permutations), 840)

    def test_random_permutation_in_permutations(self):
        if False:
            while True:
                i = 10
        word = 'asdfghi'
        self.word_permutation = WordPermutation(word)
        self.word_permutation.generate_permutations()
        self.assertIn('ihgfdsa', self.word_permutation.permutations)
if __name__ == '__main__':
    unittest.main()