import unittest
import pytest
from nltk import ConditionalFreqDist, tokenize

class TestEmptyCondFreq(unittest.TestCase):

    def test_tabulate(self):
        if False:
            while True:
                i = 10
        empty = ConditionalFreqDist()
        self.assertEqual(empty.conditions(), [])
        with pytest.raises(ValueError):
            empty.tabulate(conditions='BUG')
        self.assertEqual(empty.conditions(), [])

    def test_plot(self):
        if False:
            i = 10
            return i + 15
        empty = ConditionalFreqDist()
        self.assertEqual(empty.conditions(), [])
        empty.plot(conditions=['BUG'])
        self.assertEqual(empty.conditions(), [])

    def test_increment(self):
        if False:
            i = 10
            return i + 15
        text = 'cow cat mouse cat tiger'
        cfd = ConditionalFreqDist()
        for word in tokenize.word_tokenize(text):
            condition = len(word)
            cfd[condition][word] += 1
        self.assertEqual(cfd.conditions(), [3, 5])
        cfd[2]['hi'] += 1
        self.assertCountEqual(cfd.conditions(), [3, 5, 2])
        self.assertEqual(cfd[2]['hi'], 1)