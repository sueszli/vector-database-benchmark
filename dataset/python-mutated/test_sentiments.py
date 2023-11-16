from __future__ import unicode_literals
import unittest
from nose.tools import *
from nose.plugins.attrib import attr
from textblob.sentiments import PatternAnalyzer, NaiveBayesAnalyzer, DISCRETE, CONTINUOUS

class TestPatternSentiment(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.analyzer = PatternAnalyzer()

    def test_kind(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(self.analyzer.kind, CONTINUOUS)

    def test_analyze(self):
        if False:
            return 10
        p1 = 'I feel great this morning.'
        n1 = 'This is a terrible car.'
        p1_result = self.analyzer.analyze(p1)
        n1_result = self.analyzer.analyze(n1)
        assert_true(p1_result[0] > 0)
        assert_true(n1_result[0] < 0)
        assert_equal(p1_result.polarity, p1_result[0])
        assert_equal(p1_result.subjectivity, p1_result[1])

    def test_analyze_assessments(self):
        if False:
            i = 10
            return i + 15
        p1 = 'I feel great this morning.'
        n1 = 'This is a terrible car.'
        p1_result = self.analyzer.analyze(p1, keep_assessments=True)
        n1_result = self.analyzer.analyze(n1, keep_assessments=True)
        p1_assessment = p1_result.assessments[0]
        n1_assessment = n1_result.assessments[0]
        assert_true(p1_assessment[1] > 0)
        assert_true(n1_assessment[1] < 0)
        assert_equal(p1_result.polarity, p1_assessment[1])
        assert_equal(p1_result.subjectivity, p1_assessment[2])

class TestNaiveBayesAnalyzer(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.analyzer = NaiveBayesAnalyzer()

    def test_kind(self):
        if False:
            while True:
                i = 10
        assert_equal(self.analyzer.kind, DISCRETE)

    @attr('slow')
    def test_analyze(self):
        if False:
            print('Hello World!')
        p1 = 'I feel great this morning.'
        n1 = 'This is a terrible car.'
        p1_result = self.analyzer.analyze(p1)
        assert_equal(p1_result[0], 'pos')
        assert_equal(self.analyzer.analyze(n1)[0], 'neg')
        assert_true(isinstance(p1_result[1], float))
        assert_true(isinstance(p1_result[2], float))
        assert_about_equal(p1_result[1] + p1_result[2], 1)
        assert_equal(p1_result.classification, p1_result[0])
        assert_equal(p1_result.p_pos, p1_result[1])
        assert_equal(p1_result.p_neg, p1_result[2])

def assert_about_equal(first, second, places=4):
    if False:
        return 10
    return assert_equal(round(first, places), second)
if __name__ == '__main__':
    unittest.main()