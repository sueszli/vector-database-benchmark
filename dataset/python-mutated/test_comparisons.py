"""
Test ChatterBot's statement comparison algorithms.
"""
from unittest import TestCase
from chatterbot.conversation import Statement
from chatterbot import comparisons
from chatterbot import languages

class LevenshteinDistanceTestCase(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.compare = comparisons.LevenshteinDistance(language=languages.ENG)

    def test_levenshtein_distance_statement_false(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Falsy values should match by zero.\n        '
        statement = Statement(text='')
        other_statement = Statement(text='Hello')
        value = self.compare(statement, other_statement)
        self.assertEqual(value, 0)

    def test_levenshtein_distance_other_statement_false(self):
        if False:
            while True:
                i = 10
        '\n        Falsy values should match by zero.\n        '
        statement = Statement(text='Hello')
        other_statement = Statement(text='')
        value = self.compare(statement, other_statement)
        self.assertEqual(value, 0)

    def test_levenshtein_distance_statement_integer(self):
        if False:
            while True:
                i = 10
        '\n        Test that an exception is not raised if a statement is initialized\n        with an integer value as its text attribute.\n        '
        statement = Statement(text=2)
        other_statement = Statement(text='Hello')
        value = self.compare(statement, other_statement)
        self.assertEqual(value, 0)

    def test_exact_match_different_capitalization(self):
        if False:
            while True:
                i = 10
        '\n        Test that text capitalization is ignored.\n        '
        statement = Statement(text='Hi HoW ArE yOu?')
        other_statement = Statement(text='hI hOw are YoU?')
        value = self.compare(statement, other_statement)
        self.assertEqual(value, 1)

class SpacySimilarityTests(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.compare = comparisons.SpacySimilarity(language=languages.ENG)

    def test_exact_match_different_stopwords(self):
        if False:
            return 10
        '\n        Test sentences with different stopwords.\n        '
        statement = Statement(text='What is matter?')
        other_statement = Statement(text='What is the matter?')
        value = self.compare(statement, other_statement)
        self.assertAlmostEqual(value, 0.9, places=1)

    def test_exact_match_different_capitalization(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that text capitalization is ignored.\n        '
        statement = Statement(text='Hi HoW ArE yOu?')
        other_statement = Statement(text='hI hOw are YoU?')
        value = self.compare(statement, other_statement)
        self.assertAlmostEqual(value, 0.8, places=1)

class JaccardSimilarityTestCase(TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.compare = comparisons.JaccardSimilarity(language=languages.ENG)

    def test_exact_match_different_capitalization(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that text capitalization is ignored.\n        '
        statement = Statement(text='Hi HoW ArE yOu?')
        other_statement = Statement(text='hI hOw are YoU?')
        value = self.compare(statement, other_statement)
        self.assertEqual(value, 1)