"""
Summary module tests
"""
import unittest
from txtai.pipeline import Summary

class TestSummary(unittest.TestCase):
    """
    Summary tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        '\n        Create single summary instance.\n        '
        cls.text = "Search is the base of many applications. Once data starts to pile up, users want to be able to find it. It's the foundation of the internet and an ever-growing challenge that is never solved or done. The field of Natural Language Processing (NLP) is rapidly evolving with a number of new developments. Large-scale general language models are an exciting new capability allowing us to add amazing functionality quickly with limited compute and people. Innovation continues with new models and advancements coming in at what seems a weekly basis. This article introduces txtai, an AI-powered search engine that enables Natural Language Understanding (NLU) based search in any application."
        cls.summary = Summary('t5-small')

    def testSummary(self):
        if False:
            i = 10
            return i + 15
        '\n        Test summarization of text\n        '
        self.assertEqual(self.summary(self.text, minlength=15, maxlength=15), 'the field of natural language processing (NLP) is rapidly evolving')

    def testSummaryBatch(self):
        if False:
            print('Hello World!')
        '\n        Test batch summarization of text\n        '
        summaries = self.summary([self.text, self.text], maxlength=15)
        self.assertEqual(len(summaries), 2)

    def testSummaryNoLength(self):
        if False:
            i = 10
            return i + 15
        '\n        Test summary with no max length set\n        '
        self.assertEqual(self.summary(self.text + self.text), 'search is the base of many applications. Once data starts to pile up, users want to be able to find it. ' + 'Large-scale general language models are an exciting new capability allowing us to add amazing functionality quickly ' + 'with limited compute and people.')

    def testSummaryShort(self):
        if False:
            return 10
        '\n        Test that summarization is skipped\n        '
        self.assertEqual(self.summary('Text', maxlength=15), 'Text')