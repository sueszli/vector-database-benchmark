"""
Summary module tests
"""
import unittest
from txtai.pipeline import Textractor
from utils import Utils

class TestTextractor(unittest.TestCase):
    """
    Textractor tests.
    """

    def testBeautifulSoup(self):
        if False:
            print('Hello World!')
        '\n        Test text extraction using Beautiful Soup\n        '
        textractor = Textractor(tika=False)
        text = textractor(Utils.PATH + '/tabular.csv')
        self.assertEqual(len(text), 125)

    def testCheckJava(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the checkjava method\n        '
        textractor = Textractor()
        self.assertFalse(textractor.checkjava('1112444abc'))

    def testLines(self):
        if False:
            print('Hello World!')
        '\n        Test extraction to lines\n        '
        textractor = Textractor(lines=True)
        lines = textractor(Utils.PATH + '/article.pdf')
        self.assertEqual(len(lines), 35)

    def testParagraphs(self):
        if False:
            i = 10
            return i + 15
        '\n        Test extraction to paragraphs\n        '
        textractor = Textractor(paragraphs=True)
        paragraphs = textractor(Utils.PATH + '/article.pdf')
        self.assertEqual(len(paragraphs), 13)

    def testSentences(self):
        if False:
            while True:
                i = 10
        '\n        Test extraction to sentences\n        '
        textractor = Textractor(sentences=True)
        sentences = textractor(Utils.PATH + '/article.pdf')
        self.assertEqual(len(sentences), 17)

    def testSingle(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a single extraction with no tokenization of the results\n        '
        textractor = Textractor()
        text = textractor(Utils.PATH + '/article.pdf')
        self.assertEqual(len(text), 2301)