"""
Tabular module tests
"""
import unittest
from txtai.pipeline import Tabular
from utils import Utils

class TestTabular(unittest.TestCase):
    """
    Tabular tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        '\n        Create single tabular instance\n        '
        cls.tabular = Tabular('id', ['text'])

    def testContent(self):
        if False:
            while True:
                i = 10
        '\n        Test parsing additional content\n        '
        tabular = Tabular('id', ['text'], True)
        row = {'id': 0, 'text': 'This is a test', 'flag': 1}
        rows = tabular([row])
        (uid, data, _) = rows[1]
        self.assertEqual(uid, 0)
        self.assertEqual(data, row)
        tabular.content = ['flag']
        rows = tabular([row])
        (uid, data, _) = rows[1]
        self.assertEqual(uid, 0)
        self.assertTrue(list(data.keys()) == ['flag'])
        self.assertEqual(data['flag'], 1)

    def testCSV(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test parsing a CSV file\n        '
        rows = self.tabular([Utils.PATH + '/tabular.csv'])
        (uid, text, _) = rows[0][0]
        self.assertEqual(uid, 0)
        self.assertEqual(text, 'The first sentence')

    def testDict(self):
        if False:
            i = 10
            return i + 15
        '\n        Test parsing a dict\n        '
        rows = self.tabular([{'id': 0, 'text': 'This is a test'}])
        (uid, text, _) = rows[0]
        self.assertEqual(uid, 0)
        self.assertEqual(text, 'This is a test')

    def testList(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test parsing a list\n        '
        rows = self.tabular([[{'id': 0, 'text': 'This is a test'}]])
        (uid, text, _) = rows[0][0]
        self.assertEqual(uid, 0)
        self.assertEqual(text, 'This is a test')

    def testMissingColumns(self):
        if False:
            print('Hello World!')
        '\n        Test rows with uneven or missing columns\n        '
        tabular = Tabular('id', ['text'], True)
        rows = tabular([{'id': 0, 'text': 'This is a test', 'metadata': 'meta'}, {'id': 1, 'text': 'This is a test'}])
        (_, data, _) = rows[3]
        self.assertIsNone(data['metadata'])

    def testNoColumns(self):
        if False:
            while True:
                i = 10
        '\n        Test creating text without specifying columns\n        '
        tabular = Tabular('id')
        rows = tabular([{'id': 0, 'text': 'This is a test', 'summary': 'Describes text in more detail'}])
        (uid, text, _) = rows[0]
        self.assertEqual(uid, 0)
        self.assertEqual(text, 'This is a test. Describes text in more detail')