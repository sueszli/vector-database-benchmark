"""
SQLite module tests
"""
from txtai.embeddings import Embeddings
from .testrdbms import Common

class TestSQLite(Common.TestRDBMS):
    """
    Embeddings with content stored in SQLite tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        '\n        Initialize test data.\n        '
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.backend = 'sqlite'
        cls.embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': cls.backend})

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Cleanup data.\n        '
        if cls.embeddings:
            cls.embeddings.close()

    def testFunction(self):
        if False:
            i = 10
            return i + 15
        '\n        Test custom functions\n        '
        embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend, 'functions': [{'name': 'length', 'function': 'testdatabase.testsqlite.length'}]})
        embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        result = embeddings.search('select length(text) length from txtai where id = 0', 1)[0]
        self.assertEqual(result['length'], 39)

def length(text):
    if False:
        return 10
    '\n    Custom SQL function.\n    '
    return len(text)