"""
DuckDB module tests
"""
import os
import unittest
from txtai.embeddings import Embeddings
from .testrdbms import Common

class TestDuckDB(Common.TestRDBMS):
    """
    Embeddings with content stored in DuckDB.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        '\n        Initialize test data.\n        '
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.backend = 'duckdb'
        cls.embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': cls.backend})

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        '\n        Cleanup data.\n        '
        if cls.embeddings:
            cls.embeddings.close()

    @unittest.skipIf(os.name == 'nt', 'testArchive skipped on Windows')
    def testArchive(self):
        if False:
            return 10
        '\n        Test embeddings index archiving\n        '
        super().testArchive()