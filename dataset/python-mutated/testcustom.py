"""
Custom database tests
"""
import unittest
from txtai.database import DatabaseFactory

class TestCustom(unittest.TestCase):
    """
    Custom database backend tests.
    """

    def testCustomBackend(self):
        if False:
            i = 10
            return i + 15
        '\n        Test resolving a custom backend\n        '
        database = DatabaseFactory.create({'content': 'txtai.database.SQLite'})
        self.assertIsNotNone(database)

    def testCustomBackendNotFound(self):
        if False:
            while True:
                i = 10
        '\n        Test resolving an unresolvable backend\n        '
        with self.assertRaises(ImportError):
            DatabaseFactory.create({'content': 'notfound.database'})