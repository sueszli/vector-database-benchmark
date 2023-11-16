"""Test the pymongo module itself."""
from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import unittest
import pymongo

class TestPyMongo(unittest.TestCase):

    def test_mongo_client_alias(self):
        if False:
            print('Hello World!')
        self.assertEqual(pymongo.MongoClient, pymongo.mongo_client.MongoClient)
if __name__ == '__main__':
    unittest.main()