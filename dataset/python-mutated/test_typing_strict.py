"""Test typings in strict mode."""
from __future__ import annotations
import unittest
from typing import TYPE_CHECKING, Any, Dict
import pymongo
from pymongo.collection import Collection
from pymongo.database import Database

def test_generic_arguments() -> None:
    if False:
        print('Hello World!')
    'Ensure known usages of generic arguments pass strict typing'
    if not TYPE_CHECKING:
        raise unittest.SkipTest('Used for Type Checking Only')
    mongo_client: pymongo.MongoClient[Dict[str, Any]] = pymongo.MongoClient()
    mongo_client.drop_database('foo')
    mongo_client.get_default_database()
    db = mongo_client.get_database('test_db')
    db = Database(mongo_client, 'test_db')
    db.with_options()
    db.validate_collection('py_test')
    col = db.get_collection('py_test')
    col.insert_one({'abc': 123})
    col = Collection(db, 'py_test')
    col.with_options()