import logging
from pymongo.errors import OperationFailure
from .._util import enable_sharding, mongo_count
from ..decorators import mongo_retry
logger = logging.getLogger(__name__)
BSON_STORE_TYPE = 'BSONStore'

class BSONStore(object):
    """
    BSON Data Store. This stores any Python object that encodes to BSON correctly,
    and offers a vanilla pymongo interface. Note that strings myst be valid UTF-8.

    See: https://api.mongodb.com/python/3.4.0/api/bson/index.html

    Note that this neither defines nor ensures any indices, they are left to the user
    to create and manage according to the effective business schema applicable to their data.

    Likewise, _id is left to the user to populate if they wish, and is exposed in documents. As
    is normally the case with pymongo, _id is set to unique ObjectId if left unspecified at
    document insert time.
    """

    def __init__(self, arctic_lib):
        if False:
            return 10
        self._arctic_lib = arctic_lib
        self._reset()

    def enable_sharding(self):
        if False:
            for i in range(10):
                print('nop')
        logger.info('Trying to enable sharding...')
        arctic_lib = self._arctic_lib
        try:
            enable_sharding(arctic_lib.arctic, arctic_lib.get_name(), hashed=True, key='_id')
        except OperationFailure as exception:
            logger.warning('Could not enable sharding: %s, you probably need admin permissions.', exception)

    @classmethod
    def initialize_library(cls, arctic_lib, hashed=True, **kwargs):
        if False:
            while True:
                i = 10
        logger.info('Creating BSONStore without sharding. Use BSONStore.enable_sharding to enable sharding for large amounts of data.')
        c = arctic_lib.get_top_level_collection()
        if c.name not in mongo_retry(c.database.list_collection_names)():
            mongo_retry(c.database.create_collection)(c.name)
        else:
            logger.warning('Collection %s already exists', c.name)

    @mongo_retry
    def _reset(self):
        if False:
            return 10
        self._collection = self._arctic_lib.get_top_level_collection()

    @mongo_retry
    def stats(self):
        if False:
            return 10
        '\n        Store stats, necessary for quota to work.\n        '
        res = {}
        db = self._collection.database
        res['dbstats'] = db.command('dbstats')
        res['data'] = db.command('collstats', self._collection.name)
        res['totals'] = {'count': res['data']['count'], 'size': res['data']['size']}
        return res

    @mongo_retry
    def find(self, *args, **kwargs):
        if False:
            return 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find\n        '
        return self._collection.find(*args, **kwargs)

    @mongo_retry
    def find_one(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find_one\n        '
        return self._collection.find_one(*args, **kwargs)

    @mongo_retry
    def insert_one(self, document, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.insert_one\n        '
        self._arctic_lib.check_quota()
        return self._collection.insert_one(document, **kwargs)

    @mongo_retry
    def insert_many(self, documents, **kwargs):
        if False:
            while True:
                i = 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.insert_many\n        '
        self._arctic_lib.check_quota()
        return self._collection.insert_many(documents, **kwargs)

    def delete_one(self, filter, **kwargs):
        if False:
            return 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.delete_one\n        '
        return self._collection.delete_one(filter, **kwargs)

    @mongo_retry
    def delete_many(self, filter, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.delete_many\n        '
        return self._collection.delete_many(filter, **kwargs)

    @mongo_retry
    def update_one(self, filter, update, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.update_one\n        '
        self._arctic_lib.check_quota()
        return self._collection.update_one(filter, update, **kwargs)

    @mongo_retry
    def update_many(self, filter, update, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.update_many\n        '
        self._arctic_lib.check_quota()
        return self._collection.update_many(filter, update, **kwargs)

    @mongo_retry
    def replace_one(self, filter, replacement, **kwargs):
        if False:
            return 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.replace_one\n        '
        self._arctic_lib.check_quota()
        return self._collection.replace_one(filter, replacement, **kwargs)

    @mongo_retry
    def find_one_and_replace(self, filter, replacement, **kwargs):
        if False:
            while True:
                i = 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find_one_and_replace\n        '
        self._arctic_lib.check_quota()
        return self._collection.find_one_and_replace(filter, replacement, **kwargs)

    @mongo_retry
    def find_one_and_update(self, filter, update, **kwargs):
        if False:
            print('Hello World!')
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find_one_and_update\n        '
        self._arctic_lib.check_quota()
        return self._collection.find_one_and_update(filter, update, **kwargs)

    def find_one_and_delete(self, filter, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.find_one_and_delete\n        '
        return self._collection.find_one_and_delete(filter, **kwargs)

    @mongo_retry
    def bulk_write(self, requests, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.bulk_write\n\n        Warning: this is wrapped in mongo_retry, and is therefore potentially unsafe if the write you want to execute\n        isn't idempotent.\n        "
        self._arctic_lib.check_quota()
        return self._collection.bulk_write(requests, **kwargs)

    @mongo_retry
    def count(self, filter, **kwargs):
        if False:
            print('Hello World!')
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.count\n        '
        return mongo_count(self._collection, filter=filter, **kwargs)

    @mongo_retry
    def aggregate(self, pipeline, **kwargs):
        if False:
            while True:
                i = 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.aggregate\n        '
        return self._collection.aggregate(pipeline, **kwargs)

    @mongo_retry
    def distinct(self, key, **kwargs):
        if False:
            while True:
                i = 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.distinct\n        '
        return self._collection.distinct(key, **kwargs)

    @mongo_retry
    def create_index(self, keys, **kwargs):
        if False:
            while True:
                i = 10
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.create_index\n        '
        return self._collection.create_index(keys, **kwargs)

    @mongo_retry
    def drop_index(self, index_or_name):
        if False:
            print('Hello World!')
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.drop_index\n        '
        return self._collection.drop_index(index_or_name)

    @mongo_retry
    def index_information(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        See http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.index_information\n        '
        return self._collection.index_information()