from __future__ import print_function
from datetime import datetime as dt
from bson.binary import Binary
import pickle
from arctic import Arctic, register_library_type
from arctic.decorators import mongo_retry

class Stuff(object):
    """
    Some custom class persisted by our CustomArcticLibType Library Type
    """

    def __init__(self, field1, date_field, stuff):
        if False:
            for i in range(10):
                print('nop')
        self.field1 = field1
        self.date_field = date_field
        self.stuff = stuff

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.field1) + ' ' + str(self.date_field) + ' ' + str(self.stuff)

class CustomArcticLibType(object):
    """
    Custom Arctic Library for storing 'Stuff' items
    """
    _LIBRARY_TYPE = 'test.CustomArcticLibType'

    def __init__(self, arctic_lib):
        if False:
            print('Hello World!')
        self._arctic_lib = arctic_lib
        self._collection = arctic_lib.get_top_level_collection()
        self._sub_collection = self._collection.sub_collection
        print('My name is %s' % arctic_lib.get_name())
        self.some_metadata = arctic_lib.get_library_metadata('some_metadata')

    @classmethod
    def initialize_library(cls, arctic_lib, **kwargs):
        if False:
            print('Hello World!')
        arctic_lib.set_library_metadata('some_metadata', 'some_value')
        CustomArcticLibType(arctic_lib)._ensure_index()

    def _ensure_index(self):
        if False:
            return 10
        '\n        Index any fields used by your queries.\n        '
        collection = self._collection
        collection.create_index('field1')

    @mongo_retry
    def query(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Generic query method.\n\n        In reality, your storage class would have its own query methods,\n\n        Performs a Mongo find on the Marketdata index metadata collection.\n        See:\n        http://api.mongodb.org/python/current/api/pymongo/collection.html\n        '
        for x in self._collection.find(*args, **kwargs):
            x['stuff'] = pickle.loads(x['stuff'])
            del x['_id']
            yield Stuff(**x)

    @mongo_retry
    def stats(self):
        if False:
            print('Hello World!')
        '\n        Database usage statistics. Used by quota.\n        '
        res = {}
        db = self._collection.database
        res['dbstats'] = db.command('dbstats')
        res['data'] = db.command('collstats', self._collection.name)
        res['totals'] = {'count': res['data']['count'], 'size': res['data']['size']}
        return res

    @mongo_retry
    def store(self, thing):
        if False:
            i = 10
            return i + 15
        '\n        Simple persistence method\n        '
        to_store = {'field1': thing.field1, 'date_field': thing.date_field}
        to_store['stuff'] = Binary(pickle.dumps(thing.stuff))
        self._arctic_lib.check_quota()
        self._collection.insert_one(to_store)

    @mongo_retry
    def delete(self, query):
        if False:
            i = 10
            return i + 15
        '\n        Simple delete method\n        '
        self._collection.delete_one(query)
register_library_type(CustomArcticLibType._LIBRARY_TYPE, CustomArcticLibType)
if 'mongo_host' not in globals():
    mongo_host = 'localhost'
store = Arctic(mongo_host)
store.initialize_library('username.custom_lib', CustomArcticLibType._LIBRARY_TYPE)
lib = store['username.custom_lib']
lib.store(Stuff('thing', dt(2012, 1, 1), object()))
lib.store(Stuff('thing2', dt(2013, 1, 1), object()))
lib.store(Stuff('thing3', dt(2014, 1, 1), object()))
lib.store(Stuff(['a', 'b', 'c'], dt(2014, 1, 1), object()))
for e in list(lib.query()):
    print(e)
list(lib.query({'field1': 'thing'}))
list(lib.query({'field1': 'a'}))
list(lib.query({'field1': 'b'}))
list(lib.query({'date_field': {'$lt': dt(2013, 2, 2)}}))
list(lib.query({'field1': 'thing', 'date_field': {'$lt': dt(2013, 2, 2)}}))
lib.delete({})