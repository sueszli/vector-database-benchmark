"""
Helper functions, constants, and types to aid with PyMongo support.
"""
import pymongo
from bson import binary, json_util
from pymongo.errors import OperationFailure
PYMONGO_VERSION = tuple(pymongo.version_tuple[:2])
if PYMONGO_VERSION >= (4,):
    LEGACY_JSON_OPTIONS = json_util.LEGACY_JSON_OPTIONS.with_options(uuid_representation=binary.UuidRepresentation.PYTHON_LEGACY)
else:
    LEGACY_JSON_OPTIONS = json_util.DEFAULT_JSON_OPTIONS

def count_documents(collection, filter, skip=None, limit=None, hint=None, collation=None):
    if False:
        return 10
    'Pymongo>3.7 deprecates count in favour of count_documents'
    if limit == 0:
        return 0
    kwargs = {}
    if skip is not None:
        kwargs['skip'] = skip
    if limit is not None:
        kwargs['limit'] = limit
    if hint not in (-1, None):
        kwargs['hint'] = hint
    if collation is not None:
        kwargs['collation'] = collation
    if PYMONGO_VERSION >= (3, 7):
        try:
            return collection.count_documents(filter=filter, **kwargs)
        except OperationFailure as err:
            if PYMONGO_VERSION >= (4,):
                raise
            message = str(err)
            if not ('not allowed in this context' in message and ('$where' in message or '$geoNear' in message or '$near' in message or ('$nearSphere' in message))):
                raise
    cursor = collection.find(filter)
    for (option, option_value) in kwargs.items():
        cursor_method = getattr(cursor, option)
        cursor = cursor_method(option_value)
    with_limit_and_skip = 'skip' in kwargs or 'limit' in kwargs
    return cursor.count(with_limit_and_skip=with_limit_and_skip)

def list_collection_names(db, include_system_collections=False):
    if False:
        print('Hello World!')
    'Pymongo>3.7 deprecates collection_names in favour of list_collection_names'
    if PYMONGO_VERSION >= (3, 7):
        collections = db.list_collection_names()
    else:
        collections = db.collection_names()
    if not include_system_collections:
        collections = [c for c in collections if not c.startswith('system.')]
    return collections