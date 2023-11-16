from functools import partial
import os
import sqlite3
import sqlalchemy as sa
from six.moves import range
from .input_validation import coerce_string
SQLITE_MAX_VARIABLE_NUMBER = 998

def group_into_chunks(items, chunk_size=SQLITE_MAX_VARIABLE_NUMBER):
    if False:
        for i in range(10):
            print('nop')
    items = list(items)
    return [items[x:x + chunk_size] for x in range(0, len(items), chunk_size)]

def verify_sqlite_path_exists(path):
    if False:
        return 10
    if path != ':memory:' and (not os.path.exists(path)):
        raise ValueError("SQLite file {!r} doesn't exist.".format(path))

def check_and_create_connection(path, require_exists):
    if False:
        i = 10
        return i + 15
    if require_exists:
        verify_sqlite_path_exists(path)
    return sqlite3.connect(path)

def check_and_create_engine(path, require_exists):
    if False:
        while True:
            i = 10
    if require_exists:
        verify_sqlite_path_exists(path)
    return sa.create_engine('sqlite:///' + path)

def coerce_string_to_conn(require_exists):
    if False:
        for i in range(10):
            print('nop')
    return coerce_string(partial(check_and_create_connection, require_exists=require_exists))

def coerce_string_to_eng(require_exists):
    if False:
        while True:
            i = 10
    return coerce_string(partial(check_and_create_engine, require_exists=require_exists))