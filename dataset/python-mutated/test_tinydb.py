import re
from collections.abc import Mapping
import pytest
from tinydb import TinyDB, where, Query
from tinydb.middlewares import Middleware, CachingMiddleware
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document

def test_drop_tables(db: TinyDB):
    if False:
        while True:
            i = 10
    db.drop_tables()
    db.insert({})
    db.drop_tables()
    assert len(db) == 0

def test_all(db: TinyDB):
    if False:
        return 10
    db.drop_tables()
    for i in range(10):
        db.insert({})
    assert len(db.all()) == 10

def test_insert(db: TinyDB):
    if False:
        for i in range(10):
            print('nop')
    db.drop_tables()
    db.insert({'int': 1, 'char': 'a'})
    assert db.count(where('int') == 1) == 1
    db.drop_tables()
    db.insert({'int': 1, 'char': 'a'})
    db.insert({'int': 1, 'char': 'b'})
    db.insert({'int': 1, 'char': 'c'})
    assert db.count(where('int') == 1) == 3
    assert db.count(where('char') == 'a') == 1

def test_insert_ids(db: TinyDB):
    if False:
        while True:
            i = 10
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    assert db.insert({'int': 1, 'char': 'a'}) == 2

def test_insert_with_doc_id(db: TinyDB):
    if False:
        for i in range(10):
            print('nop')
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    assert db.insert(Document({'int': 1, 'char': 'a'}, 12)) == 12
    assert db.insert(Document({'int': 1, 'char': 'a'}, 77)) == 77
    assert db.insert({'int': 1, 'char': 'a'}) == 78

def test_insert_with_duplicate_doc_id(db: TinyDB):
    if False:
        while True:
            i = 10
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    with pytest.raises(ValueError):
        db.insert(Document({'int': 1, 'char': 'a'}, 1))

def test_insert_multiple(db: TinyDB):
    if False:
        while True:
            i = 10
    db.drop_tables()
    assert not db.contains(where('int') == 1)
    db.insert_multiple([{'int': 1, 'char': 'a'}, {'int': 1, 'char': 'b'}, {'int': 1, 'char': 'c'}])
    assert db.count(where('int') == 1) == 3
    assert db.count(where('char') == 'a') == 1

    def generator():
        if False:
            for i in range(10):
                print('nop')
        for j in range(10):
            yield {'int': j}
    db.drop_tables()
    db.insert_multiple(generator())
    for i in range(10):
        assert db.count(where('int') == i) == 1
    assert db.count(where('int').exists()) == 10
    db.drop_tables()
    db.insert_multiple(({'int': i} for i in range(10)))
    for i in range(10):
        assert db.count(where('int') == i) == 1

def test_insert_multiple_with_ids(db: TinyDB):
    if False:
        print('Hello World!')
    db.drop_tables()
    assert db.insert_multiple([{'int': 1, 'char': 'a'}, {'int': 1, 'char': 'b'}, {'int': 1, 'char': 'c'}]) == [1, 2, 3]

def test_insert_multiple_with_doc_ids(db: TinyDB):
    if False:
        while True:
            i = 10
    db.drop_tables()
    assert db.insert_multiple([Document({'int': 1, 'char': 'a'}, 12), Document({'int': 1, 'char': 'b'}, 77)]) == [12, 77]
    assert db.get(doc_id=12) == {'int': 1, 'char': 'a'}
    assert db.get(doc_id=77) == {'int': 1, 'char': 'b'}
    with pytest.raises(ValueError):
        db.insert_multiple([Document({'int': 1, 'char': 'a'}, 12)])

def test_insert_invalid_type_raises_error(db: TinyDB):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='Document is not a Mapping'):
        db.insert(object())

def test_insert_valid_mapping_type(db: TinyDB):
    if False:
        print('Hello World!')

    class CustomDocument(Mapping):

        def __init__(self, data):
            if False:
                i = 10
                return i + 15
            self.data = data

        def __getitem__(self, key):
            if False:
                print('Hello World!')
            return self.data[key]

        def __iter__(self):
            if False:
                print('Hello World!')
            return iter(self.data)

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self.data)
    db.drop_tables()
    db.insert(CustomDocument({'int': 1, 'char': 'a'}))
    assert db.count(where('int') == 1) == 1

def test_custom_mapping_type_with_json(tmpdir):
    if False:
        return 10

    class CustomDocument(Mapping):

        def __init__(self, data):
            if False:
                while True:
                    i = 10
            self.data = data

        def __getitem__(self, key):
            if False:
                for i in range(10):
                    print('nop')
            return self.data[key]

        def __iter__(self):
            if False:
                print('Hello World!')
            return iter(self.data)

        def __len__(self):
            if False:
                print('Hello World!')
            return len(self.data)
    db = TinyDB(str(tmpdir.join('test.db')))
    db.drop_tables()
    db.insert(CustomDocument({'int': 1, 'char': 'a'}))
    assert db.count(where('int') == 1) == 1
    db.insert_multiple([CustomDocument({'int': 2, 'char': 'a'}), CustomDocument({'int': 3, 'char': 'a'})])
    assert db.count(where('int') == 1) == 1
    assert db.count(where('int') == 2) == 1
    assert db.count(where('int') == 3) == 1
    doc_id = db.get(where('int') == 3).doc_id
    db.update(CustomDocument({'int': 4, 'char': 'a'}), doc_ids=[doc_id])
    assert db.count(where('int') == 3) == 0
    assert db.count(where('int') == 4) == 1

def test_remove(db: TinyDB):
    if False:
        print('Hello World!')
    db.remove(where('char') == 'b')
    assert len(db) == 2
    assert db.count(where('int') == 1) == 2

def test_remove_all_fails(db: TinyDB):
    if False:
        i = 10
        return i + 15
    with pytest.raises(RuntimeError):
        db.remove()

def test_remove_multiple(db: TinyDB):
    if False:
        i = 10
        return i + 15
    db.remove(where('int') == 1)
    assert len(db) == 0

def test_remove_ids(db: TinyDB):
    if False:
        i = 10
        return i + 15
    db.remove(doc_ids=[1, 2])
    assert len(db) == 1

def test_remove_returns_ids(db: TinyDB):
    if False:
        return 10
    assert db.remove(where('char') == 'b') == [2]

def test_update(db: TinyDB):
    if False:
        return 10
    assert len(db) == 3
    db.update({'int': 2}, where('char') == 'a')
    assert db.count(where('int') == 2) == 1
    assert db.count(where('int') == 1) == 2

def test_update_all(db: TinyDB):
    if False:
        i = 10
        return i + 15
    assert db.count(where('int') == 1) == 3
    db.update({'newField': True})
    assert db.count(where('newField') == True) == 3

def test_update_returns_ids(db: TinyDB):
    if False:
        print('Hello World!')
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    assert db.insert({'int': 1, 'char': 'a'}) == 2
    assert db.update({'char': 'b'}, where('int') == 1) == [1, 2]

def test_update_transform(db: TinyDB):
    if False:
        return 10

    def increment(field):
        if False:
            while True:
                i = 10

        def transform(el):
            if False:
                while True:
                    i = 10
            el[field] += 1
        return transform

    def delete(field):
        if False:
            i = 10
            return i + 15

        def transform(el):
            if False:
                print('Hello World!')
            del el[field]
        return transform
    assert db.count(where('int') == 1) == 3
    db.update(increment('int'), where('char') == 'a')
    db.update(delete('char'), where('char') == 'a')
    assert db.count(where('int') == 2) == 1
    assert db.count(where('char') == 'a') == 0
    assert db.count(where('int') == 1) == 2

def test_update_ids(db: TinyDB):
    if False:
        print('Hello World!')
    db.update({'int': 2}, doc_ids=[1, 2])
    assert db.count(where('int') == 2) == 2

def test_update_multiple(db: TinyDB):
    if False:
        for i in range(10):
            print('nop')
    assert len(db) == 3
    db.update_multiple([({'int': 2}, where('char') == 'a'), ({'int': 4}, where('char') == 'b')])
    assert db.count(where('int') == 1) == 1
    assert db.count(where('int') == 2) == 1
    assert db.count(where('int') == 4) == 1

def test_update_multiple_operation(db: TinyDB):
    if False:
        while True:
            i = 10

    def increment(field):
        if False:
            for i in range(10):
                print('nop')

        def transform(el):
            if False:
                for i in range(10):
                    print('nop')
            el[field] += 1
        return transform
    assert db.count(where('int') == 1) == 3
    db.update_multiple([(increment('int'), where('char') == 'a'), (increment('int'), where('char') == 'b')])
    assert db.count(where('int') == 2) == 2

def test_upsert(db: TinyDB):
    if False:
        for i in range(10):
            print('nop')
    assert len(db) == 3
    db.upsert({'int': 5}, where('char') == 'a')
    assert db.count(where('int') == 5) == 1
    assert db.upsert({'int': 9, 'char': 'x'}, where('char') == 'x') == [4]
    assert db.count(where('int') == 9) == 1

def test_upsert_by_id(db: TinyDB):
    if False:
        i = 10
        return i + 15
    assert len(db) == 3
    extant_doc = Document({'char': 'v'}, doc_id=1)
    assert db.upsert(extant_doc) == [1]
    doc = db.get(where('char') == 'v')
    assert isinstance(doc, Document)
    assert doc is not None
    assert doc.doc_id == 1
    assert len(db) == 3
    missing_doc = Document({'int': 5, 'char': 'w'}, doc_id=5)
    assert db.upsert(missing_doc) == [5]
    doc = db.get(where('char') == 'w')
    assert isinstance(doc, Document)
    assert doc is not None
    assert doc.doc_id == 5
    assert len(db) == 4
    with pytest.raises(ValueError, match='(?=.*\\bdoc_id\\b)(?=.*\\bquery\\b)'):
        db.upsert({'no_Document': 'no_query'})
    assert db.insert({'check': '_next_id'}) == 6

def test_search(db: TinyDB):
    if False:
        return 10
    assert not db._query_cache
    assert len(db.search(where('int') == 1)) == 3
    assert len(db._query_cache) == 1
    assert len(db.search(where('int') == 1)) == 3

def test_search_path(db: TinyDB):
    if False:
        while True:
            i = 10
    assert not db._query_cache
    assert len(db.search(where('int').exists())) == 3
    assert len(db._query_cache) == 1
    assert len(db.search(where('asd').exists())) == 0
    assert len(db.search(where('int').exists())) == 3

def test_search_no_results_cache(db: TinyDB):
    if False:
        print('Hello World!')
    assert len(db.search(where('missing').exists())) == 0
    assert len(db.search(where('missing').exists())) == 0

def test_get(db: TinyDB):
    if False:
        print('Hello World!')
    item = db.get(where('char') == 'b')
    assert isinstance(item, Document)
    assert item is not None
    assert item['char'] == 'b'

def test_get_ids(db: TinyDB):
    if False:
        i = 10
        return i + 15
    el = db.all()[0]
    assert db.get(doc_id=el.doc_id) == el
    assert db.get(doc_id=float('NaN')) is None

def test_get_multiple_ids(db: TinyDB):
    if False:
        return 10
    el = db.all()
    assert db.get(doc_ids=[x.doc_id for x in el]) == el

def test_get_invalid(db: TinyDB):
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError):
        db.get()

def test_count(db: TinyDB):
    if False:
        i = 10
        return i + 15
    assert db.count(where('int') == 1) == 3
    assert db.count(where('char') == 'd') == 0

def test_contains(db: TinyDB):
    if False:
        while True:
            i = 10
    assert db.contains(where('int') == 1)
    assert not db.contains(where('int') == 0)

def test_contains_ids(db: TinyDB):
    if False:
        print('Hello World!')
    assert db.contains(doc_id=1)
    assert db.contains(doc_id=2)
    assert not db.contains(doc_id=88)

def test_contains_invalid(db: TinyDB):
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError):
        db.contains()

def test_get_idempotent(db: TinyDB):
    if False:
        i = 10
        return i + 15
    u = db.get(where('int') == 1)
    z = db.get(where('int') == 1)
    assert u == z

def test_multiple_dbs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Regression test for issue #3\n    '
    db1 = TinyDB(storage=MemoryStorage)
    db2 = TinyDB(storage=MemoryStorage)
    db1.insert({'int': 1, 'char': 'a'})
    db1.insert({'int': 1, 'char': 'b'})
    db1.insert({'int': 1, 'value': 5.0})
    db2.insert({'color': 'blue', 'animal': 'turtle'})
    assert len(db1) == 3
    assert len(db2) == 1

def test_storage_closed_once():
    if False:
        print('Hello World!')

    class Storage:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.closed = False

        def read(self):
            if False:
                while True:
                    i = 10
            return {}

        def write(self, data):
            if False:
                while True:
                    i = 10
            pass

        def close(self):
            if False:
                i = 10
                return i + 15
            assert not self.closed
            self.closed = True
    with TinyDB(storage=Storage) as db:
        db.close()
    del db

def test_unique_ids(tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    :type tmpdir: py._path.local.LocalPath\n    '
    path = str(tmpdir.join('db.json'))
    with TinyDB(path) as _db:
        _db.insert({'x': 1})
    with TinyDB(path) as _db:
        _db.insert({'x': 1})
    with TinyDB(path) as _db:
        data = _db.all()
        assert data[0].doc_id != data[1].doc_id
    with TinyDB(path) as _db:
        _db.drop_tables()
        _db.insert_multiple(({'x': i} for i in range(5)))
        _db.remove(where('x') == 2)
        assert len(_db) == 4
        ids = [e.doc_id for e in _db.all()]
        assert len(ids) == len(set(ids))

def test_lastid_after_open(tmpdir):
    if False:
        return 10
    '\n    Regression test for issue #34\n\n    :type tmpdir: py._path.local.LocalPath\n    '
    NUM = 100
    path = str(tmpdir.join('db.json'))
    with TinyDB(path) as _db:
        _db.insert_multiple(({'i': i} for i in range(NUM)))
    with TinyDB(path) as _db:
        assert _db._get_next_id() - 1 == NUM

def test_doc_ids_json(tmpdir):
    if False:
        return 10
    '\n    Regression test for issue #45\n    '
    path = str(tmpdir.join('db.json'))
    with TinyDB(path) as _db:
        _db.drop_tables()
        assert _db.insert({'int': 1, 'char': 'a'}) == 1
        assert _db.insert({'int': 1, 'char': 'a'}) == 2
        _db.drop_tables()
        assert _db.insert_multiple([{'int': 1, 'char': 'a'}, {'int': 1, 'char': 'b'}, {'int': 1, 'char': 'c'}]) == [1, 2, 3]
        assert _db.contains(doc_id=1)
        assert _db.contains(doc_id=2)
        assert not _db.contains(doc_id=88)
        _db.update({'int': 2}, doc_ids=[1, 2])
        assert _db.count(where('int') == 2) == 2
        el = _db.all()[0]
        assert _db.get(doc_id=el.doc_id) == el
        assert _db.get(doc_id=float('NaN')) is None
        _db.remove(doc_ids=[1, 2])
        assert len(_db) == 1

def test_insert_string(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    path = str(tmpdir.join('db.json'))
    with TinyDB(path) as _db:
        data = [{'int': 1}, {'int': 2}]
        _db.insert_multiple(data)
        with pytest.raises(ValueError):
            _db.insert([1, 2, 3])
        with pytest.raises(ValueError):
            _db.insert({'bark'})
        assert data == _db.all()
        _db.insert({'int': 3})

def test_insert_invalid_dict(tmpdir):
    if False:
        return 10
    path = str(tmpdir.join('db.json'))
    with TinyDB(path) as _db:
        data = [{'int': 1}, {'int': 2}]
        _db.insert_multiple(data)
        with pytest.raises(TypeError):
            _db.insert({'int': _db})
        assert data == _db.all()
        _db.insert({'int': 3})

def test_gc(tmpdir):
    if False:
        return 10
    path = str(tmpdir.join('db.json'))
    db = TinyDB(path)
    table = db.table('foo')
    table.insert({'something': 'else'})
    table.insert({'int': 13})
    assert len(table.search(where('int') == 13)) == 1
    assert table.all() == [{'something': 'else'}, {'int': 13}]
    db.close()

def test_drop_table():
    if False:
        return 10
    db = TinyDB(storage=MemoryStorage)
    default_table_name = db.table(db.default_table_name).name
    assert [] == list(db.tables())
    db.drop_table(default_table_name)
    db.insert({'a': 1})
    assert [default_table_name] == list(db.tables())
    db.drop_table(default_table_name)
    assert [] == list(db.tables())
    table_name = 'some-other-table'
    db = TinyDB(storage=MemoryStorage)
    db.table(table_name).insert({'a': 1})
    assert {table_name} == db.tables()
    db.drop_table(table_name)
    assert set() == db.tables()
    assert table_name not in db._tables
    db.drop_table('non-existent-table-name')
    assert set() == db.tables()

def test_empty_write(tmpdir):
    if False:
        i = 10
        return i + 15
    path = str(tmpdir.join('db.json'))

    class ReadOnlyMiddleware(Middleware):

        def write(self, data):
            if False:
                return 10
            raise AssertionError('No write for unchanged db')
    TinyDB(path).close()
    TinyDB(path, storage=ReadOnlyMiddleware(JSONStorage)).close()

def test_query_cache():
    if False:
        while True:
            i = 10
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple([{'name': 'foo', 'value': 42}, {'name': 'bar', 'value': -1337}])
    query = where('value') > 0
    results = db.search(query)
    assert len(results) == 1
    db._tables[db.table(db.default_table_name).name]._read_table = lambda : {}
    results.extend([1])
    assert db.search(query) == [{'name': 'foo', 'value': 42}]

def test_tinydb_is_iterable(db: TinyDB):
    if False:
        while True:
            i = 10
    assert [r for r in db] == db.all()

def test_repr(tmpdir):
    if False:
        print('Hello World!')
    path = str(tmpdir.join('db.json'))
    db = TinyDB(path)
    db.insert({'a': 1})
    assert re.match("<TinyDB tables=\\[u?\\'_default\\'\\], tables_count=1, default_table_documents_count=1, all_tables_documents_count=\\[\\'_default=1\\'\\]>", repr(db))

def test_delete(tmpdir):
    if False:
        while True:
            i = 10
    path = str(tmpdir.join('db.json'))
    db = TinyDB(path, ensure_ascii=False)
    q = Query()
    db.insert({'network': {'id': '114', 'name': 'ok', 'rpc': 'dac', 'ticker': 'mkay'}})
    assert db.search(q.network.id == '114') == [{'network': {'id': '114', 'name': 'ok', 'rpc': 'dac', 'ticker': 'mkay'}}]
    db.remove(q.network.id == '114')
    assert db.search(q.network.id == '114') == []

def test_insert_multiple_with_single_dict(db: TinyDB):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        d = {'first': 'John', 'last': 'smith'}
        db.insert_multiple(d)
        db.close()

def test_access_storage():
    if False:
        i = 10
        return i + 15
    assert isinstance(TinyDB(storage=MemoryStorage).storage, MemoryStorage)
    assert isinstance(TinyDB(storage=CachingMiddleware(MemoryStorage)).storage, CachingMiddleware)

def test_empty_db_len():
    if False:
        print('Hello World!')
    db = TinyDB(storage=MemoryStorage)
    assert len(db) == 0

def test_insert_on_existing_db(tmpdir):
    if False:
        return 10
    path = str(tmpdir.join('db.json'))
    db = TinyDB(path, ensure_ascii=False)
    db.insert({'foo': 'bar'})
    assert len(db) == 1
    db.close()
    db = TinyDB(path, ensure_ascii=False)
    db.insert({'foo': 'bar'})
    db.insert({'foo': 'bar'})
    assert len(db) == 3

def test_storage_access():
    if False:
        print('Hello World!')
    db = TinyDB(storage=MemoryStorage)
    assert isinstance(db.storage, MemoryStorage)

def test_lambda_query():
    if False:
        print('Hello World!')
    db = TinyDB(storage=MemoryStorage)
    db.insert({'foo': 'bar'})
    query = lambda doc: doc.get('foo') == 'bar'
    query.is_cacheable = lambda : False
    assert db.search(query) == [{'foo': 'bar'}]
    assert not db._query_cache