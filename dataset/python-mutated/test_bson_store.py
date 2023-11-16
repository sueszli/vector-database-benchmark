from mock import sentinel, create_autospec, patch, call, Mock
from pymongo.collection import Collection
from arctic import Arctic
from arctic.arctic import ArcticLibraryBinding
from arctic.store.bson_store import BSONStore

def test_enable_sharding():
    if False:
        print('Hello World!')
    arctic_lib = create_autospec(ArcticLibraryBinding)
    arctic_lib.arctic = create_autospec(Arctic)
    with patch('arctic.store.bson_store.enable_sharding', autospec=True) as enable_sharding:
        arctic_lib.get_top_level_collection.return_value.database.create_collection.__name__ = 'some_name'
        arctic_lib.get_top_level_collection.return_value.database.collection_names.__name__ = 'some_name'
        bsons = BSONStore(arctic_lib)
        bsons.enable_sharding()
        assert enable_sharding.call_args_list == [call(arctic_lib.arctic, arctic_lib.get_name(), hashed=True, key='_id')]

def test_find():
    if False:
        i = 10
        return i + 15
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    collection.find.return_value = (doc for doc in [sentinel.document])
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    assert list(bsons.find(sentinel.filter)) == [sentinel.document]
    assert collection.find.call_count == 1
    assert collection.find.call_args_list == [call(sentinel.filter)]

def test_find_one():
    if False:
        return 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    collection.find_one.return_value = sentinel.document
    arctic_lib.get_top_level_collection.return_value = collection
    ms = BSONStore(arctic_lib)
    assert ms.find_one(sentinel.filter) == sentinel.document
    assert collection.find_one.call_count == 1
    assert collection.find_one.call_args_list == [call(sentinel.filter)]

def test_insert_one():
    if False:
        print('Hello World!')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.insert_one(sentinel.document)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.insert_one.call_count == 1
    assert collection.insert_one.call_args_list == [call(sentinel.document)]

def test_insert_many():
    if False:
        for i in range(10):
            print('nop')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.insert_many(sentinel.documents)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.insert_many.call_count == 1
    assert collection.insert_many.call_args_list == [call(sentinel.documents)]

def test_replace_one():
    if False:
        while True:
            i = 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.replace_one(sentinel.filter, sentinel.replacement)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.replace_one.call_count == 1
    assert collection.replace_one.call_args_list == [call(sentinel.filter, sentinel.replacement)]

def test_update_one():
    if False:
        print('Hello World!')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.update_one(sentinel.filter, sentinel.replacement)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.update_one.call_count == 1
    assert collection.update_one.call_args_list == [call(sentinel.filter, sentinel.replacement)]

def test_update_many():
    if False:
        return 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.update_many(sentinel.filter, sentinel.replacements)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.update_many.call_count == 1
    assert collection.update_many.call_args_list == [call(sentinel.filter, sentinel.replacements)]

def test_find_one_and_replace():
    if False:
        print('Hello World!')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.find_one_and_replace(sentinel.filter, sentinel.replacement)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.find_one_and_replace.call_count == 1
    assert collection.find_one_and_replace.call_args_list == [call(sentinel.filter, sentinel.replacement)]

def test_find_one_and_update():
    if False:
        while True:
            i = 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    ms = BSONStore(arctic_lib)
    ms.find_one_and_update(sentinel.filter, sentinel.update)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.find_one_and_update.call_count == 1
    assert collection.find_one_and_update.call_args_list == [call(sentinel.filter, sentinel.update)]

def test_find_one_and_delete():
    if False:
        while True:
            i = 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    ms = BSONStore(arctic_lib)
    ms.find_one_and_delete(sentinel.filter)
    assert collection.find_one_and_delete.call_count == 1
    assert collection.find_one_and_delete.call_args_list == [call(sentinel.filter)]

def test_bulk_write():
    if False:
        while True:
            i = 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.bulk_write(sentinel.requests)
    assert arctic_lib.check_quota.call_count == 1
    assert collection.bulk_write.call_count == 1
    assert collection.bulk_write.call_args_list == [call(sentinel.requests)]

def test_delete_one():
    if False:
        print('Hello World!')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.delete_one(sentinel.filter)
    assert collection.delete_one.call_count == 1
    assert collection.delete_one.call_args_list == [call(sentinel.filter)]

def test_count():
    if False:
        for i in range(10):
            print('nop')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True, count=Mock(), count_documents=Mock())
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.count(sentinel.filter)
    assert collection.count.call_count + collection.count_documents.call_count == 1
    assert collection.count.call_args_list == [call(filter=sentinel.filter)] or collection.count_documents.call_args_list == [call(filter=sentinel.filter)]

def test_distinct():
    if False:
        while True:
            i = 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.distinct(sentinel.key)
    assert collection.distinct.call_count == 1
    assert collection.distinct.call_args_list == [call(sentinel.key)]

def test_delete_many():
    if False:
        i = 10
        return i + 15
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.delete_many(sentinel.filter)
    assert collection.delete_many.call_count == 1
    assert collection.delete_many.call_args_list == [call(sentinel.filter)]

def test_create_index():
    if False:
        while True:
            i = 10
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.create_index([(sentinel.path1, sentinel.order1), (sentinel.path2, sentinel.path2)])
    assert collection.create_index.call_count == 1
    assert collection.create_index.call_args_list == [call([(sentinel.path1, sentinel.order1), (sentinel.path2, sentinel.path2)])]

def test_drop_index():
    if False:
        i = 10
        return i + 15
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.drop_index(sentinel.name)
    assert collection.drop_index.call_count == 1
    assert collection.drop_index.call_args_list == [call(sentinel.name)]

def test_index_information():
    if False:
        for i in range(10):
            print('nop')
    arctic_lib = create_autospec(ArcticLibraryBinding, instance=True)
    collection = create_autospec(Collection, instance=True)
    arctic_lib.get_top_level_collection.return_value = collection
    bsons = BSONStore(arctic_lib)
    bsons.index_information()
    assert collection.index_information.call_count == 1