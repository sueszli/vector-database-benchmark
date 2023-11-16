"""
This micro benchmark compares various approaches for saving large dictionaries in a MongoDB
collection using mongoengine layer.

StackStorm has a now long standing known issue where storing large executions can take a long time.

Two main reasons for that are:

    1. mongoengine adds a lot of overhead when converting large dicts to pymongo compatible format
    2. Our EscapedDictField perfoms escaping of . and $ character on every dict key recursively and
      that is slow for large nested dictionaries.

The biggest offender in our scenario is LiveActionDB.result field which contains actual execution
result which can be very large.

The good news is that even though we use Dict field for this value, for all purposes we treat this
value in the database as a binary blob aka we perform no search on this field value. This makes
things easier.

The goal of that benchmark is to determine a more efficient approach which is also not too hard to
implement in a backward compatible manner.
"""
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
from typing import Type
import os
import json
import pytest
import mongoengine as me
from st2common.service_setup import db_setup
from st2common.models.db import stormbase
from st2common.models.db.liveaction import LiveActionDB
from st2common.persistence.liveaction import LiveAction
from st2common.fields import JSONDictField
from common import FIXTURES_DIR
from common import PYTEST_FIXTURE_FILE_PARAM_DECORATOR
from common import PYTEST_FIXTURE_FILE_PARAM_NO_8MB_DECORATOR
LiveActionDB._meta['allow_inheritance'] = True

class LiveActionDB_EscapedDynamicField(LiveActionDB):
    result = stormbase.EscapedDynamicField(default={})
    field1 = stormbase.EscapedDynamicField(default={})
    field2 = stormbase.EscapedDynamicField(default={})
    field3 = stormbase.EscapedDynamicField(default={})

class LiveActionDB_EscapedDictField(LiveActionDB):
    result = stormbase.EscapedDictField(default={})
    field1 = stormbase.EscapedDynamicField(default={}, use_header=False)
    field2 = stormbase.EscapedDynamicField(default={}, use_header=False)
    field3 = stormbase.EscapedDynamicField(default={}, use_header=False)

class LiveActionDB_JSONField(LiveActionDB):
    result = JSONDictField(default={}, use_header=False)
    field1 = JSONDictField(default={}, use_header=False)
    field2 = JSONDictField(default={}, use_header=False)
    field3 = JSONDictField(default={}, use_header=False)

class LiveActionDB_JSONFieldWithHeader(LiveActionDB):
    result = JSONDictField(default={}, use_header=True, compression_algorithm='none')
    field1 = JSONDictField(default={}, use_header=True, compression_algorithm='none')
    field2 = JSONDictField(default={}, use_header=True, compression_algorithm='none')
    field3 = JSONDictField(default={}, use_header=True, compression_algorithm='none')

class LiveActionDB_JSONFieldWithHeaderAndZstandard(LiveActionDB):
    result = JSONDictField(default={}, use_header=True, compression_algorithm='zstandard')
    field1 = JSONDictField(default={}, use_header=True, compression_algorithm='zstandard')
    field2 = JSONDictField(default={}, use_header=True, compression_algorithm='zstandard')
    field3 = JSONDictField(default={}, use_header=True, compression_algorithm='zstandard')

class LiveActionDB_StringField(LiveActionDB):
    value = me.StringField()

class LiveActionDB_BinaryField(LiveActionDB):
    value = me.BinaryField()

def get_model_class_for_approach(approach: str) -> Type[LiveActionDB]:
    if False:
        return 10
    if approach == 'escaped_dynamic_field':
        model_cls = LiveActionDB_EscapedDynamicField
    elif approach == 'escaped_dict_field':
        model_cls = LiveActionDB_EscapedDictField
    elif approach == 'json_dict_field':
        model_cls = LiveActionDB_JSONField
    elif approach == 'json_dict_field_with_header':
        model_cls = LiveActionDB_JSONFieldWithHeader
    elif approach == 'json_dict_field_with_header_and_zstd':
        model_cls = LiveActionDB_JSONFieldWithHeaderAndZstandard
    else:
        raise ValueError('Invalid approach: %s' % approach)
    return model_cls

@PYTEST_FIXTURE_FILE_PARAM_DECORATOR
@pytest.mark.parametrize('approach', ['escaped_dynamic_field', 'escaped_dict_field', 'json_dict_field', 'json_dict_field_with_header', 'json_dict_field_with_header_and_zstd'], ids=['escaped_dynamic_field', 'escaped_dict_field', 'json_dict_field', 'json_dict_field_w_header', 'json_dict_field_w_header_and_zstd'])
@pytest.mark.benchmark(group='live_action_save')
def test_save_large_execution(benchmark, fixture_file: str, approach: str) -> None:
    if False:
        print('Hello World!')
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'r') as fp:
        content = fp.read()
    data = json.loads(content)
    db_setup()
    model_cls = get_model_class_for_approach(approach=approach)

    def run_benchmark():
        if False:
            for i in range(10):
                print('nop')
        live_action_db = model_cls()
        live_action_db.status = 'succeeded'
        live_action_db.action = 'core.local'
        live_action_db.result = data
        inserted_live_action_db = LiveAction.add_or_update(live_action_db)
        return inserted_live_action_db
    inserted_live_action_db = benchmark(run_benchmark)
    retrieved_live_action_db = LiveAction.get_by_id(inserted_live_action_db.id)
    assert inserted_live_action_db.result == data
    assert inserted_live_action_db == retrieved_live_action_db

@PYTEST_FIXTURE_FILE_PARAM_NO_8MB_DECORATOR
@pytest.mark.parametrize('approach', ['escaped_dynamic_field', 'escaped_dict_field', 'json_dict_field', 'json_dict_field_with_header', 'json_dict_field_with_header_and_zstd'], ids=['escaped_dynamic_field', 'escaped_dict_field', 'json_dict_field', 'json_dict_field_w_header', 'json_dict_field_w_header_and_zstd'])
@pytest.mark.benchmark(group='live_action_save_multiple_fields')
def test_save_multiple_fields(benchmark, fixture_file: str, approach: str) -> None:
    if False:
        print('Hello World!')
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'r') as fp:
        content = fp.read()
    data = json.loads(content)
    db_setup()
    model_cls = get_model_class_for_approach(approach=approach)

    def run_benchmark():
        if False:
            return 10
        live_action_db = model_cls()
        live_action_db.status = 'succeeded'
        live_action_db.action = 'core.local'
        live_action_db.field1 = data
        live_action_db.field2 = data
        live_action_db.field3 = data
        inserted_live_action_db = LiveAction.add_or_update(live_action_db)
        return inserted_live_action_db
    inserted_live_action_db = benchmark(run_benchmark)
    retrieved_live_action_db = LiveAction.get_by_id(inserted_live_action_db.id)
    assert inserted_live_action_db.field1 == data
    assert inserted_live_action_db.field2 == data
    assert inserted_live_action_db.field3 == data
    assert inserted_live_action_db == retrieved_live_action_db

@PYTEST_FIXTURE_FILE_PARAM_DECORATOR
@pytest.mark.parametrize('approach', ['escaped_dynamic_field', 'escaped_dict_field', 'json_dict_field', 'json_dict_field_with_header', 'json_dict_field_with_header_and_zstd'], ids=['escaped_dynamic_field', 'escaped_dict_field', 'json_dict_field', 'json_dict_field_w_header', 'json_dict_field_w_header_and_zstd'])
@pytest.mark.benchmark(group='live_action_read')
def test_read_large_execution(benchmark, fixture_file: str, approach: str) -> None:
    if False:
        i = 10
        return i + 15
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'r') as fp:
        content = fp.read()
    data = json.loads(content)
    db_setup()
    model_cls = get_model_class_for_approach(approach=approach)
    live_action_db = model_cls()
    live_action_db.status = 'succeeded'
    live_action_db.action = 'core.local'
    live_action_db.result = data
    inserted_live_action_db = LiveAction.add_or_update(live_action_db)

    def run_benchmark():
        if False:
            print('Hello World!')
        retrieved_live_action_db = LiveAction.get_by_id(inserted_live_action_db.id)
        return retrieved_live_action_db
    retrieved_live_action_db = benchmark(run_benchmark)
    assert retrieved_live_action_db == inserted_live_action_db
    assert retrieved_live_action_db.result == data

@PYTEST_FIXTURE_FILE_PARAM_DECORATOR
@pytest.mark.parametrize('approach', ['string_field', 'binary_field'], ids=['string_field', 'binary_field'])
@pytest.mark.benchmark(group='test_model_save')
def test_save_large_string_value(benchmark, fixture_file: str, approach: str) -> None:
    if False:
        print('Hello World!')
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'rb') as fp:
        content = fp.read()
    db_setup()
    if approach == 'string_field':
        model_cls = LiveActionDB_StringField
        content = content.decode('utf-8')
    elif approach == 'binary_field':
        model_cls = LiveActionDB_BinaryField
    else:
        raise ValueError('Unsupported approach')

    def run_benchmark():
        if False:
            for i in range(10):
                print('nop')
        live_action_db = model_cls()
        live_action_db.status = 'succeeded'
        live_action_db.action = 'core.local'
        live_action_db.value = content
        inserted_live_action_db = LiveAction.add_or_update(live_action_db)
        return inserted_live_action_db
    inserted_live_action_db = benchmark(run_benchmark)
    assert bool(inserted_live_action_db.value)

@PYTEST_FIXTURE_FILE_PARAM_DECORATOR
@pytest.mark.parametrize('approach', ['string_field', 'binary_field'], ids=['string_field', 'binary_field'])
@pytest.mark.benchmark(group='test_model_read')
def test_read_large_string_value(benchmark, fixture_file: str, approach: str) -> None:
    if False:
        return 10
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'rb') as fp:
        content = fp.read()
    db_setup()
    if approach == 'string_field':
        model_cls = LiveActionDB_StringField
        content = content.decode('utf-8')
    elif approach == 'binary_field':
        model_cls = LiveActionDB_BinaryField
    else:
        raise ValueError('Unsupported approach')
    live_action_db = model_cls()
    live_action_db.status = 'succeeded'
    live_action_db.action = 'core.local'
    live_action_db.value = content
    inserted_live_action_db = LiveAction.add_or_update(live_action_db)

    def run_benchmark():
        if False:
            print('Hello World!')
        retrieved_live_action_db = LiveAction.get_by_id(inserted_live_action_db.id)
        return retrieved_live_action_db
    retrieved_live_action_db = benchmark(run_benchmark)
    assert retrieved_live_action_db == inserted_live_action_db
    assert retrieved_live_action_db.value == content