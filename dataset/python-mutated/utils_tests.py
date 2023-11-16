import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import json
import os
import re
from typing import Any, Optional
from unittest.mock import Mock, patch
from superset.databases.commands.exceptions import DatabaseInvalidError
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
import numpy as np
import pandas as pd
import pytest
from flask import Flask, g
import marshmallow
from sqlalchemy.exc import ArgumentError
import tests.integration_tests.test_app
from superset import app, db, security_manager
from superset.constants import NO_TIME_RANGE
from superset.exceptions import CertificateException, SupersetException
from superset.models.core import Database, Log
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.utils.core import base_json_conv, cast_to_num, convert_legacy_filters_into_adhoc, create_ssl_cert_file, DTTM_ALIAS, extract_dataframe_dtypes, format_timedelta, GenericDataType, get_form_data_token, as_list, get_email_address_list, get_stacktrace, json_int_dttm_ser, json_iso_dttm_ser, merge_extra_filters, merge_extra_form_data, merge_request_params, normalize_dttm_col, parse_ssl_cert, parse_js_uri_path_item, split, validate_json, zlib_compress, zlib_decompress, DateColumn
from superset.utils.database import get_or_create_db
from superset.utils import schema
from superset.utils.hashing import md5_sha_from_str
from superset.views.utils import build_extra_filters, get_form_data
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices, load_world_bank_data
from .fixtures.certificates import ssl_certificate

class TestUtils(SupersetTestCase):

    def test_json_int_dttm_ser(self):
        if False:
            while True:
                i = 10
        dttm = datetime(2020, 1, 1)
        ts = 1577836800000.0
        assert json_int_dttm_ser(dttm) == ts
        assert json_int_dttm_ser(date(2020, 1, 1)) == ts
        assert json_int_dttm_ser(datetime(1970, 1, 1)) == 0
        assert json_int_dttm_ser(date(1970, 1, 1)) == 0
        assert json_int_dttm_ser(dttm + timedelta(milliseconds=1)) == ts + 1
        assert json_int_dttm_ser(np.int64(1)) == 1
        with self.assertRaises(TypeError):
            json_int_dttm_ser(np.datetime64())

    def test_json_iso_dttm_ser(self):
        if False:
            print('Hello World!')
        dttm = datetime(2020, 1, 1)
        dt = date(2020, 1, 1)
        t = time()
        assert json_iso_dttm_ser(dttm) == dttm.isoformat()
        assert json_iso_dttm_ser(dt) == dt.isoformat()
        assert json_iso_dttm_ser(t) == t.isoformat()
        assert json_iso_dttm_ser(np.int64(1)) == 1
        assert json_iso_dttm_ser(np.datetime64(), pessimistic=True) == "Unserializable [<class 'numpy.datetime64'>]"
        with self.assertRaises(TypeError):
            json_iso_dttm_ser(np.datetime64())

    def test_base_json_conv(self):
        if False:
            print('Hello World!')
        assert isinstance(base_json_conv(np.bool_(1)), bool)
        assert isinstance(base_json_conv(np.int64(1)), int)
        assert isinstance(base_json_conv(np.array([1, 2, 3])), list)
        assert base_json_conv(np.array(None)) is None
        assert isinstance(base_json_conv({1}), list)
        assert isinstance(base_json_conv(Decimal('1.0')), float)
        assert isinstance(base_json_conv(uuid.uuid4()), str)
        assert isinstance(base_json_conv(time()), str)
        assert isinstance(base_json_conv(timedelta(0)), str)
        assert isinstance(base_json_conv(b''), str)
        assert base_json_conv(bytes('', encoding='utf-16')) == '[bytes]'
        with pytest.raises(TypeError):
            base_json_conv(np.datetime64())

    def test_zlib_compression(self):
        if False:
            i = 10
            return i + 15
        json_str = '{"test": 1}'
        blob = zlib_compress(json_str)
        got_str = zlib_decompress(blob)
        self.assertEqual(json_str, got_str)

    def test_merge_extra_filters(self):
        if False:
            i = 10
            return i + 15
        form_data = {'A': 1, 'B': 2, 'c': 'test'}
        expected = {**form_data, 'adhoc_filters': [], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)
        form_data = {'A': 1, 'B': 2, 'c': 'test', 'extra_filters': []}
        expected = {'A': 1, 'B': 2, 'c': 'test', 'adhoc_filters': [], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': 'someval'}, {'col': 'B', 'op': '==', 'val': ['c1', 'c2']}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'filterOptionName': '90cfb3c34852eb3bc741b0cc20053b46', 'isExtra': True, 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'filterOptionName': '6c178d069965f1c02640661280415d96', 'isExtra': True, 'operator': '==', 'subject': 'B'}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': 'someval'}, {'col': 'B', 'op': '==', 'val': ['c1', 'c2']}], 'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['G1', 'g2'], 'expressionType': 'SIMPLE', 'operator': '!=', 'subject': 'D'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['G1', 'g2'], 'expressionType': 'SIMPLE', 'operator': '!=', 'subject': 'D'}, {'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'filterOptionName': '90cfb3c34852eb3bc741b0cc20053b46', 'isExtra': True, 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'filterOptionName': '6c178d069965f1c02640661280415d96', 'isExtra': True, 'operator': '==', 'subject': 'B'}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)
        form_data = {'extra_filters': [{'col': '__time_range', 'op': 'in', 'val': '1 year ago :'}, {'col': '__time_col', 'op': 'in', 'val': 'birth_year'}, {'col': '__time_grain', 'op': 'in', 'val': 'years'}, {'col': 'A', 'op': 'like', 'val': 'hello'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'hello', 'expressionType': 'SIMPLE', 'filterOptionName': 'e3cbdd92a2ae23ca92c6d7fca42e36a6', 'isExtra': True, 'operator': 'like', 'subject': 'A'}], 'time_range': '1 year ago :', 'granularity_sqla': 'birth_year', 'time_grain_sqla': 'years', 'applied_time_extras': {'__time_range': '1 year ago :', '__time_col': 'birth_year', '__time_grain': 'years'}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)

    def test_merge_extra_filters_ignores_empty_filters(self):
        if False:
            for i in range(10):
                print('nop')
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': ''}, {'col': 'B', 'op': '==', 'val': []}]}
        expected = {'adhoc_filters': [], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)

    def test_merge_extra_filters_ignores_nones(self):
        if False:
            i = 10
            return i + 15
        form_data = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': '', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': None}], 'extra_filters': [{'col': 'B', 'op': '==', 'val': []}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': '', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': None}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)

    def test_merge_extra_filters_ignores_equal_filters(self):
        if False:
            print('Hello World!')
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': 'someval'}, {'col': 'B', 'op': '==', 'val': ['c1', 'c2']}, {'col': 'c', 'op': 'in', 'val': ['c1', 1, None]}], 'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}, {'clause': 'WHERE', 'comparator': ['c1', 1, None], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'c'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}, {'clause': 'WHERE', 'comparator': ['c1', 1, None], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'c'}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)

    def test_merge_extra_filters_merges_different_val_types(self):
        if False:
            for i in range(10):
                print('nop')
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': ['g1', 'g2']}, {'col': 'B', 'op': '==', 'val': ['c1', 'c2']}], 'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}, {'clause': 'WHERE', 'comparator': ['g1', 'g2'], 'expressionType': 'SIMPLE', 'filterOptionName': 'c11969c994b40a83a4ae7d48ff1ea28e', 'isExtra': True, 'operator': 'in', 'subject': 'a'}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': 'someval'}, {'col': 'B', 'op': '==', 'val': ['c1', 'c2']}], 'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['g1', 'g2'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['g1', 'g2'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}, {'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'filterOptionName': '90cfb3c34852eb3bc741b0cc20053b46', 'isExtra': True, 'operator': 'in', 'subject': 'a'}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)

    def test_merge_extra_filters_adds_unequal_lists(self):
        if False:
            return 10
        form_data = {'extra_filters': [{'col': 'a', 'op': 'in', 'val': ['g1', 'g2', 'g3']}, {'col': 'B', 'op': '==', 'val': ['c1', 'c2', 'c3']}], 'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['g1', 'g2'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['g1', 'g2'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2'], 'expressionType': 'SIMPLE', 'operator': '==', 'subject': 'B'}, {'clause': 'WHERE', 'comparator': ['g1', 'g2', 'g3'], 'expressionType': 'SIMPLE', 'filterOptionName': '21cbb68af7b17e62b3b2f75e2190bfd7', 'isExtra': True, 'operator': 'in', 'subject': 'a'}, {'clause': 'WHERE', 'comparator': ['c1', 'c2', 'c3'], 'expressionType': 'SIMPLE', 'filterOptionName': '0a8dcb928f1f4bba97643c6e68d672f1', 'isExtra': True, 'operator': '==', 'subject': 'B'}], 'applied_time_extras': {}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, expected)

    def test_merge_extra_filters_when_applied_time_extras_predefined(self):
        if False:
            for i in range(10):
                print('nop')
        form_data = {'applied_time_extras': {'__time_range': 'Last week'}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, {'applied_time_extras': {'__time_range': 'Last week'}, 'adhoc_filters': []})

    def test_merge_request_params_when_url_params_undefined(self):
        if False:
            for i in range(10):
                print('nop')
        form_data = {'since': '2000', 'until': 'now'}
        url_params = {'form_data': form_data, 'dashboard_ids': '(1,2,3,4,5)'}
        merge_request_params(form_data, url_params)
        self.assertIn('url_params', form_data.keys())
        self.assertIn('dashboard_ids', form_data['url_params'])
        self.assertNotIn('form_data', form_data.keys())

    def test_merge_request_params_when_url_params_predefined(self):
        if False:
            while True:
                i = 10
        form_data = {'since': '2000', 'until': 'now', 'url_params': {'abc': '123', 'dashboard_ids': '(1,2,3)'}}
        url_params = {'form_data': form_data, 'dashboard_ids': '(1,2,3,4,5)'}
        merge_request_params(form_data, url_params)
        self.assertIn('url_params', form_data.keys())
        self.assertIn('abc', form_data['url_params'])
        self.assertEqual(url_params['dashboard_ids'], form_data['url_params']['dashboard_ids'])

    def test_format_timedelta(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(format_timedelta(timedelta(0)), '0:00:00')
        self.assertEqual(format_timedelta(timedelta(days=1)), '1 day, 0:00:00')
        self.assertEqual(format_timedelta(timedelta(minutes=-6)), '-0:06:00')
        self.assertEqual(format_timedelta(timedelta(0) - timedelta(days=1, hours=5, minutes=6)), '-1 day, 5:06:00')
        self.assertEqual(format_timedelta(timedelta(0) - timedelta(days=16, hours=4, minutes=3)), '-16 days, 4:03:00')

    def test_validate_json(self):
        if False:
            return 10
        valid = '{"a": 5, "b": [1, 5, ["g", "h"]]}'
        self.assertIsNone(validate_json(valid))
        invalid = '{"a": 5, "b": [1, 5, ["g", "h]]}'
        with self.assertRaises(SupersetException):
            validate_json(invalid)

    def test_convert_legacy_filters_into_adhoc_where(self):
        if False:
            i = 10
            return i + 15
        form_data = {'where': 'a = 1'}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'expressionType': 'SQL', 'filterOptionName': '46fb6d7891e23596e42ae38da94a57e0', 'sqlExpression': 'a = 1'}]}
        convert_legacy_filters_into_adhoc(form_data)
        self.assertEqual(form_data, expected)

    def test_convert_legacy_filters_into_adhoc_filters(self):
        if False:
            print('Hello World!')
        form_data = {'filters': [{'col': 'a', 'op': 'in', 'val': 'someval'}]}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'someval', 'expressionType': 'SIMPLE', 'filterOptionName': '135c7ee246666b840a3d7a9c3a30cf38', 'operator': 'in', 'subject': 'a'}]}
        convert_legacy_filters_into_adhoc(form_data)
        self.assertEqual(form_data, expected)

    def test_convert_legacy_filters_into_adhoc_present_and_empty(self):
        if False:
            return 10
        form_data = {'adhoc_filters': [], 'where': 'a = 1'}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'expressionType': 'SQL', 'filterOptionName': '46fb6d7891e23596e42ae38da94a57e0', 'sqlExpression': 'a = 1'}]}
        convert_legacy_filters_into_adhoc(form_data)
        self.assertEqual(form_data, expected)

    def test_convert_legacy_filters_into_adhoc_having(self):
        if False:
            while True:
                i = 10
        form_data = {'having': 'COUNT(1) = 1'}
        expected = {'adhoc_filters': [{'clause': 'HAVING', 'expressionType': 'SQL', 'filterOptionName': '683f1c26466ab912f75a00842e0f2f7b', 'sqlExpression': 'COUNT(1) = 1'}]}
        convert_legacy_filters_into_adhoc(form_data)
        self.assertEqual(form_data, expected)

    def test_convert_legacy_filters_into_adhoc_present_and_nonempty(self):
        if False:
            i = 10
            return i + 15
        form_data = {'adhoc_filters': [{'clause': 'WHERE', 'expressionType': 'SQL', 'sqlExpression': 'a = 1'}], 'filters': [{'col': 'a', 'op': 'in', 'val': 'someval'}], 'having': 'COUNT(1) = 1'}
        expected = {'adhoc_filters': [{'clause': 'WHERE', 'expressionType': 'SQL', 'sqlExpression': 'a = 1'}]}
        convert_legacy_filters_into_adhoc(form_data)
        self.assertEqual(form_data, expected)

    def test_parse_js_uri_path_items_eval_undefined(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(parse_js_uri_path_item('undefined', eval_undefined=True))
        self.assertIsNone(parse_js_uri_path_item('null', eval_undefined=True))
        self.assertEqual('undefined', parse_js_uri_path_item('undefined'))
        self.assertEqual('null', parse_js_uri_path_item('null'))

    def test_parse_js_uri_path_items_unquote(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('slashed/name', parse_js_uri_path_item('slashed%2fname'))
        self.assertEqual('slashed%2fname', parse_js_uri_path_item('slashed%2fname', unquote=False))

    def test_parse_js_uri_path_items_item_optional(self):
        if False:
            return 10
        self.assertIsNone(parse_js_uri_path_item(None))
        self.assertIsNotNone(parse_js_uri_path_item('item'))

    def test_get_stacktrace(self):
        if False:
            return 10
        with app.app_context():
            app.config['SHOW_STACKTRACE'] = True
            try:
                raise Exception('NONONO!')
            except Exception:
                stacktrace = get_stacktrace()
                self.assertIn('NONONO', stacktrace)
            app.config['SHOW_STACKTRACE'] = False
            try:
                raise Exception('NONONO!')
            except Exception:
                stacktrace = get_stacktrace()
                assert stacktrace is None

    def test_split(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(list(split('a b')), ['a', 'b'])
        self.assertEqual(list(split('a,b', delimiter=',')), ['a', 'b'])
        self.assertEqual(list(split('a,(b,a)', delimiter=',')), ['a', '(b,a)'])
        self.assertEqual(list(split('a,(b,a),"foo , bar"', delimiter=',')), ['a', '(b,a)', '"foo , bar"'])
        self.assertEqual(list(split("a,'b,c'", delimiter=',', quote="'")), ['a', "'b,c'"])
        self.assertEqual(list(split('a "b c"')), ['a', '"b c"'])
        self.assertEqual(list(split('a "b \\" c"')), ['a', '"b \\" c"'])

    def test_get_or_create_db(self):
        if False:
            for i in range(10):
                print('nop')
        get_or_create_db('test_db', 'sqlite:///superset.db')
        database = db.session.query(Database).filter_by(database_name='test_db').one()
        self.assertIsNotNone(database)
        self.assertEqual(database.sqlalchemy_uri, 'sqlite:///superset.db')
        self.assertIsNotNone(security_manager.find_permission_view_menu('database_access', database.perm))
        get_or_create_db('test_db', 'sqlite:///changed.db')
        database = db.session.query(Database).filter_by(database_name='test_db').one()
        self.assertEqual(database.sqlalchemy_uri, 'sqlite:///changed.db')
        db.session.delete(database)
        db.session.commit()

    def test_get_or_create_db_invalid_uri(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(DatabaseInvalidError):
            get_or_create_db('test_db', 'yoursql:superset.db/()')

    def test_get_or_create_db_existing_invalid_uri(self):
        if False:
            print('Hello World!')
        database = get_or_create_db('test_db', 'sqlite:///superset.db')
        database.sqlalchemy_uri = 'None'
        db.session.commit()
        database = get_or_create_db('test_db', 'sqlite:///superset.db')
        assert database.sqlalchemy_uri == 'sqlite:///superset.db'

    def test_as_list(self):
        if False:
            print('Hello World!')
        self.assertListEqual(as_list(123), [123])
        self.assertListEqual(as_list([123]), [123])
        self.assertListEqual(as_list('foo'), ['foo'])

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_build_extra_filters(self):
        if False:
            return 10
        world_health = db.session.query(Dashboard).filter_by(slug='world_health').one()
        layout = json.loads(world_health.position_json)
        filter_ = db.session.query(Slice).filter_by(slice_name='Region Filter').one()
        world = db.session.query(Slice).filter_by(slice_name="World's Population").one()
        box_plot = db.session.query(Slice).filter_by(slice_name='Box plot').one()
        treemap = db.session.query(Slice).filter_by(slice_name='Treemap').one()
        filter_scopes = {str(filter_.id): {'region': {'scope': ['ROOT_ID'], 'immune': [treemap.id]}, 'country_name': {'scope': ['ROOT_ID'], 'immune': [treemap.id, box_plot.id]}}}
        default_filters = {str(filter_.id): {'region': ['North America'], 'country_name': ['United States']}}
        assert build_extra_filters(layout, filter_scopes, default_filters, treemap.id) == []
        assert build_extra_filters(layout, filter_scopes, default_filters, world.id) == [{'col': 'region', 'op': '==', 'val': 'North America'}, {'col': 'country_name', 'op': 'in', 'val': ['United States']}]
        assert build_extra_filters(layout, filter_scopes, default_filters, box_plot.id) == [{'col': 'region', 'op': '==', 'val': 'North America'}]

    def test_merge_extra_filters_with_no_extras(self):
        if False:
            return 10
        form_data = {'time_range': 'Last 10 days'}
        merge_extra_form_data(form_data)
        self.assertEqual(form_data, {'time_range': 'Last 10 days', 'adhoc_filters': []})

    def test_merge_extra_filters_with_unset_legacy_time_range(self):
        if False:
            return 10
        '\n        Make sure native filter is applied if filter box time range is unset.\n        '
        form_data = {'time_range': 'Last 10 days', 'extra_filters': [{'col': '__time_range', 'op': '==', 'val': NO_TIME_RANGE}], 'extra_form_data': {'time_range': 'Last year'}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, {'time_range': 'Last year', 'applied_time_extras': {}, 'adhoc_filters': []})

    def test_merge_extra_filters_with_conflicting_time_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure filter box takes precedence if both native filter and filter box\n        time ranges are set.\n        '
        form_data = {'time_range': 'Last 10 days', 'extra_filters': [{'col': '__time_range', 'op': '==', 'val': 'Last week'}], 'extra_form_data': {'time_range': 'Last year'}}
        merge_extra_filters(form_data)
        self.assertEqual(form_data, {'time_range': 'Last week', 'applied_time_extras': {'__time_range': 'Last week'}, 'adhoc_filters': []})

    def test_merge_extra_filters_with_extras(self):
        if False:
            while True:
                i = 10
        form_data = {'time_range': 'Last 10 days', 'extra_form_data': {'filters': [{'col': 'foo', 'op': 'IN', 'val': ['bar']}], 'adhoc_filters': [{'expressionType': 'SQL', 'clause': 'WHERE', 'sqlExpression': '1 = 0'}], 'time_range': 'Last 100 years', 'time_grain_sqla': 'PT1M', 'relative_start': 'now'}}
        merge_extra_form_data(form_data)
        adhoc_filters = form_data['adhoc_filters']
        assert adhoc_filters[0] == {'clause': 'WHERE', 'expressionType': 'SQL', 'isExtra': True, 'sqlExpression': '1 = 0'}
        converted_filter = adhoc_filters[1]
        del converted_filter['filterOptionName']
        assert converted_filter == {'clause': 'WHERE', 'comparator': ['bar'], 'expressionType': 'SIMPLE', 'isExtra': True, 'operator': 'IN', 'subject': 'foo'}
        assert form_data['time_range'] == 'Last 100 years'
        assert form_data['time_grain_sqla'] == 'PT1M'
        assert form_data['extras']['relative_start'] == 'now'

    def test_ssl_certificate_parse(self):
        if False:
            for i in range(10):
                print('nop')
        parsed_certificate = parse_ssl_cert(ssl_certificate)
        self.assertEqual(parsed_certificate.serial_number, 12355228710836649848)

    def test_ssl_certificate_file_creation(self):
        if False:
            i = 10
            return i + 15
        path = create_ssl_cert_file(ssl_certificate)
        expected_filename = md5_sha_from_str(ssl_certificate)
        self.assertIn(expected_filename, path)
        self.assertTrue(os.path.exists(path))

    def test_get_email_address_list(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(get_email_address_list('a@a'), ['a@a'])
        self.assertEqual(get_email_address_list(' a@a '), ['a@a'])
        self.assertEqual(get_email_address_list('a@a\n'), ['a@a'])
        self.assertEqual(get_email_address_list(',a@a;'), ['a@a'])
        self.assertEqual(get_email_address_list(',a@a; b@b c@c a-c@c; d@d, f@f'), ['a@a', 'b@b', 'c@c', 'a-c@c', 'd@d', 'f@f'])

    def test_get_form_data_default(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with app.test_request_context():
            (form_data, slc) = get_form_data()
            self.assertEqual(slc, None)

    def test_get_form_data_request_args(self) -> None:
        if False:
            return 10
        with app.test_request_context(query_string={'form_data': json.dumps({'foo': 'bar'})}):
            (form_data, slc) = get_form_data()
            self.assertEqual(form_data, {'foo': 'bar'})
            self.assertEqual(slc, None)

    def test_get_form_data_request_form(self) -> None:
        if False:
            while True:
                i = 10
        with app.test_request_context(data={'form_data': json.dumps({'foo': 'bar'})}):
            (form_data, slc) = get_form_data()
            self.assertEqual(form_data, {'foo': 'bar'})
            self.assertEqual(slc, None)

    def test_get_form_data_request_form_with_queries(self) -> None:
        if False:
            print('Hello World!')
        with app.test_request_context(data={'form_data': json.dumps({'queries': [{'url_params': {'foo': 'bar'}}]})}):
            (form_data, slc) = get_form_data()
            self.assertEqual(form_data, {'url_params': {'foo': 'bar'}})
            self.assertEqual(slc, None)

    def test_get_form_data_request_args_and_form(self) -> None:
        if False:
            return 10
        with app.test_request_context(data={'form_data': json.dumps({'foo': 'bar'})}, query_string={'form_data': json.dumps({'baz': 'bar'})}):
            (form_data, slc) = get_form_data()
            self.assertEqual(form_data, {'baz': 'bar', 'foo': 'bar'})
            self.assertEqual(slc, None)

    def test_get_form_data_globals(self) -> None:
        if False:
            print('Hello World!')
        with app.test_request_context():
            g.form_data = {'foo': 'bar'}
            (form_data, slc) = get_form_data()
            delattr(g, 'form_data')
            self.assertEqual(form_data, {'foo': 'bar'})
            self.assertEqual(slc, None)

    def test_get_form_data_corrupted_json(self) -> None:
        if False:
            i = 10
            return i + 15
        with app.test_request_context(data={'form_data': "{x: '2324'}"}, query_string={'form_data': '{"baz": "bar"'}):
            (form_data, slc) = get_form_data()
            self.assertEqual(form_data, {})
            self.assertEqual(slc, None)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_log_this(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        slc = self.get_slice('Top 10 Girl Name Share', db.session)
        dashboard_id = 1
        assert slc.viz is not None
        resp = self.get_json_resp(f'/superset/explore_json/{slc.datasource_type}/{slc.datasource_id}/' + f'?form_data={{"slice_id": {slc.id}}}&dashboard_id={dashboard_id}', {'form_data': json.dumps(slc.viz.form_data)})
        record = db.session.query(Log).filter_by(action='explore_json', slice_id=slc.id).order_by(Log.dttm.desc()).first()
        self.assertEqual(record.dashboard_id, dashboard_id)
        self.assertEqual(json.loads(record.json)['dashboard_id'], str(dashboard_id))
        self.assertEqual(json.loads(record.json)['form_data']['slice_id'], slc.id)
        self.assertEqual(json.loads(record.json)['form_data']['viz_type'], slc.viz.form_data['viz_type'])

    def test_schema_validate_json(self):
        if False:
            for i in range(10):
                print('nop')
        valid = '{"a": 5, "b": [1, 5, ["g", "h"]]}'
        self.assertIsNone(schema.validate_json(valid))
        invalid = '{"a": 5, "b": [1, 5, ["g", "h]]}'
        self.assertRaises(marshmallow.ValidationError, schema.validate_json, invalid)

    def test_schema_one_of_case_insensitive(self):
        if False:
            i = 10
            return i + 15
        validator = schema.OneOfCaseInsensitive(choices=[1, 2, 3, 'FoO', 'BAR', 'baz'])
        self.assertEqual(1, validator(1))
        self.assertEqual(2, validator(2))
        self.assertEqual('FoO', validator('FoO'))
        self.assertEqual('FOO', validator('FOO'))
        self.assertEqual('bar', validator('bar'))
        self.assertEqual('BaZ', validator('BaZ'))
        self.assertRaises(marshmallow.ValidationError, validator, 'qwerty')
        self.assertRaises(marshmallow.ValidationError, validator, 4)

    def test_cast_to_num(self) -> None:
        if False:
            while True:
                i = 10
        assert cast_to_num('5') == 5
        assert cast_to_num('5.2') == 5.2
        assert cast_to_num(10) == 10
        assert cast_to_num(10.1) == 10.1
        assert cast_to_num(None) is None
        assert cast_to_num('this is not a string') is None

    def test_get_form_data_token(self):
        if False:
            print('Hello World!')
        assert get_form_data_token({'token': 'token_abcdefg1'}) == 'token_abcdefg1'
        generated_token = get_form_data_token({})
        assert re.match('^token_[a-z0-9]{8}$', generated_token) is not None

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_extract_dataframe_dtypes(self):
        if False:
            print('Hello World!')
        slc = self.get_slice('Girls', db.session)
        cols: tuple[tuple[str, GenericDataType, list[Any]], ...] = (('dt', GenericDataType.TEMPORAL, [date(2021, 2, 4), date(2021, 2, 4)]), ('dttm', GenericDataType.TEMPORAL, [datetime(2021, 2, 4, 1, 1, 1), datetime(2021, 2, 4, 1, 1, 1)]), ('str', GenericDataType.STRING, ['foo', 'foo']), ('int', GenericDataType.NUMERIC, [1, 1]), ('float', GenericDataType.NUMERIC, [0.5, 0.5]), ('mixed-int-float', GenericDataType.NUMERIC, [0.5, 1.0]), ('bool', GenericDataType.BOOLEAN, [True, False]), ('mixed-str-int', GenericDataType.STRING, ['abc', 1.0]), ('obj', GenericDataType.STRING, [{'a': 1}, {'a': 1}]), ('dt_null', GenericDataType.TEMPORAL, [None, date(2021, 2, 4)]), ('dttm_null', GenericDataType.TEMPORAL, [None, datetime(2021, 2, 4, 1, 1, 1)]), ('str_null', GenericDataType.STRING, [None, 'foo']), ('int_null', GenericDataType.NUMERIC, [None, 1]), ('float_null', GenericDataType.NUMERIC, [None, 0.5]), ('bool_null', GenericDataType.BOOLEAN, [None, False]), ('obj_null', GenericDataType.STRING, [None, {'a': 1}]), ('ds', GenericDataType.TEMPORAL, [None, {'ds': '2017-01-01'}]))
        df = pd.DataFrame(data={col[0]: col[2] for col in cols})
        assert extract_dataframe_dtypes(df, slc.datasource) == [col[1] for col in cols]

    def test_normalize_dttm_col(self):
        if False:
            i = 10
            return i + 15

        def normalize_col(df: pd.DataFrame, timestamp_format: Optional[str], offset: int, time_shift: Optional[str]) -> pd.DataFrame:
            if False:
                return 10
            df = df.copy()
            normalize_dttm_col(df, tuple([DateColumn.get_legacy_time_column(timestamp_format=timestamp_format, offset=offset, time_shift=time_shift)]))
            return df
        ts = pd.Timestamp(2021, 2, 15, 19, 0, 0, 0)
        df = pd.DataFrame([{'__timestamp': ts, 'a': 1}])
        assert normalize_col(df, None, 0, None)[DTTM_ALIAS][0] == ts
        assert normalize_col(df, 'epoch_ms', 0, None)[DTTM_ALIAS][0] == ts
        assert normalize_col(df, 'epoch_s', 0, None)[DTTM_ALIAS][0] == ts
        assert normalize_col(df, None, 1, None)[DTTM_ALIAS][0] == pd.Timestamp(2021, 2, 15, 20, 0, 0, 0)
        assert normalize_col(df, None, 1, '30 minutes')[DTTM_ALIAS][0] == pd.Timestamp(2021, 2, 15, 20, 30, 0, 0)
        df = pd.DataFrame([{'__timestamp': ts.timestamp(), 'a': 1}])
        assert normalize_col(df, 'epoch_s', 0, None)[DTTM_ALIAS][0] == ts
        df = pd.DataFrame([{'__timestamp': ts.timestamp() * 1000, 'a': 1}])
        assert normalize_col(df, 'epoch_ms', 0, None)[DTTM_ALIAS][0] == ts
        df = pd.DataFrame([{'__timestamp': '1677-09-21 00:00:00', 'a': 1}])
        with pytest.raises(pd.errors.OutOfBoundsDatetime):
            normalize_col(df, None, 0, None)