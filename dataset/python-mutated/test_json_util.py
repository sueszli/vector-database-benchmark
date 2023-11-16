"""Test some utilities for working with JSON and PyMongo."""
from __future__ import annotations
import datetime
import json
import re
import sys
import uuid
from typing import Any, List, MutableMapping
from bson.codec_options import CodecOptions, DatetimeConversion
sys.path[0:0] = ['']
from test import IntegrationTest, unittest
from bson import EPOCH_AWARE, EPOCH_NAIVE, SON, DatetimeMS, json_util
from bson.binary import ALL_UUID_REPRESENTATIONS, MD5_SUBTYPE, STANDARD, USER_DEFINED_SUBTYPE, Binary, UuidRepresentation
from bson.code import Code
from bson.datetime_ms import _max_datetime_ms
from bson.dbref import DBRef
from bson.int64 import Int64
from bson.json_util import LEGACY_JSON_OPTIONS, DatetimeRepresentation, JSONMode, JSONOptions
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.timestamp import Timestamp
from bson.tz_util import FixedOffset, utc
STRICT_JSON_OPTIONS = JSONOptions(strict_number_long=True, datetime_representation=DatetimeRepresentation.ISO8601, strict_uuid=True, json_mode=JSONMode.LEGACY)

class TestJsonUtil(unittest.TestCase):

    def round_tripped(self, doc, **kwargs):
        if False:
            return 10
        return json_util.loads(json_util.dumps(doc, **kwargs), **kwargs)

    def round_trip(self, doc, **kwargs):
        if False:
            while True:
                i = 10
        self.assertEqual(doc, self.round_tripped(doc, **kwargs))

    def test_basic(self):
        if False:
            while True:
                i = 10
        self.round_trip({'hello': 'world'})

    def test_loads_bytes(self):
        if False:
            print('Hello World!')
        string = b'{"hello": "world"}'
        self.assertEqual(json_util.loads(bytes(string)), {'hello': 'world'})
        self.assertEqual(json_util.loads(bytearray(string)), {'hello': 'world'})

    def test_json_options_with_options(self):
        if False:
            return 10
        opts = JSONOptions(datetime_representation=DatetimeRepresentation.NUMBERLONG, json_mode=JSONMode.LEGACY)
        self.assertEqual(opts.datetime_representation, DatetimeRepresentation.NUMBERLONG)
        opts2 = opts.with_options(datetime_representation=DatetimeRepresentation.ISO8601, json_mode=JSONMode.LEGACY)
        self.assertEqual(opts2.datetime_representation, DatetimeRepresentation.ISO8601)
        opts = JSONOptions(strict_number_long=True, json_mode=JSONMode.LEGACY)
        self.assertEqual(opts.strict_number_long, True)
        opts2 = opts.with_options(strict_number_long=False)
        self.assertEqual(opts2.strict_number_long, False)
        opts = json_util.CANONICAL_JSON_OPTIONS
        self.assertNotEqual(opts.uuid_representation, UuidRepresentation.JAVA_LEGACY)
        opts2 = opts.with_options(uuid_representation=UuidRepresentation.JAVA_LEGACY)
        self.assertEqual(opts2.uuid_representation, UuidRepresentation.JAVA_LEGACY)
        self.assertEqual(opts2.document_class, dict)
        opts3 = opts2.with_options(document_class=SON)
        self.assertEqual(opts3.uuid_representation, UuidRepresentation.JAVA_LEGACY)
        self.assertEqual(opts3.document_class, SON)

    def test_objectid(self):
        if False:
            for i in range(10):
                print('nop')
        self.round_trip({'id': ObjectId()})

    def test_dbref(self):
        if False:
            while True:
                i = 10
        self.round_trip({'ref': DBRef('foo', 5)})
        self.round_trip({'ref': DBRef('foo', 5, 'db')})
        self.round_trip({'ref': DBRef('foo', ObjectId())})
        self.assertEqual('{"$ref": "collection", "$id": 1, "$db": "db"}', json_util.dumps(DBRef('collection', 1, 'db')))

    def test_datetime(self):
        if False:
            i = 10
            return i + 15
        tz_aware_opts = json_util.DEFAULT_JSON_OPTIONS.with_options(tz_aware=True)
        self.round_trip({'date': datetime.datetime(2009, 12, 9, 15, 49, 45, 191000, utc)}, json_options=tz_aware_opts)
        self.round_trip({'date': datetime.datetime(2009, 12, 9, 15, 49, 45, 191000)})
        for jsn in ['{"dt": { "$date" : "1970-01-01T00:00:00.000+0000"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000000+0000"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000+00:00"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000000+00:00"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000000+00"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000Z"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000000Z"}}', '{"dt": { "$date" : "1970-01-01T00:00:00Z"}}', '{"dt": {"$date": "1970-01-01T00:00:00.000"}}', '{"dt": { "$date" : "1970-01-01T00:00:00"}}', '{"dt": { "$date" : "1970-01-01T00:00:00.000000"}}', '{"dt": { "$date" : "1969-12-31T16:00:00.000-0800"}}', '{"dt": { "$date" : "1969-12-31T16:00:00.000000-0800"}}', '{"dt": { "$date" : "1969-12-31T16:00:00.000-08:00"}}', '{"dt": { "$date" : "1969-12-31T16:00:00.000000-08:00"}}', '{"dt": { "$date" : "1969-12-31T16:00:00.000000-08"}}', '{"dt": { "$date" : "1970-01-01T01:00:00.000+0100"}}', '{"dt": { "$date" : "1970-01-01T01:00:00.000000+0100"}}', '{"dt": { "$date" : "1970-01-01T01:00:00.000+01:00"}}', '{"dt": { "$date" : "1970-01-01T01:00:00.000000+01:00"}}', '{"dt": { "$date" : "1970-01-01T01:00:00.000000+01"}}']:
            self.assertEqual(EPOCH_AWARE, json_util.loads(jsn, json_options=tz_aware_opts)['dt'])
            self.assertEqual(EPOCH_NAIVE, json_util.loads(jsn)['dt'])
        dtm = datetime.datetime(1, 1, 1, 1, 1, 1, 0, utc)
        jsn = '{"dt": {"$date": -62135593139000}}'
        self.assertEqual(dtm, json_util.loads(jsn, json_options=tz_aware_opts)['dt'])
        jsn = '{"dt": {"$date": {"$numberLong": "-62135593139000"}}}'
        self.assertEqual(dtm, json_util.loads(jsn, json_options=tz_aware_opts)['dt'])
        pre_epoch = {'dt': datetime.datetime(1, 1, 1, 1, 1, 1, 10000, utc)}
        post_epoch = {'dt': datetime.datetime(1972, 1, 1, 1, 1, 1, 10000, utc)}
        self.assertEqual('{"dt": {"$date": {"$numberLong": "-62135593138990"}}}', json_util.dumps(pre_epoch))
        self.assertEqual('{"dt": {"$date": "1972-01-01T01:01:01.010Z"}}', json_util.dumps(post_epoch))
        self.assertEqual('{"dt": {"$date": -62135593138990}}', json_util.dumps(pre_epoch, json_options=LEGACY_JSON_OPTIONS))
        self.assertEqual('{"dt": {"$date": 63075661010}}', json_util.dumps(post_epoch, json_options=LEGACY_JSON_OPTIONS))
        self.assertEqual('{"dt": {"$date": {"$numberLong": "-62135593138990"}}}', json_util.dumps(pre_epoch, json_options=STRICT_JSON_OPTIONS))
        self.assertEqual('{"dt": {"$date": "1972-01-01T01:01:01.010Z"}}', json_util.dumps(post_epoch, json_options=STRICT_JSON_OPTIONS))
        number_long_options = JSONOptions(datetime_representation=DatetimeRepresentation.NUMBERLONG, json_mode=JSONMode.LEGACY)
        self.assertEqual('{"dt": {"$date": {"$numberLong": "63075661010"}}}', json_util.dumps(post_epoch, json_options=number_long_options))
        self.assertEqual('{"dt": {"$date": {"$numberLong": "-62135593138990"}}}', json_util.dumps(pre_epoch, json_options=number_long_options))
        pre_epoch_naive = {'dt': datetime.datetime(1, 1, 1, 1, 1, 1, 10000)}
        post_epoch_naive = {'dt': datetime.datetime(1972, 1, 1, 1, 1, 1, 10000)}
        self.assertEqual('{"dt": {"$date": {"$numberLong": "-62135593138990"}}}', json_util.dumps(pre_epoch_naive, json_options=STRICT_JSON_OPTIONS))
        self.assertEqual('{"dt": {"$date": "1972-01-01T01:01:01.010Z"}}', json_util.dumps(post_epoch_naive, json_options=STRICT_JSON_OPTIONS))
        self.assertEqual(datetime.datetime(1972, 1, 1, 1, 1, 1, 10000, utc), json_util.loads('{"dt": {"$date": "1972-01-01T01:01:01.010+0000"}}', json_options=tz_aware_opts)['dt'])
        self.assertEqual(datetime.datetime(1972, 1, 1, 1, 1, 1, 10000, utc), json_util.loads('{"dt": {"$date": "1972-01-01T01:01:01.010+0000"}}', json_options=JSONOptions(tz_aware=True, tzinfo=utc))['dt'])
        self.assertEqual(datetime.datetime(1972, 1, 1, 1, 1, 1, 10000), json_util.loads('{"dt": {"$date": "1972-01-01T01:01:01.010+0000"}}', json_options=JSONOptions(tz_aware=False))['dt'])
        self.round_trip(pre_epoch_naive, json_options=JSONOptions(tz_aware=False))
        pacific = FixedOffset(-8 * 60, 'US/Pacific')
        aware_datetime = {'dt': datetime.datetime(2002, 10, 27, 6, 0, 0, 10000, pacific)}
        self.assertEqual('{"dt": {"$date": "2002-10-27T06:00:00.010-0800"}}', json_util.dumps(aware_datetime, json_options=STRICT_JSON_OPTIONS))
        self.round_trip(aware_datetime, json_options=JSONOptions(json_mode=JSONMode.LEGACY, tz_aware=True, tzinfo=pacific))
        self.round_trip(aware_datetime, json_options=JSONOptions(datetime_representation=DatetimeRepresentation.ISO8601, json_mode=JSONMode.LEGACY, tz_aware=True, tzinfo=pacific))

    def test_datetime_ms(self):
        if False:
            print('Hello World!')
        dat_min: dict[str, Any] = {'x': DatetimeMS(0)}
        dat_max: dict[str, Any] = {'x': DatetimeMS(_max_datetime_ms())}
        opts = JSONOptions(datetime_representation=DatetimeRepresentation.ISO8601)
        self.assertEqual(dat_min['x'].as_datetime(CodecOptions(tz_aware=False)), json_util.loads(json_util.dumps(dat_min))['x'])
        self.assertEqual(dat_max['x'].as_datetime(CodecOptions(tz_aware=False)), json_util.loads(json_util.dumps(dat_max))['x'])
        dat_min = {'x': DatetimeMS(-1)}
        dat_max = {'x': DatetimeMS(_max_datetime_ms() + 1)}
        self.assertEqual('{"x": {"$date": {"$numberLong": "-1"}}}', json_util.dumps(dat_min))
        self.assertEqual('{"x": {"$date": {"$numberLong": "' + str(int(dat_max['x'])) + '"}}}', json_util.dumps(dat_max))
        opts = JSONOptions(datetime_representation=DatetimeRepresentation.LEGACY, json_mode=JSONMode.LEGACY)
        self.assertEqual('{"x": {"$date": "-1"}}', json_util.dumps(dat_min, json_options=opts))
        self.assertEqual('{"x": {"$date": "' + str(int(dat_max['x'])) + '"}}', json_util.dumps(dat_max, json_options=opts))
        opts = JSONOptions(datetime_representation=DatetimeRepresentation.NUMBERLONG, json_mode=JSONMode.LEGACY)
        self.assertEqual('{"x": {"$date": {"$numberLong": "-1"}}}', json_util.dumps(dat_min, json_options=opts))
        self.assertEqual('{"x": {"$date": {"$numberLong": "' + str(int(dat_max['x'])) + '"}}}', json_util.dumps(dat_max, json_options=opts))
        dat_min = {'x': datetime.datetime.min}
        dat_max = {'x': DatetimeMS(_max_datetime_ms()).as_datetime(CodecOptions(tz_aware=False))}
        opts = JSONOptions(datetime_representation=DatetimeRepresentation.ISO8601, datetime_conversion=DatetimeConversion.DATETIME_MS)
        self.assertEqual(DatetimeMS(dat_min['x']), json_util.loads(json_util.dumps(dat_min), json_options=opts)['x'])
        self.assertEqual(DatetimeMS(dat_max['x']), json_util.loads(json_util.dumps(dat_max), json_options=opts)['x'])

    def test_regex_object_hook(self):
        if False:
            for i in range(10):
                print('nop')
        pat = 'a*b'
        json_re = '{"$regex": "%s", "$options": "u"}' % pat
        loaded = json_util.object_hook(json.loads(json_re))
        self.assertTrue(isinstance(loaded, Regex))
        self.assertEqual(pat, loaded.pattern)
        self.assertEqual(re.U, loaded.flags)

    def test_regex(self):
        if False:
            print('Hello World!')
        for regex_instance in (re.compile('a*b', re.IGNORECASE), Regex('a*b', re.IGNORECASE)):
            res = self.round_tripped({'r': regex_instance})['r']
            self.assertEqual('a*b', res.pattern)
            res = self.round_tripped({'r': Regex('a*b', re.IGNORECASE)})['r']
            self.assertEqual('a*b', res.pattern)
            self.assertEqual(re.IGNORECASE, res.flags)
        unicode_options = re.I | re.M | re.S | re.U | re.X
        regex = re.compile('a*b', unicode_options)
        res = self.round_tripped({'r': regex})['r']
        self.assertEqual(unicode_options, res.flags)
        res = json_util.loads('{"r": {"$regex": "a*b"}}')['r']
        self.assertEqual(0, res.flags)
        self.assertEqual(Regex('.*', 'ilm'), json_util.loads('{"r": {"$regex": ".*", "$options": "ilm"}}')['r'])
        self.assertEqual('{"$regularExpression": {"pattern": ".*", "options": "mx"}}', json_util.dumps(Regex('.*', re.M | re.X)))
        self.assertEqual('{"$regularExpression": {"pattern": ".*", "options": "mx"}}', json_util.dumps(re.compile(b'.*', re.M | re.X)))
        self.assertEqual('{"$regex": ".*", "$options": "mx"}', json_util.dumps(Regex('.*', re.M | re.X), json_options=LEGACY_JSON_OPTIONS))

    def test_regex_validation(self):
        if False:
            i = 10
            return i + 15
        non_str_types = [10, {}, []]
        docs = [{'$regex': i} for i in non_str_types]
        for doc in docs:
            self.assertEqual(doc, json_util.loads(json.dumps(doc)))
        doc = {'$regex': ''}
        self.assertIsInstance(json_util.loads(json.dumps(doc)), Regex)

    def test_minkey(self):
        if False:
            print('Hello World!')
        self.round_trip({'m': MinKey()})

    def test_maxkey(self):
        if False:
            while True:
                i = 10
        self.round_trip({'m': MaxKey()})

    def test_timestamp(self):
        if False:
            while True:
                i = 10
        dct = {'ts': Timestamp(4, 13)}
        res = json_util.dumps(dct, default=json_util.default)
        rtdct = json_util.loads(res)
        self.assertEqual(dct, rtdct)
        self.assertEqual('{"ts": {"$timestamp": {"t": 4, "i": 13}}}', res)

    def test_uuid_default(self):
        if False:
            i = 10
            return i + 15
        doc = {'uuid': uuid.UUID('f47ac10b-58cc-4372-a567-0e02b2c3d479')}
        with self.assertRaisesRegex(ValueError, 'cannot encode native uuid'):
            json_util.dumps(doc)
        legacy_jsn = '{"uuid": {"$uuid": "f47ac10b58cc4372a5670e02b2c3d479"}}'
        expected = {'uuid': Binary(b'\xf4z\xc1\x0bX\xccCr\xa5g\x0e\x02\xb2\xc3\xd4y', 4)}
        self.assertEqual(json_util.loads(legacy_jsn), expected)

    def test_uuid(self):
        if False:
            i = 10
            return i + 15
        doc = {'uuid': uuid.UUID('f47ac10b-58cc-4372-a567-0e02b2c3d479')}
        uuid_legacy_opts = LEGACY_JSON_OPTIONS.with_options(uuid_representation=UuidRepresentation.PYTHON_LEGACY)
        self.round_trip(doc, json_options=uuid_legacy_opts)
        self.assertEqual('{"uuid": {"$uuid": "f47ac10b58cc4372a5670e02b2c3d479"}}', json_util.dumps(doc, json_options=LEGACY_JSON_OPTIONS))
        self.assertEqual('{"uuid": {"$binary": "9HrBC1jMQ3KlZw4CssPUeQ==", "$type": "03"}}', json_util.dumps(doc, json_options=STRICT_JSON_OPTIONS.with_options(uuid_representation=UuidRepresentation.PYTHON_LEGACY)))
        self.assertEqual('{"uuid": {"$binary": "9HrBC1jMQ3KlZw4CssPUeQ==", "$type": "04"}}', json_util.dumps(doc, json_options=JSONOptions(strict_uuid=True, json_mode=JSONMode.LEGACY, uuid_representation=STANDARD)))
        self.assertEqual(doc, json_util.loads('{"uuid": {"$binary": "9HrBC1jMQ3KlZw4CssPUeQ==", "$type": "03"}}', json_options=uuid_legacy_opts))
        for uuid_representation in set(ALL_UUID_REPRESENTATIONS) - {UuidRepresentation.UNSPECIFIED}:
            options = JSONOptions(strict_uuid=True, json_mode=JSONMode.LEGACY, uuid_representation=uuid_representation)
            self.round_trip(doc, json_options=options)
            self.assertEqual(doc, json_util.loads('{"uuid": {"$binary": "9HrBC1jMQ3KlZw4CssPUeQ==", "$type": "04"}}', json_options=options))

    def test_uuid_uuid_rep_unspecified(self):
        if False:
            while True:
                i = 10
        _uuid = uuid.uuid4()
        options = JSONOptions(strict_uuid=True, json_mode=JSONMode.LEGACY, uuid_representation=UuidRepresentation.UNSPECIFIED)
        doc: dict[str, Any] = {'uuid': _uuid}
        with self.assertRaises(ValueError):
            json_util.dumps(doc, json_options=options)
        doc = {'uuid': Binary(_uuid.bytes, subtype=3)}
        ext_json_str = json_util.dumps(doc)
        self.assertEqual(doc, json_util.loads(ext_json_str, json_options=options))
        doc = {'uuid': Binary(_uuid.bytes, subtype=4)}
        ext_json_str = json_util.dumps(doc)
        self.assertEqual(doc, json_util.loads(ext_json_str, json_options=options))
        doc = {'uuid': Binary(_uuid.bytes, subtype=4)}
        ext_json_str = json_util.dumps({'uuid': _uuid}, json_options=LEGACY_JSON_OPTIONS)
        self.assertEqual(doc, json_util.loads(ext_json_str, json_options=options))

    def test_binary(self):
        if False:
            while True:
                i = 10
        bin_type_dict = {'bin': b'\x00\x01\x02\x03\x04'}
        md5_type_dict = {'md5': Binary(b' n7\x18\xaf\t/\xd1\xd1/\x80\xca\xe7q\xcc\xac', MD5_SUBTYPE)}
        custom_type_dict = {'custom': Binary(b'hello', USER_DEFINED_SUBTYPE)}
        self.round_trip(bin_type_dict)
        self.round_trip(md5_type_dict)
        self.round_trip(custom_type_dict)
        bin = json_util.loads('{"bin": {"$binary": "AAECAwQ=", "$type": "00"}}')['bin']
        self.assertEqual(type(bin), bytes)
        json_bin_dump = json_util.dumps(bin_type_dict, json_options=LEGACY_JSON_OPTIONS)
        self.assertIn('"$type": "00"', json_bin_dump)
        self.assertEqual(bin_type_dict, json_util.loads('{"bin": {"$type": 0, "$binary": "AAECAwQ="}}'))
        json_bin_dump = json_util.dumps(md5_type_dict, json_options=LEGACY_JSON_OPTIONS)
        self.assertEqual('{"md5": {"$binary": "IG43GK8JL9HRL4DK53HMrA==", "$type": "05"}}', json_bin_dump)
        self.assertEqual(md5_type_dict, json_util.loads('{"md5": {"$type": 5, "$binary": "IG43GK8JL9HRL4DK53HMrA=="}}'))
        json_bin_dump = json_util.dumps(custom_type_dict, json_options=LEGACY_JSON_OPTIONS)
        self.assertIn('"$type": "80"', json_bin_dump)
        self.assertEqual(custom_type_dict, json_util.loads('{"custom": {"$type": 128, "$binary": "aGVsbG8="}}'))
        self.assertEqual(128, json_util.loads('{"custom": {"$type": "ffffff80", "$binary": "aGVsbG8="}}')['custom'].subtype)
        self.assertEqual(255, json_util.loads('{"custom": {"$type": "ffffffff", "$binary": "aGVsbG8="}}')['custom'].subtype)

    def test_code(self):
        if False:
            return 10
        self.round_trip({'code': Code('function x() { return 1; }')})
        code = Code('return z', z=2)
        res = json_util.dumps(code)
        self.assertEqual(code, json_util.loads(res))
        self.assertEqual('{"$code": "return z", "$scope": {"z": 2}}', res)
        no_scope = Code('function() {}')
        self.assertEqual('{"$code": "function() {}"}', json_util.dumps(no_scope))

    def test_undefined(self):
        if False:
            print('Hello World!')
        jsn = '{"name": {"$undefined": true}}'
        self.assertIsNone(json_util.loads(jsn)['name'])

    def test_numberlong(self):
        if False:
            print('Hello World!')
        jsn = '{"weight": {"$numberLong": "65535"}}'
        self.assertEqual(json_util.loads(jsn)['weight'], Int64(65535))
        self.assertEqual(json_util.dumps({'weight': Int64(65535)}), '{"weight": 65535}')
        json_options = JSONOptions(strict_number_long=True, json_mode=JSONMode.LEGACY)
        self.assertEqual(json_util.dumps({'weight': Int64(65535)}, json_options=json_options), jsn)

    def test_loads_document_class(self):
        if False:
            return 10
        self.assertEqual({'foo': 'bar'}, json_util.loads('{"foo": "bar"}', json_options=JSONOptions(document_class=dict)))
        self.assertEqual(SON([('foo', 'bar'), ('b', 1)]), json_util.loads('{"foo": "bar", "b": 1}', json_options=JSONOptions(document_class=SON)))

class TestJsonUtilRoundtrip(IntegrationTest):

    def test_cursor(self):
        if False:
            return 10
        db = self.db
        db.drop_collection('test')
        docs: List[MutableMapping[str, Any]] = [{'foo': [1, 2]}, {'bar': {'hello': 'world'}}, {'code': Code('function x() { return 1; }')}, {'bin': Binary(b'\x00\x01\x02\x03\x04', USER_DEFINED_SUBTYPE)}, {'dbref': {'_ref': DBRef('simple', ObjectId('509b8db456c02c5ab7e63c34'))}}]
        db.test.insert_many(docs)
        reloaded_docs = json_util.loads(json_util.dumps(db.test.find()))
        for doc in docs:
            self.assertTrue(doc in reloaded_docs)
if __name__ == '__main__':
    unittest.main()