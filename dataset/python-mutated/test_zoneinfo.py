from __future__ import annotations
import base64
import contextlib
import dataclasses
import importlib.metadata
import io
import json
import os
import pathlib
import pickle
import re
import shutil
import struct
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from functools import cached_property
from . import _support as test_support
from ._support import OS_ENV_LOCK, TZPATH_TEST_LOCK, ZoneInfoTestBase
from test.support.import_helper import import_module
lzma = import_module('lzma')
(py_zoneinfo, c_zoneinfo) = test_support.get_modules()
try:
    importlib.metadata.metadata('tzdata')
    HAS_TZDATA_PKG = True
except importlib.metadata.PackageNotFoundError:
    HAS_TZDATA_PKG = False
ZONEINFO_DATA = None
ZONEINFO_DATA_V1 = None
TEMP_DIR = None
DATA_DIR = pathlib.Path(__file__).parent / 'data'
ZONEINFO_JSON = DATA_DIR / 'zoneinfo_data.json'
ZERO = timedelta(0)
ONE_H = timedelta(hours=1)

def setUpModule():
    if False:
        for i in range(10):
            print('nop')
    global TEMP_DIR
    global ZONEINFO_DATA
    global ZONEINFO_DATA_V1
    TEMP_DIR = pathlib.Path(tempfile.mkdtemp(prefix='zoneinfo'))
    ZONEINFO_DATA = ZoneInfoData(ZONEINFO_JSON, TEMP_DIR / 'v2')
    ZONEINFO_DATA_V1 = ZoneInfoData(ZONEINFO_JSON, TEMP_DIR / 'v1', v1=True)

def tearDownModule():
    if False:
        for i in range(10):
            print('nop')
    shutil.rmtree(TEMP_DIR)

class TzPathUserMixin:
    """
    Adds a setUp() and tearDown() to make TZPATH manipulations thread-safe.

    Any tests that require manipulation of the TZPATH global are necessarily
    thread unsafe, so we will acquire a lock and reset the TZPATH variable
    to the default state before each test and release the lock after the test
    is through.
    """

    @property
    def tzpath(self):
        if False:
            return 10
        return None

    @property
    def block_tzdata(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.tzpath_context(self.tzpath, block_tzdata=self.block_tzdata, lock=TZPATH_TEST_LOCK))
            self.addCleanup(stack.pop_all().close)
        super().setUp()

class DatetimeSubclassMixin:
    """
    Replaces all ZoneTransition transition dates with a datetime subclass.
    """

    class DatetimeSubclass(datetime):

        @classmethod
        def from_datetime(cls, dt):
            if False:
                print('Hello World!')
            return cls(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, tzinfo=dt.tzinfo, fold=dt.fold)

    def load_transition_examples(self, key):
        if False:
            i = 10
            return i + 15
        transition_examples = super().load_transition_examples(key)
        for zt in transition_examples:
            dt = zt.transition
            new_dt = self.DatetimeSubclass.from_datetime(dt)
            new_zt = dataclasses.replace(zt, transition=new_dt)
            yield new_zt

class ZoneInfoTest(TzPathUserMixin, ZoneInfoTestBase):
    module = py_zoneinfo
    class_name = 'ZoneInfo'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.klass.clear_cache()

    @property
    def zoneinfo_data(self):
        if False:
            i = 10
            return i + 15
        return ZONEINFO_DATA

    @property
    def tzpath(self):
        if False:
            print('Hello World!')
        return [self.zoneinfo_data.tzpath]

    def zone_from_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.klass(key)

    def zones(self):
        if False:
            for i in range(10):
                print('nop')
        return ZoneDumpData.transition_keys()

    def fixed_offset_zones(self):
        if False:
            for i in range(10):
                print('nop')
        return ZoneDumpData.fixed_offset_zones()

    def load_transition_examples(self, key):
        if False:
            i = 10
            return i + 15
        return ZoneDumpData.load_transition_examples(key)

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        for key in self.zones():
            with self.subTest(key):
                zi = self.zone_from_key(key)
                self.assertEqual(str(zi), key)
        file_key = self.zoneinfo_data.keys[0]
        file_path = self.zoneinfo_data.path_from_key(file_key)
        with open(file_path, 'rb') as f:
            with self.subTest(test_name='Repr test', path=file_path):
                zi_ff = self.klass.from_file(f)
                self.assertEqual(str(zi_ff), repr(zi_ff))

    def test_repr(self):
        if False:
            print('Hello World!')
        key = next(iter(self.zones()))
        zi = self.klass(key)
        class_name = self.class_name
        with self.subTest(name='from key'):
            self.assertRegex(repr(zi), class_name)
        file_key = self.zoneinfo_data.keys[0]
        file_path = self.zoneinfo_data.path_from_key(file_key)
        with open(file_path, 'rb') as f:
            zi_ff = self.klass.from_file(f, key=file_key)
        with self.subTest(name='from file with key'):
            self.assertRegex(repr(zi_ff), class_name)
        with open(file_path, 'rb') as f:
            zi_ff_nk = self.klass.from_file(f)
        with self.subTest(name='from file without key'):
            self.assertRegex(repr(zi_ff_nk), class_name)

    def test_key_attribute(self):
        if False:
            print('Hello World!')
        key = next(iter(self.zones()))

        def from_file_nokey(key):
            if False:
                return 10
            with open(self.zoneinfo_data.path_from_key(key), 'rb') as f:
                return self.klass.from_file(f)
        constructors = (('Primary constructor', self.klass, key), ('no_cache', self.klass.no_cache, key), ('from_file', from_file_nokey, None))
        for (msg, constructor, expected) in constructors:
            zi = constructor(key)
            with self.subTest(msg):
                self.assertEqual(zi.key, expected)
            with self.subTest(f'{msg}: readonly'):
                with self.assertRaises(AttributeError):
                    zi.key = 'Some/Value'

    def test_bad_keys(self):
        if False:
            i = 10
            return i + 15
        bad_keys = ['Eurasia/Badzone', 'BZQ', 'America.Los_Angeles', '🇨🇦', 'America/New\ud800York']
        for bad_key in bad_keys:
            with self.assertRaises(self.module.ZoneInfoNotFoundError):
                self.klass(bad_key)

    def test_bad_keys_paths(self):
        if False:
            i = 10
            return i + 15
        bad_keys = ['/America/Los_Angeles', 'America/Los_Angeles/', '../zoneinfo/America/Los_Angeles', 'America/../America/Los_Angeles', 'America/./Los_Angeles']
        for bad_key in bad_keys:
            with self.assertRaises(ValueError):
                self.klass(bad_key)

    def test_bad_zones(self):
        if False:
            for i in range(10):
                print('nop')
        bad_zones = [b'', b'AAAA3' + b' ' * 15]
        for bad_zone in bad_zones:
            fobj = io.BytesIO(bad_zone)
            with self.assertRaises(ValueError):
                self.klass.from_file(fobj)

    def test_fromutc_errors(self):
        if False:
            i = 10
            return i + 15
        key = next(iter(self.zones()))
        zone = self.zone_from_key(key)
        bad_values = [(datetime(2019, 1, 1, tzinfo=timezone.utc), ValueError), (datetime(2019, 1, 1), ValueError), (date(2019, 1, 1), TypeError), (time(0), TypeError), (0, TypeError), ('2019-01-01', TypeError)]
        for (val, exc_type) in bad_values:
            with self.subTest(val=val):
                with self.assertRaises(exc_type):
                    zone.fromutc(val)

    def test_utc(self):
        if False:
            return 10
        zi = self.klass('UTC')
        dt = datetime(2020, 1, 1, tzinfo=zi)
        self.assertEqual(dt.utcoffset(), ZERO)
        self.assertEqual(dt.dst(), ZERO)
        self.assertEqual(dt.tzname(), 'UTC')

    def test_unambiguous(self):
        if False:
            for i in range(10):
                print('nop')
        test_cases = []
        for key in self.zones():
            for zone_transition in self.load_transition_examples(key):
                test_cases.append((key, zone_transition.transition - timedelta(days=2), zone_transition.offset_before))
                test_cases.append((key, zone_transition.transition + timedelta(days=2), zone_transition.offset_after))
        for (key, dt, offset) in test_cases:
            with self.subTest(key=key, dt=dt, offset=offset):
                tzi = self.zone_from_key(key)
                dt = dt.replace(tzinfo=tzi)
                self.assertEqual(dt.tzname(), offset.tzname, dt)
                self.assertEqual(dt.utcoffset(), offset.utcoffset, dt)
                self.assertEqual(dt.dst(), offset.dst, dt)

    def test_folds_and_gaps(self):
        if False:
            while True:
                i = 10
        test_cases = []
        for key in self.zones():
            tests = {'folds': [], 'gaps': []}
            for zt in self.load_transition_examples(key):
                if zt.fold:
                    test_group = tests['folds']
                elif zt.gap:
                    test_group = tests['gaps']
                else:
                    no_peephole_opt = None
                    continue
                dt = zt.anomaly_start - timedelta(seconds=1)
                test_group.append((dt, 0, zt.offset_before))
                test_group.append((dt, 1, zt.offset_before))
                dt = zt.anomaly_start
                test_group.append((dt, 0, zt.offset_before))
                test_group.append((dt, 1, zt.offset_after))
                dt = zt.anomaly_start + timedelta(seconds=1)
                test_group.append((dt, 0, zt.offset_before))
                test_group.append((dt, 1, zt.offset_after))
                dt = zt.anomaly_end - timedelta(seconds=1)
                test_group.append((dt, 0, zt.offset_before))
                test_group.append((dt, 1, zt.offset_after))
                dt = zt.anomaly_end
                test_group.append((dt, 0, zt.offset_after))
                test_group.append((dt, 1, zt.offset_after))
                dt = zt.anomaly_end + timedelta(seconds=1)
                test_group.append((dt, 0, zt.offset_after))
                test_group.append((dt, 1, zt.offset_after))
            for (grp, test_group) in tests.items():
                test_cases.append(((key, grp), test_group))
        for ((key, grp), tests) in test_cases:
            with self.subTest(key=key, grp=grp):
                tzi = self.zone_from_key(key)
                for (dt, fold, offset) in tests:
                    dt = dt.replace(fold=fold, tzinfo=tzi)
                    self.assertEqual(dt.tzname(), offset.tzname, dt)
                    self.assertEqual(dt.utcoffset(), offset.utcoffset, dt)
                    self.assertEqual(dt.dst(), offset.dst, dt)

    def test_folds_from_utc(self):
        if False:
            while True:
                i = 10
        for key in self.zones():
            zi = self.zone_from_key(key)
            with self.subTest(key=key):
                for zt in self.load_transition_examples(key):
                    if not zt.fold:
                        continue
                    dt_utc = zt.transition_utc
                    dt_before_utc = dt_utc - timedelta(seconds=1)
                    dt_after_utc = dt_utc + timedelta(seconds=1)
                    dt_before = dt_before_utc.astimezone(zi)
                    self.assertEqual(dt_before.fold, 0, (dt_before, dt_utc))
                    dt_after = dt_after_utc.astimezone(zi)
                    self.assertEqual(dt_after.fold, 1, (dt_after, dt_utc))

    def test_time_variable_offset(self):
        if False:
            return 10
        for key in self.zones():
            zi = self.zone_from_key(key)
            t = time(11, 15, 1, 34471, tzinfo=zi)
            with self.subTest(key=key):
                self.assertIs(t.tzname(), None)
                self.assertIs(t.utcoffset(), None)
                self.assertIs(t.dst(), None)

    def test_time_fixed_offset(self):
        if False:
            while True:
                i = 10
        for (key, offset) in self.fixed_offset_zones():
            zi = self.zone_from_key(key)
            t = time(11, 15, 1, 34471, tzinfo=zi)
            with self.subTest(key=key):
                self.assertEqual(t.tzname(), offset.tzname)
                self.assertEqual(t.utcoffset(), offset.utcoffset)
                self.assertEqual(t.dst(), offset.dst)

class CZoneInfoTest(ZoneInfoTest):
    module = c_zoneinfo

    def test_fold_mutate(self):
        if False:
            i = 10
            return i + 15
        "Test that fold isn't mutated when no change is necessary.\n\n        The underlying C API is capable of mutating datetime objects, and\n        may rely on the fact that addition of a datetime object returns a\n        new datetime; this test ensures that the input datetime to fromutc\n        is not mutated.\n        "

        def to_subclass(dt):
            if False:
                return 10

            class SameAddSubclass(type(dt)):

                def __add__(self, other):
                    if False:
                        return 10
                    if other == timedelta(0):
                        return self
                    return super().__add__(other)
            return SameAddSubclass(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, fold=dt.fold, tzinfo=dt.tzinfo)
        subclass = [False, True]
        key = 'Europe/London'
        zi = self.zone_from_key(key)
        for zt in self.load_transition_examples(key):
            if zt.fold and zt.offset_after.utcoffset == ZERO:
                example = zt.transition_utc.replace(tzinfo=zi)
                break
        for subclass in [False, True]:
            if subclass:
                dt = to_subclass(example)
            else:
                dt = example
            with self.subTest(subclass=subclass):
                dt_fromutc = zi.fromutc(dt)
                self.assertEqual(dt_fromutc.fold, 1)
                self.assertEqual(dt.fold, 0)

class ZoneInfoDatetimeSubclassTest(DatetimeSubclassMixin, ZoneInfoTest):
    pass

class CZoneInfoDatetimeSubclassTest(DatetimeSubclassMixin, CZoneInfoTest):
    pass

class ZoneInfoSubclassTest(ZoneInfoTest):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()

        class ZISubclass(cls.klass):
            pass
        cls.class_name = 'ZISubclass'
        cls.parent_klass = cls.klass
        cls.klass = ZISubclass

    def test_subclass_own_cache(self):
        if False:
            print('Hello World!')
        base_obj = self.parent_klass('Europe/London')
        sub_obj = self.klass('Europe/London')
        self.assertIsNot(base_obj, sub_obj)
        self.assertIsInstance(base_obj, self.parent_klass)
        self.assertIsInstance(sub_obj, self.klass)

class CZoneInfoSubclassTest(ZoneInfoSubclassTest):
    module = c_zoneinfo

class ZoneInfoV1Test(ZoneInfoTest):

    @property
    def zoneinfo_data(self):
        if False:
            while True:
                i = 10
        return ZONEINFO_DATA_V1

    def load_transition_examples(self, key):
        if False:
            print('Hello World!')
        epoch = datetime(1970, 1, 1)
        max_offset_32 = timedelta(seconds=2 ** 31)
        min_dt = epoch - max_offset_32
        max_dt = epoch + max_offset_32
        for zt in ZoneDumpData.load_transition_examples(key):
            if min_dt <= zt.transition <= max_dt:
                yield zt

class CZoneInfoV1Test(ZoneInfoV1Test):
    module = c_zoneinfo

@unittest.skipIf(not HAS_TZDATA_PKG, 'Skipping tzdata-specific tests: tzdata not installed')
class TZDataTests(ZoneInfoTest):
    """
    Runs all the ZoneInfoTest tests, but against the tzdata package

    NOTE: The ZoneDumpData has frozen test data, but tzdata will update, so
    some of the tests (particularly those related to the far future) may break
    in the event that the time zone policies in the relevant time zones change.
    """

    @property
    def tzpath(self):
        if False:
            while True:
                i = 10
        return []

    @property
    def block_tzdata(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def zone_from_key(self, key):
        if False:
            return 10
        return self.klass(key=key)

@unittest.skipIf(not HAS_TZDATA_PKG, 'Skipping tzdata-specific tests: tzdata not installed')
class CTZDataTests(TZDataTests):
    module = c_zoneinfo

class WeirdZoneTest(ZoneInfoTestBase):
    module = py_zoneinfo

    def test_one_transition(self):
        if False:
            return 10
        LMT = ZoneOffset('LMT', -timedelta(hours=6, minutes=31, seconds=2))
        STD = ZoneOffset('STD', -timedelta(hours=6))
        transitions = [ZoneTransition(datetime(1883, 6, 9, 14), LMT, STD)]
        after = 'STD6'
        zf = self.construct_zone(transitions, after)
        zi = self.klass.from_file(zf)
        dt0 = datetime(1883, 6, 9, 1, tzinfo=zi)
        dt1 = datetime(1883, 6, 10, 1, tzinfo=zi)
        for (dt, offset) in [(dt0, LMT), (dt1, STD)]:
            with self.subTest(name='local', dt=dt):
                self.assertEqual(dt.tzname(), offset.tzname)
                self.assertEqual(dt.utcoffset(), offset.utcoffset)
                self.assertEqual(dt.dst(), offset.dst)
        dts = [(datetime(1883, 6, 9, 1, tzinfo=zi), datetime(1883, 6, 9, 7, 31, 2, tzinfo=timezone.utc)), (datetime(2010, 4, 1, 12, tzinfo=zi), datetime(2010, 4, 1, 18, tzinfo=timezone.utc))]
        for (dt_local, dt_utc) in dts:
            with self.subTest(name='fromutc', dt=dt_local):
                dt_actual = dt_utc.astimezone(zi)
                self.assertEqual(dt_actual, dt_local)
                dt_utc_actual = dt_local.astimezone(timezone.utc)
                self.assertEqual(dt_utc_actual, dt_utc)

    def test_one_zone_dst(self):
        if False:
            for i in range(10):
                print('nop')
        DST = ZoneOffset('DST', ONE_H, ONE_H)
        transitions = [ZoneTransition(datetime(1970, 1, 1), DST, DST)]
        after = 'STD0DST-1,0/0,J365/25'
        zf = self.construct_zone(transitions, after)
        zi = self.klass.from_file(zf)
        dts = [datetime(1900, 3, 1), datetime(1965, 9, 12), datetime(1970, 1, 1), datetime(2010, 11, 3), datetime(2040, 1, 1)]
        for dt in dts:
            dt = dt.replace(tzinfo=zi)
            with self.subTest(dt=dt):
                self.assertEqual(dt.tzname(), DST.tzname)
                self.assertEqual(dt.utcoffset(), DST.utcoffset)
                self.assertEqual(dt.dst(), DST.dst)

    def test_no_tz_str(self):
        if False:
            return 10
        STD = ZoneOffset('STD', ONE_H, ZERO)
        DST = ZoneOffset('DST', 2 * ONE_H, ONE_H)
        transitions = []
        for year in range(1996, 2000):
            transitions.append(ZoneTransition(datetime(year, 3, 1, 2), STD, DST))
            transitions.append(ZoneTransition(datetime(year, 11, 1, 2), DST, STD))
        after = ''
        zf = self.construct_zone(transitions, after)
        zi = self.klass.from_file(zf)
        cases = [(datetime(1995, 1, 1), STD), (datetime(1996, 4, 1), DST), (datetime(1996, 11, 2), STD), (datetime(2001, 1, 1), STD)]
        for (dt, offset) in cases:
            dt = dt.replace(tzinfo=zi)
            with self.subTest(dt=dt):
                self.assertEqual(dt.tzname(), offset.tzname)
                self.assertEqual(dt.utcoffset(), offset.utcoffset)
                self.assertEqual(dt.dst(), offset.dst)
        t = time(0, tzinfo=zi)
        with self.subTest('Testing datetime.time'):
            self.assertIs(t.tzname(), None)
            self.assertIs(t.utcoffset(), None)
            self.assertIs(t.dst(), None)

    def test_tz_before_only(self):
        if False:
            while True:
                i = 10
        offsets = [ZoneOffset('STD', ZERO, ZERO), ZoneOffset('DST', ONE_H, ONE_H)]
        for offset in offsets:
            transitions = [ZoneTransition(None, offset, offset)]
            after = ''
            zf = self.construct_zone(transitions, after)
            zi = self.klass.from_file(zf)
            dts = [datetime(1900, 1, 1), datetime(1970, 1, 1), datetime(2000, 1, 1)]
            for dt in dts:
                dt = dt.replace(tzinfo=zi)
                with self.subTest(offset=offset, dt=dt):
                    self.assertEqual(dt.tzname(), offset.tzname)
                    self.assertEqual(dt.utcoffset(), offset.utcoffset)
                    self.assertEqual(dt.dst(), offset.dst)

    def test_empty_zone(self):
        if False:
            i = 10
            return i + 15
        zf = self.construct_zone([], '')
        with self.assertRaises(ValueError):
            self.klass.from_file(zf)

    def test_zone_very_large_timestamp(self):
        if False:
            while True:
                i = 10
        'Test when a transition is in the far past or future.\n\n        Particularly, this is a concern if something:\n\n            1. Attempts to call ``datetime.timestamp`` for a datetime outside\n               of ``[datetime.min, datetime.max]``.\n            2. Attempts to construct a timedelta outside of\n               ``[timedelta.min, timedelta.max]``.\n\n        This actually occurs "in the wild", as some time zones on Ubuntu (at\n        least as of 2020) have an initial transition added at ``-2**58``.\n        '
        LMT = ZoneOffset('LMT', timedelta(seconds=-968))
        GMT = ZoneOffset('GMT', ZERO)
        transitions = [(-(1 << 62), LMT, LMT), ZoneTransition(datetime(1912, 1, 1), LMT, GMT), (1 << 62, GMT, GMT)]
        after = 'GMT0'
        zf = self.construct_zone(transitions, after)
        zi = self.klass.from_file(zf, key='Africa/Abidjan')
        offset_cases = [(datetime.min, LMT), (datetime.max, GMT), (datetime(1911, 12, 31), LMT), (datetime(1912, 1, 2), GMT)]
        for (dt_naive, offset) in offset_cases:
            dt = dt_naive.replace(tzinfo=zi)
            with self.subTest(name='offset', dt=dt, offset=offset):
                self.assertEqual(dt.tzname(), offset.tzname)
                self.assertEqual(dt.utcoffset(), offset.utcoffset)
                self.assertEqual(dt.dst(), offset.dst)
        utc_cases = [(datetime.min, datetime.min + timedelta(seconds=968)), (datetime(1898, 12, 31, 23, 43, 52), datetime(1899, 1, 1)), (datetime(1911, 12, 31, 23, 59, 59, 999999), datetime(1912, 1, 1, 0, 16, 7, 999999)), (datetime(1912, 1, 1, 0, 16, 8), datetime(1912, 1, 1, 0, 16, 8)), (datetime(1970, 1, 1), datetime(1970, 1, 1)), (datetime.max, datetime.max)]
        for (naive_dt, naive_dt_utc) in utc_cases:
            dt = naive_dt.replace(tzinfo=zi)
            dt_utc = naive_dt_utc.replace(tzinfo=timezone.utc)
            self.assertEqual(dt_utc.astimezone(zi), dt)
            self.assertEqual(dt, dt_utc)

    def test_fixed_offset_phantom_transition(self):
        if False:
            for i in range(10):
                print('nop')
        UTC = ZoneOffset('UTC', ZERO, ZERO)
        transitions = [ZoneTransition(datetime(1970, 1, 1), UTC, UTC)]
        after = 'UTC0'
        zf = self.construct_zone(transitions, after)
        zi = self.klass.from_file(zf, key='UTC')
        dt = datetime(2020, 1, 1, tzinfo=zi)
        with self.subTest('datetime.datetime'):
            self.assertEqual(dt.tzname(), UTC.tzname)
            self.assertEqual(dt.utcoffset(), UTC.utcoffset)
            self.assertEqual(dt.dst(), UTC.dst)
        t = time(0, tzinfo=zi)
        with self.subTest('datetime.time'):
            self.assertEqual(t.tzname(), UTC.tzname)
            self.assertEqual(t.utcoffset(), UTC.utcoffset)
            self.assertEqual(t.dst(), UTC.dst)

    def construct_zone(self, transitions, after=None, version=3):
        if False:
            print('Hello World!')
        isutc = []
        isstd = []
        leap_seconds = []
        offset_lists = [[], []]
        trans_times_lists = [[], []]
        trans_idx_lists = [[], []]
        v1_range = (-2 ** 31, 2 ** 31)
        v2_range = (-2 ** 63, 2 ** 63)
        ranges = [v1_range, v2_range]

        def zt_as_tuple(zt):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(zt, tuple):
                return zt
            if zt.transition:
                trans_time = int(zt.transition_utc.timestamp())
            else:
                trans_time = None
            return (trans_time, zt.offset_before, zt.offset_after)
        transitions = sorted(map(zt_as_tuple, transitions), key=lambda x: x[0])
        for zt in transitions:
            (trans_time, offset_before, offset_after) = zt
            for (v, (dt_min, dt_max)) in enumerate(ranges):
                offsets = offset_lists[v]
                trans_times = trans_times_lists[v]
                trans_idx = trans_idx_lists[v]
                if trans_time is not None and (not dt_min <= trans_time <= dt_max):
                    continue
                if offset_before not in offsets:
                    offsets.append(offset_before)
                if offset_after not in offsets:
                    offsets.append(offset_after)
                if trans_time is not None:
                    trans_times.append(trans_time)
                    trans_idx.append(offsets.index(offset_after))
        isutcnt = len(isutc)
        isstdcnt = len(isstd)
        leapcnt = len(leap_seconds)
        zonefile = io.BytesIO()
        time_types = ('l', 'q')
        for v in range(min((version, 2))):
            offsets = offset_lists[v]
            trans_times = trans_times_lists[v]
            trans_idx = trans_idx_lists[v]
            time_type = time_types[v]
            abbrstr = bytearray()
            ttinfos = []
            for offset in offsets:
                utcoff = int(offset.utcoffset.total_seconds())
                isdst = bool(offset.dst)
                abbrind = len(abbrstr)
                ttinfos.append((utcoff, isdst, abbrind))
                abbrstr += offset.tzname.encode('ascii') + b'\x00'
            abbrstr = bytes(abbrstr)
            typecnt = len(offsets)
            timecnt = len(trans_times)
            charcnt = len(abbrstr)
            zonefile.write(b'TZif')
            zonefile.write(b'%d' % version)
            zonefile.write(b' ' * 15)
            zonefile.write(struct.pack('>6l', isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt))
            zonefile.write(struct.pack(f'>{timecnt}{time_type}', *trans_times))
            zonefile.write(struct.pack(f'>{timecnt}B', *trans_idx))
            for ttinfo in ttinfos:
                zonefile.write(struct.pack('>lbb', *ttinfo))
            zonefile.write(bytes(abbrstr))
            zonefile.write(struct.pack(f'{isutcnt}b', *isutc))
            zonefile.write(struct.pack(f'{isstdcnt}b', *isstd))
            zonefile.write(struct.pack(f'>{leapcnt}l', *leap_seconds))
            if v > 0:
                zonefile.write(b'\n')
                zonefile.write(after.encode('ascii'))
                zonefile.write(b'\n')
        zonefile.seek(0)
        return zonefile

class CWeirdZoneTest(WeirdZoneTest):
    module = c_zoneinfo

class TZStrTest(ZoneInfoTestBase):
    module = py_zoneinfo
    NORMAL = 0
    FOLD = 1
    GAP = 2

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls._populate_test_cases()
        cls.populate_tzstr_header()

    @classmethod
    def populate_tzstr_header(cls):
        if False:
            return 10
        out = bytearray()
        for _ in range(2):
            out += b'TZif'
            out += b'3'
            out += b' ' * 15
            out += struct.pack('>6l', 0, 0, 0, 0, 0, 0)
        cls._tzif_header = bytes(out)

    def zone_from_tzstr(self, tzstr):
        if False:
            while True:
                i = 10
        'Creates a zoneinfo file following a POSIX rule.'
        zonefile = io.BytesIO(self._tzif_header)
        zonefile.seek(0, 2)
        zonefile.write(b'\n')
        zonefile.write(tzstr.encode('ascii'))
        zonefile.write(b'\n')
        zonefile.seek(0)
        return self.klass.from_file(zonefile, key=tzstr)

    def test_tzstr_localized(self):
        if False:
            for i in range(10):
                print('nop')
        for (tzstr, cases) in self.test_cases.items():
            with self.subTest(tzstr=tzstr):
                zi = self.zone_from_tzstr(tzstr)
            for (dt_naive, offset, _) in cases:
                dt = dt_naive.replace(tzinfo=zi)
                with self.subTest(tzstr=tzstr, dt=dt, offset=offset):
                    self.assertEqual(dt.tzname(), offset.tzname)
                    self.assertEqual(dt.utcoffset(), offset.utcoffset)
                    self.assertEqual(dt.dst(), offset.dst)

    def test_tzstr_from_utc(self):
        if False:
            while True:
                i = 10
        for (tzstr, cases) in self.test_cases.items():
            with self.subTest(tzstr=tzstr):
                zi = self.zone_from_tzstr(tzstr)
            for (dt_naive, offset, dt_type) in cases:
                if dt_type == self.GAP:
                    continue
                dt_utc = (dt_naive - offset.utcoffset).replace(tzinfo=timezone.utc)
                dt_act = dt_utc.astimezone(zi)
                dt_exp = dt_naive.replace(tzinfo=zi)
                self.assertEqual(dt_act, dt_exp)
                if dt_type == self.FOLD:
                    self.assertEqual(dt_act.fold, dt_naive.fold, dt_naive)
                else:
                    self.assertEqual(dt_act.fold, 0)
                dt_act = dt_exp.astimezone(timezone.utc)
                self.assertEqual(dt_act, dt_utc)

    def test_invalid_tzstr(self):
        if False:
            while True:
                i = 10
        invalid_tzstrs = ['PST8PDT', '+11', 'GMT,M3.2.0/2,M11.1.0/3', 'GMT0+11,M3.2.0/2,M11.1.0/3', 'PST8PDT,M3.2.0/2', 'STD+25', 'STD-25', 'STD+374', 'STD+374DST,M3.2.0/2,M11.1.0/3', 'STD+23DST+25,M3.2.0/2,M11.1.0/3', 'STD-23DST-25,M3.2.0/2,M11.1.0/3', 'AAA4BBB,M1443339,M11.1.0/3', 'AAA4BBB,M3.2.0/2,0349309483959c', 'AAA4BBB,M13.1.1/2,M1.1.1/2', 'AAA4BBB,M1.1.1/2,M13.1.1/2', 'AAA4BBB,M0.1.1/2,M1.1.1/2', 'AAA4BBB,M1.1.1/2,M0.1.1/2', 'AAA4BBB,M1.6.1/2,M1.1.1/2', 'AAA4BBB,M1.1.1/2,M1.6.1/2', 'AAA4BBB,M1.1.7/2,M2.1.1/2', 'AAA4BBB,M1.1.1/2,M2.1.7/2', 'AAA4BBB,-1/2,20/2', 'AAA4BBB,1/2,-1/2', 'AAA4BBB,367,20/2', 'AAA4BBB,1/2,367/2', 'AAA4BBB,J0/2,J20/2', 'AAA4BBB,J20/2,J366/2']
        for invalid_tzstr in invalid_tzstrs:
            with self.subTest(tzstr=invalid_tzstr):
                tzstr_regex = re.escape(invalid_tzstr)
                with self.assertRaisesRegex(ValueError, tzstr_regex):
                    self.zone_from_tzstr(invalid_tzstr)

    @classmethod
    def _populate_test_cases(cls):
        if False:
            for i in range(10):
                print('nop')

        def call(f):
            if False:
                for i in range(10):
                    print('nop')
            'Decorator to call the addition methods.\n\n            This will call a function which adds at least one new entry into\n            the `cases` dictionary. The decorator will also assert that\n            something was added to the dictionary.\n            '
            prev_len = len(cases)
            f()
            assert len(cases) > prev_len, 'Function did not add a test case!'
        NORMAL = cls.NORMAL
        FOLD = cls.FOLD
        GAP = cls.GAP
        cases = {}

        @call
        def _add():
            if False:
                print('Hello World!')
            tzstr = 'EST5EDT,M3.2.0/4:00,M11.1.0/3:00'
            EST = ZoneOffset('EST', timedelta(hours=-5), ZERO)
            EDT = ZoneOffset('EDT', timedelta(hours=-4), ONE_H)
            cases[tzstr] = ((datetime(2019, 3, 9), EST, NORMAL), (datetime(2019, 3, 10, 3, 59), EST, NORMAL), (datetime(2019, 3, 10, 4, 0, fold=0), EST, GAP), (datetime(2019, 3, 10, 4, 0, fold=1), EDT, GAP), (datetime(2019, 3, 10, 4, 1, fold=0), EST, GAP), (datetime(2019, 3, 10, 4, 1, fold=1), EDT, GAP), (datetime(2019, 11, 2), EDT, NORMAL), (datetime(2019, 11, 3, 1, 59, fold=1), EDT, NORMAL), (datetime(2019, 11, 3, 2, 0, fold=0), EDT, FOLD), (datetime(2019, 11, 3, 2, 0, fold=1), EST, FOLD), (datetime(2020, 3, 8, 3, 59), EST, NORMAL), (datetime(2020, 3, 8, 4, 0, fold=0), EST, GAP), (datetime(2020, 3, 8, 4, 0, fold=1), EDT, GAP), (datetime(2020, 11, 1, 1, 59, fold=1), EDT, NORMAL), (datetime(2020, 11, 1, 2, 0, fold=0), EDT, FOLD), (datetime(2020, 11, 1, 2, 0, fold=1), EST, FOLD))

        @call
        def _add():
            if False:
                return 10
            tzstr = 'GMT0BST-1,M3.5.0/1:00,M10.5.0/2:00'
            GMT = ZoneOffset('GMT', ZERO, ZERO)
            BST = ZoneOffset('BST', ONE_H, ONE_H)
            cases[tzstr] = ((datetime(2019, 3, 30), GMT, NORMAL), (datetime(2019, 3, 31, 0, 59), GMT, NORMAL), (datetime(2019, 3, 31, 2, 0), BST, NORMAL), (datetime(2019, 10, 26), BST, NORMAL), (datetime(2019, 10, 27, 0, 59, fold=1), BST, NORMAL), (datetime(2019, 10, 27, 1, 0, fold=0), BST, GAP), (datetime(2019, 10, 27, 2, 0, fold=1), GMT, GAP), (datetime(2020, 3, 29, 0, 59), GMT, NORMAL), (datetime(2020, 3, 29, 2, 0), BST, NORMAL), (datetime(2020, 10, 25, 0, 59, fold=1), BST, NORMAL), (datetime(2020, 10, 25, 1, 0, fold=0), BST, FOLD), (datetime(2020, 10, 25, 2, 0, fold=1), GMT, NORMAL))

        @call
        def _add():
            if False:
                return 10
            tzstr = 'AEST-10AEDT,M10.1.0/2,M4.1.0/3'
            AEST = ZoneOffset('AEST', timedelta(hours=10), ZERO)
            AEDT = ZoneOffset('AEDT', timedelta(hours=11), ONE_H)
            cases[tzstr] = ((datetime(2019, 4, 6), AEDT, NORMAL), (datetime(2019, 4, 7, 1, 59), AEDT, NORMAL), (datetime(2019, 4, 7, 1, 59, fold=1), AEDT, NORMAL), (datetime(2019, 4, 7, 2, 0, fold=0), AEDT, FOLD), (datetime(2019, 4, 7, 2, 1, fold=0), AEDT, FOLD), (datetime(2019, 4, 7, 2, 0, fold=1), AEST, FOLD), (datetime(2019, 4, 7, 2, 1, fold=1), AEST, FOLD), (datetime(2019, 4, 7, 3, 0, fold=0), AEST, NORMAL), (datetime(2019, 4, 7, 3, 0, fold=1), AEST, NORMAL), (datetime(2019, 10, 5, 0), AEST, NORMAL), (datetime(2019, 10, 6, 1, 59), AEST, NORMAL), (datetime(2019, 10, 6, 2, 0, fold=0), AEST, GAP), (datetime(2019, 10, 6, 2, 0, fold=1), AEDT, GAP), (datetime(2019, 10, 6, 3, 0), AEDT, NORMAL))

        @call
        def _add():
            if False:
                for i in range(10):
                    print('nop')
            tzstr = 'IST-1GMT0,M10.5.0,M3.5.0/1'
            GMT = ZoneOffset('GMT', ZERO, -ONE_H)
            IST = ZoneOffset('IST', ONE_H, ZERO)
            cases[tzstr] = ((datetime(2019, 3, 30), GMT, NORMAL), (datetime(2019, 3, 31, 0, 59), GMT, NORMAL), (datetime(2019, 3, 31, 2, 0), IST, NORMAL), (datetime(2019, 10, 26), IST, NORMAL), (datetime(2019, 10, 27, 0, 59, fold=1), IST, NORMAL), (datetime(2019, 10, 27, 1, 0, fold=0), IST, FOLD), (datetime(2019, 10, 27, 1, 0, fold=1), GMT, FOLD), (datetime(2019, 10, 27, 2, 0, fold=1), GMT, NORMAL), (datetime(2020, 3, 29, 0, 59), GMT, NORMAL), (datetime(2020, 3, 29, 2, 0), IST, NORMAL), (datetime(2020, 10, 25, 0, 59, fold=1), IST, NORMAL), (datetime(2020, 10, 25, 1, 0, fold=0), IST, FOLD), (datetime(2020, 10, 25, 2, 0, fold=1), GMT, NORMAL))

        @call
        def _add():
            if False:
                i = 10
                return i + 15
            tzstr = '<+11>-11'
            cases[tzstr] = ((datetime(2020, 1, 1), ZoneOffset('+11', timedelta(hours=11)), NORMAL),)

        @call
        def _add():
            if False:
                return 10
            tzstr = '<-04>4<-03>,M9.1.6/24,M4.1.6/24'
            M04 = ZoneOffset('-04', timedelta(hours=-4))
            M03 = ZoneOffset('-03', timedelta(hours=-3), ONE_H)
            cases[tzstr] = ((datetime(2020, 5, 1), M04, NORMAL), (datetime(2020, 11, 1), M03, NORMAL))

        @call
        def _add():
            if False:
                while True:
                    i = 10
            tzstr = 'EST5EDT,0/0,J365/25'
            EDT = ZoneOffset('EDT', timedelta(hours=-4), ONE_H)
            cases[tzstr] = ((datetime(2019, 1, 1), EDT, NORMAL), (datetime(2019, 6, 1), EDT, NORMAL), (datetime(2019, 12, 31, 23, 59, 59, 999999), EDT, NORMAL), (datetime(2020, 1, 1), EDT, NORMAL), (datetime(2020, 3, 1), EDT, NORMAL), (datetime(2020, 6, 1), EDT, NORMAL), (datetime(2020, 12, 31, 23, 59, 59, 999999), EDT, NORMAL), (datetime(2400, 1, 1), EDT, NORMAL), (datetime(2400, 3, 1), EDT, NORMAL), (datetime(2400, 12, 31, 23, 59, 59, 999999), EDT, NORMAL))

        @call
        def _add():
            if False:
                while True:
                    i = 10
            tzstr = 'AAA3BBB,J60/12,J305/12'
            AAA = ZoneOffset('AAA', timedelta(hours=-3))
            BBB = ZoneOffset('BBB', timedelta(hours=-2), ONE_H)
            cases[tzstr] = ((datetime(2019, 1, 1), AAA, NORMAL), (datetime(2019, 2, 28), AAA, NORMAL), (datetime(2019, 3, 1, 11, 59), AAA, NORMAL), (datetime(2019, 3, 1, 12, fold=0), AAA, GAP), (datetime(2019, 3, 1, 12, fold=1), BBB, GAP), (datetime(2019, 3, 1, 13), BBB, NORMAL), (datetime(2019, 11, 1, 10, 59), BBB, NORMAL), (datetime(2019, 11, 1, 11, fold=0), BBB, FOLD), (datetime(2019, 11, 1, 11, fold=1), AAA, FOLD), (datetime(2019, 11, 1, 12), AAA, NORMAL), (datetime(2019, 12, 31, 23, 59, 59, 999999), AAA, NORMAL), (datetime(2020, 1, 1), AAA, NORMAL), (datetime(2020, 2, 29), AAA, NORMAL), (datetime(2020, 3, 1, 11, 59), AAA, NORMAL), (datetime(2020, 3, 1, 12, fold=0), AAA, GAP), (datetime(2020, 3, 1, 12, fold=1), BBB, GAP), (datetime(2020, 3, 1, 13), BBB, NORMAL), (datetime(2020, 11, 1, 10, 59), BBB, NORMAL), (datetime(2020, 11, 1, 11, fold=0), BBB, FOLD), (datetime(2020, 11, 1, 11, fold=1), AAA, FOLD), (datetime(2020, 11, 1, 12), AAA, NORMAL), (datetime(2020, 12, 31, 23, 59, 59, 999999), AAA, NORMAL))

        @call
        def _add():
            if False:
                for i in range(10):
                    print('nop')
            tzstr = '<-03>3<-02>,M3.5.0/-2,M10.5.0/-1'
            N03 = ZoneOffset('-03', timedelta(hours=-3))
            N02 = ZoneOffset('-02', timedelta(hours=-2), ONE_H)
            cases[tzstr] = ((datetime(2020, 3, 27), N03, NORMAL), (datetime(2020, 3, 28, 21, 59, 59), N03, NORMAL), (datetime(2020, 3, 28, 22, fold=0), N03, GAP), (datetime(2020, 3, 28, 22, fold=1), N02, GAP), (datetime(2020, 3, 28, 23), N02, NORMAL), (datetime(2020, 10, 24, 21), N02, NORMAL), (datetime(2020, 10, 24, 22, fold=0), N02, FOLD), (datetime(2020, 10, 24, 22, fold=1), N03, FOLD), (datetime(2020, 10, 24, 23), N03, NORMAL))

        @call
        def _add():
            if False:
                i = 10
                return i + 15
            tzstr = 'AAA3BBB,M3.2.0/01:30,M11.1.0/02:15:45'
            AAA = ZoneOffset('AAA', timedelta(hours=-3))
            BBB = ZoneOffset('BBB', timedelta(hours=-2), ONE_H)
            cases[tzstr] = ((datetime(2012, 3, 11, 1, 0), AAA, NORMAL), (datetime(2012, 3, 11, 1, 30, fold=0), AAA, GAP), (datetime(2012, 3, 11, 1, 30, fold=1), BBB, GAP), (datetime(2012, 3, 11, 2, 30), BBB, NORMAL), (datetime(2012, 11, 4, 1, 15, 44, 999999), BBB, NORMAL), (datetime(2012, 11, 4, 1, 15, 45, fold=0), BBB, FOLD), (datetime(2012, 11, 4, 1, 15, 45, fold=1), AAA, FOLD), (datetime(2012, 11, 4, 2, 15, 45), AAA, NORMAL))
        cls.test_cases = cases

class CTZStrTest(TZStrTest):
    module = c_zoneinfo

class ZoneInfoCacheTest(TzPathUserMixin, ZoneInfoTestBase):
    module = py_zoneinfo

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.klass.clear_cache()
        super().setUp()

    @property
    def zoneinfo_data(self):
        if False:
            return 10
        return ZONEINFO_DATA

    @property
    def tzpath(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.zoneinfo_data.tzpath]

    def test_ephemeral_zones(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(self.klass('America/Los_Angeles'), self.klass('America/Los_Angeles'))

    def test_strong_refs(self):
        if False:
            for i in range(10):
                print('nop')
        tz0 = self.klass('Australia/Sydney')
        tz1 = self.klass('Australia/Sydney')
        self.assertIs(tz0, tz1)

    def test_no_cache(self):
        if False:
            return 10
        tz0 = self.klass('Europe/Lisbon')
        tz1 = self.klass.no_cache('Europe/Lisbon')
        self.assertIsNot(tz0, tz1)

    def test_cache_reset_tzpath(self):
        if False:
            i = 10
            return i + 15
        'Test that the cache persists when tzpath has been changed.\n\n        The PEP specifies that as long as a reference exists to one zone\n        with a given key, the primary constructor must continue to return\n        the same object.\n        '
        zi0 = self.klass('America/Los_Angeles')
        with self.tzpath_context([]):
            zi1 = self.klass('America/Los_Angeles')
        self.assertIs(zi0, zi1)

    def test_clear_cache_explicit_none(self):
        if False:
            i = 10
            return i + 15
        la0 = self.klass('America/Los_Angeles')
        self.klass.clear_cache(only_keys=None)
        la1 = self.klass('America/Los_Angeles')
        self.assertIsNot(la0, la1)

    def test_clear_cache_one_key(self):
        if False:
            print('Hello World!')
        'Tests that you can clear a single key from the cache.'
        la0 = self.klass('America/Los_Angeles')
        dub0 = self.klass('Europe/Dublin')
        self.klass.clear_cache(only_keys=['America/Los_Angeles'])
        la1 = self.klass('America/Los_Angeles')
        dub1 = self.klass('Europe/Dublin')
        self.assertIsNot(la0, la1)
        self.assertIs(dub0, dub1)

    def test_clear_cache_two_keys(self):
        if False:
            i = 10
            return i + 15
        la0 = self.klass('America/Los_Angeles')
        dub0 = self.klass('Europe/Dublin')
        tok0 = self.klass('Asia/Tokyo')
        self.klass.clear_cache(only_keys=['America/Los_Angeles', 'Europe/Dublin'])
        la1 = self.klass('America/Los_Angeles')
        dub1 = self.klass('Europe/Dublin')
        tok1 = self.klass('Asia/Tokyo')
        self.assertIsNot(la0, la1)
        self.assertIsNot(dub0, dub1)
        self.assertIs(tok0, tok1)

class CZoneInfoCacheTest(ZoneInfoCacheTest):
    module = c_zoneinfo

class ZoneInfoPickleTest(TzPathUserMixin, ZoneInfoTestBase):
    module = py_zoneinfo

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.klass.clear_cache()
        with contextlib.ExitStack() as stack:
            stack.enter_context(test_support.set_zoneinfo_module(self.module))
            self.addCleanup(stack.pop_all().close)
        super().setUp()

    @property
    def zoneinfo_data(self):
        if False:
            return 10
        return ZONEINFO_DATA

    @property
    def tzpath(self):
        if False:
            print('Hello World!')
        return [self.zoneinfo_data.tzpath]

    def test_cache_hit(self):
        if False:
            for i in range(10):
                print('nop')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                zi_in = self.klass('Europe/Dublin')
                pkl = pickle.dumps(zi_in, protocol=proto)
                zi_rt = pickle.loads(pkl)
                with self.subTest(test='Is non-pickled ZoneInfo'):
                    self.assertIs(zi_in, zi_rt)
                zi_rt2 = pickle.loads(pkl)
                with self.subTest(test='Is unpickled ZoneInfo'):
                    self.assertIs(zi_rt, zi_rt2)

    def test_cache_miss(self):
        if False:
            while True:
                i = 10
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                zi_in = self.klass('Europe/Dublin')
                pkl = pickle.dumps(zi_in, protocol=proto)
                del zi_in
                self.klass.clear_cache()
                zi_rt = pickle.loads(pkl)
                zi_rt2 = pickle.loads(pkl)
                self.assertIs(zi_rt, zi_rt2)

    def test_no_cache(self):
        if False:
            print('Hello World!')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                zi_no_cache = self.klass.no_cache('Europe/Dublin')
                pkl = pickle.dumps(zi_no_cache, protocol=proto)
                zi_rt = pickle.loads(pkl)
                with self.subTest(test='Not the pickled object'):
                    self.assertIsNot(zi_rt, zi_no_cache)
                zi_rt2 = pickle.loads(pkl)
                with self.subTest(test='Not a second unpickled object'):
                    self.assertIsNot(zi_rt, zi_rt2)
                zi_cache = self.klass('Europe/Dublin')
                with self.subTest(test='Not a cached object'):
                    self.assertIsNot(zi_rt, zi_cache)

    def test_from_file(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'Europe/Dublin'
        with open(self.zoneinfo_data.path_from_key(key), 'rb') as f:
            zi_nokey = self.klass.from_file(f)
            f.seek(0)
            zi_key = self.klass.from_file(f, key=key)
        test_cases = [(zi_key, 'ZoneInfo with key'), (zi_nokey, 'ZoneInfo without key')]
        for (zi, test_name) in test_cases:
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(test_name=test_name, proto=proto):
                    with self.assertRaises(pickle.PicklingError):
                        pickle.dumps(zi, protocol=proto)

    def test_pickle_after_from_file(self):
        if False:
            while True:
                i = 10
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                key = 'Europe/Dublin'
                zi = self.klass(key)
                pkl_0 = pickle.dumps(zi, protocol=proto)
                zi_rt_0 = pickle.loads(pkl_0)
                self.assertIs(zi, zi_rt_0)
                with open(self.zoneinfo_data.path_from_key(key), 'rb') as f:
                    zi_ff = self.klass.from_file(f, key=key)
                pkl_1 = pickle.dumps(zi, protocol=proto)
                zi_rt_1 = pickle.loads(pkl_1)
                self.assertIs(zi, zi_rt_1)
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(zi_ff, protocol=proto)
                pkl_2 = pickle.dumps(zi, protocol=proto)
                zi_rt_2 = pickle.loads(pkl_2)
                self.assertIs(zi, zi_rt_2)

class CZoneInfoPickleTest(ZoneInfoPickleTest):
    module = c_zoneinfo

class CallingConventionTest(ZoneInfoTestBase):
    """Tests for functions with restricted calling conventions."""
    module = py_zoneinfo

    @property
    def zoneinfo_data(self):
        if False:
            for i in range(10):
                print('nop')
        return ZONEINFO_DATA

    def test_from_file(self):
        if False:
            while True:
                i = 10
        with open(self.zoneinfo_data.path_from_key('UTC'), 'rb') as f:
            with self.assertRaises(TypeError):
                self.klass.from_file(fobj=f)

    def test_clear_cache(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self.klass.clear_cache(['UTC'])

class CCallingConventionTest(CallingConventionTest):
    module = c_zoneinfo

class TzPathTest(TzPathUserMixin, ZoneInfoTestBase):
    module = py_zoneinfo

    @staticmethod
    @contextlib.contextmanager
    def python_tzpath_context(value):
        if False:
            for i in range(10):
                print('nop')
        path_var = 'PYTHONTZPATH'
        try:
            with OS_ENV_LOCK:
                old_env = os.environ.get(path_var, None)
                os.environ[path_var] = value
                yield
        finally:
            if old_env is None:
                del os.environ[path_var]
            else:
                os.environ[path_var] = old_env

    def test_env_variable(self):
        if False:
            return 10
        'Tests that the environment variable works with reset_tzpath.'
        new_paths = [('', []), ('/etc/zoneinfo', ['/etc/zoneinfo']), (f'/a/b/c{os.pathsep}/d/e/f', ['/a/b/c', '/d/e/f'])]
        for (new_path_var, expected_result) in new_paths:
            with self.python_tzpath_context(new_path_var):
                with self.subTest(tzpath=new_path_var):
                    self.module.reset_tzpath()
                    tzpath = self.module.TZPATH
                    self.assertSequenceEqual(tzpath, expected_result)

    def test_env_variable_relative_paths(self):
        if False:
            return 10
        test_cases = [[('path/to/somewhere',), ()], [('/usr/share/zoneinfo', 'path/to/somewhere'), ('/usr/share/zoneinfo',)], [('../relative/path',), ()], [('/usr/share/zoneinfo', '../relative/path'), ('/usr/share/zoneinfo',)], [('path/to/somewhere', '../relative/path'), ()], [('/usr/share/zoneinfo', 'path/to/somewhere', '../relative/path'), ('/usr/share/zoneinfo',)]]
        for (input_paths, expected_paths) in test_cases:
            path_var = os.pathsep.join(input_paths)
            with self.python_tzpath_context(path_var):
                with self.subTest('warning', path_var=path_var):
                    with self.assertWarns(self.module.InvalidTZPathWarning):
                        self.module.reset_tzpath()
                tzpath = self.module.TZPATH
                with self.subTest('filtered', path_var=path_var):
                    self.assertSequenceEqual(tzpath, expected_paths)

    def test_reset_tzpath_kwarg(self):
        if False:
            print('Hello World!')
        self.module.reset_tzpath(to=['/a/b/c'])
        self.assertSequenceEqual(self.module.TZPATH, ('/a/b/c',))

    def test_reset_tzpath_relative_paths(self):
        if False:
            return 10
        bad_values = [('path/to/somewhere',), ('/usr/share/zoneinfo', 'path/to/somewhere'), ('../relative/path',), ('/usr/share/zoneinfo', '../relative/path'), ('path/to/somewhere', '../relative/path'), ('/usr/share/zoneinfo', 'path/to/somewhere', '../relative/path')]
        for input_paths in bad_values:
            with self.subTest(input_paths=input_paths):
                with self.assertRaises(ValueError):
                    self.module.reset_tzpath(to=input_paths)

    def test_tzpath_type_error(self):
        if False:
            print('Hello World!')
        bad_values = ['/etc/zoneinfo:/usr/share/zoneinfo', b'/etc/zoneinfo:/usr/share/zoneinfo', 0]
        for bad_value in bad_values:
            with self.subTest(value=bad_value):
                with self.assertRaises(TypeError):
                    self.module.reset_tzpath(bad_value)

    def test_tzpath_attribute(self):
        if False:
            print('Hello World!')
        tzpath_0 = ['/one', '/two']
        tzpath_1 = ['/three']
        with self.tzpath_context(tzpath_0):
            query_0 = self.module.TZPATH
        with self.tzpath_context(tzpath_1):
            query_1 = self.module.TZPATH
        self.assertSequenceEqual(tzpath_0, query_0)
        self.assertSequenceEqual(tzpath_1, query_1)

class CTzPathTest(TzPathTest):
    module = c_zoneinfo

class TestModule(ZoneInfoTestBase):
    module = py_zoneinfo

    @property
    def zoneinfo_data(self):
        if False:
            for i in range(10):
                print('nop')
        return ZONEINFO_DATA

    @cached_property
    def _UTC_bytes(self):
        if False:
            return 10
        zone_file = self.zoneinfo_data.path_from_key('UTC')
        with open(zone_file, 'rb') as f:
            return f.read()

    def touch_zone(self, key, tz_root):
        if False:
            for i in range(10):
                print('nop')
        'Creates a valid TZif file at key under the zoneinfo root tz_root.\n\n        tz_root must exist, but all folders below that will be created.\n        '
        if not os.path.exists(tz_root):
            raise FileNotFoundError(f'{tz_root} does not exist.')
        (root_dir, *tail) = key.rsplit('/', 1)
        if tail:
            os.makedirs(os.path.join(tz_root, root_dir), exist_ok=True)
        zonefile_path = os.path.join(tz_root, key)
        with open(zonefile_path, 'wb') as f:
            f.write(self._UTC_bytes)

    def test_getattr_error(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AttributeError):
            self.module.NOATTRIBUTE

    def test_dir_contains_all(self):
        if False:
            return 10
        'dir(self.module) should at least contain everything in __all__.'
        module_all_set = set(self.module.__all__)
        module_dir_set = set(dir(self.module))
        difference = module_all_set - module_dir_set
        self.assertFalse(difference)

    def test_dir_unique(self):
        if False:
            print('Hello World!')
        'Test that there are no duplicates in dir(self.module)'
        module_dir = dir(self.module)
        module_unique = set(module_dir)
        self.assertCountEqual(module_dir, module_unique)

    def test_available_timezones(self):
        if False:
            i = 10
            return i + 15
        with self.tzpath_context([self.zoneinfo_data.tzpath]):
            self.assertTrue(self.zoneinfo_data.keys)
            available_keys = self.module.available_timezones()
            zoneinfo_keys = set(self.zoneinfo_data.keys)
            union = zoneinfo_keys & available_keys
            self.assertEqual(zoneinfo_keys, union)

    def test_available_timezones_weirdzone(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as td:
            self.touch_zone('Mars/Olympus_Mons', td)
            with self.tzpath_context([td]):
                available_keys = self.module.available_timezones()
                self.assertIn('Mars/Olympus_Mons', available_keys)

    def test_folder_exclusions(self):
        if False:
            print('Hello World!')
        expected = {'America/Los_Angeles', 'America/Santiago', 'America/Indiana/Indianapolis', 'UTC', 'Europe/Paris', 'Europe/London', 'Asia/Tokyo', 'Australia/Sydney'}
        base_tree = list(expected)
        posix_tree = [f'posix/{x}' for x in base_tree]
        right_tree = [f'right/{x}' for x in base_tree]
        cases = [('base_tree', base_tree), ('base_and_posix', base_tree + posix_tree), ('base_and_right', base_tree + right_tree), ('all_trees', base_tree + right_tree + posix_tree)]
        with tempfile.TemporaryDirectory() as td:
            for (case_name, tree) in cases:
                tz_root = os.path.join(td, case_name)
                os.mkdir(tz_root)
                for key in tree:
                    self.touch_zone(key, tz_root)
                with self.tzpath_context([tz_root]):
                    with self.subTest(case_name):
                        actual = self.module.available_timezones()
                        self.assertEqual(actual, expected)

    def test_exclude_posixrules(self):
        if False:
            for i in range(10):
                print('nop')
        expected = {'America/New_York', 'Europe/London'}
        tree = list(expected) + ['posixrules']
        with tempfile.TemporaryDirectory() as td:
            for key in tree:
                self.touch_zone(key, td)
            with self.tzpath_context([td]):
                actual = self.module.available_timezones()
                self.assertEqual(actual, expected)

class CTestModule(TestModule):
    module = c_zoneinfo

class ExtensionBuiltTest(unittest.TestCase):
    """Smoke test to ensure that the C and Python extensions are both tested.

    Because the intention is for the Python and C versions of ZoneInfo to
    behave identically, these tests necessarily rely on implementation details,
    so the tests may need to be adjusted if the implementations change. Do not
    rely on these tests as an indication of stable properties of these classes.
    """

    def test_cache_location(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(hasattr(c_zoneinfo.ZoneInfo, '_weak_cache'))
        self.assertTrue(hasattr(py_zoneinfo.ZoneInfo, '_weak_cache'))

    def test_gc_tracked(self):
        if False:
            return 10
        import gc
        self.assertTrue(gc.is_tracked(py_zoneinfo.ZoneInfo))
        self.assertFalse(gc.is_tracked(c_zoneinfo.ZoneInfo))

@dataclasses.dataclass(frozen=True)
class ZoneOffset:
    tzname: str
    utcoffset: timedelta
    dst: timedelta = ZERO

@dataclasses.dataclass(frozen=True)
class ZoneTransition:
    transition: datetime
    offset_before: ZoneOffset
    offset_after: ZoneOffset

    @property
    def transition_utc(self):
        if False:
            return 10
        return (self.transition - self.offset_before.utcoffset).replace(tzinfo=timezone.utc)

    @property
    def fold(self):
        if False:
            return 10
        'Whether this introduces a fold'
        return self.offset_before.utcoffset > self.offset_after.utcoffset

    @property
    def gap(self):
        if False:
            print('Hello World!')
        'Whether this introduces a gap'
        return self.offset_before.utcoffset < self.offset_after.utcoffset

    @property
    def delta(self):
        if False:
            i = 10
            return i + 15
        return self.offset_after.utcoffset - self.offset_before.utcoffset

    @property
    def anomaly_start(self):
        if False:
            print('Hello World!')
        if self.fold:
            return self.transition + self.delta
        else:
            return self.transition

    @property
    def anomaly_end(self):
        if False:
            i = 10
            return i + 15
        if not self.fold:
            return self.transition + self.delta
        else:
            return self.transition

class ZoneInfoData:

    def __init__(self, source_json, tzpath, v1=False):
        if False:
            return 10
        self.tzpath = pathlib.Path(tzpath)
        self.keys = []
        self.v1 = v1
        self._populate_tzpath(source_json)

    def path_from_key(self, key):
        if False:
            i = 10
            return i + 15
        return self.tzpath / key

    def _populate_tzpath(self, source_json):
        if False:
            print('Hello World!')
        with open(source_json, 'rb') as f:
            zoneinfo_dict = json.load(f)
        zoneinfo_data = zoneinfo_dict['data']
        for (key, value) in zoneinfo_data.items():
            self.keys.append(key)
            raw_data = self._decode_text(value)
            if self.v1:
                data = self._convert_to_v1(raw_data)
            else:
                data = raw_data
            destination = self.path_from_key(key)
            destination.parent.mkdir(exist_ok=True, parents=True)
            with open(destination, 'wb') as f:
                f.write(data)

    def _decode_text(self, contents):
        if False:
            i = 10
            return i + 15
        raw_data = b''.join(map(str.encode, contents))
        decoded = base64.b85decode(raw_data)
        return lzma.decompress(decoded)

    def _convert_to_v1(self, contents):
        if False:
            i = 10
            return i + 15
        assert contents[0:4] == b'TZif', 'Invalid TZif data found!'
        version = int(contents[4:5])
        header_start = 4 + 16
        header_end = header_start + 24
        assert version >= 2, 'Version 1 file found: no conversion necessary'
        (isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt) = struct.unpack('>6l', contents[header_start:header_end])
        file_size = timecnt * 5 + typecnt * 6 + charcnt + leapcnt * 8 + isstdcnt + isutcnt
        file_size += header_end
        out = b'TZif' + b'\x00' + contents[5:file_size]
        assert contents[file_size:file_size + 4] == b'TZif', 'Version 2 file not truncated at Version 2 header'
        return out

class ZoneDumpData:

    @classmethod
    def transition_keys(cls):
        if False:
            i = 10
            return i + 15
        return cls._get_zonedump().keys()

    @classmethod
    def load_transition_examples(cls, key):
        if False:
            for i in range(10):
                print('nop')
        return cls._get_zonedump()[key]

    @classmethod
    def fixed_offset_zones(cls):
        if False:
            while True:
                i = 10
        if not cls._FIXED_OFFSET_ZONES:
            cls._populate_fixed_offsets()
        return cls._FIXED_OFFSET_ZONES.items()

    @classmethod
    def _get_zonedump(cls):
        if False:
            i = 10
            return i + 15
        if not cls._ZONEDUMP_DATA:
            cls._populate_zonedump_data()
        return cls._ZONEDUMP_DATA

    @classmethod
    def _populate_fixed_offsets(cls):
        if False:
            for i in range(10):
                print('nop')
        cls._FIXED_OFFSET_ZONES = {'UTC': ZoneOffset('UTC', ZERO, ZERO)}

    @classmethod
    def _populate_zonedump_data(cls):
        if False:
            return 10

        def _Africa_Abidjan():
            if False:
                return 10
            LMT = ZoneOffset('LMT', timedelta(seconds=-968))
            GMT = ZoneOffset('GMT', ZERO)
            return [ZoneTransition(datetime(1912, 1, 1), LMT, GMT)]

        def _Africa_Casablanca():
            if False:
                return 10
            P00_s = ZoneOffset('+00', ZERO, ZERO)
            P01_d = ZoneOffset('+01', ONE_H, ONE_H)
            P00_d = ZoneOffset('+00', ZERO, -ONE_H)
            P01_s = ZoneOffset('+01', ONE_H, ZERO)
            return [ZoneTransition(datetime(2018, 3, 25, 2), P00_s, P01_d), ZoneTransition(datetime(2018, 5, 13, 3), P01_d, P00_s), ZoneTransition(datetime(2018, 6, 17, 2), P00_s, P01_d), ZoneTransition(datetime(2018, 10, 28, 3), P01_d, P01_s), ZoneTransition(datetime(2019, 5, 5, 3), P01_s, P00_d), ZoneTransition(datetime(2019, 6, 9, 2), P00_d, P01_s)]

        def _America_Los_Angeles():
            if False:
                while True:
                    i = 10
            LMT = ZoneOffset('LMT', timedelta(seconds=-28378), ZERO)
            PST = ZoneOffset('PST', timedelta(hours=-8), ZERO)
            PDT = ZoneOffset('PDT', timedelta(hours=-7), ONE_H)
            PWT = ZoneOffset('PWT', timedelta(hours=-7), ONE_H)
            PPT = ZoneOffset('PPT', timedelta(hours=-7), ONE_H)
            return [ZoneTransition(datetime(1883, 11, 18, 12, 7, 2), LMT, PST), ZoneTransition(datetime(1918, 3, 31, 2), PST, PDT), ZoneTransition(datetime(1918, 3, 31, 2), PST, PDT), ZoneTransition(datetime(1918, 10, 27, 2), PDT, PST), ZoneTransition(datetime(1942, 2, 9, 2), PST, PWT), ZoneTransition(datetime(1945, 8, 14, 16), PWT, PPT), ZoneTransition(datetime(1945, 9, 30, 2), PPT, PST), ZoneTransition(datetime(2015, 3, 8, 2), PST, PDT), ZoneTransition(datetime(2015, 11, 1, 2), PDT, PST), ZoneTransition(datetime(2450, 3, 13, 2), PST, PDT), ZoneTransition(datetime(2450, 11, 6, 2), PDT, PST)]

        def _America_Santiago():
            if False:
                for i in range(10):
                    print('nop')
            LMT = ZoneOffset('LMT', timedelta(seconds=-16966), ZERO)
            SMT = ZoneOffset('SMT', timedelta(seconds=-16966), ZERO)
            N05 = ZoneOffset('-05', timedelta(seconds=-18000), ZERO)
            N04 = ZoneOffset('-04', timedelta(seconds=-14400), ZERO)
            N03 = ZoneOffset('-03', timedelta(seconds=-10800), ONE_H)
            return [ZoneTransition(datetime(1890, 1, 1), LMT, SMT), ZoneTransition(datetime(1910, 1, 10), SMT, N05), ZoneTransition(datetime(1916, 7, 1), N05, SMT), ZoneTransition(datetime(2008, 3, 30), N03, N04), ZoneTransition(datetime(2008, 10, 12), N04, N03), ZoneTransition(datetime(2040, 4, 8), N03, N04), ZoneTransition(datetime(2040, 9, 2), N04, N03)]

        def _Asia_Tokyo():
            if False:
                print('Hello World!')
            JST = ZoneOffset('JST', timedelta(seconds=32400), ZERO)
            JDT = ZoneOffset('JDT', timedelta(seconds=36000), ONE_H)
            return [ZoneTransition(datetime(1948, 5, 2), JST, JDT), ZoneTransition(datetime(1948, 9, 12, 1), JDT, JST), ZoneTransition(datetime(1951, 9, 9, 1), JDT, JST)]

        def _Australia_Sydney():
            if False:
                while True:
                    i = 10
            LMT = ZoneOffset('LMT', timedelta(seconds=36292), ZERO)
            AEST = ZoneOffset('AEST', timedelta(seconds=36000), ZERO)
            AEDT = ZoneOffset('AEDT', timedelta(seconds=39600), ONE_H)
            return [ZoneTransition(datetime(1895, 2, 1), LMT, AEST), ZoneTransition(datetime(1917, 1, 1, 0, 1), AEST, AEDT), ZoneTransition(datetime(1917, 3, 25, 2), AEDT, AEST), ZoneTransition(datetime(2012, 4, 1, 3), AEDT, AEST), ZoneTransition(datetime(2012, 10, 7, 2), AEST, AEDT), ZoneTransition(datetime(2040, 4, 1, 3), AEDT, AEST), ZoneTransition(datetime(2040, 10, 7, 2), AEST, AEDT)]

        def _Europe_Dublin():
            if False:
                return 10
            LMT = ZoneOffset('LMT', timedelta(seconds=-1500), ZERO)
            DMT = ZoneOffset('DMT', timedelta(seconds=-1521), ZERO)
            IST_0 = ZoneOffset('IST', timedelta(seconds=2079), ONE_H)
            GMT_0 = ZoneOffset('GMT', ZERO, ZERO)
            BST = ZoneOffset('BST', ONE_H, ONE_H)
            GMT_1 = ZoneOffset('GMT', ZERO, -ONE_H)
            IST_1 = ZoneOffset('IST', ONE_H, ZERO)
            return [ZoneTransition(datetime(1880, 8, 2, 0), LMT, DMT), ZoneTransition(datetime(1916, 5, 21, 2), DMT, IST_0), ZoneTransition(datetime(1916, 10, 1, 3), IST_0, GMT_0), ZoneTransition(datetime(1917, 4, 8, 2), GMT_0, BST), ZoneTransition(datetime(2016, 3, 27, 1), GMT_1, IST_1), ZoneTransition(datetime(2016, 10, 30, 2), IST_1, GMT_1), ZoneTransition(datetime(2487, 3, 30, 1), GMT_1, IST_1), ZoneTransition(datetime(2487, 10, 26, 2), IST_1, GMT_1)]

        def _Europe_Lisbon():
            if False:
                print('Hello World!')
            WET = ZoneOffset('WET', ZERO, ZERO)
            WEST = ZoneOffset('WEST', ONE_H, ONE_H)
            CET = ZoneOffset('CET', ONE_H, ZERO)
            CEST = ZoneOffset('CEST', timedelta(seconds=7200), ONE_H)
            return [ZoneTransition(datetime(1992, 3, 29, 1), WET, WEST), ZoneTransition(datetime(1992, 9, 27, 2), WEST, CET), ZoneTransition(datetime(1993, 3, 28, 2), CET, CEST), ZoneTransition(datetime(1993, 9, 26, 3), CEST, CET), ZoneTransition(datetime(1996, 3, 31, 2), CET, WEST), ZoneTransition(datetime(1996, 10, 27, 2), WEST, WET)]

        def _Europe_London():
            if False:
                for i in range(10):
                    print('nop')
            LMT = ZoneOffset('LMT', timedelta(seconds=-75), ZERO)
            GMT = ZoneOffset('GMT', ZERO, ZERO)
            BST = ZoneOffset('BST', ONE_H, ONE_H)
            return [ZoneTransition(datetime(1847, 12, 1), LMT, GMT), ZoneTransition(datetime(2005, 3, 27, 1), GMT, BST), ZoneTransition(datetime(2005, 10, 30, 2), BST, GMT), ZoneTransition(datetime(2043, 3, 29, 1), GMT, BST), ZoneTransition(datetime(2043, 10, 25, 2), BST, GMT)]

        def _Pacific_Kiritimati():
            if False:
                for i in range(10):
                    print('nop')
            LMT = ZoneOffset('LMT', timedelta(seconds=-37760), ZERO)
            N1040 = ZoneOffset('-1040', timedelta(seconds=-38400), ZERO)
            N10 = ZoneOffset('-10', timedelta(seconds=-36000), ZERO)
            P14 = ZoneOffset('+14', timedelta(seconds=50400), ZERO)
            return [ZoneTransition(datetime(1901, 1, 1), LMT, N1040), ZoneTransition(datetime(1979, 10, 1), N1040, N10), ZoneTransition(datetime(1994, 12, 31), N10, P14)]
        cls._ZONEDUMP_DATA = {'Africa/Abidjan': _Africa_Abidjan(), 'Africa/Casablanca': _Africa_Casablanca(), 'America/Los_Angeles': _America_Los_Angeles(), 'America/Santiago': _America_Santiago(), 'Australia/Sydney': _Australia_Sydney(), 'Asia/Tokyo': _Asia_Tokyo(), 'Europe/Dublin': _Europe_Dublin(), 'Europe/Lisbon': _Europe_Lisbon(), 'Europe/London': _Europe_London(), 'Pacific/Kiritimati': _Pacific_Kiritimati()}
    _ZONEDUMP_DATA = None
    _FIXED_OFFSET_ZONES = None