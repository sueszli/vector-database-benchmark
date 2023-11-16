import datetime as dt
import sys
import warnings
import pytest
from hypothesis import assume, given
from hypothesis.errors import InvalidArgument
from hypothesis.strategies import data, datetimes, just, sampled_from, times
from hypothesis.strategies._internal.datetime import datetime_does_not_exist
from tests.common.debug import assert_all_examples, assert_can_trigger_event, find_any, minimal
with warnings.catch_warnings():
    if sys.version_info[:2] >= (3, 12):
        warnings.simplefilter('ignore', DeprecationWarning)
    import pytz
    from dateutil.tz import datetime_exists
from hypothesis.extra.pytz import timezones

def test_utc_is_minimal():
    if False:
        for i in range(10):
            print('nop')
    assert pytz.UTC is minimal(timezones())

def test_can_generate_non_naive_time():
    if False:
        while True:
            i = 10
    assert minimal(times(timezones=timezones()), lambda d: d.tzinfo).tzinfo == pytz.UTC

def test_can_generate_non_naive_datetime():
    if False:
        print('Hello World!')
    assert minimal(datetimes(timezones=timezones()), lambda d: d.tzinfo).tzinfo == pytz.UTC

@given(datetimes(timezones=timezones()))
def test_timezone_aware_datetimes_are_timezone_aware(dt):
    if False:
        while True:
            i = 10
    assert dt.tzinfo is not None

@given(sampled_from(['min_value', 'max_value']), datetimes(timezones=timezones()))
def test_datetime_bounds_must_be_naive(name, val):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        datetimes(**{name: val}).validate()

def test_underflow_in_simplify():
    if False:
        for i in range(10):
            print('nop')
    minimal(datetimes(max_value=dt.datetime.min + dt.timedelta(days=3), timezones=timezones()), lambda x: x.tzinfo != pytz.UTC)

def test_overflow_in_simplify():
    if False:
        return 10
    minimal(datetimes(min_value=dt.datetime.max - dt.timedelta(days=3), timezones=timezones()), lambda x: x.tzinfo != pytz.UTC)

def test_timezones_arg_to_datetimes_must_be_search_strategy():
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidArgument):
        datetimes(timezones=pytz.all_timezones).validate()
    tz = [pytz.timezone(t) for t in pytz.all_timezones]
    with pytest.raises(InvalidArgument):
        datetimes(timezones=tz).validate()

@given(times(timezones=timezones()))
def test_timezone_aware_times_are_timezone_aware(dt):
    if False:
        while True:
            i = 10
    assert dt.tzinfo is not None

def test_can_generate_non_utc():
    if False:
        while True:
            i = 10
    times(timezones=timezones()).filter(lambda d: assume(d.tzinfo) and d.tzinfo.zone != 'UTC').validate()

@given(sampled_from(['min_value', 'max_value']), times(timezones=timezones()))
def test_time_bounds_must_be_naive(name, val):
    if False:
        return 10
    with pytest.raises(InvalidArgument):
        times(**{name: val}).validate()

@pytest.mark.parametrize('bound', [{'min_value': dt.datetime.max - dt.timedelta(days=3)}, {'max_value': dt.datetime.min + dt.timedelta(days=3)}])
def test_can_trigger_error_in_draw_near_boundary(bound):
    if False:
        return 10
    assert_can_trigger_event(datetimes(**bound, timezones=timezones()), lambda event: 'Failed to draw a datetime' in event)

@given(data(), datetimes(), datetimes())
def test_datetimes_stay_within_naive_bounds(data, lo, hi):
    if False:
        for i in range(10):
            print('nop')
    if lo > hi:
        (lo, hi) = (hi, lo)
    out = data.draw(datetimes(lo, hi, timezones=timezones()))
    assert lo <= out.replace(tzinfo=None) <= hi

@pytest.mark.parametrize('kw', [{'min_value': dt.datetime(2019, 3, 31), 'max_value': dt.datetime(2019, 4, 1), 'timezones': just(pytz.timezone('Europe/Dublin'))}, {'min_value': dt.datetime(2020, 10, 4), 'max_value': dt.datetime(2020, 10, 5), 'timezones': just(pytz.timezone('Australia/Sydney'))}])
def test_datetimes_can_exclude_imaginary(kw):
    if False:
        i = 10
        return i + 15
    find_any(datetimes(**kw, allow_imaginary=True), lambda x: not datetime_exists(x))
    assert_all_examples(datetimes(**kw, allow_imaginary=False), datetime_exists)

def test_really_weird_tzinfo_case():
    if False:
        print('Hello World!')
    x = dt.datetime(2019, 3, 31, 2, 30, tzinfo=pytz.timezone('Europe/Dublin'))
    assert x.tzinfo is not x.astimezone(dt.timezone.utc).astimezone(x.tzinfo)
    assert datetime_does_not_exist(x)