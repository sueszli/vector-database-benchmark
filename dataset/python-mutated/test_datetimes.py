import datetime as dt
import pytest
from hypothesis import given, settings
from hypothesis.strategies import dates, datetimes, timedeltas, times
from tests.common.debug import find_any, minimal

def test_can_find_positive_delta():
    if False:
        for i in range(10):
            print('nop')
    assert minimal(timedeltas(), lambda x: x.days > 0) == dt.timedelta(1)

def test_can_find_negative_delta():
    if False:
        return 10
    assert minimal(timedeltas(max_value=dt.timedelta(10 ** 6)), lambda x: x.days < 0) == dt.timedelta(-1)

def test_can_find_on_the_second():
    if False:
        i = 10
        return i + 15
    find_any(timedeltas(), lambda x: x.seconds == 0)

def test_can_find_off_the_second():
    if False:
        for i in range(10):
            print('nop')
    find_any(timedeltas(), lambda x: x.seconds != 0)

def test_simplifies_towards_zero_delta():
    if False:
        return 10
    d = minimal(timedeltas())
    assert d.days == d.seconds == d.microseconds == 0

def test_min_value_is_respected():
    if False:
        while True:
            i = 10
    assert minimal(timedeltas(min_value=dt.timedelta(days=10))).days == 10

def test_max_value_is_respected():
    if False:
        while True:
            i = 10
    assert minimal(timedeltas(max_value=dt.timedelta(days=-10))).days == -10

@given(timedeltas())
def test_single_timedelta(val):
    if False:
        while True:
            i = 10
    assert find_any(timedeltas(val, val)) is val

def test_simplifies_towards_millenium():
    if False:
        print('Hello World!')
    d = minimal(datetimes())
    assert d.year == 2000
    assert d.month == d.day == 1
    assert d.hour == d.minute == d.second == d.microsecond == 0

@given(datetimes())
def test_default_datetimes_are_naive(dt):
    if False:
        print('Hello World!')
    assert dt.tzinfo is None

def test_bordering_on_a_leap_year():
    if False:
        return 10
    x = minimal(datetimes(dt.datetime.min.replace(year=2003), dt.datetime.max.replace(year=2005)), lambda x: x.month == 2 and x.day == 29, timeout_after=60)
    assert x.year == 2004

def test_can_find_after_the_year_2000():
    if False:
        for i in range(10):
            print('nop')
    assert minimal(dates(), lambda x: x.year > 2000).year == 2001

def test_can_find_before_the_year_2000():
    if False:
        i = 10
        return i + 15
    assert minimal(dates(), lambda x: x.year < 2000).year == 1999

@pytest.mark.parametrize('month', range(1, 13))
def test_can_find_each_month(month):
    if False:
        while True:
            i = 10
    find_any(dates(), lambda x: x.month == month, settings(max_examples=10 ** 6))

def test_min_year_is_respected():
    if False:
        i = 10
        return i + 15
    assert minimal(dates(min_value=dt.date.min.replace(2003))).year == 2003

def test_max_year_is_respected():
    if False:
        print('Hello World!')
    assert minimal(dates(max_value=dt.date.min.replace(1998))).year == 1998

@given(dates())
def test_single_date(val):
    if False:
        while True:
            i = 10
    assert find_any(dates(val, val)) is val

def test_can_find_midnight():
    if False:
        i = 10
        return i + 15
    find_any(times(), lambda x: x.hour == x.minute == x.second == 0)

def test_can_find_non_midnight():
    if False:
        while True:
            i = 10
    assert minimal(times(), lambda x: x.hour != 0).hour == 1

def test_can_find_on_the_minute():
    if False:
        return 10
    find_any(times(), lambda x: x.second == 0)

def test_can_find_off_the_minute():
    if False:
        print('Hello World!')
    find_any(times(), lambda x: x.second != 0)

def test_simplifies_towards_midnight():
    if False:
        print('Hello World!')
    d = minimal(times())
    assert d.hour == d.minute == d.second == d.microsecond == 0

def test_can_generate_naive_time():
    if False:
        for i in range(10):
            print('nop')
    find_any(times(), lambda d: not d.tzinfo)

@given(times())
def test_naive_times_are_naive(dt):
    if False:
        while True:
            i = 10
    assert dt.tzinfo is None

def test_can_generate_datetime_with_fold_1():
    if False:
        return 10
    find_any(datetimes(), lambda d: d.fold)

def test_can_generate_time_with_fold_1():
    if False:
        return 10
    find_any(times(), lambda d: d.fold)

@given(datetimes(allow_imaginary=False))
def test_allow_imaginary_is_not_an_error_for_naive_datetimes(d):
    if False:
        print('Hello World!')
    pass