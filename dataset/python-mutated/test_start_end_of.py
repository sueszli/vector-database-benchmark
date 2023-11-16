from __future__ import annotations
import pytest
import pendulum
from tests.conftest import assert_datetime

def test_start_of_second():
    if False:
        return 10
    d = pendulum.now()
    new = d.start_of('second')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, d.second, 0)

def test_end_of_second():
    if False:
        i = 10
        return i + 15
    d = pendulum.now()
    new = d.end_of('second')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, d.second, 999999)

def test_start_of_minute():
    if False:
        print('Hello World!')
    d = pendulum.now()
    new = d.start_of('minute')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, 0, 0)

def test_end_of_minute():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.now()
    new = d.end_of('minute')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, d.minute, 59, 999999)

def test_start_of_hour():
    if False:
        print('Hello World!')
    d = pendulum.now()
    new = d.start_of('hour')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, 0, 0, 0)

def test_end_of_hour():
    if False:
        while True:
            i = 10
    d = pendulum.now()
    new = d.end_of('hour')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, d.hour, 59, 59, 999999)

def test_start_of_day():
    if False:
        return 10
    d = pendulum.now()
    new = d.start_of('day')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, 0, 0, 0, 0)

def test_end_of_day():
    if False:
        return 10
    d = pendulum.now()
    new = d.end_of('day')
    assert isinstance(new, pendulum.DateTime)
    assert_datetime(new, d.year, d.month, d.day, 23, 59, 59, 999999)

def test_start_of_month_is_fluid():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.now()
    assert isinstance(d.start_of('month'), pendulum.DateTime)

def test_start_of_month_from_now():
    if False:
        while True:
            i = 10
    d = pendulum.now()
    new = d.start_of('month')
    assert_datetime(new, d.year, d.month, 1, 0, 0, 0, 0)

def test_start_of_month_from_last_day():
    if False:
        return 10
    d = pendulum.datetime(2000, 1, 31, 2, 3, 4)
    new = d.start_of('month')
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)

def test_start_of_year_is_fluid():
    if False:
        i = 10
        return i + 15
    d = pendulum.now()
    new = d.start_of('year')
    assert isinstance(new, pendulum.DateTime)

def test_start_of_year_from_now():
    if False:
        return 10
    d = pendulum.now()
    new = d.start_of('year')
    assert_datetime(new, d.year, 1, 1, 0, 0, 0, 0)

def test_start_of_year_from_first_day():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.start_of('year')
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)

def test_start_of_year_from_last_day():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2000, 12, 31, 23, 59, 59)
    new = d.start_of('year')
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)

def test_end_of_month_is_fluid():
    if False:
        i = 10
        return i + 15
    d = pendulum.now()
    assert isinstance(d.end_of('month'), pendulum.DateTime)

def test_end_of_month():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2000, 1, 1, 2, 3, 4).end_of('month')
    new = d.end_of('month')
    assert_datetime(new, 2000, 1, 31, 23, 59, 59)

def test_end_of_month_from_last_day():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2000, 1, 31, 2, 3, 4)
    new = d.end_of('month')
    assert_datetime(new, 2000, 1, 31, 23, 59, 59)

def test_end_of_year_is_fluid():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.now()
    assert isinstance(d.end_of('year'), pendulum.DateTime)

def test_end_of_year_from_now():
    if False:
        print('Hello World!')
    d = pendulum.now().end_of('year')
    new = d.end_of('year')
    assert_datetime(new, d.year, 12, 31, 23, 59, 59, 999999)

def test_end_of_year_from_first_day():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.end_of('year')
    assert_datetime(new, 2000, 12, 31, 23, 59, 59, 999999)

def test_end_of_year_from_last_day():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2000, 12, 31, 23, 59, 59, 999999)
    new = d.end_of('year')
    assert_datetime(new, 2000, 12, 31, 23, 59, 59, 999999)

def test_start_of_decade_is_fluid():
    if False:
        print('Hello World!')
    d = pendulum.now()
    assert isinstance(d.start_of('decade'), pendulum.DateTime)

def test_start_of_decade_from_now():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.now()
    new = d.start_of('decade')
    assert_datetime(new, d.year - d.year % 10, 1, 1, 0, 0, 0, 0)

def test_start_of_decade_from_first_day():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.start_of('decade')
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)

def test_start_of_decade_from_last_day():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(2009, 12, 31, 23, 59, 59)
    new = d.start_of('decade')
    assert_datetime(new, 2000, 1, 1, 0, 0, 0, 0)

def test_end_of_decade_is_fluid():
    if False:
        return 10
    d = pendulum.now()
    assert isinstance(d.end_of('decade'), pendulum.DateTime)

def test_end_of_decade_from_now():
    if False:
        i = 10
        return i + 15
    d = pendulum.now()
    new = d.end_of('decade')
    assert_datetime(new, d.year - d.year % 10 + 9, 12, 31, 23, 59, 59, 999999)

def test_end_of_decade_from_first_day():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(2000, 1, 1, 1, 1, 1)
    new = d.end_of('decade')
    assert_datetime(new, 2009, 12, 31, 23, 59, 59, 999999)

def test_end_of_decade_from_last_day():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(2009, 12, 31, 23, 59, 59, 999999)
    new = d.end_of('decade')
    assert_datetime(new, 2009, 12, 31, 23, 59, 59, 999999)

def test_start_of_century_is_fluid():
    if False:
        while True:
            i = 10
    d = pendulum.now()
    assert isinstance(d.start_of('century'), pendulum.DateTime)

def test_start_of_century_from_now():
    if False:
        while True:
            i = 10
    d = pendulum.now()
    new = d.start_of('century')
    assert_datetime(new, d.year - d.year % 100 + 1, 1, 1, 0, 0, 0, 0)

def test_start_of_century_from_first_day():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(2001, 1, 1, 1, 1, 1)
    new = d.start_of('century')
    assert_datetime(new, 2001, 1, 1, 0, 0, 0, 0)

def test_start_of_century_from_last_day():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(2100, 12, 31, 23, 59, 59)
    new = d.start_of('century')
    assert_datetime(new, 2001, 1, 1, 0, 0, 0, 0)

def test_end_of_century_is_fluid():
    if False:
        return 10
    d = pendulum.now()
    assert isinstance(d.end_of('century'), pendulum.DateTime)

def test_end_of_century_from_now():
    if False:
        return 10
    now = pendulum.now()
    d = now.end_of('century')
    assert_datetime(d, now.year - now.year % 100 + 100, 12, 31, 23, 59, 59, 999999)

def test_end_of_century_from_first_day():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2001, 1, 1, 1, 1, 1)
    new = d.end_of('century')
    assert_datetime(new, 2100, 12, 31, 23, 59, 59, 999999)

def test_end_of_century_from_last_day():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2100, 12, 31, 23, 59, 59, 999999)
    new = d.end_of('century')
    assert_datetime(new, 2100, 12, 31, 23, 59, 59, 999999)

def test_average_is_fluid():
    if False:
        while True:
            i = 10
    d = pendulum.now().average()
    assert isinstance(d, pendulum.DateTime)

def test_average_from_same():
    if False:
        i = 10
        return i + 15
    d1 = pendulum.datetime(2000, 1, 31, 2, 3, 4)
    d2 = pendulum.datetime(2000, 1, 31, 2, 3, 4).average(d1)
    assert_datetime(d2, 2000, 1, 31, 2, 3, 4)

def test_average_from_greater():
    if False:
        for i in range(10):
            print('nop')
    d1 = pendulum.datetime(2000, 1, 1, 1, 1, 1, tz='local')
    d2 = pendulum.datetime(2009, 12, 31, 23, 59, 59, tz='local').average(d1)
    assert_datetime(d2, 2004, 12, 31, 12, 30, 30)

def test_average_from_lower():
    if False:
        return 10
    d1 = pendulum.datetime(2009, 12, 31, 23, 59, 59, tz='local')
    d2 = pendulum.datetime(2000, 1, 1, 1, 1, 1, tz='local').average(d1)
    assert_datetime(d2, 2004, 12, 31, 12, 30, 30)

def start_of_with_invalid_unit():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        pendulum.now().start_of('invalid')

def end_of_with_invalid_unit():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        pendulum.now().end_of('invalid')

def test_start_of_with_transition():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2013, 10, 27, 23, 59, 59, tz='Europe/Paris')
    assert d.offset == 3600
    assert d.start_of('month').offset == 7200
    assert d.start_of('day').offset == 7200
    assert d.start_of('year').offset == 3600

def test_start_of_on_date_before_transition():
    if False:
        return 10
    d = pendulum.datetime(2013, 10, 27, 0, 59, 59, tz='UTC').in_timezone('Europe/Paris')
    assert d.offset == 7200
    assert d.start_of('minute').offset == 7200
    assert d.start_of('hour').offset == 7200
    assert d.start_of('day').offset == 7200
    assert d.start_of('month').offset == 7200
    assert d.start_of('year').offset == 3600

def test_start_of_on_date_after_transition():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2013, 10, 27, 1, 59, 59, tz='UTC').in_timezone('Europe/Paris')
    assert d.offset == 3600
    assert d.start_of('minute').offset == 3600
    assert d.start_of('hour').offset == 3600
    assert d.start_of('day').offset == 7200
    assert d.start_of('month').offset == 7200
    assert d.start_of('year').offset == 3600

def test_end_of_with_transition():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2013, 3, 31, tz='Europe/Paris')
    assert d.offset == 3600
    assert d.end_of('month').offset == 7200
    assert d.end_of('day').offset == 7200
    assert d.end_of('year').offset == 3600

def test_end_of_on_date_before_transition():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(2013, 10, 27, 0, 0, 0, tz='UTC').in_timezone('Europe/Paris')
    assert d.offset == 7200
    assert d.end_of('minute').offset == 7200
    assert d.end_of('hour').offset == 7200
    assert d.end_of('day').offset == 3600
    assert d.end_of('month').offset == 3600
    assert d.end_of('year').offset == 3600

def test_end_of_on_date_after_transition():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(2013, 10, 27, 1, 0, 0, tz='UTC').in_timezone('Europe/Paris')
    assert d.offset == 3600
    assert d.end_of('minute').offset == 3600
    assert d.end_of('hour').offset == 3600
    assert d.end_of('day').offset == 3600
    assert d.end_of('month').offset == 3600
    assert d.end_of('year').offset == 3600