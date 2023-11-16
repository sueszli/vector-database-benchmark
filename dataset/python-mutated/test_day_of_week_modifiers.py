from __future__ import annotations
import pytest
import pendulum
from pendulum.exceptions import PendulumException
from tests.conftest import assert_datetime

def test_start_of_week():
    if False:
        return 10
    d = pendulum.datetime(1980, 8, 7, 12, 11, 9).start_of('week')
    assert_datetime(d, 1980, 8, 4, 0, 0, 0)

def test_start_of_week_from_week_start():
    if False:
        print('Hello World!')
    d = pendulum.datetime(1980, 8, 4).start_of('week')
    assert_datetime(d, 1980, 8, 4, 0, 0, 0)

def test_start_of_week_crossing_year_boundary():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2014, 1, 1).start_of('week')
    assert_datetime(d, 2013, 12, 30, 0, 0, 0)

def test_end_of_week():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1980, 8, 7, 12, 11, 9).end_of('week')
    assert_datetime(d, 1980, 8, 10, 23, 59, 59)

def test_end_of_week_from_week_end():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1980, 8, 10).end_of('week')
    assert_datetime(d, 1980, 8, 10, 23, 59, 59)

def test_end_of_week_crossing_year_boundary():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(2013, 12, 31).end_of('week')
    assert_datetime(d, 2014, 1, 5, 23, 59, 59)

def test_next():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 5, 21).next()
    assert_datetime(d, 1975, 5, 28, 0, 0, 0)

def test_next_monday():
    if False:
        print('Hello World!')
    d = pendulum.datetime(1975, 5, 21).next(pendulum.MONDAY)
    assert_datetime(d, 1975, 5, 26, 0, 0, 0)

def test_next_saturday():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 5, 21).next(5)
    assert_datetime(d, 1975, 5, 24, 0, 0, 0)

def test_next_keep_time():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 5, 21, 12).next()
    assert_datetime(d, 1975, 5, 28, 0, 0, 0)
    d = pendulum.datetime(1975, 5, 21, 12).next(keep_time=True)
    assert_datetime(d, 1975, 5, 28, 12, 0, 0)

def test_next_invalid():
    if False:
        for i in range(10):
            print('nop')
    dt = pendulum.datetime(1975, 5, 21, 12)
    with pytest.raises(ValueError):
        dt.next(7)

def test_previous():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 5, 21).previous()
    assert_datetime(d, 1975, 5, 14, 0, 0, 0)

def test_previous_monday():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 5, 21).previous(pendulum.MONDAY)
    assert_datetime(d, 1975, 5, 19, 0, 0, 0)

def test_previous_saturday():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 5, 21).previous(5)
    assert_datetime(d, 1975, 5, 17, 0, 0, 0)

def test_previous_keep_time():
    if False:
        return 10
    d = pendulum.datetime(1975, 5, 21, 12).previous()
    assert_datetime(d, 1975, 5, 14, 0, 0, 0)
    d = pendulum.datetime(1975, 5, 21, 12).previous(keep_time=True)
    assert_datetime(d, 1975, 5, 14, 12, 0, 0)

def test_previous_invalid():
    if False:
        print('Hello World!')
    dt = pendulum.datetime(1975, 5, 21, 12)
    with pytest.raises(ValueError):
        dt.previous(7)

def test_first_day_of_month():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 11, 21).first_of('month')
    assert_datetime(d, 1975, 11, 1, 0, 0, 0)

def test_first_wednesday_of_month():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 11, 21).first_of('month', pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 11, 5, 0, 0, 0)

def test_first_friday_of_month():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 11, 21).first_of('month', 4)
    assert_datetime(d, 1975, 11, 7, 0, 0, 0)

def test_last_day_of_month():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 12, 5).last_of('month')
    assert_datetime(d, 1975, 12, 31, 0, 0, 0)

def test_last_tuesday_of_month():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 12, 1).last_of('month', pendulum.TUESDAY)
    assert_datetime(d, 1975, 12, 30, 0, 0, 0)

def test_last_friday_of_month():
    if False:
        return 10
    d = pendulum.datetime(1975, 12, 5).last_of('month', 4)
    assert_datetime(d, 1975, 12, 26, 0, 0, 0)

def test_nth_of_month_outside_scope():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 6, 5)
    with pytest.raises(PendulumException):
        d.nth_of('month', 6, pendulum.MONDAY)

def test_nth_of_month_outside_year():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 12, 5)
    with pytest.raises(PendulumException):
        d.nth_of('month', 55, pendulum.MONDAY)

def test_nth_of_month_first():
    if False:
        print('Hello World!')
    d = pendulum.datetime(1975, 12, 5).nth_of('month', 1, pendulum.MONDAY)
    assert_datetime(d, 1975, 12, 1, 0, 0, 0)

def test_2nd_monday_of_month():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 12, 5).nth_of('month', 2, pendulum.MONDAY)
    assert_datetime(d, 1975, 12, 8, 0, 0, 0)

def test_3rd_wednesday_of_month():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 12, 5).nth_of('month', 3, 2)
    assert_datetime(d, 1975, 12, 17, 0, 0, 0)

def test_first_day_of_quarter():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 11, 21).first_of('quarter')
    assert_datetime(d, 1975, 10, 1, 0, 0, 0)

def test_first_wednesday_of_quarter():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 11, 21).first_of('quarter', pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 10, 1, 0, 0, 0)

def test_first_friday_of_quarter():
    if False:
        print('Hello World!')
    d = pendulum.datetime(1975, 11, 21).first_of('quarter', 4)
    assert_datetime(d, 1975, 10, 3, 0, 0, 0)

def test_first_of_quarter_from_a_day_that_will_not_exist_in_the_first_month():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2014, 5, 31).first_of('quarter')
    assert_datetime(d, 2014, 4, 1, 0, 0, 0)

def test_last_day_of_quarter():
    if False:
        return 10
    d = pendulum.datetime(1975, 8, 5).last_of('quarter')
    assert_datetime(d, 1975, 9, 30, 0, 0, 0)

def test_last_tuesday_of_quarter():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 8, 5).last_of('quarter', pendulum.TUESDAY)
    assert_datetime(d, 1975, 9, 30, 0, 0, 0)

def test_last_friday_of_quarter():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 8, 5).last_of('quarter', pendulum.FRIDAY)
    assert_datetime(d, 1975, 9, 26, 0, 0, 0)

def test_last_day_of_quarter_that_will_not_exist_in_the_last_month():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(2014, 5, 31).last_of('quarter')
    assert_datetime(d, 2014, 6, 30, 0, 0, 0)

def test_nth_of_quarter_outside_scope():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 1, 5)
    with pytest.raises(PendulumException):
        d.nth_of('quarter', 20, pendulum.MONDAY)

def test_nth_of_quarter_outside_year():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 1, 5)
    with pytest.raises(PendulumException):
        d.nth_of('quarter', 55, pendulum.MONDAY)

def test_nth_of_quarter_first():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 12, 5).nth_of('quarter', 1, pendulum.MONDAY)
    assert_datetime(d, 1975, 10, 6, 0, 0, 0)

def test_nth_of_quarter_from_a_day_that_will_not_exist_in_the_first_month():
    if False:
        print('Hello World!')
    d = pendulum.datetime(2014, 5, 31).nth_of('quarter', 2, pendulum.MONDAY)
    assert_datetime(d, 2014, 4, 14, 0, 0, 0)

def test_2nd_monday_of_quarter():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 8, 5).nth_of('quarter', 2, pendulum.MONDAY)
    assert_datetime(d, 1975, 7, 14, 0, 0, 0)

def test_3rd_wednesday_of_quarter():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 8, 5).nth_of('quarter', 3, 2)
    assert_datetime(d, 1975, 7, 16, 0, 0, 0)

def test_first_day_of_year():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 11, 21).first_of('year')
    assert_datetime(d, 1975, 1, 1, 0, 0, 0)

def test_first_wednesday_of_year():
    if False:
        return 10
    d = pendulum.datetime(1975, 11, 21).first_of('year', pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 1, 1, 0, 0, 0)

def test_first_friday_of_year():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 11, 21).first_of('year', 4)
    assert_datetime(d, 1975, 1, 3, 0, 0, 0)

def test_last_day_of_year():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 8, 5).last_of('year')
    assert_datetime(d, 1975, 12, 31, 0, 0, 0)

def test_last_tuesday_of_year():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 8, 5).last_of('year', pendulum.TUESDAY)
    assert_datetime(d, 1975, 12, 30, 0, 0, 0)

def test_last_friday_of_year():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 8, 5).last_of('year', 4)
    assert_datetime(d, 1975, 12, 26, 0, 0, 0)

def test_nth_of_year_outside_scope():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 1, 5)
    with pytest.raises(PendulumException):
        d.nth_of('year', 55, pendulum.MONDAY)

def test_nth_of_year_first():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 12, 5).nth_of('year', 1, pendulum.MONDAY)
    assert_datetime(d, 1975, 1, 6, 0, 0, 0)

def test_2nd_monday_of_year():
    if False:
        while True:
            i = 10
    d = pendulum.datetime(1975, 8, 5).nth_of('year', 2, pendulum.MONDAY)
    assert_datetime(d, 1975, 1, 13, 0, 0, 0)

def test_2rd_wednesday_of_year():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 8, 5).nth_of('year', 3, pendulum.WEDNESDAY)
    assert_datetime(d, 1975, 1, 15, 0, 0, 0)

def test_7th_thursday_of_year():
    if False:
        for i in range(10):
            print('nop')
    d = pendulum.datetime(1975, 8, 31).nth_of('year', 7, pendulum.THURSDAY)
    assert_datetime(d, 1975, 2, 13, 0, 0, 0)

def test_first_of_invalid_unit():
    if False:
        return 10
    d = pendulum.datetime(1975, 8, 5)
    with pytest.raises(ValueError):
        d.first_of('invalid')

def test_last_of_invalid_unit():
    if False:
        i = 10
        return i + 15
    d = pendulum.datetime(1975, 8, 5)
    with pytest.raises(ValueError):
        d.last_of('invalid')

def test_nth_of_invalid_unit():
    if False:
        return 10
    d = pendulum.datetime(1975, 8, 5)
    with pytest.raises(ValueError):
        d.nth_of('invalid', 3, pendulum.MONDAY)