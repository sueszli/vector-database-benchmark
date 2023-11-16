from __future__ import annotations
import pendulum

def test_dst_add():
    if False:
        for i in range(10):
            print('nop')
    start = pendulum.datetime(2017, 3, 7, tz='America/Toronto')
    end = start.add(days=6)
    interval = end - start
    new_end = start + interval
    assert new_end == end

def test_dst_add_non_variable_units():
    if False:
        i = 10
        return i + 15
    start = pendulum.datetime(2013, 3, 31, 1, 30, tz='Europe/Paris')
    end = start.add(hours=1)
    interval = end - start
    new_end = start + interval
    assert new_end == end

def test_dst_subtract():
    if False:
        return 10
    start = pendulum.datetime(2017, 3, 7, tz='America/Toronto')
    end = start.add(days=6)
    interval = end - start
    new_start = end - interval
    assert new_start == start

def test_naive_subtract():
    if False:
        i = 10
        return i + 15
    start = pendulum.naive(2013, 3, 31, 1, 30)
    end = start.add(hours=1)
    interval = end - start
    new_end = start + interval
    assert new_end == end

def test_negative_difference_subtract():
    if False:
        for i in range(10):
            print('nop')
    start = pendulum.datetime(2018, 5, 28, 12, 34, 56, 123456)
    end = pendulum.datetime(2018, 1, 1)
    interval = end - start
    new_end = start + interval
    assert new_end == end