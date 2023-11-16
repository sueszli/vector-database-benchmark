from __future__ import annotations
from datetime import timedelta
import pendulum
from tests.conftest import assert_duration

def test_add_interval():
    if False:
        for i in range(10):
            print('nop')
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = pendulum.duration(days=12, seconds=30)
    p = p1 + p2
    assert_duration(p, 0, 0, 5, 0, 0, 1, 2)

def test_add_timedelta():
    if False:
        for i in range(10):
            print('nop')
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = timedelta(days=12, seconds=30)
    p = p1 + p2
    assert_duration(p, 0, 0, 5, 0, 0, 1, 2)

def test_add_unsupported():
    if False:
        for i in range(10):
            print('nop')
    p = pendulum.duration(days=23, seconds=32)
    assert NotImplemented == p.__add__(5)

def test_sub_interval():
    if False:
        for i in range(10):
            print('nop')
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = pendulum.duration(days=12, seconds=28)
    p = p1 - p2
    assert_duration(p, 0, 0, 1, 4, 0, 0, 4)

def test_sub_timedelta():
    if False:
        return 10
    p1 = pendulum.duration(days=23, seconds=32)
    p2 = timedelta(days=12, seconds=28)
    p = p1 - p2
    assert_duration(p, 0, 0, 1, 4, 0, 0, 4)

def test_sub_unsupported():
    if False:
        for i in range(10):
            print('nop')
    p = pendulum.duration(days=23, seconds=32)
    assert NotImplemented == p.__sub__(5)

def test_neg():
    if False:
        return 10
    p = pendulum.duration(days=23, seconds=32)
    assert_duration(-p, 0, 0, -3, -2, 0, 0, -32)