from __future__ import annotations
import pendulum

def test_in_weeks():
    if False:
        return 10
    it = pendulum.duration(days=17)
    assert round(it.total_weeks(), 2) == 2.43

def test_in_days():
    if False:
        while True:
            i = 10
    it = pendulum.duration(days=3)
    assert it.total_days() == 3

def test_in_hours():
    if False:
        i = 10
        return i + 15
    it = pendulum.duration(days=3, minutes=72)
    assert it.total_hours() == 73.2

def test_in_minutes():
    if False:
        return 10
    it = pendulum.duration(minutes=6, seconds=72)
    assert it.total_minutes() == 7.2

def test_in_seconds():
    if False:
        print('Hello World!')
    it = pendulum.duration(seconds=72, microseconds=123456)
    assert it.total_seconds() == 72.123456