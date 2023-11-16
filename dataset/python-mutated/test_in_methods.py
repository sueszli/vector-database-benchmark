from __future__ import annotations
import pendulum

def test_in_weeks():
    if False:
        print('Hello World!')
    it = pendulum.duration(days=17)
    assert it.in_weeks() == 2

def test_in_days():
    if False:
        i = 10
        return i + 15
    it = pendulum.duration(days=3)
    assert it.in_days() == 3

def test_in_hours():
    if False:
        return 10
    it = pendulum.duration(days=3, minutes=72)
    assert it.in_hours() == 73

def test_in_minutes():
    if False:
        return 10
    it = pendulum.duration(minutes=6, seconds=72)
    assert it.in_minutes() == 7

def test_in_seconds():
    if False:
        for i in range(10):
            print('nop')
    it = pendulum.duration(seconds=72)
    assert it.in_seconds() == 72