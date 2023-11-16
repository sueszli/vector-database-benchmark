from __future__ import print_function
from sys import hexversion
import random
from sortedcontainers import SortedSet
from functools import wraps
import operator
if hexversion < 50331648:
    from itertools import izip as zip
    range = xrange
random.seed(0)
actions = []

def actor(func):
    if False:
        print('Hello World!')
    actions.append(func)
    return func

def test_init():
    if False:
        i = 10
        return i + 15
    sst = SortedSet()
    sst._check()
    sst = SortedSet()
    sst._reset(10000)
    assert sst._list._load == 10000
    sst._check()
    sst = SortedSet(range(10000))
    assert all((tup[0] == tup[1] for tup in zip(sst, range(10000))))
    sst.clear()
    assert len(sst) == 0
    assert list(iter(sst)) == []
    sst._check()

@actor
def stress_contains(sst):
    if False:
        i = 10
        return i + 15
    values = list(sst)
    assert all((val in sst for val in values))

@actor
def stress_delitem(sst):
    if False:
        while True:
            i = 10
    for rpt in range(100):
        pos = random.randrange(0, len(sst))
        del sst[pos]

@actor
def stress_operator(sst):
    if False:
        while True:
            i = 10
    other = SortedSet(sst)
    stress_delitem(other)
    assert other < sst
    assert sst > other

@actor
def stress_getitem(sst):
    if False:
        print('Hello World!')
    other = list(sst)
    assert all((sst[pos] == other[pos] for pos in range(len(sst))))

@actor
def stress_reversed(sst):
    if False:
        print('Hello World!')
    other = list(reversed(list(sst)))
    assert all((tup[0] == tup[1] for tup in zip(reversed(sst), other)))

@actor
def stress_add(sst):
    if False:
        return 10
    for rpt in range(100):
        val = random.randrange(0, 1000)
        sst.add(val)

@actor
def stress_count(sst):
    if False:
        while True:
            i = 10
    for val in sst:
        assert sst.count(val) == 1

@actor
def stress_difference(sst):
    if False:
        i = 10
        return i + 15
    copy_one = sst.copy()
    stress_delitem(copy_one)
    copy_two = sst.copy()
    stress_delitem(copy_two)
    sst.difference_update(copy_one, copy_two)

@actor
def stress_discard(sst):
    if False:
        while True:
            i = 10
    for rpt in range(100):
        pos = random.randrange(0, len(sst))
        val = sst[pos]
        sst.discard(val)

@actor
def stress_index(sst):
    if False:
        for i in range(10):
            print('nop')
    for rpt in range(100):
        pos = random.randrange(0, len(sst))
        val = sst[pos]
        assert pos == sst.index(val)

@actor
def stress_intersection(sst):
    if False:
        i = 10
        return i + 15
    copy_one = sst.copy()
    stress_delitem(copy_one)
    copy_two = sst.copy()
    stress_delitem(copy_two)
    sst.intersection_update(copy_one, copy_two)

@actor
def stress_symmetric_difference(sst):
    if False:
        i = 10
        return i + 15
    copy_one = sst.copy()
    stress_delitem(copy_one)
    sst.symmetric_difference_update(copy_one)

@actor
def stress_pop(sst):
    if False:
        i = 10
        return i + 15
    val = sst[-1]
    assert val == sst.pop()
    for rpt in range(100):
        pos = random.randrange(0, len(sst))
        val = sst[pos]
        assert val == sst.pop(pos)

@actor
def stress_remove(sst):
    if False:
        print('Hello World!')
    for rpt in range(100):
        pos = random.randrange(0, len(sst))
        val = sst[pos]
        sst.remove(val)

@actor
def stress_update(sst):
    if False:
        print('Hello World!')

    def iter_randomly(start, stop, count):
        if False:
            print('Hello World!')
        for rpt in range(count):
            yield random.randrange(start, stop)
    sst.update(iter_randomly(0, 500, 100), iter_randomly(500, 1000, 100), iter_randomly(1000, 1500, 100), iter_randomly(1500, 2000, 100))

@actor
def stress_isdisjoint(sst):
    if False:
        return 10
    values = [-1, -2, -3]
    assert sst.isdisjoint(values)

@actor
def stress_issubset(sst):
    if False:
        return 10
    that = SortedSet(sst)
    that.update(range(1000))
    assert sst.issubset(that)

@actor
def stress_issuperset(sst):
    if False:
        i = 10
        return i + 15
    that = SortedSet(sst)
    assert sst.issuperset(that)

def test_stress(repeat=1000):
    if False:
        print('Hello World!')
    sst = SortedSet(range(1000))
    for rpt in range(repeat):
        action = random.choice(actions)
        action(sst)
        try:
            sst._check()
        except AssertionError:
            print(action)
            raise
        start_len = len(sst)
        while len(sst) < 500:
            sst.add(random.randrange(0, 2000))
        while len(sst) > 2000:
            del sst[random.randrange(0, len(sst))]
        if start_len != len(sst):
            sst._check()
if __name__ == '__main__':
    import sys
    from datetime import datetime
    start = datetime.now()
    print('Python', sys.version_info)
    try:
        num = int(sys.argv[1])
        print('Setting iterations to', num)
    except:
        print('Setting iterations to 1000 (default)')
        num = 1000
    try:
        pea = int(sys.argv[2])
        random.seed(pea)
        print('Setting seed to', pea)
    except:
        print('Setting seed to 0 (default)')
        random.seed(0)
    try:
        test_stress(num)
    except:
        raise
    finally:
        print('Exiting after', datetime.now() - start)