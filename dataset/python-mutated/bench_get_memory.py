"""
Benchmarking for getting the memoryview of an object.

https://github.com/gevent/gevent/issues/1318
"""
from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range
try:
    buffer
except NameError:
    buffer = memoryview
import perf
from gevent._greenlet_primitives import get_memory as cy_get_memory

def get_memory_gevent14(data):
    if False:
        while True:
            i = 10
    try:
        mv = memoryview(data)
        if mv.shape:
            return mv
        return mv.tobytes()
    except TypeError:
        return buffer(data)

def get_memory_is(data):
    if False:
        i = 10
        return i + 15
    try:
        mv = memoryview(data) if type(data) is not memoryview else data
        if mv.shape:
            return mv
        return mv.tobytes()
    except TypeError:
        return buffer(data)

def get_memory_inst(data):
    if False:
        return 10
    try:
        mv = memoryview(data) if not isinstance(data, memoryview) else data
        if mv.shape:
            return mv
        return mv.tobytes()
    except TypeError:
        return buffer(data)
N = 100
DATA = {'bytestring': b'abc123', 'bytearray': bytearray(b'abc123'), 'memoryview': memoryview(b'abc123')}

def test(loops, func, arg):
    if False:
        i = 10
        return i + 15
    t0 = perf.perf_counter()
    for __ in range(loops):
        for _ in xrange(N):
            func(arg)
    return perf.perf_counter() - t0

def main():
    if False:
        i = 10
        return i + 15
    runner = perf.Runner()
    for (func, name) in ((get_memory_gevent14, 'gevent14-py'), (cy_get_memory, 'inst-cy'), (get_memory_inst, 'inst-py'), (get_memory_is, 'is-py')):
        for (arg_name, arg) in DATA.items():
            runner.bench_time_func('%s - %s' % (name, arg_name), test, func, arg, inner_loops=N)
if __name__ == '__main__':
    main()