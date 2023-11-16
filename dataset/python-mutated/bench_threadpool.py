"""
Benchmarks for thread pool.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import perf
from gevent.threadpool import ThreadPool
try:
    xrange = xrange
except NameError:
    xrange = range

def noop():
    if False:
        for i in range(10):
            print('nop')
    'Does nothing'

def identity(i):
    if False:
        while True:
            i = 10
    return i
PAR_COUNT = 5
N = 20

def bench_apply(loops):
    if False:
        print('Hello World!')
    pool = ThreadPool(1)
    t0 = perf.perf_counter()
    for _ in xrange(loops):
        for _ in xrange(N):
            pool.apply(noop)
    pool.join()
    pool.kill()
    return perf.perf_counter() - t0

def bench_spawn_wait(loops):
    if False:
        i = 10
        return i + 15
    pool = ThreadPool(1)
    t0 = perf.perf_counter()
    for _ in xrange(loops):
        for _ in xrange(N):
            r = pool.spawn(noop)
            r.get()
    pool.join()
    pool.kill()
    return perf.perf_counter() - t0

def _map(pool, pool_func, loops):
    if False:
        return 10
    data = [1] * N
    t0 = perf.perf_counter()
    for _ in xrange(loops):
        list(pool_func(identity, data))
    pool.join()
    pool.kill()
    return perf.perf_counter() - t0

def _ppool():
    if False:
        while True:
            i = 10
    pool = ThreadPool(PAR_COUNT)
    pool.size = PAR_COUNT
    return pool

def bench_map_seq(loops):
    if False:
        print('Hello World!')
    pool = ThreadPool(1)
    return _map(pool, pool.map, loops)

def bench_map_par(loops):
    if False:
        return 10
    pool = _ppool()
    return _map(pool, pool.map, loops)

def bench_imap_seq(loops):
    if False:
        print('Hello World!')
    pool = ThreadPool(1)
    return _map(pool, pool.imap, loops)

def bench_imap_par(loops):
    if False:
        for i in range(10):
            print('nop')
    pool = _ppool()
    return _map(pool, pool.imap, loops)

def bench_imap_un_seq(loops):
    if False:
        print('Hello World!')
    pool = ThreadPool(1)
    return _map(pool, pool.imap_unordered, loops)

def bench_imap_un_par(loops):
    if False:
        return 10
    pool = _ppool()
    return _map(pool, pool.imap_unordered, loops)

def main():
    if False:
        i = 10
        return i + 15
    runner = perf.Runner()
    runner.bench_time_func('imap_unordered_seq', bench_imap_un_seq)
    runner.bench_time_func('imap_unordered_par', bench_imap_un_par)
    runner.bench_time_func('imap_seq', bench_imap_seq)
    runner.bench_time_func('imap_par', bench_imap_par)
    runner.bench_time_func('map_seq', bench_map_seq)
    runner.bench_time_func('map_par', bench_map_par)
    runner.bench_time_func('apply', bench_apply)
    runner.bench_time_func('spawn', bench_spawn_wait)
if __name__ == '__main__':
    main()