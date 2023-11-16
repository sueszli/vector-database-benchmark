"""
Benchmarks for hub primitive operations.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import perf
from perf import perf_counter
import gevent
from greenlet import greenlet
from greenlet import getcurrent
N = 1000

def bench_switch():
    if False:
        i = 10
        return i + 15

    class Parent(type(gevent.get_hub())):

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            parent = self.parent
            for _ in range(N):
                parent.switch()

    def child():
        if False:
            while True:
                i = 10
        parent = getcurrent().parent
        for _ in range(N):
            parent.switch()
    hub = Parent(None, None)
    child_greenlet = greenlet(child, hub)
    for _ in range(N):
        child_greenlet.switch()

def bench_wait_ready():
    if False:
        return 10

    class Watcher(object):

        def start(self, cb, obj):
            if False:
                return 10
            cb(obj)

        def stop(self):
            if False:
                while True:
                    i = 10
            pass
    watcher = Watcher()
    hub = gevent.get_hub()
    for _ in range(1000):
        hub.wait(watcher)

def bench_cancel_wait():
    if False:
        i = 10
        return i + 15

    class Watcher(object):
        active = True
        callback = object()

        def close(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    watcher = Watcher()
    hub = gevent.get_hub()
    loop = hub.loop
    for _ in range(1000):
        hub.cancel_wait(watcher, None, True)
    for cb in loop._callbacks:
        if cb.callback:
            cb.callback(*cb.args)
            cb.stop()
    hub.destroy(True)

def bench_wait_func_ready():
    if False:
        i = 10
        return i + 15
    from gevent.hub import wait

    class ToWatch(object):

        def rawlink(self, cb):
            if False:
                while True:
                    i = 10
            cb(self)
    watched_objects = [ToWatch() for _ in range(N)]
    t0 = perf_counter()
    wait(watched_objects)
    return perf_counter() - t0

def main():
    if False:
        for i in range(10):
            print('nop')
    runner = perf.Runner()
    runner.bench_func('multiple wait ready', bench_wait_func_ready, inner_loops=N)
    runner.bench_func('wait ready', bench_wait_ready, inner_loops=N)
    runner.bench_func('cancel wait', bench_cancel_wait, inner_loops=N)
    runner.bench_func('switch', bench_switch, inner_loops=N)
if __name__ == '__main__':
    main()