"""
See how fast deferreds are.

This is mainly useful to compare cdefer.Deferred to defer.Deferred
"""
from timer import timeit
from twisted.internet import defer
from twisted.python.compat import range
benchmarkFuncs = []

def benchmarkFunc(iter, args=()):
    if False:
        print('Hello World!')
    '\n    A decorator for benchmark functions that measure a single iteration\n    count. Registers the function with the given iteration count to the global\n    benchmarkFuncs list\n    '

    def decorator(func):
        if False:
            while True:
                i = 10
        benchmarkFuncs.append((func, args, iter))
        return func
    return decorator

def benchmarkNFunc(iter, ns):
    if False:
        i = 10
        return i + 15
    '\n    A decorator for benchmark functions that measure multiple iteration\n    counts. Registers the function with the given iteration count to the global\n    benchmarkFuncs list.\n    '

    def decorator(func):
        if False:
            i = 10
            return i + 15
        for n in ns:
            benchmarkFuncs.append((func, (n,), iter))
        return func
    return decorator

def instantiate():
    if False:
        while True:
            i = 10
    '\n    Only create a deferred\n    '
    d = defer.Deferred()
instantiate = benchmarkFunc(100000)(instantiate)

def instantiateShootCallback():
    if False:
        print('Hello World!')
    '\n    Create a deferred and give it a normal result\n    '
    d = defer.Deferred()
    d.callback(1)
instantiateShootCallback = benchmarkFunc(100000)(instantiateShootCallback)

def instantiateShootErrback():
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a deferred and give it an exception result. To avoid Unhandled\n    Errors, also register an errback that eats the error\n    '
    d = defer.Deferred()
    try:
        1 / 0
    except BaseException:
        d.errback()
    d.addErrback(lambda x: None)
instantiateShootErrback = benchmarkFunc(200)(instantiateShootErrback)
ns = [10, 1000, 10000]

def instantiateAddCallbacksNoResult(n):
    if False:
        while True:
            i = 10
    '\n    Creates a deferred and adds a trivial callback/errback/both to it the given\n    number of times.\n    '
    d = defer.Deferred()

    def f(result):
        if False:
            i = 10
            return i + 15
        return result
    for i in range(n):
        d.addCallback(f)
        d.addErrback(f)
        d.addBoth(f)
        d.addCallbacks(f, f)
instantiateAddCallbacksNoResult = benchmarkNFunc(20, ns)(instantiateAddCallbacksNoResult)

def instantiateAddCallbacksBeforeResult(n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a deferred and adds a trivial callback/errback/both to it the given\n    number of times, and then shoots a result through all of the callbacks.\n    '
    d = defer.Deferred()

    def f(result):
        if False:
            for i in range(10):
                print('nop')
        return result
    for i in range(n):
        d.addCallback(f)
        d.addErrback(f)
        d.addBoth(f)
        d.addCallbacks(f)
    d.callback(1)
instantiateAddCallbacksBeforeResult = benchmarkNFunc(20, ns)(instantiateAddCallbacksBeforeResult)

def instantiateAddCallbacksAfterResult(n):
    if False:
        while True:
            i = 10
    '\n    Create a deferred, shoots it and then adds a trivial callback/errback/both\n    to it the given number of times. The result is processed through the\n    callbacks as they are added.\n    '
    d = defer.Deferred()

    def f(result):
        if False:
            for i in range(10):
                print('nop')
        return result
    d.callback(1)
    for i in range(n):
        d.addCallback(f)
        d.addErrback(f)
        d.addBoth(f)
        d.addCallbacks(f)
instantiateAddCallbacksAfterResult = benchmarkNFunc(20, ns)(instantiateAddCallbacksAfterResult)

def pauseUnpause(n):
    if False:
        print('Hello World!')
    '\n    Adds the given number of callbacks/errbacks/both to a deferred while it is\n    paused, and unpauses it, trigerring the processing of the value through the\n    callbacks.\n    '
    d = defer.Deferred()

    def f(result):
        if False:
            print('Hello World!')
        return result
    d.callback(1)
    d.pause()
    for i in range(n):
        d.addCallback(f)
        d.addErrback(f)
        d.addBoth(f)
        d.addCallbacks(f)
    d.unpause()
pauseUnpause = benchmarkNFunc(20, ns)(pauseUnpause)

def benchmark():
    if False:
        while True:
            i = 10
    '\n    Run all of the benchmarks registered in the benchmarkFuncs list\n    '
    print(defer.Deferred.__module__)
    for (func, args, iter) in benchmarkFuncs:
        print(func.__name__, args, timeit(func, iter, *args))
if __name__ == '__main__':
    benchmark()