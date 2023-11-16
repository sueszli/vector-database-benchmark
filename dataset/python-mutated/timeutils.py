"""Simple tools for timing functions' execution, when IPython is not available. """
import timeit
import math
_scales = [1.0, 1000.0, 1000000.0, 1000000000.0]
_units = ['s', 'ms', 'μs', 'ns']

def timed(func, setup='pass', limit=None):
    if False:
        print('Hello World!')
    'Adaptively measure execution time of a function. '
    timer = timeit.Timer(func, setup=setup)
    (repeat, number) = (3, 1)
    for i in range(1, 10):
        if timer.timeit(number) >= 0.2:
            break
        elif limit is not None and number >= limit:
            break
        else:
            number *= 10
    time = min(timer.repeat(repeat, number)) / number
    if time > 0.0:
        order = min(-int(math.floor(math.log10(time)) // 3), 3)
    else:
        order = 3
    return (number, time, time * _scales[order], _units[order])

def __do_timings():
    if False:
        for i in range(10):
            print('nop')
    import os
    res = os.getenv('SYMPY_TIMINGS', '')
    res = [x.strip() for x in res.split(',')]
    return set(res)
_do_timings = __do_timings()
_timestack = None

def _print_timestack(stack, level=1):
    if False:
        while True:
            i = 10
    print('-' * level, '%.2f %s%s' % (stack[2], stack[0], stack[3]))
    for s in stack[1]:
        _print_timestack(s, level + 1)

def timethis(name):
    if False:
        i = 10
        return i + 15

    def decorator(func):
        if False:
            print('Hello World!')
        global _do_timings
        if name not in _do_timings:
            return func

        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            from time import time
            global _timestack
            oldtimestack = _timestack
            _timestack = [func.func_name, [], 0, args]
            t1 = time()
            r = func(*args, **kwargs)
            t2 = time()
            _timestack[2] = t2 - t1
            if oldtimestack is not None:
                oldtimestack[1].append(_timestack)
                _timestack = oldtimestack
            else:
                _print_timestack(_timestack)
                _timestack = None
            return r
        return wrapper
    return decorator