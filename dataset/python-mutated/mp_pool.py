import multiprocessing
import time
import random
import sys

def calculate(func, args):
    if False:
        return 10
    result = func(*args)
    return '%s says that %s%s = %s' % (multiprocessing.current_process().name, func.__name__, args, result)

def calculatestar(args):
    if False:
        i = 10
        return i + 15
    return calculate(*args)

def mul(a, b):
    if False:
        while True:
            i = 10
    time.sleep(0.5 * random.random())
    return a * b

def plus(a, b):
    if False:
        for i in range(10):
            print('nop')
    time.sleep(0.5 * random.random())
    return a + b

def f(x):
    if False:
        while True:
            i = 10
    return 1.0 / (x - 5.0)

def pow3(x):
    if False:
        for i in range(10):
            print('nop')
    return x ** 3

def noop(x):
    if False:
        while True:
            i = 10
    pass

def test():
    if False:
        return 10
    PROCESSES = 4
    print('Creating pool with %d processes\n' % PROCESSES)
    with multiprocessing.Pool(PROCESSES) as pool:
        TASKS = [(mul, (i, 7)) for i in range(10)] + [(plus, (i, 8)) for i in range(10)]
        results = [pool.apply_async(calculate, t) for t in TASKS]
        imap_it = pool.imap(calculatestar, TASKS)
        imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)
        print('Ordered results using pool.apply_async():')
        for r in results:
            print('\t', r.get())
        print()
        print('Ordered results using pool.imap():')
        for x in imap_it:
            print('\t', x)
        print()
        print('Unordered results using pool.imap_unordered():')
        for x in imap_unordered_it:
            print('\t', x)
        print()
        print('Ordered results using pool.map() --- will block till complete:')
        for x in pool.map(calculatestar, TASKS):
            print('\t', x)
        print()
        print('Testing error handling:')
        try:
            print(pool.apply(f, (5,)))
        except ZeroDivisionError:
            print('\tGot ZeroDivisionError as expected from pool.apply()')
        else:
            raise AssertionError('expected ZeroDivisionError')
        try:
            print(pool.map(f, list(range(10))))
        except ZeroDivisionError:
            print('\tGot ZeroDivisionError as expected from pool.map()')
        else:
            raise AssertionError('expected ZeroDivisionError')
        try:
            print(list(pool.imap(f, list(range(10)))))
        except ZeroDivisionError:
            print('\tGot ZeroDivisionError as expected from list(pool.imap())')
        else:
            raise AssertionError('expected ZeroDivisionError')
        it = pool.imap(f, list(range(10)))
        for i in range(10):
            try:
                x = next(it)
            except ZeroDivisionError:
                if i == 5:
                    pass
            except StopIteration:
                break
            else:
                if i == 5:
                    raise AssertionError('expected ZeroDivisionError')
        assert i == 9
        print('\tGot ZeroDivisionError as expected from IMapIterator.next()')
        print()
        print('Testing ApplyResult.get() with timeout:', end=' ')
        res = pool.apply_async(calculate, TASKS[0])
        while 1:
            sys.stdout.flush()
            try:
                sys.stdout.write('\n\t%s' % res.get(0.02))
                break
            except multiprocessing.TimeoutError:
                sys.stdout.write('.')
        print()
        print()
        print('Testing IMapIterator.next() with timeout:', end=' ')
        it = pool.imap(calculatestar, TASKS)
        while 1:
            sys.stdout.flush()
            try:
                sys.stdout.write('\n\t%s' % it.next(0.02))
            except StopIteration:
                break
            except multiprocessing.TimeoutError:
                sys.stdout.write('.')
        print()
        print()
if __name__ == '__main__':
    multiprocessing.freeze_support()
    test()