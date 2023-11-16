""" Generator expression tests

"""
from __future__ import print_function
import inspect
print('Generator expression that demonstrates the timing:')

def iteratorCreationTiming():
    if False:
        while True:
            i = 10

    def getIterable(x):
        if False:
            while True:
                i = 10
        print('Getting iterable', x)
        return Iterable(x)

    class Iterable:

        def __init__(self, x):
            if False:
                i = 10
                return i + 15
            self.x = x
            self.values = list(range(x))
            self.count = 0

        def __iter__(self):
            if False:
                return 10
            print('Giving iterator now', self.x)
            return self

        def __next__(self):
            if False:
                for i in range(10):
                    print('nop')
            print('Next of', self.x, 'is', self.count)
            if len(self.values) > self.count:
                self.count += 1
                return self.values[self.count - 1]
            else:
                print('Raising StopIteration for', self.x)
                raise StopIteration
        next = __next__

        def __del__(self):
            if False:
                return 10
            print('Deleting', self.x)
    gen = ((y, z) for y in getIterable(3) for z in getIterable(2))
    print('Using generator', gen)
    next(gen)
    res = tuple(gen)
    print(res)
    print('*' * 20)
    try:
        next(gen)
    except StopIteration:
        print('Usage past end gave StopIteration exception as expected.')
        try:
            print('Generator state then is', inspect.getgeneratorstate(gen))
        except AttributeError:
            pass
        print('Its frame is now', gen.gi_frame)
    print('Early aborting generator:')
    gen2 = ((y, z) for y in getIterable(3) for z in getIterable(2))
    del gen2
iteratorCreationTiming()
print('Generator expressions that demonstrate the use of conditions:')
print(tuple((x for x in range(8) if x % 2 == 1)))
print(tuple((x for x in range(8) if x % 2 == 1 for z in range(8) if z == x)))
print(tuple((x for (x, y) in zip(list(range(2)), list(range(4))))))
print('Directory of generator expressions:')
for_dir = (x for x in [1])
gen_dir = dir(for_dir)
print(sorted((g for g in gen_dir)))

def genexprSend():
    if False:
        for i in range(10):
            print('nop')
    x = (x for x in range(9))
    print('Sending too early:')
    try:
        x.send(3)
    except TypeError as e:
        print('Gave expected TypeError with text:', e)
    try:
        z = next(x)
    except StopIteration as e:
        print('Gave expected (3.10.0/1 only) StopIteration with text:', repr(e))
    else:
        print('Next return value (pre 3.10)', z)
    try:
        y = x.send(3)
    except StopIteration as e:
        print('Gave expected (3.10.0/1 only) StopIteration with text:', repr(e))
    else:
        print('Send return value', y)
    try:
        print('And then next gave', next(x))
    except StopIteration as e:
        print('Gave expected (3.10.0/1 only) StopIteration with text:', repr(e))
    print('Throwing an exception to it.')
    try:
        x.throw(2)
        assert False
    except TypeError as e:
        print('Gave expected TypeError:', e)
    print('Throwing an exception to it.')
    try:
        x.throw(ValueError(2))
    except ValueError as e:
        print('Gave expected ValueError:', e)
    try:
        next(x)
        print('Next worked even after thrown error')
    except StopIteration as e:
        print('Gave expected stop iteration after throwing exception in it:', e)
    print('Throwing another exception from it.')
    try:
        x.throw(ValueError(5))
    except ValueError as e:
        print('Gave expected ValueError with text:', e)
print('Generator expressions have send too:')
genexprSend()

def genexprClose():
    if False:
        return 10
    x = (x for x in range(9))
    print('Immediate close:')
    x.close()
    print('Closed once')
    x.close()
    print('Closed again without any trouble')
genexprClose()

def genexprThrown():
    if False:
        while True:
            i = 10

    def checked(z):
        if False:
            return 10
        if z == 3:
            raise ValueError
        return z
    x = (checked(x) for x in range(9))
    try:
        for (count, value) in enumerate(x):
            print(count, value)
    except ValueError:
        print(count + 1, ValueError)
    try:
        next(x)
        print('Allowed to do next() after raised exception from the generator expression')
    except StopIteration:
        print('Exception in generator, disallowed next() afterwards.')
genexprThrown()

def nestedExpressions():
    if False:
        while True:
            i = 10
    a = [x for x in range(10)]
    b = (x for x in (y for y in a))
    print('nested generator expression', list(b))
nestedExpressions()

def lambdaGenerators():
    if False:
        while True:
            i = 10
    a = 1
    x = lambda : (yield a)
    print('Simple lambda generator', x, x(), list(x()))
    y = lambda : ((yield 1), (yield 2))
    print('Complex lambda generator', y, y(), list(y()))
lambdaGenerators()

def functionGenerators():
    if False:
        print('Hello World!')
    a = 1

    def x():
        if False:
            while True:
                i = 10
        yield a
    print('Simple function generator', x, x(), list(x()))

    def y():
        if False:
            for i in range(10):
                print('nop')
        yield ((yield 1), (yield 2))
    print('Complex function generator', y, y(), list(y()))
functionGenerators()