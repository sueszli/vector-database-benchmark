def gen():
    if False:
        i = 10
        return i + 15
    yield 1
    yield 2
    yield 3
    yield 4

def gen2():
    if False:
        print('Hello World!')
    yield (-1)
    print((yield from gen()))
    yield 10
    yield 11
g = gen2()
print(next(g))
print(next(g))
g.close()
try:
    print(next(g))
except StopIteration:
    print('StopIteration')

def gen3():
    if False:
        i = 10
        return i + 15
    yield 1
    try:
        yield 2
    except GeneratorExit:
        print('leaf caught GeneratorExit and swallowed it')
        return
    yield 3
    yield 4

def gen4():
    if False:
        print('Hello World!')
    yield (-1)
    try:
        print((yield from gen3()))
    except GeneratorExit:
        print('delegating caught GeneratorExit')
        raise
    yield 10
    yield 11
g = gen4()
print(next(g))
print(next(g))
print(next(g))
g.close()
try:
    print(next(g))
except StopIteration:
    print('StopIteration')

def gen5():
    if False:
        print('Hello World!')
    yield 1
    try:
        yield 2
    except GeneratorExit:
        print('leaf caught GeneratorExit and reraised GeneratorExit')
        raise GeneratorExit(123)
    yield 3
    yield 4

def gen6():
    if False:
        i = 10
        return i + 15
    yield (-1)
    try:
        print((yield from gen5()))
    except GeneratorExit:
        print('delegating caught GeneratorExit')
        raise
    yield 10
    yield 11
g = gen6()
print(next(g))
print(next(g))
print(next(g))
g.close()
try:
    print(next(g))
except StopIteration:
    print('StopIteration')

def gen7():
    if False:
        i = 10
        return i + 15
    try:
        yield 123
    except GeneratorExit:
        yield 456
g = gen7()
print(next(g))
try:
    g.close()
except RuntimeError:
    print('RuntimeError')

def gen8():
    if False:
        while True:
            i = 10
    g = range(2)
    yield from g
g = gen8()
print(next(g))
g.close()

class Iter:

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        return 1

    def close(self):
        if False:
            return 10
        print('close')

def gen9():
    if False:
        print('Hello World!')
    yield from Iter()
g = gen9()
print(next(g))
g.close()