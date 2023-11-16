def gen():
    if False:
        return 10
    try:
        yield 1
    except ValueError as e:
        print('got ValueError from upstream!', repr(e.args))
    yield 'str1'
    raise TypeError

def gen2():
    if False:
        print('Hello World!')
    print((yield from gen()))
g = gen2()
print(next(g))
print(g.throw(ValueError))
try:
    print(next(g))
except TypeError:
    print('got TypeError from downstream!')
g = gen2()
print(next(g))
print(g.throw(ValueError, None))
try:
    print(next(g))
except TypeError:
    print('got TypeError from downstream!')
g = gen2()
print(next(g))
print(g.throw(ValueError, ValueError(123)))
try:
    print(next(g))
except TypeError:
    print('got TypeError from downstream!')

def gen():
    if False:
        i = 10
        return i + 15
    try:
        yield 123
    except ValueError:
        print('ValueError')

def gen2():
    if False:
        for i in range(10):
            print('nop')
    yield from gen()
    yield 789
g = gen2()
print(next(g))
print(g.throw(ValueError))